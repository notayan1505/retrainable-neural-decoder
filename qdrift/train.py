"""Training loop for the neural decoders.

RTX 5070 Ti (Blackwell, 16 GB) notes:
  * Needs a CUDA 12.8+ torch build, e.g.
    ``pip install torch --index-url https://download.pytorch.org/whl/cu128``.
  * The whole dataset lives on the GPU as uint8; use large batches (8k-64k).
  * bf16 autocast + TF32 are enabled automatically on CUDA (``amp=True``).
  * ``compile=True`` is optional; on failure (e.g. Windows without Triton) it
    falls back to eager mode with a warning.
On MPS/CPU everything runs in plain float32.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qdrift.data.stein import Dataset
from qdrift.models.net import build_model


def pick_device(name: str | None = None) -> torch.device:
    if name is None or name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (need an NVIDIA GPU and a CUDA-enabled torch build, "
                           "e.g. the cu128 wheel for an RTX 5070 Ti)")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine")
    return dev


@dataclass
class TrainConfig:
    kind: str = "mlp"
    width: int = 256
    depth: int = 3
    dropout: float = 0.0
    epochs: int = 20
    batch_size: int = 8192
    lr: float = 2e-3
    weight_decay: float = 1e-4
    device: str | None = None
    compile: bool = False
    amp: bool = True
    seed: int = 0
    val_frac: float = 0.1
    log_every: int = 1


def _autocast(dev: torch.device, enabled: bool):
    if dev.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _clean_state(model: nn.Module) -> dict:
    sd = getattr(model, "_orig_mod", model).state_dict()
    return {k.replace("_orig_mod.", ""): v.detach().cpu().clone() for k, v in sd.items()}


def _to_dev(arr: np.ndarray, dev: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.uint8)).to(dev)


@torch.no_grad()
def _eval(model, X, y, dev, amp, bs=65536):
    model.eval()
    tot, wrong, n = 0.0, 0, X.shape[0]
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    for i in range(0, n, bs):
        with _autocast(dev, amp):
            logit = model(X[i:i + bs].float())
        logit = logit.float()
        yb = y[i:i + bs].float()
        tot += lossf(logit, yb).item()
        wrong += int(((logit > 0).float() != yb).sum().item())
    return tot / max(n, 1), wrong / max(n, 1)


def training_split_indices(n: int, val_frac: float = 0.1, seed: int = 0):
    """Deterministically split indices into validation and optimizer-training subsets."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    perm = np.random.default_rng(seed).permutation(n)
    nv = int(round(val_frac * n))
    nv = min(max(nv, 1), n - 1) if n > 1 and val_frac > 0 else 0
    return perm[nv:], perm[:nv]


def train_model(train: Dataset, cfg: TrainConfig, val: Dataset | None = None,
                log=print, model: nn.Module | None = None) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(cfg.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = bool(cfg.amp) and dev.type == "cuda"

    if val is None:
        tr_idx, val_idx = training_split_indices(len(train.labels), cfg.val_frac, cfg.seed)
        Xv_np, yv_np = train.detectors[val_idx], train.labels[val_idx]
        Xt_np, yt_np = train.detectors[tr_idx], train.labels[tr_idx]
    else:
        Xt_np, yt_np = train.detectors, train.labels
        Xv_np, yv_np = val.detectors, val.labels
    Xt, yt = _to_dev(Xt_np, dev), _to_dev(yt_np, dev)
    Xv, yv = _to_dev(Xv_np, dev), _to_dev(yv_np, dev)

    n = Xt.shape[0]
    d, r = train.d, train.r
    if model is None:
        model = build_model(cfg.kind, d, r, cfg.width, cfg.depth, cfg.dropout)
    model = model.to(dev)
    run = model
    if cfg.compile:
        try:
            run = torch.compile(model)
            # torch.compile is lazy: a missing backend (e.g. Triton on Windows) only fails on
            # the first call, so make that call here where we can still fall back.
            with torch.no_grad(), _autocast(dev, amp):
                run(Xt[: min(2, n)].float())
        except Exception as e:  # noqa: BLE001
            log(f"warning: torch.compile failed ({e!r}); continuing uncompiled")
            run = model

    bs = max(1, min(cfg.batch_size, n))
    steps = (n + bs - 1) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * steps))
    lossf = nn.BCEWithLogitsLoss()

    best_loss, best_state = float("inf"), None
    history: list[dict] = []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        te = time.time()
        run.train()
        perm = torch.randperm(n, device=dev)
        tot = torch.zeros((), device=dev)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb = Xt[idx].float(), yt[idx].float()
            opt.zero_grad(set_to_none=True)
            with _autocast(dev, amp):
                logit = run(xb)
            loss = lossf(logit.float(), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach() * idx.numel()
        train_loss = float(tot.item()) / n
        try:
            vl, ve = _eval(run, Xv, yv, dev, amp) if Xv.shape[0] else (train_loss, float("nan"))
        except Exception:  # compiled eval failure: use eager module
            vl, ve = _eval(model, Xv, yv, dev, amp)
        secs = time.time() - te
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": vl, "val_err": ve, "seconds": secs})
        if vl < best_loss:
            best_loss, best_state = vl, _clean_state(model)
        if cfg.log_every and (ep % cfg.log_every == 0 or ep == cfg.epochs):
            log(f"epoch {ep:3d}/{cfg.epochs} loss {train_loss:.4f} val_loss {vl:.4f} "
                f"val_err {ve:.4f} {n / max(secs, 1e-9):,.0f} samples/s")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def predict(model, detectors: np.ndarray, device=None, batch_size: int = 65536) -> np.ndarray:
    detectors = np.asarray(detectors)
    if detectors.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    if device is None:
        device = next(model.parameters()).device
    dev = pick_device(device) if not isinstance(device, torch.device) else device
    model = model.to(dev).eval()
    X = _to_dev(detectors, dev)
    out = []
    for i in range(0, X.shape[0], batch_size):
        logit = model(X[i:i + batch_size].float())
        out.append((logit > 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(path, model, cfg: TrainConfig, d: int, r: int, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": _clean_state(model), "cfg": asdict(cfg), "d": int(d), "r": int(r),
               "extra": dict(extra or {})}
    torch.save(payload, path)
    return path


def load_checkpoint(path, device=None) -> tuple[nn.Module, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = payload["cfg"]
    model = build_model(cfg["kind"], payload["d"], payload["r"], cfg["width"], cfg["depth"], cfg["dropout"])
    model.load_state_dict(payload["state_dict"])
    model = model.to(pick_device(device) if not isinstance(device, torch.device) else device).eval()
    return model, {"cfg": cfg, "d": payload["d"], "r": payload["r"], "extra": payload["extra"]}


def learning_curve_subsets(n: int, fractions, seed: int = 0) -> list[tuple[float, np.ndarray]]:
    """Nested training subsets (fraction, indices) shared by the NN and baseline learning curves."""
    perm = np.random.default_rng(seed).permutation(n)
    return [(float(f), perm[:max(2, int(round(f * n)))]) for f in fractions]


def learning_curve(train: Dataset, test: Dataset, cfg: TrainConfig,
                   fractions=(0.01, 0.03, 0.1, 0.3, 1.0), log=print) -> list[dict]:
    rows = []
    for f, idx in learning_curve_subsets(len(train.labels), fractions, cfg.seed):
        k = len(idx)
        sub = train.subset(idx)
        fit_idx, val_idx = training_split_indices(k, cfg.val_frac, cfg.seed)
        fit, val = sub.subset(fit_idx), sub.subset(val_idx)
        # Small subsets would get only a handful of optimizer steps at the full batch size and
        # look worse than they are, so keep at least ~32 steps per epoch.
        sub_cfg = replace(cfg, batch_size=min(cfg.batch_size, max(64, len(fit) // 32)))
        model, _ = train_model(fit, sub_cfg, val=val, log=lambda *_: None)
        pred = predict(model, test.detectors, device=cfg.device and pick_device(cfg.device))
        errs = int((pred != test.labels).sum())
        n = len(test.labels)
        p = errs / max(n, 1)
        row = {"decoder": f"nn_{cfg.kind}", "fraction": float(f), "n_train": int(len(fit)), "n_test": int(n),
               "errors": errs, "p_L": p, "stderr": float(np.sqrt(p * (1 - p) / max(n, 1)))}
        log(f"fraction {f:g} n_train {len(fit)} p_L {p:.4f} +- {row['stderr']:.4f}")
        rows.append(row)
    return rows


class NNDecoder:
    def __init__(self, cfg: TrainConfig | None = None, model=None, device=None):
        self.cfg = cfg or TrainConfig()
        self.model = model
        self.device = device if device is not None else self.cfg.device
        self.name = f"nn_{self.cfg.kind}"
        self.history: list[dict] = []

    def fit(self, train: Dataset) -> "NNDecoder":
        self.model, self.history = train_model(train, self.cfg, log=lambda *_: None)
        return self

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NNDecoder is not fitted")
        return predict(self.model, detectors, device=self.device)
