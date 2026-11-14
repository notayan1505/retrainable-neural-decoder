"""Evaluation helpers: baselines vs. neural decoders, drift tests, learning curves, plots."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qdrift.data.stein import DEFAULT_PARSED, Dataset, load_dataset, split
from qdrift.decoders.baselines import (LookupTable, MatchingFitted, error_per_round, logical_error,
                                       make_baselines, run_baselines)

RESULTS = Path("results/qdrift")


class _Fitted:
    """Wrap an already-fitted decoder so run_baselines does not refit it."""

    def __init__(self, dec):
        self.dec, self.name = dec, dec.name

    def fit(self, train):
        return self

    def predict(self, detectors):
        return self.dec.predict(detectors)


def evaluate_config(train: Dataset, test: Dataset, nn=None) -> list[dict]:
    rows = run_baselines(train, test)
    if nn is not None:
        rows.extend(run_baselines(train, test, decoders=[_Fitted(nn)]))
    return rows


def _tag(rows, **kw):
    for r in rows:
        r.update(kw)
    return rows


def evaluate(backend, configs, basis=None, ckpt_dir=None, test_backend=None, test_frac=0.25, seed=0,
             parsed_dir=DEFAULT_PARSED, device=None) -> list[dict]:
    out: list[dict] = []
    for d, r in configs:
        ds = load_dataset(backend, d, r, basis, parsed_dir=parsed_dir)
        tr_i, te_i = split(ds, test_frac, seed)
        train, test = ds.subset(tr_i), ds.subset(te_i)
        nns = []
        if ckpt_dir is not None:
            from qdrift.models.net import MODEL_KINDS
            from qdrift.train import NNDecoder, TrainConfig, load_checkpoint, pick_device
            for kind in MODEL_KINDS:
                p = Path(ckpt_dir) / f"{backend}_d{d}_r{r}_{kind}.pt"
                if p.exists():
                    # Run on THIS machine's device. The checkpoint records the device it was
                    # trained on (e.g. "cuda"), which may not exist here.
                    dev = pick_device(device)
                    model, info = load_checkpoint(p, dev)
                    cfg = TrainConfig(**{**info["cfg"], "device": str(dev)})
                    nns.append(NNDecoder(cfg, model=model, device=dev))
        base_rows = run_baselines(train, test)
        for nn in nns:
            base_rows.extend(run_baselines(train, test, decoders=[_Fitted(nn)]))
        out.extend(_tag(base_rows, backend=backend, test_backend=backend, basis=basis or "all"))
        if test_backend and test_backend != backend:
            try:
                ds2 = load_dataset(test_backend, d, r, basis, parsed_dir=parsed_dir)
            except ValueError as e:
                print(f"note: skipping d={d} r={r}: no data for test backend {test_backend} ({e})")
                continue
            tr2, te2 = split(ds2, test_frac, seed)
            train2, test2 = ds2.subset(tr2), ds2.subset(te2)
            decs = make_baselines(d, r)
            fitted = [_Fitted(x.fit(train)) for x in decs] + [_Fitted(nn) for nn in nns]
            rows = run_baselines(train, test2, decoders=fitted)
            refit = [MatchingFitted(d, r, diagonals=True)]
            refit[0].name = "mwpm_refit"
            if any(isinstance(x, LookupTable) for x in decs):
                lk = LookupTable()
                lk.name = "lookup_refit"
                refit.append(lk)
            rows.extend(run_baselines(train2, test2, decoders=refit))
            out.extend(_tag(rows, backend=backend, test_backend=test_backend, basis=basis or "all"))
    return out


def baseline_learning_curve(train: Dataset, test: Dataset, fractions, seed=0) -> list[dict]:
    from qdrift.train import learning_curve_subsets
    rows = []
    for f, idx in learning_curve_subsets(len(train.labels), fractions, seed):
        sub = train.subset(idx)
        decs = [x for x in make_baselines(train.d, train.r) if x.name in ("mwpm_fitted", "lookup")]
        for dec in decs:
            dec.fit(sub)
            le = logical_error(dec.predict(test.detectors), test.labels)
            rows.append({"decoder": dec.name, "fraction": float(f), "n_train": int(len(idx)),
                         "n_test": le["n"], "errors": le["errors"], "p_L": le["p_L"], "stderr": le["stderr"]})
    return rows


def write_csv(rows, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _eps_err(p, se, r):
    lo, hi = max(p - se, 0.0), min(p + se, 0.499)
    e = error_per_round(p, r)
    return max(e - error_per_round(lo, r), 0.0), max(error_per_round(hi, r) - e, 0.0)


def plot_error_vs_d(rows, path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maxr: dict = {}
    for r in rows:
        maxr[r["d"]] = max(maxr.get(r["d"], 0), r["r"])
    rows = [r for r in rows if r["r"] == maxr[r["d"]]]
    tb = sorted({str(r.get("test_backend", "")) for r in rows} - {""})
    fig, ax = plt.subplots(figsize=(6, 4))
    for dec in dict.fromkeys(r["decoder"] for r in rows):
        pts = sorted((r for r in rows if r["decoder"] == dec), key=lambda r: r["d"])
        x = [p["d"] for p in pts]
        y = [max(p["eps_round"], 1e-6) for p in pts]
        err = np.array([_eps_err(p["p_L"], p["stderr"], p["r"]) for p in pts]).T
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=dec)
    ax.set_yscale("log")
    ax.set_xticks(sorted(maxr))
    ax.set_xlabel("code distance d")
    ax.set_ylabel("logical error per round")
    ax.set_title("Test backend: " + (", ".join(tb) if tb else "?") + " (largest r per d)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_learning_curve(rows, path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for dec in dict.fromkeys(r["decoder"] for r in rows):
        pts = sorted((r for r in rows if r["decoder"] == dec), key=lambda r: r["n_train"])
        ax.errorbar([p["n_train"] for p in pts], [p["p_L"] for p in pts],
                    yerr=[p["stderr"] for p in pts], marker="o", capsize=3, label=dec)
    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("logical error p_L")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
