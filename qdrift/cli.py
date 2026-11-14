"""Command line interface: ``python -m qdrift <command>``. Heavy imports happen inside each command."""
from __future__ import annotations

import os

# Aer and torch ship different OpenMP runtimes; avoid segfaults when both load in one process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/qdrift")
CU128_HINT = ("hint: this torch build has no kernel for your GPU (RTX 50-series needs CUDA 12.8+). "
              "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cu128")


# ------------------------------------------------------------------------------ helpers
def _csv_list(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out.extend(x.strip() for x in str(v).split(",") if x.strip())
    return out


def _parse_configs(text: str) -> list[tuple[int, int]]:
    out = []
    for tok in _csv_list([text]):
        try:
            d, r = tok.lower().split("x")
            out.append((int(d), int(r)))
        except ValueError:
            raise ValueError(f"bad config {tok!r}; use DxR, e.g. 7x5") from None
    if not out:
        raise ValueError("no configs given")
    return out


def _fractions(text: str) -> list[float]:
    try:
        return [float(x) for x in _csv_list([text])]
    except ValueError:
        raise ValueError(f"bad --fractions {text!r}") from None


def _describe(ds) -> str:
    dates = sorted({str(m.get("date", ""))[:10] for m in ds.metas if m.get("date")})
    bases = sorted({str(m.get("basis", "")) for m in ds.metas})
    backs = sorted({str(m.get("backend", "")) for m in ds.metas})
    return (f"backend={','.join(backs)} basis={','.join(bases)} snapshots={len(ds.metas)} "
            f"dates={','.join(dates) or 'n/a'} samples={len(ds)}")


def _load_split(backends, d, r, basis, parsed_dir, test_frac, seed, split_strategy='iid'):
    from qdrift.data.stein import load_dataset, split
    ds = load_dataset(backends, d, r, basis, parsed_dir=parsed_dir)
    tr, te = split(ds, test_frac, seed, strategy=split_strategy)
    return ds, ds.subset(tr), ds.subset(te)


def _available_r(backends, d, parsed_dir, basis=None) -> list[int]:
    from qdrift.data.stein import build_index
    rs = sorted({m["r"] for m in build_index(parsed_dir)
                 if m["backend"] in backends and m["d"] == d and (basis is None or m["basis"] == basis)})
    if not rs:
        raise ValueError(f"no parsed data for backends={backends}, d={d}. "
                         "Run `python -m qdrift download` and `python -m qdrift parse` first.")
    return rs


# ------------------------------------------------------------------------------ commands
def cmd_download(a) -> int:
    from qdrift.data.stein import BACKENDS, download
    names = list(BACKENDS) if a.all else (_csv_list(a.backends) or ["ibm_fez"])
    for b in names:
        print(f"downloading {b} ...")
        print(f"  -> {download(b, a.raw_dir, force=a.force)}")
    return 0


def cmd_parse(a) -> int:
    from qdrift.data.stein import build_index, parse_backend
    raw = Path(a.raw_dir)
    names = _csv_list(a.backends)
    if not names:
        names = sorted(p.name for p in raw.iterdir()
                       if p.is_dir() and not p.name.startswith("_") and (p / p.name).is_dir()) if raw.is_dir() else []
        if not names:
            raise FileNotFoundError(f"no <name>/<name>/ folders under {raw}. Run `python -m qdrift download` first.")
    for b in names:
        outs = parse_backend(b, a.raw_dir, a.parsed_dir, workers=a.workers, force=a.force)
        n = sum(m["n_samples"] for m in build_index(a.parsed_dir) if m["backend"] == b)
        print(f"{b}: {len(outs)} snapshots, {n} samples")
    return 0


def cmd_index(a) -> int:
    from qdrift.data.stein import build_index
    rows = sorted(build_index(a.parsed_dir), key=lambda m: (m["backend"], m["d"], m["r"], str(m["date"])))
    if not rows:
        raise FileNotFoundError(f"no parsed data in {a.parsed_dir}")
    head = ["backend", "d", "r", "basis", "date", "n_samples"]
    body = [[str(m.get(k, "")) for k in head] for m in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    for b in body:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(b)))
    return 0


def cmd_baseline(a) -> int:
    from qdrift.decoders.baselines import format_table
    from qdrift.evaluate import RESULTS as RES, evaluate_config, write_csv
    backends = _csv_list(a.backend) or ["ibm_fez"]
    rs = [a.r] if a.r is not None else _available_r(backends, a.d, a.parsed_dir, a.basis)
    all_rows = []
    for r in rs:
        ds, train, test = _load_split(backends, a.d, r, a.basis, a.parsed_dir, a.test_frac, a.seed, a.split_strategy)
        print(f"\n[d={a.d} r={r}] {_describe(ds)}")
        rows = evaluate_config(train, test)
        for row in rows:
            row.update(backend="+".join(backends), basis=a.basis or "all")
        print(format_table(rows))
        all_rows.extend(rows)
    out = Path(a.out) if a.out else RES / f"baseline_{'+'.join(backends)}_d{a.d}.csv"
    print(f"\nwrote {write_csv(all_rows, out)}")
    return 0


def _make_cfg(a, TrainConfig):
    return TrainConfig(kind=a.kind, width=a.width, depth=a.depth, dropout=a.dropout, epochs=a.epochs,
                       batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
                       device=None if a.device == "auto" else a.device, compile=a.compile,
                       amp=not a.no_amp, seed=a.seed, val_frac=a.val_frac)


def cmd_train(a) -> int:
    import torch
    from qdrift.decoders.baselines import format_table
    from qdrift.evaluate import (baseline_learning_curve, evaluate_config, plot_learning_curve, write_csv)
    from qdrift.data.stein import split
    from qdrift.train import (NNDecoder, TrainConfig, learning_curve, load_checkpoint, pick_device,
                              save_checkpoint, train_model)
    backends = _csv_list(a.backend)
    if not backends:
        raise ValueError("--backend is required")
    tag = "+".join(backends)
    dev = pick_device(a.device)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    cfg = _make_cfg(a, TrainConfig)
    cfg.device = str(dev)
    if dev.type == "cuda":
        try:
            (torch.zeros(8, device=dev) + 1).sum().item()
        except RuntimeError as e:
            if "no kernel image" in str(e):
                print(CU128_HINT, file=sys.stderr)
            raise
    ds, train, test = _load_split(backends, a.d, a.r, a.basis, a.parsed_dir, a.test_frac, a.seed, a.split_strategy)
    print(_describe(ds))
    init_model = None
    if a.init:
        init_model, info = load_checkpoint(a.init, dev)
        c = info["cfg"]
        if (info["d"], info["r"]) != (a.d, a.r):
            raise ValueError(f"--init checkpoint is for d={info['d']} r={info['r']}, not d={a.d} r={a.r}")
        cfg.kind, cfg.width, cfg.depth, cfg.dropout = c["kind"], c["width"], c["depth"], c["dropout"]
        print(f"fine-tuning from {a.init} ({cfg.kind} width={cfg.width} depth={cfg.depth})")
    stem = f"{tag}_d{a.d}_r{a.r}_{cfg.kind}"
    try:
        if a.learning_curve:
            fr = _fractions(a.fractions)
            rows = learning_curve(train, test, cfg, fr, log=print)
            rows += baseline_learning_curve(train, test, fr, seed=cfg.seed, val_frac=cfg.val_frac)
            csv_path = write_csv(rows, RESULTS / f"learning_curve_{stem}.csv")
            png = plot_learning_curve(rows, csv_path.with_suffix(".png"))
            print(f"wrote {csv_path}\nwrote {png}")
            return 0
        fit_idx, val_idx = split(train, cfg.val_frac, cfg.seed + 1, strategy=a.split_strategy)
        fit_train, val = train.subset(fit_idx), train.subset(val_idx)
        model, hist = train_model(fit_train, cfg, val=val, log=print, model=init_model)
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print(CU128_HINT, file=sys.stderr)
        raise
    nn = NNDecoder(cfg, model=model, device=dev)
    rows = evaluate_config(fit_train, test, nn)
    print(format_table(rows))
    ckpt = Path(a.out) if a.out else RESULTS / "checkpoints" / f"{stem}.pt"
    save_checkpoint(ckpt, model, cfg, a.d, a.r, extra={"backend": tag, "basis": a.basis, "seed": a.seed,
                                                       "test_frac": a.test_frac, "init": a.init})
    write_csv(hist, ckpt.with_name(ckpt.stem + "_history.csv"))
    write_csv(rows, ckpt.with_name(ckpt.stem + "_rows.csv"))
    print(f"saved checkpoint {ckpt}")
    return 0


_TRAIN_OPTS = [  # (dest, flag, type, default, help); type None = store_true
    ("kind", "--kind", str, "mlp", "model kind: mlp, conv or gru"),
    ("epochs", "--epochs", int, 30, "training epochs"),
    ("batch_size", "--batch-size", int, 16384, "batch size"),
    ("lr", "--lr", float, 2e-3, "peak learning rate"),
    ("width", "--width", int, 512, "hidden width"),
    ("depth", "--depth", int, 4, "number of layers"),
    ("dropout", "--dropout", float, 0.0, "dropout"),
    ("weight_decay", "--weight-decay", float, 1e-4, "AdamW weight decay"),
    ("device", "--device", str, "auto", "auto, cuda, mps or cpu"),
    ("compile", "--compile", None, False, "torch.compile (needs Triton on Windows)"),
    ("no_amp", "--no-amp", None, False, "disable bf16 autocast"),
    ("seed", "--seed", int, 0, "random seed"),
    ("val_frac", "--val-frac", float, 0.1, "fraction of outer training data reserved for validation"),
    ("split_strategy", "--split-strategy", str, "iid", "iid, chain, or snapshot split"),
    ("test_frac", "--test-frac", float, 0.25, "test fraction"),
    ("init", "--init", str, None, "start from this checkpoint (fine-tuning)"),
    ("learning_curve", "--learning-curve", None, False, "run a learning curve instead of one fit"),
    ("fractions", "--fractions", str, "0.01,0.03,0.1,0.3,1.0", "learning-curve training fractions"),
    ("basis", "--basis", str, None, "X or Z (default: both)"),
    ("parsed_dir", "--parsed-dir", str, "data/parsed", "parsed data folder"),
]


def _add_train_opts(p, passthrough=False):
    sup = argparse.SUPPRESS
    for dest, flag, typ, default, hlp in _TRAIN_OPTS:
        d = sup if passthrough else default
        if typ is None:
            p.add_argument(flag, dest=dest, action="store_true", default=d if passthrough else False, help=hlp)
        else:
            p.add_argument(flag, dest=dest, type=typ, default=d, help=hlp)


def cmd_sweep(a) -> int:
    configs = _parse_configs(a.configs)
    kinds = _csv_list([a.kinds])
    backend = a.backend
    cmds = []
    for d, r in configs:
        for kind in kinds:
            name = f"{backend}_d{d}_r{r}_{kind}"
            cmd = [sys.executable, "-m", "qdrift", "train", "--backend", backend, "--d", str(d), "--r", str(r),
                   "--kind", kind]
            for dest, flag, typ, _, _ in _TRAIN_OPTS:
                if dest == "kind" or not hasattr(a, dest):
                    continue
                v = getattr(a, dest)
                if typ is None:
                    if v:
                        cmd.append(flag)
                elif v is not None:
                    cmd += [flag, str(v)]
            cmds.append((name, cmd))
    if a.dry_run:
        for name, cmd in cmds:
            print(" ".join(cmd))
        return 0
    logdir = RESULTS / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    pending, running, status = list(cmds), [], {}
    import time
    while pending or running:
        while pending and len(running) < max(1, a.jobs):
            name, cmd = pending.pop(0)
            fh = open(logdir / f"{name}.log", "w")
            print(f"start {name} -> {logdir / (name + '.log')}")
            running.append((name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
        time.sleep(0.5)
        for item in running[:]:
            name, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                status[name] = proc.returncode
                running.remove(item)
                print(f"done  {name} exit={proc.returncode}")
    print("\nname".ljust(40) + "status")
    for name, _ in cmds:
        print(name.ljust(39) + ("ok" if status[name] == 0 else f"failed ({status[name]})"))
    return 0 if all(v == 0 for v in status.values()) else 1


def cmd_evaluate(a) -> int:
    from qdrift.decoders.baselines import format_table
    from qdrift.evaluate import RESULTS as RES, evaluate, plot_error_vs_d, write_csv
    configs = _parse_configs(a.configs)
    rows = evaluate(a.backend, configs, basis=a.basis, ckpt_dir=a.ckpt_dir, test_backend=a.test_backend,
                    test_frac=a.test_frac, seed=a.seed, parsed_dir=a.parsed_dir, device=a.device,
                    split_strategy=a.split_strategy)
    if a.test_backend and a.test_backend != a.backend and not any(r["test_backend"] == a.test_backend for r in rows):
        raise ValueError(f"no parsed data for test backend {a.test_backend!r} at configs {a.configs}; nothing to "
                         f"score. Run `python -m qdrift parse {a.test_backend}` or check `python -m qdrift index`.")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["test_backend"], r["d"], r["r"]), []).append(r)
    for (tb, d, r), g in groups.items():
        print(f"\n[trained on {a.backend}, scored on {tb}, d={d} r={r}]")
        print(format_table(g))
    suffix = f"_on_{a.test_backend}" if a.test_backend else ""
    out_dir = Path(a.out_dir) if a.out_dir else RES
    csv_path = write_csv(rows, out_dir / f"eval_{a.backend}{suffix}.csv")
    png = plot_error_vs_d([r for r in rows if r["test_backend"] == (a.test_backend or a.backend)],
                          csv_path.with_suffix(".png"))
    print(f"\nwrote {csv_path}\nwrote {png}")
    return 0


def cmd_run_ibm(a) -> int:
    configs = [(d, r, (a.basis or "Z").upper()) for d, r in _parse_configs(a.configs)]
    label = a.label or (f"{a.backend}_fresh" if a.real else f"{a.backend}_fake")
    if not a.real and label.endswith("_fresh"):
        raise ValueError(f"label {label!r} is reserved for real hardware runs; simulated data would get mixed "
                         f"into it. Use the default ({a.backend}_fake) or another label, or add --real.")
    out_root = Path(a.raw_dir) / label  # run_ibm writes <out_root>/<label>/d<D>_r<R>/job_<n>
    n_circ = 2 * len(configs)
    if a.real:
        if not os.environ.get("QISKIT_IBM_TOKEN"):
            raise RuntimeError("QISKIT_IBM_TOKEN is not set; refusing to contact IBM Quantum. "
                               "Export your API token as an environment variable first.")
        print(f"REAL HARDWARE run on {a.backend}: {len(configs)} snapshots, {n_circ} circuits, "
              f"{a.shots} shots each ({n_circ * a.shots} shots total), up to {a.n_chains} chains each.")
        if not a.yes and input("Type 'yes' to submit to IBM Quantum: ").strip() != "yes":
            print("aborted")
            return 1
    else:
        print(f"Using the FAKE (simulated) {a.backend} backend; no IBM hardware is contacted.")
    from qdrift.data.circuits import run_ibm
    paths = run_ibm(a.backend, out_root, configs, shots=a.shots, n_chains=a.n_chains, real=a.real,
                    seed=a.seed, label=label)
    for p in paths:
        print(f"wrote {p}")
    print(f"next: python -m qdrift parse {label} --raw-dir {a.raw_dir}")
    return 0


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m qdrift", description="Repetition-code decoders on IBM data.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="download Stein et al. data from Zenodo")
    s.add_argument("backends", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--raw-dir", default="data/raw")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("parse", help="parse raw JSON into compact .npz files")
    s.add_argument("backends", nargs="*")
    s.add_argument("--raw-dir", default="data/raw")
    s.add_argument("--parsed-dir", default="data/parsed")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_parse)

    s = sub.add_parser("index", help="list parsed snapshots")
    s.add_argument("--parsed-dir", default="data/parsed")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("baseline", help="run the non-neural decoders")
    s.add_argument("--backend", action="append", default=None, help="repeatable or comma list (default ibm_fez)")
    s.add_argument("--d", type=int, required=True)
    s.add_argument("--r", type=int, default=None)
    s.add_argument("--basis", choices=["X", "Z"], default=None)
    s.add_argument("--test-frac", type=float, default=0.25)
    s.add_argument("--split-strategy", choices=["iid", "chain", "snapshot"], default="iid")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--parsed-dir", default="data/parsed")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("train", help="train one neural decoder")
    s.add_argument("--backend", action="append", default=None, required=True)
    s.add_argument("--d", type=int, required=True)
    s.add_argument("--r", type=int, required=True)
    s.add_argument("--out", default=None, help="checkpoint path")
    _add_train_opts(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("sweep", help="launch several train runs as subprocesses")
    s.add_argument("--backend", required=True)
    s.add_argument("--configs", required=True, help="e.g. 7x5,9x7,11x11")
    s.add_argument("--kinds", default="mlp")
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    _add_train_opts(s, passthrough=True)
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("evaluate", help="baselines + checkpoints, optional drift test")
    s.add_argument("--backend", required=True)
    s.add_argument("--configs", required=True)
    s.add_argument("--basis", choices=["X", "Z"], default=None)
    s.add_argument("--ckpt-dir", default=str(RESULTS / "checkpoints"))
    s.add_argument("--test-backend", default=None)
    s.add_argument("--test-frac", type=float, default=0.25)
    s.add_argument("--split-strategy", choices=["iid", "chain", "snapshot"], default="iid")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--parsed-dir", default="data/parsed")
    s.add_argument("--out-dir", default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("run-ibm", help="generate fresh data on a fake (default) or real IBM backend")
    s.add_argument("--backend", default="ibm_fez")
    s.add_argument("--configs", required=True)
    s.add_argument("--basis", choices=["X", "Z"], default="Z")
    s.add_argument("--shots", type=int, default=4096,
                   help="shots per circuit; ~4096 gives >4000 test samples per config at 4 chains")
    s.add_argument("--n-chains", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--real", action="store_true", help="use real hardware (needs QISKIT_IBM_TOKEN)")
    s.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    s.add_argument("--label", default=None)
    s.add_argument("--raw-dir", default="data/raw")
    s.set_defaults(fn=cmd_run_ibm)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
