"""Loader and parser for the Stein et al. repetition-code dataset (see docs/stein_format.md).

Raw files are untrusted: they are only ever read with ``json.loads``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

ZENODO_RECORD = "20768087"
BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_pittsburgh")
DEFAULT_RAW = Path("data/raw")
DEFAULT_PARSED = Path("data/parsed")

_DATA_PREFIX = "c_data_"
_SYND_PREFIX = "c_syndrome_"


@dataclass
class Dataset:
    detectors: np.ndarray  # uint8 (N, r+1, d-1)
    labels: np.ndarray  # uint8 (N,)  q0 ^ b
    states: np.ndarray  # uint8 (N,)
    data: np.ndarray  # uint8 (N, d)
    chain: np.ndarray  # int16 (N,)
    snapshot: np.ndarray  # int32 (N,) index into metas
    metas: list[dict]

    @property
    def d(self) -> int:
        return int(self.data.shape[1])

    @property
    def r(self) -> int:
        return int(self.detectors.shape[1]) - 1

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def subset(self, idx) -> "Dataset":
        return Dataset(
            self.detectors[idx], self.labels[idx], self.states[idx], self.data[idx],
            self.chain[idx], self.snapshot[idx], self.metas,
        )


def dataset_fingerprint(ds: Dataset) -> str:
    """Stable SHA-256 identity for the exact parsed dataset used by a decoder."""
    digest = hashlib.sha256()
    for name in ("detectors", "labels", "states", "data", "chain", "snapshot"):
        arr = np.ascontiguousarray(getattr(ds, name))
        digest.update(name.encode("utf-8") + b"\\0")
        digest.update(arr.dtype.str.encode("ascii") + b"\\0")
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(arr).cast("B"))
    meta = json.dumps(ds.metas, sort_keys=True, separators=(",", ":"), default=str)
    digest.update(meta.encode("utf-8"))
    return digest.hexdigest()


# ----------------------------------------------------------------------------- core math
def detectors_from_raw(syndrome, data, state):
    """Return (detectors (N, r+1, d-1) uint8, labels (N,) uint8). See docs/stein_format.md."""
    syndrome = np.asarray(syndrome)
    data = np.asarray(data)
    if data.ndim != 2 or syndrome.ndim != 2 or syndrome.shape[0] != data.shape[0]:
        raise ValueError(f"bad shapes: syndrome {syndrome.shape}, data {data.shape}")
    n, d = data.shape
    if d < 2 or syndrome.shape[1] == 0 or syndrome.shape[1] % (d - 1) != 0:
        raise ValueError(
            f"syndrome width {syndrome.shape[1]} is not a positive multiple of d-1={d - 1}"
        )
    for name, a in (("syndrome", syndrome), ("data", data)):
        if a.size and (a.min() < 0 or a.max() > 1):
            raise ValueError(f"{name} values must be 0/1")
    r = syndrome.shape[1] // (d - 1)
    st = np.asarray(state)
    if st.ndim == 1 and st.shape[0] != n:
        raise ValueError(f"state has length {st.shape[0]}, expected {n}")
    if st.size and (st.min() < 0 or st.max() > 1):
        raise ValueError("state values must be 0/1")
    m = syndrome.astype(np.uint8).reshape(n, r, d - 1)
    q = data.astype(np.uint8)
    det = np.empty((n, r + 1, d - 1), dtype=np.uint8)
    det[:, 0] = m[:, 0]
    if r > 1:
        det[:, 1:r] = m[:, 1:] ^ m[:, :-1]
    det[:, r] = (q[:, :-1] ^ q[:, 1:]) ^ m[:, r - 1]
    labels = (q[:, 0] ^ st.astype(np.uint8)).astype(np.uint8)
    return det, labels


# ----------------------------------------------------------------------------- download
def _is_unsafe_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return True
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        return True
    if PureWindowsPath(name).drive:
        return True
    return ".." in PurePosixPath(name).parts


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar archive, refusing unsafe paths and anything but files/directories."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            if _is_unsafe_name(m.name):
                raise ValueError(f"unsafe path in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported member type in archive: {m.name!r}")
        tf.extractall(dest, members=members, filter="data")


def download(backend, raw_dir=DEFAULT_RAW, force=False) -> Path:
    """Fetch <backend>.tar.gz from Zenodo and extract it. Returns raw_dir/<backend>/<backend>."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    raw_dir = Path(raw_dir)
    out = raw_dir / backend / backend
    if out.is_dir() and not force:
        return out
    arch_dir = raw_dir / "_archives"
    arch_dir.mkdir(parents=True, exist_ok=True)
    archive = arch_dir / f"{backend}.tar.gz"
    if force or not archive.exists():
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{backend}.tar.gz?download=1"
        tmp = archive.with_name(archive.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "qdrift"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, 1 << 20)
        os.replace(tmp, archive)
    if force and (raw_dir / backend).exists():
        shutil.rmtree(raw_dir / backend)
    _safe_extract(archive, raw_dir / backend)
    if not out.is_dir():
        raise FileNotFoundError(f"archive did not contain {backend}/ ; looked in {out}")
    return out


# ----------------------------------------------------------------------------- parsing
def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.loads(fh.read())


def _chain_qubits(suffix: str, d: int, job: Path) -> list[int]:
    try:
        qs = [int(x) for x in suffix.split("_")]
    except ValueError:
        raise ValueError(f"{job}: cannot parse chain name suffix {suffix!r}") from None
    if len(qs) != 2 * d - 1:
        raise ValueError(f"{job}: chain {suffix!r} has {len(qs)} qubits, expected {2 * d - 1}")
    return qs


def _to_bits(rows, shape, what: str, job: Path) -> np.ndarray:
    try:
        arr = np.asarray(rows, dtype=np.uint8)
    except (OverflowError, ValueError, TypeError) as e:
        raise ValueError(f"{job}: cannot read {what} as 0/1 array: {e}") from None
    if arr.shape != shape:
        raise ValueError(f"{job}: {what} has shape {arr.shape}, expected {shape}")
    if arr.size and arr.max() > 1:
        raise ValueError(f"{job}: {what} contains values other than 0/1")
    return arr


def parse_snapshot(job_dir) -> Dataset:
    job = Path(job_dir)
    info = _read_json(job / "info.json")
    try:
        shots, d, r = int(info["shots"]), int(info["d"]), int(info["rounds"])
        n_chains = int(info["n_chains"])
        backend = str(info["backend"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{job}: bad info.json ({e!r})") from None
    basis = str(info.get("basis", ""))
    date = ""
    cal = job / "calibration.json"
    if cal.exists():
        date = str(_read_json(cal).get("last_update_date", ""))[:10]
    parts = job.resolve().parts if len(job.parts) < 3 else job.parts
    snap_id = "/".join(parts[-3:])

    entries = _read_json(job / "bitstrings.json")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{job}: bitstrings.json must be a non-empty list")
    parsed = []
    for e in entries:
        try:
            b = int(e["metadata"]["logical_state"])
            regs = e["per_shot_cregs"]
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(f"{job}: malformed bitstrings entry ({ex!r})") from None
        if b not in (0, 1):
            raise ValueError(f"{job}: logical_state {b} not in (0, 1)")
        parsed.append((b, regs))
    parsed.sort(key=lambda t: t[0])  # state 0 first; stable
    if [b for b, _ in parsed] != sorted(set(b for b, _ in parsed)):
        raise ValueError(f"{job}: duplicate logical states")

    dets, labs, sts, dats, chns, chain_names = [], [], [], [], [], None
    for b, regs in parsed:
        keys = [k for k in regs if k.startswith(_DATA_PREFIX)]
        if len(keys) != n_chains or len(regs) != 2 * n_chains:
            raise ValueError(f"{job}: expected {n_chains} chains, found {len(keys)} data registers")
        names = []
        for ci, k in enumerate(keys):
            suffix = k[len(_DATA_PREFIX):]
            sk = _SYND_PREFIX + suffix
            if sk not in regs:
                raise ValueError(f"{job}: missing register {sk}")
            q = _to_bits(regs[k], (shots, d), f"{k}", job)
            s = _to_bits(regs[sk], (shots, r * (d - 1)), f"{sk}", job)
            det, lab = detectors_from_raw(s, q, b)
            dets.append(det)
            labs.append(lab)
            dats.append(q)
            sts.append(np.full(shots, b, dtype=np.uint8))
            chns.append(np.full(shots, ci, dtype=np.int16))
            names.append(suffix)
        if chain_names is None:
            chain_names = names
        elif names != chain_names:
            raise ValueError(f"{job}: chains differ between logical states")
    chains = [_chain_qubits(s, d, job) for s in chain_names]
    meta = {
        "snapshot": snap_id, "backend": backend, "d": d, "r": r, "basis": basis,
        "date": date, "shots": shots, "n_chains": n_chains, "chains": chains,
    }
    n = sum(len(x) for x in labs)
    return Dataset(
        np.concatenate(dets), np.concatenate(labs), np.concatenate(sts), np.concatenate(dats),
        np.concatenate(chns), np.zeros(n, dtype=np.int32), [meta],
    )


def save_snapshot(ds: Dataset, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, detectors=ds.detectors, labels=ds.labels, states=ds.states, data=ds.data,
            chain=ds.chain, snapshot=ds.snapshot, meta=np.array(json.dumps(ds.metas)),
        )
    os.replace(tmp, path)


def load_snapshot(path) -> Dataset:
    with np.load(path, allow_pickle=False) as z:
        metas = json.loads(str(z["meta"]))
        if isinstance(metas, dict):
            metas = [metas]
        return Dataset(
            z["detectors"], z["labels"], z["states"], z["data"], z["chain"],
            z["snapshot"], metas,
        )


def _parse_and_save(args) -> str:
    job, out = args
    save_snapshot(parse_snapshot(job), out)
    return str(out)


def _job_dirs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.glob("d*_r*/job_*") if p.is_dir() and (p / "info.json").exists()
    )


def parse_backend(backend, raw_dir=DEFAULT_RAW, parsed_dir=DEFAULT_PARSED, workers=None,
                  force=False) -> list[Path]:
    raw_dir, parsed_dir = Path(raw_dir), Path(parsed_dir)
    root = raw_dir / backend / backend
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Run `python -m qdrift download {backend}` first."
        )
    jobs = _job_dirs(root)
    if not jobs:
        raise FileNotFoundError(f"no job folders under {root}")
    outs = [parsed_dir / backend / j.parent.name / f"{j.name}.npz" for j in jobs]
    todo = [(j, o) for j, o in zip(jobs, outs) if force or not o.exists()]
    if todo:
        if workers is None:
            workers = min(os.cpu_count() or 1, len(todo))
        if workers <= 1 or len(todo) == 1:
            for t in todo:
                _parse_and_save(t)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_parse_and_save, todo))
    return outs


# ----------------------------------------------------------------------------- cache access
def build_index(parsed_dir=DEFAULT_PARSED) -> list[dict]:
    parsed_dir = Path(parsed_dir)
    out = []
    for p in sorted(parsed_dir.glob("*/d*_r*/job_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            metas = json.loads(str(z["meta"]))
            n = int(z["labels"].shape[0])
        meta = dict(metas[0] if isinstance(metas, list) else metas)
        meta["path"] = str(p)
        meta["n_samples"] = n
        out.append(meta)
    return out


def concat(datasets) -> Dataset:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("concat needs at least one dataset")
    metas, snaps, off = [], [], 0
    for ds in datasets:
        metas.extend(ds.metas)
        snaps.append(ds.snapshot + np.int32(off))
        off += len(ds.metas)
    return Dataset(
        np.concatenate([x.detectors for x in datasets]),
        np.concatenate([x.labels for x in datasets]),
        np.concatenate([x.states for x in datasets]),
        np.concatenate([x.data for x in datasets]),
        np.concatenate([x.chain for x in datasets]),
        np.concatenate(snaps).astype(np.int32),
        metas,
    )


def load_dataset(backends, d, r, basis=None, parsed_dir=DEFAULT_PARSED, snapshots=None) -> Dataset:
    if isinstance(backends, str):
        backends = [backends]
    backends = list(backends)
    index = build_index(parsed_dir)
    if not index:
        raise FileNotFoundError(
            f"no parsed data in {parsed_dir}. Run `python -m qdrift download <backend>` "
            "then `python -m qdrift parse <backend>`."
        )
    want = set(snapshots) if snapshots is not None else None
    sel = [
        m for m in index
        if m["backend"] in backends and m["d"] == d and m["r"] == r
        and (basis is None or m["basis"] == basis)
        and (want is None or m["snapshot"] in want)
    ]
    if not sel:
        raise ValueError(
            f"no parsed snapshots for backends={backends}, d={d}, r={r}, basis={basis}. "
            "Run `python -m qdrift download <backend>` and `python -m qdrift parse <backend>`; "
            "available: " + ", ".join(sorted({f"{m['backend']}/d{m['d']}_r{m['r']}" for m in index}))
        )
    sel.sort(key=lambda m: m["snapshot"])
    return concat(load_snapshot(m["path"]) for m in sel)


def split(ds: Dataset, test_frac=0.25, seed=0, strategy="iid"):
    """Return train/test indices using IID, held-out-chain, or held-out-snapshot splitting.

    iid preserves the original per-(snapshot, state) shot split. chain holds out complete
    physical chains within each snapshot (both prepared states stay together). snapshot
    holds out complete snapshots and tests transfer across calibration sessions.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    if strategy not in {"iid", "chain", "snapshot"}:
        raise ValueError("strategy must be one of: iid, chain, snapshot")

    n = len(ds)
    if strategy == "iid":
        key = ds.snapshot.astype(np.int64) * 2 + ds.states.astype(np.int64)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        bounds = np.flatnonzero(np.diff(sk)) + 1
        test_parts = []
        for grp in np.split(order, bounds):
            if grp.size == 0:
                continue
            k = int(key[grp[0]])
            snap_id, state = ds.metas[k // 2]["snapshot"], k % 2
            rng = np.random.default_rng([seed, zlib.crc32(snap_id.encode()), state])
            perm = rng.permutation(grp.size)
            n_test = min(max(int(round(test_frac * grp.size)), 1), grp.size - 1) if grp.size > 1 else 0
            test_parts.append(grp[perm[:n_test]])
        test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.zeros(0, dtype=np.int64)
    elif strategy == "snapshot":
        groups = np.unique(ds.snapshot)
        if len(groups) < 2:
            raise ValueError("snapshot split requires at least two snapshots")
        rng = np.random.default_rng(seed)
        n_test = min(max(int(round(test_frac * len(groups))), 1), len(groups) - 1)
        held = groups[rng.permutation(len(groups))[:n_test]]
        test_idx = np.flatnonzero(np.isin(ds.snapshot, held))
    else:
        test_mask = np.zeros(n, dtype=bool)
        for snap in np.unique(ds.snapshot):
            rows = np.flatnonzero(ds.snapshot == snap)
            chains = np.unique(ds.chain[rows])
            if len(chains) < 2:
                raise ValueError(
                    f"chain split requires at least two chains per snapshot; snapshot "
                    f"{ds.metas[int(snap)].get('snapshot', snap)!r} has {len(chains)}"
                )
            snap_id = ds.metas[int(snap)]["snapshot"]
            rng = np.random.default_rng([seed, zlib.crc32(snap_id.encode())])
            n_test = min(max(int(round(test_frac * len(chains))), 1), len(chains) - 1)
            held = chains[rng.permutation(len(chains))[:n_test]]
            test_mask[rows[np.isin(ds.chain[rows], held)]] = True
        test_idx = np.flatnonzero(test_mask)

    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    return np.flatnonzero(mask), test_idx
