"""Baseline decoders for the repetition-code memory experiment.

Detector layout is (r+1, d-1): node (t, i) is the i-th check comparison in row t.
Every decoder maps detectors (N, r+1, d-1) uint8 -> predicted label (N,) uint8, where
label 1 means "data qubit 0 ended flipped".

Matching graph (see the module docs in the task):
  space edge    (t,i)-(t,i+1)      data qubit i+1 flipped (row r: its final readout)
  time edge     (t,i)-(t+1,i)      check measurement error
  diagonal      (t,i)-(t+1,i+1), (t,i+1)-(t+1,i)   mid-round data flip (fitted only)
  boundary "L"  (t,0)              data qubit 0 flipped -> the ONLY edge that flips the label
  boundary "R"  (t,d-2)            data qubit d-1 flipped
"""
from __future__ import annotations

import math

import numpy as np
import pymatching

from qdrift.data.stein import Dataset

P_MIN = 1e-6
P_MAX = 0.5 - 1e-6


# --------------------------------------------------------------------------- graph

def graph_edges(d: int, r: int, diagonals: bool = False):
    """Return (pair_edges, boundary_edges) on nodes (t, i).

    pair_edges: list of ((t,i), (t',i')); boundary_edges: list of ((t,i), "L"/"R").
    """
    if d < 3:
        raise ValueError("matching decoders need d >= 3")
    n = d - 1
    pairs = []
    for t in range(r + 1):
        for i in range(n - 1):
            pairs.append(((t, i), (t, i + 1)))
    for t in range(r):
        for i in range(n):
            pairs.append(((t, i), (t + 1, i)))
    if diagonals:
        for t in range(r):
            for i in range(n - 1):
                pairs.append(((t, i), (t + 1, i + 1)))
                pairs.append(((t, i + 1), (t + 1, i)))
    bounds = []
    for t in range(r + 1):
        bounds.append(((t, 0), "L"))
        bounds.append(((t, n - 1), "R"))
    return pairs, bounds


def _weight(p: float) -> float:
    return math.log((1.0 - p) / p)


def _clip_p(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p) & (p > 0), p, P_MIN)
    return np.clip(p, P_MIN, P_MAX)


def _build_matching(d, r, edge_probs=None, pairs=None, bounds=None) -> pymatching.Matching:
    """edge_probs None -> all weights 1."""
    n = d - 1
    idx = lambda node: node[0] * n + node[1]  # noqa: E731  row-major, matches reshape
    m = pymatching.Matching()
    for a, b in pairs:
        if edge_probs is None:
            m.add_edge(idx(a), idx(b), fault_ids=set(), weight=1.0)
        else:
            p = edge_probs[(a, b)]
            m.add_edge(idx(a), idx(b), fault_ids=set(), weight=_weight(p), error_probability=p)
    for a, side in bounds:
        fid = {0} if side == "L" else set()
        if edge_probs is None:
            m.add_boundary_edge(idx(a), fault_ids=fid, weight=1.0)
        else:
            p = edge_probs[(a, side)]
            m.add_boundary_edge(idx(a), fault_ids=fid, weight=_weight(p), error_probability=p)
    return m


def fit_edge_probs(detectors: np.ndarray, diagonals: bool = True,
                   boundary_floor: bool = False) -> dict:
    """Pairwise (2-point correlation) estimate of every graph edge probability.

    A boundary probability is what is left of a detector's rate after its pair edges are
    accounted for. On real hardware that remainder often comes out at or below zero (the
    noise is not independent edge flips), which would make the label-flipping left boundary
    edges almost unusable and the decoder worse than equal weights at d >= 7. With
    ``boundary_floor`` those estimates are floored at the row's median space-edge probability.
    """
    N, R1, n = detectors.shape
    d, r = n + 1, R1 - 1
    X = detectors.reshape(N, -1).astype(np.float64)
    mu = X.mean(axis=0)
    xx = (X.T @ X) / max(N, 1)
    cov = xx - np.outer(mu, mu)
    denom = 1.0 - 2.0 * mu[:, None] - 2.0 * mu[None, :] + 4.0 * xx
    with np.errstate(divide="ignore", invalid="ignore"):
        P = 0.5 - np.sqrt(0.25 - cov / denom)
    P = _clip_p(P)

    pairs, bounds = graph_edges(d, r, diagonals)
    idx = lambda node: node[0] * n + node[1]  # noqa: E731
    probs = {}
    log_prod = np.zeros(n * R1)  # sum log(1 - 2 p_ij) over graph neighbours
    for a, b in pairs:
        p = float(P[idx(a), idx(b)])
        probs[(a, b)] = p
        log_prod[idx(a)] += math.log1p(-2.0 * p)
        log_prod[idx(b)] += math.log1p(-2.0 * p)
    # Floor for boundary edges: the median space-edge probability of the same row, i.e. a
    # chain-end data qubit is taken to be at least as error-prone as a typical interior one.
    row_floor = {}
    if boundary_floor:
        for t in range(R1):
            row = [probs[((t, i), (t, i + 1))] for i in range(n - 1)]
            row_floor[t] = float(np.median(row)) if row else P_MIN
    for a, side in bounds:
        k = idx(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (1.0 - 2.0 * mu[k]) / math.exp(log_prod[k])
        p = float(_clip_p((1.0 - ratio) / 2.0))
        probs[(a, side)] = max(p, row_floor.get(a[0], P_MIN))
    return probs


# --------------------------------------------------------------------------- decoders

class Decoder:
    name: str = "base"

    def fit(self, train: Dataset) -> "Decoder":
        return self

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


class NoDecode(Decoder):
    """Raw readout of data qubit 0: always predicts 'not flipped'."""
    name = "none"

    def predict(self, detectors):
        return np.zeros(len(detectors), dtype=np.uint8)


class _MatchingBase(Decoder):
    def __init__(self, d=None, r=None):
        self.d, self.r = d, r
        self._m = None

    def _make(self):
        raise NotImplementedError

    def _ensure(self, detectors):
        if self._m is None:
            if self.d is None:
                self.r, self.d = detectors.shape[1] - 1, detectors.shape[2] + 1
            self._m = self._make()

    def predict(self, detectors):
        detectors = np.asarray(detectors, dtype=np.uint8)
        N = len(detectors)
        if N == 0:
            return np.zeros(0, dtype=np.uint8)
        self._ensure(detectors)
        flat = np.ascontiguousarray(detectors.reshape(N, -1))
        out = self._m.decode_batch(flat)
        return np.asarray(out, dtype=np.uint8).reshape(N, -1)[:, 0].copy()


class MatchingEqual(_MatchingBase):
    """MWPM with every edge weight 1 and no diagonal edges."""
    name = "mwpm_equal"

    def fit(self, train):
        self.d, self.r = train.d, train.r
        self._m = self._make()
        return self

    def _make(self):
        pairs, bounds = graph_edges(self.d, self.r, diagonals=False)
        return _build_matching(self.d, self.r, None, pairs, bounds)


class MatchingFitted(_MatchingBase):
    """MWPM with weights log((1-p)/p) from pairwise detector correlations on the train set."""
    name = "mwpm_fitted"

    def __init__(self, d=None, r=None, diagonals=True, boundary_floor=False):
        super().__init__(d, r)
        self.diagonals = diagonals
        self.boundary_floor = boundary_floor
        self.edge_probs: dict = {}

    def fit(self, train):
        self.fit_detectors(train.detectors)
        return self

    def fit_detectors(self, detectors):
        detectors = np.asarray(detectors, dtype=np.uint8)
        self.r, self.d = detectors.shape[1] - 1, detectors.shape[2] + 1
        self.edge_probs = fit_edge_probs(detectors, self.diagonals, self.boundary_floor)
        self._m = self._make()
        return self

    def _make(self):
        if not self.edge_probs:
            raise RuntimeError("MatchingFitted.predict called before fit")
        pairs, bounds = graph_edges(self.d, self.r, self.diagonals)
        return _build_matching(self.d, self.r, self.edge_probs, pairs, bounds)


def _pattern_keys(detectors: np.ndarray) -> np.ndarray:
    """One sortable key per row: uint64 when <= 64 bits, else fixed-width bytes."""
    N = len(detectors)
    flat = np.ascontiguousarray(detectors.reshape(N, -1), dtype=np.uint8)
    packed = np.packbits(flat, axis=1, bitorder="little")
    nb = packed.shape[1]
    if nb <= 8:
        buf = np.zeros((N, 8), dtype=np.uint8)
        buf[:, :nb] = packed
        return buf.view("<u8").ravel()
    # sentinel 0x01 byte so numpy's trailing-NUL stripping of 'S' cannot merge keys
    buf = np.concatenate([packed, np.ones((N, 1), dtype=np.uint8)], axis=1)
    return np.ascontiguousarray(buf).view(f"S{nb + 1}").ravel()


class LookupTable(Decoder):
    """Empirical MAP per full detector pattern; ties / unseen -> fitted matching."""
    name = "lookup"

    def __init__(self):
        self.keys = None
        self.decision = None  # 0, 1, or 255 (undecided)
        self.fallback = MatchingFitted()

    def fit(self, train):
        keys = _pattern_keys(train.detectors)
        uk, inv = np.unique(keys, return_inverse=True)
        tot = np.bincount(inv.ravel(), minlength=len(uk))
        ones = np.bincount(inv.ravel(), weights=train.labels.astype(np.float64), minlength=len(uk))
        dec = np.full(len(uk), 255, dtype=np.uint8)
        dec[2 * ones > tot] = 1
        dec[2 * ones < tot] = 0
        self.keys, self.decision = uk, dec
        self.fallback.fit(train)
        return self

    def predict(self, detectors):
        detectors = np.asarray(detectors, dtype=np.uint8)
        N = len(detectors)
        out = np.zeros(N, dtype=np.uint8)
        if N == 0:
            return out
        keys = _pattern_keys(detectors)
        pos = np.searchsorted(self.keys, keys)
        pos_c = np.minimum(pos, len(self.keys) - 1)
        found = self.keys[pos_c] == keys
        dec = np.where(found, self.decision[pos_c], 255).astype(np.uint8)
        known = dec != 255
        out[known] = dec[known]
        if (~known).any():
            out[~known] = self.fallback.predict(detectors[~known])
        return out


def make_baselines(d, r) -> list:
    decs = [NoDecode(), MatchingEqual(d, r), MatchingFitted(d, r, diagonals=True)]
    if (d - 1) * (r + 1) <= 24:
        decs.append(LookupTable())
    return decs


# --------------------------------------------------------------------------- metrics

def logical_error(pred, labels) -> dict:
    pred = np.asarray(pred).astype(np.uint8).ravel()
    labels = np.asarray(labels).astype(np.uint8).ravel()
    n = int(labels.size)
    errors = int(np.count_nonzero(pred != labels))
    p = errors / n if n else float("nan")
    stderr = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return {"n": n, "errors": errors, "p_L": p, "stderr": stderr}


def error_per_round(p_L, r) -> float:
    if not (p_L < 0.5):
        return 0.5
    return (1.0 - (1.0 - 2.0 * p_L) ** (1.0 / r)) / 2.0


def run_baselines(train: Dataset, test: Dataset, decoders=None) -> list:
    if decoders is None:
        decoders = make_baselines(train.d, train.r)
    rows = []
    for dec in decoders:
        dec.fit(train)
        pred = dec.predict(test.detectors)
        le = logical_error(pred, test.labels)
        per_state = {}
        for b in (0, 1):
            m = test.states == b
            per_state[b] = logical_error(pred[m], test.labels[m])["p_L"] if m.any() else float("nan")
        rows.append({
            "decoder": dec.name, "d": test.d, "r": test.r,
            "n_train": len(train.labels), "n_test": le["n"], "errors": le["errors"],
            "p_L": le["p_L"], "stderr": le["stderr"], "eps_round": error_per_round(le["p_L"], test.r),
            "p_L_state0": per_state[0], "p_L_state1": per_state[1],
        })
    return rows


def format_table(rows) -> str:
    cols = [
        ("decoder", "decoder", "{}"), ("d", "d", "{}"), ("r", "r", "{}"),
        ("n_train", "n_train", "{}"), ("n_test", "n_test", "{}"), ("errors", "errors", "{}"),
        ("p_L", "p_L", "{:.5f}"), ("stderr", "+-", "{:.5f}"), ("eps_round", "eps/round", "{:.5f}"),
        ("p_L_state0", "p_L|0", "{:.5f}"), ("p_L_state1", "p_L|1", "{:.5f}"),
    ]
    body = [[fmt.format(row[k]) for k, _, fmt in cols] for row in rows]
    head = [h for _, h, _ in cols]
    widths = [max(len(h), *(len(b[j]) for b in body)) if body else len(h) for j, h in enumerate(head)]

    def line(cells):
        return "  ".join(c.ljust(w) if j == 0 else c.rjust(w) for j, (c, w) in enumerate(zip(cells, widths)))

    out = [line(head), "  ".join("-" * w for w in widths)] + [line(b) for b in body]
    return "\n".join(out)
