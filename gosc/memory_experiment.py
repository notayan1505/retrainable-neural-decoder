"""Repeated-round, circuit-level-noise surface-code MEMORY experiment (Qiskit/Aer).

Question answered: if we throw simulated noise at a qubit and let Qiskit code
error-correct it, does the encoded qubit survive better than a bare one?

Pipeline
--------
1. :func:`rotated_surface_code` builds the rotated surface code of distance
   ``d`` from coordinates (d=3 reproduces Surface-17 of :mod:`gosc.qiskit_check`
   exactly, d=5 gives 25 data + 24 ancilla) and :func:`verify_code` checks it.
2. :func:`memory_ops` produces ONE abstract gate schedule (reset, H, CX,
   measure).  It is emitted twice: :func:`build_memory_circuit` -> Qiskit
   ``QuantumCircuit`` (the data generator) and :func:`build_stim_circuit` ->
   ``stim.Circuit`` (only used for the exact detector error model / cross-check).
3. Noise (:func:`memory_noise_model`): Aer ``NoiseModel`` with single knob p:
   1q depolarizing p1=p/10 on h/x/y/z/id, 2q depolarizing p2=p on cx, readout
   flip pmeas=p, and an X flip with probability p after every ``reset``.
   Aer's ``depolarizing_error(lam, n)`` is ``(1-lam) rho + lam I/2^n``: every
   non-identity Pauli has probability lam/4^n (verified), so the stim twin
   uses DEPOLARIZE1(3 p1/4) and DEPOLARIZE2(15 p2/16).
4. Detectors (Z memory): Z-check outcomes of round 0 (deterministic 0 for
   |0..0>), XOR of consecutive rounds, and final data-implied Z-check value XOR
   the last round.  X-checks are measured (realism) but not used; their round-0
   outcome is random and is never post-selected.  Observable = parity of the
   final data bits on Z_L (top row; {0,1,2} for d=3).
5. Decoders (PyMatching): (a) a hand-built space-time graph (spacelike +
   boundary + timelike edges, weights from documented effective q's, or
   uniform); (b) the exact detector error model of the stim twin.  Both decode
   the same Aer samples.  (c) "raw" = no decoding (read Z_L parity directly).
6. Baseline: one physical qubit, reset, ``IDLE_PER_ROUND`` noisy ``id`` gates
   per round, measured; same NoiseModel.

Schedule (fixed, non-interleaved): per round, for each Z-check in index order
``reset anc; CX data->anc in N order (TL, BL, TR, BR); measure``, then for each
X-check ``reset anc; H; CX anc->data in Z order (TL, TR, BL, BR); H; measure``.
An X fault on an X-check ancilla after its 2nd CX spreads to the horizontal
pair (BL, BR) - perpendicular to the vertical X_L, so for Z memory the hook
does NOT reduce the distance (verified with stim: circuit distance = d).
Z-check hooks (vertical Z pairs) only matter for X memory, where they are
likewise perpendicular to the horizontal Z_L.  There is no idle noise: Aer
attaches errors to gates only, for both the encoded and the bare qubit.

Bit conventions follow :mod:`gosc.qiskit_check`: Qiskit bitstrings are
little-endian, ``anc[0..nz-1]`` are Z-check ancillas and ``anc[nz..]`` X-check
ancillas; per-round register ``r{t}`` bit i = check i (Z checks first).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from . import qiskit_check as qk

BASIS_GATES = ["h", "x", "y", "z", "cx", "measure", "reset"]
BASELINE_BASIS = ["id", "x", "measure", "reset"]
IDLE_PER_ROUND = 4  # CX gates a bulk data qubit takes part in per round


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceCode:
    """Rotated surface code.  Stabilizer lists are in CNOT order."""

    d: int
    z_stabs: tuple[tuple[int, ...], ...]
    x_stabs: tuple[tuple[int, ...], ...]
    z_logical: tuple[int, ...]
    x_logical: tuple[int, ...]
    coords: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def n_data(self) -> int:
        return self.d * self.d

    @property
    def n_z(self) -> int:
        return len(self.z_stabs)

    @property
    def n_x(self) -> int:
        return len(self.x_stabs)

    @property
    def n_anc(self) -> int:
        return self.n_z + self.n_x

    def matrix(self, kind: str) -> np.ndarray:
        stabs = self.z_stabs if kind == "Z" else self.x_stabs
        m = np.zeros((len(stabs), self.n_data), np.uint8)
        for i, s in enumerate(stabs):
            m[i, list(s)] = 1
        return m


@lru_cache(maxsize=None)
def rotated_surface_code(d: int = 3) -> SurfaceCode:
    """Rotated surface code from coordinates.

    Data qubit (r, c) -> index r*d + c.  Plaquette with top-left corner
    (i, j), i, j in -1..d-1, covers the in-grid qubits of {i,i+1}x{j,j+1}; it is
    an X-check if (i+j) is even, else a Z-check.  All weight-4 plaquettes are
    kept; weight-2 ones only if X on the top/bottom or Z on the left/right
    boundary.  Z_L = top row, X_L = column d//2.  For d=3 the check *sets* and
    their *index order* are asserted to equal ``qiskit_check.Z_STABS/X_STABS``.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError("d must be odd and >= 3")
    zs, xs = [], []
    for i in range(-1, d):
        for j in range(-1, d):
            slots = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]  # TL, BL, TR, BR
            inside = [(r, c) for r, c in slots if 0 <= r < d and 0 <= c < d]
            is_x = (i + j) % 2 == 0
            if len(inside) == 4:
                pass
            elif len(inside) == 2:
                if is_x and not (i in (-1, d - 1)):
                    continue
                if not is_x and not (j in (-1, d - 1)):
                    continue
            else:
                continue
            if is_x:  # Z order: TL, TR, BL, BR
                order = [slots[0], slots[2], slots[1], slots[3]]
                xs.append(tuple(r * d + c for r, c in order if 0 <= r < d and 0 <= c < d))
            else:     # N order: TL, BL, TR, BR
                zs.append(tuple(r * d + c for r, c in slots if 0 <= r < d and 0 <= c < d))
    if d == 3:  # reuse Surface-17 index order
        zs = [next(s for s in zs if set(s) == set(t)) for t in qk.Z_STABS]
        xs = [next(s for s in xs if set(s) == set(t)) for t in qk.X_STABS]
    zl = tuple(range(d))
    xl = tuple(r * d + d // 2 for r in range(d))
    if d == 3:
        assert list(zl) == qk.LOGICAL_Z_SUPPORT and list(xl) == qk.LOGICAL_X_SUPPORT
    code = SurfaceCode(d, tuple(zs), tuple(xs), zl, xl)
    verify_code(code)
    return code


def verify_code(code: SurfaceCode) -> dict:
    """Assert: checks commute, are independent (k = 1), logicals valid."""
    hz, hx = code.matrix("Z"), code.matrix("X")
    n = code.n_data
    assert ((hz @ hx.T) % 2 == 0).all(), "X and Z checks anticommute"
    rz, rx = qk._gf2_rank(hz), qk._gf2_rank(hx)
    assert rz == code.n_z and rx == code.n_x, "dependent checks"
    k = n - rz - rx
    assert k == 1, f"k = {k}"
    zl = np.zeros(n, np.uint8)
    zl[list(code.z_logical)] = 1
    xl = np.zeros(n, np.uint8)
    xl[list(code.x_logical)] = 1
    assert not ((hx @ zl) % 2).any() and not ((hz @ xl) % 2).any()
    assert int(zl @ xl) % 2 == 1
    return {"n": n, "k": k, "n_z": code.n_z, "n_x": code.n_x}


# ---------------------------------------------------------------------------
# Schedule (single source of truth for Qiskit and stim)
# ---------------------------------------------------------------------------

def memory_ops(code: SurfaceCode, rounds: int, basis: str = "Z",
               inject: Mapping[tuple[int, int], str] | None = None,
               meas_flips: Sequence[tuple[int, int]] = ()) -> list[tuple]:
    """Abstract schedule.  Qubits: data 0..n-1, ancilla n + a.

    Ops: ("R", q) reset, ("H", q), ("X"/"Y"/"Z", q), ("CX", c, t),
    ("M", q, key) with key ("r", t, a) or ("final", j), ("B",) barrier.
    ``inject[(t, q)] = 'X'|'Y'|'Z'`` applies that Pauli to data qubit q at the
    start of round t (t == rounds: just before the final readout).
    ``meas_flips`` = [(t, a)] applies X on ancilla a just before it is measured
    in round t (= a measurement error on that check).
    """
    if rounds < 1:
        raise ValueError("rounds >= 1")
    n = code.n_data
    inject = dict(inject or {})
    flips = set(meas_flips)
    ops: list[tuple] = [("R", q) for q in range(n)]
    if basis == "X":
        ops += [("H", q) for q in range(n)]
    elif basis != "Z":
        raise ValueError(basis)
    for t in range(rounds + 1):
        ops.append(("B",))
        for (tt, q), pauli in sorted(inject.items()):
            if tt == t:
                ops.append((pauli.upper(), q))
        if t == rounds:
            break
        for a, stab in enumerate(code.z_stabs):
            anc = n + a
            ops.append(("R", anc))
            ops += [("CX", q, anc) for q in stab]
            if (t, a) in flips:
                ops.append(("X", anc))
            ops.append(("M", anc, ("r", t, a)))
        for i, stab in enumerate(code.x_stabs):
            a = code.n_z + i
            anc = n + a
            ops += [("R", anc), ("H", anc)]
            ops += [("CX", anc, q) for q in stab]
            ops.append(("H", anc))
            if (t, a) in flips:
                ops.append(("X", anc))
            ops.append(("M", anc, ("r", t, a)))
    if basis == "X":
        ops += [("H", q) for q in range(n)]
    ops += [("M", q, ("final", q)) for q in range(n)]
    return ops


def build_memory_circuit(rounds: int, basis: str = "Z", d: int = 3,
                         inject: Mapping[tuple[int, int], str] | None = None,
                         meas_flips: Sequence[tuple[int, int]] = ()) -> QuantumCircuit:
    """Qiskit circuit of the memory experiment.

    Registers: ``data`` (d^2), ``anc`` (d^2-1; Z-check ancillas first),
    classical ``r0 .. r{rounds-1}`` (d^2-1 bits each, bit i = check i) and
    ``final`` (d^2 data bits).  d=3: 17 qubits, 8*rounds+9 clbits.
    """
    code = rotated_surface_code(d)
    data = QuantumRegister(code.n_data, "data")
    anc = QuantumRegister(code.n_anc, "anc")
    cregs = [ClassicalRegister(code.n_anc, f"r{t}") for t in range(rounds)]
    fin = ClassicalRegister(code.n_data, "final")
    qc = QuantumCircuit(data, anc, *cregs, fin, name=f"memory_d{d}_r{rounds}_{basis}")
    qubits = list(data) + list(anc)
    for op in memory_ops(code, rounds, basis, inject, meas_flips):
        kind = op[0]
        if kind == "R":
            qc.reset(qubits[op[1]])
        elif kind == "H":
            qc.h(qubits[op[1]])
        elif kind in ("X", "Y", "Z"):
            getattr(qc, kind.lower())(qubits[op[1]])
        elif kind == "CX":
            qc.cx(qubits[op[1]], qubits[op[2]])
        elif kind == "M":
            key = op[2]
            cb = fin[key[1]] if key[0] == "final" else cregs[key[1]][key[2]]
            qc.measure(qubits[op[1]], cb)
        elif kind == "B":
            qc.barrier()
    return qc


def measurement_index(code: SurfaceCode, rounds: int, key: tuple) -> int:
    """Global clbit index of a measurement key in :func:`build_memory_circuit`."""
    if key[0] == "final":
        return rounds * code.n_anc + key[1]
    return key[1] * code.n_anc + key[2]


def detector_definitions(code: SurfaceCode, rounds: int, basis: str = "Z"):
    """Return (detectors, observable): lists of measurement keys to XOR.

    Detector index = t * n_checks + c, t = 0..rounds (t = rounds uses the
    final data readout), checks of type ``basis`` only.
    """
    stabs = code.z_stabs if basis == "Z" else code.x_stabs
    off = 0 if basis == "Z" else code.n_z
    dets = []
    for t in range(rounds + 1):
        for c, stab in enumerate(stabs):
            keys = []
            if t < rounds:
                keys.append(("r", t, off + c))
            else:
                keys += [("final", q) for q in stab]
            if t > 0:
                keys.append(("r", t - 1, off + c))
            dets.append(keys)
    obs = [("final", q) for q in (code.z_logical if basis == "Z" else code.x_logical)]
    return dets, obs


def _parity_matrices(code, rounds, basis):
    dets, obs = detector_definitions(code, rounds, basis)
    nclb = rounds * code.n_anc + code.n_data
    D = np.zeros((nclb, len(dets)), np.uint8)
    for j, keys in enumerate(dets):
        for k in keys:
            D[measurement_index(code, rounds, k), j] ^= 1
    O = np.zeros((nclb, 1), np.uint8)
    for k in obs:
        O[measurement_index(code, rounds, k), 0] ^= 1
    return D, O


def counts_to_arrays(counts: Mapping[str, int], qc: QuantumCircuit, code: SurfaceCode,
                     rounds: int, basis: str = "Z"):
    """Turn Aer counts into (detectors[u, nd], observable[u], weights[u]) over unique keys.

    Keys are parsed with :func:`gosc.qiskit_check.split_counts_key`
    (little-endian, first register rightmost).
    """
    keys = list(counts)
    bits = np.zeros((len(keys), qc.num_clbits), np.uint8)
    for u, key in enumerate(keys):
        regs = qk.split_counts_key(key, qc)
        row = []
        for reg in qc.cregs:
            row.extend(qk.bits_to_tuple(regs[reg.name]))
        bits[u] = row
    D, O = _parity_matrices(code, rounds, basis)
    dets = (bits @ D) % 2
    obs = ((bits @ O) % 2)[:, 0]
    w = np.array([counts[k] for k in keys], np.int64)
    return dets.astype(np.uint8), obs.astype(np.uint8), w


# ---------------------------------------------------------------------------
# Noise and simulation
# ---------------------------------------------------------------------------

def memory_noise_model(p: float, p1: float | None = None, p_idle: float | None = None) -> NoiseModel:
    """p1 = p/10 (h,x,y,z,id), p2 = p (cx), pmeas = p, p_reset = p.

    Reuses :func:`gosc.qiskit_check.build_noise_model` (gate lists
    ``ONE_Q_GATES`` / ``TWO_Q_GATES``, readout error) and adds X-after-reset.
    ``p_idle`` (baseline only) replaces the depolarizing rate on ``id``.
    """
    p1 = p / 10 if p1 is None else p1
    if p_idle is None:
        nm = qk.build_noise_model(p1, p, p)
    else:
        nm = qk.build_noise_model(0.0, p, p)
        if p1 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                           [g for g in qk.ONE_Q_GATES if g != "id"])
        if p_idle > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(p_idle, 1), ["id"])
    if p > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["reset"])
    return nm


@lru_cache(maxsize=None)
def _transpiled(d: int, rounds: int, basis: str) -> QuantumCircuit:
    pm = generate_preset_pass_manager(optimization_level=0, basis_gates=BASIS_GATES)
    return pm.run(build_memory_circuit(rounds, basis, d))


def sample_memory_aer(d: int, rounds: int, p: float, shots: int, seed: int = 0,
                      basis: str = "Z", qc: QuantumCircuit | None = None):
    """Run the (transpiled) memory circuit on ``AerSimulator(method='stabilizer')``.

    Returns (detectors[u, nd], observable[u], weights[u]) over unique outcomes.
    """
    code = rotated_surface_code(d)
    if qc is None:
        qc = _transpiled(d, rounds, basis)
    else:
        pm = generate_preset_pass_manager(optimization_level=0, basis_gates=BASIS_GATES)
        qc = pm.run(qc)
    sim = AerSimulator(method="stabilizer", noise_model=memory_noise_model(p) if p > 0 else None)
    counts = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    return counts_to_arrays(counts, qc, code, rounds, basis)


# ---------------------------------------------------------------------------
# stim twin (exact DEM + independent sampler)
# ---------------------------------------------------------------------------

def build_stim_circuit(d: int, rounds: int, p: float, basis: str = "Z",
                       inject: Mapping[tuple[int, int], str] | None = None,
                       meas_flips: Sequence[tuple[int, int]] = (), p1: float | None = None):
    """stim circuit with the SAME schedule, detectors and noise as the Aer run."""
    import stim

    code = rotated_surface_code(d)
    p1 = p / 10 if p1 is None else p1
    c = stim.Circuit()
    rec: dict = {}
    nmeas = 0
    for op in memory_ops(code, rounds, basis, inject, meas_flips):
        kind = op[0]
        if kind == "R":
            c.append("R", [op[1]])
            if p > 0:
                c.append("X_ERROR", [op[1]], p)
        elif kind in ("H", "X", "Y", "Z"):
            c.append(kind, [op[1]])
            if p1 > 0:
                c.append("DEPOLARIZE1", [op[1]], 0.75 * p1)
        elif kind == "CX":
            c.append("CX", [op[1], op[2]])
            if p > 0:
                c.append("DEPOLARIZE2", [op[1], op[2]], 15 / 16 * p)
        elif kind == "M":
            c.append("M", [op[1]], p)
            rec[op[2]] = nmeas
            nmeas += 1
        elif kind == "B":
            c.append("TICK")
    dets, obs = detector_definitions(code, rounds, basis)
    for keys in dets:
        c.append("DETECTOR", [stim.target_rec(rec[k] - nmeas) for k in keys])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec[k] - nmeas) for k in obs], 0)
    return c


def stim_matching(d: int, rounds: int, p: float, basis: str = "Z"):
    """PyMatching decoder from the exact stim detector error model."""
    import pymatching

    dem = build_stim_circuit(d, rounds, p, basis).detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def circuit_distance(d: int, rounds: int, basis: str = "Z", p: float = 1e-3) -> int:
    """Size of the smallest graphlike undetected logical error (stim)."""
    c = build_stim_circuit(d, rounds, p, basis)
    return len(c.shortest_graphlike_error())


def sample_memory_stim(d: int, rounds: int, p: float, shots: int, seed: int = 0, basis: str = "Z"):
    """Independent stim sampler (cross-check only).  Returns (dets, obs, weights=1)."""
    c = build_stim_circuit(d, rounds, p, basis)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return dets.astype(np.uint8), obs[:, 0].astype(np.uint8), np.ones(shots, np.int64)


# ---------------------------------------------------------------------------
# Hand-built space-time matching graph
# ---------------------------------------------------------------------------

# Effective per-round flip probabilities used for weights (rough, documented):
#  * data X-flip per round: a bulk data qubit is in 4 CX, each flipping it
#    (X or Y on that qubit) with prob 8/16 * p = p/2  ->  Q_SPACE = 2 p
#  * check-outcome flip: readout p + reset p + ancilla flips from its CXs
#    (~1 p)                                      ->  Q_TIME = 3 p
Q_SPACE_FACTOR = 2.0
Q_TIME_FACTOR = 3.0


def hand_built_matching(d: int, rounds: int, p: float, basis: str = "Z", weighting: str = "p"):
    """Space-time graph: nodes (t, c), t = 0..rounds.

    * spacelike: X error on data q at the start of round t (t = rounds: before
      the final readout) flips every check containing q at layer t; a qubit in
      one check gives a boundary edge.  Fault id 0 if q is in the logical.
    * timelike: a wrong outcome of check c in round t flips (t, c), (t+1, c).
    Weights log((1-q)/q) with q = Q_SPACE_FACTOR*p / Q_TIME_FACTOR*p
    (``weighting='p'``) or all 1 (``'uniform'``).  Diagonal (hook /
    mid-round) edges are NOT included - see :func:`stim_matching`.
    """
    import pymatching

    code = rotated_surface_code(d)
    h = code.matrix(basis)
    logical = set(code.z_logical if basis == "Z" else code.x_logical)
    nc = h.shape[0]

    def w(q):
        if weighting == "uniform":
            return 1.0
        q = min(max(q, 1e-12), 0.49)
        return math.log((1 - q) / q)

    ws, wt = w(Q_SPACE_FACTOR * p), w(Q_TIME_FACTOR * p)
    m = pymatching.Matching()
    for t in range(rounds + 1):
        for q in range(code.n_data):
            checks = np.flatnonzero(h[:, q])
            fid = {0} if q in logical else set()
            if len(checks) == 1:
                m.add_boundary_edge(t * nc + int(checks[0]), fault_ids=fid, weight=ws,
                                    merge_strategy="smallest-weight")
            else:
                m.add_edge(t * nc + int(checks[0]), t * nc + int(checks[1]), fault_ids=fid,
                           weight=ws, merge_strategy="smallest-weight")
        if t < rounds:
            for c in range(nc):
                m.add_edge(t * nc + c, (t + 1) * nc + c, fault_ids=set(), weight=wt,
                           merge_strategy="smallest-weight")
    return m


def decode_failures(matching, dets: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> int:
    """Number of shots where the decoder's observable prediction is wrong."""
    pred = np.asarray(matching.decode_batch(dets), np.uint8)[:, 0]
    return int(weights[pred != obs].sum())


# ---------------------------------------------------------------------------
# Unencoded baseline
# ---------------------------------------------------------------------------

def unencoded_circuit(rounds: int, idle_per_round: int = IDLE_PER_ROUND, basis: str = "Z") -> QuantumCircuit:
    """Bare qubit: reset, [H], rounds*idle_per_round ``id`` gates, [H], measure."""
    qc = QuantumCircuit(1, 1, name=f"bare_r{rounds}")
    qc.reset(0)
    if basis == "X":
        qc.h(0)
    for _ in range(rounds * idle_per_round):
        qc.id(0)
    if basis == "X":
        qc.h(0)
    qc.measure(0, 0)
    return qc


def run_unencoded(rounds: int, p: float, shots: int, seed: int = 0,
                  idle_per_round: int = IDLE_PER_ROUND, p_idle: float | None = None) -> int:
    """Failures (outcome 1) of the bare qubit on Aer under the same NoiseModel.

    ``p_idle=None`` -> idle gates use the 1q rate p/10 (primary baseline);
    ``p_idle=p`` -> 'strict' baseline where each idle step is as noisy as a CX slot.
    """
    pm = generate_preset_pass_manager(optimization_level=0, basis_gates=BASELINE_BASIS)
    qc = pm.run(unencoded_circuit(rounds, idle_per_round))
    nm = memory_noise_model(p, p_idle=p_idle)
    counts = AerSimulator(method="stabilizer", noise_model=nm).run(
        qc, shots=shots, seed_simulator=seed).result().get_counts()
    return int(counts.get("1", 0))


def unencoded_analytic(rounds: int, p: float, idle_per_round: int = IDLE_PER_ROUND,
                       p_idle: float | None = None) -> float:
    """P(odd number of flips): reset p, n idle flips lam/2 each, readout p."""
    lam = p / 10 if p_idle is None else p_idle
    flips = [p, p] + [lam / 2] * (rounds * idle_per_round)
    return float((1 - np.prod([1 - 2 * f for f in flips])) / 2)


# ---------------------------------------------------------------------------
# Statistics and experiment driver
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def run_memory_point(d: int, rounds: int, p: float, shots: int, seed: int = 0,
                     basis: str = "Z", unencoded_shots: int | None = None) -> dict:
    """One (d, rounds, p) point: Aer data decoded 4 ways + bare-qubit baselines."""
    dets, obs, w = sample_memory_aer(d, rounds, p, shots, seed, basis)
    row = {"d": d, "rounds": rounds, "p": p, "basis": basis, "shots": shots}
    row["fail_raw"] = int(w[obs == 1].sum())
    row["fail_stim_dem"] = decode_failures(stim_matching(d, rounds, p, basis), dets, obs, w)
    row["fail_hand_p"] = decode_failures(hand_built_matching(d, rounds, p, basis, "p"), dets, obs, w)
    row["fail_hand_uniform"] = decode_failures(hand_built_matching(d, rounds, p, basis, "uniform"),
                                               dets, obs, w)
    us = unencoded_shots or shots
    row["unenc_shots"] = us
    row["fail_unenc"] = run_unencoded(rounds, p, us, seed + 1)
    row["fail_unenc_strict"] = run_unencoded(rounds, p, us, seed + 2, p_idle=p)
    row["unenc_analytic"] = unencoded_analytic(rounds, p)
    row["unenc_strict_analytic"] = unencoded_analytic(rounds, p, p_idle=p)
    for k in ("raw", "stim_dem", "hand_p", "hand_uniform", "unenc", "unenc_strict"):
        n = us if k.startswith("unenc") else shots
        f = row[f"fail_{k}"]
        lo, hi = wilson(f, n)
        row[f"PL_{k}"], row[f"lo_{k}"], row[f"hi_{k}"] = f / n, lo, hi
    return row


def break_even(ps: Sequence[float], enc: Sequence[float], unenc: Sequence[float]) -> float | None:
    """Log-log interpolated p where enc/unenc crosses 1 (first crossing), else None."""
    r = [math.log(max(e, 1e-12) / max(u, 1e-12)) for e, u in zip(enc, unenc)]
    for i in range(len(ps) - 1):
        if (r[i] <= 0 < r[i + 1]) or (r[i] > 0 >= r[i + 1]):
            x0, x1 = math.log(ps[i]), math.log(ps[i + 1])
            return float(math.exp(x0 + (x1 - x0) * (-r[i]) / (r[i + 1] - r[i])))
    return None
