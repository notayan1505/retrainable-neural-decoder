"""Qiskit cross-check of the distance-3 rotated surface code ("Surface-17").

This module independently re-derives, with Qiskit 2.x / qiskit-aer, the
surface-code facts used by the workshop notebook and the tile game:

1. the Surface-17 stabilizers and logical operators (checked symplectically);
2. a *proper* logical |0>_L (projected onto the +1 eigenspace of all 8 checks);
3. single-qubit X/Y/Z syndromes (Aer stabilizer simulation vs. numpy);
4. a lookup-table decoder and a PyMatching MWPM decoder;
5. code-capacity logical error rates (Monte Carlo + exact enumeration);
6. one noisy round of the Qiskit syndrome circuit on AerSimulator;
7. Bell-pair creation by ZZ measurement and generic Pauli-product measurement;
8. the notebook's 2-qubit ZZ stabilizer circuit for an optional hardware run.

Conventions
-----------
* Data qubits are 0..8 laid out row-major on a 3x3 grid::

      0 1 2
      3 4 5
      6 7 8

* A Pauli on the 9 data qubits is a pair of uint8 vectors ``(x, z)`` indexed by
  qubit (``x[q] = 1`` means an X component on qubit ``q``; Y = X and Z).
* Qiskit ``Pauli`` *labels* are big-endian strings whose LAST character acts on
  qubit 0.  To avoid mistakes this module never hand-writes labels for the code;
  it builds ``Pauli((z_bool, x_bool))`` from arrays indexed by qubit.
* Qiskit classical bitstrings (``get_counts`` keys) are little-endian: within a
  register, bit 0 is the RIGHTMOST character.  With several registers, the key
  is space-separated and the register added FIRST is the RIGHTMOST group.
  :func:`split_counts_key` parses keys accordingly.
* Syndromes are returned as tuples indexed by check number (``s[i]`` = outcome
  of check ``i``, 1 meaning eigenvalue -1).  When packed into an integer we use
  ``sum(s[i] << i)`` (little-endian, matching Qiskit).
* ``sz`` = outcomes of the 4 Z-checks (these detect X errors);
  ``sx`` = outcomes of the 4 X-checks (these detect Z errors).

No deprecated APIs are used (no ``execute``, ``IBMQ``, ``BasicAer``, V1
primitives, or ``c_if``); classical feed-forward uses ``QuantumCircuit.if_test``.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, StabilizerState
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# ---------------------------------------------------------------------------
# 1. Surface-17 definition
# ---------------------------------------------------------------------------

N_DATA = 9
N_ANC = 8
Z_STABS: list[list[int]] = [[0, 3], [1, 2, 4, 5], [3, 4, 6, 7], [5, 8]]
X_STABS: list[list[int]] = [[0, 1, 3, 4], [1, 2], [4, 5, 7, 8], [6, 7]]
LOGICAL_X_SUPPORT: list[int] = [1, 4, 7]  # X_L = X1 X4 X7 (vertical column)
LOGICAL_Z_SUPPORT: list[int] = [0, 1, 2]  # Z_L = Z0 Z1 Z2 (top row)

# For each X-check, one data qubit that belongs to that X-check only.  A Z on
# that qubit flips exactly that X-check (used for |0>_L preparation).
X_CHECK_PIVOTS: list[int] = [0, 2, 8, 6]


def _support_matrix(stabs: Sequence[Sequence[int]]) -> np.ndarray:
    m = np.zeros((len(stabs), N_DATA), dtype=np.uint8)
    for i, s in enumerate(stabs):
        m[i, list(s)] = 1
    return m


HZ = _support_matrix(Z_STABS)  # 4x9 Z-check matrix (detects X errors)
HX = _support_matrix(X_STABS)  # 4x9 X-check matrix (detects Z errors)
XL = _support_matrix([LOGICAL_X_SUPPORT])[0]
ZL = _support_matrix([LOGICAL_Z_SUPPORT])[0]


def symplectic_commutes(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """Return True iff Paulis ``a=(xa,za)`` and ``b=(xb,zb)`` commute.

    Uses the symplectic form ``xa.zb + za.xb (mod 2)``.
    """
    xa, za = a
    xb, zb = b
    return int(np.dot(xa, zb) + np.dot(za, xb)) % 2 == 0


def stabilizer_paulis() -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    """All 8 stabilizers as ``(name, (x, z))``; names 'Z0'..'Z3', 'X0'..'X3'."""
    zero = np.zeros(N_DATA, dtype=np.uint8)
    out = [(f"Z{i}", (zero.copy(), HZ[i].copy())) for i in range(4)]
    out += [(f"X{i}", (HX[i].copy(), zero.copy())) for i in range(4)]
    return out


def logical_paulis() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Logical operators ``{'X_L': (x, z), 'Z_L': (x, z)}``."""
    zero = np.zeros(N_DATA, dtype=np.uint8)
    return {"X_L": (XL.copy(), zero.copy()), "Z_L": (zero.copy(), ZL.copy())}


def verify_code() -> dict[str, bool]:
    """Verify the code definition symplectically; raise AssertionError if wrong.

    Checks: (a) all 8 stabilizers mutually commute, (b) X_L and Z_L commute
    with every stabilizer, (c) X_L and Z_L anticommute, (d) the 8 checks are
    linearly independent (so k = 9 - 8 = 1 logical qubit), (e) each pivot
    qubit in :data:`X_CHECK_PIVOTS` lies in exactly its own X-check.
    The workshop representatives X_L = X{1,4,7}, Z_L = Z{0,1,2} pass.
    """
    stabs = stabilizer_paulis()
    for (_, a), (_, b) in itertools.combinations(stabs, 2):
        assert symplectic_commutes(a, b), "stabilizers do not commute"
    logs = logical_paulis()
    for name, lp in logs.items():
        for sname, s in stabs:
            assert symplectic_commutes(lp, s), f"{name} anticommutes with {sname}"
    assert not symplectic_commutes(logs["X_L"], logs["Z_L"]), "X_L, Z_L commute"
    assert _gf2_rank(HX) == 4 and _gf2_rank(HZ) == 4
    for i, q in enumerate(X_CHECK_PIVOTS):
        col = HX[:, q]
        assert col[i] == 1 and col.sum() == 1
    return {"stabilizers_commute": True, "logicals_commute_with_stabilizers": True,
            "logicals_anticommute": True, "independent_checks": True}


def _gf2_rank(m: np.ndarray) -> int:
    a = m.copy() % 2
    r = 0
    for c in range(a.shape[1]):
        piv = [i for i in range(r, a.shape[0]) if a[i, c]]
        if not piv:
            continue
        a[[r, piv[0]]] = a[[piv[0], r]]
        for i in range(a.shape[0]):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
    return r


def to_qiskit_pauli(x: np.ndarray, z: np.ndarray, num_qubits: int = N_DATA) -> Pauli:
    """Build a ``qiskit.quantum_info.Pauli`` from qubit-indexed (x, z) vectors.

    Vectors are zero-padded to ``num_qubits`` (e.g. 17 when ancillas exist).
    """
    xx = np.zeros(num_qubits, dtype=bool)
    zz = np.zeros(num_qubits, dtype=bool)
    xx[: len(x)] = np.asarray(x, dtype=bool)
    zz[: len(z)] = np.asarray(z, dtype=bool)
    return Pauli((zz, xx))


def pauli_from_dict(err: Mapping[int, str] | None) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``{qubit: 'X'|'Y'|'Z'|'I'}`` into ``(x, z)`` uint8 vectors."""
    x = np.zeros(N_DATA, dtype=np.uint8)
    z = np.zeros(N_DATA, dtype=np.uint8)
    for q, p in (err or {}).items():
        p = p.upper()
        if p in ("X", "Y"):
            x[q] ^= 1
        if p in ("Z", "Y"):
            z[q] ^= 1
    return x, z


def pauli_to_list(x: np.ndarray, z: np.ndarray) -> list[str]:
    """``(x, z)`` -> list of 9 chars in 'IXYZ', index = qubit."""
    return ["IXZY"[int(a) + 2 * int(b)] for a, b in zip(x, z)]


# ---------------------------------------------------------------------------
# 2. Syndrome circuit and proper |0>_L
# ---------------------------------------------------------------------------

def split_counts_key(key: str, circuit: QuantumCircuit) -> dict[str, str]:
    """Split a ``get_counts`` key into per-register bitstrings.

    Qiskit prints registers separated by spaces, last-added register first
    (leftmost).  Within each returned string the rightmost char is bit 0.
    """
    parts = key.split()
    regs = circuit.cregs
    assert len(parts) == len(regs)
    return {reg.name: bits for reg, bits in zip(regs, reversed(parts))}


def bits_to_tuple(bits: str) -> tuple[int, ...]:
    """Little-endian Qiskit bitstring -> tuple indexed by bit number."""
    return tuple(int(c) for c in reversed(bits))


def _measure_z_check(qc, data, anc, a, stab, cbit):
    qc.reset(anc[a])
    for d in stab:
        qc.cx(data[d], anc[a])
    qc.measure(anc[a], cbit)


def _measure_x_check(qc, data, anc, a, stab, cbit):
    qc.reset(anc[a])
    qc.h(anc[a])
    for d in stab:
        qc.cx(anc[a], data[d])
    qc.h(anc[a])
    qc.measure(anc[a], cbit)


def prepare_logical_zero(qc: QuantumCircuit, data: QuantumRegister, anc: QuantumRegister,
                         prep: ClassicalRegister) -> None:
    """Append a measurement-based preparation of the true |0>_L.

    |0...0> is a +1 eigenstate of every Z-check and of Z_L.  Measuring the 4
    X-checks projects onto a random X-syndrome; for each X-check that returns
    -1 we apply Z on its pivot qubit (:data:`X_CHECK_PIVOTS`), which flips that
    X-check only and commutes with all Z-checks and Z_L.  The result is the
    unique state with all 8 checks = +1 and Z_L = +1, i.e. |0>_L.  Uses the
    X-check ancillas ``anc[4..7]`` and the 4-bit register ``prep``.
    """
    for i, stab in enumerate(X_STABS):
        _measure_x_check(qc, data, anc, 4 + i, stab, prep[i])
    for i, q in enumerate(X_CHECK_PIVOTS):
        with qc.if_test((prep[i], 1)):
            qc.z(data[q])


def logical_zero_encoder() -> QuantumCircuit:
    """Unitary (measurement-free) Clifford encoder for |0>_L on 9 qubits.

    H on each X-check pivot, then CX from the pivot to the other qubits of its
    X-check.  Pivots are never targets and all these CXs commute, so the
    output is ``prod_i (I + X_check_i)|0..0>`` normalised = |0>_L.  Useful for
    direct verification with ``StabilizerState``.
    """
    qc = QuantumCircuit(N_DATA, name="enc0L")
    for stab, piv in zip(X_STABS, X_CHECK_PIVOTS):
        qc.h(piv)
        for d in stab:
            if d != piv:
                qc.cx(piv, d)
    return qc


def syndrome_circuit(error: Mapping[int, str] | None = None, logical: str | None = "0",
                     correction: Mapping[int, str] | None = None,
                     save_state: bool = False) -> QuantumCircuit:
    """Surface-17 circuit: prepare, inject error, measure all 8 stabilizers.

    Registers: ``data`` (9), ``anc`` (8: anc[0..3] Z-checks, anc[4..7]
    X-checks), classical ``s`` (8 bits: s[0..3] = Z-checks, s[4..7] = X-checks)
    and, if ``logical`` is not None, classical ``prep`` (4 bits, X-check
    projection outcomes during preparation).  In ``get_counts`` keys the
    register order is ``"prep s"`` (``s`` was added first so it is rightmost).

    Z-check: reset anc; CX data->anc; measure.  X-check: reset anc; H; CX
    anc->data; H; measure.  Outcome 1 means eigenvalue -1.

    Args:
        error: ``{qubit: 'X'|'Y'|'Z'}`` injected after preparation.
        logical: '0' for |0>_L, '1' for |1>_L (= X_L|0>_L), None for |0..0>.
        correction: optional Pauli applied after the syndrome round.
        save_state: append ``save_stabilizer`` (Aer stabilizer method only).
    """
    data = QuantumRegister(N_DATA, "data")
    anc = QuantumRegister(N_ANC, "anc")
    s = ClassicalRegister(8, "s")
    qc = QuantumCircuit(data, anc, s)
    if logical is not None:
        prep = ClassicalRegister(4, "prep")
        qc.add_register(prep)
        prepare_logical_zero(qc, data, anc, prep)
        if logical == "1":
            for q in LOGICAL_X_SUPPORT:
                qc.x(data[q])
        qc.barrier()
    _apply_pauli_gates(qc, data, error)
    qc.barrier()
    for i, stab in enumerate(Z_STABS):
        _measure_z_check(qc, data, anc, i, stab, s[i])
    for i, stab in enumerate(X_STABS):
        _measure_x_check(qc, data, anc, 4 + i, stab, s[4 + i])
    if correction:
        qc.barrier()
        _apply_pauli_gates(qc, data, correction)
    if save_state:
        qc.save_stabilizer()
    return qc


def _apply_pauli_gates(qc, data, pauli: Mapping[int, str] | None):
    for q, p in (pauli or {}).items():
        p = p.upper()
        if p == "X":
            qc.x(data[q])
        elif p == "Y":
            qc.y(data[q])
        elif p == "Z":
            qc.z(data[q])


def verify_logical_zero(seeds: Iterable[int] = range(20)) -> list[dict[str, float]]:
    """Run the measurement-based |0>_L prep on Aer (stabilizer method).

    For each seed returns expectation values of all 8 stabilizers and Z_L on
    the final 17-qubit state plus the random ``prep`` outcomes.
    """
    sim = AerSimulator(method="stabilizer")
    qc = syndrome_circuit(logical="0", save_state=True)
    rows = []
    for seed in seeds:
        res = sim.run(qc, shots=1, seed_simulator=seed).result()
        st: StabilizerState = res.data(0)["stabilizer"]
        row = {name: float(np.real(st.expectation_value(to_qiskit_pauli(x, z, 17))))
               for name, (x, z) in stabilizer_paulis()}
        x, z = logical_paulis()["Z_L"]
        row["Z_L"] = float(np.real(st.expectation_value(to_qiskit_pauli(x, z, 17))))
        key = next(iter(res.get_counts()))
        row["prep"] = split_counts_key(key, qc)["prep"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 3. Syndromes
# ---------------------------------------------------------------------------

def syndrome_of(x: np.ndarray, z: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Symplectic syndrome ``(sz, sx)`` of a data Pauli (x, z).

    ``sz = HZ x mod 2`` (Z-checks see X parts); ``sx = HX z mod 2``.
    """
    sz = tuple(int(v) for v in (HZ @ x) % 2)
    sx = tuple(int(v) for v in (HX @ z) % 2)
    return sz, sx


def aer_syndromes(errors: Sequence[Mapping[int, str]], logical: str = "0",
                  seed: int = 1234) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Run the Qiskit syndrome circuit for each error on Aer (stabilizer method).

    Returns ``[(sz, sx), ...]`` parsed from the ``s`` register (little-endian:
    s[0..3] = Z-checks = rightmost 4 chars, s[4..7] = X-checks).  Each circuit
    is run with 4 shots and the syndrome is asserted deterministic.
    """
    sim = AerSimulator(method="stabilizer")
    circs = [syndrome_circuit(error=e, logical=logical) for e in errors]
    res = sim.run(circs, shots=4, seed_simulator=seed).result()
    out = []
    for i, qc in enumerate(circs):
        syns = {split_counts_key(k, qc)["s"] for k in res.get_counts(i)}
        assert len(syns) == 1, f"non-deterministic syndrome {syns}"
        s = bits_to_tuple(syns.pop())
        out.append((s[:4], s[4:]))
    return out


# ---------------------------------------------------------------------------
# 4. Decoders
# ---------------------------------------------------------------------------

def _syn_int(s: Sequence[int]) -> int:
    return int(sum(int(b) << i for i, b in enumerate(s)))


def _min_weight_table(h: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]], set[int]]:
    """Min-weight correction for every one of the 16 syndromes of ``h``.

    Returns (table[16, 9], weight1_candidates, extended_syndromes).  Ties are
    broken by enumerating by weight then lexicographically by qubit tuple.
    """
    table = np.zeros((16, N_DATA), dtype=np.uint8)
    found = {0}
    cands: dict[int, list[int]] = {}
    for q in range(N_DATA):
        cands.setdefault(_syn_int((h[:, q]) % 2), []).append(q)
    extended = set()
    for w in range(1, N_DATA + 1):
        for qs in itertools.combinations(range(N_DATA), w):
            e = np.zeros(N_DATA, dtype=np.uint8)
            e[list(qs)] = 1
            k = _syn_int((h @ e) % 2)
            if k not in found:
                found.add(k)
                table[k] = e
                if w > 1:
                    extended.add(k)
        if len(found) == 16:
            break
    return table, cands, extended


def build_lookup_tables() -> dict:
    """Lookup tables for single X errors (via Z-checks) and single Z errors.

    Returns a dict with:
      * ``'X'``: {sz tuple: [qubits whose X error gives sz]} (weight 1 only),
      * ``'Z'``: {sx tuple: [qubits whose Z error gives sx]},
      * ``'X_table'`` / ``'Z_table'``: 16x9 arrays, row = little-endian
        syndrome integer, giving the correction (X resp. Z support).  Weight-1
        entries use the lowest-index candidate; the 8 syndromes per type not
        reachable by a single error are filled by brute-force minimum weight
        (listed in ``'X_extended'`` / ``'Z_extended'``) so the decoder is total.

    Degeneracy: X1 and X2 share sz=(0,1,0,0) and X6, X7 share sz=(0,0,1,0);
    Z0, Z3 share sx=(1,0,0,0) and Z5, Z8 share sx=(0,0,1,0).  Each pair's
    product (X1X2, X6X7, Z0Z3, Z5Z8) is itself a stabilizer, so picking either
    member is logically equivalent: the degeneracy is benign.
    """
    xt, xc, xext = _min_weight_table(HZ)
    zt, zc, zext = _min_weight_table(HX)

    def as_tuple(k):
        return tuple((k >> i) & 1 for i in range(4))

    # weight-1 entries: choose lowest-index qubit
    for k, qs in xc.items():
        xt[k] = 0
        xt[k, qs[0]] = 1
    for k, qs in zc.items():
        zt[k] = 0
        zt[k, qs[0]] = 1
    return {
        "X": {as_tuple(k): v for k, v in sorted(xc.items())},
        "Z": {as_tuple(k): v for k, v in sorted(zc.items())},
        "X_table": xt, "Z_table": zt,
        "X_extended": sorted(as_tuple(k) for k in xext),
        "Z_extended": sorted(as_tuple(k) for k in zext),
    }


_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = build_lookup_tables()
    return _TABLES


def decode(sx: Sequence[int], sz: Sequence[int]) -> list[str]:
    """Lookup decoder.

    Args:
        sx: X-check outcomes (4 bits; detect Z errors) -> Z corrections.
        sz: Z-check outcomes (4 bits; detect X errors) -> X corrections.
    Returns:
        list of 9 chars in 'IXYZ' (index = data qubit) whose syndrome equals
        the input, for every one of the 256 possible inputs.
    """
    t = _tables()
    cx = t["X_table"][_syn_int(sz)]
    cz = t["Z_table"][_syn_int(sx)]
    return pauli_to_list(cx, cz)


def apply_correction(error: tuple[np.ndarray, np.ndarray] | Mapping[int, str],
                     correction: Sequence[str] | Mapping[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Residual Pauli ``correction * error`` (phases dropped) as ``(x, z)``.

    ``error`` may be ``(x, z)`` arrays or a ``{qubit: pauli}`` dict;
    ``correction`` may be a 9-char list (as returned by :func:`decode`) or dict.
    """
    ex, ez = error if isinstance(error, tuple) else pauli_from_dict(error)
    if not isinstance(correction, Mapping):
        correction = {q: p for q, p in enumerate(correction) if p != "I"}
    cx, cz = pauli_from_dict(correction)
    return (np.asarray(ex, np.uint8) ^ cx, np.asarray(ez, np.uint8) ^ cz)


def logical_flips(x: np.ndarray, z: np.ndarray) -> tuple[int, int]:
    """For a residual in the normaliser: (X_L flipped?, Z_L flipped?).

    Returns ``(x.ZL mod 2, z.XL mod 2)``: the first is 1 when the residual
    anticommutes with Z_L (a logical X was applied), the second when it
    anticommutes with X_L (a logical Z was applied).
    """
    return int(np.dot(x, ZL) % 2), int(np.dot(z, XL) % 2)


def _mwpm():
    import pymatching

    return pymatching.Matching(HZ), pymatching.Matching(HX)


def mwpm_decode(sx: Sequence[int], sz: Sequence[int]) -> list[str]:
    """PyMatching MWPM decoder (``Matching(HZ)`` for X errors, ``Matching(HX)``
    for Z errors).  Same signature/return as :func:`decode`."""
    mz, mx = _mwpm()
    cx = np.asarray(mz.decode(np.asarray(sz, np.uint8)), np.uint8)
    cz = np.asarray(mx.decode(np.asarray(sx, np.uint8)), np.uint8)
    return pauli_to_list(cx, cz)


def _batch_decoder(name: str) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return f(SZ, SX) -> (CX, CZ) for (N,4) syndrome arrays."""
    if name == "lookup":
        t = _tables()
        w = 1 << np.arange(4)

        def f(SZ, SX):
            return t["X_table"][SZ @ w], t["Z_table"][SX @ w]
        return f
    if name == "mwpm":
        mz, mx = _mwpm()

        def g(SZ, SX):
            return (np.asarray(mz.decode_batch(SZ.astype(np.uint8)), np.uint8),
                    np.asarray(mx.decode_batch(SX.astype(np.uint8)), np.uint8))
        return g
    raise ValueError(name)


def compare_decoders(max_weight: int = 2) -> dict:
    """Compare lookup vs MWPM on every Pauli error of weight 1..max_weight.

    Returns {'counts': {w: n_errors}, 'lookup_fail': {w: n}, 'mwpm_fail':
    {w: n}, 'differ': [(error_dict, lookup_flips, mwpm_flips), ...]}.  A
    failure means the residual is a nontrivial logical.  Residual syndromes
    are asserted zero for both decoders.
    """
    out = {"counts": {}, "lookup_fail": {}, "mwpm_fail": {}, "differ": []}
    for w in range(1, max_weight + 1):
        n = lf = mf = 0
        for qs in itertools.combinations(range(N_DATA), w):
            for ps in itertools.product("XYZ", repeat=w):
                err = dict(zip(qs, ps))
                x, z = pauli_from_dict(err)
                sz, sx = syndrome_of(x, z)
                res = {}
                for name, dec in (("lookup", decode), ("mwpm", mwpm_decode)):
                    rx, rz = apply_correction((x, z), dec(sx, sz))
                    assert syndrome_of(rx, rz) == ((0,) * 4, (0,) * 4)
                    res[name] = logical_flips(rx, rz)
                n += 1
                lf += any(res["lookup"])
                mf += any(res["mwpm"])
                if res["lookup"] != res["mwpm"]:
                    out["differ"].append((err, res["lookup"], res["mwpm"]))
        out["counts"][w], out["lookup_fail"][w], out["mwpm_fail"][w] = n, lf, mf
    return out


# ---------------------------------------------------------------------------
# 5. Code-capacity logical error rate
# ---------------------------------------------------------------------------

def logical_error_rate(p: float, shots: int, decoder: str = "lookup", seed: int = 0) -> dict:
    """Monte Carlo logical error rate under code-capacity depolarizing noise.

    Each data qubit independently suffers X, Y or Z with probability p/3 each
    (perfect syndrome extraction, one round).  The syndrome is computed
    symplectically with numpy, decoded (``decoder`` in {'lookup', 'mwpm'}),
    and a failure is declared if the residual anticommutes with X_L or Z_L.
    The residual syndrome is asserted to be zero.  Seeded for reproducibility.

    Returns {'p', 'shots', 'failures', 'rate', 'stderr'}.
    """
    rng = np.random.default_rng(seed)
    f = _batch_decoder(decoder)
    fails = 0
    done = 0
    chunk = 1_000_000
    while done < shots:
        n = min(chunk, shots - done)
        r = rng.random((n, N_DATA))
        ex = (r < 2 * p / 3).astype(np.uint8)          # X or Y
        ez = ((r >= p / 3) & (r < p)).astype(np.uint8)  # Y or Z
        SZ = (ex @ HZ.T) % 2
        SX = (ez @ HX.T) % 2
        cx, cz = f(SZ, SX)
        rx, rz = ex ^ cx, ez ^ cz
        assert not ((rx @ HZ.T) % 2).any() and not ((rz @ HX.T) % 2).any()
        fail = ((rx @ ZL) % 2 == 1) | ((rz @ XL) % 2 == 1)
        fails += int(fail.sum())
        done += n
    rate = fails / shots
    return {"p": p, "shots": shots, "failures": fails, "rate": rate,
            "stderr": float(np.sqrt(max(rate * (1 - rate), 1e-300) / shots))}


def exact_logical_error_rate(p: float, decoder: str = "lookup") -> float:
    """Exact code-capacity logical error rate by enumerating all 4^9 Paulis.

    P_L(p) = sum over failing errors E of (p/3)^|E| (1-p)^(9-|E|).
    """
    f = _batch_decoder(decoder)
    allv = np.array(list(itertools.product([0, 1], repeat=N_DATA)), dtype=np.uint8)  # 512x9
    cx, _ = f((allv @ HZ.T) % 2, np.zeros((512, 4), np.uint8))
    _, cz = f(np.zeros((512, 4), np.uint8), (allv @ HX.T) % 2)
    fx = ((allv ^ cx) @ ZL) % 2  # X part failure, per X pattern
    fz = ((allv ^ cz) @ XL) % 2  # Z part failure, per Z pattern
    # weight of combined Pauli = |x OR z|
    wt = (allv[:, None, :] | allv[None, :, :]).sum(-1)  # 512x512
    fail = (fx[:, None] | fz[None, :]).astype(bool)
    probs = (p / 3) ** wt * (1 - p) ** (N_DATA - wt)
    return float(probs[fail].sum())


def loglog_slope(ps: Sequence[float], rates: Sequence[float]) -> float:
    """Least-squares slope of log(rate) vs log(p)."""
    return float(np.polyfit(np.log(ps), np.log(rates), 1)[0])


def logical_rate_sweep(ps: Sequence[float] = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1),
                       shots: int = 200_000, decoder: str = "lookup", seed: int = 0) -> list[dict]:
    """Sweep :func:`logical_error_rate` and attach the exact value."""
    out = []
    for i, p in enumerate(ps):
        r = logical_error_rate(p, shots, decoder, seed + i)
        r["exact"] = exact_logical_error_rate(p, decoder)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 6. Noisy Aer syndrome round
# ---------------------------------------------------------------------------

ONE_Q_GATES = ["h", "x", "y", "z", "id", "s", "sdg", "sx"]
TWO_Q_GATES = ["cx", "cz", "cy"]
NOISY_BASIS = ["h", "x", "y", "z", "cx"]  # all covered by build_noise_model


def build_noise_model(p1: float, p2: float, pmeas: float) -> NoiseModel:
    """Depolarizing (1q: p1, 2q: p2) + symmetric readout flip pmeas."""
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ONE_Q_GATES)
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_Q_GATES)
    if pmeas > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - pmeas, pmeas], [pmeas, 1 - pmeas]]))
    return nm


def noisy_aer_syndrome_run(p1: float, p2: float, pmeas: float, shots: int = 2000,
                           seed: int = 7, logical: str | None = "0") -> dict:
    """One noisy round of the Qiskit syndrome circuit on AerSimulator.

    The circuit (|0>_L preparation + one round of all 8 checks) is transpiled
    with ``generate_preset_pass_manager(optimization_level=0, basis_gates=
    NOISY_BASIS)`` (explicit basis so every gate is one the noise model
    covers; with ``backend=sim`` and an empty noise model Aer's generic target
    rewrites H as rz/sx) and run with a depolarizing + readout ``NoiseModel`` (stabilizer method;
    Pauli noise keeps it Clifford).  Noise also acts on the preparation
    gates, so the prepared state itself is imperfect.

    Returns {'syndromes': {s-bitstring: count}, 'zero_fraction': float,
    'ops': transpiled count_ops}.  Bitstrings are the 8-bit ``s`` register,
    little-endian (rightmost = Z-check 0, leftmost = X-check 3).
    """
    sim = AerSimulator(method="stabilizer", noise_model=build_noise_model(p1, p2, pmeas))
    qc = syndrome_circuit(logical=logical)
    pm = generate_preset_pass_manager(optimization_level=0, basis_gates=NOISY_BASIS)
    tqc = pm.run(qc)
    counts = sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
    syn: dict[str, int] = {}
    for k, v in counts.items():
        s = split_counts_key(k, tqc)["s"]
        syn[s] = syn.get(s, 0) + v
    return {"syndromes": syn, "zero_fraction": syn.get("0" * 8, 0) / shots,
            "ops": dict(tqc.count_ops())}


def noisy_zero_fraction_sweep(ps: Sequence[float] = (0.0, 1e-3, 3e-3, 1e-2, 3e-2),
                              shots: int = 2000, seed: int = 7) -> list[tuple[float, float]]:
    """All-zero-syndrome fraction vs p with p1 = p/10, p2 = p, pmeas = p."""
    return [(p, noisy_aer_syndrome_run(p / 10, p, p, shots, seed)["zero_fraction"]) for p in ps]


# ---------------------------------------------------------------------------
# 7. Bell pair by ZZ measurement and generic Pauli-product measurement
# ---------------------------------------------------------------------------

def bell_pair_by_zz_measurement() -> QuantumCircuit:
    """Tile-game Bell protocol: q0, q1 in |+>, measure Z0 Z1 via ancilla q2.

    Unitary part: H q0, H q1, CX q0->q2, CX q1->q2.  Measuring q2 (classical
    bit 0) gives m=0 (ZZ=+1, post-state (|00>+|11>)/sqrt2) or m=1 (ZZ=-1,
    post-state (|01>+|10>)/sqrt2), each with probability 1/2.  XX=+1 is
    preserved because |++> is an XX eigenstate and XX commutes with ZZ.
    """
    qc = QuantumCircuit(3, 1, name="bell_zz")
    qc.h(0)
    qc.h(1)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.measure(2, 0)
    return qc


def bell_branches(max_tries: int = 64) -> dict[int, StabilizerState]:
    """Return {outcome: post-measurement 3-qubit StabilizerState} for both
    outcomes, by sampling ``StabilizerState.measure`` with different seeds."""
    unitary = bell_pair_by_zz_measurement().remove_final_measurements(inplace=False)
    out: dict[int, StabilizerState] = {}
    for seed in range(max_tries):
        st = StabilizerState(unitary)
        st.seed(seed)
        outcome, post = st.measure([2])
        out.setdefault(int(outcome), post)
        if len(out) == 2:
            break
    return out


def bell_target_state(outcome: int) -> StabilizerState:
    """(|00>+|11>)/sqrt2 (outcome 0) or (|01>+|10>)/sqrt2 (outcome 1) on q0,q1,
    tensored with the ancilla q2 in |outcome>."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    if outcome:
        qc.x(1)
        qc.x(2)
    return StabilizerState(qc)


def pauli_product_measurement_circuit(pauli: str) -> QuantumCircuit:
    """Measure a multi-qubit Pauli product with one ancilla.

    ``pauli`` uses the Qiskit label convention: optional leading sign ('+',
    '-'), then characters over 'IXYZ' where the LAST character acts on qubit
    0 (e.g. 'YXZX' = X on q0, Z on q1, X on q2, Y on q3).  The circuit has
    n data qubits plus ancilla qubit n and one classical bit.

    Gadget: H(anc); controlled-P_q (CX/CY/CZ, anc control) for each non-I
    factor; H(anc); [X(anc) if sign is '-']; measure.  Outcome 0 <=> +1
    eigenvalue of the signed operator.
    """
    sign = 1
    lab = pauli
    if lab and lab[0] in "+-":
        sign = -1 if lab[0] == "-" else 1
        lab = lab[1:]
    n = len(lab)
    qc = QuantumCircuit(n + 1, 1, name=f"ppm_{pauli}")
    anc = n
    qc.h(anc)
    for q in range(n):
        c = lab[n - 1 - q]
        if c == "X":
            qc.cx(anc, q)
        elif c == "Y":
            qc.cy(anc, q)
        elif c == "Z":
            qc.cz(anc, q)
        elif c != "I":
            raise ValueError(f"bad Pauli char {c!r}")
    qc.h(anc)
    if sign < 0:
        qc.x(anc)
    qc.measure(anc, 0)
    return qc


def measure_pauli_on_state(state_prep: QuantumCircuit, pauli: str, shots: int = 64,
                           seed: int = 11) -> dict[str, int]:
    """Prepare ``state_prep`` on the data qubits, then run
    :func:`pauli_product_measurement_circuit` on Aer; return counts."""
    ppm = pauli_product_measurement_circuit(pauli)
    qc = QuantumCircuit(ppm.num_qubits, 1)
    qc.compose(state_prep, qubits=range(state_prep.num_qubits), inplace=True)
    qc.compose(ppm, inplace=True)
    sim = AerSimulator(method="stabilizer")
    pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
    return sim.run(pm.run(qc), shots=shots, seed_simulator=seed).result().get_counts()


# ---------------------------------------------------------------------------
# 8. Two-qubit ZZ stabilizer circuit (hardware-ready)
# ---------------------------------------------------------------------------

def two_qubit_stabilizer_circuit(error_qubits: Iterable[int] = (0,)) -> QuantumCircuit:
    """Notebook-style ZZ stabilizer check: data q0, q1 (in |00>), ancilla q2.

    Injects X on each qubit in ``error_qubits``, then CX q0->anc, CX q1->anc
    (data controls; the notebook comment "CNOT from ancilla" is wrong) and
    measures the ancilla into c[0].  Ideal outcome = parity of the number of
    flipped data qubits (0 for no/two errors, 1 for one error).

    Optional hardware run (not executed in tests; needs an IBM account)::

        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService()          # saved credentials
        backend = service.least_busy(operational=True, simulator=False)
        isa = generate_preset_pass_manager(optimization_level=1,
                                           backend=backend).run(qc)
        job = SamplerV2(mode=backend).run([isa], shots=1024)
        counts = job.result()[0].data.c.get_counts()   # register name 'c'

    (The notebook already uses V2-style ``Sampler(mode=backend)`` / PUB
    results, but imports the obsolete ``Options`` and hard-codes
    ``ibm_brisbane``; ``least_busy`` avoids the hard-coded backend.)
    """
    data = QuantumRegister(2, "data")
    anc = QuantumRegister(1, "anc")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(data, anc, c)
    for q in error_qubits:
        qc.x(data[q])
    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])
    qc.measure(anc[0], c[0])
    return qc


# ---------------------------------------------------------------------------
# Notebook discrepancy helper
# ---------------------------------------------------------------------------

def notebook_logical_zero_check(seeds: Iterable[int] = range(8)) -> dict:
    """Analyse the notebook's ``prepare_logical_zero`` (X on {0,3,6}).

    Returns symplectic facts and Aer measurement stats:
      * 'x036_commutes_with_Zchecks': X{0,3,6} commutes with all Z-checks;
      * 'x036_anticommutes_ZL': it anticommutes with Z_L (so it is a logical X);
      * 'x036_times_XL_is_stabilizer': X{0,3,6}X{1,4,7} = X-check0 * X-check3;
      * 'x_syndrome_outcomes': set of X-check syndromes seen on Aer for the
        notebook state (random => not an X-check eigenstate).
    """
    x036 = np.zeros(N_DATA, np.uint8)
    x036[[0, 3, 6]] = 1
    zero = np.zeros(N_DATA, np.uint8)
    p = (x036, zero)
    facts = {
        "x036_commutes_with_Zchecks": all(symplectic_commutes(p, (zero, HZ[i])) for i in range(4)),
        "x036_anticommutes_ZL": not symplectic_commutes(p, (zero, ZL)),
        "x036_times_XL_is_stabilizer": bool(np.array_equal(x036 ^ XL, HX[0] ^ HX[3])),
    }
    data = QuantumRegister(N_DATA, "data")
    anc = QuantumRegister(N_ANC, "anc")
    s = ClassicalRegister(8, "s")
    qc = QuantumCircuit(data, anc, s)
    for q in (0, 3, 6):
        qc.x(data[q])
    for i, stab in enumerate(Z_STABS):
        _measure_z_check(qc, data, anc, i, stab, s[i])
    for i, stab in enumerate(X_STABS):
        _measure_x_check(qc, data, anc, 4 + i, stab, s[4 + i])
    sim = AerSimulator(method="stabilizer")
    counts = sim.run(qc, shots=256, seed_simulator=3).result().get_counts()
    xs = {bits_to_tuple(k)[4:] for k in counts}
    zs = {bits_to_tuple(k)[:4] for k in counts}
    facts["x_syndrome_outcomes"] = xs
    facts["z_syndrome_outcomes"] = zs
    return facts


if __name__ == "__main__":  # pragma: no cover - quick report
    print(verify_code())
    t = build_lookup_tables()
    print("X lookup:", t["X"])
    print("Z lookup:", t["Z"])
    cmp_ = compare_decoders(2)
    print("decoder comparison:", {k: v for k, v in cmp_.items() if k != "differ"},
          "n_differ =", len(cmp_["differ"]))
    sweep = logical_rate_sweep(shots=400_000)
    for r in sweep:
        print(f"p={r['p']:.0e} MC={r['rate']:.3e}±{r['stderr']:.1e} exact={r['exact']:.3e}")
    ps = [1e-3, 2e-3, 5e-3, 1e-2]
    print("exact slope:", loglog_slope(ps, [exact_logical_error_rate(p) for p in ps]))
    print("noisy sweep:", noisy_zero_fraction_sweep())
    print("notebook |0>_L:", notebook_logical_zero_check())
