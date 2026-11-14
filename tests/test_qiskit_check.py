"""Tests for gosc.qiskit_check (Qiskit/Aer/PyMatching cross-check of Surface-17)."""
import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, StabilizerState, random_clifford
from qiskit_aer import AerSimulator

from gosc import qiskit_check as qc_mod
from gosc.qiskit_check import (
    HX, HZ, N_DATA, X_STABS, Z_STABS, apply_correction, aer_syndromes,
    bell_branches, bell_pair_by_zz_measurement, bell_target_state, build_lookup_tables,
    compare_decoders, decode, exact_logical_error_rate, logical_error_rate,
    logical_flips, logical_paulis, logical_zero_encoder, loglog_slope,
    measure_pauli_on_state, mwpm_decode, noisy_aer_syndrome_run,
    notebook_logical_zero_check, pauli_from_dict, pauli_product_measurement_circuit,
    split_counts_key, stabilizer_paulis, symplectic_commutes, syndrome_circuit,
    syndrome_of, to_qiskit_pauli, two_qubit_stabilizer_circuit, verify_code,
    verify_logical_zero,
)

SINGLE_ERRORS = [{q: p} for q in range(N_DATA) for p in "XYZ"]
ZERO4 = (0, 0, 0, 0)


# 1. code definition ---------------------------------------------------------

def test_code_definition_verified():
    assert all(verify_code().values())


def test_all_stabilizers_commute_qiskit():
    paulis = [to_qiskit_pauli(x, z) for _, (x, z) in stabilizer_paulis()]
    for a, b in itertools.combinations(paulis, 2):
        assert a.commutes(b)


def test_logicals_qiskit():
    lp = {k: to_qiskit_pauli(*v) for k, v in logical_paulis().items()}
    for _, (x, z) in stabilizer_paulis():
        s = to_qiskit_pauli(x, z)
        assert lp["X_L"].commutes(s) and lp["Z_L"].commutes(s)
    assert lp["X_L"].anticommutes(lp["Z_L"])
    # label convention: last char = qubit 0
    assert lp["Z_L"].to_label() == "IIIIIIZZZ"
    assert lp["X_L"].to_label() == "IXIIXIIXI"


# 2. proper |0>_L ------------------------------------------------------------

def test_measurement_based_logical_zero_all_plus_one():
    rows = verify_logical_zero(range(16))
    for row in rows:
        for k, v in row.items():
            if k != "prep":
                assert v == pytest.approx(1.0), (k, row)
    # the projection really was random (corrections exercised)
    assert len({r["prep"] for r in rows}) > 1


def test_unitary_encoder_logical_zero():
    st = StabilizerState(logical_zero_encoder())
    for _, (x, z) in stabilizer_paulis():
        assert st.expectation_value(to_qiskit_pauli(x, z)) == pytest.approx(1)
    x, z = logical_paulis()["Z_L"]
    assert st.expectation_value(to_qiskit_pauli(x, z)) == pytest.approx(1)
    x, z = logical_paulis()["X_L"]
    assert st.expectation_value(to_qiskit_pauli(x, z)) == pytest.approx(0)


def test_no_error_syndrome_is_zero_for_both_logicals():
    for logical in ("0", "1"):
        assert aer_syndromes([{}], logical=logical) == [(ZERO4, ZERO4)]


def test_counts_key_little_endian_parsing():
    # X on qubit 8 flips only Z-check 3 => s[3]=1 => 4th char from the right
    qc = syndrome_circuit(error={8: "X"})
    counts = AerSimulator(method="stabilizer").run(qc, shots=1, seed_simulator=0).result().get_counts()
    key = next(iter(counts))
    s = split_counts_key(key, qc)["s"]
    assert s == "00001000"
    assert len(split_counts_key(key, qc)["prep"]) == 4


# 3. syndromes ---------------------------------------------------------------

def test_single_error_syndromes_aer_match_symplectic():
    aer = aer_syndromes(SINGLE_ERRORS)
    for err, (sz, sx) in zip(SINGLE_ERRORS, aer):
        x, z = pauli_from_dict(err)
        assert (sz, sx) == syndrome_of(x, z), err
        (q, p), = err.items()
        exp_sz = tuple(int(q in s) for s in Z_STABS) if p in "XY" else ZERO4
        exp_sx = tuple(int(q in s) for s in X_STABS) if p in "ZY" else ZERO4
        assert (sz, sx) == (exp_sz, exp_sx)


def test_documented_syndromes():
    assert syndrome_of(*pauli_from_dict({4: "X"})) == ((0, 1, 1, 0), ZERO4)
    assert syndrome_of(*pauli_from_dict({0: "Z"})) == (ZERO4, (1, 0, 0, 0))
    assert syndrome_of(*pauli_from_dict({4: "Y"})) == ((0, 1, 1, 0), (1, 0, 1, 0))


# 4. decoders ----------------------------------------------------------------

def test_lookup_tables_and_degeneracy():
    t = build_lookup_tables()
    assert t["X"][(0, 1, 0, 0)] == [1, 2]
    assert t["X"][(0, 0, 1, 0)] == [6, 7]
    assert t["Z"][(1, 0, 0, 0)] == [0, 3]
    assert t["Z"][(0, 0, 1, 0)] == [5, 8]
    assert len(t["X"]) == 7 and len(t["Z"]) == 7
    assert len(t["X_extended"]) == 8 and len(t["Z_extended"]) == 8
    # degenerate pairs differ by a stabilizer
    zero = np.zeros(N_DATA, np.uint8)
    for a, b in [(1, 2), (6, 7)]:
        e = zero.copy(); e[[a, b]] = 1
        assert any(np.array_equal(e, r) for r in HX)
    for a, b in [(0, 3), (5, 8)]:
        e = zero.copy(); e[[a, b]] = 1
        assert any(np.array_equal(e, r) for r in HZ)


def test_decoder_total():
    for sx in itertools.product([0, 1], repeat=4):
        for sz in itertools.product([0, 1], repeat=4):
            corr = decode(sx, sz)
            assert syndrome_of(*pauli_from_dict(dict(enumerate(corr)))) == (sz, sx)


@pytest.mark.parametrize("dec", [decode, mwpm_decode])
def test_single_errors_corrected(dec):
    for err in SINGLE_ERRORS:
        x, z = pauli_from_dict(err)
        sz, sx = syndrome_of(x, z)
        rx, rz = apply_correction((x, z), dec(sx, sz))
        assert syndrome_of(rx, rz) == (ZERO4, ZERO4)
        assert logical_flips(rx, rz) == (0, 0), err


def test_correction_in_aer_returns_to_codespace():
    sim = AerSimulator(method="stabilizer")
    aer = aer_syndromes(SINGLE_ERRORS)
    circs = []
    for err, (sz, sx) in zip(SINGLE_ERRORS, aer):
        corr = {q: p for q, p in enumerate(decode(sx, sz)) if p != "I"}
        circs.append(syndrome_circuit(error=err, correction=corr, save_state=True))
    res = sim.run(circs, shots=1, seed_simulator=5).result()
    zl = to_qiskit_pauli(*logical_paulis()["Z_L"], 17)
    for i in range(len(circs)):
        st = res.data(i)["stabilizer"]
        for _, (x, z) in stabilizer_paulis():
            assert st.expectation_value(to_qiskit_pauli(x, z, 17)) == pytest.approx(1)
        assert st.expectation_value(zl) == pytest.approx(1), SINGLE_ERRORS[i]


def test_mwpm_vs_lookup_weight_two():
    r = compare_decoders(2)
    assert r["counts"] == {1: 27, 2: 324}
    assert r["lookup_fail"][1] == 0 and r["mwpm_fail"][1] == 0
    # d=3 cannot correct all weight-2 errors; both decoders fail on the same set
    assert r["lookup_fail"][2] == r["mwpm_fail"][2] == 144
    assert r["differ"] == []


# 5. logical error rate ------------------------------------------------------

def test_logical_rate_reproducible_and_matches_exact():
    a = logical_error_rate(0.05, 100_000, seed=3)
    b = logical_error_rate(0.05, 100_000, seed=3)
    assert a == b
    ex = exact_logical_error_rate(0.05)
    assert abs(a["rate"] - ex) < 5 * a["stderr"]
    m = logical_error_rate(0.05, 100_000, decoder="mwpm", seed=3)
    assert abs(m["rate"] - ex) < 5 * m["stderr"]


def test_exact_slope_quadratic():
    ps = [1e-3, 2e-3, 5e-3, 1e-2]
    s = loglog_slope(ps, [exact_logical_error_rate(p) for p in ps])
    assert 1.9 < s < 2.05
    assert exact_logical_error_rate(1e-3) < 1e-3  # below break-even


def test_monte_carlo_slope_quadratic():
    ps = [1e-3, 3e-3, 1e-2]
    rates = [logical_error_rate(p, 3_000_000, seed=10 + i)["rate"] for i, p in enumerate(ps)]
    assert all(r > 0 for r in rates)
    assert 1.7 <= loglog_slope(ps, rates) <= 2.4


# 6. noisy Aer ---------------------------------------------------------------

def test_noisy_aer_zero_noise_is_clean():
    r = noisy_aer_syndrome_run(0, 0, 0, shots=200)
    assert r["zero_fraction"] == 1.0
    assert set(r["ops"]) <= {"h", "cx", "z", "x", "reset", "measure", "if_else", "barrier"}


def test_noisy_aer_zero_fraction_decreases():
    lo = noisy_aer_syndrome_run(1e-4, 1e-3, 1e-3, shots=1000)["zero_fraction"]
    hi = noisy_aer_syndrome_run(3e-3, 3e-2, 3e-2, shots=1000)["zero_fraction"]
    assert 1.0 >= lo > hi
    assert lo > 0.85


# 7. Bell and PPM ------------------------------------------------------------

def test_bell_both_branches():
    br = bell_branches()
    assert set(br) == {0, 1}
    for m, st in br.items():
        assert st.equiv(bell_target_state(m))
        assert st.expectation_value(Pauli("IXX")) == pytest.approx(1)
        assert st.expectation_value(Pauli("IZZ")) == pytest.approx(1 if m == 0 else -1)


def test_bell_circuit_on_aer_random_outcome():
    counts = AerSimulator().run(bell_pair_by_zz_measurement(), shots=400, seed_simulator=1).result().get_counts()
    assert set(counts) == {"0", "1"}


def _prep_from_clifford(cliff):
    return cliff.to_circuit()


def test_ppm_product_eigenstates():
    from qiskit import QuantumCircuit
    prep = QuantumCircuit(4)
    # 'YXZX': q0 X(+), q1 Z(|0>), q2 X(+), q3 Y(|+i>)
    prep.h(0); prep.h(2); prep.h(3); prep.s(3)
    assert measure_pauli_on_state(prep, "YXZX") == {"0": 64}
    assert measure_pauli_on_state(prep, "-YXZX") == {"1": 64}
    prep.x(1)  # now Z on q1 is -1 => YXZX eigenvalue -1
    assert measure_pauli_on_state(prep, "YXZX") == {"1": 64}


def test_ppm_random_clifford_stabilizers():
    rng = np.random.default_rng(42)
    for trial in range(3):
        cl = random_clifford(4, seed=100 + trial)
        prep = _prep_from_clifford(cl)
        gens = cl.to_labels(mode="S")  # signed stabilizer generators of C|0>
        st = StabilizerState(prep)
        for g in gens:
            assert st.expectation_value(Pauli(g)) == pytest.approx(1)
            sign, lab = ("-", g[1:]) if g[0] == "-" else ("+", g.lstrip("+"))
            assert measure_pauli_on_state(prep, sign + lab, shots=32) == {"0": 32}
            neg = ("+" if sign == "-" else "-") + lab
            assert measure_pauli_on_state(prep, neg, shots=32) == {"1": 32}
        # a product of two generators is also a stabilizer
        prod = Pauli(gens[0]).dot(Pauli(gens[1]))
        lab = prod.to_label()
        assert measure_pauli_on_state(prep, lab, shots=32) == {"0": 32}


def test_ppm_label_convention():
    qc = pauli_product_measurement_circuit("ZX")
    names = [(i.operation.name, [qc.find_bit(q).index for q in i.qubits]) for i in qc.data]
    assert ("cx", [2, 0]) in names and ("cz", [2, 1]) in names


# 8. 2-qubit circuit ---------------------------------------------------------

@pytest.mark.parametrize("errs,expected", [((), "0"), ((0,), "1"), ((1,), "1"), ((0, 1), "0")])
def test_two_qubit_stabilizer(errs, expected):
    qc = two_qubit_stabilizer_circuit(errs)
    counts = AerSimulator().run(qc, shots=50, seed_simulator=0).result().get_counts()
    assert counts == {expected: 50}


# notebook discrepancy -------------------------------------------------------

def test_notebook_logical_zero_is_really_logical_one():
    f = notebook_logical_zero_check()
    assert f["x036_commutes_with_Zchecks"] and f["x036_anticommutes_ZL"]
    assert f["x036_times_XL_is_stabilizer"]
    assert f["z_syndrome_outcomes"] == {ZERO4}
    assert len(f["x_syndrome_outcomes"]) > 1  # not an X-check eigenstate
