import json
import os

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from qdrift.data import circuits as C
from qdrift.data.stein import detectors_from_raw, parse_snapshot


def _fez_edges():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return [tuple(e) for e in FakeFez().coupling_map.get_edges()]


def test_find_chains_fez():
    edges = _fez_edges()
    es = {frozenset(e) for e in edges}
    chains = C.find_chains(edges, 5, 4, seed=1)
    assert len(chains) == 4
    flat = [q for c in chains for q in c]
    assert len(flat) == len(set(flat))
    for c in chains:
        assert len(c) == 9
        assert all(frozenset(p) in es for p in zip(c, c[1:]))
    # CouplingMap input and avoid
    chains2 = C.find_chains(CouplingMap(edges), 3, 2, avoid={0, 1, 2}, seed=2)
    assert not ({0, 1, 2} & {q for c in chains2 for q in c})


def test_find_chains_line_and_errors():
    line = [[i, i + 1] for i in range(9)]
    chains = C.find_chains(line, 3, 5)
    assert len(chains) == 2 and all(len(c) == 5 for c in chains)
    two = C.find_chains(line, 3, 2)
    assert len(two) == 2
    with pytest.raises(ValueError):
        C.find_chains(line, 6, 1)  # needs 11 qubits


def test_register_names_and_sizes():
    qc = C.build_circuit([[4, 5, 6, 7, 8], [10, 11, 12, 13, 14]], 3, "Z", 0)
    got = {c.name: c.size for c in qc.cregs}
    assert got == {"c_data_4_5_6_7_8": 3, "c_syndrome_4_5_6_7_8": 6,
                   "c_data_10_11_12_13_14": 3, "c_syndrome_10_11_12_13_14": 6}


def _bits(qc, shots=20, method="stabilizer"):
    res = AerSimulator(method=method).run(qc, shots=shots, seed_simulator=1).result()
    return res.get_counts()


def _ideal_arrays(qc, chain, shots=20):
    from qiskit_aer.primitives import SamplerV2
    pub = SamplerV2(options={"backend_options": {"method": "stabilizer"}}).run([qc], shots=shots).result()[0]
    d = (len(chain) + 1) // 2
    suf = C._suffix(chain)
    return (C.bitarray_to_bits(pub.data["c_data_" + suf]),
            C.bitarray_to_bits(pub.data["c_syndrome_" + suf]))


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("state", [0, 1])
def test_ideal_no_errors(basis, state):
    chain = [0, 1, 2, 3, 4]
    qc = C.build_circuit([chain], 2, basis, state)
    data, synd = _ideal_arrays(qc, chain)
    assert data.shape == (20, 3) and synd.shape == (20, 4)
    assert (data == state).all()
    det, lab = detectors_from_raw(synd, data, state)
    assert det.shape == (20, 3, 2)
    assert det.sum() == 0 and lab.sum() == 0


def test_bit_order_proof():
    qc = QuantumCircuit(3)
    from qiskit import ClassicalRegister
    cr = ClassicalRegister(3, "cc")
    qc.add_register(cr)
    qc.x(1)  # only classical bit 1 is 1
    qc.measure(0, cr[0])
    qc.measure(1, cr[1])
    qc.measure(2, cr[2])
    from qiskit_aer.primitives import SamplerV2
    pub = SamplerV2().run([qc], shots=8).result()[0]
    bits = C.bitarray_to_bits(pub.data["cc"])
    assert (bits == np.array([0, 1, 0])).all()
    # wider than a byte too
    qc2 = QuantumCircuit(10)
    cr2 = ClassicalRegister(10, "w")
    qc2.add_register(cr2)
    qc2.x(9)
    qc2.measure(range(10), cr2)
    pub = SamplerV2().run([qc2], shots=4).result()[0]
    bits = C.bitarray_to_bits(pub.data["w"])
    exp = np.zeros(10, dtype=int)
    exp[9] = 1
    assert (bits == exp).all()


def test_injected_data_error_fires_expected_detector():
    # d=3, r=2, Z basis; X on data 0 (qubit 0) between round 0 and round 1.
    chain = [0, 1, 2, 3, 4]
    base = C.build_circuit([chain], 2, "Z", 0)
    ops = list(base.data)
    # first measure of a check ends round 0 (reset follows the first measure); insert after the 2nd reset
    resets = [i for i, inst in enumerate(ops) if inst.operation.name == "reset"]
    pos = resets[1] + 1
    qc = base.copy_empty_like()
    for i, inst in enumerate(ops):
        if i == pos:
            qc.x(0)
        qc.append(inst)
    data, synd = _ideal_arrays(qc, chain)
    det, lab = detectors_from_raw(synd, data, 0)
    assert (lab == 1).all()
    expected = np.zeros((3, 2), dtype=int)
    expected[1, 0] = 1  # round 1 check 0 flips; later rounds / final compare against it
    expected[2, 0] = 0  # final: q0^q1 = 1, ^ m[r-1,0] = 1 -> 0
    assert (det == expected).all()


def test_fake_backend_end_to_end(tmp_path):
    paths = C.run_ibm("ibm_fez", tmp_path, [(3, 3, "Z")], shots=256, n_chains=2)
    assert len(paths) == 1
    p = paths[0]
    assert p.parts[-3:] == ("ibm_fez_fake", "d3_r3", "job_1") or p.parts[-2] == "d3_r3"
    ds = parse_snapshot(p)
    assert len(ds) == 2 * 256 * 2
    assert ds.detectors.shape == (len(ds), 4, 2)
    assert 0.001 < ds.detectors.mean() < 0.35
    assert ds.labels.mean() < 0.3
    assert (p / "circuit_state0.qasm").exists() and (p / "circuit_state1.qasm").exists()
    info = json.loads((p / "info.json").read_text())
    assert info["backend"] == "ibm_fez_fake" and info["n_chains"] == 2
    cal = json.loads((p / "calibration.json").read_text())
    assert cal["backend_name"] == "ibm_fez_fake" and "last_update_date" in cal
    # second run takes the next job number
    p2 = C.run_ibm("ibm_fez", tmp_path, [(3, 1, "X")], shots=64, n_chains=1)[0]
    assert p2.name == "job_1" and p2.parent.name == "d3_r1"


@pytest.mark.skipif(os.environ.get("QDRIFT_SKIP_SLOW") == "1", reason="slow")
def test_fake_backend_x_basis(tmp_path):
    p = C.run_ibm("ibm_fez", tmp_path, [(3, 1, "X")], shots=128, n_chains=2)[0]
    ds = parse_snapshot(p)
    assert ds.detectors.shape[1:] == (2, 2)
    assert 0.001 < ds.detectors.mean() < 0.45


def test_real_requires_token_and_never_touches_network(monkeypatch, tmp_path):
    import qiskit_ibm_runtime

    def boom(*a, **k):
        pytest.fail("QiskitRuntimeService must not be constructed")

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", boom)
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="QISKIT_IBM_TOKEN"):
        C.run_ibm("ibm_fez", tmp_path, [(3, 1, "Z")], real=True)
