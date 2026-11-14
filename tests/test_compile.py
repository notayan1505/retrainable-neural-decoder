import random

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from gosc.compile import (Rot, gate_to_rotations, merge_equal, naive_layers,
                          simulate_unitary, t_layers, to_pauli_rotations)
from gosc.pauli import PauliProduct

GATES1 = ["h", "s", "sdg", "t", "tdg", "x", "y", "z"]


def same_up_to_phase(a, b):
    k = np.argmax(np.abs(b))
    ph = a.flat[k] / b.flat[k]
    return abs(abs(ph) - 1) < 1e-9 and np.allclose(a, ph * b, atol=1e-9)


def qiskit_unitary(circ, n):
    qc = QuantumCircuit(n)
    for g, qs in circ:
        getattr(qc, g)(*qs)
    return Operator(qc).data


def random_circuit(rng, n, m):
    c = []
    for _ in range(m):
        if n > 1 and rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            c.append(("cx", (a, b)))
        else:
            c.append((rng.choice(GATES1), (rng.randrange(n),)))
    return c


def circuits():
    rng = random.Random(1234)
    for n in (1, 2, 3):
        for _ in range(25):
            yield n, random_circuit(rng, n, rng.randint(1, 10))


@pytest.mark.parametrize("gate,qs,n", [(g, (0,), 2) for g in GATES1] + [(g, (1,), 2) for g in GATES1]
                         + [("cx", (0, 1), 2), ("cx", (1, 0), 2), ("cx", (2, 0), 3)])
def test_single_gates(gate, qs, n):
    u = simulate_unitary(gate_to_rotations(gate, qs, n), n)
    assert same_up_to_phase(u, qiskit_unitary([(gate, qs)], n))


def test_full_circuits():
    for n, c in circuits():
        cc = to_pauli_rotations(c, n)
        assert cc.t_count == sum(g in ("t", "tdg") for g, _ in c)
        u = simulate_unitary(cc.t_rotations + [], n)
        u = simulate_unitary(cc.clifford_tail, n) @ u
        assert same_up_to_phase(u, qiskit_unitary(c, n)), c


def flat(layers):
    return [r for l in layers for r in l]


def check_layers(cc, n):
    ref = simulate_unitary(cc.t_rotations, n)
    naive = naive_layers(cc.t_rotations)
    greedy = t_layers(cc.t_rotations)
    assert len(greedy) <= len(naive)
    for ls in (naive, greedy):
        assert sorted(map(repr, flat(ls))) == sorted(map(repr, cc.t_rotations))
        assert np.allclose(simulate_unitary(flat(ls), n), ref)
    from gosc.pauli import commutes
    for l in greedy:
        assert all(commutes(a.axis, b.axis) for a in l for b in l)


def test_layers_random():
    for n, c in circuits():
        check_layers(to_pauli_rotations(c, n), n)


def test_merge_random():
    for n, c in circuits():
        cc = to_pauli_rotations(c, n)
        layers = t_layers(cc.t_rotations)
        new, pushed = merge_equal(layers)
        assert len(flat(new)) <= cc.t_count
        assert (cc.t_count - len(flat(new))) % 2 == 0
        u = simulate_unitary(cc.clifford_tail, n) @ simulate_unitary(pushed, n) \
            @ simulate_unitary(flat(new), n)
        assert same_up_to_phase(u, qiskit_unitary(c, n)), c


def test_merge_reduces():
    c = [("t", (0,)), ("t", (0,)), ("t", (1,)), ("tdg", (1,)), ("t", (0,))]
    cc = to_pauli_rotations(c, 2)
    new, pushed = merge_equal(t_layers(cc.t_rotations))
    assert cc.t_count == 5 and len(flat(new)) == 1 and len(pushed) == 1
    u = simulate_unitary(cc.clifford_tail, 2) @ simulate_unitary(pushed, 2) @ simulate_unitary(flat(new), 2)
    assert same_up_to_phase(u, qiskit_unitary(c, 2))


def test_greedy_reduction_example():
    n = 3
    P = lambda s: Rot(PauliProduct(s), "pi8")
    rots = [P("ZII"), P("XII"), P("IZI"), P("IXI")]  # a, b, c, d
    assert len(naive_layers(rots)) == 3
    g = t_layers(rots)
    assert len(g) == 2
    assert {r.axis.ops for r in g[0]} == {"ZII", "IZI"}
    assert {r.axis.ops for r in g[1]} == {"XII", "IXI"}
    assert np.allclose(simulate_unitary(flat(g), n), simulate_unitary(rots, n))
    # same thing from a real circuit: H gates give X-type axes via the tail
    c = [("t", (2,)), ("h", (2,)), ("t", (2,)), ("t", (1,)), ("h", (1,)), ("t", (1,))]
    cc = to_pauli_rotations(c, 3)
    assert len(t_layers(cc.t_rotations)) <= len(naive_layers(cc.t_rotations))
    check_layers(cc, 3)
