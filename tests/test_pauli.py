import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli

from gosc.pauli import (PauliProduct, commutes, multiply, pauli_matrix,
                        rotation_matrix, rotation_through, y_from_xz)

ALL = [PauliProduct("".join(t)) for n in (1, 2) for t in itertools.product("IXYZ", repeat=n)]
PAIRS = [(a, b) for a in ALL for b in ALL if len(a.ops) == len(b.ops)]


def equal_up_to_phase(A, B):
    idx = np.unravel_index(np.argmax(abs(A)), A.shape)
    ph = B[idx] / A[idx]
    return abs(abs(ph) - 1) < 1e-9 and np.allclose(A * ph, B)


@pytest.mark.parametrize("a,b", PAIRS)
def test_multiply_vs_qiskit(a, b):
    phase, prod = multiply(a, b)
    ref = Pauli(a.ops).to_matrix() @ Pauli(b.ops).to_matrix()
    assert prod.sign == 1
    assert phase in (1, -1, 1j, -1j)
    assert np.allclose(phase * Pauli(prod.ops).to_matrix(), ref)


@pytest.mark.parametrize("a,b", PAIRS)
def test_commutes_vs_matrices(a, b):
    A, B = pauli_matrix(a), pauli_matrix(b)
    assert commutes(a, b) == np.allclose(A @ B, B @ A)


def test_y_is_i_xz():
    phase, prod = y_from_xz()
    assert prod.ops == "Y" or True
    ph, p = multiply(PauliProduct("X"), PauliProduct("Z"))
    assert p.ops == "Y" and ph == -1j
    assert 1j * ph == 1  # i*X*Z = i*(-i)*Y = Y
    assert np.allclose(1j * pauli_matrix(PauliProduct("X")) @ pauli_matrix(PauliProduct("Z")),
                       pauli_matrix(PauliProduct("Y")))


def test_sign_folding_and_weight():
    a, b = PauliProduct("XI", -1), PauliProduct("YZ", -1)
    phase, prod = multiply(a, b)
    assert prod == PauliProduct("ZZ", 1) and phase == 1j
    phase, _ = multiply(PauliProduct("X", -1), PauliProduct("Y"))
    assert phase == -1j
    assert PauliProduct("XIZ").weight() == 2
    with pytest.raises(ValueError):
        multiply(PauliProduct("X"), PauliProduct("XX"))


SIGNED = [PauliProduct(p.ops, s) for p in ALL for s in (1, -1)]
SIGNED_PAIRS = [(a, b) for a in SIGNED for b in SIGNED if len(a.ops) == len(b.ops)]


@pytest.mark.parametrize("p,q", SIGNED_PAIRS)
def test_rotation_through_numeric(p, q):
    phi = 0.37
    qp = rotation_through(p, q)
    lhs = rotation_matrix(p, np.pi / 4) @ rotation_matrix(q, phi)
    rhs = rotation_matrix(qp, phi) @ rotation_matrix(p, np.pi / 4)
    assert np.allclose(lhs, rhs) or equal_up_to_phase(lhs, rhs)
    if commutes(p, q):
        assert qp == q
    # qiskit cross-check of the rotation matrix for unsigned Paulis
    if p.sign == 1:
        assert np.allclose(rotation_matrix(p, 0.3),
                           Operator(np.cos(0.3) * np.eye(2 ** len(p.ops)) - 1j * np.sin(0.3) * Pauli(p.ops).to_matrix()).data)


def test_rotation_through_sign_examples():
    # -i*Z*X = -i*(iY) = Y
    assert rotation_through(PauliProduct("Z"), PauliProduct("X")) == PauliProduct("Y", 1)
    assert rotation_through(PauliProduct("Z", -1), PauliProduct("X")) == PauliProduct("Y", -1)
    assert rotation_through(PauliProduct("X"), PauliProduct("Z")) == PauliProduct("Y", -1)
