"""Pauli-product algebra (pure Python + numpy).

Conventions
-----------
``PauliProduct(ops, sign)`` represents ``sign * ops[0] (x) ops[1] (x) ...`` with
``ops`` over 'IXYZ'; ops[0] is the leftmost tensor factor (same as the string
label of ``qiskit.quantum_info.Pauli``).

Rotation commutation
--------------------
With U_P(t) = exp(-i t P) and U_P(pi/4) = (I - iP)/sqrt(2), matrix products
compose right-to-left.  For anticommuting P, Q:

    U_P(pi/4) Q U_P(pi/4)^dag = -i P Q

hence the identity (verified numerically in tests):

    U_P(pi/4) U_Q(phi) = U_{Q'}(phi) U_P(pi/4),   Q' = -i P Q  (anticommuting)
                                                   Q' = Q      (commuting)

Note: Q' = -i*P*Q is the convention for moving P_{pi/4} from before to after
Q_phi in this (matrix product) order.  The opposite direction (moving it from
after to before) gives Q' = +i*P*Q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID = set("IXYZ")

_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# single-qubit product table: (a, b) -> (phase, c) with a*b = phase*c
_SINGLE: dict[tuple[str, str], tuple[complex, str]] = {}
for _a in "IXYZ":
    _SINGLE[("I", _a)] = (1, _a)
    _SINGLE[(_a, "I")] = (1, _a)
    _SINGLE[(_a, _a)] = (1, "I")
for _a, _b, _c in (("X", "Y", "Z"), ("Y", "Z", "X"), ("Z", "X", "Y")):
    _SINGLE[(_a, _b)] = (1j, _c)
    _SINGLE[(_b, _a)] = (-1j, _c)


@dataclass(frozen=True)
class PauliProduct:
    ops: str
    sign: int = 1

    def __post_init__(self):
        if not set(self.ops) <= _VALID:
            raise ValueError(f"invalid Pauli string {self.ops!r}")
        if self.sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")

    def weight(self) -> int:
        return sum(c != "I" for c in self.ops)


def multiply(a: PauliProduct, b: PauliProduct) -> tuple[complex, PauliProduct]:
    """Return (phase, product) with a*b = phase*product; product.sign == +1.

    The signs of a and b are folded into phase.
    """
    if len(a.ops) != len(b.ops):
        raise ValueError("length mismatch")
    phase: complex = a.sign * b.sign
    out = []
    for x, y in zip(a.ops, b.ops):
        p, c = _SINGLE[(x, y)]
        phase *= p
        out.append(c)
    return complex(phase), PauliProduct("".join(out), 1)


def commutes(a: PauliProduct, b: PauliProduct) -> bool:
    if len(a.ops) != len(b.ops):
        raise ValueError("length mismatch")
    anti = sum(x != "I" and y != "I" and x != y for x, y in zip(a.ops, b.ops))
    return anti % 2 == 0


def y_from_xz() -> tuple[complex, PauliProduct]:
    """Documents Y = i*X*Z: returns (phase, product) of X*Z = -i*Y, so Y = i*X*Z."""
    return multiply(PauliProduct("X"), PauliProduct("Z"))


def rotation_through(p_quarter: PauliProduct, q: PauliProduct) -> PauliProduct:
    """Q' such that U_P(pi/4) U_Q(phi) = U_Q'(phi) U_P(pi/4) (see module doc).

    Commuting: Q' = Q.  Anticommuting: Q' = -i*P*Q (Hermitian, sign tracked).
    """
    if commutes(p_quarter, q):
        return q
    phase, prod = multiply(p_quarter, q)
    s = -1j * phase  # real (+-1) for anticommuting pair
    assert abs(s.imag) < 1e-12
    return PauliProduct(prod.ops, 1 if s.real > 0 else -1)


def pauli_matrix(p: PauliProduct) -> np.ndarray:
    m = np.array([[1]], dtype=complex)
    for c in p.ops:
        m = np.kron(m, _MATS[c])
    return p.sign * m


def rotation_matrix(p: PauliProduct, theta: float) -> np.ndarray:
    """exp(-i theta P) = cos(theta) I - i sin(theta) P (sign of p included)."""
    m = pauli_matrix(p)
    return np.cos(theta) * np.eye(m.shape[0], dtype=complex) - 1j * np.sin(theta) * m
