"""Clifford+T -> Pauli-product rotations (Litinski, sec. 1).

Conventions
-----------
* Rot(axis, kind) is exp(-i*theta*axis) with theta = pi/8, pi/4, pi/2 for kind
  'pi8','pi4','pi2'.  Negative angles are encoded as axis.sign == -1.
* Qubit q of a circuit lives at ``ops[n-1-q]`` of a PauliProduct, so axes and
  matrices agree with qiskit's little-endian ``Operator`` / ``Pauli`` labels.
* Rotation lists are in TIME order; the unitary is R_k ... R_1 (matrix product).
* Result: U_circuit = (clifford_tail applied in order) . (t_rotations in order),
  up to global phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pauli import PauliProduct, commutes, rotation_matrix, rotation_through

_ANGLE = {"pi8": np.pi / 8, "pi4": np.pi / 4, "pi2": np.pi / 2}


@dataclass(frozen=True)
class Rot:
    axis: PauliProduct
    angle_kind: str

    def __post_init__(self):
        if self.angle_kind not in _ANGLE:
            raise ValueError(f"bad angle_kind {self.angle_kind!r}")


@dataclass
class CompiledCircuit:
    n: int
    t_rotations: list = field(default_factory=list)
    clifford_tail: list = field(default_factory=list)

    @property
    def t_count(self) -> int:
        return len(self.t_rotations)


def _pp(n: int, paulis: dict, sign: int = 1) -> PauliProduct:
    ops = ["I"] * n
    for q, c in paulis.items():
        ops[n - 1 - q] = c
    return PauliProduct("".join(ops), sign)


def gate_to_rotations(gate: str, qubits, n: int) -> list:
    g = gate.lower()
    q = qubits[0]
    z = lambda s=1: Rot(_pp(n, {q: "Z"}, s), "pi4")
    if g == "s":
        return [Rot(_pp(n, {q: "Z"}), "pi4")]
    if g == "sdg":
        return [Rot(_pp(n, {q: "Z"}, -1), "pi4")]
    if g == "t":
        return [Rot(_pp(n, {q: "Z"}), "pi8")]
    if g == "tdg":
        return [Rot(_pp(n, {q: "Z"}, -1), "pi8")]
    if g == "h":
        # Z(pi/4) X(pi/4) Z(pi/4): palindromic, so time order is unambiguous
        return [Rot(_pp(n, {q: "Z"}), "pi4"), Rot(_pp(n, {q: "X"}), "pi4"),
                Rot(_pp(n, {q: "Z"}), "pi4")]
    if g in ("x", "y", "z"):
        return [Rot(_pp(n, {q: g.upper()}), "pi2")]
    if g == "cx":
        c, t = qubits
        # all three commute; paper: (ZcXt)_{pi/4} (Xt)_{-pi/4} (Zc)_{-pi/4}
        return [Rot(_pp(n, {c: "Z"}, -1), "pi4"), Rot(_pp(n, {t: "X"}, -1), "pi4"),
                Rot(_pp(n, {c: "Z", t: "X"}), "pi4")]
    raise ValueError(f"unsupported gate {gate!r}")


def _conj(c: Rot, q: PauliProduct) -> PauliProduct:
    """Q' with Q . C = C . Q'  (i.e. C^dag Q C), C a pi/4 or pi/2 rotation."""
    if commutes(c.axis, q):
        return q
    if c.angle_kind == "pi2":
        return PauliProduct(q.ops, -q.sign)
    inv = PauliProduct(c.axis.ops, -c.axis.sign)
    return rotation_through(inv, q)


def _through_tail(q: PauliProduct, tail) -> PauliProduct:
    for c in reversed(tail):
        q = _conj(c, q)
    return q


def to_pauli_rotations(circuit, n: int) -> CompiledCircuit:
    out = CompiledCircuit(n)
    for gate, qubits in circuit:
        for r in gate_to_rotations(gate, qubits, n):
            if r.angle_kind == "pi8":
                # Q . Tail = Tail . Q'  (Tail already sits at the end)
                out.t_rotations.append(Rot(_through_tail(r.axis, out.clifford_tail), "pi8"))
            else:
                out.clifford_tail.append(r)  # G . Tail: already at the end
    return out


def naive_layers(rotations) -> list:
    layers: list = []
    for r in rotations:
        if layers and all(commutes(r.axis, s.axis) for s in layers[-1]):
            layers[-1].append(r)
        else:
            layers.append([r])
    return layers


def t_layers(rotations, greedy: bool = True) -> list:
    layers = naive_layers(rotations)
    if not greedy:
        return layers
    changed = True
    while changed:
        changed = False
        for i in range(len(layers) - 1):
            for r in list(layers[i + 1]):
                if all(commutes(r.axis, s.axis) for s in layers[i]):
                    layers[i + 1].remove(r)
                    layers[i].append(r)
                    changed = True
        layers = [l for l in layers if l]
    return layers


def merge_equal(layers):
    """Merge equal pi/8 rotations in a layer into pi/4 (T.T = S); opposite signs cancel.

    Returns (new_layers, pushed_pi4) where pushed_pi4 are the created pi/4
    rotations moved to the end, in application order.  The total circuit is
    ``clifford_tail . pushed_pi4 . layers`` i.e. pushed_pi4 goes BEFORE the old
    clifford_tail.  Later layers' axes are adjusted accordingly.
    """
    pushed: list = []
    new_layers = []
    for layer in layers:
        axes = [_through_tail(r.axis, pushed) for r in layer]
        kept = []
        merged = []
        pool = list(axes)
        while pool:
            a = pool.pop(0)
            opp = PauliProduct(a.ops, -a.sign)
            if opp in pool:
                pool.remove(opp)
            elif a in pool:
                pool.remove(a)
                merged.append(Rot(a, "pi4"))
            else:
                kept.append(Rot(a, "pi8"))
        pushed.extend(merged)
        if kept:
            new_layers.append(kept)
    return new_layers, pushed


def simulate_unitary(rotations, n: int) -> np.ndarray:
    u = np.eye(2 ** n, dtype=complex)
    for r in rotations:
        u = rotation_matrix(r.axis, _ANGLE[r.angle_kind]) @ u
    return u
