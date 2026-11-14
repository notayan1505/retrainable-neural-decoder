"""Data blocks from Litinski, "A Game of Surface Codes" (Quantum 3, 128, 2019).

Cost unit: one *time step* (~ d code cycles). Tile = one d x d patch footprint.

Three data-block designs for n data qubits:

* compact      : tiles = ceil(1.5 n) + 3, up to 9 time steps per magic state.
* intermediate : tiles = 2 n + 4,         up to 5 time steps per magic state.
* fast         : tiles = 2 n + ceil(sqrt(8 n)) + 1, always 1 time step.

``consume_time`` is a *simplified* per-Pauli cost model (see its docstring);
it reproduces the paper's worst cases (9 / 5 / 1) and the Fig. 11 example,
but does not simulate the actual patch layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DataBlock",
    "compact",
    "intermediate",
    "fast",
    "consume_time",
    "pi4_cost",
    "x_support",
    "fig11_example",
]


@dataclass(frozen=True)
class DataBlock:
    name: str
    n: int
    tiles: int
    max_consume: int


def compact(n: int) -> DataBlock:
    """Compact block: two rows of n/2 patches plus ancilla region.

    tiles = ceil(1.5 n) + 3. For even n this is ``3n//2 + 3`` (paper's
    formula); for odd n the paper does not give a layout, we round up
    (one row gets the extra qubit), i.e. ``(3n + 1)//2 + 3``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return DataBlock("compact", n, (3 * n + 1) // 2 + 3, 9)


def intermediate(n: int) -> DataBlock:
    """Intermediate block: one row of n patches, tiles = 2n + 4."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return DataBlock("intermediate", n, 2 * n + 4, 5)


def fast(n: int) -> DataBlock:
    """Fast block: two-tile patches, tiles = 2n + ceil(sqrt(8n)) + 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return DataBlock("fast", n, 2 * n + math.ceil(math.sqrt(8 * n)) + 1, 1)


def _validate(block: DataBlock, pauli: str) -> str:
    p = pauli.upper()
    if len(p) != block.n:
        raise ValueError(f"Pauli length {len(p)} != block size {block.n}")
    if any(c not in "IXYZ" for c in p):
        raise ValueError(f"invalid Pauli string {pauli!r}")
    return p


def pi4_cost(pauli: str) -> int:
    """Cost of the pi/4 rotations that handle Y operators.

    Simplification of the paper's rule: 0 if no Y, 1 if the number of Y's is
    odd (one pi/4 rotation), 2 if the number of Y's is even and > 0.
    """
    ny = pauli.upper().count("Y")
    if ny == 0:
        return 0
    return 1 if ny % 2 else 2


def x_support(pauli: str) -> list[int]:
    """Indices of qubits with an X *component* (X or Y, since Y = iXZ).

    In the compact/intermediate blocks only Z edges face the ancilla region,
    so any qubit whose operator involves X (including Y, which after the
    pi/4 handling is still accessed via its X edge -- cf. Fig. 11 where Y's
    are turned into X's) requires a patch rotation.
    """
    return [i for i, c in enumerate(pauli.upper()) if c in "XY"]


def consume_time(block: DataBlock, pauli: str) -> int:
    """Time steps to consume one magic state for a pi/8 rotation about ``pauli``.

    Simplified model (documented choices):

    * fast: always 1.
    * intermediate: 1 + pi4_cost + (2 if any qubit has an X component
      [X or Y] else 0). The paper's patch-rotation overhead for the
      intermediate block is in {0,1,2}; we use the worst value 2 whenever a
      rotation is needed. Max = 1 + 2 + 2 = 5.
    * compact: 1 + pi4_cost + patch-rotation cost, where qubits are laid out
      in two rows: row 0 = indices [0, ceil(n/2)), row 1 = the rest.
      Rotation cost is 0 if no qubit has an X component (X or Y), 3 if all
      such qubits are in one row, 6 if they are in both rows (3 + 3, rows
      rotated one after another as in Fig. 11). Capped at 9.
      Max = 1 + 2 + 6 = 9.

    "X component" includes Y: this is the rule that makes the paper's Fig. 11
    example 'YIYZYY' cost exactly 9 (4 Y's -> 2, Y's in both rows -> 6).
    """
    p = _validate(block, pauli)
    if block.name == "fast":
        return 1
    xs = x_support(p)
    if block.name == "intermediate":
        return min(5, 1 + pi4_cost(p) + (2 if xs else 0))
    if block.name == "compact":
        half = math.ceil(block.n / 2)
        rows = {0 if i < half else 1 for i in xs}
        rot = 0 if not rows else (3 if len(rows) == 1 else 6)
        return min(9, 1 + pi4_cost(p) + rot)
    raise ValueError(f"unknown block type {block.name!r}")


def fig11_example() -> tuple[DataBlock, str]:
    """Fig. 11 of the paper: compact(6) block, rotation about 'YIYZYY' -> 9 steps."""
    return compact(6), "YIYZYY"
