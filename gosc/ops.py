"""Operations of the tile game and their costs (single source of truth: COSTS)."""
from __future__ import annotations

from dataclasses import dataclass

COSTS = {
    "init_patch": 0,
    "measure_single": 0,
    "pp_measure": 1,
    "grow": 1,
    "shrink": 0,
    "corner_move": 1,
    "rotate_patch": 3,
    "y_measure_with_shared_edge": 1,
    "y_measure_single": 2,  # deform 1 + init ancilla 0 + joint measure 1 + deform back 0
}


def _is_tile(x):
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) for i in x)


def _tile_seq(x):
    return (x,) if _is_tile(x) else tuple(tuple(t) for t in x)


@dataclass(frozen=True)
class Init:
    patch_id: object
    state: str  # '0' or '+'
    tiles: object  # a tile (x, y) or an iterable of tiles; normalised to frozenset
    orientation: str = "ZNS"

    def __post_init__(self):
        if self.state not in ("0", "+"):
            raise ValueError("Init state must be '0' or '+'")
        object.__setattr__(self, "tiles", frozenset(_tile_seq(self.tiles)))


@dataclass(frozen=True)
class MeasureSingle:
    patch_id: object
    basis: str  # 'X' or 'Z'

    def __post_init__(self):
        if self.basis not in ("X", "Z"):
            raise ValueError("MeasureSingle basis must be 'X' or 'Z'")


@dataclass(frozen=True)
class PPM:
    """Pauli product measurement. terms: ((patch_id, 'X'|'Y'|'Z'), ...).
    With empty ancilla_tiles the (two) patches must be directly adjacent with
    compatible facing edges; otherwise the free, connected ancilla region must
    touch an edge of the right type on every term's patch (Y: both X and Z)."""
    terms: tuple
    ancilla_tiles: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(tuple(t) for t in self.terms))
        object.__setattr__(self, "ancilla_tiles", frozenset(tuple(t) for t in self.ancilla_tiles))
        for _, p in self.terms:
            if p not in ("X", "Y", "Z"):
                raise ValueError("PPM Pauli must be X, Y or Z")


@dataclass(frozen=True)
class Grow:
    """Extend a patch onto a straight, contiguous run of free tiles (one op, cost 1).
    `tiles` is one tile or a run starting next to the patch. `edge_overrides`
    optionally sets types of new boundary segments: ((tile, side), 'X'|'Z'), ..."""
    patch_id: object
    tiles: object
    edge_overrides: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tiles", _tile_seq(self.tiles))

    @property
    def tile(self):
        return self.tiles[0]


@dataclass(frozen=True)
class Shrink:
    patch_id: object
    tiles: object  # tile or iterable of tiles to release

    def __post_init__(self):
        object.__setattr__(self, "tiles", frozenset(_tile_seq(self.tiles)))

    @property
    def tile(self):
        return next(iter(self.tiles))


@dataclass(frozen=True)
class CornerMove:
    """Simplified: release `remove_tile` and take adjacent free `add_tile` (cost 1)."""
    patch_id: object
    remove_tile: tuple
    add_tile: tuple


@dataclass(frozen=True)
class Rotate:
    patch_id: object


def cost(op) -> int:
    return COSTS[{
        Init: "init_patch", MeasureSingle: "measure_single", PPM: "pp_measure",
        Grow: "grow", Shrink: "shrink", CornerMove: "corner_move", Rotate: "rotate_patch",
    }[type(op)]]
