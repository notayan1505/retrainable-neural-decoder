"""Tile board and patches for the Game of Surface Codes (Litinski).

Coordinates: tile (x, y), x to the East, y to the South (row 0 is the top row).
A Patch owns a set of tiles; every boundary segment of that tile union is a
(tile, side) pair mapped to 'X' (dotted edge) or 'Z' (solid edge).
"""
from __future__ import annotations

from dataclasses import dataclass, field

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
Tile = tuple
Segment = tuple  # (tile, side)


class IllegalMove(Exception):
    """Raised for any violation of the game rules."""


def neighbor(tile, side):
    dx, dy = DIRS[side]
    return (tile[0] + dx, tile[1] + dy)


def is_connected(tiles) -> bool:
    tiles = set(tiles)
    if not tiles:
        return False
    seen, stack = set(), [next(iter(tiles))]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        stack.extend(n for n in (neighbor(t, s) for s in DIRS) if n in tiles and n not in seen)
    return seen == tiles


def boundary_segments(tiles) -> list:
    tiles = set(tiles)
    return [(t, s) for t in sorted(tiles) for s in "NESW" if neighbor(t, s) not in tiles]


def _orientation_types(orientation: str) -> dict:
    if orientation not in ("ZNS", "XNS"):
        raise ValueError(f"orientation must be 'ZNS' or 'XNS', got {orientation!r}")
    ns = orientation[0]
    ew = "X" if ns == "Z" else "Z"
    return {"N": ns, "S": ns, "E": ew, "W": ew}


def _flip(t):
    return "X" if t == "Z" else "Z"


@dataclass
class Patch:
    id: object
    tiles: frozenset
    edges: dict = field(default_factory=dict)  # (tile, side) -> 'X' | 'Z'
    state: str | None = None  # informational: '0' / '+' from Init

    @classmethod
    def square(cls, id, tile, orientation="ZNS"):
        """Single-tile patch. 'ZNS': Z edges on N/S, X on E/W; 'XNS' swaps."""
        return cls.from_tiles(id, [tile], None, orientation)

    @classmethod
    def from_tiles(cls, id, tiles, edge_types=None, orientation="ZNS"):
        """Patch on a connected tile union. `edge_types` is a (possibly partial)
        dict {(tile, side): 'X'|'Z'}; unspecified boundary segments follow
        `orientation`. Keys that are not boundary segments are rejected."""
        tiles = frozenset(tuple(t) for t in tiles)
        if not is_connected(tiles):
            raise IllegalMove(f"patch {id!r}: tiles must be non-empty and connected")
        defaults = _orientation_types(orientation)
        segs = boundary_segments(tiles)
        edge_types = dict(edge_types or {})
        for k, v in edge_types.items():
            if k not in segs:
                raise IllegalMove(f"patch {id!r}: {k} is not a boundary segment")
            if v not in ("X", "Z"):
                raise IllegalMove(f"patch {id!r}: edge type must be 'X' or 'Z'")
        return cls(id, tiles, {seg: edge_types.get(seg, defaults[seg[1]]) for seg in segs})

    def axis_types(self) -> dict:
        """Dominant edge type on the N/S axis and E/W axis (for retyping)."""
        ns = next((v for (t, s), v in self.edges.items() if s in "NS"), None)
        ew = next((v for (t, s), v in self.edges.items() if s in "EW"), None)
        if ns is None and ew is None:
            ns, ew = "Z", "X"
        elif ns is None:
            ns = _flip(ew)
        elif ew is None:
            ew = _flip(ns)
        return {"N": ns, "S": ns, "E": ew, "W": ew}

    def with_tiles(self, tiles, overrides=None) -> "Patch":
        """New patch on `tiles`; surviving segments keep their type, new
        segments take the axis default, `overrides` win."""
        tiles = frozenset(tiles)
        if not is_connected(tiles):
            raise IllegalMove(f"patch {self.id!r}: resulting tiles are not connected")
        ax = self.axis_types()
        overrides = dict(overrides or {})
        edges = {}
        for seg in boundary_segments(tiles):
            edges[seg] = overrides.get(seg, self.edges.get(seg, ax[seg[1]]))
        return Patch(self.id, tiles, edges, self.state)

    def rotated(self) -> "Patch":
        return Patch(self.id, self.tiles, {k: _flip(v) for k, v in self.edges.items()}, self.state)


class Board:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.patches: dict = {}
        self.occupancy: dict = {}  # tile -> patch id

    def in_range(self, tile) -> bool:
        return 0 <= tile[0] < self.width and 0 <= tile[1] < self.height

    def is_free(self, tile) -> bool:
        return self.in_range(tile) and tile not in self.occupancy

    def add_patch(self, patch: Patch):
        if patch.id in self.patches:
            raise IllegalMove(f"patch {patch.id!r} already exists")
        for t in patch.tiles:
            if not self.in_range(t):
                raise IllegalMove(f"tile {t} out of range for patch {patch.id!r}")
            if t in self.occupancy:
                raise IllegalMove(f"tile {t} already occupied by patch {self.occupancy[t]!r}")
        self.patches[patch.id] = patch
        for t in patch.tiles:
            self.occupancy[t] = patch.id

    def remove_patch(self, patch_id) -> Patch:
        if patch_id not in self.patches:
            raise IllegalMove(f"missing patch {patch_id!r}")
        p = self.patches.pop(patch_id)
        for t in p.tiles:
            del self.occupancy[t]
        return p

    def replace_patch(self, new: Patch):
        """Swap the patch with id new.id for `new` (validated atomically)."""
        old = self.remove_patch(new.id)
        try:
            self.add_patch(new)
        except IllegalMove:
            self.add_patch(old)
            raise

    def get(self, patch_id) -> Patch:
        if patch_id not in self.patches:
            raise IllegalMove(f"missing patch {patch_id!r}")
        return self.patches[patch_id]

    def free_tiles(self) -> set:
        return {(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.occupancy}

    def adjacent_edges(self, patch_a, patch_b) -> list:
        """Pairs ((tile_a, side_a), (tile_b, side_b)) of facing edge segments."""
        a = patch_a if isinstance(patch_a, Patch) else self.get(patch_a)
        b = patch_b if isinstance(patch_b, Patch) else self.get(patch_b)
        out = []
        for (t, s) in a.edges:
            n = neighbor(t, s)
            if n in b.tiles and (n, OPP[s]) in b.edges:
                out.append(((t, s), (n, OPP[s])))
        return out

    def render_ascii(self) -> str:
        """One char per tile: first char of patch id, '.' for free tiles."""
        rows = []
        for y in range(self.height):
            rows.append(" ".join(
                str(self.occupancy[(x, y)])[0] if (x, y) in self.occupancy else "." for x in range(self.width)))
        return "\n".join(rows)
