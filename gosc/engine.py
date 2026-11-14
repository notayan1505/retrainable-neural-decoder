"""Schedule executor / validator for the tile game.

A schedule is a list of time-steps; ops within a step run in parallel and must
use pairwise disjoint tiles (checked against the board at the START of the
step, so an op cannot use a patch created in the same step). SIMPLIFICATION:
a step lasts max(cost(op)) time units, so a cost-3 Rotate in a step with other
ops simply makes that step 3 long. total_time is the sum of step durations.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .board import DIRS, OPP, Board, IllegalMove, Patch, is_connected, neighbor
from .ops import CornerMove, Grow, Init, MeasureSingle, PPM, Rotate, Shrink, cost

__all__ = ["Engine", "Result", "IllegalMove"]


@dataclass
class Result:
    total_time: int
    trace: list = field(default_factory=list)  # ASCII board snapshot after each step
    step_times: list = field(default_factory=list)
    boards: list = field(default_factory=list)  # deepcopy of Board: initial (index 0) + after each step


def tiles_used(op, board: Board) -> set:
    if isinstance(op, Init):
        return set(op.tiles)
    if isinstance(op, PPM):
        used = set(op.ancilla_tiles)
        for pid, _ in op.terms:
            used |= board.get(pid).tiles
        return used
    used = set(board.get(op.patch_id).tiles)
    if isinstance(op, Grow):
        used |= set(op.tiles)
    elif isinstance(op, CornerMove):
        used.add(tuple(op.add_tile))
    return used


def _facing(patch: Patch, region) -> dict:
    """Edge types of `patch` segments whose outside neighbor lies in `region`."""
    return {seg: ty for seg, ty in patch.edges.items() if neighbor(*seg) in region}


def _check_ppm(op: PPM, board: Board):
    ids = [pid for pid, _ in op.terms]
    if len(op.terms) < 2 and not op.ancilla_tiles:
        raise IllegalMove("PPM needs at least two patches (or an ancilla region)")
    if len(set(ids)) != len(ids):
        raise IllegalMove("PPM lists a patch more than once")
    patches = {pid: board.get(pid) for pid in ids}
    if op.ancilla_tiles:
        for t in op.ancilla_tiles:
            if not board.is_free(t):
                raise IllegalMove(f"PPM ancilla tile {t} is not a free tile")
        if not is_connected(op.ancilla_tiles):
            raise IllegalMove("PPM ancilla tiles are not a connected region")
        for pid, pauli in op.terms:
            types = set(_facing(patches[pid], op.ancilla_tiles).values())
            need = {"X", "Z"} if pauli == "Y" else {pauli}
            if not types:
                raise IllegalMove(f"patch {pid!r} is not adjacent to the PPM ancilla region")
            if not need <= types:
                what = "Y requires both an X and a Z edge exposed" if pauli == "Y" else f"needs an exposed {pauli} edge"
                raise IllegalMove(f"patch {pid!r}: {what} on the ancilla boundary (exposed: {sorted(types)})")
    else:
        if len(op.terms) != 2:
            raise IllegalMove("direct PPM (no ancilla region) needs exactly two patches")
        (a, pa), (b, pb) = op.terms
        pairs = board.adjacent_edges(patches[a], patches[b])
        if not pairs:
            raise IllegalMove(f"patches {a!r} and {b!r} are not adjacent (no facing edges)")
        for pid, pauli in op.terms:
            idx = 0 if pid == a else 1
            need = {"X", "Z"} if pauli == "Y" else {pauli}
            ok = {patches[a].edges[p[0]] for p in pairs if patches[a].edges[p[0]] == patches[b].edges[p[1]]}
            if not need <= ok:
                raise IllegalMove(
                    f"PPM {a!r}-{b!r}: no compatible facing edge pair of type(s) {sorted(need - ok)} for patch {pid!r}")


class Engine:
    def __init__(self, board: Board):
        self.board = board

    # -- single op application (validates, mutates board) ---------------------
    def _apply(self, op):
        b = self.board
        if isinstance(op, Init):
            if len(op.tiles) == 1:
                p = Patch.square(op.patch_id, next(iter(op.tiles)), op.orientation)
            else:
                p = Patch.from_tiles(op.patch_id, op.tiles, None, op.orientation)
            p.state = op.state
            b.add_patch(p)
        elif isinstance(op, MeasureSingle):
            b.remove_patch(op.patch_id)
        elif isinstance(op, PPM):
            _check_ppm(op, b)
        elif isinstance(op, Grow):
            p = b.get(op.patch_id)
            run = list(op.tiles)
            for t in run:
                if not b.is_free(t):
                    raise IllegalMove(f"Grow: tile {t} is occupied or out of range")
            if len(run) > 1:
                d = (run[1][0] - run[0][0], run[1][1] - run[0][1])
                if d not in DIRS.values() or any(
                        (run[i + 1][0] - run[i][0], run[i + 1][1] - run[i][1]) != d for i in range(len(run) - 1)):
                    raise IllegalMove("Grow: tiles must form a straight contiguous run")
            if not any(neighbor(t, s) == run[0] for t in p.tiles for s in DIRS):
                raise IllegalMove(f"Grow: tile {run[0]} is not adjacent to patch {op.patch_id!r}")
            b.replace_patch(p.with_tiles(p.tiles | set(run), dict(op.edge_overrides)))
        elif isinstance(op, Shrink):
            p = b.get(op.patch_id)
            if not op.tiles <= p.tiles:
                raise IllegalMove(f"Shrink: tiles {sorted(op.tiles - p.tiles)} not in patch {op.patch_id!r}")
            if op.tiles == p.tiles:
                raise IllegalMove("Shrink would remove the whole patch (use MeasureSingle)")
            b.replace_patch(p.with_tiles(p.tiles - op.tiles))
        elif isinstance(op, CornerMove):
            p = b.get(op.patch_id)
            rem, add = tuple(op.remove_tile), tuple(op.add_tile)
            if rem not in p.tiles:
                raise IllegalMove(f"CornerMove: {rem} not in patch {op.patch_id!r}")
            if not b.is_free(add):
                raise IllegalMove(f"CornerMove: tile {add} is occupied or out of range")
            rest = p.tiles - {rem}
            if not any(neighbor(t, s) == add for t in rest for s in DIRS):
                raise IllegalMove(f"CornerMove: {add} is not adjacent to the rest of the patch")
            b.replace_patch(p.with_tiles(rest | {add}))
        elif isinstance(op, Rotate):
            b.replace_patch(b.get(op.patch_id).rotated())
        else:
            raise IllegalMove(f"unknown op {op!r}")

    def run(self, schedule) -> Result:
        res = Result(0)
        res.boards.append(copy.deepcopy(self.board))
        for i, step in enumerate(schedule):
            used_by = {}
            for op in step:
                for t in tiles_used(op, self.board):
                    if t in used_by:
                        raise IllegalMove(
                            f"step {i}: ops {used_by[t]!r} and {op!r} both use tile {t}")
                    used_by[t] = op
            try:
                for op in step:
                    self._apply(op)
            except IllegalMove as e:
                raise IllegalMove(f"step {i}: {e}") from None
            dur = max((cost(op) for op in step), default=0)
            res.step_times.append(dur)
            res.total_time += dur
            res.trace.append(self.board.render_ascii())
            res.boards.append(copy.deepcopy(self.board))
        return res
