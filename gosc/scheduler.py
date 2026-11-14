"""State-dependent magic-state consumption scheduler for data blocks.

Model (see docs/scheduler_notes.md for the full discussion)
-----------------------------------------------------------
*Compact block* state: one orientation per data qubit, ``'Z'`` or ``'X'``,
naming the patch edge that currently faces the ancilla aisle. Initially all
``'Z'``. Qubits are split into two rows like :mod:`gosc.blocks`:
row 0 = string indices ``[0, ceil(n/2))``, row 1 = the rest. (Indices are
positions in ``PauliProduct.ops``; with :mod:`gosc.compile`, circuit qubit q
sits at index ``n-1-q``.)

Rotation about Pauli string P (the sign is irrelevant for the cost):

* need of qubit i: ``'X'`` if P_i in {X, Y} (Y is accessed through its X edge,
  as in :func:`gosc.blocks.x_support`), ``'Z'`` if P_i == 'Z', none if 'I'.
* switching cost = 3 * (number of rows containing >= 1 qubit whose current
  orientation differs from its need). Rows switch one after another; all
  qubits of one row switch together.
* only qubits that *must* change are flipped; 'I' qubits and already-correct
  qubits keep their orientation (pre-emptive free flips are not allowed).
  The new orientation persists into the next rotation.
* cost = 1 + pi4_cost(P) + switching cost  (max 1 + 2 + 6 = 9).

From the all-'Z' state this equals :func:`gosc.blocks.consume_time`
(e.g. 'YIYZYY' on n=6 costs 9); repeating the same rotation then costs
only ``1 + pi4_cost``.

*Intermediate block*: stateless, cost = ``blocks.consume_time``.
*Fast block*: cost 1.

Pipeline: with k distillation blocks, interval = DistillationBlock.time / k,
magic state i (1-based) is ready at ``i*interval``; consumption is serial on
the single aisle: ``start_i = max(end_{i-1}, ready_i)``,
``end_i = start_i + cost_i``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from gosc.blocks import DataBlock, compact, consume_time, pi4_cost
from gosc.distillation import DistillationBlock
from gosc.pauli import PauliProduct, commutes

__all__ = [
    "SWITCH_COST_PER_ROW",
    "EXACT_LIMIT",
    "initial_state",
    "needs",
    "rotation_cost",
    "sequence_cost",
    "order_layer",
    "TraceEntry",
    "ScheduleResult",
    "LowerBound",
    "schedule",
    "simulate",
    "lower_bound",
]

SWITCH_COST_PER_ROW = 3
EXACT_LIMIT = 9  # layers up to this size are solved exactly by DP over subsets
STRATEGIES = ("naive", "greedy", "optimal")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ops(item: Any) -> str:
    """Pauli string of a PauliProduct, a compile.Rot (``.axis``) or a str."""
    if isinstance(item, str):
        return item.upper()
    if isinstance(item, PauliProduct):
        return item.ops
    axis = getattr(item, "axis", None)
    if isinstance(axis, PauliProduct):
        return axis.ops
    raise TypeError(f"cannot interpret {item!r} as a Pauli rotation")


def _pp(item: Any) -> PauliProduct:
    if isinstance(item, PauliProduct):
        return item
    if isinstance(item, str):
        return PauliProduct(item.upper())
    return item.axis


def initial_state(n: int) -> str:
    """All-'Z' orientation string of length n."""
    return "Z" * n


def needs(pauli: str) -> str:
    """Per-qubit required orientation: 'X' for X/Y, 'Z' for Z, '.' for I."""
    return "".join("X" if c in "XY" else ("Z" if c == "Z" else ".")
                   for c in pauli.upper())


def _rows(n: int) -> int:
    return math.ceil(n / 2)


def _switch(p: str, state: str, half: int) -> tuple[int, str]:
    rows = set()
    new = list(state)
    for i, c in enumerate(p):
        if c == "I":
            continue
        need = "Z" if c == "Z" else "X"
        if state[i] != need:
            rows.add(0 if i < half else 1)
            new[i] = need
    return SWITCH_COST_PER_ROW * len(rows), "".join(new)


def rotation_cost(P: Any, state: str | None, block: DataBlock) -> tuple[int, str | None]:
    """Cost of consuming one magic state for a pi/8 rotation about P.

    Returns ``(cost, new_state)``. For the compact block ``state`` is the
    orientation string (None means all-'Z'); for intermediate/fast blocks the
    model is stateless and ``state`` is returned unchanged.
    """
    p = _ops(P)
    if len(p) != block.n:
        raise ValueError(f"Pauli length {len(p)} != block size {block.n}")
    if block.name == "fast":
        return 1, state
    if block.name == "intermediate":
        return consume_time(block, p), state
    if block.name != "compact":
        raise ValueError(f"unknown block type {block.name!r}")
    if state is None:
        state = initial_state(block.n)
    if len(state) != block.n or set(state) - {"X", "Z"}:
        raise ValueError(f"bad state {state!r}")
    sw, new = _switch(p, state, _rows(block.n))
    return 1 + pi4_cost(p) + sw, new


def sequence_cost(seq: Sequence[Any], state: str | None, block: DataBlock) -> tuple[int, str | None]:
    """Total cost of consuming ``seq`` in order from ``state``; returns (cost, final_state)."""
    total = 0
    for r in seq:
        c, state = rotation_cost(r, state, block)
        total += c
    return total, state


# ---------------------------------------------------------------------------
# layer ordering
# ---------------------------------------------------------------------------

def _greedy_idx(paulis: list[str], state: str, block: DataBlock) -> list[int]:
    """Nearest neighbour on switching cost; ties -> smallest original index."""
    half = _rows(block.n)
    left = list(range(len(paulis)))
    out = []
    while left:
        best = min(left, key=lambda j: (_switch(paulis[j], state, half)[0], j))
        _, state = _switch(paulis[best], state, half)
        out.append(best)
        left.remove(best)
    return out


def _cost_idx(paulis: list[str], order: Sequence[int], state: str, block: DataBlock) -> tuple[int, str]:
    half = _rows(block.n)
    tot = 0
    for j in order:
        c, state = _switch(paulis[j], state, half)
        tot += c
    return tot, state


def _lookahead(state: str, nxt: list[str] | None, block: DataBlock) -> int:
    """Switching cost of the next layer under greedy from ``state`` (0 if none)."""
    if not nxt:
        return 0
    return _cost_idx(nxt, _greedy_idx(nxt, state, block), state, block)[0]


def _exact_idx(paulis: list[str], state: str, block: DataBlock,
               nxt: list[str] | None) -> list[int]:
    """Exact DP over (subset, orientation-state).

    The future cost depends only on the current orientation string, so
    keeping, per subset, the cheapest way to reach each distinct state is
    exact (a (subset, last-rotation) DP would not be: the state after a
    prefix depends on its whole order).
    """
    m = len(paulis)
    half = _rows(block.n)
    # dp[mask]: state -> (cost, order tuple); ties broken lexicographically
    dp: list[dict[str, tuple[int, tuple[int, ...]]]] = [dict() for _ in range(1 << m)]
    dp[0][state] = (0, ())
    for mask in range(1 << m):
        for st, (c, order) in dp[mask].items():
            for j in range(m):
                if mask >> j & 1:
                    continue
                sw, ns = _switch(paulis[j], st, half)
                cand = (c + sw, order + (j,))
                tgt = dp[mask | (1 << j)]
                old = tgt.get(ns)
                if old is None or cand < old:
                    tgt[ns] = cand
    final = dp[(1 << m) - 1]
    best_cost = min(c for c, _ in final.values())
    cands = [(_lookahead(st, nxt, block), order)
             for st, (c, order) in final.items() if c == best_cost]
    return list(min(cands)[1])


def _two_opt(paulis: list[str], order: list[int], state: str, block: DataBlock,
             max_rounds: int = 50) -> list[int]:
    best = list(order)
    best_c = _cost_idx(paulis, best, state, block)[0]
    m = len(best)
    for _ in range(max_rounds):
        improved = False
        for i in range(m - 1):
            for j in range(i + 1, m):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = _cost_idx(paulis, cand, state, block)[0]
                if c < best_c:
                    best, best_c, improved = cand, c, True
        if not improved:
            break
    return best


def order_layer(layer: Sequence[Any], state: str | None, strategy: str,
                block: DataBlock | None = None,
                next_layer: Sequence[Any] | None = None) -> list:
    """Reorder one layer of mutually commuting pi/8 rotations.

    Items may be PauliProduct, compile.Rot or str; the same objects are
    returned (a permutation of ``layer``).

    * 'naive'  : given order.
    * 'greedy' : nearest neighbour on switching cost from the entering state,
      ties broken by original index.
    * 'optimal': minimises the layer's total cost from the entering state.
      Exact (DP over subsets x orientation states) for <= EXACT_LIMIT
      rotations; among cost-optimal orders, the final state with the lowest
      greedy switching cost of ``next_layer`` is preferred, then the
      lexicographically smallest index order. Larger layers: 2-opt local
      search seeded with the better of greedy and naive (strict improvements
      only), so optimal <= min(greedy, naive) always.

    Only the compact block is state-dependent; other blocks return the
    given order for every strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    items = list(layer)
    if not items:
        return []
    paulis = [_ops(x) for x in items]
    n = len(paulis[0])
    block = block or compact(n)
    if strategy == "naive" or block.name != "compact" or len(items) == 1:
        return items
    if state is None:
        state = initial_state(n)
    if strategy == "greedy":
        idx = _greedy_idx(paulis, state, block)
    else:
        nxt = [_ops(x) for x in next_layer] if next_layer else None
        if len(items) <= EXACT_LIMIT:
            idx = _exact_idx(paulis, state, block, nxt)
        else:
            g = _greedy_idx(paulis, state, block)
            nv = list(range(len(items)))
            seed = min((g, nv), key=lambda o: _cost_idx(paulis, o, state, block)[0])
            idx = _two_opt(paulis, seed, state, block)
    return [items[j] for j in idx]


# ---------------------------------------------------------------------------
# pipeline simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceEntry:
    index: int          # 1-based magic-state / rotation index
    layer: int          # 0-based layer index
    pauli: str
    ready: float
    start: float
    end: float
    cost: int


@dataclass
class ScheduleResult:
    makespan: float
    trace: list[TraceEntry]
    idle: float            # aisle waiting time after the first start
    initial_wait: float    # = interval (first state cannot be earlier)
    busy: int              # sum of rotation costs
    interval: float
    bottleneck: str        # 'distillation' or 'consumption'
    final_state: str | None
    strategy: str = ""
    ordered_layers: list = field(default_factory=list)

    @property
    def T(self) -> int:
        return len(self.trace)


def _check_layers(layers: Sequence[Sequence[Any]]) -> None:
    for li, layer in enumerate(layers):
        pps = [_pp(x) for x in layer]
        for a, b in itertools.combinations(pps, 2):
            if not commutes(a, b):
                raise ValueError(
                    f"layer {li}: rotations {a.ops} and {b.ops} anticommute; "
                    "reordering inside a layer requires mutual commutation")


def _interval(k_distill: int, distill: DistillationBlock | None) -> float:
    if k_distill < 1:
        raise ValueError("need at least one distillation block")
    d = distill or DistillationBlock()
    return d.time / k_distill


def simulate(ordered_layers: Sequence[Sequence[Any]], block: DataBlock,
             k_distill: int, distill: DistillationBlock | None = None,
             state: str | None = None, strategy: str = "") -> ScheduleResult:
    """Run the serial pipeline on already-ordered layers (no reordering).

    bottleneck (critical path): 'distillation' if the LAST rotation starts
    exactly when its magic state is ready (the makespan is then
    T*interval + last cost, so consumption savings earlier on are invisible),
    else 'consumption' (the aisle is still busy when the last state arrives).
    """
    interval = _interval(k_distill, distill)
    if block.name == "compact" and state is None:
        state = initial_state(block.n)
    trace = []
    end = 0.0
    idle = 0.0
    busy = 0
    i = 0
    for li, layer in enumerate(ordered_layers):
        for r in layer:
            i += 1
            ready = i * interval
            start = max(end, ready)
            if i > 1:
                idle += start - end
            c, state = rotation_cost(r, state, block)
            end = start + c
            busy += c
            trace.append(TraceEntry(i, li, _ops(r), ready, start, end, c))
    T = len(trace)
    return ScheduleResult(
        makespan=end, trace=trace, idle=idle,
        initial_wait=interval if T else 0.0, busy=busy, interval=interval,
        bottleneck=("distillation" if T and trace[-1].start <= trace[-1].ready + 1e-9
                    else "consumption"),
        final_state=state, strategy=strategy,
        ordered_layers=[list(l) for l in ordered_layers])


def schedule(layers: Sequence[Sequence[Any]], block: DataBlock, k_distill: int,
             strategy: str = "optimal", distill: DistillationBlock | None = None,
             state: str | None = None) -> ScheduleResult:
    """Order each layer (only within layers: their rotations mutually commute,
    checked with gosc.pauli.commutes -> ValueError otherwise) and simulate the
    pipeline. Layers are processed in the given order; the state entering
    layer l is the state left by layer l-1. 'optimal' uses layer l+1 as
    lookahead for tie-breaking. Per-layer optimality is myopic: it does not
    guarantee the globally minimal makespan.
    """
    _check_layers(layers)
    if block.name == "compact" and state is None:
        state = initial_state(block.n)
    cur = state
    ordered = []
    for li, layer in enumerate(layers):
        nxt = layers[li + 1] if li + 1 < len(layers) else None
        o = order_layer(layer, cur, strategy, block, nxt)
        _, cur = sequence_cost(o, cur, block)
        ordered.append(o)
    return simulate(ordered, block, k_distill, distill, state, strategy)


# ---------------------------------------------------------------------------
# lower bounds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LowerBound:
    loose: float  # max(T * 1, T*interval + 1)
    tight: float  # max(T*interval + min last cost, interval + sum of min costs)


def _min_cost(p: str, block: DataBlock) -> int:
    """Smallest conceivable cost of one rotation (any state)."""
    if block.name == "fast":
        return 1
    if block.name == "intermediate":
        return consume_time(block, p)
    return 1 + pi4_cost(p)


def lower_bound(layers: Sequence[Sequence[Any]], block: DataBlock, k_distill: int,
                distill: DistillationBlock | None = None) -> LowerBound:
    """Makespan lower bounds valid for every strategy.

    * loose = max(T, T*interval + 1): every rotation costs >= 1 and the last
      state is ready at T*interval.
    * tight = max(T*interval + min cost in last layer,
                  interval + sum of per-rotation minimum costs), where the
      minimum cost is 1+pi4_cost (compact), consume_time (intermediate),
      1 (fast). The first state is ready at ``interval`` and consumption is
      serial afterwards. tight >= max(T*interval+1, sum(min costs)).
    """
    interval = _interval(k_distill, distill)
    flat = [_ops(r) for l in layers for r in l]
    T = len(flat)
    if T == 0:
        return LowerBound(0.0, 0.0)
    last = [l for l in layers if l][-1]
    loose = max(float(T), T * interval + 1)
    tight = max(T * interval + min(_min_cost(_ops(r), block) for r in last),
                interval + sum(_min_cost(p, block) for p in flat))
    return LowerBound(loose, max(loose, tight))
