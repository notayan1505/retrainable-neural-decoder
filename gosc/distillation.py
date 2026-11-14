"""15-to-1 distillation blocks and data-block/factory pipelines (Litinski 2019).

A 15-to-1 block occupies 11 tiles and outputs one magic state every 11 time
steps with infidelity ~35 p^3. With k blocks, a magic state is available every
11/k time steps; the time per T gate is max(magic_interval, worst-case
consumption time of the data block).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from gosc.blocks import DataBlock, compact, intermediate, fast

__all__ = [
    "DistillationBlock",
    "output_infidelity",
    "success_probability",
    "pipeline",
    "total_time_steps",
    "physical_qubits",
    "wall_clock_seconds",
    "sweep",
]


def output_infidelity(p: float) -> float:
    """Leading-order 15-to-1 output error: 35 p^3."""
    return 35 * p**3


@dataclass(frozen=True)
class DistillationBlock:
    tiles: int = 11
    time: int = 11
    protocol: str = "15-to-1"
    out_infidelity: Callable[[float], float] = field(default=output_infidelity)


def success_probability(p: float, n: int = 15) -> float:
    """Probability that none of the n input states is faulty: (1-p)^n."""
    return (1 - p) ** n


def pipeline(data_block: DataBlock, n_distill_blocks: int,
             distill: DistillationBlock | None = None) -> dict:
    """Combine a data block with n_distill_blocks distillation blocks.

    Routing tiles between blocks are not counted (this is why intermediate(100)
    + 2 blocks gives 226 tiles here vs ~222 quoted in the paper).
    """
    if n_distill_blocks < 1:
        raise ValueError("need at least one distillation block")
    d = distill or DistillationBlock()
    magic_interval = d.time / n_distill_blocks
    consume_worst = data_block.max_consume
    return {
        "total_tiles": data_block.tiles + n_distill_blocks * d.tiles,
        "magic_interval": float(magic_interval),
        "consume_worst": consume_worst,
        "time_per_T": max(float(magic_interval), float(consume_worst)),
    }


def total_time_steps(T_count: float, time_per_T: float) -> float:
    """Total time steps = T_count * time_per_T."""
    return T_count * time_per_T


def physical_qubits(tiles: int, d: int) -> int:
    """2 d^2 physical qubits per tile (data + measurement qubits)."""
    return 2 * d * d * tiles


def wall_clock_seconds(T_count: float, time_per_T: float, d: int,
                       cycle_us: float = 1.0) -> float:
    """Seconds = T_count * time_per_T * d cycles * cycle_us microseconds."""
    return T_count * time_per_T * d * cycle_us * 1e-6


def sweep(n: int, T_count: float, d: int, max_blocks: int,
          cycle_us: float = 1.0) -> list[tuple]:
    """Space-time trade-off rows (name, k, tiles, time_per_T, hours, physical_qubits)
    for compact/intermediate/fast data blocks with k = 1..max_blocks factories."""
    rows = []
    for ctor in (compact, intermediate, fast):
        block = ctor(n)
        for k in range(1, max_blocks + 1):
            pl = pipeline(block, k)
            hours = wall_clock_seconds(T_count, pl["time_per_T"], d, cycle_us) / 3600
            rows.append((block.name, k, pl["total_tiles"], pl["time_per_T"],
                         hours, physical_qubits(pl["total_tiles"], d)))
    return rows
