import pytest

from gosc.blocks import compact, intermediate, fast
from gosc.distillation import (
    DistillationBlock, output_infidelity, success_probability, pipeline,
    total_time_steps, physical_qubits, wall_clock_seconds, sweep,
)

N, D, T = 100, 13, 1e8


def test_block_defaults():
    b = DistillationBlock()
    assert (b.tiles, b.time, b.protocol) == (11, 11, "15-to-1")
    assert b.out_infidelity(1e-4) == pytest.approx(35e-12)


def test_infidelity_and_success():
    assert output_infidelity(1e-3) == pytest.approx(3.5e-8)
    assert success_probability(1e-3) == pytest.approx((1 - 1e-3) ** 15)
    assert success_probability(0.0) == 1.0


def test_minimal_setup():
    pl = pipeline(compact(N), 1)
    assert pl["total_tiles"] == 164
    assert pl["magic_interval"] == 11.0
    assert pl["consume_worst"] == 9
    assert pl["time_per_T"] == 11.0
    assert total_time_steps(T, pl["time_per_T"]) == pytest.approx(1.1e9)
    hours = wall_clock_seconds(T, pl["time_per_T"], D) / 3600
    assert 3.9 <= hours <= 4.1
    assert physical_qubits(pl["total_tiles"], D) == 55432


def test_intermediate_setup():
    # Paper quotes ~222 tiles; we get 204 + 22 = 226 (routing/layout details
    # of the paper's figure are not modelled).
    pl = pipeline(intermediate(N), 2)
    assert 220 <= pl["total_tiles"] <= 230
    assert pl["total_tiles"] == 226
    assert pl["magic_interval"] == 5.5
    assert pl["time_per_T"] == 5.5
    hours = wall_clock_seconds(T, pl["time_per_T"], D) / 3600
    assert 1.9 <= hours <= 2.1


def test_fast_saturates_with_blocks():
    pl = pipeline(fast(N), 11)
    assert pl["time_per_T"] == 1.0


def test_sweep():
    rows = sweep(N, T, D, 4)
    assert len(rows) == 12
    names = {r[0] for r in rows}
    assert names == {"compact", "intermediate", "fast"}
    first = rows[0]
    assert first[:4] == ("compact", 1, 164, 11.0)
    assert first[5] == 55432
    # more blocks never slower, always more tiles
    for name in names:
        sub = [r for r in rows if r[0] == name]
        assert all(a[3] >= b[3] and a[2] < b[2] for a, b in zip(sub, sub[1:]))


def test_pipeline_requires_block():
    with pytest.raises(ValueError):
        pipeline(compact(4), 0)
