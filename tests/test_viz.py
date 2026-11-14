import numpy as np
import pytest

from gosc.board import Board
from gosc.engine import Engine
from gosc.ops import PPM, Init
from gosc.viz import draw_board, draw_timeline, draw_trace, plot_logical_error, plot_space_time


def bell():
    e = Engine(Board(4, 2))
    return e.run([[Init("a", "+", (0, 0), "XNS"), Init("b", "+", (1, 0), "XNS")],
                  [PPM((("a", "Z"), ("b", "Z")))]])


def four_patch():
    e = Engine(Board(6, 4))
    region = frozenset({(1, 0), (1, 1), (2, 1), (3, 1)})
    return e.run([
        [Init("y", "+", (2, 0)), Init("z", "+", (3, 0)), Init("x1", "+", (1, 2), "XNS"), Init("x2", "+", (4, 1))],
        [PPM((("y", "Y"), ("x1", "X"), ("z", "Z"), ("x2", "X")), region)],
    ]), region


def nonempty(p):
    assert p.exists() and p.stat().st_size > 0


def test_boards_recorded():
    r = bell()
    assert len(r.boards) == len(r.trace) + 1 and not r.boards[0].patches


def test_bell_figures(tmp_path):
    r = bell()
    draw_trace(r, tmp_path / "t.png")
    draw_timeline(r, tmp_path / "l.png")
    draw_board(r.boards[-1], title="x", edge_labels=True)
    nonempty(tmp_path / "t.png")
    nonempty(tmp_path / "l.png")


def test_four_patch_figures(tmp_path):
    r, region = four_patch()
    draw_trace(r, tmp_path / "t.png", ncols=2)
    draw_timeline(r, tmp_path / "l.png")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    draw_board(r.boards[-1], ax, highlight_tiles=region)
    fig.savefig(tmp_path / "b.png")
    for n in "tlb":
        nonempty(tmp_path / f"{n}.png")


def test_space_time(tmp_path):
    rows = [dict(name=n, k=k, tiles=10 * k, time_per_T=5 / k, hours=100 / k * f, physical_qubits=1e5 * k * f)
            for n, f in (("15-1", 1), ("20-4", 2)) for k in (1, 2, 4)]
    plot_space_time(rows, tmp_path / "s.png")
    nonempty(tmp_path / "s.png")


@pytest.mark.parametrize("fit", [True, False])
def test_logical_error(tmp_path, fit):
    ps = np.array([1e-3, 2e-3, 4e-3, 8e-3])
    plot_logical_error(ps, 30 * ps ** 2, tmp_path / "e.png", slope_fit=fit)
    nonempty(tmp_path / "e.png")
