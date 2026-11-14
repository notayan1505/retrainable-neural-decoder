import pytest

from gosc.board import Board, Patch
from gosc.engine import Engine, IllegalMove
from gosc.ops import COSTS, PPM, Grow, Init, MeasureSingle, Rotate, Shrink, cost


def test_bell_pair():
    e = Engine(Board(4, 2))
    r = e.run([
        [Init("a", "+", (0, 0), "XNS"), Init("b", "+", (1, 0), "XNS")],
        [PPM((("a", "Z"), ("b", "Z")))],
    ])
    assert r.total_time == 1 and len(r.trace) == 2


def test_move_patch_three_tiles():
    e = Engine(Board(6, 1))
    r = e.run([
        [Init("p", "0", (0, 0))],
        [Grow("p", [(1, 0), (2, 0), (3, 0)])],
        [Shrink("p", [(0, 0), (1, 0), (2, 0)])],
    ])
    assert r.total_time == 1
    assert e.board.get("p").tiles == {(3, 0)}
    assert e.board.occupancy == {(3, 0): "p"}


def test_y_measure_single_patch():
    b = Board(4, 3)
    # |0> ancilla (init cost 0) pre-placed with a Z/X split on its facing side
    b.add_patch(Patch.from_tiles("a", [(1, 0), (2, 0)], {((1, 0), "S"): "Z", ((2, 0), "S"): "X"}))
    e = Engine(b)
    r = e.run([
        [Init("p", "+", (1, 1))],
        [Grow("p", (2, 1), ((((2, 1), "N"), "X"),))],  # deform: expose X and Z toward ancilla (1)
        [PPM((("p", "Y"), ("a", "Z")))],               # joint measure (1)
        [MeasureSingle("a", "X"), Shrink("p", (2, 1))],  # measure ancilla, deform back (0)
    ])
    assert r.total_time == 2


def test_four_patch_ppm_through_ancilla():
    e = Engine(Board(6, 4))
    region = frozenset({(1, 0), (1, 1), (2, 1), (3, 1)})
    r = e.run([
        [Init("y", "+", (2, 0)), Init("z", "+", (3, 0)), Init("x1", "+", (1, 2), "XNS"),
         Init("x2", "+", (4, 1))],
        [PPM((("y", "Y"), ("x1", "X"), ("z", "Z"), ("x2", "X")), region)],
    ])
    assert r.total_time == 1


def test_costs():
    assert cost(Rotate("a")) == COSTS["rotate_patch"] == 3
    assert cost(Grow("a", (1, 1))) == 1 and cost(Shrink("a", (1, 1))) == 0


def test_illegal_overlap():
    e = Engine(Board(3, 3))
    with pytest.raises(IllegalMove, match="occupied"):
        e.run([[Init("a", "0", (1, 1))], [Init("b", "0", (1, 1))]])
    with pytest.raises(IllegalMove):
        Engine(Board(2, 2)).run([[Init("a", "0", (5, 5))]])


def test_illegal_nonadjacent_ppm():
    e = Engine(Board(5, 1))
    with pytest.raises(IllegalMove, match="not adjacent"):
        e.run([[Init("a", "+", (0, 0), "XNS"), Init("b", "+", (3, 0), "XNS")], [PPM((("a", "Z"), ("b", "Z")))]])


def test_illegal_y_without_x_and_z():
    e = Engine(Board(4, 3))
    with pytest.raises(IllegalMove, match="Y requires"):
        e.run([[Init("y", "+", (1, 1)), Init("z", "+", (3, 0))],
               [PPM((("y", "Y"), ("z", "Z")), frozenset({(1, 0), (2, 0)}))]])


def test_illegal_parallel_shared_tile_and_missing_patch():
    e = Engine(Board(5, 1))
    e.run([[Init("a", "0", (0, 0)), Init("b", "0", (2, 0))]])
    with pytest.raises(IllegalMove, match="both use tile"):
        e.run([[Grow("a", (1, 0)), Grow("b", (1, 0))]])
    with pytest.raises(IllegalMove, match="missing"):
        e.run([[MeasureSingle("zz", "X")]])
