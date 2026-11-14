import itertools
import math

import pytest

from gosc.blocks import (
    compact, intermediate, fast, consume_time, pi4_cost, fig11_example,
)


@pytest.mark.parametrize("n,c,i,f", [
    (4, 9, 12, 2 * 4 + math.ceil(math.sqrt(32)) + 1),
    (6, 12, 16, 12 + math.ceil(math.sqrt(48)) + 1),
    (10, 18, 24, 20 + math.ceil(math.sqrt(80)) + 1),
    (100, 153, 204, 200 + 29 + 1),
])
def test_tile_formulas(n, c, i, f):
    assert compact(n).tiles == c
    assert intermediate(n).tiles == i
    assert fast(n).tiles == f


def test_fast_100_value():
    assert fast(100).tiles == 230


def test_odd_compact_rounds_up():
    assert compact(5).tiles == 8 + 3  # ceil(7.5) + 3


def test_max_consume():
    assert (compact(4).max_consume, intermediate(4).max_consume, fast(4).max_consume) == (9, 5, 1)


def test_pi4_rule():
    assert pi4_cost("ZZII") == 0
    assert pi4_cost("YZII") == 1
    assert pi4_cost("YYYI") == 1
    assert pi4_cost("YYII") == 2
    assert pi4_cost("YYYY") == 2


def test_compact_rules():
    b = compact(4)  # rows: [0,1] and [2,3]
    assert consume_time(b, "ZZZZ") == 1
    assert consume_time(b, "XZII") == 4       # X in one row
    assert consume_time(b, "XIIX") == 7       # X in both rows
    assert consume_time(b, "YZII") == 1 + 1 + 3
    assert consume_time(b, "YIIY") == 9       # worst case


def test_compact_worst_case_exactly_9():
    b = compact(4)
    vals = [consume_time(b, "".join(p)) for p in itertools.product("IXYZ", repeat=4)]
    assert max(vals) == 9


def test_intermediate_rules_and_worst_case():
    b = intermediate(4)
    assert consume_time(b, "ZZZZ") == 1
    assert consume_time(b, "XZII") == 3
    assert consume_time(b, "YZII") == 4
    assert consume_time(b, "YYZI") == 5
    vals = [consume_time(b, "".join(p)) for p in itertools.product("IXYZ", repeat=4)]
    assert max(vals) == 5


def test_fast_always_one():
    b = fast(4)
    assert all(consume_time(b, "".join(p)) == 1 for p in itertools.product("IXYZ", repeat=4))


def test_monotonic_exhaustive_n4():
    c, i, f = compact(4), intermediate(4), fast(4)
    for p in itertools.product("IXYZ", repeat=4):
        s = "".join(p)
        assert consume_time(f, s) <= consume_time(i, s) <= consume_time(c, s), s


def test_fig11_example():
    block, p = fig11_example()
    assert p == "YIYZYY" and block.n == 6 and block.name == "compact"
    assert consume_time(block, p) == 9


def test_bad_input():
    with pytest.raises(ValueError):
        consume_time(compact(4), "XYZ")
    with pytest.raises(ValueError):
        consume_time(compact(4), "XYZQ")
