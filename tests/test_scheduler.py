import random
from collections import Counter

import pytest

from gosc.blocks import compact, consume_time, fast, intermediate, pi4_cost
from gosc.pauli import PauliProduct, commutes
from gosc.scheduler import (lower_bound, order_layer, rotation_cost, schedule,
                            sequence_cost)

STRATS = ("naive", "greedy", "optimal")
# n=4: row0 = {0,1}, row1 = {2,3}
ALT = ["XXII", "ZZII", "IIXX", "IIZZ", "XXXX", "ZZZZ"]


def random_layer(rng, n, m, density=0.6):
    """Mutually commuting Pauli strings by rejection sampling."""
    layer = []
    while len(layer) < m:
        ops = "".join(rng.choice("XYZ") if rng.random() < density else "I"
                      for _ in range(n))
        p = PauliProduct(ops)
        if p.weight() and all(commutes(p, q) for q in layer):
            layer.append(p)
    return layer


def random_state(rng, n):
    return "".join(rng.choice("XZ") for _ in range(n))


# --------------------------------------------------------------- cost model

def test_fig11_cost_and_persistence():
    b = compact(6)
    c, st = rotation_cost("YIYZYY", None, b)
    assert c == 9 == consume_time(b, "YIYZYY")
    assert st == "XZXZXX"
    c2, st2 = rotation_cost(PauliProduct("YIYZYY", -1), st, b)
    assert c2 == 1 + pi4_cost("YIYZYY") == 3 and st2 == st


def test_from_all_z_matches_blocks_consume_time():
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(1, 12)
        p = "".join(rng.choice("IXYZ") for _ in range(n))
        b = compact(n)
        assert rotation_cost(p, None, b)[0] == consume_time(b, p)


def test_switch_rules():
    b = compact(4)
    # only the qubits that must change are flipped; I keeps orientation
    assert rotation_cost("XIII", "ZZZZ", b) == (4, "XZZZ")
    assert rotation_cost("ZIII", "XXXX", b) == (4, "ZXXX")
    assert rotation_cost("IIII", "XZXZ", b) == (1, "XZXZ")
    # one row flips together: cost 3, both rows: 6
    assert rotation_cost("XXII", "ZZZZ", b)[0] == 4
    assert rotation_cost("XIXI", "ZZZZ", b)[0] == 7
    # Y needs X orientation; odd #Y -> +1
    assert rotation_cost("YIII", "XZZZ", b) == (2, "XZZZ")
    # worst case 9 reachable from a non-initial state too
    assert rotation_cost("ZZYY", "XXZZ", b)[0] == 1 + 2 + 6


def test_stateless_blocks():
    for b in (intermediate(5), fast(5)):
        for p in ("XYZIZ", "ZZZZZ", "IIIII"):
            c, st = rotation_cost(p, "anything", b)
            assert c == (1 if b.name == "fast" else consume_time(b, p))
            assert st == "anything"


# --------------------------------------------------------------- ordering

def test_hand_built_layer_exact_numbers():
    b = compact(4)
    assert sequence_cost(order_layer(ALT, None, "naive", b), None, b)[0] == 30
    assert sequence_cost(order_layer(ALT, None, "greedy", b), None, b)[0] == 12
    assert sequence_cost(order_layer(ALT, None, "optimal", b), None, b)[0] == 12
    four = ALT[:4]
    assert sequence_cost(four, None, b)[0] == 16
    assert sequence_cost(order_layer(four, None, "optimal", b), None, b)[0] == 10


def test_random_layers_strategy_ordering():
    rng = random.Random(2024)
    tot = Counter()
    for _ in range(250):
        n = rng.randint(4, 12)
        b = compact(n)
        layer = random_layer(rng, n, rng.randint(2, 8))
        st = random_state(rng, n)
        costs = {}
        for s in STRATS:
            o = order_layer(layer, st, s, b)
            assert Counter(x.ops for x in o) == Counter(x.ops for x in layer)
            costs[s] = sequence_cost(o, st, b)[0]
            tot[s] += costs[s]
        assert costs["optimal"] <= costs["greedy"]
        assert costs["optimal"] <= costs["naive"]
    # nearest-neighbour greedy is NOT guaranteed <= naive per instance
    # (rare counterexamples exist), but it is better on aggregate.
    assert tot["greedy"] < tot["naive"]
    assert tot["optimal"] <= tot["greedy"]


def test_optimal_matches_brute_force_small():
    import itertools
    rng = random.Random(5)
    for _ in range(40):
        n = rng.randint(3, 8)
        b = compact(n)
        layer = random_layer(rng, n, rng.randint(2, 6))
        st = random_state(rng, n)
        brute = min(sequence_cost(p, st, b)[0] for p in itertools.permutations(layer))
        assert sequence_cost(order_layer(layer, st, "optimal", b), st, b)[0] == brute


def test_large_layer_two_opt_not_worse():
    rng = random.Random(11)
    n = 12
    b = compact(n)
    layer = random_layer(rng, n, 12, density=0.4)
    st = "Z" * n
    co = sequence_cost(order_layer(layer, st, "optimal", b), st, b)[0]
    assert co <= sequence_cost(order_layer(layer, st, "greedy", b), st, b)[0]
    assert co <= sequence_cost(layer, st, b)[0]


def test_schedule_rejects_anticommuting_layer():
    with pytest.raises(ValueError):
        schedule([["XIII", "ZIII"]], compact(4), 1)


def test_schedule_preserves_multiset_and_layers():
    rng = random.Random(3)
    n = 8
    layers = [random_layer(rng, n, rng.randint(2, 6)) for _ in range(5)]
    for s in STRATS:
        r = schedule(layers, compact(n), 4, s)
        for l_in, l_out in zip(layers, r.ordered_layers):
            assert Counter(x.ops for x in l_in) == Counter(x.ops for x in l_out)
        assert [t.layer for t in r.trace] == sorted(t.layer for t in r.trace)


# --------------------------------------------------------------- pipeline

def test_distillation_bound_all_equal():
    b = compact(4)
    # final rotation's cost is order-independent: ZZZZ forces all-Z first
    layers = [ALT, ["ZZZZ"], ["XXXX"]]
    T = 8
    res = {s: schedule(layers, b, 1, s) for s in STRATS}
    for r in res.values():
        assert r.interval == 11.0
        assert r.bottleneck == "distillation"
        assert all(t.start == t.ready for t in r.trace)  # never waits on consumption
        assert r.makespan == T * 11 + 7 == T * 11 + r.trace[-1].cost
        assert r.makespan >= lower_bound(layers, b, 1).tight
    assert len({r.makespan for r in res.values()}) == 1


def test_consumption_bound_optimal_beats_naive():
    b = compact(4)
    layers = [ALT]
    rn = schedule(layers, b, 16, "naive")
    ro = schedule(layers, b, 16, "optimal")
    iv = 11 / 16
    assert rn.bottleneck == ro.bottleneck == "consumption"
    assert rn.makespan == pytest.approx(iv + 30)
    assert ro.makespan == pytest.approx(iv + 12)
    assert ro.idle == pytest.approx(0.0)


def test_makespan_at_least_lower_bound():
    rng = random.Random(99)
    for _ in range(30):
        n = rng.randint(4, 10)
        layers = [random_layer(rng, n, rng.randint(1, 6)) for _ in range(rng.randint(1, 5))]
        for ctor in (compact, intermediate, fast):
            b = ctor(n)
            for k in (1, 2, 4, 8, 16):
                lb = lower_bound(layers, b, k)
                assert lb.tight >= lb.loose
                for s in STRATS:
                    r = schedule(layers, b, k, s)
                    assert r.makespan >= lb.tight - 1e-9
                    assert r.makespan == pytest.approx(r.initial_wait + r.idle + r.busy)
