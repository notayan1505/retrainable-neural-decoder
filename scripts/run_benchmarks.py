"""Benchmark naive / greedy / optimal magic-state consumption ordering.

Writes results/benchmarks.csv and results/benchmarks.md.

(1) synthetic: random commuting layers (n = 8..20, layer sizes 4..9) on
    compact(n), k in {1,2,4,8,16} distillation blocks;
(2) compiled circuits via gosc.compile (if available): Toffoli (7-T
    decomposition), a small Toffoli/CNOT adder-like circuit, random Clifford+T.
"""
from __future__ import annotations

import csv
import os
import random
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gosc.blocks import compact  # noqa: E402
from gosc.pauli import PauliProduct, commutes  # noqa: E402
from gosc.scheduler import (lower_bound, order_layer, sequence_cost,  # noqa: E402
                            simulate)

KS = (1, 2, 4, 8, 16)
STRATS = ("naive", "greedy", "optimal")
NS = (8, 12, 16, 20)
SEEDS = 5
N_LAYERS = 10
COMPILED_INFO: list = []


def random_layer(rng, n, m, density=0.5):
    layer = []
    while len(layer) < m:
        ops = "".join(rng.choice("XYZ") if rng.random() < density else "I"
                      for _ in range(n))
        p = PauliProduct(ops)
        if p.weight() and all(commutes(p, q) for q in layer):
            layer.append(p)
    return layer


def ordered(layers, block, strategy):
    """Order every layer once (orders do not depend on k)."""
    st = "Z" * block.n
    out = []
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        o = order_layer(layer, st, strategy, block, nxt)
        _, st = sequence_cost(o, st, block)
        out.append(o)
    return out


def bench(name, layers, block, rows):
    from gosc.scheduler import _check_layers  # precondition check
    _check_layers(layers)
    orders = {s: ordered(layers, block, s) for s in STRATS}
    T = sum(len(l) for l in layers)
    for k in KS:
        lb = lower_bound(layers, block, k)
        res = {s: simulate(orders[s], block, k, strategy=s) for s in STRATS}
        for s, r in res.items():
            rows.append(dict(
                instance=name, n=block.n, T=T, layers=len(layers), k=k,
                interval=round(r.interval, 4), strategy=s,
                makespan=round(r.makespan, 4), busy=r.busy,
                idle=round(r.idle, 4), bottleneck=r.bottleneck,
                lb_tight=round(lb.tight, 4),
                speedup_vs_naive=round(res["naive"].makespan / r.makespan, 4)))


def synthetic(rows):
    for n in NS:
        for seed in range(SEEDS):
            rng = random.Random(1000 * n + seed)
            layers = [random_layer(rng, n, rng.randint(4, 9)) for _ in range(N_LAYERS)]
            bench(f"synthetic_n{n}_s{seed}", layers, compact(n), rows)


def toffoli(a, b, c):
    return [("h", (c,)), ("cx", (b, c)), ("tdg", (c,)), ("cx", (a, c)), ("t", (c,)),
            ("cx", (b, c)), ("tdg", (c,)), ("cx", (a, c)), ("t", (b,)), ("t", (c,)),
            ("h", (c,)), ("cx", (a, b)), ("t", (a,)), ("tdg", (b,)), ("cx", (a, b))]


def circuits():
    # adder-like (Cuccaro-style carry/sum pattern): 5 qubits c0,a0,b0,a1,b1(+carry)
    adder = (toffoli(0, 1, 2) + [("cx", (0, 1)), ("cx", (1, 2))]
             + toffoli(2, 3, 4) + [("cx", (2, 3)), ("cx", (3, 4))]
             + [("cx", (0, 4))])
    rng = random.Random(42)
    rnd = []
    for _ in range(80):
        if rng.random() < 0.3:
            a, b = rng.sample(range(6), 2)
            rnd.append(("cx", (a, b)))
        else:
            rnd.append((rng.choice(["h", "s", "t", "t", "tdg"]), (rng.randrange(6),)))
    return [("toffoli_3q", toffoli(0, 1, 2), 3), ("adder_5q", adder, 5),
            ("random_clifford_t_6q", rnd, 6)]


def compiled(rows):
    try:
        import gosc.compile as gc
    except Exception as e:  # pragma: no cover
        print(f"[skip] gosc.compile unavailable: {e}")
        return False
    if not (hasattr(gc, "to_pauli_rotations") and hasattr(gc, "t_layers")):
        print("[skip] gosc.compile lacks to_pauli_rotations / t_layers")
        return False
    for name, circ, n in circuits():
        cc = gc.to_pauli_rotations(circ, n)
        layers = [[r.axis for r in l] for l in gc.t_layers(cc.t_rotations)]
        xs = sum(1 for l in layers for p in l if any(c in "XY" for c in p.ops))
        print(f"{name}: n={n} T={cc.t_count} layers={len(layers)} "
              f"sizes={[len(l) for l in layers]} rotations_with_X_or_Y={xs}/{cc.t_count}")
        COMPILED_INFO.append(f"{name}: n={n}, T={cc.t_count}, layer sizes "
                             f"{[len(l) for l in layers]}, rotations with an X/Y "
                             f"component {xs}/{cc.t_count}")
        bench(name, layers, compact(n), rows)
    return True


def summarize(rows, have_compiled):
    agg = defaultdict(list)
    for r in rows:
        kind = "synthetic" if r["instance"].startswith("synthetic") else r["instance"]
        key = (kind, r["n"], r["k"], r["strategy"])
        agg[key].append(r)
    lines = ["# Magic-state consumption ordering benchmark", "",
             "Compact data block, state-dependent patch-orientation cost model "
             "(gosc/scheduler.py). Speedup = naive makespan / strategy makespan "
             "(geometric mean over seeds for synthetic). Makespan in time steps.",
             "Distillation interval = 11/k. Synthetic: 10 layers per instance, "
             "layer sizes 4..9, 5 seeds per n.", ""]

    def gmean(x):
        return statistics.geometric_mean(x)

    lines += ["## Synthetic: speedup naive/optimal (naive/greedy in parentheses)", "",
              "| n | " + " | ".join(f"k={k}" for k in KS) + " |",
              "|---|" + "---|" * len(KS)]
    for n in NS:
        cells = []
        for k in KS:
            o = gmean([r["speedup_vs_naive"] for r in agg[("synthetic", n, k, "optimal")]])
            g = gmean([r["speedup_vs_naive"] for r in agg[("synthetic", n, k, "greedy")]])
            cells.append(f"{o:.3f} ({g:.3f})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Synthetic: mean makespan / tight lower bound, and bottleneck (naive -> optimal)", "",
              "| n | k | naive | greedy | optimal | LB | bottleneck naive -> optimal |",
              "|---|---|---|---|---|---|---|"]
    for n in NS:
        for k in KS:
            m = {s: statistics.mean(r["makespan"] for r in agg[("synthetic", n, k, s)])
                 for s in STRATS}
            lb = statistics.mean(r["lb_tight"] for r in agg[("synthetic", n, k, "optimal")])
            bn = Counter_str(agg[("synthetic", n, k, "naive")])
            bo = Counter_str(agg[("synthetic", n, k, "optimal")])
            lines.append(f"| {n} | {k} | {m['naive']:.1f} | {m['greedy']:.1f} | "
                         f"{m['optimal']:.1f} | {lb:.1f} | {bn} -> {bo} |")
    # greedy worse than naive / optimal worse than greedy (multi-layer, makespan)
    worse_g = sum(1 for r in rows if r["strategy"] == "greedy" and r["speedup_vs_naive"] < 1 - 1e-9)
    worse_o = sum(1 for r in rows if r["strategy"] == "optimal" and r["speedup_vs_naive"] < 1 - 1e-9)
    ng = sum(1 for r in rows if r["strategy"] == "greedy")
    by = {(r["instance"], r["k"], r["strategy"]): r["makespan"] for r in rows}
    o_gt_g = sum(1 for (i, k, s), v in by.items()
                 if s == "optimal" and v > by[(i, k, "greedy")] + 1e-9)
    lines += ["", f"Makespan regressions over all (instance, k) cases ({ng} each): "
              f"greedy > naive in {worse_g}, optimal > naive in {worse_o}, "
              f"optimal > greedy in {o_gt_g}.", ""]
    lines += [single_layer_check(), ""]
    if have_compiled:
        lines += ["## Compiled circuits (gosc.compile, t_layers greedy layering)", ""]
        lines += [f"* {x}" for x in COMPILED_INFO]
        lines += ["* layers with > 9 rotations use greedy/naive-seeded 2-opt, not the exact DP.", "",
                  "| circuit | n | T | layers | k | naive | greedy | optimal | LB | busy naive/opt | speedup naive/opt | bottleneck naive -> opt |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        names = []
        for r in rows:
            if not r["instance"].startswith("synthetic") and r["instance"] not in names:
                names.append(r["instance"])
        for name in names:
            for k in KS:
                rr = {r["strategy"]: r for r in rows if r["instance"] == name and r["k"] == k}
                o = rr["optimal"]
                lines.append(f"| {name} | {o['n']} | {o['T']} | {o['layers']} | {k} | "
                             f"{rr['naive']['makespan']:.2f} | {rr['greedy']['makespan']:.2f} | "
                             f"{o['makespan']:.2f} | {o['lb_tight']:.2f} | "
                             f"{rr['naive']['busy']}/{o['busy']} | "
                             f"{o['speedup_vs_naive']:.3f} | "
                             f"{rr['naive']['bottleneck']} -> {o['bottleneck']} |")
    else:
        lines += ["Compiled-circuit benchmark skipped (gosc.compile not available)."]
    lines += ["", "## When reordering helps", "",
              "* Distillation-bound (k=1, interval 11 > max consume 9): every rotation "
              "starts when its magic state arrives, makespan = T*11 + cost of the last "
              "rotation, so reordering can save at most 8 steps in total -- speedup ~1.0.",
              "* Consumption-bound (large k, interval << cost): makespan ~ interval + sum of "
              "costs, so every saved orientation switch is saved wall-clock time; this is "
              "where optimal/greedy ordering pays off.",
              "* In between (k=2..4) the gain depends on whether the naive cost per rotation "
              "exceeds the interval; reordering can move a run from consumption- to "
              "distillation-bound, after which further savings are invisible.", ""]
    return "\n".join(lines)


def single_layer_check(n_trials=2000, seed=0):
    """Per-instance single-layer cost comparison from random entering states."""
    rng = random.Random(seed)
    g_worse = o_worse = 0
    tot = {s: 0 for s in STRATS}
    for _ in range(n_trials):
        n = rng.randint(4, 10)
        b = compact(n)
        layer = random_layer(rng, n, rng.randint(3, 7), density=0.6)
        st = "".join(rng.choice("XZ") for _ in range(n))
        c = {s: sequence_cost(order_layer(layer, st, s, b), st, b)[0] for s in STRATS}
        for s in STRATS:
            tot[s] += c[s]
        g_worse += c["greedy"] > c["naive"]
        o_worse += c["optimal"] > min(c["greedy"], c["naive"])
    return (f"Single-layer cost check ({n_trials} random layers, n=4..10, 3..7 rotations, "
            f"random entering state, seed {seed}): greedy > naive in {g_worse} "
            f"({100 * g_worse / n_trials:.2f}%), optimal > min(greedy, naive) in {o_worse}; "
            f"total cost naive {tot['naive']}, greedy {tot['greedy']}, optimal {tot['optimal']}.")


def Counter_str(rs):
    d = sum(1 for r in rs if r["bottleneck"] == "distillation")
    return f"{d}/{len(rs)} dist"


def main():
    rows = []
    synthetic(rows)
    have = compiled(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "benchmarks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = summarize(rows, have)
    with open(os.path.join(ROOT, "results", "benchmarks.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
