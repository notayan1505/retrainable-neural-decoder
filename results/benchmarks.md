# Magic-state consumption ordering benchmark

Compact data block, state-dependent patch-orientation cost model (gosc/scheduler.py). Speedup = naive makespan / strategy makespan (geometric mean over seeds for synthetic). Makespan in time steps.
Distillation interval = 11/k. Synthetic: 10 layers per instance, layer sizes 4..9, 5 seeds per n.

## Synthetic: speedup naive/optimal (naive/greedy in parentheses)

| n | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| 8 | 1.002 (1.000) | 1.040 (1.029) | 1.294 (1.164) | 1.296 (1.167) | 1.297 (1.167) |
| 12 | 0.999 (0.995) | 1.216 (1.108) | 1.259 (1.120) | 1.261 (1.121) | 1.262 (1.122) |
| 16 | 1.001 (0.999) | 1.222 (1.113) | 1.226 (1.115) | 1.227 (1.115) | 1.228 (1.116) |
| 20 | 1.001 (1.000) | 1.172 (1.057) | 1.173 (1.058) | 1.174 (1.058) | 1.174 (1.058) |

## Synthetic: mean makespan / tight lower bound, and bottleneck (naive -> optimal)

| n | k | naive | greedy | optimal | LB | bottleneck naive -> optimal |
|---|---|---|---|---|---|---|
| 8 | 1 | 734.0 | 734.0 | 732.8 | 729.2 | 5/5 dist -> 5/5 dist |
| 8 | 2 | 383.5 | 372.6 | 368.7 | 365.1 | 2/5 dist -> 5/5 dist |
| 8 | 4 | 358.3 | 307.9 | 276.8 | 183.1 | 0/5 dist -> 0/5 dist |
| 8 | 8 | 356.6 | 305.7 | 275.1 | 131.6 | 0/5 dist -> 0/5 dist |
| 8 | 16 | 355.9 | 304.9 | 274.3 | 130.9 | 0/5 dist -> 0/5 dist |
| 12 | 1 | 699.4 | 703.4 | 700.6 | 696.6 | 5/5 dist -> 5/5 dist |
| 12 | 2 | 438.2 | 396.6 | 361.0 | 349.0 | 0/5 dist -> 1/5 dist |
| 12 | 4 | 435.4 | 389.5 | 345.7 | 175.2 | 0/5 dist -> 0/5 dist |
| 12 | 8 | 434.0 | 387.9 | 344.1 | 144.8 | 0/5 dist -> 0/5 dist |
| 12 | 16 | 433.3 | 387.1 | 343.3 | 144.1 | 0/5 dist -> 0/5 dist |
| 16 | 1 | 673.8 | 674.6 | 673.0 | 668.6 | 5/5 dist -> 5/5 dist |
| 16 | 2 | 466.5 | 418.8 | 381.6 | 335.3 | 0/5 dist -> 0/5 dist |
| 16 | 4 | 463.8 | 415.8 | 377.9 | 168.7 | 0/5 dist -> 0/5 dist |
| 16 | 8 | 462.4 | 414.4 | 376.6 | 146.8 | 0/5 dist -> 0/5 dist |
| 16 | 16 | 461.7 | 413.7 | 375.9 | 146.1 | 0/5 dist -> 0/5 dist |
| 20 | 1 | 701.8 | 701.8 | 701.0 | 695.0 | 5/5 dist -> 5/5 dist |
| 20 | 2 | 508.5 | 480.9 | 433.5 | 348.5 | 0/5 dist -> 0/5 dist |
| 20 | 4 | 505.8 | 478.1 | 430.8 | 175.2 | 0/5 dist -> 0/5 dist |
| 20 | 8 | 504.4 | 476.8 | 429.4 | 152.8 | 0/5 dist -> 0/5 dist |
| 20 | 16 | 503.7 | 476.1 | 428.7 | 152.1 | 0/5 dist -> 0/5 dist |

Makespan regressions over all (instance, k) cases (115 each): greedy > naive in 10, optimal > naive in 4, optimal > greedy in 0.

Single-layer cost check (2000 random layers, n=4..10, 3..7 rotations, random entering state, seed 0): greedy > naive in 55 (2.75%), optimal > min(greedy, naive) in 0; total cost naive 58351, greedy 51100, optimal 47356.

## Compiled circuits (gosc.compile, t_layers greedy layering)

* toffoli_3q: n=3, T=7, layer sizes [7], rotations with an X/Y component 4/7
* adder_5q: n=5, T=14, layer sizes [10, 4], rotations with an X/Y component 8/14
* random_clifford_t_6q: n=6, T=31, layer sizes [8, 10, 9, 4], rotations with an X/Y component 24/31
* layers with > 9 rotations use greedy/naive-seeded 2-opt, not the exact DP.

| circuit | n | T | layers | k | naive | greedy | optimal | LB | busy naive/opt | speedup naive/opt | bottleneck naive -> opt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toffoli_3q | 3 | 7 | 1 | 1 | 78.00 | 78.00 | 78.00 | 78.00 | 10/10 | 1.000 | distillation -> distillation |
| toffoli_3q | 3 | 7 | 1 | 2 | 39.50 | 39.50 | 39.50 | 39.50 | 10/10 | 1.000 | distillation -> distillation |
| toffoli_3q | 3 | 7 | 1 | 4 | 20.25 | 20.25 | 20.25 | 20.25 | 10/10 | 1.000 | distillation -> distillation |
| toffoli_3q | 3 | 7 | 1 | 8 | 11.38 | 12.50 | 11.38 | 10.62 | 10/10 | 1.000 | consumption -> consumption |
| toffoli_3q | 3 | 7 | 1 | 16 | 10.69 | 10.69 | 10.69 | 7.69 | 10/10 | 1.000 | consumption -> consumption |
| adder_5q | 5 | 14 | 2 | 1 | 155.00 | 155.00 | 155.00 | 155.00 | 23/23 | 1.000 | distillation -> distillation |
| adder_5q | 5 | 14 | 2 | 2 | 78.00 | 78.00 | 78.00 | 78.00 | 23/23 | 1.000 | distillation -> distillation |
| adder_5q | 5 | 14 | 2 | 4 | 39.50 | 39.50 | 39.50 | 39.50 | 23/23 | 1.000 | distillation -> distillation |
| adder_5q | 5 | 14 | 2 | 8 | 24.38 | 25.88 | 25.88 | 20.25 | 23/23 | 0.942 | consumption -> consumption |
| adder_5q | 5 | 14 | 2 | 16 | 23.69 | 23.69 | 23.69 | 14.69 | 23/23 | 1.000 | consumption -> consumption |
| random_clifford_t_6q | 6 | 31 | 4 | 1 | 349.00 | 349.00 | 349.00 | 342.00 | 104/77 | 1.000 | distillation -> distillation |
| random_clifford_t_6q | 6 | 31 | 4 | 2 | 187.50 | 184.50 | 182.00 | 171.50 | 104/77 | 1.030 | consumption -> consumption |
| random_clifford_t_6q | 6 | 31 | 4 | 4 | 123.25 | 111.00 | 102.00 | 86.25 | 104/77 | 1.208 | consumption -> consumption |
| random_clifford_t_6q | 6 | 31 | 4 | 8 | 107.25 | 93.00 | 80.25 | 43.62 | 104/77 | 1.336 | consumption -> consumption |
| random_clifford_t_6q | 6 | 31 | 4 | 16 | 104.69 | 89.69 | 77.69 | 35.69 | 104/77 | 1.347 | consumption -> consumption |

## When reordering helps

* Distillation-bound (k=1, interval 11 > max consume 9): every rotation starts when its magic state arrives, makespan = T*11 + cost of the last rotation, so reordering can save at most 8 steps in total -- speedup ~1.0.
* Consumption-bound (large k, interval << cost): makespan ~ interval + sum of costs, so every saved orientation switch is saved wall-clock time; this is where optimal/greedy ordering pays off.
* In between (k=2..4) the gain depends on whether the naive cost per rotation exceeds the interval; reordering can move a run from consumption- to distillation-bound, after which further savings are invisible.
