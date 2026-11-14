# Magic-state consumption scheduler (`gosc/scheduler.py`)

## Cost model

Unit: one time step (about d code cycles), the same unit as `gosc/blocks.py`.

### Compact block (state-dependent)

- **State.** Each data qubit has an orientation `'Z'` or `'X'`: the edge type that currently faces the ancilla aisle. The initial state is all `'Z'`. The state is stored as a string of length n.
- **Rows.** Row 0 holds string indices `[0, ceil(n/2))` and row 1 holds the rest, as in `blocks.consume_time`. Indices are positions in `PauliProduct.ops`. In `gosc.compile`, circuit qubit q sits at `ops[n-1-q]`.
- **Needs** of a pi/8 rotation about P:
  - `P_i` in {X, Y} needs `'X'`. Y is reached through its X edge after the pi/4 handling, matching `blocks.x_support`.
  - `P_i = Z` needs `'Z'`.
  - `P_i = I` needs nothing.
  - The sign of P does not matter.
- **Switching cost.** Each row that contains at least one qubit whose orientation must change costs 3. Rows switch one after another, and all qubits in a row switch together, so the switching cost is 0, 3 or 6.
- **Flips.** Only the qubits that *must* change are flipped. `'I'` qubits and qubits that are already correct keep their orientation. Free pre-emptive flips are not allowed: they would let any order reach any state for nothing, and the ordering problem would collapse.
- **Persistence.** The new orientation carries over to the next rotation.
- **Cost** = `1 + pi4_cost(P) + switching cost`. `pi4_cost` comes from `blocks.py`: 0 with no Y, 1 with an odd number of Ys, 2 with an even number greater than 0. The maximum is 1 + 2 + 6 = 9.
- **Consistency check.** From the all-`'Z'` state this matches `blocks.consume_time` exactly (tested on 500 random strings). For example, `YIYZYY` on n=6 costs 9 and leaves state `XZXZXX`. Repeating the same rotation then costs only `1 + pi4_cost = 3`.

### Other blocks

- **Intermediate block:** stateless, cost = `blocks.consume_time`.
- **Fast block:** cost 1.

On these blocks ordering never changes the cost, so every strategy returns the given order.

## Pipeline

- With k 15-to-1 blocks, `interval = DistillationBlock.time / k`, which is `11/k`.
- Magic state i (i = 1..T) is ready at `i*interval`.
- One ancilla aisle consumes the states serially:
  - `start_i = max(end_{i-1}, ready_i)`
  - `end_i = start_i + cost_i`
  - `cost_i` depends on the state left by rotation i-1.
- Reported quantities:
  - `makespan` is `end_T`.
  - `busy` is the sum of the costs.
  - `initial_wait = interval` is the unavoidable wait for the first state.
  - `idle` is the aisle's waiting time after the first start, so `makespan = initial_wait + idle + busy`.
- **Bottleneck** is decided on the critical path. It is `'distillation'` if the *last* rotation starts exactly at its ready time. The makespan is then `T*interval + last cost`, so any consumption saved earlier is invisible. Otherwise it is `'consumption'`: the aisle is still busy when the last state arrives. Two aggregate rules were rejected:
  - "idle > busy" mislabels k=1 runs that have many cost-9 rotations, and the initial wait always makes idle > 0.
  - `T*interval >= busy` mislabels front-loaded work. With it, `random_clifford_t_6q` at k=4 was labelled distillation-bound even though reordering gave 1.21x.

## Strategies (`order_layer`)

Reordering happens only inside a layer, because the rotations in a layer commute with one another. `schedule` checks this with `gosc.pauli.commutes` and raises `ValueError` if any pair anticommutes. Layers keep their order, and the state that leaves layer l is the state that enters layer l+1.

- **naive:** the given order.
- **greedy:** nearest neighbour on switching cost from the current state. Ties go to the smallest original index.
- **optimal:**
  - **Objective.** It minimizes the total cost of the layer from the entering state. `pi4_cost` does not depend on order, so only switching cost matters.
  - **Up to 9 rotations: exact DP over (subset, orientation state).** A DP over (subset, last rotation) would *not* be exact, because the state after a prefix depends on the prefix's whole order (each qubit keeps the need of the last rotation that touched it). Future cost depends only on the orientation string, so keeping the cheapest way to reach each state for each subset is exact. The tests check it against brute-force permutations.
  - **Ties.** Among cost-optimal final states, it prefers the one with the lowest greedy switching cost on the next layer (lookahead). Remaining ties go to the lexicographically smallest order.
  - **More than 9 rotations.** 2-opt (segment reversal) starts from the better of greedy and naive and accepts only strict improvements.

### Guarantees and non-guarantees

- `optimal <= greedy` and `optimal <= naive` hold for the cost of a single layer from a given entering state.
- Greedy is **not** guaranteed to beat naive on every instance. `scripts/run_benchmarks.py` tests 2000 random single layers (n=4..10, 3..7 rotations, random entering state, seed 0). Pure nearest neighbour was worse than naive in 55 of them (2.75%). Optimal was never worse than min(greedy, naive). Total costs were naive 58351, greedy 51100, optimal 47356. The tests assert greedy <= naive only in aggregate.
- Per-layer optimality is **myopic** and uses **total cost, not time**:
  - Makespan is `max_i(ready_i + sum_{j>=i} cost_j)`. Two orders with the same total cost can have different makespans: putting the expensive rotations late is worse when the early ones would otherwise wait for magic states.
  - The end state of one layer affects the next layer, and the one-layer lookahead only breaks ties.
  - So `optimal` can lose to `naive` on makespan. This happened in 4 of 115 benchmark cases, including `adder_5q` at k=8 (25.88 vs 24.38, with equal busy time 23).
  - A ready-time-aware DP (minimizing end time per (subset, state), which is monotone and therefore also exact) would remove the within-layer part of this. It is not implemented.

## Lower bounds (`lower_bound`)

- `loose = max(T, T*interval + 1)`
- `tight = max(loose, T*interval + min cost in the last layer, interval + sum of per-rotation minimum costs)`

The minimum cost of a rotation in any state is `1 + pi4_cost` for compact, `consume_time` for intermediate, and 1 for fast. Using `1 + pi4` on the fast block would overestimate the cost and break the bound. The tests check `makespan >= tight` for every block, strategy and k.

## Benchmark summary (`scripts/run_benchmarks.py` -> `results/benchmarks.{csv,md}`)

- **Synthetic** (compact block, 10 random commuting layers of 4..9 rotations, 5 seeds per n). Geometric-mean speedup naive/optimal:
  - k=1: 0.999 to 1.002 (distillation-bound in all 20 runs: makespan = T*11 + last cost).
  - k=2: 1.04 to 1.22.
  - k=4..16: 1.17 (n=20) to 1.30 (n=8).
  - Greedy recovers about half of the gain (1.06 to 1.17).
- **Compiled circuits** (gosc.compile, `t_layers`; layers with more than 9 rotations go through 2-opt, not the exact DP):
  - **Toffoli** (7 T, one layer; 4 of 7 rotations have an X/Y component) and **5-qubit adder** (14 T, layers [10, 4]; 8 of 14): busy time is identical for every strategy (10 and 23). Reordering finds no switches to save, so the speedup is 1.000.
  - The adder at k=8 gets *worse* with greedy and optimal (25.88 vs 24.38, speedup 0.942), for the time-blind reason above.
  - **Random 6-qubit Clifford+T** (31 T, layers [8, 10, 9, 4]): busy time drops from 104 to 77. Speedup is 1.03 at k=2, 1.21 at k=4, 1.34 at k=8 and 1.35 at k=16. At k=1 it is 1.000.
- **When reordering helps.** It helps only in the consumption-bound regime (interval < typical cost, so k >= 2 for compact). In the distillation-bound regime (k=1), each rotation waits for its magic state anyway, and reordering can save at most 8 steps in total.
