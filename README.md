# Learning a Chip's Noise: A Neural Decoder Trained on Real IBM Data

QuantUM Qiskit Fall Fest 2026 · 3-hour build · status: plan v3, approved to build. This file is the single source of truth.

## Goal
A neural decoder trained on **real IBM hardware data** and compared fairly against PyMatching,
built on Qiskit, with an open, one-command retrain.

## Plain-English background
Error correction spreads one logical bit over several physical qubits and keeps measuring
parity checks. A **decoder** reads the check results and guesses whether the stored bit
flipped. PyMatching is the standard decoder: fast and strong, but only as good as the noise
model it is given. Real chips have uneven, drifting noise. A learned decoder can pick that up
from data.

## Data

### Primary: Stein et al., IBM repetition-code dataset
[zenodo.org/records/20768087](https://zenodo.org/records/20768087) (CC BY 4.0, ~146 MB compressed)
- Raw hardware runs on **ibm_kingston, ibm_pittsburgh, ibm_fez**.
- d = 3, 5, 7, 9, 11; rounds r = 1–11; X and Z bases; 352 snapshots. The paper reports about 2.7M shots.
- Each job folder holds `info.json` (parameters), `calibration.json` (T1/T2, gate and readout
  errors, coupling map at run time), `circuit_state0/1.qasm` (transpiled circuits) and
  `bitstrings.json` (raw per-shot measurements for each parallel chain).
- **Detection events and logical outcomes are NOT precomputed** (per the record page; the
  README still needs checking). We derive them from the raw bitstrings, using the qasm to map
  classical bits to qubits and rounds. **This parsing is the first real task. Budget for it.**

### Fallback: Google 2022 surface-code dataset
[zenodo.org/records/6804040](https://zenodo.org/records/6804040). Stim/PyMatching-ready, but
it is a surface code on Google hardware, not IBM.

### Our own 10 IBM minutes: a fresh test set
- We run the same repetition-code circuits (via `qiskit-ibm-runtime`) **today**, months after
  Stein's data was taken.
- These runs are never trained on. That makes this the honest test, and it doubles as a
  **drift test**: does a decoder trained on June data still work on November hardware?
- Before running on hardware, everything is checked on a fake backend
  (`AerSimulator.from_backend(...)`) first.

## Input / output
- **Input:** detector bits shaped **(r+1) × (d−1)**.
  - Round 1: `D_{1,i} = m_{1,i}`.
  - Rounds 2..r: `D_{t,i} = m_{t,i} ⊕ m_{t-1,i}`.
  - Final row: data-implied parity `(q_i ⊕ q_{i+1}) ⊕ m_{r,i}`.
- **Output:** "did the stored bit flip?" (0/1). The label is `q_0 ⊕ b`, where `b` is the
  prepared state (state0 or state1 circuit).

## Decoders, all compared on the same held-out runs
| # | Decoder | Notes |
|---|---|---|
| 0 | No decoding | raw readout |
| 1 | PyMatching, equal weights | graph only |
| 2 | PyMatching, weights fitted from data | pairwise estimator, **including diagonal space-time edges** |
| 3 | Lookup table | empirical MAP per detector pattern; small d only (d=3: 2^((d−1)(r+1)) patterns) |
| 4 | Our network | trained on Stein data |

Weight fitting (decoder 2): for detectors `i, j`,
`p_ij = 1/2 − sqrt(1/4 − (⟨x_i x_j⟩ − ⟨x_i⟩⟨x_j⟩) / (1 − 2⟨x_i⟩ − 2⟨x_j⟩ + 4⟨x_i x_j⟩))`,
with weight `w = log((1−p)/p)`. Fit it on the **training** split only.

## What to expect (be honest in the pitch)
- **A tie at small sizes.** At d=3 the lookup table is near-optimal, and fitted-weight
  matching is close behind.
- **A possible win only around d ≥ 7, r ≥ 5.** That's where Stein et al. started beating
  matching.
- Stein et al. reported up to **11.1× lower logical error than hardware-informed matching at
  d = 11, r = 11** (arXiv abstract; one news summary attributes it differently, so cite the
  paper's own wording).

## Measurement rules (gotchas)
- **Error per round** when comparing different round counts:
  `ε = (1 − (1 − 2P_L)^(1/r)) / 2`.
- **Sample size:** about **4,000+ test runs** to resolve a 1-percentage-point difference.
  Show binomial error bars.
- **Learning curve:** plot test error vs. training-set size for the NN (and the lookup
  table). It shows where learning beats fitting.
- **Strict splits:** fitted weights, lookup table and NN all train on the same training split
  and are tested on the same held-out runs.

## How we differ from Stein et al. (we cite them, we don't claim "first")
Stein et al. already trained neural decoders on this data. Their key idea is
**calibration-conditioned FiLM**, which feeds the chip's calibration numbers into the network.
Our contribution:
1. **Fresh-hardware drift test:** decoders trained on their June runs, tested on our new
   November runs on the same chips. How much does performance decay, and does refitting
   matching weights or fine-tuning the NN recover it?
2. **Open, one-command retrain** (`train --backend ibm_kingston --d 7 --r 5`), Qiskit end to
   end, runs on a consumer GPU (RTX 5070 Ti).
3. **A learning-curve view:** how much real data each approach needs before it wins.

## Other prior work to cite
- NVIDIA Ising (Apr 2026): open, fine-tunable surface-code AI pre-decoders.
- AlphaQubit 1/2 (Google DeepMind): weights not public as far as we found.
- Lange et al. 2023: open GNN decoder.

## 3-hour schedule
| Time | Task |
|---|---|
| 0:00–0:40 | Download Stein data, read the README, write the parser (bitstrings → detectors + labels). Sanity check: detector rates are low, and the label is balanced across state0/state1 |
| 0:40–1:10 | Decoders 0–3 on d=3 and d=7; first table. **Submit our IBM job here** so it runs while we work |
| 1:10–2:00 | NN training on GPU, d=7–11, r ≥ 5; learning curve |
| 2:00–2:30 | Evaluate everything on our fresh IBM runs (drift test) |
| 2:30–3:00 | Plots, slides, rehearse |

## Hardware and environment

**Dev machine:** a Mac (Apple Silicon, MPS). Code is written and smoke-tested here on small
data. The `.venv` in the project root already has qiskit 2.6, qiskit-aer 0.18,
qiskit-ibm-runtime 0.52, stim 1.16, PyMatching 2.4, torch 2.15 and numpy.

**Training machine:** a separate laptop with an **NVIDIA RTX 5070 Ti** (Blackwell, sm_120, 16 GB).
The code is copied over and run there. Its OS is unknown, so everything must work on
**Windows and Linux**: use `pathlib`, no shell-specific scripts, `num_workers=0`.

5070 Ti optimizations (all automatic, falling back cleanly on Mac/CPU):
- **PyTorch built for CUDA 12.8 or newer is required** for Blackwell. The README must give
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` (or newer). Older
  builds fail with "no kernel image".
- **Put the whole dataset on the GPU.** It is a few million samples of ≤ 120 bits each, so it
  fits easily in 16 GB as uint8. Shuffle with `torch.randperm` on the device. No DataLoader.
- **Large batches** (8k–64k), **bf16 autocast**, `torch.backends.cuda.matmul.allow_tf32 = True`
  and AdamW with OneCycle LR.
- **`torch.compile` behind a `--compile` flag**, off by default. On Windows it needs Triton,
  which may be missing.
- **Sweeps:** the GPU is mostly idle with small models, so run several (d, r, size) configs
  as separate processes at once. A `sweep` command launches them.
- **Device selection:** `cuda` → `mps` → `cpu`, overridable with `--device`.
- PyMatching and data parsing are CPU-bound. Use `multiprocessing` across snapshots for
  parsing and cache parsed arrays as `.npz`.

## Repo layout (new package; do not touch the old `gosc/` folder)
```
qdrift/
  data/stein.py        # download + parse Stein bitstrings -> detectors, labels, metadata
  data/circuits.py     # Qiskit repetition-code circuit (same layout as Stein), fake + real backends
  decoders/baselines.py# no-decode, PyMatching equal, PyMatching fitted (diagonal edges), lookup table
  models/net.py        # NN (start: MLP; then conv/recurrent over rounds)
  train.py             # GPU-optimized training, checkpoints, learning curve
  evaluate.py          # all decoders on the same held-out split -> CSV + plots (error per round, error bars)
  cli.py               # `python -m qdrift <download|parse|baseline|train|sweep|evaluate|run-ibm>`
tests/qdrift/          # pytest; synthetic Stim repetition-code data as ground truth for parser/decoders
data/ (gitignored)     # raw downloads in data/raw/<archive>/, parsed cache in data/parsed/
results/qdrift/        # CSVs + figures
README_qdrift.md       # setup on the 5070 Ti laptop, every command, credits
requirements.txt
```

## Build tasks (the orchestrator splits work along these lines)
1. **Inspect the data (read-only).** Download one archive (start with `ibm_fez.tar.gz`, 9 MB),
   read `README.md`, `index.csv` and one job folder. Write down the exact bit layout in
   `docs/stein_format.md`. This is fuzzy work, so the orchestrator does it or supervises it closely.
2. **Parser.** Raw bitstrings → `(N, r+1, d−1)` uint8 detectors + `(N,)` labels + metadata
   (backend, d, r, basis, snapshot id, date). Validate it on synthetic Stim data and on real
   data (detector rates low, labels sensible).
3. **Baselines.** Decoders 0–3 with a deterministic train/test split per snapshot.
4. **Model + training.** GPU-optimized as above, with a learning-curve mode (train-size fractions).
5. **Evaluation + plots.** Logical error per round vs. d, plus a learning curve, with binomial error bars.
6. **IBM circuits.** Rebuild Stein's repetition-code circuits in Qiskit from their qasm or
   layout, test them on a fake backend through the same parser, and add `run-ibm` with
   `--real` (needs `QISKIT_IBM_TOKEN`; never hard-code the token). No real submissions until
   the user provides a token.
7. **CLI, README, requirements, tests.**

Acceptance: `pytest` passes on the Mac, `python -m qdrift baseline --d 3` prints a table, and
a small `train` run completes on CPU/MPS.

## Risks
- **Parsing the raw bitstrings is fiddly** (parallel chains, bit ordering, qasm mapping).
  Highest risk, so do it first. Fallback: the Google dataset, which is Stim-ready.
- **The IBM queue or our minutes don't come through.** Then hold out a later Stein snapshot
  as the "fresh" set instead.
- **The NN doesn't beat fitted matching in our time budget.** We still present the drift test
  and the learning curve. Both are results either way.

---

## STATUS / HANDOFF (updated 2026-11-14, ~14:15) — read this first when resuming

### Progress by build task
| # | Task | State | Where |
|---|---|---|---|
| 1 | Inspect data | **Done** | `docs/stein_format.md` (layout verified on all 41 fez snapshots) |
| 2 | Parser | **Done**, reviewed | `qdrift/data/stein.py`, `qdrift/data/synthetic.py`, `tests/qdrift/test_stein.py` |
| 3 | Baselines | **Done**, reviewed, one fix by orchestrator (boundary floor) | `qdrift/decoders/baselines.py`, `tests/qdrift/test_baselines.py` |
| 4 | Model + training | **Done**, reviewed, two fixes (compile fallback, learning-curve batch size) | `qdrift/models/net.py`, `qdrift/train.py`, `tests/qdrift/test_train.py` |
| 5 | Evaluation + plots | **Written and tested, NOT yet reviewed by the orchestrator** | `qdrift/evaluate.py`, `tests/qdrift/test_evaluate.py` |
| 6 | IBM circuits | **Done**, partly reviewed (sampler fix); fake backend only | `qdrift/data/circuits.py`, `tests/qdrift/test_circuits.py` |
| 7 | CLI, README, requirements, tests | **Written and tested, NOT yet reviewed by the orchestrator** | `qdrift/cli.py`, `qdrift/__main__.py`, `README_qdrift.md`, `requirements.txt`, `.gitignore`, `tests/qdrift/conftest.py`, `tests/qdrift/test_cli.py` |

**Current state: `.venv/bin/pytest tests/qdrift -q` gives 68 passed in 28 s** (three runs in a
row, the last one after all orchestrator fixes; no segfault). Acceptance checklist is met:
pytest passes, `python -m qdrift baseline --d 3` prints tables and writes
`results/qdrift/baseline_ibm_fez_d3.csv`, and a small train run completed on MPS.

### To do next, in order
1. Review `qdrift/evaluate.py`, `qdrift/cli.py` and `tests/qdrift/test_cli.py` (not read by the
   orchestrator yet). Check: lazy imports (`run-ibm` must not import torch, `baseline` must not
   import torch/qiskit), `run-ibm` output lands where `parse <label>` finds it, `--real`
   asks for confirmation, plots look right. Do not train on the Mac.
2. One end-to-end fake drift check (about 1 minute, no training):
   `python -m qdrift run-ibm --backend ibm_fez --configs 3x3` then `parse ibm_fez_fake` then
   `evaluate --backend ibm_fez --configs 3x3 --test-backend ibm_fez_fake`. Not run yet.
3. Read `README_qdrift.md` against the real CLI flags (`python -m qdrift <cmd> --help`).
4. Note: `sweep` defaults to `--jobs 1` and `--kinds mlp`; pass `--jobs 3` or more on the GPU.
5. Check `README_qdrift.md` has every command for the 5070 Ti laptop (Windows + Linux).
6. On the 5070 Ti laptop: download + parse kingston and pittsburgh, run baselines, sweeps,
   learning curves. Nothing has been trained beyond one 6-second smoke run.
7. When a token exists: `run-ibm --real` for the drift test. Until then only fake backends.

### First numbers (real ibm_fez, X and Z pooled, 75/25 split per snapshot and state)
d=3, r=3, 69,120 train / 23,040 test samples, before the boundary-floor fix:
| decoder | logical error | ± | state 0 | state 1 |
|---|---:|---:|---:|---:|
| none | 9.67% | 0.20 | 4.05% | 15.30% |
| mwpm_equal | 4.34% | 0.13 | 1.75% | 6.93% |
| mwpm_fitted | 3.37% | 0.12 | 1.21% | 5.53% |
| lookup | 3.40% | 0.12 | 1.28% | 5.53% |
| MLP (15 epochs, MPS, 5.8 s) | 4.02% | – | – | – |

After the boundary-floor fix `mwpm_fitted` is 3.54% at d=3, r=3 (about one standard error
worse) and clearly better at larger sizes: d=7, r=5: 2.31% (was 2.80%; equal weights 2.73%);
d=11, r=11: 9.69% (was 12.71%; equal weights 10.13%). The MLP above is a smoke run, not tuned,
and it does not yet beat fitted matching or the lookup table at d=3. That matches the
"tie at small sizes" expectation.

### Facts found that change the plan
- **Dates:** fez calibration dates are 2025-08-25 and 2025-10-03, not June 2026. The drift
  gap to "today" is over a year. Fix the pitch wording. Kingston: 2025-08-28 to 2025-10-08. Pittsburgh: 2025-09-02 to 2025-10-08.
- **Noise is high and lopsided:** detector rates 5–30%. In Z basis state 1 is far worse than
  state 0 (raw `q_0` error at d=11, r=11: 0.9% vs 47%). Report per state; splits are balanced
  by state.
- **Coverage:** fez has only odd r ≤ d. Kingston and pittsburgh also have some even r (kingston d=5 r=2,4; d=9 r=2,4) and r > d (kingston d=3 r=5; pittsburgh d=3 r=5,7, d=5 r=7).
- **`index.csv` is not on Zenodo.** `qdrift.data.stein.build_index` rebuilds it.
- **Shot counts:** README says 3,779,584 shots / 20,654,080 samples in total (the plan's
  "2.7M" is out of date). Fez: 164,864 shots, 1,099,776 samples, both confirmed.
- All three backends downloaded and parsed (2026-11-14, cloud session): 20,654,080 samples, exactly the README total. Raw unpacks to ~11 GB; parsed cache < 100 MB.

### Kingston + pittsburgh baselines (2026-11-14, `results/qdrift/baseline_ibm_{kingston,pittsburgh}_d*.csv`)
- **The lookup table beats matching by a lot wherever it fits** (d=3 up to r=7, d=5 up to r=5): kingston d=5 r=5: lookup 2.22% vs fitted 3.65% / equal 3.52%; kingston d=3 r=5: 2.75% vs 5.00% / 8.36%; pittsburgh d=3 r=7: 2.44% vs 4.41%. The noise has structure a matching graph cannot express (correlated errors, leakage-like effects). That is the room a neural decoder can learn into, and the best argument for the NN at d ≥ 7 where no lookup table fits.
- **Fitted weights do not beat equal weights at r ≥ 5 on kingston, and barely on pittsburgh** (kingston d=11 r=11: equal 8.83%, fitted 9.21%). This is NOT caused by pooling: fitting per snapshot (9.23%) or per chain (9.80%) is no better. The pairwise estimator itself breaks down at these noise rates. For the pitch: compare the NN against the better of `mwpm_equal` and `mwpm_fitted` per config, and say so.
- `mwpm_equal` is worse than no decoding at kingston d=3 r=5 (8.36% vs 7.86%). Not investigated.

### Design decisions made
- One sample = one (shot, chain). Label = `q_0 XOR prepared state`. Same rule for X and Z.
- Parsed cache: one `.npz` per snapshot, format in `docs/stein_format.md`.
- Split: random 25% test per (snapshot, state), seeded by snapshot id, so it is stable when
  more data is added.
- Fitted matching pools all chains and both bases of a config into one graph. A per-chain
  fit would be a stronger baseline. Not done.
- **Boundary floor** (`fit_edge_probs(boundary_floor=True)`): the pairwise estimator gave
  zero or negative chain-end probabilities on real data, making fitted matching worse than
  equal weights from d=7 up. Chain-end probabilities are now floored at the row's median
  space-edge probability. Caveat: this rule was picked after comparing variants on the test
  split (differences were within noise).
- Fresh/fake IBM runs are stored under their own backend label (`ibm_fez_fake`,
  `ibm_fez_fresh`) so they never mix with Stein's `ibm_fez`.
- `find_chains` picks chains from the coupling map only. It does not avoid bad qubits using
  calibration data. Worth adding before real runs.

### Python crashes seen and what was found
- **Symptom:** `pytest tests/qdrift` segfaulted intermittently (7 macOS crash reports,
  `Python-2026-11-14-1351xx … 1354xx.ips`, all `EXC_BAD_ACCESS / SIGSEGV`).
- **Cause (confirmed from the crash reports):** every crash is inside OpenMP worker threads
  (`__kmp_fork_barrier`, `__kmp_launch_worker`). qiskit-aer and torch each ship their own
  OpenMP runtime and crash when both are loaded in one process.
- **Workaround in place:** `tests/qdrift/conftest.py` sets `OMP_NUM_THREADS=1` (and
  `KMP_DUPLICATE_LIB_OK=TRUE`) before anything is imported; `qdrift/cli.py` is specified to
  do the same and to import heavy modules lazily per command.
- **Verified:** three full-suite runs in a row with the conftest in place, no crash. If it
  ever comes back, run the circuits tests in their own process:
  `pytest tests/qdrift/test_circuits.py` and `pytest tests/qdrift --ignore=tests/qdrift/test_circuits.py`.
- `OMP_NUM_THREADS=1` only matters on the Mac test runs. On the 5070 Ti laptop, training does
  not load qiskit-aer, so it is unaffected.
- Other pitfalls hit: plain `pytest` could not import `qdrift` without the project root on
  `sys.path` (conftest now adds it); run `python -m qdrift` from the project root.

### Rules still in force
- Do not touch `gosc/`. No real IBM submissions (no token yet). No training on the Mac.
- Downloads are untrusted: JSON/text parsing only.
