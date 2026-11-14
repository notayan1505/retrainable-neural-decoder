# Orchestrator Run-Book — QuantUM Hackathon

**Read first:** `CLAUDE.md` (your operating rules) and `HACKATHON.md` (the challenge, rules, rubric). This file is the plan to execute them. You are the Lead Architect: plan, delegate, review. Do not write routine implementation code in the main session.

## 0. Before anything else (10% planning phase)
1. Re-read `HACKATHON.md` §1. The exact S and N values and the real challenge notebooks may not be available yet. **If the user has the real prompt/notebook, ask them to drop it in `Notes/` and fold it into `HACKATHON.md` before building.** If not, proceed with the generic engine below. It is designed so that any layout, op list, and N plug in as data.
2. Set up the environment (single shot, one worker): Python 3.12 venv in `./.venv`, `pip install "qiskit[visualization]>=2.5.0" qiskit-aer qiskit-ibm-runtime matplotlib numpy networkx pytest`; attach the Qiskit Docs MCP (`uvx qiskit-docs-mcp-server`) so workers write current APIs. **Never** use `execute`, `IBMQ`, or `ibmq_*` backends.
3. Ask the user once for: IBM API key + CRN (only if real-hardware runs are wanted), and presentation length/format. Do not block the build on these.

## 0b. REFOCUS (user clarification): the core deliverable is noise → Qiskit error correction
Priority order from now on: (1) **QEC pipeline in Qiskit**: simulated noise injected into a surface-code memory experiment, syndrome extraction over multiple rounds, decoding (lookup + PyMatching), correction, and a clear before/after result; (2) the tile-game / scheduler / blocks work as the "Game of Surface Codes" framing and the source of the insight; (3) presentation. Concretely add these tasks after T7 lands:
- **T11 (Opus 5.5):** multi-round memory experiment on Surface-17 under circuit-level noise (Aer NoiseModel: depolarizing 1q/2q, readout, optionally amplitude/phase damping), repeated syndrome rounds, detection-event graph decoding with PyMatching; compare **encoded logical qubit vs a single unencoded physical qubit** over the same time and report the break-even (pseudo-threshold) physical error rate. Also a d=5 (rotated surface code, 25 data qubits) comparison to show scaling if time allows.
- **T12 (Sonnet 5):** an end-to-end demo notebook: "inject noise → watch syndromes → decode → corrected state", Colab-ready, plus the figures for the slides.
- **T13 (you):** decide the headline claim from real numbers only (e.g. encoded beats unencoded below p≈X; logical error ∝ p² at d=3), and state the model's limits honestly.

## 0c. NEW REQUIREMENT: a neural-network surface-code decoder
The user wants a **neural surface-code decoder** as a centrepiece (learned decoder vs the classical baselines we already have). Treat it as task **T14** and the main "Methods"/"Effect" differentiator.

**Spec**
- **Task:** map syndrome data to a correction / logical-flip prediction for the Surface-17 (d=3) memory experiment, optionally d=5 if the code from T11 supports it.
- **Data:** generate labelled training data from the project's own Qiskit/Aer noisy memory experiment (`gosc/memory_experiment.py`, `gosc/qiskit_check.py`): inputs = detection events (Z-check and, if used, X-check syndrome history over R rounds, plus the final data-qubit readout-derived detectors); label = whether the logical Z_L (and separately X_L) flipped, or the full per-qubit Pauli correction. Use fixed seeds, separate train/val/test splits by seed, and a documented noise model (depolarizing 1q/2q + readout + reset, p as the knob). Aer is slow for millions of shots; it is acceptable to generate the bulk training set with a fast Pauli-frame/`stim` sampler of the *same* circuit and noise, as long as the final reported test numbers come from Qiskit/Aer-generated shots (state this clearly).
- **Models to try (small first):** (a) MLP on flattened detection events; (b) 1-D conv or GRU/LSTM over rounds (for repeated-round data); optionally (c) a small transformer or graph NN over the detector graph. Binary cross-entropy on the logical flip, optionally with the matching decoder's prediction as an extra input or as a residual ("learn the correction to MWPM"). Keep models small enough to train on a laptop CPU/MPS in minutes.
- **Framework:** PyTorch if a wheel exists for this Python (`pip install torch`; the venv is Python 3.14, so if no wheel is available create a second venv with Python 3.12 via `brew install python@3.12` / `uv`, or fall back to scikit-learn `MLPClassifier`/JAX/numpy). Do not block on this; decide in the first step and note it.
- **Evaluation (this is the point, be rigorous):** logical error rate vs physical error rate p for: unencoded qubit, raw (undecoded) encoded, lookup decoder, PyMatching MWPM, and the neural decoder, on the **same held-out Aer shots**, with Wilson confidence intervals. Also report: generalisation to noise levels not seen in training (train on p in a band, test outside it), generalisation across number of rounds, calibration (does the predicted probability match frequency), and inference latency per shot vs MWPM. Report honestly if the neural decoder only matches (or fails to beat) MWPM at d=3. Matching is near-optimal for independent noise, so the expected honest finding is "matches MWPM", with possible gains on **correlated/non-Pauli noise** where MWPM's assumptions break. To make a real insight, add one noise model that violates MWPM's independence assumptions (e.g. correlated two-qubit errors / crosstalk, biased noise X>>Z, or amplitude-damping/coherent over-rotation twirled badly) and test whether the learned decoder wins there. Only claim a win if the numbers show it.
- **Qiskit integration:** the training data, the circuits and the noise come from Qiskit/Aer; provide `decode_with_nn(syndrome_history) -> correction` with the same signature as the lookup/MWPM decoders so the three plug into one evaluation harness (`gosc/decoders.py`).
- **Deliverables:** `gosc/neural_decoder.py` (model, training loop, save/load weights to `models/`), `gosc/decoders.py` (common interface + evaluation harness), `scripts/train_nn.py`, `scripts/eval_decoders.py`, `tests/test_neural_decoder.py` (tiny-data smoke tests: shapes, training reduces loss on a toy set, it decodes all 27 single errors exactly like the lookup table once trained on the noiseless-plus-one-error set, determinism with seeds), `results/neural_decoder.md`, figures in `presentation/figures/nn_*.png`, and a notebook cell block for T12.
- **Delegation:** data generation + harness: Sonnet 5; model/training/eval: Opus 5.5; deciding the headline claim and the honest limitations: you (Fable 5.2).

## 0d. Status snapshot (what already exists, do not rebuild)
Built and tested (`python -m pytest -q` → 1739 passed at last run), all in `gosc/` unless noted: `pauli.py` (Pauli algebra; NOTE the correct commutation sign is Q' = −i·P·Q, tested against Qiskit), `board.py`/`ops.py`/`engine.py` (tile game, costs from the paper), `compile.py` (Clifford+T → Pauli rotations + T layers), `blocks.py`/`distillation.py` (compact/intermediate/fast blocks, 15-to-1 distillation, paper examples reproduced), `scheduler.py` (state-aware rotation scheduler; reordering gives ~1.17–1.30× when consumption-bound, nothing when distillation-bound), `qiskit_check.py` (Surface-17 with a *proper* |0>_L, decoders, noisy Aer, Bell/PPM circuits; finds that the workshop notebook's "|0>_L" is really |1>_L), `viz.py` (figures). In progress at the time of writing: `memory_experiment.py` (T11, repeated-round circuit-level noise, encoded vs unencoded, break-even). Environment: `.venv` with Qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-ibm-runtime, pymatching, matplotlib, numpy, networkx, pytest.

## 1. Target architecture (you own this design)
```
QuantUM/
  gosc/                    # "Game of Surface Codes" package
    board.py               # Tile grid, Patch (edges: X/Z, tiles), occupancy, validation
    ops.py                 # Op types + costs: init(0), measure1(0), pp_measure(1), grow(1), shrink(0), corner_move(1), rotate(3)
    engine.py              # Execute an op list on a board; enforce legality; return time N + trace
    pauli.py               # Pauli-product algebra: multiply, commute, Y = iXZ, rotation commutation rules
    compile.py             # Circuit -> Pauli-product rotations -> T layers (greedy algo); Clifford pushing
    blocks.py              # Compact / intermediate / fast data-block layouts (tile counts, magic-state cost)
    scheduler.py           # Given layout + ops: find schedule minimizing time; report N, tiles, space-time
    qiskit_check.py        # Verify protocols with real Qiskit circuits (stabilizer/Aer); Surface-17 syndrome demo
    memory_experiment.py   # Repeated-round noisy memory experiment, encoded vs unencoded (T11)
    decoders.py            # Common decoder interface + evaluation harness (lookup, MWPM, neural) (T14)
    neural_decoder.py      # Neural surface-code decoder: model, training, save/load (T14)
    viz.py                 # Matplotlib board renderer + timeline (for slides)
  tests/                   # pytest, one file per module
  notebooks/               # Colab-ready demo notebook(s)
  presentation/            # slides outline, figures, speaker script
```
Design rules: pure data in / data out; layouts and op lists are JSON-serializable; all costs live in one table in `ops.py` (single source of truth matching HACKATHON.md §2).

## 2. Work breakdown and delegation (80% phase)
Per `CLAUDE.md` §3, route by tier. Give every worker: exact file paths, function signatures, the cost table, and acceptance tests. Workers must not re-read the repo.

| # | Task | Tier | Depends on | Acceptance |
|---|---|---|---|---|
| T0 | Env setup + MCP attach + `pytest` smoke | Sonnet 5 | – | `python -c "import qiskit; print(qiskit.__version__)"` ≥ 2.5; no deprecated imports |
| T1 | Locate/inspect any new files the user adds to `Notes/` | Haiku (`explore`) | – | Summary only |
| T2 | `pauli.py` + tests: product, commutation, Y=iXZ, π/4 rotation pushing (P′_φ → (iPP′)_φ) | Sonnet 5 | – | Property tests vs Qiskit `Pauli` |
| T3 | `board.py`, `ops.py`, `engine.py` + tests: legality (adjacent edges, free tiles), cost accounting | Sonnet 5 | – | Reproduces paper examples: Bell pair = 1, move patch = 1, Y-measure = 2, 4-patch Y⊗X⊗Z⊗X measure = 1 |
| T4 | `compile.py` + tests: Clifford+T → π/8 rotations + final measurements; greedy T-layer reduction | Sonnet 5 / Opus 5.5 | T2 | Equivalence to the original unitary (up to Clifford) checked on ≤4 qubits with Qiskit `Operator` |
| T5 | `blocks.py` + tests: compact (1.5n+3 tiles, ≤9⏱), intermediate (2n+4, ≤5⏱), fast (2n+√8n+1, 1⏱) with explicit tile maps for small n | Opus 5.5 | T3 | Tile counts match formulas; consumption-time counts match the Fig. 11 worked example (9⏱) |
| T5b | `distillation.py` + tests: 15-to-1 block model (11 tiles, 11🕒, output infidelity 35p³), production rate with k blocks; combined data+distillation pipeline giving time per T gate = max(distillation interval, consumption time) | Opus 5.5 | T5 | Reproduces PennyLane worked examples: 164 tiles & 11🕒/T (minimal); 222 tiles & 5.5🕒 T-production with intermediate block (≈2 h at d=13, 10⁸ T gates) |
| T6 | `scheduler.py`: given layout + op list/circuit, produce a schedule and N; compare against naive serial baseline; sweep block type × distillation blocks for a space-time curve | **Fable 5.2 designs the algorithm; Opus 5.5 implements** | T3–T5 | N never below a proven lower bound; schedule re-validates in `engine.py` |
| T7 | `qiskit_check.py`: (a) Bell-pair-by-ZZ-measurement, (b) multi-patch Pauli-product measurement as Qiskit circuits on `StabilizerState`/Aer, (c) Surface-17 syndrome extraction (reuse the intro notebook's `z_stabs`/`x_stabs` and measure functions) + a decoder covering X, Z and Y errors (lookup table; optionally PyMatching) with an `apply_correction` step + noisy-Aer logical-error-rate vs p. Fix the notebook's known shortcut: prepare a |0⟩_L that is a +1 eigenstate of the X checks too (e.g. measure the X-stabilizers once and post-select/correct), and say plainly in docs which approximation is used | Opus 5.5 | T2 | Circuits run on `AerSimulator`; outcome statistics match theory; fully current APIs (SamplerV2, preset pass manager) |
| T8 | `viz.py`: board snapshots per time step, dashed X / solid Z edges, timeline, space-time curve | Sonnet 5 | T3 | PNGs render; used in slides |
| T9 | Demo notebook (Colab-ready; pip cell on top) | Sonnet 5 | T3–T8 | Runs top-to-bottom on a clean Colab |
| T10 | Optional real-hardware run of the 2-qubit stabilizer circuit via `SamplerV2` (only if user supplies credentials) | Sonnet 5 | T7 | Job ID logged; sim vs hardware histogram figure |

Parallelize T2/T3/T8 immediately; T4/T5 once T2/T3 land; T6/T7 after. Ambiguous cross-module bugs stay in the main session.

## 3. The insight to chase (30% of grade — you own this)
Pick whichever the data supports, and say so honestly in the deck:
- **Scheduler result:** for a given layout and S ops, the minimum N found vs the naive serial N, and a lower bound showing how close it is (ops on disjoint ancilla regions can be measured in the same time step; magic-state consumption pipelining).
- **Space-time trade-off:** time per T gate vs tiles for compact/intermediate/fast blocks, plus the effect of extra distillation blocks, reproducing the paper's trend in our own engine.
- **QEC-to-game bridge:** use Surface-17 + noisy Aer to show how code distance d converts "1 time step" into d code cycles and what that does to logical error rate; link the degeneracy observation from the notebook to why decoding is local.
Do not claim results the code didn't produce. Report negatives plainly.

## 4. Review phase (final 10% — you)
- Run `pytest -q` and the notebook end to end. Inspect every worker diff; verify module integration and that costs match the single cost table.
- Re-check every number in the slides against HACKATHON.md and the paper.
- Grep the repo for `execute(`, `IBMQ`, `ibmq_` and remove.
- Confirm no secrets are committed (API key/CRN stay in env vars or `save_account` run interactively).

## 5. Presentation (40% — do not skimp)
Produce `presentation/outline.md` plus figures, structured as the rubric asks:
1. **Problem in one slide:** why logical qubits cost time; the game rules (board, clock, objective).
2. **Method:** engine + compiler + scheduler (one architecture diagram).
3. **QEC grounding:** Surface-17, stabilizers, syndrome, decoding; how a tile and a time step map to d² qubits and d cycles.
4. **Qiskit slide:** exactly where Qiskit is used (circuits, Aer noise, SamplerV2, transpile, Docs MCP grounding to avoid deprecated APIs, optional hardware run).
5. **Result/insight:** the one chart that matters (space-time or N vs baseline).
6. **Design justification + limits:** why these choices; what the model ignores (decoder latency, state-injection error, routing congestion).
7. **Demo + takeaway** in ≤ 5 minutes of speaking. Include a short speaker script.
Offer to also build the final deck via the slides skill if the user wants.

## 6. Guardrails
- Do not fabricate: the real S/N values and challenge notebooks are unknown until provided. Flag this in the final report.
- Keep the whole thing runnable on Colab (no local-only dependencies).
- Real-hardware jobs consume the user's QPU time; run them only when asked.
- Finish with a short report: what was built, what was verified, what is unverified, and what the user must do (credentials, final prompt, rehearsal).
