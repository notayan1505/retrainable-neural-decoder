# QuantUM Qiskit Fall Fest 2026 — Hackathon Brief

Source of truth for *what the hackathon is*. The execution plan lives in `ORCHESTRATOR.md`; model/delegation rules live in `CLAUDE.md`.

Sources digested: Google Slides "QuantUM Qiskit Fall Fest 2026 Pre-Hackathon Workshop" (same as the `.pptx` in `Notes/`; speaker notes are empty), `Notes/Hackathon Learning Resources.docx`, `Notes/Qiskit 101 AI.pdf`, both notebooks in `Notes/`, and Litinski, *A Game of Surface Codes* (arXiv:1808.02892).

---

## 1. The challenge: "Game of Surface Codes"

The slide deck reveals the prompt as: **"A game of surface codes: Large-scale quantum computing with lattice surgery"** (Litinski 2019).

**Rules as stated on the slide**
- **Board (space):** Logical data qubits are *patches* occupying one or more tiles on a 2D grid. Patch boundaries set the available operators: **dotted edges = X-edges, solid edges = Z-edges**.
- **Clock (time):** Every action costs time in surface-code cycles. Qubit initialization is instantaneous (0). Multi-qubit Pauli measurements via lattice surgery extend a patch boundary and cost **1 time step**.
- **Objective:** *For a given surface code layout and time cost N, perform S operations in N time.*

> **Clarification from the user (supersedes the guess below):** at its core the task is to **throw noise (simulated noise is allowed) at a quantum system and have the Qiskit code error-correct it.** The Game-of-Surface-Codes material is the framework/context; the graded substance is a working noisy-simulation → syndrome extraction → decode → correct pipeline in Qiskit, with evidence that correction helps (e.g. logical vs unencoded error rate). The tile-game engine, scheduler and block models are supporting material, not the centre.

> **Team decision:** the project will also include a **neural-network surface-code decoder** (learned decoder trained on simulated noisy syndromes from Qiskit/Aer), evaluated against lookup-table and PyMatching decoders. See `ORCHESTRATOR.md` §0c for the spec and evaluation rules. This is aimed at the Methods (20%) and Effect (30%) criteria; claim only what the held-out numbers show.

> **Decision (user, 2026-11-14): S and N are ours to define.** We state a general rule instead of one fixed prompt. An instance = (layout: block type, n, distillation blocks) + (S-operation list) + (budget N). Solved iff the scheduler returns a schedule that `engine.py` validates with total time <= N. N is reported against a naive serial upper bound and a proven lower bound (T-layer count / distillation interval). Demo instances are our own and must be labelled as such, not as the official prompt. Each time step maps to d code cycles, linking N to the Aer logical-error results. If the real prompt appears, drop it in as data.

> **Open gap (older note, now superseded by the decision above):** the slide gives the objective only in this abstract form. The concrete challenge notebooks, layouts, and the values of S and N are not in any provided file. They are probably released at the event ("Prompt Reveal…"; the resources doc says "all the challenge notebooks are executable on Google Colab"). Treat everything below as the framework to build against, and re-check this section when the real prompt drops.

### Grading rubric
| Criterion | Weight | What judges ask |
|---|---|---|
| Qiskit | 10% | How did you integrate Qiskit into the project? |
| Methods | 20% | Creative **and accurate** use of QEC elements? |
| Effect | 30% | Conceptual breakthroughs / further insights? |
| **Presentation** | **40%** | Can you explain the methods succinctly and justify design choices? |

Implication: 70% of the score is insight + communication. A clean, well-explained, correct result beats a sprawling one. Qiskit use is only 10% but is a hard requirement, so it must be real, not decorative.

---

## 2. Game rules in full (from the paper)

**Patches.** One-qubit patch: 2 dashed (X) + 2 solid (Z) edges; can be 1 tile (square) or stretched over several. Two-qubit patch: 6 edges (X1, Z1, X2, Z2, plus Z1·Z2 and X1·X2).

**Operations and costs** (⏱ = one time step ≈ d surface-code cycles)
| Operation | Cost |
|---|---|
| Init one-qubit patch in \|0⟩ or \|+⟩ (also two-qubit \|00⟩, \|++⟩) | 0 |
| Init arbitrary state (state injection; may carry error probability p) | 0 |
| Single-patch X or Z measurement (frees the tiles) | 0 |
| Two-patch Pauli product measurement (adjacent edges, e.g. Z⊗Z) | 1 |
| Multi-patch measurement via arbitrary-shaped ancilla patch (any product of adjacent edge operators, e.g. Y⊗X⊗Z⊗X) | 1 |
| Grow patch edge onto a free tile | 1 |
| Shrink patch edge inward | 0 |
| Move a corner of a patch (reshape) | 1 |
| Patch rotation (swap which edge faces a neighbour, compact block) | 3 |

**Derived protocols**
- Bell pair: init two patches in \|+⟩, measure Z⊗Z → 1.
- Move a square patch any distance: enlarge then shrink → 1.
- Y-basis measurement: deform so X and Z edges are on one side, ancilla in \|0⟩, measure Z⊗Y, discard ancilla → 2.
- Adjacent edges: if qubit A's Z edge touches both B's X and Z edges, you measure Z_A ⊗ Y_B (Y = iXZ).
- Space cost = s·d² physical data qubits; time cost ≈ t·d cycles; space-time ≈ s·t·d³.

**Circuit model.** Cliffords are free/classical; T gates cost a magic state |m⟩ = |0⟩ + e^{iπ/4}|1⟩.
- Write circuits as Pauli-product rotations P_φ = exp(−iPφ). S = Z_{π/4}, T = Z_{π/8}, H = Z_{π/4}·X_{π/4}·Z_{π/4}, CNOT = (Z⊗X)_{π/4}·(1⊗X)_{−π/4}·(Z⊗1)_{−π/4}.
- Commute all π/4 rotations to the end. If P and P′ commute they pass freely. If they anticommute, P′_φ becomes (iPP′)_φ. A circuit becomes a sequence of π/8 rotations (T count) plus final Pauli-product measurements.
- Group mutually commuting π/8 rotations into **T layers** (T depth). Greedy: move rotation j from layer i+1 to layer i if it commutes with everything in layer i; repeat to fixpoint. Equal rotations in one layer merge into a π/4 rotation (lowers T count).
- A P_π/8 rotation = measure P⊗Z with a magic state (1⏱). Outcome −1 → corrective P_π/4 (commuted to the end). Then measure the magic state in X; outcome −1 → P_π/2 Pauli correction (commuted to the end).

**Data blocks (n data qubits)**
| Block | Tiles | Time to consume one magic state |
|---|---|---|
| 2-tile-per-qubit baseline | 3n | 1 |
| Compact | 1.5n + 3 | up to 9 (worst case: even number of Y's in the product) |
| Intermediate | 2n + 4 | up to 5 |
| Fast | 2n + √(8n) + 1 | 1 |

Compact-block worst case (even # of Y): two explicit π/4 rotations (Y_π/4 = Z_π/4 X_π/4 Z_−π/4; each is a P⊗Y measurement with a |0⟩ ancilla, then X-measure the ancilla) + patch rotations (3) + the magic-state measurement.

**Headline numbers** (100 qubits, 10⁸ T gates, 10⁶ T layers, p=10⁻⁴, d=13, 1 µs cycle): ~55,000 qubits / 4 h (compact + 1 distillation block); ~120,000 qubits / 22 min (+10 distillation blocks); ~330M qubits / 1 s (Fowler time-optimal, 1500 parallel units). Two-qubit proof-of-principle device: 48 physical data qubits (Appendix C).

### 2b. Extra detail from the linked PennyLane tutorial
Source: https://pennylane.ai/demos/tutorial_game_of_surface_codes (the link on the challenge slide; it is a conceptual walkthrough with **no code or exercises**, and the phrase "S operations in N time" does not appear there). It adds:
- Physical qubits are statically arranged at **2d² per tile**; patches are dynamic (appear, move, deform, vanish). Cost unit is the code cycle (🕒).
- **Y measurement** has two costs depending on layout: 1🕒 when a shared edge exposes both X and Z of the patch (with an auxiliary |0⟩); 2🕒 for a single-qubit Y measurement (deform 1 + init aux 0 + joint measure 1 + deform back 0).
- **Compact block worst case = 9🕒** = 1 (measurement) + 2 (even number of Y ⇒ two π/4 rotations) + 6 (two opposite patches both need X ⇒ rotation 3🕒 each). Only Z edges face the auxiliary aisle in the compact block.
- **Intermediate block** (2n+4 tiles): linear qubit layout removes the opposite-X case; auxiliary region can be the upper or lower row; max rotation overhead 2🕒; max total 5🕒 (1+2+2).
- **Fast block:** two-qubit patches on two tiles (six edges) with an all-encompassing auxiliary region ⇒ all 15 non-identity two-qubit Paulis directly available; non-Clifford rotation ≤ 1🕒.
- Rewrite used for Y rotations: e^{iπ/8 Y} = e^{−iπ/4 Z} e^{iπ/8 X} e^{iπ/4 Z}. The auto-corrected injection protocol removes the extra π/4 rotation at the cost of one more measured qubit.
- **Distillation (15-to-1):** 15 noisy magic states of infidelity p → 1 state of infidelity **35p³**; uses 4+11 injections on a compact block; yields a state in **11🕒 on 11 tiles**; failure is detected by X measurements on qubits 1–4; success probability ≈ (1−p)^n for an n-qubit protocol.
- **Worked examples** (d=13, 1 µs cycle, 10⁸ T gates): *minimal* = 100 logical qubits on 153 tiles + 11 distillation tiles (164 tiles ≈ 55k physical qubits), 11🕒 per T gate ⇒ d·11·10⁸ µs ≈ 4 h, distillation is the bottleneck. *Intermediate* = add a second distillation block (T every 5.5🕒) + intermediate data block (≤5🕒): 222 tiles ≈ 75k qubits ⇒ ≈ 2 h. Others: ≈22 min with ~120k qubits; ≈1 s with 1500 interconnected computers of ~220k qubits.
- Points to **§4 and §5 of the paper** for advanced techniques (parallel distillation, T-layer / time-optimal schemes).

---

## 3. QEC foundations from the workshop (for the "Methods" score)

- **NISQ:** 10s–1000s of noisy qubits, finite coherence; scaling qubit count alone doesn't create reliability.
- **Errors:** X (bit), Z (phase), Y; coherent (over/under-rotation) vs incoherent (relaxation/dephasing); readout/reset errors; leakage, crosstalk, correlated errors. Pauli basis lets QEC discretize continuous noise into syndromes.
- **Classical repetition code:** 0→000, 1→111, majority vote; corrects t = ⌊(d−1)/2⌋ errors. Quantum adds: no-cloning, both X and Z errors, measurement collapse → encode nonlocally in entangled qubits, measure **stabilizers** via ancillas.
- **Stabilizer:** S|ψ⟩ = +|ψ⟩; an error E flips the sign (SE|ψ⟩ = −E|ψ⟩).
- **Surface-17 (distance 3):** 9 data + 8 ancilla qubits.
  - Z-stabilizers: `[0,3] [1,2,4,5] [3,4,6,7] [5,8]`
  - X-stabilizers: `[0,1,3,4] [1,2] [4,5,7,8] [6,7]`
  - Qiskit returns bitstrings little-endian (bit 0 = rightmost).
  - Prepared |0⟩_L is a +1 eigenstate of all Z-stabilizers, so a Z error on it is invisible; X errors give a syndrome with degeneracy (e.g. qubits 1 or 2 share a Z-syndrome).
  - Decoding: lookup table for d=3; MWPM is the dominant general decoder.
- **Hardware extension:** IBM uses heavy-hex lattices (max 3 neighbours), which constrains planar surface codes.

### 3b. Details from `Introduction to Quantum Error Correction with Surface Codes.ipynb` (reusable code and known caveats)
- **Structure:** §1 repetition → 2-qubit toy detection code (encode α|00⟩+β|11⟩, Z₁Z₂ stabilizer with ancilla, inject X₁/X₂); §2 stabilizer formalism (codespace, rank r, k = n − r logical qubits; 3-qubit bit-flip code S=⟨Z₀Z₁, Z₁Z₂⟩ with syndromes X₀⇒[−1,+1], X₁⇒[−1,−1], X₂⇒[+1,−1]); §3 surface code + Surface-17 + decoding; §4 real hardware; extension on heavy-hex.
- **Surface-17 functions to reuse:** `initialize_surface_code()` (9 data `data`, 8 ancilla `anc` where anc[0–3] are Z-checks and anc[4–7] X-checks, 8-bit classical register `s`), `measure_z_stabilizer` / `measure_x_stabilizer` (reset ancilla, CNOT/CZ pattern, measure), `measure_all_stabilizers`, `run_surface_code(logical_state, add_error, error_type, error_location, shots)` using `AerSimulator(method='statevector')` and `transpile(..., optimization_level=3)`.
- **Known simplifications (do not overclaim):** `prepare_logical_zero` is only an *approximation* — it applies X on data qubits 0, 3, 6 and is a +1 eigenstate of the **Z-type** stabilizers only, not the X-type ones. Logical X_L is X on qubits 1, 4, 7 (vertical path). A single round of stabilizer measurement is used; true fault tolerance needs multiple rounds. A Z error on a Z-basis state gives the *same* syndrome as no error (it commutes with every Z-check). The notebook's `prepare_logical_one` calls `prepare_logical_zero` (the approximate state) first.
- **Decoding:** lookup table built by injecting each single X error and keeping `full_syndrome[-4:]` (Z-syndrome). Degeneracy: syndrome `0010` ⇒ qubit 1 or 2; `0100` ⇒ qubit 6 or 7. MWPM treats flipped stabilizers as defects and pairs them with minimum-weight chains; libraries: **PyMatching, Stim, Fusion Blossom**. MWPM returns a correction up to logical equivalence, not the exact error.
- **Open exercises in the notebook** (good, cheap proof of Methods): (1) `correct_bit` for the classical repetition code; (2) a function that reverses an injected error (apply the same Pauli on the decoded qubit); (3) the Z/Y-error lookup table (only the X table is built); (4) a proper |0⟩_L preparation that is stabilized by the X checks too; (5) try X errors on both qubits of the 2-qubit hardware circuit and predict the outcome.
- **Hardware section (§4):** reduce to 2 Z and 2 X stabilizers or the 2-qubit ZZ circuit (CX data→ancilla, measure) because full Surface-17 depth exceeds coherence on current devices. `QiskitRuntimeService`, `SamplerV2`, `transpile(qc, backend, optimization_level=1)`. The old `save_account(..., channel='ibm_quantum')` / `Options` / `ibm_brisbane` snippets are legacy: use the Fall Fest 101 pattern (`save_account(token, instance=CRN, ...)`, `service.least_busy()`).
- **Heavy-hex extension:** max 3 connections per qubit ⇒ planar/rotated surface codes need elongated ancilla chains or non-local CNOT sequences; Bacon-Shor (low-weight gauge checks) and subsystem surface codes fit better; rotated surface codes (Surface-17) are a middle ground.
- **Reading references:** Tomita & Svore (arXiv:1404.3747), Roffe (arXiv:1907.11157), Fowler et al. (arXiv:1208.0928), Preskill (arXiv:1801.00862).

---

## 4. Qiskit and tooling facts (for the 10% Qiskit score and to avoid wasted time)

- Pin **Qiskit ≥ 2.5** (`qiskit[visualization]>=2.5.0 qiskit-aer qiskit-ibm-runtime matplotlib numpy`; labs also use `qc-grader`). Python ≥ 3.10 (3.12 recommended). Challenge notebooks run on Google Colab or qBraid.
- **Qiskit Patterns:** Map → Optimize (transpile via `generate_preset_pass_manager`) → Execute → Post-process. Use `SamplerV2` / `EstimatorV2` from `qiskit_ibm_runtime`, `AerSimulator`, `SparsePauliOp`, `Statevector`, `plot_histogram`, `plot_state_qsphere`.
- **Dead APIs LLMs still write:** `execute()` (removed in Qiskit 1.0), `IBMQ`, `provider.get_backend('ibmq_qasm_simulator')`. Never use them.
- **Grounding ladder** (from `Qiskit 101 AI.pdf`): attach the Qiskit MCP servers (`pip install "qiskit-mcp-servers[all]"`: `qiskit-docs-mcp-server`, `qiskit-mcp-server`, `qiskit-ibm-runtime-mcp-server` [needs `QISKIT_IBM_TOKEN`], `qiskit-ibm-transpiler-mcp-server`); feed `quantum.cloud.ibm.com/llms.txt` or append `.md` to docs URLs; paste the migration guide; pin versions; run everything on simulators first. The deck says *"this is your time to integrate Qiskit into your AI agent"*, so documenting the use of the Docs MCP is itself good Qiskit/presentation material.
- **Real hardware:** `QiskitRuntimeService`, API key + instance CRN from quantum.cloud.ibm.com. `service.least_busy()` for QPU runs; use `AerSimulator.from_backend(backend)` for a noisy simulation to save QPU time. Credentials are the team's to supply; never commit them.

---

## 5. What a winning submission looks like

1. **Faithful game engine** whose operations and costs exactly match Section 2.
2. **Qiskit-verified**: the lattice-surgery logic is checked with real Qiskit circuits (e.g. Pauli-product measurement / Bell-pair protocols, Surface-17 syndrome extraction) and noisy simulation, not just asserted.
3. **A genuine insight** (30%): e.g. a scheduler that provably meets or beats the naive time N, a space-time trade-off curve across compact/intermediate/fast blocks, or a decoding/degeneracy observation tied to the layout.
4. **A tight story** (40%): problem → rules → method → result → why it works → limits, with figures.

## 6. Raw resource links (from the workshop docs)
- Litinski paper: arXiv:1808.02892 (abridged version, full paper, and YouTube talk are linked in the deck)
- IBM Quantum Learning: *Foundations of Quantum Error Correction*; Roffe, *QEC: An Introductory Guide* (arXiv:1907.11157); Fowler et al., *Surface codes: Towards practical large-scale quantum computation*; Preskill, *Quantum Computing in the NISQ era and beyond*; Tomita & Svore (arXiv:1404.3747)
- doQumentation, Quantum Country, QGSS26 lectures (Kevin Sung, Ibrahim Shehzad)
