# Qiskit cross-check of Surface-17 (`gosc/qiskit_check.py`)

Stack: Qiskit 2.6.2, qiskit-aer 0.18.2 (`AerSimulator`, mostly `method="stabilizer"`), `qiskit.quantum_info`, and PyMatching 2.x.
It uses no deprecated APIs: no `execute`, `IBMQ`, `BasicAer`, or V1 primitives. It also avoids `c_if` and uses `if_test` for feed-forward.
Tests are in `tests/test_qiskit_check.py` (30 tests, about 20 s, all seeded).

## What was verified

1. **Code definition.** The data qubits are 0..8 in row-major order. The checks are Z: `[0,3] [1,2,4,5] [3,4,6,7] [5,8]` and X: `[0,1,3,4] [1,2] [4,5,7,8] [6,7]`.
   - All 8 checks commute pairwise and are independent (rank 4 + 4, so k = 1).
   - The workshop logicals are correct. X_L = X{1,4,7} and Z_L = Z{0,1,2} commute with every check and anticommute with each other (they overlap only on qubit 1).
   - This was checked twice: once with numpy symplectic arithmetic and once with `qiskit.quantum_info.Pauli`.
2. **Proper |0>_L.** The circuit starts from |0…0>, measures the 4 X-checks, and conditionally applies Z on the pivot qubits 0, 2, 8, 6. Each pivot belongs only to its own X-check.
   - Aer `save_stabilizer` gives +1 for all 8 checks and for Z_L on every seed.
   - The `prep` outcomes vary from seed to seed, so the corrections are actually exercised.
   - A unitary encoder (H on the pivots, then CX fan-out) was also verified directly with `StabilizerState`.
3. **Syndromes.** All 27 single X/Y/Z errors were run through the Qiskit circuit on Aer. Each gives a deterministic syndrome equal to the symplectic one, `sz = HZ·x`, `sx = HX·z`. A Y error triggers both. For example, X4 gives `sz=(0,1,1,0)`.
   - **Bit order.** `get_counts` keys are little-endian, so bit 0 is the rightmost character. When there are several registers they are space-separated, and the register added first is the rightmost group. The keys therefore read `"prep s"`.
   - In `s`, bits s[0..3] are the Z-checks and s[4..7] are the X-checks. For example, X8 gives `s="00001000"`.
4. **Decoders.**
   - **Lookup tables.** Single errors reach 7 of the 15 nonzero syndromes for each error type. The other 8 are filled by brute-force minimum weight, so `decode` is total: every one of the 256 inputs returns a correction with matching syndrome.
   - **Degeneracies.** X1 and X2 share a syndrome, as do X6/X7, Z0/Z3 and Z5/Z8. Each pair's product is a stabilizer, so the degeneracy is benign.
   - **Lookup vs MWPM.** MWPM uses `pymatching.Matching(HZ)` for X errors and `Matching(HX)` for Z errors. Both decoders correct all 27 weight-1 errors (checked symplectically, and also in Aer by verifying the final state is +1 on all checks and Z_L).
   - On all 324 weight-2 errors, both decoders fail on the same 144 and never disagree. This is expected for d = 3, and neither decoder is "more right" on this set.
5. **Code-capacity logical error rate.** This is a vectorised Monte Carlo with a seeded RNG. Each qubit gets X, Y or Z with probability p/3 each. The residual syndrome is asserted to be 0, and a failure means the residual anticommutes with X_L or Z_L.
   - An exact enumeration over all 4⁹ Paulis gives P_L(p) directly.

   | p | MC (4e5 shots) | exact |
   |---|---|---|
   | 1e-3 | 1.0e-5 ± 5e-6 | 1.60e-5 |
   | 1e-2 | 1.47e-3 | 1.55e-3 |
   | 1e-1 | 0.113 | 0.114 |

   The exact log-log slope over p ∈ [1e-3, 1e-2] is **1.99**. The Monte Carlo slope (3e6 shots per point) is tested to lie in [1.7, 2.4]. P_L ≈ 15.5·p² at small p. This is consistent with the 144 failing weight-2 errors (item 4): 144·(p/3)² = 16·p², reduced by the (1−p)⁷ factor and higher-order terms, so d = 3 behaves quadratically, as expected below threshold.
6. **Noisy Aer round.** One round includes the |0>_L prep and is transpiled with `generate_preset_pass_manager(optimization_level=0, basis_gates=["h","x","y","z","cx"])`.
   - Noise is 1q depolarizing (p/10), 2q depolarizing (p) and symmetric readout error (p).
   - The fraction of the all-zero syndrome falls as p grows: 1.00, 0.958, 0.879, 0.669 and 0.300 for p = 0, 1e-3, 3e-3, 1e-2 and 3e-2.
   - The basis is set explicitly because with `backend=sim`, Aer's target basis depends on which gates the noise model lists. With no noise, H became rz/sx; with noise it stayed h. The explicit basis keeps the circuit identical across the whole p-sweep, and every gate in it is covered by the noise model.
7. **Bell pair by ZZ measurement.** The protocol is |++>, then measuring Z⊗Z through an ancilla. Both branches were reached by seeding `StabilizerState.measure`.
   - m = 0 gives (|00>+|11>)/√2 and m = 1 gives (|01>+|10>)/√2. This was checked with `StabilizerState.equiv` and with ⟨XX⟩ = +1, ⟨ZZ⟩ = ±1.
   - **Pauli-product measurement.** `pauli_product_measurement_circuit` uses the Qiskit label convention: the last character acts on qubit 0, and an optional `-` sign flips the outcome.
   - It was tested on product eigenstates and on the signed stabilizer generators (and one product of them) of random Cliffords. A +1 eigenstate gives outcome 0 with certainty, and the negated operator gives 1.
8. **`two_qubit_stabilizer_circuit(error_qubits)`.** This is the notebook's ZZ check, verified on Aer for errors ∅, {0}, {1} and {0,1}. Its docstring shows a SamplerV2 + `service.least_busy()` hardware run, which is not executed and needs no credentials.

## Discrepancies vs the workshop notebook

- **The notebook's `prepare_logical_zero` (X on {0,3,6}) is not |0>_L.**
  - It satisfies only the Z-checks. The X-check syndrome comes out random, and all 16 values appear on Aer.
  - X{0,3,6} anticommutes with Z_L, and X{0,3,6}·X_L equals X-check0 · X-check3. So X{0,3,6} is a logical-X representative.
  - Once X-projected, the notebook's "|0>_L" is therefore actually |1>_L, and its "|1>_L" (an extra X{1,4,7}) is |0>_L.
- The notebook's 2-qubit cell comments say "CNOT from ancilla", but the gates are data→ancilla CNOTs.
- The notebook's hardware cell already calls `Sampler` in the V2 style (`Sampler(mode=backend)`, PUB results via `join_data()`). However, it also imports the obsolete `Options` and hard-codes `ibm_brisbane`. The docstring here uses SamplerV2 + `least_busy`.
- The notebook's lookup table only covers single-error syndromes. Our decoder is total.

## Remaining approximations and fault-tolerance caveats

- The logical-rate study uses **code-capacity noise only**: errors on data qubits, perfect syndrome extraction, and a single round. It is not a circuit-level or phenomenological threshold. A d = 3 code cannot show a threshold crossing by itself; only the O(p²) scaling is shown.
- The noisy Aer run is **one round** with no repetition and no decoding over time. Noise also hits the preparation, so a non-zero syndrome mixes prep errors, data errors and readout flips. The CX ordering is the naive notebook ordering, which is not hook-error-safe. Measurement errors would need d rounds plus a space-time decoder.
- The |0>_L prep relies on the measured X-check outcomes being correct (feed-forward). On a noisy device it is not fault-tolerant.
