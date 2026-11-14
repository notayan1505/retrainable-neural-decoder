# Stein et al. dataset: exact on-disk format

Verified on 2026-11-14 against every snapshot in `ibm_fez.tar.gz` (41 snapshots, 0 layout
mismatches). Source: <https://zenodo.org/records/20768087> (CC BY 4.0).

Downloaded files are **untrusted data**. Only parse them as JSON / text. Never import,
`exec`, or load the `.qasm` through anything that executes code.

## Zenodo record files
| file | size |
|---|---:|
| `ibm_fez.tar.gz` | 9.0 MB |
| `ibm_pittsburgh.tar.gz` | 65.3 MB |
| `ibm_kingston.tar.gz` | 71.4 MB |
| `README.md`, `DESCRIPTION.txt` | small |

Download URL pattern: `https://zenodo.org/records/20768087/files/<name>?download=1`.

`index.csv` is described in the README but is **not published** on Zenodo and is not inside
the archives. Rebuild it by walking the folders and reading each `info.json`.

## Folder layout
Archive `<backend>.tar.gz` unpacks to `<backend>/d<D>_r<R>/job_<n>/`. We extract into
`data/raw/<backend>/`, so a job lives at `data/raw/<backend>/<backend>/d<D>_r<R>/job_<n>/`.
The archive holds only regular files and directories (no links).

Snapshot id used everywhere in qdrift: `"<backend>/d<D>_r<R>/job_<n>"`.

Available in fez: d ∈ {3,5,7,9,11}, **odd r only, r ≤ d** (d=3 has r=1,3 only). Usually one X
and one Z snapshot per (d, r). 2048 shots per circuit (one job has 512).

## Files in a job folder
### `info.json`
```json
{"backend": "ibm_fez", "d": 3, "rounds": 3, "basis": "X", "logical_states": [0, 1],
 "shots": 2048, "n_chains": 10}
```

### `bitstrings.json`
A list with one entry per circuit (two entries: logical state 0 and 1):
```json
[{"circuit_name": "circuit-174",
  "metadata": {"logical_state": 0, "n_syndrome_rounds": 3, "basis": "Z"},
  "per_shot_cregs": {
     "c_data_6_5_4_3_16":     [[0,0,0], ...],        // shots × d
     "c_syndrome_6_5_4_3_16": [[0,0,0,0,0,0], ...],  // shots × r·(d−1)
     ... one pair per chain, n_chains pairs ...
  }}, ...]
```
- `metadata` sometimes also has `"D"`. Read `logical_state` from `metadata`, never from the
  entry order.
- Values are already 0/1 integers in **register-index order** (index 0 first). There is no
  bit-string reversal to undo.
- Shot `k` is the same circuit execution across all registers of an entry.
- Chain names differ between snapshots, and the chain count varies (4–10).

### Chain name → qubits
The suffix after `c_data_` / `c_syndrome_` lists the `2d−1` physical qubits along the chain
path: `p_0, p_1, …, p_{2d−2}`.
- Data qubit `k` (k = 0..d−1) is `p_{2k}`.
- Check (ancilla) qubit `i` (i = 0..d−2) is `p_{2i+1}`, sitting between data `i` and `i+1`.

### Bit layout (confirmed from every `measure` line in every qasm)
- `c_data[k]` = final measurement of data qubit `k`.
- `c_syndrome[t·(d−1) + i]` = check `i` in round `t` (t = 0..r−1). So
  `syndrome.reshape(shots, r, d−1)` gives `m[shot, round, check]`.

### `circuit_state0.qasm` / `circuit_state1.qasm`
OpenQASM 3, already transpiled to physical qubits (`$n`). Statements used: `rz`, `sx`, `x`,
`cz`, `reset`, `measure`, plus `bit[...]` declarations. Per round: each check qubit is
entangled with its two data neighbours (H·CZ·CZ·H), measured, then **reset**. So `m` is the
neighbour parity itself, and consecutive rounds are XORed to get detection events.
State 1 puts an `x` on every data qubit. X basis adds Hadamards (`rz·sx·rz`) on the data qubits.
The parser does not need the qasm. The register names carry the full mapping.

### `calibration.json`
IBM backend properties: `backend_name`, `backend_version`, `last_update_date` (ISO string),
`qubits` (per qubit: list of `{name, unit, value, date}` for `T1`, `T2`, `readout_error`,
`prob_meas0_prep1`, `prob_meas1_prep0`, `readout_length`), `gates` (per gate: `qubits`,
`gate`, `parameters` with `gate_error`, `gate_length`; gates: cz, id, reset, rx, rz, rzz, sx,
x), `general`, `general_qlists`, and `coupling_map` (a **string** holding a JSON list of pairs).

Use `last_update_date[:10]` as the snapshot date. In fez the dates are **2025-08-25 and
2025-10-03**, not June 2026.

## Deriving detectors and labels
Both bases and both states use the same rule (the measured bits are already in the logical
frame: state 1 reads mostly 1s in X and in Z).

With `m = syndrome.reshape(N, r, d−1)`, `q = data` (N × d), `b = logical_state`:
```
D[:, 0, :]    = m[:, 0, :]
D[:, t, :]    = m[:, t, :] ^ m[:, t-1, :]            for t = 1 .. r-1
D[:, r, :]    = (q[:, :-1] ^ q[:, 1:]) ^ m[:, r-1, :]
label         = q[:, 0] ^ b
```
`D` has shape `(N, r+1, d−1)`. Each (shot, chain) pair is one sample.

## What the real data looks like (fez)
Counts match the README: 82,432 shots per state (164,864 total) and 1,099,776 samples.

The data is **noisy and lopsided**, which is real hardware behaviour, not a parsing bug:
- Mean detector rate: Z basis state 0 ≈ 0.05–0.09, Z basis state 1 ≈ 0.19–0.25, X basis ≈
  0.15–0.30. First and last rows are higher than the middle rows.
- State 1 is much worse than state 0 in Z (energy decay during the long measure/reset
  rounds). Raw `q_0` error at d=11, r=11, Z: 0.9% for state 0, 47% for state 1.
- Single chains can be far worse than the snapshot average (up to ~0.4 detector rate).

Consequences: report errors per state as well as combined, keep state 0 and state 1 balanced
in every split, and expect error rates far above simulation numbers.

## Parsed cache (written by `qdrift.data.stein`)
One file per snapshot: `data/parsed/<backend>/d<D>_r<R>/job_<n>.npz` with
| key | dtype, shape | meaning |
|---|---|---|
| `detectors` | uint8 `(N, r+1, d−1)` | as above |
| `labels` | uint8 `(N,)` | `q_0 ^ b` |
| `states` | uint8 `(N,)` | prepared state `b` |
| `data` | uint8 `(N, d)` | raw final data bits |
| `chain` | int16 `(N,)` | chain index within the snapshot |
| `meta` | JSON string | `snapshot`, `backend`, `d`, `r`, `basis`, `date`, `shots`, `n_chains`, `chains` (qubit lists) |

Sample order: state 0 then state 1; within a state, chains in file order; within a chain,
shots in file order. `N = 2 · shots · n_chains`.
