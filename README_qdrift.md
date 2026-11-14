# qdrift: neural decoders for repetition codes on IBM hardware

qdrift trains small neural networks to decode repetition-code experiments run on real IBM quantum computers, and compares them with classic decoders (matching, lookup table). The data comes from Stein et al., Zenodo record 10.5281/zenodo.20768087, licensed CC BY 4.0. We are not the first to train decoders on this data; this is our own open, reproducible pipeline.

Jargon, defined once:
- **Repetition code**: a chain of qubits that stores one logical bit. Distance `d` is the number of data qubits.
- **Round (`r`)**: one cycle of parity checks (syndrome measurements).
- **Detector**: a check result that changed between rounds. A detector firing means something probably went wrong nearby.
- **Decoder**: a program that reads the detectors and guesses whether the stored bit flipped.
- **p_L**: logical error rate, the share of wrong guesses. **eps/round**: the same, converted to an error per round.
- **MWPM**: minimum-weight perfect matching, the standard classic decoder.
- **Drift**: hardware noise changes over time, so a decoder trained earlier may get worse.

All commands run from the project root as `python -m qdrift <command>`.

## Setup on the RTX 5070 Ti laptop

Install torch FIRST (CUDA 12.8+ build), then the rest.

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Moving the project between machines
- Copy the code only: `qdrift/`, `tests/qdrift/`, `docs/`, `README_qdrift.md`, `requirements.txt`. Do not copy `.venv` (it is built for one operating system) and there is no need to copy `data/`; `download --all` and `parse` rebuild it on the new machine.
- Checkpoints (`results/qdrift/checkpoints/*.pt`) are portable in both directions. One trained on the GPU can be evaluated on a Mac or CPU: `evaluate` always uses the device of the machine it runs on.
- Run `pytest tests/qdrift -q` once after installing to confirm the setup.

The check should print `True` and the GPU name. If you see "no kernel image is available for execution on the device", your torch build is too old for the Blackwell GPU (RTX 50-series). Reinstall with the cu128 command above. On a Mac, plain `pip install torch` is enough.

## Commands, in the order you would run them

```bash
# 1. Download all three backends (about the size of the Zenodo files; parsed as JSON only)
python -m qdrift download --all

# 2. Turn the raw JSON into compact files (cached in data/parsed)
python -m qdrift parse

# 3. See what is available
python -m qdrift index

# 4. Classic decoders (every r for d=5; add --r 5 for one)
python -m qdrift baseline --backend ibm_fez --d 5

# 5. Train one neural decoder (saved in results/qdrift/checkpoints)
python -m qdrift train --backend ibm_fez --d 7 --r 5 --kind mlp

# 6. Several configs, three at a time (logs in results/qdrift/logs)
python -m qdrift sweep --backend ibm_fez --configs 7x5,9x7,11x11 --kinds mlp,conv --jobs 3
#    add --dry-run to only print the commands

# 7. Learning curve: error vs. amount of training data, NN against matching and lookup
python -m qdrift train --backend ibm_fez --d 7 --r 5 --learning-curve --fractions 0.01,0.03,0.1,0.3,1.0

# 8. Evaluate baselines plus any saved checkpoints; writes a CSV and a PNG
python -m qdrift evaluate --backend ibm_fez --configs 3x3,7x5,9x7

# 9. Generate fresh data on the FAKE (simulated) ibm_fez: safe, no account needed
python -m qdrift run-ibm --backend ibm_fez --configs 3x3,7x5 --shots 1024
```

### Drift test flow

Train on the reference data, then score on newly collected data.

Rehearsal on the simulated chip (no account needed; data is stored as `ibm_fez_fake`):
```bash
python -m qdrift run-ibm --backend ibm_fez --configs 7x5
python -m qdrift parse ibm_fez_fake
python -m qdrift evaluate --backend ibm_fez --configs 7x5 --basis Z --test-backend ibm_fez_fake
```
The real thing (needs `QISKIT_IBM_TOKEN`; data is stored as `ibm_fez_fresh`):
```bash
python -m qdrift run-ibm --backend ibm_fez --configs 7x5 --real
python -m qdrift parse ibm_fez_fresh
python -m qdrift evaluate --backend ibm_fez --configs 7x5 --basis Z --test-backend ibm_fez_fresh
```
`run-ibm` collects one basis per call (Z by default, `--basis X` for the other), so pass the same `--basis` to `evaluate`. Otherwise the old-data decoders are fitted on X and Z pooled and scored on Z only. The `_fresh` label is reserved for real runs; a simulated run refuses to write there. The simulated chip is much quieter than the real one, so its numbers are a plumbing check, not a forecast.

The table includes `mwpm_refit` and `lookup_refit`: classic decoders refitted on the new data. Compare them with the same decoders fitted on the old data to see the cost of drift.

Fine-tune a trained network on the new data:
```bash
python -m qdrift train --backend ibm_fez_fresh --d 7 --r 5 --init results/qdrift/checkpoints/ibm_fez_d7_r5_mlp.pt --out results/qdrift/checkpoints/ibm_fez_fresh_d7_r5_mlp.pt
```
(One line on purpose: the same command works in PowerShell and bash.)

## GPU settings
- The whole dataset is moved to the GPU once. Use big batches: 8k to 64k (`--batch-size`).
- On CUDA, bf16 autocast and TF32 are on. Turn off with `--no-amp`.
- `--compile` (torch.compile) is optional. On Windows it needs Triton; if it fails, training falls back to normal mode with a warning.
- `--jobs N` in `sweep` runs N trainings at once on the same GPU. Start with 3.
- `--device auto|cuda|mps|cpu` picks the device. The device and GPU name are printed at the start.

## Data facts
- Only odd `r <= d` exist in the data. For d=3 that is r = 1 and 3.
- The ibm_fez data was taken on 2025-08-25 and 2025-10-03.
- Logical state 1 is much noisier than state 0 (see `p_L|1` versus `p_L|0` in the tables).
- Detectors fire often: about 5 to 30 percent of the time.
- Train and test splits are fixed by `--seed` and `--test-frac`, per snapshot and state, so all decoders see the same test set.

## Tests
```bash
pytest tests/qdrift -q
```
Training tests use tiny synthetic data on the CPU and take under a minute.

## Safety
- Downloaded files are only read as JSON. Nothing from `data/raw` is ever imported or executed.
- `run-ibm` uses a fake simulated backend by default. `--real` needs `QISKIT_IBM_TOKEN` in your environment, prints the circuit and shot plan, and asks you to type `yes` (skip with `--yes`). Real runs go to `<backend>_fresh` unless you set `--label`, so they never overwrite the reference `ibm_fez` data.
