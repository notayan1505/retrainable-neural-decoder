"""Stim ground truth mirroring the hardware repetition-code layout (used by tests)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stim

from qdrift.data.stein import Dataset, detectors_from_raw


def repetition_circuit(d, r, p_data=0.02, p_meas=0.02, p_reset=0.0, state=0) -> stim.Circuit:
    """Qubits 0..2d-2 on a line; data at even, checks at odd. Measure then reset each round."""
    data = list(range(0, 2 * d - 1, 2))
    checks = list(range(1, 2 * d - 1, 2))
    c = stim.Circuit()
    c.append("R", range(2 * d - 1))
    if state == 1:
        c.append("X", data)
    n_chk = d - 1
    total = r * n_chk + d
    for _ in range(r):
        if p_data:
            c.append("X_ERROR", data, p_data)
        pairs = []
        for i, a in enumerate(checks):
            pairs += [data[i], a, data[i + 1], a]
        c.append("CX", pairs)
        if p_meas:
            c.append("X_ERROR", checks, p_meas)
        c.append("M", checks)
        c.append("R", checks)
        if p_reset:
            c.append("X_ERROR", checks, p_reset)
    if p_meas:
        c.append("X_ERROR", data, p_meas)
    c.append("M", data)

    def rec(idx):
        return stim.target_rec(idx - total)

    for t in range(r + 1):
        for i in range(n_chk):
            if t == 0:
                ts = [rec(i)]
            elif t < r:
                ts = [rec(t * n_chk + i), rec((t - 1) * n_chk + i)]
            else:
                ts = [rec(r * n_chk + i), rec(r * n_chk + i + 1), rec((r - 1) * n_chk + i)]
            c.append("DETECTOR", ts)
    c.append("OBSERVABLE_INCLUDE", [rec(r * n_chk)], 0)
    return c


def _sample_meas(circuit, n, seed) -> np.ndarray:
    return circuit.compile_sampler(seed=seed).sample(n).astype(np.uint8)


def sample_raw(d, r, n, state=0, seed=0, **noise):
    meas = _sample_meas(repetition_circuit(d, r, state=state, **noise), n, seed)
    k = r * (d - 1)
    return np.ascontiguousarray(meas[:, :k]), np.ascontiguousarray(meas[:, k:k + d])


def make_dataset(d, r, n_per_state=5000, seed=0, **noise) -> Dataset:
    dets, labs, sts, dats = [], [], [], []
    for b in (0, 1):
        s, q = sample_raw(d, r, n_per_state, state=b, seed=seed * 2 + b, **noise)
        det, lab = detectors_from_raw(s, q, b)
        dets.append(det)
        labs.append(lab)
        dats.append(q)
        sts.append(np.full(n_per_state, b, dtype=np.uint8))
    n = 2 * n_per_state
    meta = {
        "snapshot": f"synthetic/d{d}_r{r}/job_0", "backend": "synthetic", "d": d, "r": r,
        "basis": "Z", "date": "", "shots": n_per_state, "n_chains": 1,
        "chains": [list(range(2 * d - 1))],
    }
    return Dataset(
        np.concatenate(dets), np.concatenate(labs), np.concatenate(sts), np.concatenate(dats),
        np.zeros(n, dtype=np.int16), np.zeros(n, dtype=np.int32), [meta],
    )


def write_stein_job(job_dir, d, r, shots=200, n_chains=3, seed=0, basis="Z",
                    backend="synthetic", **noise) -> Path:
    job = Path(job_dir)
    job.mkdir(parents=True, exist_ok=True)
    width = 2 * d - 1
    entries = []
    for b in (0, 1):
        regs = {}
        for c in range(n_chains):
            suffix = "_".join(str(c * width + j) for j in range(width))
            s, q = sample_raw(d, r, shots, state=b, seed=(seed * 2 + b) * 1000 + c, **noise)
            regs[f"c_data_{suffix}"] = q.tolist()
            regs[f"c_syndrome_{suffix}"] = s.tolist()
        entries.append({
            "circuit_name": f"circuit-{b}",
            "metadata": {"logical_state": b, "n_syndrome_rounds": r, "basis": basis},
            "per_shot_cregs": regs,
        })
    info = {"backend": backend, "d": d, "rounds": r, "basis": basis, "logical_states": [0, 1],
            "shots": shots, "n_chains": n_chains}
    (job / "info.json").write_text(json.dumps(info))
    (job / "bitstrings.json").write_text(json.dumps(entries))
    (job / "calibration.json").write_text(
        json.dumps({"backend_name": backend, "last_update_date": "2025-01-02T03:04:05+00:00"})
    )
    return job
