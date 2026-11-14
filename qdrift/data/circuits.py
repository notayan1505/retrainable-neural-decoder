"""Rebuild the Stein et al. repetition-code experiment in Qiskit.

New runs (fake backend now, real IBM later) land on disk in the same folder
format that ``qdrift.data.stein.parse_snapshot`` reads.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3, transpile


# --------------------------------------------------------------------------- chains
def _edges_of(coupling) -> list[tuple[int, int]]:
    if hasattr(coupling, "get_edges"):
        return [(int(a), int(b)) for a, b in coupling.get_edges()]
    return [(int(a), int(b)) for a, b in coupling]


def find_chains(coupling_edges, d: int, n_chains: int, avoid: set[int] = frozenset(),
                seed: int = 0) -> list[list[int]]:
    """Disjoint simple paths of 2d-1 qubits, preferring chains with no coupling between them."""
    length = 2 * d - 1
    adj: dict[int, set[int]] = {}
    for a, b in _edges_of(coupling_edges):
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    avoid = set(avoid)
    rng = random.Random(seed)
    nodes = sorted(n for n in adj if n not in avoid)

    def grow(path, blocked):
        if len(path) == length:
            return path
        nxt = [n for n in adj[path[-1]] if n not in blocked and n not in path]
        rng.shuffle(nxt)
        for n in nxt:
            path.append(n)
            res = grow(path, blocked)
            if res:
                return res
            path.pop()
        return None

    def greedy(strict: bool, chains: list[list[int]]) -> list[list[int]]:
        used = set(c for ch in chains for c in ch)
        halo = set(used)
        if strict:
            for q in used:
                halo |= adj[q]
        order = nodes[:]
        rng.shuffle(order)
        for s in order:
            if len(chains) >= n_chains:
                break
            if s in used or s in halo:
                continue
            blocked = set(avoid) | used | (halo if strict else set())
            p = grow([s], blocked)
            if p:
                chains.append(list(p))
                used |= set(p)
                halo |= set(p)
                if strict:
                    for q in p:
                        halo |= adj[q]
        return chains

    best: list[list[int]] = []
    for strict_only in (True, False):  # prefer non-adjacent chains; relax if too few fit
        for _ in range(30):
            cand = greedy(strict_only, [])
            if len(cand) > len(best):
                best = cand
            if len(best) >= n_chains:
                break
        if len(best) >= n_chains:
            break
    chains = best
    if not chains:
        raise ValueError(f"no chain of {length} qubits fits on the coupling graph (d={d})")
    return chains[:n_chains]


# --------------------------------------------------------------------------- circuit
def _suffix(chain) -> str:
    return "_".join(str(int(q)) for q in chain)


def build_circuit(chains: list[list[int]], r: int, basis: str = "Z", state: int = 0,
                  n_qubits: int | None = None) -> QuantumCircuit:
    """All chains in parallel on one physical-qubit circuit (qubit index = physical qubit)."""
    basis = basis.upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if state not in (0, 1):
        raise ValueError("state must be 0 or 1")
    maxq = max(q for ch in chains for q in ch)
    n = max(n_qubits or 0, maxq + 1)
    qr = QuantumRegister(n, "q")
    cregs = []
    for ch in chains:
        d = (len(ch) + 1) // 2
        cregs.append((ClassicalRegister(d, "c_data_" + _suffix(ch)),
                      ClassicalRegister(r * (d - 1), "c_syndrome_" + _suffix(ch))))
    qc = QuantumCircuit(qr, *[c for pair in cregs for c in pair])
    datas = [list(ch[0::2]) for ch in chains]
    checks = [list(ch[1::2]) for ch in chains]
    alld = [q for ds in datas for q in ds]
    if state:
        for q in alld:
            qc.x(q)
    if basis == "X":
        for q in alld:
            qc.h(q)
    for t in range(r):
        if basis == "X":  # rotate to the Z frame so the check measures XX parity
            for q in alld:
                qc.h(q)
        for (cd, cs), ds, cks in zip(cregs, datas, checks):
            d = len(ds)
            for i, a in enumerate(cks):
                qc.h(a)
                qc.cz(a, ds[i])
                qc.cz(a, ds[i + 1])
                qc.h(a)
                qc.measure(a, cs[t * (d - 1) + i])
                qc.reset(a)
        if basis == "X":
            for q in alld:
                qc.h(q)
    if basis == "X":
        for q in alld:
            qc.h(q)
    for (cd, cs), ds in zip(cregs, datas):
        for k, q in enumerate(ds):
            qc.measure(q, cd[k])
    return qc


# --------------------------------------------------------------------------- results
def bitarray_to_bits(ba) -> np.ndarray:
    """BitArray -> (shots, width) uint8, index 0 = classical bit 0 (little-endian)."""
    arr = np.asarray(ba.array, dtype=np.uint8)
    bits = np.unpackbits(arr, axis=1)[:, arr.shape[1] * 8 - ba.num_bits:]
    return np.ascontiguousarray(bits[:, ::-1])


def per_shot_cregs(circuit: QuantumCircuit, pub_data) -> dict[str, list]:
    out = {}
    for creg in circuit.cregs:
        out[creg.name] = bitarray_to_bits(pub_data[creg.name]).astype(int).tolist()
    return out


# --------------------------------------------------------------------------- backends
_FAKES = {"ibm_fez": "FakeFez", "ibm_kingston": "FakeKingston", "ibm_pittsburgh": "FakePittsburgh"}


def fake_backend(name: str = "ibm_fez", method: str = "matrix_product_state"):
    """Noisy Aer simulator of a fake IBM device. MPS keeps ~20-60 active qubits fast."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider

    key = name.lower()
    if key not in _FAKES:
        raise ValueError(f"unknown fake backend {name!r}; choose from {sorted(_FAKES)}")
    fake = getattr(fake_provider, _FAKES[key])()
    sim = AerSimulator.from_backend(fake, method=method)
    sim._qdrift_source = fake
    sim._qdrift_name = key
    return sim


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (_dt.datetime, _dt.date)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _calibration(backend, name: str) -> dict:
    cal: dict = {"backend_name": name}
    src = getattr(backend, "_qdrift_source", backend)
    try:
        props = src.properties()
        if props is not None:
            cal.update(_jsonable(props.to_dict()))
    except Exception:
        pass
    cal["backend_name"] = name
    cal["last_update_date"] = cal.get("last_update_date") or _dt.datetime.now(_dt.timezone.utc).isoformat()
    if "coupling_map" not in cal:
        try:
            cmap = src.coupling_map
            cal["coupling_map"] = json.dumps([[int(a), int(b)] for a, b in cmap.get_edges()])
        except Exception:
            pass
    return cal


def _coupling(backend):
    src = getattr(backend, "_qdrift_source", backend)
    cm = getattr(src, "coupling_map", None)
    if cm is None:
        cm = backend.coupling_map
    return cm


# --------------------------------------------------------------------------- run
def run_snapshot(backend, out_dir, d: int, r: int, basis: str = "Z", shots: int = 1024,
                 n_chains: int = 4, seed: int = 0, chains=None, backend_name: str | None = None) -> Path:
    from qiskit_ibm_runtime import SamplerV2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = backend_name or getattr(backend, "_qdrift_name", None) or getattr(backend, "name", "backend")
    if chains is None:
        chains = find_chains(_coupling(backend), d, n_chains, seed=seed)
    chains = [list(map(int, c)) for c in chains]
    nq = backend.num_qubits
    circs = [build_circuit(chains, r, basis, s, n_qubits=nq) for s in (0, 1)]
    for s, c in enumerate(circs):
        c.name = f"circuit_state{s}"
    used = sorted({q for ch in chains for q in ch})
    tcs = transpile(circs, backend=backend, optimization_level=1,
                    initial_layout=list(range(nq)), seed_transpiler=seed)
    for tc in tcs:  # sanity: the chosen qubits are the ones used
        act = {tc.find_bit(q).index for inst in tc.data for q in inst.qubits}
        assert act <= set(range(nq)) and set(used) <= act

    # Pick the sampler explicitly. Never fall back from a real backend to a local simulation:
    # that would write simulated data under a hardware label.
    from qiskit_aer import AerSimulator
    if isinstance(backend, AerSimulator):
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        sampler = AerSamplerV2.from_backend(backend)
    else:
        sampler = SamplerV2(mode=backend)
    result = sampler.run(tcs, shots=shots).result()

    entries = []
    for s, (tc, pub) in enumerate(zip(tcs, result)):
        entries.append({
            "circuit_name": f"circuit_state{s}",
            "metadata": {"logical_state": s, "n_syndrome_rounds": r, "basis": basis.upper(), "D": d},
            "per_shot_cregs": per_shot_cregs(tc, pub.data),
        })
    (out / "bitstrings.json").write_text(json.dumps(entries))
    (out / "info.json").write_text(json.dumps({
        "backend": name, "d": d, "rounds": r, "basis": basis.upper(),
        "logical_states": [0, 1], "shots": shots, "n_chains": len(chains)}))
    (out / "calibration.json").write_text(json.dumps(_calibration(backend, name)))
    for s, tc in enumerate(tcs):
        (out / f"circuit_state{s}.qasm").write_text(qasm3.dumps(tc))
    return out


def _next_job(folder: Path) -> Path:
    n = 1
    while (folder / f"job_{n}").exists():
        n += 1
    return folder / f"job_{n}"


def run_ibm(backend_name: str, out_root, configs: list[tuple[int, int, str]], shots: int = 1024,
            n_chains: int = 4, real: bool = False, seed: int = 0,
            label: str | None = None) -> list[Path]:
    expected_label = f"{backend_name}_{'fresh' if real else 'fake'}"
    if label is not None and label != expected_label:
        raise ValueError(f"run labels are reserved: use {expected_label!r}; custom labels could mix "
                         "simulated or fresh data with historical datasets")
    label = expected_label
    if real:
        token = os.environ.get("QISKIT_IBM_TOKEN")
        if not token:
            raise RuntimeError("QISKIT_IBM_TOKEN is not set; refusing to contact IBM Quantum. "
                               "Export your API token as an environment variable first.")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend = service.backend(backend_name)
        folder_name = info_name = label or f"{backend_name}_fresh"
    else:
        backend = fake_backend(backend_name)
        folder_name = info_name = label or f"{backend_name}_fake"
    print(f"run_ibm({info_name}): {len(configs)} snapshots x 2 circuits = {2 * len(configs)} circuits, "
          f"{shots} shots each = {2 * len(configs) * shots} shots total (up to {n_chains} chains each)")
    paths = []
    for d, r, basis in configs:
        job = _next_job(Path(out_root) / folder_name / f"d{d}_r{r}")
        paths.append(run_snapshot(backend, job, d, r, basis, shots=shots, n_chains=n_chains,
                                  seed=seed, backend_name=info_name))
    return paths
