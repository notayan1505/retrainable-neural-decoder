import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from qdrift.data import stein
from qdrift.data.stein import (
    concat, detectors_from_raw, load_dataset, load_snapshot, parse_backend, parse_snapshot,
    save_snapshot, split,
)
from qdrift.data.synthetic import (
    make_dataset, repetition_circuit, sample_raw, write_stein_job,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("d,r", [(3, 1), (3, 3), (5, 2), (5, 5)])
@pytest.mark.parametrize("state", [0, 1])
def test_matches_stim(d, r, state):
    c = repetition_circuit(d, r, state=state)
    meas = c.compile_sampler(seed=3).sample(400)
    dets_ref, obs_ref = c.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True
    )
    s, q = sample_raw(d, r, 400, state=state, seed=3)
    k = r * (d - 1)
    assert np.array_equal(s, meas[:, :k]) and np.array_equal(q, meas[:, k:])
    det, lab = detectors_from_raw(s, q, state)
    assert det.shape == (400, r + 1, d - 1)
    assert np.array_equal(det.reshape(400, -1).astype(bool), dets_ref)
    assert np.array_equal(lab.astype(bool), obs_ref[:, 0])
    assert det.any()  # noise actually produced events


@pytest.mark.parametrize("state", [0, 1])
def test_noise_free(state):
    s, q = sample_raw(5, 4, 50, state=state, p_data=0, p_meas=0)
    det, lab = detectors_from_raw(s, q, state)
    assert not det.any() and not lab.any()
    ds = make_dataset(3, 3, 20, p_data=0, p_meas=0)
    assert not ds.detectors.any() and not ds.labels.any()


def test_hand_example():
    syn = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    q = np.array([[1, 1, 0]], dtype=np.uint8)
    det, lab = detectors_from_raw(syn, q, 0)
    assert det.tolist() == [[[1, 0], [0, 1], [1, 0]]]
    assert lab.tolist() == [1]
    det, lab = detectors_from_raw(syn, q, np.array([1]))
    assert lab.tolist() == [0]


def test_bad_input():
    q = np.zeros((2, 3), np.uint8)
    with pytest.raises(ValueError):
        detectors_from_raw(np.zeros((2, 3), np.uint8), q, 0)
    with pytest.raises(ValueError):
        detectors_from_raw(np.full((2, 4), 2, np.uint8), q, 0)


def test_roundtrip_and_parse(tmp_path):
    job = write_stein_job(tmp_path / "synthetic" / "d3_r2" / "job_5", 3, 2, shots=20, n_chains=3)
    ds = parse_snapshot(job)
    assert len(ds) == 2 * 20 * 3 and ds.d == 3 and ds.r == 2
    assert ds.detectors.shape == (120, 3, 2) and ds.data.shape == (120, 3)
    assert ds.states.tolist() == [0] * 60 + [1] * 60
    assert ds.chain[:60].tolist() == np.repeat([0, 1, 2], 20).tolist()
    m = ds.metas[0]
    assert m["snapshot"] == "synthetic/d3_r2/job_5" and m["date"] == "2025-01-02"
    assert m["basis"] == "Z" and m["n_chains"] == 3 and m["shots"] == 20
    assert m["chains"][1] == [5, 6, 7, 8, 9]
    f = tmp_path / "x" / "a.npz"
    save_snapshot(ds, f)
    ds2 = load_snapshot(f)
    for k in ("detectors", "labels", "states", "data", "chain", "snapshot"):
        assert np.array_equal(getattr(ds, k), getattr(ds2, k))
    assert ds2.metas == ds.metas
    # corrupted value
    import json
    p = job / "bitstrings.json"
    e = json.loads(p.read_text())
    key = next(iter(e[0]["per_shot_cregs"]))
    e[0]["per_shot_cregs"][key][0][0] = 2
    p.write_text(json.dumps(e))
    with pytest.raises(ValueError):
        parse_snapshot(job)
    e[0]["per_shot_cregs"][key][0] = [0]
    p.write_text(json.dumps(e))
    with pytest.raises(ValueError):
        parse_snapshot(job)


@pytest.mark.parametrize("workers", [1, 2])
def test_parse_backend_and_load(tmp_path, workers):
    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    base = raw / "synthetic" / "synthetic"
    write_stein_job(base / "d3_r3" / "job_0", 3, 3, shots=30, n_chains=2, seed=1)
    write_stein_job(base / "d3_r3" / "job_1", 3, 3, shots=30, n_chains=2, seed=2, basis="X")
    outs = parse_backend("synthetic", raw, parsed, workers=workers)
    assert len(outs) == 2 and all(o.exists() for o in outs)
    assert len(parse_backend("synthetic", raw, parsed, workers=workers)) == 2  # cached
    ds = load_dataset("synthetic", 3, 3, parsed_dir=parsed)
    assert len(ds) == 2 * 120 and [m["snapshot"] for m in ds.metas] == [
        "synthetic/d3_r3/job_0", "synthetic/d3_r3/job_1"]
    assert set(ds.snapshot.tolist()) == {0, 1}
    assert len(load_dataset(["synthetic"], 3, 3, basis="X", parsed_dir=parsed)) == 120
    with pytest.raises(ValueError, match="qdrift"):
        load_dataset("synthetic", 5, 3, parsed_dir=parsed)
    with pytest.raises(FileNotFoundError, match="qdrift"):
        load_dataset("synthetic", 3, 3, parsed_dir=tmp_path / "none")


def test_split(tmp_path):
    a = make_dataset(3, 2, 100, seed=1)
    b = make_dataset(3, 2, 60, seed=2)
    b.metas[0] = dict(b.metas[0], snapshot="synthetic/d3_r2/job_1")
    both = concat([a, b])
    tr, te = split(both, 0.25, seed=4)
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(len(both)))
    assert len(set(tr) & set(te)) == 0
    assert len(te) == 2 * 25 + 2 * 15
    for idx in (tr, te):
        assert set(both.states[idx].tolist()) == {0, 1}
    tr2, te2 = split(both, 0.25, seed=4)
    assert np.array_equal(te, te2) and np.array_equal(tr, tr2)
    _, te_a = split(a, 0.25, seed=4)
    assert np.array_equal(te[te < len(a)], te_a)
    assert len(both.subset(te)) == len(te) and both.subset(te).metas is both.metas

    # Snapshot holdout never places samples from one calibration session in both sets.
    snap_tr, snap_te = split(both, 0.5, seed=4, strategy="snapshot")
    assert set(both.snapshot[snap_tr]).isdisjoint(set(both.snapshot[snap_te]))
    assert set(both.snapshot[snap_tr]) | set(both.snapshot[snap_te]) == {0, 1}

    # Chain holdout keeps every shot from a physical chain (and both states) together.
    c1 = stein.parse_snapshot(write_stein_job(
        tmp_path / "chain_a" / "d3_r2" / "job_0", 3, 2, shots=20, n_chains=4, seed=10
    ))
    c2 = stein.parse_snapshot(write_stein_job(
        tmp_path / "chain_b" / "d3_r2" / "job_0", 3, 2, shots=20, n_chains=4, seed=11
    ))
    chained = concat([c1, c2])
    chain_tr, chain_te = split(chained, 0.25, seed=2, strategy="chain")
    for snap in np.unique(chained.snapshot):
        tr_chains = set(chained.chain[chain_tr[chained.snapshot[chain_tr] == snap]])
        te_chains = set(chained.chain[chain_te[chained.snapshot[chain_te] == snap]])
        assert tr_chains.isdisjoint(te_chains)
        assert tr_chains | te_chains == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="at least two chains"):
        split(a, 0.25, seed=0, strategy="chain")


def _make_tar(path, add):
    with tarfile.open(path, "w:gz") as tf:
        add(tf)


def test_safe_extract(tmp_path):
    def evil(tf):
        ti = tarfile.TarInfo("../evil")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))

    def link(tf):
        ti = tarfile.TarInfo("ok/link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)

    def good(tf):
        ti = tarfile.TarInfo("ok/f.txt")
        ti.size = 2
        tf.addfile(ti, io.BytesIO(b"hi"))

    for i, fn in enumerate((evil, link)):
        t = tmp_path / f"t{i}.tar.gz"
        _make_tar(t, fn)
        with pytest.raises(ValueError):
            stein._safe_extract(t, tmp_path / f"out{i}")
    assert not (tmp_path / "evil").exists()
    t = tmp_path / "good.tar.gz"
    _make_tar(t, good)
    stein._safe_extract(t, tmp_path / "outg")
    assert (tmp_path / "outg" / "ok" / "f.txt").read_text() == "hi"


REAL = ROOT / "data/raw/ibm_fez/ibm_fez/d3_r3/job_2"


@pytest.mark.skipif(not REAL.exists(), reason="real ibm_fez data not downloaded")
def test_real_job():
    ds = parse_snapshot(REAL)
    assert len(ds) == 2 * 512 * 10
    assert ds.detectors.shape == (len(ds), 4, 2)
    assert 0.02 < ds.detectors.mean() < 0.4
