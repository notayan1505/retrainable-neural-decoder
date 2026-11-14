import math
from pathlib import Path

import numpy as np
import pymatching
import pytest

from qdrift.data.stein import detectors_from_raw, load_dataset, split
from qdrift.data.synthetic import make_dataset, repetition_circuit
from qdrift.decoders.baselines import (
    LookupTable, MatchingEqual, MatchingFitted, NoDecode, error_per_round, format_table,
    logical_error, make_baselines, run_baselines,
)

ROOT = Path(__file__).resolve().parents[2]
KEYS = {"decoder", "d", "r", "n_train", "n_test", "errors", "p_L", "stderr", "eps_round",
        "p_L_state0", "p_L_state1"}


def _split(ds):
    tr, te = split(ds)
    return ds.subset(tr), ds.subset(te)


@pytest.fixture(scope="module")
def sym():
    noise = dict(p_data=0.03, p_meas=0.03)
    return noise, *_split(make_dataset(5, 5, n_per_state=20000, seed=1, **noise))


@pytest.fixture(scope="module")
def fitted(sym):
    return MatchingFitted().fit(sym[1])


def test_matching_beats_none_and_matches_true_dem(sym, fitted):
    noise, train, test = sym
    rows = {row["decoder"]: row for row in run_baselines(train, test)}
    assert set(rows) == {"none", "mwpm_equal", "mwpm_fitted", "lookup"}
    for name in ("mwpm_equal", "mwpm_fitted"):
        assert rows[name]["p_L"] < rows["none"]["p_L"] / 5
    dem = repetition_circuit(5, 5, **noise).detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(test.detectors.reshape(len(test.labels), -1))[:, 0]
    ref = logical_error(pred, test.labels)
    tol = 3 * ref["stderr"] + 0.001
    assert rows["mwpm_fitted"]["p_L"] <= ref["p_L"] + tol


def test_asymmetric_noise_fitted_beats_equal():
    train, test = _split(make_dataset(5, 5, n_per_state=20000, seed=3, p_data=0.005, p_meas=0.08))
    eq = logical_error(MatchingEqual().fit(train).predict(test.detectors), test.labels)
    fi = logical_error(MatchingFitted().fit(train).predict(test.detectors), test.labels)
    assert fi["p_L"] + 2 * fi["stderr"] < eq["p_L"]


def test_fitted_edge_probs_close_to_truth(fitted):
    d, r = 5, 5
    ep = fitted.edge_probs
    for t in range(r):
        for i in range(d - 2):
            assert abs(ep[((t, i), (t, i + 1))] - 0.03) < 0.008  # data flips
        for i in range(d - 1):
            assert abs(ep[((t, i), (t + 1, i))] - 0.03) < 0.008  # measurement errors
            if i < d - 2:
                assert ep[((t, i), (t + 1, i + 1))] < 0.005
                assert ep[((t, i + 1), (t + 1, i))] < 0.005
    assert all(1e-6 <= p <= 0.5 for p in ep.values())


def test_single_faults(sym, fitted):
    _, train, _ = sym
    d, r = 5, 5
    decs = [MatchingEqual(d, r), fitted, LookupTable().fit(train)]
    syn, dat, expect = [], [], []

    def data_flip(q, t):  # data qubit q flipped before round t (state 0, no other noise)
        s = np.zeros((r, d - 1), np.uint8)
        for c in (q - 1, q):
            if 0 <= c < d - 1:
                s[t:, c] = 1
        x = np.zeros(d, np.uint8)
        x[q] = 1
        return s, x

    for t in range(r):
        for q, fired in ((0, [(t, 0)]), (2, [(t, 1), (t, 2)])):
            s, x = data_flip(q, t)
            syn.append(s.ravel()), dat.append(x), expect.append(fired)
        s = np.zeros((r, d - 1), np.uint8)
        s[t, 1] = 1  # measurement error on check 1 in round t
        syn.append(s.ravel()), dat.append(np.zeros(d, np.uint8)), expect.append([(t, 1), (t + 1, 1)])
    X, y = detectors_from_raw(np.stack(syn), np.stack(dat), 0)
    for k, fired in enumerate(expect):
        assert sorted(zip(*np.nonzero(X[k]))) == fired
    np.testing.assert_array_equal(y, [1, 0, 0] * r)
    for dec in decs:
        np.testing.assert_array_equal(dec.predict(X), y, err_msg=dec.name)


def test_empty_and_deterministic(fitted):
    empty = np.zeros((0, 6, 4), np.uint8)
    for dec in (NoDecode(), MatchingEqual(5, 5), fitted):
        out = dec.predict(empty)
        assert out.shape == (0,) and out.dtype == np.uint8
    X = np.random.default_rng(0).integers(0, 2, (500, 6, 4), dtype=np.uint8)
    np.testing.assert_array_equal(fitted.predict(X), fitted.predict(X))


def test_lookup_d3_r3():
    train, test = _split(make_dataset(3, 3, n_per_state=50000, seed=5, p_data=0.04, p_meas=0.06))
    lk = logical_error(LookupTable().fit(train).predict(test.detectors), test.labels)
    fi = logical_error(MatchingFitted().fit(train).predict(test.detectors), test.labels)
    assert lk["p_L"] <= fi["p_L"] + 2 * fi["stderr"]
    # tiny table: most patterns unseen -> fallback, no crash
    small = LookupTable().fit(train.subset(np.arange(50)))
    allpat = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8).reshape(256, 4, 2)
    out = small.predict(allpat)
    assert out.shape == (256,) and set(np.unique(out)) <= {0, 1}
    assert small.predict(np.zeros((0, 4, 2), np.uint8)).shape == (0,)


def test_make_baselines_sizes():
    assert [x.name for x in make_baselines(3, 3)][-1] == "lookup"
    assert "lookup" not in [x.name for x in make_baselines(7, 5)]


def test_metrics():
    assert error_per_round(0.1, 1) == pytest.approx(0.1)
    assert error_per_round(0.2, 3) == pytest.approx((1 - 0.6 ** (1 / 3)) / 2)
    assert error_per_round(0.5, 4) == 0.5 and error_per_round(0.7, 4) == 0.5
    le = logical_error(np.array([1, 0, 0, 0]), np.array([0, 0, 0, 0]))
    assert le == {"n": 4, "errors": 1, "p_L": 0.25, "stderr": pytest.approx(math.sqrt(0.25 * 0.75 / 4))}


def test_run_baselines_rows_and_table():
    train, test = _split(make_dataset(3, 3, n_per_state=2000, seed=7, p_data=0.03, p_meas=0.03))
    rows = run_baselines(train, test)
    assert [r["decoder"] for r in rows] == ["none", "mwpm_equal", "mwpm_fitted", "lookup"]
    for row in rows:
        assert set(row) == KEYS
    s = format_table(rows)
    assert isinstance(s, str) and all(r["decoder"] in s for r in rows)


@pytest.mark.skipif(not (ROOT / "data" / "parsed" / "ibm_fez").exists(), reason="no parsed ibm_fez")
def test_real_ibm_fez():
    ds = load_dataset("ibm_fez", 3, 3)
    tr, te = split(ds)
    rows = run_baselines(ds.subset(tr), ds.subset(te))
    assert all(0 <= r["p_L"] <= 0.5 for r in rows)


def test_wilson_interval():
    from qdrift.decoders.baselines import wilson_interval
    lo, hi = wilson_interval(0, 2048)
    assert lo == 0.0 and 0.0 < hi < 0.002  # zero errors still gets a non-zero upper bar
    lo, hi = wilson_interval(500, 1000)
    se = math.sqrt(0.25 / 1000)
    assert lo == pytest.approx(0.5 - se, rel=1e-3) and hi == pytest.approx(0.5 + se, rel=1e-3)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))
