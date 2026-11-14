import pytest

from qdrift.data.stein import load_dataset, parse_backend, split
from qdrift.data.synthetic import write_stein_job
from qdrift.evaluate import (baseline_learning_curve, evaluate, evaluate_config, plot_error_vs_d,
                             plot_learning_curve, write_csv)
from qdrift.train import learning_curve_subsets


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    root = tmp_path_factory.mktemp("ev")
    raw, parsed = root / "raw", root / "parsed"
    for i, (name, noise) in enumerate([("synthA", 0.02), ("synthB", 0.05)]):
        write_stein_job(raw / name / name / "d3_r3" / "job_1", 3, 3, shots=150, n_chains=3, seed=i,
                        backend=name, p_data=noise, p_meas=noise)
        parse_backend(name, raw, parsed, workers=1)
    return parsed


def _split(parsed, name="synthA"):
    ds = load_dataset(name, 3, 3, parsed_dir=parsed)
    tr, te = split(ds)
    return ds.subset(tr), ds.subset(te)


def test_evaluate_config_and_csv_plot(parsed, tmp_path):
    train, test = _split(parsed)
    rows = evaluate_config(train, test)
    assert {r["decoder"] for r in rows} == {"none", "mwpm_equal", "mwpm_fitted", "lookup"}
    p = write_csv(rows, tmp_path / "sub" / "x.csv")
    assert p.exists() and p.stat().st_size > 0
    png = plot_error_vs_d(rows, tmp_path / "e.png")
    assert png.stat().st_size > 0


def test_evaluate_drift(parsed):
    rows = evaluate("synthA", [(3, 3)], test_backend="synthB", parsed_dir=parsed)
    names = {(r["decoder"], r["test_backend"]) for r in rows}
    assert ("mwpm_refit", "synthB") in names and ("lookup_refit", "synthB") in names
    assert ("mwpm_fitted", "synthA") in names


def test_evaluate_drift_skips_missing(parsed, capsys):
    rows = evaluate("synthA", [(3, 3)], test_backend="nope", parsed_dir=parsed)
    assert all(r["test_backend"] == "synthA" for r in rows)
    assert "skipping" in capsys.readouterr().out


def test_baseline_learning_curve_matches_train_subsets(parsed, tmp_path):
    train, test = _split(parsed)
    fr = [0.5, 1.0]
    rows = baseline_learning_curve(train, test, fr, seed=0)
    expect = {f: len(idx) for f, idx in learning_curve_subsets(len(train.labels), fr, 0)}
    assert {r["fraction"]: r["n_train"] for r in rows} == expect
    assert {r["decoder"] for r in rows} >= {"mwpm_fitted"}
    assert plot_learning_curve(rows, tmp_path / "lc.png").stat().st_size > 0


def test_train_learning_curve_uses_same_subsets(parsed):
    from qdrift.train import TrainConfig, learning_curve
    train, test = _split(parsed)
    cfg = TrainConfig(kind="mlp", width=16, depth=1, epochs=1, device="cpu")
    nn_rows = learning_curve(train, test, cfg, [0.5, 1.0], log=lambda *_: None)
    b_rows = baseline_learning_curve(train, test, [0.5, 1.0], seed=cfg.seed)
    for f in (0.5, 1.0):
        assert {r["n_train"] for r in nn_rows if r["fraction"] == f} == \
               {r["n_train"] for r in b_rows if r["fraction"] == f}
