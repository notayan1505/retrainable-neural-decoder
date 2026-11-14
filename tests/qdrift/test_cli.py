import csv
from pathlib import Path

import pytest

from qdrift.cli import main
from qdrift.data.synthetic import write_stein_job


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    raw = root / "raw"
    for i, (name, noise) in enumerate([("synthA", 0.02), ("synthB", 0.05)]):
        write_stein_job(raw / name / name / "d3_r3" / "job_1", 3, 3, shots=150, n_chains=3, seed=i,
                        backend=name, p_data=noise, p_meas=noise)
    (raw / "_archives").mkdir()
    return root


@pytest.fixture(scope="module")
def parsed_tree(tree):
    assert main(["parse", "--raw-dir", str(tree / "raw"), "--parsed-dir", str(tree / "parsed"),
                 "--workers", "1"]) == 0
    return tree


def _args(t):
    return ["--parsed-dir", str(t / "parsed")]


def test_parse_index(parsed_tree, capsys):
    capsys.readouterr()
    assert main(["index", *_args(parsed_tree)]) == 0
    out = capsys.readouterr().out
    assert "synthA" in out and "synthB" in out and "n_samples" in out


def test_baseline(parsed_tree, capsys):
    out_csv = parsed_tree / "b.csv"
    assert main(["baseline", "--backend", "synthA", "--d", "3", *_args(parsed_tree), "--out", str(out_csv)]) == 0
    out = capsys.readouterr().out
    for n in ("none", "mwpm_equal", "mwpm_fitted", "lookup"):
        assert n in out
    assert out_csv.stat().st_size > 0


def test_train_eval_init_curve(parsed_tree, tmp_path, capsys):
    t = parsed_tree
    ck = tmp_path / "ck"
    common = ["--backend", "synthA", "--d", "3", "--r", "3", *_args(t), "--device", "cpu", "--epochs", "2",
              "--width", "16", "--depth", "1", "--batch-size", "256"]
    assert main(["train", *common, "--out", str(ck / "synthA_d3_r3_mlp.pt")]) == 0
    assert (ck / "synthA_d3_r3_mlp.pt").exists()
    capsys.readouterr()
    od = tmp_path / "ev"
    assert main(["evaluate", "--backend", "synthA", "--configs", "3x3", *_args(t), "--ckpt-dir", str(ck),
                 "--out-dir", str(od), "--device", "cpu"]) == 0
    assert "nn_mlp" in capsys.readouterr().out
    assert main(["evaluate", "--backend", "synthA", "--configs", "3x3", *_args(t), "--ckpt-dir", str(ck),
                 "--test-backend", "synthB", "--out-dir", str(od), "--device", "cpu"]) == 0
    rows = list(csv.DictReader(open(od / "eval_synthA_on_synthB.csv")))
    assert any(r["decoder"] == "mwpm_refit" for r in rows)
    assert (od / "eval_synthA_on_synthB.png").stat().st_size > 0
    assert (od / "eval_synthA.png").stat().st_size > 0
    # fine-tune on the other backend
    assert main(["train", "--backend", "synthB", "--d", "3", "--r", "3", *_args(t), "--device", "cpu",
                 "--epochs", "1", "--init", str(ck / "synthA_d3_r3_mlp.pt"),
                 "--out", str(ck / "synthB_d3_r3_mlp.pt")]) == 0
    assert (ck / "synthB_d3_r3_mlp.pt").exists()
    # a checkpoint trained on another machine's device (e.g. the GPU laptop) must still evaluate here
    import torch
    payload = torch.load(ck / "synthA_d3_r3_mlp.pt", weights_only=True)
    payload["cfg"]["device"] = "cuda" if not torch.cuda.is_available() else "mps"
    torch.save(payload, ck / "synthA_d3_r3_mlp.pt")
    capsys.readouterr()
    assert main(["evaluate", "--backend", "synthA", "--configs", "3x3", *_args(t), "--ckpt-dir", str(ck),
                 "--out-dir", str(tmp_path / "ev_foreign")]) == 0
    assert "nn_mlp" in capsys.readouterr().out


def test_evaluate_missing_test_backend_is_an_error(parsed_tree, tmp_path, capsys):
    od = tmp_path / "ev"
    assert main(["evaluate", "--backend", "synthA", "--configs", "3x3", *_args(parsed_tree),
                 "--ckpt-dir", str(tmp_path / "none"), "--test-backend", "nope", "--out-dir", str(od)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not od.exists()


def test_run_ibm_fake_refuses_fresh_label(tmp_path, capsys):
    assert main(["run-ibm", "--configs", "3x1", "--label", "ibm_fez_fresh", "--raw-dir", str(tmp_path)]) == 2
    assert "reserved" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_learning_curve_cli(parsed_tree, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["train", "--backend", "synthA", "--d", "3", "--r", "3", *_args(parsed_tree), "--device", "cpu",
                 "--epochs", "1", "--width", "16", "--depth", "1", "--learning-curve",
                 "--fractions", "0.5,1.0"]) == 0
    lc = Path("results/qdrift/learning_curve_synthA_d3_r3_mlp.csv")
    assert lc.exists() and lc.with_suffix(".png").stat().st_size > 0
    decs = {r["decoder"] for r in csv.DictReader(open(lc))}
    assert {"nn_mlp", "mwpm_fitted"} <= decs


def test_sweep_dry_run(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    assert main(["sweep", "--backend", "ibm_fez", "--configs", "7x5,9x7", "--kinds", "mlp,conv",
                 "--epochs", "3", "--dry-run"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 4 and all("qdrift train" in l.replace('" "', " ") or " train " in l for l in lines)
    assert any("--epochs 3" in l for l in lines)
    assert not Path("results").exists()


def test_run_ibm_real_without_token(monkeypatch, tmp_path, capsys):
    import qiskit_ibm_runtime

    def boom(*a, **k):
        pytest.fail("QiskitRuntimeService must not be constructed")

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", boom)
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    assert main(["run-ibm", "--configs", "3x1", "--real", "--yes", "--raw-dir", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_data_is_friendly(tmp_path, capsys):
    assert main(["baseline", "--backend", "nope", "--d", "3", "--parsed-dir", str(tmp_path / "p")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["parse", "--raw-dir", str(tmp_path / "none")]) == 2


@pytest.mark.skipif(not Path("data/parsed/ibm_fez").exists(), reason="real parsed data missing")
def test_real_fez_baseline(tmp_path):
    assert main(["baseline", "--backend", "ibm_fez", "--d", "3", "--out", str(tmp_path / "b.csv")]) == 0
