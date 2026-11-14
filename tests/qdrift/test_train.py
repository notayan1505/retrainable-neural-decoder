import numpy as np
import pytest
import torch

from qdrift.data.stein import split
from qdrift.data.synthetic import make_dataset
from qdrift.models.net import MODEL_KINDS, build_model, count_params
from qdrift.train import (NNDecoder, TrainConfig, learning_curve, load_checkpoint, pick_device,
                          predict, save_checkpoint, train_model)

quiet = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def ds():
    return make_dataset(3, 3, n_per_state=20000, seed=1, p_data=0.05, p_meas=0.05)


@pytest.mark.parametrize("kind", MODEL_KINDS)
@pytest.mark.parametrize("d,r", [(3, 1), (3, 3), (7, 5)])
def test_models_shape_backward(kind, d, r):
    m = build_model(kind, d, r, width=32, depth=2)
    x = torch.randint(0, 2, (5, r + 1, d - 1)).float()
    out = m(x)
    assert out.shape == (5,)
    out.sum().backward()
    assert count_params(m) > 0


def test_pick_device():
    assert pick_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        pytest.skip("cuda available")
    with pytest.raises(RuntimeError, match="CUDA"):
        pick_device("cuda")


def test_mlp_beats_no_decode(ds):
    tr, te = split(ds, seed=0)
    cfg = TrainConfig(kind="mlp", width=64, depth=2, epochs=8, batch_size=512, device="cpu")
    model, hist = train_model(ds.subset(tr), cfg, log=quiet)
    assert len(hist) == 8 and set(hist[0]) == {"epoch", "train_loss", "val_loss", "val_err", "seconds"}
    test = ds.subset(te)
    err = float((predict(model, test.detectors, device="cpu") != test.labels).mean())
    base = float(test.labels.mean())
    assert err < 0.8 * base, (err, base)


def test_checkpoint_roundtrip_and_compile(tmp_path, ds):
    small = ds.subset(np.arange(2000))
    cfg = TrainConfig(kind="conv", width=32, depth=2, epochs=2, batch_size=512, device="cpu", compile=True)
    model, _ = train_model(small, cfg, log=quiet)
    p = save_checkpoint(tmp_path / "m.pt", model, cfg, ds.d, ds.r, extra={"a": 1})
    m2, info = load_checkpoint(p, device="cpu")
    assert info["d"] == 3 and info["r"] == 3 and info["extra"] == {"a": 1} and info["cfg"]["kind"] == "conv"
    assert not any("_orig_mod" in k for k in m2.state_dict())
    a = predict(model, small.detectors, device="cpu")
    b = predict(m2, small.detectors, device="cpu")
    assert np.array_equal(a, b)


def test_learning_curve(ds):
    tr, te = split(ds, seed=0)
    cfg = TrainConfig(width=16, depth=1, epochs=2, batch_size=512, device="cpu")
    rows = learning_curve(ds.subset(tr), ds.subset(te), cfg, fractions=(0.1, 1.0), log=quiet)
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"decoder", "fraction", "n_train", "n_test", "errors", "p_L", "stderr"}
        assert row["decoder"] == "nn_mlp"
    assert rows[0]["n_train"] < rows[1]["n_train"] < len(tr)


def test_nndecoder_and_empty(ds):
    cfg = TrainConfig(width=16, depth=1, epochs=2, batch_size=512, device="cpu")
    dec = NNDecoder(cfg).fit(ds.subset(np.arange(1500)))
    assert dec.name == "nn_mlp"
    out = dec.predict(ds.detectors[:100])
    assert out.shape == (100,) and out.dtype == np.uint8
    e = dec.predict(ds.detectors[:0])
    assert e.shape == (0,)


def test_batch_larger_than_dataset(ds):
    cfg = TrainConfig(width=16, depth=1, epochs=2, batch_size=10**6, device="cpu")
    model, hist = train_model(ds.subset(np.arange(300)), cfg, log=quiet)
    assert len(hist) == 2
