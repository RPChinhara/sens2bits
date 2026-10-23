from __future__ import annotations

import logging
import math
import types
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml

from sens2bits import sens


def write_config(tmp_path, monkeypatch, *, smoke=False, test_windows=1):
    data = {
        "model_id": "fake/model",
        "dtype": "float32",
        "device": "cpu",
        "max_len": 4,
        "batch_size": 2,
        "sens_windows": 2,
        "test_windows": test_windows,
        "bits_options": [4, 2],
        "budgets": [3],
        "unit": "part",
        "random_seeds": [0],
        "smoke": smoke,
        "smoke_sens_windows": 1,
        "smoke_test_windows": 1,
        "smoke_bits_options": [4, 2],
    }
    if smoke:
        data["sens_windows"] = 4
        data["test_windows"] = None
        data["bits_options"] = [8, 6]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(sens, "CONFIG_PATH", path)
    return data


class FakeLogitsModel:
    def __call__(self, input_ids=None, attention_mask=None):
        b, s = input_ids.shape
        return types.SimpleNamespace(logits=torch.zeros(b, s, 4))


def make_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            layer0 = self.model.layers[0]
            layer0.self_attn = nn.Linear(4, 5)
            layer0.mlp = nn.Module()
            layer0.mlp.proj = nn.Linear(5, 6)
            layer0.visual_encoder = nn.Linear(2, 2)
            layer1 = self.model.layers[1]
            layer1.linear_attn = nn.Module()
            layer1.linear_attn.proj = nn.Linear(4, 5)
            layer1.mlp = nn.Module()
            layer1.mlp.proj = nn.Linear(5, 6)

    return M()


def make_unit():
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target("layer_0_mixer_0", 0, "mixer", module)
    return sens.Unit("layer_0_mixer", 0, "mixer", [target]), target


def test_setup_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "test.log"
    monkeypatch.setattr(sens, "LOG_PATH", log_path)
    sens.setup_logging()
    assert len(sens.LOGGER.handlers) == 2
    assert sens.LOGGER.propagate is False
    assert sens.LOGGER.level == logging.INFO
    sens.LOGGER.info("hello")
    assert log_path.exists()


def test_uncaught_exception_hook_keyboard_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "LOG_PATH", tmp_path / "test.log")
    sens.setup_logging()
    original = sens.sys.__excepthook__
    calls = []
    monkeypatch.setattr(sens.sys, "__excepthook__", lambda *args: calls.append(args))
    hook = sens.sys.excepthook
    hook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert calls
    assert calls[0][0] is KeyboardInterrupt
    monkeypatch.setattr(sens.sys, "__excepthook__", original)


def test_uncaught_exception_hook_logs_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "LOG_PATH", tmp_path / "test.log")
    sens.setup_logging()
    original = sens.sys.__excepthook__
    calls = []
    monkeypatch.setattr(sens.sys, "__excepthook__", lambda *args: calls.append(args))
    hook = sens.sys.excepthook
    hook(RuntimeError, RuntimeError("boom"), None)
    assert calls
    assert calls[0][0] is RuntimeError
    monkeypatch.setattr(sens.sys, "__excepthook__", original)


def test_resolve_dtype():
    assert sens.resolve_dtype("float16") is torch.float16
    assert sens.resolve_dtype("float32") is torch.float32
    assert sens.resolve_dtype("bfloat16") is torch.bfloat16
    try:
        sens.resolve_dtype("int8")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_load_config_smoke_false(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, smoke=False)
    cfg = sens.load_config()
    assert cfg["sens_windows"] == 2
    assert cfg["bits_options"] == [4, 2]


def test_load_config_smoke_true(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, smoke=True)
    cfg = sens.load_config()
    assert cfg["sens_windows"] == 1
    assert cfg["bits_options"] == [4, 2]


class FakeCausal:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        obj = cls()
        obj.model_id = model_id
        obj.kwargs = kwargs
        return obj

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def named_children(self):
        return []


class FakeMultimodal(FakeCausal):
    pass


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()


def test_load_model_and_tokenizer_causal(tmp_path, monkeypatch):
    cfg = {"model_id": "m", "dtype": "float32", "device": "cpu"}
    monkeypatch.setattr(sens, "AutoModelForCausalLM", FakeCausal)
    monkeypatch.setattr(sens, "AutoModelForMultimodalLM", FakeMultimodal)
    monkeypatch.setattr(sens, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(sens, "LOGGER", logging.getLogger("x"))
    model, tokenizer = sens.load_model_and_tokenizer(cfg)
    assert isinstance(model, FakeCausal)
    assert model.evaluated is True
    assert model.device == "cpu"


def test_load_model_and_tokenizer_fallback(tmp_path, monkeypatch):
    class BadCausal:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(sens, "AutoModelForCausalLM", BadCausal)
    monkeypatch.setattr(sens, "AutoModelForMultimodalLM", FakeMultimodal)
    monkeypatch.setattr(sens, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(sens, "LOGGER", logging.getLogger("x"))
    cfg = {"model_id": "m", "dtype": "float32", "device": "cpu"}
    model, tokenizer = sens.load_model_and_tokenizer(cfg)
    assert isinstance(model, FakeMultimodal)


def test_load_model_and_tokenizer_raises(tmp_path, monkeypatch):
    class BadCausal:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("boom")

    class BadMulti:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("boom2")

    monkeypatch.setattr(sens, "AutoModelForCausalLM", BadCausal)
    monkeypatch.setattr(sens, "AutoModelForMultimodalLM", BadMulti)
    monkeypatch.setattr(sens, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(sens, "LOGGER", logging.getLogger("x"))
    cfg = {"model_id": "m", "dtype": "float32", "device": "cpu"}
    try:
        sens.load_model_and_tokenizer(cfg)
    except RuntimeError as e:
        assert "Both causal and multimodal model loading failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_find_targets():
    model = make_model()
    targets = sens.find_targets(model)
    names = [t.name for t in targets]
    assert any(".mlp." in n for n in names)
    assert all("visual" not in n for n in names)
    assert all("vision" not in n for n in names)
    assert targets == sorted(targets, key=lambda t: (t.layer, t.name))
    parts = {t.part for t in targets}
    assert parts <= {"mixer", "mlp"}


def test_print_target_summary():
    model = make_model()
    targets = sens.find_targets(model)
    sens.print_target_summary(model, targets)


def test_make_units_part():
    model = make_model()
    targets = sens.find_targets(model)
    units = sens.make_units(targets, "part")
    assert all(isinstance(u, sens.Unit) for u in units)
    if units:
        assert units[0].name.startswith("layer_")
    assert units[0].params > 0


def test_make_units_matrix():
    model = make_model()
    targets = sens.find_targets(model)
    units = sens.make_units(targets, "matrix")
    assert len(units) == len(targets)


def test_make_units_empty():
    assert sens.make_units([], "part") == []


def test_token_windows():
    def tokenizer(text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.arange(0, 10).unsqueeze(0)}

    split = [{"text": "a"}, {"text": "b"}]
    windows = sens.token_windows(tokenizer, split, 4)
    assert len(windows) == 2
    assert all(w.numel() == 4 for w in windows)


def test_token_windows_empty():
    def tokenizer(text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.arange(0, 3)}

    windows = sens.token_windows(tokenizer, [{"text": "a"}], 4)
    assert windows == []


def test_load_windows(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, smoke=False)

    def fake_load_dataset(name, subset):
        return {"train": [{"text": "train text"}], "test": [{"text": "test text"}]}

    monkeypatch.setattr(sens, "load_dataset", fake_load_dataset)

    def fake_token_windows(tokenizer, split, max_len):
        return [torch.arange(0, 4), torch.arange(0, 4)]

    monkeypatch.setattr(sens, "token_windows", fake_token_windows)
    train, test = sens.load_windows(lambda *a, **k: None, sens.load_config())
    assert len(train) == 2
    assert len(test) == 1


def test_load_windows_empty_raises(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, smoke=False)
    monkeypatch.setattr(sens, "load_dataset", lambda *a, **k: {"train": [{"text": "x"}], "test": [{"text": "y"}]})
    monkeypatch.setattr(sens, "token_windows", lambda *a, **k: [])
    try:
        sens.load_windows(lambda *a, **k: None, sens.load_config())
    except RuntimeError as e:
        assert "no complete" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_perplexity():
    cfg = {"batch_size": 2, "device": "cpu"}
    windows = [torch.arange(4), torch.arange(4), torch.arange(4)]
    model = FakeLogitsModel()
    val = sens.perplexity(model, windows, cfg)
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_fake_quantize_weight_errors():
    try:
        sens.fake_quantize_weight(torch.tensor([1.0, 2.0]), 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_fake_quantize_weight_basic():
    w = torch.tensor([[2.0, 4.0], [0.0, 0.0]])
    q, s = sens.fake_quantize_weight(w, 4)
    assert q.shape == w.shape
    assert s.shape == (2,)
    assert q.dtype == w.dtype
    assert s[1].item() == 1.0


def test_fake_quantize_weight_roundtrip_idempotent():
    w = torch.tensor([[1.0, 2.3], [4.5, -1.2]])
    q1, _ = sens.fake_quantize_weight(w, 4)
    q2, _ = sens.fake_quantize_weight(q1, 4)
    assert torch.equal(q1, q2)


def test_quantize_unit_and_restore():
    unit, target = make_unit()
    original = target.module.weight.data.clone()
    saved = {target.name: original.clone()}
    sens.quantize_unit(unit, 2)
    assert not torch.equal(target.module.weight.data, original)
    sens.restore_unit(unit, saved)
    assert torch.equal(target.module.weight.data, original)


def test_restore_unit_failure(monkeypatch):
    unit, target = make_unit()
    original = target.module.weight.data.clone()
    saved = {target.name: original.clone() + 1.0}
    monkeypatch.setattr(sens.torch, "equal", lambda a, b: False)
    try:
        sens.restore_unit(unit, saved)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")


def test_quantizer_check_pass():
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target("t", 0, "mixer", module)
    sens.quantizer_check(target, 4)


def test_quantizer_check_scale_mismatch(monkeypatch):
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target("t", 0, "mixer", module)
    orig = sens.fake_quantize_weight
    def fake(weight, bits):
        q, s = orig(weight, bits)
        return q, torch.ones(1)
    monkeypatch.setattr(sens, "fake_quantize_weight", fake)
    try:
        sens.quantizer_check(target, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")


def test_quantizer_check_too_many_unique(monkeypatch):
    module = nn.Linear(16, 2)
    with torch.no_grad():
        module.weight.copy_(torch.randn(2, 16))
    target = sens.Target("t", 0, "mixer", module)
    orig = sens.fake_quantize_weight
    def fake(weight, bits):
        q, s = orig(weight, bits)
        q = q.clone()
        q[0, :] = torch.arange(q.shape[1], dtype=q.dtype)
        return q, s
    monkeypatch.setattr(sens, "fake_quantize_weight", fake)
    try:
        sens.quantizer_check(target, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")


def test_quantizer_check_error_bound(monkeypatch):
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target("t", 0, "mixer", module)
    orig = sens.fake_quantize_weight
    def fake(weight, bits):
        q = weight.clone() + 100.0
        s = torch.ones(weight.shape[0])
        return q.to(weight.dtype), s
    monkeypatch.setattr(sens, "fake_quantize_weight", fake)
    try:
        sens.quantizer_check(target, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")


def test_quantizer_check_idempotence(monkeypatch):
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target("t", 0, "mixer", module)
    orig = sens.fake_quantize_weight
    calls = {"n": 0}
    def fake(weight, bits):
        q, s = orig(weight, bits)
        calls["n"] += 1
        if calls["n"] == 2:
            return q + 1.0, s
        return q, s
    monkeypatch.setattr(sens, "fake_quantize_weight", fake)
    try:
        sens.quantizer_check(target, 4)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")


def test_read_completed_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "SENS_PATH", tmp_path / "nope.csv")
    assert sens.read_completed() == set()


def test_read_completed_existing(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    p.write_text("unit,layer,part,bits,params,ppl,delta_ppl\nu,1,mlp,4,10,1.1,0.1\n")
    monkeypatch.setattr(sens, "SENS_PATH", p)
    assert sens.read_completed() == {("u", 4)}


def test_append_sensitivity(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    monkeypatch.setattr(sens, "SENS_PATH", p)
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    sens.append_sensitivity({"unit": "u", "layer": 1, "part": "mlp", "bits": 4, "params": 10, "ppl": 1.1, "delta_ppl": 0.1})
    assert p.exists()
    text = p.read_text()
    assert "unit,layer" in text
    sens.append_sensitivity({"unit": "u2", "layer": 2, "part": "mixer", "bits": 2, "params": 5, "ppl": 1.2, "delta_ppl": 0.2})
    lines = p.read_text().splitlines()
    assert len(lines) == 3


def test_save_heatmap(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    hp = tmp_path / "heatmap.png"
    monkeypatch.setattr(sens, "SENS_PATH", p)
    monkeypatch.setattr(sens, "HEATMAP_PATH", hp)
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    frame = pd.DataFrame(
        [
            {"unit": "u1", "layer": 0, "part": "mlp", "bits": 4, "params": 10, "ppl": 2.0, "delta_ppl": 10.0},
            {"unit": "u1", "layer": 0, "part": "mlp", "bits": 2, "params": 10, "ppl": 3.0, "delta_ppl": 1.0},
            {"unit": "u2", "layer": 1, "part": "mixer", "bits": 4, "params": 5, "ppl": 2.1, "delta_ppl": 5.0},
            {"unit": "u2", "layer": 1, "part": "mixer", "bits": 2, "params": 5, "ppl": 2.2, "delta_ppl": 6.0},
        ]
    )
    frame.to_csv(p, index=False)
    units = [
        sens.Unit("u1", 0, "mlp", []),
        sens.Unit("u2", 1, "mixer", []),
    ]
    non_mono = sens.save_heatmap(units, [4, 2])
    assert hp.exists()
    assert isinstance(non_mono, int)


def test_save_heatmap_use_log(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    hp = tmp_path / "heatmap.png"
    monkeypatch.setattr(sens, "SENS_PATH", p)
    monkeypatch.setattr(sens, "HEATMAP_PATH", hp)
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    frame = pd.DataFrame(
        [
            {"unit": "u1", "layer": 0, "part": "mlp", "bits": 4, "params": 10, "ppl": 2.0, "delta_ppl": 0.0},
            {"unit": "u1", "layer": 0, "part": "mlp", "bits": 2, "params": 10, "ppl": 3.0, "delta_ppl": 500.0},
        ]
    )
    frame.to_csv(p, index=False)
    units = [sens.Unit("u1", 0, "mlp", [])]
    sens.save_heatmap(units, [4, 2])
    assert hp.exists()


def test_run_sensitivity(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "SENS_PATH", tmp_path / "sens.csv")
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sens, "HEATMAP_PATH", tmp_path / "heatmap.png")
    monkeypatch.setattr(sens, "read_completed", lambda: set())

    unit, target = make_unit()
    saved_weight = target.module.weight.data.clone()
    model = FakeLogitsModel()

    def fake_perplexity(m, windows, cfg):
        return 1.5

    monkeypatch.setattr(sens, "perplexity", fake_perplexity)
    cfg = {"bits_options": [4, 2], "batch_size": 2, "device": "cpu"}
    sens.run_sensitivity(model, [unit], [torch.arange(4)], cfg, 1.5)
    assert (tmp_path / "sens.csv").exists()
    frame = pd.read_csv(tmp_path / "sens.csv")
    assert len(frame) == 2
    assert torch.equal(target.module.weight.data, saved_weight)


def test_run_sensitivity_skip_completed(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    monkeypatch.setattr(sens, "SENS_PATH", p)
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sens, "HEATMAP_PATH", tmp_path / "heatmap.png")
    frame = pd.DataFrame([{"unit": "layer_0_mixer", "layer": 0, "part": "mixer", "bits": 4, "params": 4, "ppl": 1.5, "delta_ppl": 0.0}])
    frame.to_csv(p, index=False)
    monkeypatch.setattr(sens, "read_completed", lambda: {("layer_0_mixer", 4)})

    unit, target = make_unit()
    unit.name = "layer_0_mixer"
    model = FakeLogitsModel()
    monkeypatch.setattr(sens, "perplexity", lambda m, w, c: 1.5)
    cfg = {"bits_options": [4, 2], "batch_size": 2, "device": "cpu"}
    sens.run_sensitivity(model, [unit], [torch.arange(4)], cfg, 1.5)
    frame = pd.read_csv(p)
    assert len(frame) == 2


def test_run_sensitivity_baseline_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "SENS_PATH", tmp_path / "sens.csv")
    monkeypatch.setattr(sens, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sens, "HEATMAP_PATH", tmp_path / "heatmap.png")
    monkeypatch.setattr(sens, "read_completed", lambda: set())
    unit, target = make_unit()
    model = FakeLogitsModel()
    calls = {"n": 0}

    def fake_perplexity(m, w, c):
        calls["n"] += 1
        if calls["n"] == 2:
            return 9.99
        return 1.5

    monkeypatch.setattr(sens, "perplexity", fake_perplexity)
    cfg = {"bits_options": [4], "batch_size": 2, "device": "cpu"}
    try:
        sens.run_sensitivity(model, [unit], [torch.arange(4)], cfg, 1.5)
    except AssertionError as e:
        assert "baseline changed" in str(e)
    else:
        raise AssertionError("expected AssertionError")


def test_sens_main(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "setup_logging", lambda: None)
    write_config(tmp_path, monkeypatch, smoke=False)
    cfg = sens.load_config()
    monkeypatch.setattr(sens, "load_config", lambda: cfg)
    fake_model = FakeLogitsModel()
    fake_tok = object()
    monkeypatch.setattr(sens, "load_model_and_tokenizer", lambda config: (fake_model, fake_tok))
    model = make_model()
    targets = sens.find_targets(model)
    monkeypatch.setattr(sens, "find_targets", lambda m: targets)
    units = sens.make_units(targets, "part")
    monkeypatch.setattr(sens, "make_units", lambda t, k: units)
    monkeypatch.setattr(sens, "load_windows", lambda tok, c: ([torch.arange(4)], [torch.arange(4)]))
    monkeypatch.setattr(sens, "perplexity", lambda m, w, c: 1.0)
    monkeypatch.setattr(sens, "quantizer_check", lambda t, b: None)
    monkeypatch.setattr(sens, "run_sensitivity", lambda *a, **k: None)
    monkeypatch.setattr(sens, "print_target_summary", lambda m, t: None)
    monkeypatch.setattr(sens, "LOGGER", logging.getLogger("x"))
    sens.main()
    assert True


def test_sens_main_no_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(sens, "setup_logging", lambda: None)
    write_config(tmp_path, monkeypatch, smoke=False)
    cfg = sens.load_config()
    monkeypatch.setattr(sens, "load_config", lambda: cfg)
    monkeypatch.setattr(sens, "load_model_and_tokenizer", lambda config: (FakeLogitsModel(), object()))
    monkeypatch.setattr(sens, "find_targets", lambda m: [])
    monkeypatch.setattr(sens, "LOGGER", logging.getLogger("x"))
    try:
        sens.main()
    except RuntimeError as e:
        assert "No target" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
