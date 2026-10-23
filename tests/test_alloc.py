from __future__ import annotations

import json
import random
import types

import pandas as pd
import pytest
import torch
import torch.nn as nn

from sens2bits import alloc


from sens2bits import sens


def real_unit(name="u1", layer=0, part="mixer"):
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target = sens.Target(name + "_t", layer, part, module)
    return sens.Unit(name, layer, part, [target])


def write_sens_csv(path, rows=None):
    if rows is None:
        rows = [
            {"unit": "u1", "layer": 0, "part": "mixer", "bits": 8, "params": 4, "ppl": 1.0, "delta_ppl": 0.0},
            {"unit": "u1", "layer": 0, "part": "mixer", "bits": 4, "params": 4, "ppl": 1.2, "delta_ppl": 0.2},
            {"unit": "u2", "layer": 1, "part": "mlp", "bits": 8, "params": 6, "ppl": 1.0, "delta_ppl": 0.0},
            {"unit": "u2", "layer": 1, "part": "mlp", "bits": 4, "params": 6, "ppl": 1.3, "delta_ppl": 0.3},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_average_bits():
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    allocn = {u1.name: 8, u2.name: 4}
    # weights 4 and 4 => (4*8+4*4)/8 = 6
    assert alloc.average_bits(units, allocn) == 6.0


def test_load_costs(tmp_path, monkeypatch):
    p = tmp_path / "sens.csv"
    write_sens_csv(p)
    monkeypatch.setattr(alloc, "SENS_PATH", p)
    costs = alloc.load_costs()
    assert "u1" in costs
    assert costs["u1"][4] == 0.2
    assert costs["u2"][8] == 0.0


def test_load_checkpoint_missing_columns(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    pd.DataFrame([{"budget": 4, "method": "greedy"}]).to_csv(path, index=False)
    monkeypatch.setattr(alloc, "RESULTS_PATH", path)
    with pytest.raises(ValueError, match="missing columns"):
        alloc.load_checkpoint()


def test_checkpoint_row_missing():
    frame = pd.DataFrame([{"budget": 4, "method": "greedy"}])
    with pytest.raises(KeyError, match="Missing checkpoint row"):
        alloc.checkpoint_row(frame, 8, "greedy")


def test_greedy_allocation_basic():
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    bits_options = [8, 4]
    costs = {"u1": {8: 0.0, 4: 0.2}, "u2": {8: 0.0, 4: 0.3}}
    out = alloc.greedy_allocation(units, bits_options, costs, 4.0)
    # budget 4 means all at 4
    assert all(v == 4 for v in out.values())
    # avg bits 4 <=4
    assert alloc.average_bits(units, out) <= 4.0


def test_greedy_allocation_unreachable():
    u1 = real_unit("u1")
    units = [u1]
    bits_options = [8, 4]
    costs = {"u1": {8: 0.0, 4: 0.2}}
    with pytest.raises(ValueError, match="unreachable"):
        alloc.greedy_allocation(units, bits_options, costs, 2.0)


def test_random_allocation_basic():
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    bits_options = [8, 4]
    out = alloc.random_allocation(units, bits_options, 4.0, 0)
    assert list(out.values())
    assert alloc.average_bits(units, out) <= 4.0


def test_random_allocation_unreachable():
    u1 = real_unit("u1")
    units = [u1]
    bits_options = [8, 4]
    with pytest.raises(ValueError):
        alloc.random_allocation(units, bits_options, 2.0, 0)


def test_uniform_allocation():
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    out = alloc.uniform_allocation(units, [8, 4], 4)
    assert out == {u1.name: 4, u2.name: 4}


def test_predicted_ppl():
    units = [real_unit("u1"), real_unit("u2")]
    costs = {"u1": {8: 0.1, 4: 0.2}, "u2": {8: 0.3, 4: 0.4}}
    for allocation in (
        alloc.greedy_allocation(units, [8, 4], costs, 4.0),
        alloc.uniform_allocation(units, [8, 4], 4),
        alloc.random_allocation(units, [8, 4], 4.0, 0),
    ):
        assert alloc.predicted_ppl(1.0, units, costs, allocation) == pytest.approx(
            1.0 + sum(costs[unit.name][allocation[unit.name]] for unit in units)
        )


def test_save_allocation(tmp_path, monkeypatch):
    adir = tmp_path / "allocation"
    monkeypatch.setattr(alloc, "ALLOCATION_DIR", adir)
    alloc.save_allocation(4.0, "greedy", {"a": 4})
    path = adir / "allocation_greedy_4.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data == {"a": 4}


def test_evaluate_allocation(tmp_path, monkeypatch):
    u = real_unit("u1")
    units = [u]
    original = u.targets[0].module.weight.data.clone()
    model = types.SimpleNamespace()

    def fake_perplexity(m, windows, cfg):
        return 1.23

    monkeypatch.setattr(alloc, "perplexity", fake_perplexity)
    monkeypatch.setattr(alloc, "quantize_unit", lambda unit, bits: None)
    # make quantize_unit actually mutate to evaluate restore path? We'll rely on monkeypatch no-op and restore separately handled
    result = alloc.evaluate_allocation(
        model, units, {"u1": 4}, [torch.arange(4)], [torch.arange(4)], {"batch_size": 1, "device": "cpu"}, 2.0
    )
    assert torch.equal(u.targets[0].module.weight.data, original)
    assert result["avg_bits"] == 4.0  # params/params*4
    assert result["ppl_sens"] == 1.23
    assert result["ppl_test"] == 1.23
    assert result["pred_ppl_sens"] == 2.0


def test_evaluate_allocation_restore_on_exception(tmp_path, monkeypatch):
    u = real_unit("u1")
    units = [u]
    original = u.targets[0].module.weight.data.clone()
    model = types.SimpleNamespace()

    def fake_perplexity(m, windows, cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(alloc, "perplexity", fake_perplexity)
    monkeypatch.setattr(alloc, "quantize_unit", lambda unit, bits: None)
    with pytest.raises(RuntimeError, match="boom"):
        alloc.evaluate_allocation(model, units, {"u1": 4}, [torch.arange(4)], [torch.arange(4)], {"batch_size": 1, "device": "cpu"}, None)
    assert torch.equal(u.targets[0].module.weight.data, original)


def test_alloc_main(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc, "setup_logging", lambda: None)
    monkeypatch.setattr(alloc, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(alloc, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(alloc, "SENS_PATH", tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "ALLOCATION_DIR", tmp_path / "alloc")
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    # create sens csv
    write_sens_csv(tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "load_costs", lambda: {"u1": {8: 0.0, 4: 0.2}, "u2": {8: 0.0, 4: 0.3}})

    cfg = {
        "dtype": "float32",
        "device": "cpu",
        "max_len": 4,
        "batch_size": 2,
        "sens_windows": 2,
        "test_windows": 1,
        "bits_options": [8, 4],
        "budgets": [4],
        "unit": "part",
        "random_seeds": [0],
        "smoke": False,
    }
    # need config load
    import yaml
    cfgpath = tmp_path / "config.yaml"
    cfgpath.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(alloc, "load_config", lambda: cfg)
    # no smoke override? load_config mocked directly returns cfg
    umodel = None
    model = types.SimpleNamespace()
    fake_tok = object()
    monkeypatch.setattr(alloc, "load_model_and_tokenizer", lambda config: (model, fake_tok))
    # find targets / make units
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    monkeypatch.setattr(alloc, "find_targets", lambda m: [])
    monkeypatch.setattr(alloc, "make_units", lambda t, k: units)
    monkeypatch.setattr(alloc, "load_windows", lambda tok, c: ([torch.arange(4)], [torch.arange(4)]))
    monkeypatch.setattr(alloc, "perplexity", lambda m, w, c: 1.0)
    # greedy_allocation etc run actual; costs okay; average bits works
    monkeypatch.setattr(alloc, "save_allocation", lambda *a, **k: None)
    calls = []

    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"avg_bits": 4.0, "ppl_sens": 1.0, "ppl_test": 1.0, "pred_ppl_sens": 1.0}

    monkeypatch.setattr(alloc, "evaluate_allocation", evaluate)
    monkeypatch.setattr(alloc, "LOGGER", __import__("logging").getLogger("x"))
    alloc.main()
    assert (tmp_path / "results.csv").exists()
    rows = pd.read_csv(tmp_path / "results.csv")
    assert set(rows["method"]) == {"unquantized", "greedy", "uniform_below", "random_seed_0"}
    assert len(calls) == 3
    assert all(isinstance(call[0][-1], (int, float)) for call in calls)
    alloc.main()
    assert len(calls) == 3


def test_alloc_main_backfills_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc, "setup_logging", lambda: None)
    monkeypatch.setattr(alloc, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(alloc, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(alloc, "SENS_PATH", tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "ALLOCATION_DIR", tmp_path / "alloc")
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    write_sens_csv(tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "load_costs", lambda: {"u1": {8: 0.0, 4: 0.2}, "u2": {8: 0.0, 4: 0.3}})
    cfg = {
        "dtype": "float32", "device": "cpu", "max_len": 4, "batch_size": 2,
        "sens_windows": 2, "test_windows": 1, "bits_options": [8, 4],
        "budgets": [4], "unit": "part", "random_seeds": [0], "smoke": False,
    }
    baseline = {"budget": "baseline", "method": "unquantized", "avg_bits": 8, "ppl_sens": 1.0, "ppl_test": 1.1, "pred_ppl_sens": ""}
    greedy = {"budget": 4, "method": "greedy", "avg_bits": 4, "ppl_sens": 1.5, "ppl_test": 1.6, "pred_ppl_sens": 1.2}
    uniform = {"budget": 4, "method": "uniform_below", "avg_bits": 4, "ppl_sens": 2.0, "ppl_test": 2.1, "pred_ppl_sens": ""}
    random_row = {"budget": 4, "method": "random_seed_0", "avg_bits": 4, "ppl_sens": 3.0, "ppl_test": 3.1, "pred_ppl_sens": ""}
    pd.DataFrame([baseline, greedy, uniform, random_row]).to_csv(tmp_path / "results.csv", index=False)
    model = types.SimpleNamespace()
    units = [real_unit("u1"), real_unit("u2")]
    monkeypatch.setattr(alloc, "load_config", lambda: cfg)
    monkeypatch.setattr(alloc, "load_model_and_tokenizer", lambda config: (model, object()))
    monkeypatch.setattr(alloc, "find_targets", lambda m: [])
    monkeypatch.setattr(alloc, "make_units", lambda t, k: units)
    monkeypatch.setattr(alloc, "load_windows", lambda tok, c: ([], []))
    monkeypatch.setattr(alloc, "save_allocation", lambda *a, **k: None)
    monkeypatch.setattr(alloc, "evaluate_allocation", lambda *a, **k: pytest.fail("unexpected evaluation"))
    monkeypatch.setattr(alloc, "LOGGER", __import__("logging").getLogger("x"))
    before = pd.read_csv(tmp_path / "results.csv")
    alloc.main()
    after = pd.read_csv(tmp_path / "results.csv")
    pd.testing.assert_series_equal(after["ppl_sens"], before["ppl_sens"])
    pd.testing.assert_series_equal(after["ppl_test"], before["ppl_test"])
    assert after.loc[after["method"] == "uniform_below", "pred_ppl_sens"].notna().all()
    assert after.loc[after["method"] == "random_seed_0", "pred_ppl_sens"].notna().all()


def test_alloc_main_assertion(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc, "setup_logging", lambda: None)
    monkeypatch.setattr(alloc, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(alloc, "RESULTS_PATH", tmp_path / "results.csv")
    monkeypatch.setattr(alloc, "SENS_PATH", tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "ALLOCATION_DIR", tmp_path / "alloc")
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    write_sens_csv(tmp_path / "sens.csv")
    monkeypatch.setattr(alloc, "load_costs", lambda: {"u1": {8: 0.0, 4: 0.2}, "u2": {8: 0.0, 4: 0.3}})

    cfg = {
        "dtype": "float32",
        "device": "cpu",
        "max_len": 4,
        "batch_size": 2,
        "sens_windows": 2,
        "test_windows": 1,
        "bits_options": [8, 4],
        "budgets": [4],
        "unit": "part",
        "random_seeds": [0],
        "smoke": False,
    }
    import yaml
    cfgpath = tmp_path / "config.yaml"
    cfgpath.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(alloc, "load_config", lambda: cfg)

    model = __import__("types").SimpleNamespace()
    fake_tok = object()
    monkeypatch.setattr(alloc, "load_model_and_tokenizer", lambda config: (model, fake_tok))
    u1 = real_unit("u1")
    u2 = real_unit("u2")
    units = [u1, u2]
    monkeypatch.setattr(alloc, "find_targets", lambda m: [])
    monkeypatch.setattr(alloc, "make_units", lambda t, k: units)
    monkeypatch.setattr(alloc, "load_windows", lambda tok, c: ([torch.arange(4)], [torch.arange(4)]))
    monkeypatch.setattr(alloc, "perplexity", lambda m, w, c: 1.0)
    monkeypatch.setattr(alloc, "save_allocation", lambda *a, **k: None)
    monkeypatch.setattr(alloc, "evaluate_allocation", lambda *a, **k: {
        "avg_bits": 5.0,
        "ppl_sens": 1.0,
        "ppl_test": 1.0,
        "pred_ppl_sens": 1.0,
    })
    monkeypatch.setattr(alloc, "LOGGER", __import__("logging").getLogger("x"))
    with pytest.raises(AssertionError, match="exceeds budget"):
        alloc.main()
