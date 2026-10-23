from __future__ import annotations

import csv
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    __version__ as transformers_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_PATH = PROJECT_ROOT / "quantization_study.log"
RESULTS_DIR = PROJECT_ROOT / "results"
SENS_PATH = RESULTS_DIR / "sens.csv"
HEATMAP_PATH = RESULTS_DIR / "heatmap.png"
CSV_FIELDS = ["unit", "layer", "part", "bits", "params", "ppl", "delta_ppl"]
LOGGER = logging.getLogger("sens2bits")


def setup_logging() -> None:
    """Configure console and append-only file logging for both workflows."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    LOGGER.propagate = False


@dataclass
class Target:
    """A decoder linear module selected for quantization experiments."""
    name: str
    layer: int
    part: str
    module: nn.Linear


@dataclass
class Unit:
    """One allocation unit, containing one or more target linear modules."""
    name: str
    layer: int
    part: str
    targets: list[Target]

    @property
    def params(self) -> int:
        """Return the number of weight parameters represented by this unit."""
        return sum(target.module.weight.numel() for target in self.targets)


def load_config() -> dict[str, Any]:
    """Load YAML settings and apply smoke-mode overrides when enabled."""
    with CONFIG_PATH.open() as stream:
        config = yaml.safe_load(stream)
    if config["smoke"]:
        config = dict(config)
        config["sens_windows"] = config["smoke_sens_windows"]
        config["test_windows"] = config["smoke_test_windows"]
        config["bits_options"] = config["smoke_bits_options"]
    return config


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Convert a configured dtype name to its PyTorch dtype."""
    dtypes = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        return dtypes[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}") from exc


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[nn.Module, Any]:
    """Load the configured causal model, tokenizer, and place the model on-device."""
    dtype = resolve_dtype(config["dtype"])
    kwargs = {"dtype": dtype}
    try:
        # Qwen3.5 normally exposes a causal LM class; retain the multimodal
        # fallback for model revisions whose auto-mapping requires it.
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], **kwargs)
    except Exception as causal_error:
        LOGGER.warning(f"AutoModelForCausalLM failed; trying AutoModelForMultimodalLM: {causal_error}")
        try:
            model = AutoModelForMultimodalLM.from_pretrained(config["model_id"], **kwargs)
        except Exception as multimodal_error:
            raise RuntimeError("Both causal and multimodal model loading failed") from multimodal_error
    model.to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model.eval()
    LOGGER.info(f"model_class: {model.__class__.__name__}")
    LOGGER.info(f"transformers_version: {transformers_version}")
    LOGGER.info("slow_path_warning: inspect runtime warnings above; no kernel packages were installed")
    LOGGER.info(f"top_level_children: {[name for name, _ in model.named_children()]}")
    return model, tokenizer


def find_targets(model: nn.Module) -> list[Target]:
    """Find decoder linear weights while excluding visual paths and non-target modules."""
    targets = []
    pattern = re.compile(r"\.layers\.(\d+)\.")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if (
            match
            and isinstance(module, nn.Linear)
            and "visual" not in name
            and "vision" not in name
        ):
            part = "mlp" if ".mlp." in name else "mixer"
            targets.append(Target(name, int(match.group(1)), part, module))
    targets.sort(key=lambda target: (target.layer, target.name))
    return targets


def print_target_summary(model: nn.Module, targets: list[Target]) -> None:
    """Log layer types, representative modules, and target parameter statistics."""
    layer_numbers = sorted({target.layer for target in targets})
    by_layer = defaultdict(list)
    for target in targets:
        by_layer[target.layer].append(target)
    layer_pattern = re.compile(r"\.layers\.(\d+)$")
    layer_types = {}
    for name, module in model.named_modules():
        match = layer_pattern.search(name)
        if match:
            # Child names are more stable across Qwen revisions than class names.
            child_names = {child_name for child_name, _ in module.named_children()}
            if "self_attn" in child_names:
                layer_types[int(match.group(1))] = "attention"
            elif "linear_attn" in child_names:
                layer_types[int(match.group(1))] = "deltanet"
    first_attention = next((layer for layer in layer_numbers if layer_types.get(layer) == "attention"), layer_numbers[0])
    first_delta = next((layer for layer in layer_numbers if layer_types.get(layer) == "deltanet"), layer_numbers[-1])
    for label, layer in (("first_attention_type_layer", first_attention), ("first_deltanet_type_layer", first_delta)):
        LOGGER.info(f"{label}: {layer}")
        for target in by_layer[layer]:
            LOGGER.info(f"  {target.name}: {tuple(target.module.weight.shape)}")
    layer_type_pattern = [layer_types.get(index, "unknown") for index in layer_numbers]
    all_params = sum(parameter.numel() for parameter in model.parameters())
    target_params = sum(target.module.weight.numel() for target in targets)
    part_counts = {part: sum(target.module.weight.numel() for target in targets if target.part == part) for part in ("mixer", "mlp")}
    LOGGER.info(f"decoder_layers: {len(layer_numbers)}")
    LOGGER.info(f"layer_type_pattern: {layer_type_pattern}")
    LOGGER.info(f"attention_deltanet_ratio: {layer_type_pattern.count('attention')}:{layer_type_pattern.count('deltanet')}")
    LOGGER.info(f"linears_per_part: {dict((part, sum(target.part == part for target in targets)) for part in ('mixer', 'mlp'))}")
    LOGGER.info(f"target_params: {target_params}")
    LOGGER.info(f"target_fraction_all_params: {target_params / all_params}")
    LOGGER.info(f"target_params_by_part: {part_counts}")


def make_units(targets: list[Target], unit_kind: str) -> list[Unit]:
    """Group targets by layer/part or keep each matrix as its own unit."""
    grouped: dict[tuple[int, str], list[Target]] = defaultdict(list)
    for target in targets:
        key = (target.layer, target.part) if unit_kind == "part" else (target.layer, target.name)
        grouped[key].append(target)
    units = []
    for (layer, part_or_name), members in sorted(grouped.items()):
        part = members[0].part
        unit_name = f"layer_{layer}_{part}" if unit_kind == "part" else members[0].name
        units.append(Unit(unit_name, layer, part, members))
    return units


def token_windows(tokenizer: Any, split: Any, max_len: int) -> list[torch.Tensor]:
    """Join a dataset split, tokenize it once, and create non-overlapping windows."""
    text = "\n\n".join(item["text"] for item in split)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return [encoded[start : start + max_len] for start in range(0, encoded.numel() - max_len + 1, max_len)]


def load_windows(tokenizer: Any, config: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load Wikitext-2 and return configured train-sensitivity and test windows."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train = token_windows(tokenizer, dataset["train"], config["max_len"])[: config["sens_windows"]]
    test = token_windows(tokenizer, dataset["test"], config["max_len"])
    if config["test_windows"] is not None:
        test = test[: config["test_windows"]]
    if not train or not test:
        raise RuntimeError("Wikitext produced no complete evaluation windows")
    return train, test


def perplexity(model: nn.Module, windows: list[torch.Tensor], config: dict[str, Any]) -> float:
    """Measure perplexity over all tokens in the supplied evaluation windows."""
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(windows), config["batch_size"]):
            # Padding lets the final partial batch use the same forward path.
            batch = torch.nn.utils.rnn.pad_sequence(
                windows[start : start + config["batch_size"]], batch_first=True, padding_value=0
            ).to(config["device"])
            attention_mask = torch.ones_like(batch)
            logits = model(input_ids=batch, attention_mask=attention_mask).logits[:, :-1].float()
            labels = batch[:, 1:]
            # Shift logits and labels for causal next-token prediction.
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            total_nll += float(loss)
            total_tokens += labels.numel()
    elapsed = time.perf_counter() - started
    value = math.exp(total_nll / total_tokens)
    LOGGER.info(f"evaluation: windows={len(windows)} ppl={value:.6f} seconds={elapsed:.3f}")
    return value


def fake_quantize_weight(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply symmetric per-output-channel fake quantization to a matrix weight."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix weight, got shape {tuple(weight.shape)}")
    qmax = 2 ** (bits - 1) - 1
    float_weight = weight.float()
    # Scales are per output row because Linear weights are shaped (out, in).
    scale = float_weight.abs().amax(dim=1) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(float_weight / scale[:, None]), -qmax, qmax) * scale[:, None]
    return quantized.to(weight.dtype), scale


def quantize_unit(unit: Unit, bits: int) -> None:
    """Replace every matrix in a unit with its fake-quantized weights."""
    for target in unit.targets:
        target.module.weight.data.copy_(fake_quantize_weight(target.module.weight.data, bits)[0])


def restore_unit(unit: Unit, saved: dict[str, torch.Tensor]) -> None:
    """Restore a unit and assert that every original tensor is recovered exactly."""
    for target in unit.targets:
        target.module.weight.data.copy_(saved[target.name])
        if not torch.equal(target.module.weight.data, saved[target.name]):
            raise AssertionError(f"weight restoration failed for {target.name}")


def quantizer_check(target: Target, bits: int) -> None:
    """Validate scale shape, codebook size, error bound, and idempotence."""
    original = target.module.weight.data.clone()
    quantized, scales = fake_quantize_weight(original, bits)
    qmax = 2 ** (bits - 1) - 1
    if scales.numel() != target.module.out_features:
        raise AssertionError("scale length does not match output channels")
    # Exercise the quantizer on an actual model matrix, not a toy tensor.
    for row in quantized.float():
        if torch.unique(row).numel() > 2**bits - 1:
            raise AssertionError("quantized row has too many unique values")
    error = (quantized.float() - original.float()).abs().amax(dim=1)
    if not torch.all(error <= scales / 2 + torch.finfo(original.dtype).eps * 4):
        raise AssertionError("quantization error exceeds half-scale bound")
    twice = fake_quantize_weight(quantized, bits)[0]
    if not torch.equal(quantized, twice):
        raise AssertionError("quantization is not idempotent")
    LOGGER.info(f"quantizer_check: passed target={target.name} bits={bits} qmax={qmax}")


def read_completed() -> set[tuple[str, int]]:
    """Read completed unit/bit rows so an interrupted grid can resume safely."""
    if not SENS_PATH.exists():
        return set()
    frame = pd.read_csv(SENS_PATH)
    return {(str(row.unit), int(row.bits)) for row in frame.itertuples()}


def append_sensitivity(row: dict[str, Any]) -> None:
    """Append one completed sensitivity measurement to the results CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not SENS_PATH.exists()
    with SENS_PATH.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_heatmap(units: list[Unit], bits_options: list[int]) -> int:
    """Save the sensitivity heatmap and count units with non-monotone deltas."""
    LOGGER.info(f"Saving heatmap: {HEATMAP_PATH}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SENS_PATH)
    # Rows are ordered by layer and part; columns follow the configured bit grid.
    pivot = frame.pivot_table(index=["layer", "part"], columns="bits", values="delta_ppl", aggfunc="first")
    pivot = pivot.reindex(columns=bits_options)
    plt.figure(figsize=(max(6, len(bits_options) * 1.5), max(6, len(pivot) * 0.25)))
    values = pivot.to_numpy()
    use_log = float(values.max() - values.min()) > 100 if values.size else False
    image = plt.imshow(values, aspect="auto", cmap="coolwarm", norm=None if not use_log else plt.matplotlib.colors.SymLogNorm(linthresh=1e-3))
    plt.colorbar(image, label="delta perplexity")
    plt.xticks(range(len(bits_options)), bits_options)
    plt.yticks(range(len(pivot)), [f"{layer}:{part}" for layer, part in pivot.index])
    plt.xlabel("bits")
    plt.ylabel("layer:part")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH)
    plt.close()
    non_monotone = 0
    for unit in units:
        values = frame[frame["unit"] == unit.name].sort_values("bits")["delta_ppl"].to_numpy()
        if len(values) > 1 and not (values[1:] >= values[:-1]).all():
            non_monotone += 1
    LOGGER.info(f"non_monotone_units: {non_monotone}")
    return non_monotone


def run_sensitivity(model: nn.Module, units: list[Unit], windows: list[torch.Tensor], config: dict[str, Any], baseline: float) -> None:
    """Run the resumable sensitivity grid while restoring weights after each swap."""
    completed = read_completed()
    # Count only missing rows so the first timing estimate reflects remaining work.
    total_evaluations = sum((unit.name, bits) not in completed for unit in units for bits in config["bits_options"])
    completed_evaluations = 0
    seconds_per_evaluation = None
    for unit in units:
        # Every bit setting for this unit starts from the same saved baseline.
        saved = {target.name: target.module.weight.data.clone() for target in unit.targets}
        try:
            for bits in config["bits_options"]:
                if (unit.name, bits) in completed:
                    # Existing rows may come from a prior interrupted run.
                    LOGGER.info(f"skipping_completed: {unit.name} bits={bits}")
                    continue
                quantize_unit(unit, bits)
                LOGGER.info(f"quantized: {unit.name} bits={bits}")
                started = time.perf_counter()
                ppl = perplexity(model, windows, config)
                seconds_per_evaluation = time.perf_counter() - started
                completed_evaluations += 1
                if completed_evaluations == 1:
                    remaining = total_evaluations - completed_evaluations
                    LOGGER.info(f"projected_grid_seconds: {remaining * seconds_per_evaluation:.3f}")
                # Persist each row immediately so an interruption loses little work.
                append_sensitivity(
                    {
                        "unit": unit.name,
                        "layer": unit.layer,
                        "part": unit.part,
                        "bits": bits,
                        "params": unit.params,
                        "ppl": ppl,
                        "delta_ppl": ppl - baseline,
                    }
                )
                restore_unit(unit, saved)
        finally:
            # Also restore after exceptions during evaluation or CSV writing.
            restore_unit(unit, saved)
        LOGGER.info(f"unit_restored: {unit.name} true")
    remeasured = perplexity(model, windows, config)
    if abs(remeasured - baseline) > 1e-3:
        raise AssertionError(f"baseline changed after grid: {baseline} -> {remeasured}")
    save_heatmap(units, config["bits_options"])


def main() -> None:
    """Run model inspection, baselines, quantizer checks, and sensitivity analysis."""
    setup_logging()
    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config)
    targets = find_targets(model)
    if not targets:
        raise RuntimeError("No target linear modules found")
    # All later measurements depend on this discovered target set and baseline.
    print_target_summary(model, targets)
    units = make_units(targets, config["unit"])
    train_windows, test_windows = load_windows(tokenizer, config)
    baseline_sens = perplexity(model, train_windows, config)
    baseline_test = perplexity(model, test_windows, config)
    LOGGER.info(f"baseline_sens: {baseline_sens:.6f}")
    LOGGER.info(f"baseline_test: {baseline_test:.6f}")
    quantizer_check(targets[0], config["bits_options"][0])
    run_sensitivity(model, units, train_windows, config, baseline_sens)
    LOGGER.info(f"targets: {len(targets)} units: {len(units)}")


if __name__ == "__main__":
    main()
