from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd

from sens import (
    LOGGER,
    load_config,
    load_model_and_tokenizer,
    load_windows,
    make_units,
    find_targets,
    perplexity,
    quantize_unit,
    restore_unit,
    setup_logging,
)


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
ALLOCATION_DIR = RESULTS_DIR / "allocation"
SENS_PATH = RESULTS_DIR / "sens.csv"
RESULTS_PATH = RESULTS_DIR / "results.csv"
RESULT_FIELDS = ["budget", "method", "avg_bits", "ppl_sens", "ppl_test", "pred_ppl_sens"]


def average_bits(units: list[Any], allocation: dict[str, int]) -> float:
    total_params = sum(unit.params for unit in units)
    return sum(unit.params * allocation[unit.name] for unit in units) / total_params


def load_costs() -> dict[str, dict[int, float]]:
    frame = pd.read_csv(SENS_PATH)
    return {
        str(unit): {int(row.bits): float(row.delta_ppl) for row in rows.itertuples()}
        for unit, rows in frame.groupby("unit")
    }


def greedy_allocation(units: list[Any], bits_options: list[int], costs: dict[str, dict[int, float]], budget: float) -> dict[str, int]:
    descending = sorted(bits_options, reverse=True)
    allocation = {unit.name: descending[0] for unit in units}
    while average_bits(units, allocation) > budget:
        candidates = []
        for unit in units:
            current = allocation[unit.name]
            index = descending.index(current)
            if index + 1 == len(descending):
                continue
            lower = descending[index + 1]
            saved = current - lower
            cost_increase = costs[unit.name][lower] - costs[unit.name][current]
            candidates.append((cost_increase / (unit.params * saved), unit.name, lower))
        if not candidates:
            raise ValueError(f"Budget {budget} is unreachable with configured bit options")
        _, unit_name, lower = min(candidates)
        allocation[unit_name] = lower
    return allocation


def random_allocation(units: list[Any], bits_options: list[int], budget: float, seed: int) -> dict[str, int]:
    descending = sorted(bits_options, reverse=True)
    allocation = {unit.name: descending[0] for unit in units}
    rng = random.Random(seed)
    while average_bits(units, allocation) > budget:
        choices = [
            unit
            for unit in units
            if descending.index(allocation[unit.name]) + 1 < len(descending)
        ]
        if not choices:
            raise ValueError(f"Budget {budget} is unreachable with configured bit options")
        unit = rng.choice(choices)
        allocation[unit.name] = descending[descending.index(allocation[unit.name]) + 1]
    return allocation


def uniform_allocation(units: list[Any], bits_options: list[int], bits: int) -> dict[str, int]:
    return {unit.name: bits for unit in units}


def save_allocation(budget: float, method: str, allocation: dict[str, int]) -> None:
    ALLOCATION_DIR.mkdir(parents=True, exist_ok=True)
    path = ALLOCATION_DIR / f"allocation_{method}_{budget:g}.json"
    path.write_text(json.dumps(allocation, indent=2, sort_keys=True) + "\n")


def evaluate_allocation(
    model: Any,
    units: list[Any],
    allocation: dict[str, int],
    train_windows: list[Any],
    test_windows: list[Any],
    config: dict[str, Any],
    predicted: float | None,
) -> dict[str, Any]:
    saved = {
        target.name: target.module.weight.data.clone()
        for unit in units
        for target in unit.targets
    }
    try:
        for unit in units:
            quantize_unit(unit, allocation[unit.name])
        ppl_sens = perplexity(model, train_windows, config)
        ppl_test = perplexity(model, test_windows, config)
    finally:
        for unit in units:
            restore_unit(unit, {target.name: saved[target.name] for target in unit.targets})
    return {
        "avg_bits": average_bits(units, allocation),
        "ppl_sens": ppl_sens,
        "ppl_test": ppl_test,
        "pred_ppl_sens": predicted,
    }


def main() -> None:
    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config)
    targets = find_targets(model)
    units = make_units(targets, config["unit"])
    train_windows, test_windows = load_windows(tokenizer, config)
    baseline_sens = perplexity(model, train_windows, config)
    baseline_test = perplexity(model, test_windows, config)
    costs = load_costs()
    bits_options = sorted(config["bits_options"], reverse=True)
    rows = [
        {
            "budget": "baseline",
            "method": "unquantized",
            "avg_bits": bits_options[0],
            "ppl_sens": baseline_sens,
            "ppl_test": baseline_test,
            "pred_ppl_sens": "",
        }
    ]
    LOGGER.info("scales_in_avg_bits: false")
    LOGGER.info(f"baseline: sens={baseline_sens:.6f} test={baseline_test:.6f}")
    for budget in config["budgets"]:
        greedy = greedy_allocation(units, bits_options, costs, budget)
        greedy_predicted = baseline_sens + sum(costs[unit.name][greedy[unit.name]] for unit in units)
        result = evaluate_allocation(model, units, greedy, train_windows, test_windows, config, greedy_predicted)
        if result["avg_bits"] > budget:
            raise AssertionError(f"greedy allocation exceeds budget {budget}")
        save_allocation(budget, "greedy", greedy)
        rows.append({"budget": budget, "method": "greedy", **result})
        LOGGER.info(f"budget={budget:g} method=greedy avg_bits={result['avg_bits']:.6f}")

        below = max(bit for bit in bits_options if bit <= budget)
        below_result = evaluate_allocation(
            model, units, uniform_allocation(units, bits_options, below), train_windows, test_windows, config, None
        )
        rows.append({"budget": budget, "method": "uniform_below", **below_result})

        above = min(bit for bit in bits_options if bit >= budget)
        above_result = evaluate_allocation(
            model, units, uniform_allocation(units, bits_options, above), train_windows, test_windows, config, None
        )
        rows.append({"budget": budget, "method": "uniform_above", **above_result})

        random_results = []
        for seed in config["random_seeds"]:
            allocation = random_allocation(units, bits_options, budget, seed)
            random_result = evaluate_allocation(model, units, allocation, train_windows, test_windows, config, None)
            random_results.append(random_result)
            rows.append({"budget": budget, "method": f"random_seed_{seed}", **random_result})
        sens_values = [result["ppl_sens"] for result in random_results]
        test_values = [result["ppl_test"] for result in random_results]
        LOGGER.info(
            f"budget={budget:g} random_mean_std="
            f"sens={mean(sens_values):.6f}/{stdev(sens_values) if len(sens_values) > 1 else 0.0:.6f} "
            f"test={mean(test_values):.6f}/{stdev(test_values) if len(test_values) > 1 else 0.0:.6f}"
        )
    pd.DataFrame(rows, columns=RESULT_FIELDS).to_csv(RESULTS_PATH, index=False)
    LOGGER.info("\n" + pd.DataFrame(rows, columns=RESULT_FIELDS).to_string(index=False))


if __name__ == "__main__":
    main()
