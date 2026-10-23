from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd

from .sens import (
    LOGGER,
    find_targets,
    load_config,
    load_model_and_tokenizer,
    load_windows,
    make_units,
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


def budget_key(budget: Any) -> str:
    """Normalize numeric budget representations for stable checkpoint keys."""
    if str(budget) == "baseline":
        return "baseline"
    return f"{float(budget):g}"


def load_checkpoint() -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Load completed allocation rows and return keys used for resume decisions."""
    if not RESULTS_PATH.exists():
        return pd.DataFrame(columns=RESULT_FIELDS), set()
    frame = pd.read_csv(RESULTS_PATH)
    missing = set(RESULT_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Allocation checkpoint is missing columns: {sorted(missing)}")
    keys = {(budget_key(row.budget), str(row.method)) for row in frame.itertuples()}
    return frame[RESULT_FIELDS], keys


def append_checkpoint(row: dict[str, Any]) -> None:
    """Append one completed allocation evaluation immediately to results.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=RESULT_FIELDS).to_csv(
        RESULTS_PATH,
        mode="a",
        header=not RESULTS_PATH.exists(),
        index=False,
    )


def checkpoint_row(frame: pd.DataFrame, budget: Any, method: str) -> dict[str, Any]:
    """Return a previously completed row for a skipped evaluation."""
    matches = frame[frame["budget"].map(budget_key).eq(budget_key(budget)) & frame["method"].eq(method)]
    if matches.empty:
        raise KeyError(f"Missing checkpoint row for budget={budget}, method={method}")
    return matches.iloc[-1].to_dict()


def average_bits(units: list[Any], allocation: dict[str, int]) -> float:
    """Return the parameter-weighted average bit width of an allocation."""
    total_params = sum(unit.params for unit in units)
    return sum(unit.params * allocation[unit.name] for unit in units) / total_params


def load_costs() -> dict[str, dict[int, float]]:
    """Load measured per-unit perplexity deltas from the sensitivity CSV."""
    frame = pd.read_csv(SENS_PATH)
    return {
        str(unit): {int(row.bits): float(row.delta_ppl) for row in rows.itertuples()}
        for unit, rows in frame.groupby("unit")
    }


def greedy_allocation(units: list[Any], bits_options: list[int], costs: dict[str, dict[int, float]], budget: float) -> dict[str, int]:
    """Downgrade the cheapest next bit step until the budget is satisfied."""
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
            # Normalize measured cost by parameter-weighted bits saved.
            candidates.append((cost_increase / (unit.params * saved), unit.name, lower))
        if not candidates:
            raise ValueError(f"Budget {budget} is unreachable with configured bit options")
        _, unit_name, lower = min(candidates)
        allocation[unit_name] = lower
    return allocation


def random_allocation(units: list[Any], bits_options: list[int], budget: float, seed: int) -> dict[str, int]:
    """Create a reproducible random sequence of one-step downgrades."""
    descending = sorted(bits_options, reverse=True)
    allocation = {unit.name: descending[0] for unit in units}
    rng = random.Random(seed)
    while average_bits(units, allocation) > budget:
        # Randomness chooses the next legal downgrade; each step remains adjacent.
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
    """Assign one bit width to every unit."""
    return {unit.name: bits for unit in units}


def predicted_ppl(
    baseline_sens: float,
    units: list[Any],
    costs: dict[str, dict[int, float]],
    allocation: dict[str, int],
) -> float:
    """Predict PPL using additive per-unit sensitivity deltas."""
    return baseline_sens + sum(costs[unit.name][allocation[unit.name]] for unit in units)


def save_allocation(budget: float, method: str, allocation: dict[str, int]) -> None:
    """Persist an allocation as JSON under the allocation results directory."""
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
    """Measure both PPL splits for an allocation and restore all original weights."""
    saved = {
        target.name: target.module.weight.data.clone()
        for unit in units
        for target in unit.targets
    }
    try:
        # Apply the complete allocation before either perplexity measurement.
        for unit in units:
            quantize_unit(unit, allocation[unit.name])
        ppl_sens = perplexity(model, train_windows, config)
        ppl_test = perplexity(model, test_windows, config)
    finally:
        # Never leave quantized weights in the model for the next method or budget.
        for unit in units:
            restore_unit(unit, {target.name: saved[target.name] for target in unit.targets})
    return {
        "avg_bits": average_bits(units, allocation),
        "ppl_sens": ppl_sens,
        "ppl_test": ppl_test,
        "pred_ppl_sens": predicted,
    }


def main() -> None:
    """Run greedy, uniform, and random allocation evaluations for every budget."""
    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config)
    targets = find_targets(model)
    units = make_units(targets, config["unit"])
    train_windows, test_windows = load_windows(tokenizer, config)
    costs = load_costs()
    bits_options = sorted(config["bits_options"], reverse=True)
    checkpoint, completed = load_checkpoint()
    # Every completed row is durable before the next expensive evaluation starts.
    baseline_key = ("baseline", "unquantized")
    if ("baseline", "unquantized") not in completed:
        # Costs and evaluation data must use the same active configuration.
        baseline_sens = perplexity(model, train_windows, config)
        baseline_test = perplexity(model, test_windows, config)
        baseline_row = {
            "budget": "baseline",
            "method": "unquantized",
            "avg_bits": bits_options[0],
            "ppl_sens": baseline_sens,
            "ppl_test": baseline_test,
            "pred_ppl_sens": "",
        }
        append_checkpoint(baseline_row)
        checkpoint = pd.concat([checkpoint, pd.DataFrame([baseline_row])], ignore_index=True)
        completed.add(baseline_key)
    else:
        baseline_row = checkpoint_row(checkpoint, "baseline", "unquantized")
        baseline_sens = float(baseline_row["ppl_sens"])
        baseline_test = float(baseline_row["ppl_test"])

    def evaluate_checkpoint(
        budget: float, method: str, allocation: dict[str, int], predicted: float
    ) -> dict[str, Any]:
        nonlocal checkpoint
        key = (budget_key(budget), method)
        if key in completed:
            result = checkpoint_row(checkpoint, budget, method)
            if pd.isna(result["pred_ppl_sens"]):
                mask = checkpoint["budget"].map(budget_key).eq(budget_key(budget)) & checkpoint["method"].eq(method)
                checkpoint.loc[mask, "pred_ppl_sens"] = predicted
                checkpoint[RESULT_FIELDS].to_csv(RESULTS_PATH, index=False)
                result["pred_ppl_sens"] = predicted
            LOGGER.info(f"resume: skipping budget={budget:g} method={method}")
            return result
        result = evaluate_allocation(model, units, allocation, train_windows, test_windows, config, predicted)
        if method == "greedy" and result["avg_bits"] > budget:
            raise AssertionError(f"greedy allocation exceeds budget {budget}")
        row = {"budget": budget, "method": method, **result}
        append_checkpoint(row)
        checkpoint = pd.concat([checkpoint, pd.DataFrame([row])], ignore_index=True)
        completed.add(key)
        return result

    LOGGER.info("scales_in_avg_bits: false")
    LOGGER.info(f"baseline: sens={baseline_sens:.6f} test={baseline_test:.6f}")
    for budget in config["budgets"]:
        # Each evaluator restores weights, so every strategy starts unquantized.
        greedy = greedy_allocation(units, bits_options, costs, budget)
        result = evaluate_checkpoint(budget, "greedy", greedy, predicted_ppl(baseline_sens, units, costs, greedy))
        save_allocation(budget, "greedy", greedy)
        LOGGER.info(f"budget={budget:g} method=greedy avg_bits={result['avg_bits']:.6f}")

        below = max(bit for bit in bits_options if bit <= budget)
        below_allocation = uniform_allocation(units, bits_options, below)
        below_result = evaluate_checkpoint(
            budget, "uniform_below", below_allocation,
            predicted_ppl(baseline_sens, units, costs, below_allocation),
        )

        above = min(bit for bit in bits_options if bit >= budget)
        above_allocation = uniform_allocation(units, bits_options, above)
        if below == above:
            LOGGER.info(f"budget={budget:g} method=uniform_above reused uniform_below result")
            if (budget_key(budget), "uniform_above") in completed:
                above_result = evaluate_checkpoint(
                    budget, "uniform_above", above_allocation,
                    predicted_ppl(baseline_sens, units, costs, above_allocation),
                )
            else:
                above_result = below_result
        else:
            above_result = evaluate_checkpoint(
                budget, "uniform_above", above_allocation,
                predicted_ppl(baseline_sens, units, costs, above_allocation),
            )
        if below == above and (budget_key(budget), "uniform_above") not in completed:
            row = {"budget": budget, "method": "uniform_above", **above_result}
            append_checkpoint(row)
            checkpoint = pd.concat([checkpoint, pd.DataFrame([row])], ignore_index=True)
            completed.add((budget_key(budget), "uniform_above"))

        random_results = []
        for seed in config["random_seeds"]:
            # Keep seeds as separate rows so random variability remains auditable.
            method = f"random_seed_{seed}"
            allocation = random_allocation(units, bits_options, budget, seed)
            random_result = evaluate_checkpoint(
                budget, method, allocation,
                predicted_ppl(baseline_sens, units, costs, allocation),
            )
            random_results.append(random_result)
        sens_values = [result["ppl_sens"] for result in random_results]
        test_values = [result["ppl_test"] for result in random_results]
        LOGGER.info(
            f"budget={budget:g} random_mean_std="
            f"sens={mean(sens_values):.6f}/{stdev(sens_values) if len(sens_values) > 1 else 0.0:.6f} "
            f"test={mean(test_values):.6f}/{stdev(test_values) if len(test_values) > 1 else 0.0:.6f}"
        )
    LOGGER.info("\n" + checkpoint[RESULT_FIELDS].to_string(index=False))


if __name__ == "__main__":
    main()
