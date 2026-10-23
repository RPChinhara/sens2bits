# Qwen3.5 low-bit sensitivity study

This project measures the perplexity sensitivity of Qwen/Qwen3.5-4B decoder
linear weights to symmetric per-output-channel fake quantization. It then
allocates bits per layer/part under average-bit budgets and compares greedy,
uniform, and random allocations.

## Setup and usage

```bash
uv sync
uv run python main.py
```

The central command runs sensitivity measurement first and then allocation.
The individual stages can also be run separately:

```bash
uv run python src/sens2bits/sens.py
uv run python src/sens2bits/alloc.py
```

All hyperparameters are in `config.yaml`.

### Start a fresh run:

```bash
rm -rf results quantization_study.log
uv run python main.py
```

## Measured results
Average bits exclude per-channel quantization scales.
Generated files are organized as follows:

- `results/sens.csv`
- `results/heatmap.png`
- `results/results.csv`
- `results/allocation/allocation_greedy_<budget>.json`

Warnings and uncaught errors are written to `quantization_study.log`. Allocation
results are checkpointed row by row in `results/results.csv`; rerunning the
allocator skips completed `(budget, method)` evaluations and continues with
missing rows.
