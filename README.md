# Qwen3.5 low-bit sensitivity study

This project measures the perplexity sensitivity of Qwen/Qwen3.5-4B decoder
linear weights to symmetric per-output-channel fake quantization. It then
allocates bits per layer/part under average-bit budgets and compares greedy,
uniform, and random allocations.

## Setup and usage

```bash
uv sync
uv run python sens.py
uv run python alloc.py
```

All hyperparameters are in `config.yaml`. The current checked-in configuration
uses smoke mode: 2 sensitivity windows, 4 test windows, and 2 bit options.
The model and dataset are loaded from the Hugging Face cache/network as needed.

## Measured results
Average bits exclude per-channel quantization scales.
Generated files are organized as follows:

- `results/sens.csv`
- `results/heatmap.png`
- `results/results.csv`
- `results/allocation/allocation_greedy_<budget>.json`

Run logs are appended to the project-root `log.txt` file while also being
shown in the terminal.
