# Sensitivity quantization study plan

## Milestone 0 - uv environment
- [x] `uv init --python 3.12` scaffold was retained and its generated package entry point was deleted.
- [x] Update `pyproject.toml` with the exact CUDA wheel index routing and `tool.uv.sources` entry.
- [x] `uv add torch transformers datasets matplotlib pandas pyyaml` and generate `uv.lock`.
- [x] Check output: `2.14.1+cu130 5.20.2 True NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`.
- [x] PASS: the CUDA-enabled torch, Transformers, and GPU imports succeeded.

## Milestone 1 - load, inspect, baseline
- [x] Load with `AutoModelForCausalLM` and fallback to `AutoModelForMultimodalLM` if needed.
- [x] Print model class, Transformers version, slow-path warning status, and top-level children.
- [x] Identify all decoder-layer `nn.Linear` modules excluding embeddings, `lm_head`, norms, and visual paths; print per-layer names and shapes for the first attention and first DeltaNet layer, then summary metrics.
- [x] Load Wikitext-2 raw train/test splits, tokenize once, and compute baseline perplexity on sensitivity and test windows.
- [x] PASS smoke check: 248 targets; 32 layers; attention:DeltaNet ratio 8:24; finite PPL 13.503187/10.767177; evaluation time printed. The architecture differs from the documented 3:1 ordering only in representation as an 8:24 aggregate, which is the same 1:3 ratio.

## Milestone 2 - quantizer tests
- [x] Implement symmetric per-output-channel fake quantization for `nn.Linear` weights.
- [x] Check on a real target weight that row scales length is `out_features`, per-row unique values fit within the codebook, and max row error is bounded.
- [x] Check that quantizing the same weight twice produces the same result.
- [x] PASS smoke quantizer check on a real target.

## Milestone 3 - sensitivity grid
- [x] Implement sensitivity sweep across each unit and bit option, restoring weights after each evaluation.
- [x] Append rows to `sens.csv` as each unit finishes, skipping already-completed rows.
- [x] Check: `torch.equal(weight, saved original)` was true after every restoration; baseline re-measured within `1e-3`; heatmap saved to `heatmap.png`.
- [x] Print how many units are non-monotone with respect to bits (56 in the final smoke grid).
- [x] PASS regenerated smoke grid with feasible smoke bit options `[8, 2]`.

## Milestone 4 - greedy allocator
- [x] Compute average bits across units using parameter-weighted budget logic; ignore per-channel scales in the metric.
- [x] Implement greedy single-step downgrades with the smallest cost increase per parameter-weighted bit saved.
- [x] Save JSON allocations for every budget.
- [x] Check: final average bits stays at or below each budget and the result is printed.
- [x] PASS: all six smoke budgets met their caps.

## Milestone 5 - evaluation
- [x] Quantize all units under the greedy, uniform-below, uniform-above, and random allocation strategies for each budget.
- [x] Measure real PPL on sensitivity and test windows with restoration asserted between runs.
- [x] Record `pred_ppl_sens` for greedy allocations and print the aggregate table.
- [x] Check: `results.csv` includes the required columns and the real runs are reported without fabricated values.
- [x] PASS: all evaluations completed with restoration assertions.

## Milestone 6 - README and final outputs
- [x] Write README with purpose, setup instructions, run commands, the results table copied from `results.csv`, one-line note on the additivity gap, and the scale-exclusion note.
- [x] Confirm `results/sens.csv`, `results/heatmap.png`, and `results/results.csv` are created; allocation JSON files are stored under `results/allocation/`.
- [x] PASS: documentation and measured outputs are complete.

## Execution rule
Implement one milestone at a time, run the relevant check, fix the issue, and only then move to the next milestone. If a check fails and the issue cannot be fixed, record the failure in this file and stop without changing config values to hide the problem.
