# tau2-sft-official-native-expanded

Purpose: build validated official-native Agent SFT data, train two epochs from
raw Qwen3-4B-Instruct-2507, and evaluate every retained checkpoint with one
matched three-domain protocol.

## Tasks

- Job `12439` (`build-data`) was rejected before execution because the requested
  1-GPU/8-CPU/32-GB shape was unavailable; it has no run log. [Scheduler log](jobs/12439-build-data-0902-182628091/jobm.log).
- Job `12440` (`build-data-cpu`) failed after setup because the submitted shell
  snapshot resolved the Python file relative to the snapshot directory. [Run log](jobs/12440-build-data-cpu-0902-182654989/run_0_20260902_182654989.log).
- Job `12442` (`build-data-cpu-v2`) used the available 0-GPU/8-CPU/32-GB shape
  and completed the build plus independent full and smoke validation. [Run log](jobs/12442-build-data-cpu-v2-0902-183029113/run_0_20260902_183029113.log).
- Job `12566` (`sft-smoke-longest32`) completed without OOM or non-finite
  metrics, but exposed that automatic resume indexing skipped rollout id 0 and
  performed only one update; it is not the accepted smoke. [Run log](jobs/12566-sft-smoke-longest32-0903-012829255/run_0_20260903_012829255.log).
- Job `12568` (`sft-smoke-longest32-v2`) explicitly starts at rollout id 0 and
  completed the accepted 8-GPU smoke: two finite updates, zero truncation, and
  one final `iter_0000001` checkpoint. [Run log](jobs/12568-sft-smoke-longest32-v2-0903-013518701/run_0_20260903_013518701.log).
- Job `12569` (`smoke-convert-hf`) was rejected before execution because the
  requested 0-GPU/8-CPU/64-GB shape is unavailable. [Scheduler log](jobs/12569-smoke-convert-hf-0903-014121180/jobm.log).
- Job `12570` (`smoke-convert-hf-v2`) was stopped before execution while queued
  on a 0-GPU/16-CPU/64-GB shape; it was superseded by the single-GPU conversion.
  [Scheduler log](jobs/12570-smoke-convert-hf-v2-0903-014141802/jobm.log).
- Job `12584` (`smoke-convert-hf-gpu`) used 1 GPU, 8 CPUs, and 128 GB and
  successfully converted the accepted smoke checkpoint to `final_hf`.
  [Run log](jobs/12584-smoke-convert-hf-gpu-0903-014909310/run_0_20260903_014909310.log).
- Job `12586` (`smoke-eval-official-native`) completed the three-domain,
  one-task per domain native evaluation against the converted smoke checkpoint.
  [Run log](jobs/12586-smoke-eval-official-native-0903-015553644/run_0_20260903_015553644.log).
- Job `12587` (`sft-full-2epoch`) completed the validated 8-GPU, two-epoch
  formal training from the raw Instruct torch-dist checkpoint. [Run log](jobs/12587-sft-full-2epoch-0903-020811199/run_0_20260903_020811199.log).
- Job `12612` (`convert-iter-0000399-hf`) converted `iter_0000399` with the
  selected 1-GPU/8-CPU/128-GB job shape. [Run log](jobs/12612-convert-iter-0000399-hf-0903-050628042/run_0_20260903_050628042.log).
- Job `12613` (`convert-iter-0000799-hf`) converted `iter_0000799` with the same
  single-GPU shape. [Run log](jobs/12613-convert-iter-0000799-hf-0903-051025012/run_0_20260903_051025012.log).
- Job `12615` (`convert-iter-0001199-hf`) converted `iter_0001199` with the same
  single-GPU shape. [Run log](jobs/12615-convert-iter-0001199-hf-0903-051344868/run_0_20260903_051344868.log).
- Job `12616` (`convert-iter-0001599-hf`) converted `iter_0001599` with the same
  single-GPU shape. [Run log](jobs/12616-convert-iter-0001599-hf-0903-051716401/run_0_20260903_051716401.log).
- Job `12617` (`convert-iter-0001897-hf`) converted the epoch-one boundary
  `iter_0001897` with the same single-GPU shape. [Run log](jobs/12617-convert-iter-0001897-hf-0903-052021022/run_0_20260903_052021022.log).
- Job `12618` (`convert-iter-0001999-hf`) converted `iter_0001999` with the same
  single-GPU shape. [Run log](jobs/12618-convert-iter-0001999-hf-0903-052325058/run_0_20260903_052325058.log).
- Job `12619` (`convert-iter-0002399-hf`) converted `iter_0002399` with the same
  single-GPU shape. [Run log](jobs/12619-convert-iter-0002399-hf-0903-052627959/run_0_20260903_052627959.log).
- Job `12620` (`convert-iter-0002799-hf`) converted `iter_0002799` with the same
  single-GPU shape. [Run log](jobs/12620-convert-iter-0002799-hf-0903-053000848/run_0_20260903_053000848.log).
- Job `12621` (`convert-iter-0003199-hf`) converted `iter_0003199` with the same
  single-GPU shape. [Run log](jobs/12621-convert-iter-0003199-hf-0903-053302322/run_0_20260903_053302322.log).
- Job `12622` (`convert-iter-0003599-hf`) converted `iter_0003599` with the same
  single-GPU shape. [Run log](jobs/12622-convert-iter-0003599-hf-0903-053605806/run_0_20260903_053605806.log).
- Job `12639` (`convert-final-hf`) converted `iter_0003795` to `final_hf` with
  the same single-GPU shape. [Run log](jobs/12639-convert-final-hf-0903-053948739/run_0_20260903_053948739.log).
- Job `12643` evaluates `iter_0000399` at seed 300. [Run log](jobs/12643-eval-iter_0000399-seed300-0903-064030112/run_0_20260903_064030112.log).
- Job `12644` evaluates `iter_0000799` at seed 300. [Run log](jobs/12644-eval-iter_0000799-seed300-0903-064030880/run_0_20260903_064030880.log).
- Job `12645` evaluates `iter_0001199` at seed 300. [Run log](jobs/12645-eval-iter_0001199-seed300-0903-064031611/run_0_20260903_064031611.log).
- Job `12646` evaluates `iter_0001599` at seed 300. [Run log](jobs/12646-eval-iter_0001599-seed300-0903-064037355/run_0_20260903_064037355.log).
- Job `12647` evaluates `iter_0001897` at seed 300. [Run log](jobs/12647-eval-iter_0001897-seed300-0903-064043139/run_0_20260903_064043139.log).
- Job `12648` evaluates `iter_0001999` at seed 300. [Run log](jobs/12648-eval-iter_0001999-seed300-0903-064054413/run_0_20260903_064054413.log).
- Job `12649` evaluates `iter_0002399` at seed 300. [Run log](jobs/12649-eval-iter_0002399-seed300-0903-064056724/run_0_20260903_064056724.log).
- Job `12650` evaluates `iter_0002799` at seed 300. [Run log](jobs/12650-eval-iter_0002799-seed300-0903-064103487/run_0_20260903_064103487.log).
- Job `12651` evaluates `iter_0003199` at seed 300. [Run log](jobs/12651-eval-iter_0003199-seed300-0903-064110137/run_0_20260903_064110137.log).
- Job `12652` evaluates `iter_0003599` at seed 300. [Run log](jobs/12652-eval-iter_0003599-seed300-0903-064117089/run_0_20260903_064117089.log).
- Job `12653` evaluates final `iter_0003795` at seed 300. [Run log](jobs/12653-eval-iter_0003795-seed300-0903-064123918/run_0_20260903_064123918.log).
- Job `12656` strictly validates and summarizes all 11 seed-300 results. [Run log](jobs/12656-summarize-seed300-0903-072946100/run_0_20260903_072946100.log).
- Job `12657` repeats selected/final `iter_0003795` at seed 301. [Run log](jobs/12657-eval-iter_0003795-seed301-0903-073204022/run_0_20260903_073204022.log).
- Job `12659` strictly validates seed 301 and writes the final combined report.
  [Run log](jobs/12659-summarize-final-0903-080848295/run_0_20260903_080848295.log).

## Result

The converter read 33,531 source rows, found 24,816 rows in consistent
double-success dialogs, expanded 31,679 targets, dropped 1,303 over-cap rows,
and retained 30,376 rows: 13,783 Airline, 13,614 Retail, and 2,979 Telecom.
Targets comprise 13,979 text and 16,397 tool-call rows. The maximum retained
length is 16,382 tokens. The smoke output contains the 32 longest retained rows;
both files passed the standalone validator.

The accepted smoke completed two finite training updates with zero truncation,
converted to a loadable two-shard HF model, and produced all three requested
official-native trajectories with no infrastructure error.

The formal run completed all 3,796 updates. Every logged loss and gradient norm
is finite, and the final update has loss 0.04980 and gradient norm 1.41144. The
checkpoint root contains exactly the ten planned intermediate points and
`iter_0003795`; each has complete metadata and 16 torch-dist shards.

All 11 formal checkpoints were converted by successful single-GPU jobs. Every
HF output has a valid Qwen3 configuration, 398 indexed tensors, and two readable
safetensors shards; the final training point is named `final_hf`.

All 11 seed-300 summaries passed strict validation with 100 tasks, 400 unique
task/trial trajectories, the 20/40/40 domain quotas, and zero infrastructure
errors. The selected checkpoint is final `iter_0003795`, which has the highest
seed-300 pass@1. [Full checkpoint table](CHECKPOINT_EVAL.md) and
[machine-readable diagnostics](CHECKPOINT_EVAL.json).

| Seed | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. | wall |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% | 21.36 min |
| 301 | 55.25% | 81.00% | 26.00% | 75.25% | 44.53% | 25.28 min |

| Seed | Domain | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Airline | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| 300 | Retail | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| 300 | Telecom | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |
| 301 | Airline | 46.25% | 70.00% | 30.00% | 47.81% | 46.25% |
| 301 | Retail | 58.75% | 80.00% | 40.00% | 87.63% | 63.87% |
| 301 | Telecom | 56.25% | 87.50% | 10.00% | 71.04% | 23.49% |

Selected-checkpoint termination counts are `user_stop/too_many_errors/max_steps`
389/8/3 at seed 300 and 384/10/6 at seed 301. Nonexistent tool calls are zero;
invalid-argument calls are 6 and 1, respectively. Job `11878` remains a
historical raw-Instruct reference only because it predates official-native
`llm_agent` evaluation.

## Fixed recipe

The full run uses 8 GPUs, batch size 16, two epochs (3,796 updates), cosine
`1e-5` to `1e-6` learning rate with 10% warmup, `qwen3_full` target-only loss,
and a 16,384-token dynamic-batch ceiling. It saves at updates 400, 800, 1200,
1600, 1898, 2000, 2400, 2800, 3200, 3600, and 3796. The final checkpoint is
converted to `final_hf`; the other ten are converted to matching `iter_*_hf`
directories.

Every full eval uses official-native `llm_agent`, the non-thinking Qwen3.6-27B
User, Airline/Retail/Telecom test quotas 20/40/40, four trials, seed 300,
temperature 0.6, top-p 1.0, 1,200 output tokens, and 200 steps. Selection is by
overall pass@1, then pass^4, then the earlier checkpoint. The selected point and
final are repeated at seed 301.
