# tau2-sft-official-native-expanded-turn-filtered

Purpose: train and evaluate the turn-quality-filtered official-native Agent SFT
data as a controlled data ablation of the prior Processed SFT.

Name: `tau2-sft-official-native-expanded-turn-filtered`.

## Data ablation

- Control: `tau2-sft-official-native-expanded`, 30,376 supervised turns.
- Treatment: `../tau2-sft-turn-quality-qwen38-xhigh-v1/data/agent_official_native_expanded_turn_filtered.jsonl`, 29,701 turns.
- The treatment removes 675 `drop/drop` turns and otherwise preserves source
  bytes and order.
- Both arms start from raw `Qwen3-4B-Instruct-2507` rather than from an existing
  SFT checkpoint.

## Fixed SFT recipe

The treatment preserves the Processed SFT recipe: 8 GPUs, batch size 16, two
epochs, cosine `1e-5` to `1e-6` learning rate with 10% warmup, training seed
1234, `qwen3_full` target-only loss, and a 16,384-token dynamic-batch ceiling.
It runs 3,712 updates and retains checkpoints at updates 400, 800, 1200, 1600,
1856, 2000, 2400, 2800, 3200, 3600, and 3712.

## Evaluation

Every checkpoint is evaluated first at seed 300 with the Processed SFT
official-native protocol: Airline/Retail/Telecom quotas 20/40/40, four trials,
Agent temperature 0.6, top-p 1.0, 1,200 output tokens, and 200 steps. Selection
uses overall pass@1, then pass^4, then the earlier checkpoint. The selected and
final checkpoints are repeated at seed 301. The primary ablation comparison
matches both arms at update 3,600. Equal-epoch final and selected-checkpoint
comparisons are retained separately. All report pass@1, pass@4(any), pass^4,
action accuracy, and DB accuracy.

## Tasks

- Job `16023` / `pt-fltjc2st` runs the 8-GPU, two-epoch filtered-data SFT with
  112 CPUs and 1,584 GB at normal priority.
  It reached update 1095 and saved complete updates 400 and 800 before a
  cluster-side interruption at 10:25.
  [Run log](jobs/16023-sft-full-2epoch-0908-092228328/run_*_20260908_092228328.log).
- Job `16049` / `pt-538n71lu` was the first update-800 resume attempt. It was
  stopped during setup after unrelated jobs filled the account's 24-GPU quota;
  no training update ran.
  [Run log](jobs/16049-sft-resume-iter799-0908-113018433/run_*_20260908_113018433.log).
- Job `16050` / `pt-veqzq3yu` converts `iter_0000799` to Hugging Face format.
  [Run log](jobs/16050-convert-iter_0000799-hf-0908-113104895/run_*_20260908_113104895.log).
- Job `16051` / `pt-ccftct8p` converts `iter_0000399` to Hugging Face format.
  [Run log](jobs/16051-convert-iter_0000399-hf-0908-113105388/run_*_20260908_113105388.log).
- Job `16060` was the first `iter_0000399` seed-300 evaluation attempt. It was
  stopped during setup for the same quota conflict.
  [Run log](jobs/16060-eval-iter_0000399-seed300-0908-113524565/run_*_20260908_113524565.log).
- Job `16061` was the first `iter_0000799` seed-300 evaluation attempt. It was
  stopped during setup for the same quota conflict.
  [Run log](jobs/16061-eval-iter_0000799-seed300-0908-113525009/run_*_20260908_113525009.log).
- Job `16065` / `pt-0dg2c5ty` is the second update-800 resume attempt.
  [Run log](jobs/16065-sft-resume-iter799-r2-0908-114511410/run_*_20260908_114511410.log).
- Job `16066` / `pt-m8wlsh1f` is the second `iter_0000799` seed-300 evaluation attempt.
  [Run log](jobs/16066-eval-iter_0000799-seed300-r2-0908-114511856/run_*_20260908_114511856.log).
- Job `16067` / `pt-tndhrf6s` is the second `iter_0000399` seed-300 evaluation attempt.
  [Run log](jobs/16067-eval-iter_0000399-seed300-r2-0908-114512290/run_*_20260908_114512290.log).
- Job `16089` converts `iter_0001199` to Hugging Face format.
  [Run log](jobs/16089-convert-iter_0001199-hf-0908-122349222/run_*_20260908_122349222.log).
- Job `16090` evaluates `iter_0001199` at seed 300.
  [Run log](jobs/16090-eval-iter_0001199-seed300-0908-122625871/run_*_20260908_122625871.log).
- Job `16095` / `pt-7v2ylw24` converts `iter_0001599` to Hugging Face format.
  [Run log](jobs/16095-convert-iter_0001599-hf-0908-125615986/run_*_20260908_125615986.log).
- Job `16096` / `pt-eoowvb6u` converts `iter_0001855` to Hugging Face format.
  [Run log](jobs/16096-convert-iter_0001855-hf-0908-125616922/run_*_20260908_125616922.log).
- Job `16097` / `pt-qnzvl6mi` converts `iter_0001999` to Hugging Face format.
  [Run log](jobs/16097-convert-iter_0001999-hf-0908-125617863/run_*_20260908_125617863.log).
- Job `16098` evaluates `iter_0001855` at seed 300.
  [Run log](jobs/16098-eval-iter_0001855-seed300-0908-130011021/run_*_20260908_130011021.log).
- Job `16099` evaluates `iter_0001599` at seed 300.
  [Run log](jobs/16099-eval-iter_0001599-seed300-0908-130239387/run_*_20260908_130239387.log).
- Job `16100` evaluates `iter_0001999` at seed 300.
  [Run log](jobs/16100-eval-iter_0001999-seed300-0908-130240323/run_*_20260908_130240323.log).
- Job `16102` converts `iter_0002399` to Hugging Face format.
  [Run log](jobs/16102-convert-iter_0002399-hf-0908-130903768/run_*_20260908_130903768.log).
- Job `16107` evaluates `iter_0002399` at seed 300.
  [Run log](jobs/16107-eval-iter_0002399-seed300-0908-131137591/run_*_20260908_131137591.log).
- Job `16110` converts `iter_0002799` to Hugging Face format.
  [Run log](jobs/16110-convert-iter_0002799-hf-0908-132351292/run_*_20260908_132351292.log).
- Job `16111` evaluates `iter_0002799` at seed 300.
  [Run log](jobs/16111-eval-iter_0002799-seed300-0908-132624291/run_*_20260908_132624291.log).
- Job `16113` converts `iter_0003199` to Hugging Face format.
  [Run log](jobs/16113-convert-iter_0003199-hf-0908-133834288/run_*_20260908_133834288.log).
- Job `16116` evaluates `iter_0003199` at seed 300.
  [Run log](jobs/16116-eval-iter_0003199-seed300-0908-134141749/run_*_20260908_134141749.log).
- Job `16131` converts `iter_0003599` to Hugging Face format.
  [Run log](jobs/16131-convert-iter_0003599-hf-0908-135348964/run_*_20260908_135348964.log).
- Job `16140` evaluates `iter_0003599` at seed 300.
  [Run log](jobs/16140-eval-iter_0003599-seed300-0908-135709690/run_*_20260908_135709690.log).
- Job `16141` converts final `iter_0003711` to Hugging Face format.
  [Run log](jobs/16141-convert-iter_0003711-final-hf-0908-135823808/run_*_20260908_135823808.log).
- Job `16149` evaluates final `iter_0003711` at seed 300.
  [Run log](jobs/16149-eval-iter_0003711-seed300-0908-140150278/run_*_20260908_140150278.log).
- Job `16192` evaluates selected `iter_0003599` at seed 301.
  [Run log](jobs/16192-eval-iter_0003599-seed301-0908-143321244/run_*_20260908_143321244.log).
- Job `16193` evaluates final `iter_0003711` at seed 301.
  [Run log](jobs/16193-eval-iter_0003711-seed301-0908-143324376/run_*_20260908_143324376.log).
- Job `16206` builds the checkpoint-selection and Processed-SFT ablation report.
  [Run log](jobs/16206-summarize-ablation-0908-150601942/run_*_20260908_150601942.log).
- Job `16214` / `pt-e1s4swwh` evaluates Processed SFT `iter_0003599` at
  seed 301 for the equal-update, two-seed comparison.
  [Run log](jobs/16214-eval-processed-iter_0003599-seed301-0908-153549508/run_*_20260908_153549508.log).
- Job `16233` / `pt-73zv054q` rebuilds the ablation report with equal-update,
  equal-epoch final, and selected-checkpoint comparisons.
  [Run log](jobs/16233-summarize-equal-update-ablation-0908-160747650/run_*_20260908_160747650.log).

## Result

The initial training job was interrupted by a shared cluster event after
update 1095. Job `16065` resumed from complete update 800 and finished all
3,712 updates with finite final loss and gradient norm. All eleven checkpoints
were converted and evaluated at seed 300; the selected and final checkpoints
were also evaluated at seed 301. Every completed evaluation contains 100 tasks,
400 trajectories, and zero infrastructure errors. The Processed-SFT update-3,600
control was also evaluated at both seeds. Report job `16233` succeeded.

`iter_0003599` is selected: it ties final `iter_0003711` on seed-300 pass@1 and
pass^4, so the earlier checkpoint wins the fixed tie-break.

Primary equal-update comparison (`iter_0003599` versus `iter_0003599`):

| Two-seed mean | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---|---:|---:|---:|---:|---:|
| Turn-filtered | 52.12% | 80.00% | 21.50% | 73.00% | 42.31% |
| Processed SFT | 55.75% | 84.00% | 25.00% | 74.63% | 43.62% |
| Difference | -3.63pp | -4.00pp | -3.50pp | -1.63pp | -1.31pp |

The equal-epoch final comparison (`iter_0003711` versus `iter_0003795`) is
`-6.87/-3.00/-6.50/-3.68/-1.64pp`. The selected-checkpoint comparison
(`iter_0003599` versus Processed final `iter_0003795`) is
`-3.63/-1.00/-6.50/-1.76/-1.81pp`, in the same metric order as the table.
Equal-update controls optimizer steps and batch count; equal-epoch controls two
passes over each arm's differently sized dataset.

The turn-quality filter does not win this controlled data ablation; retain the
Processed SFT data recipe. See [the full checkpoint curve and domain
breakdown](CHECKPOINT_EVAL.md) and [machine-readable report](CHECKPOINT_EVAL.json).
