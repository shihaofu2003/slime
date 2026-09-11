# tau2-sft-raw-all-max16384

Purpose: train a raw-data control from every AReaL Tau2 SFT row that fits the
16,384-token training limit, then compare its final checkpoint with the
processed official-native SFT under the same three-domain evaluation protocol.

## Tasks

- Job `12680` failed before tokenization completed because numeric Telecom
  dialog IDs required policy-based domain inference; no output was promoted.
  [Run log](jobs/12680-build-data-0903-093525689/run_*_20260903_093525689.log).
- Job `12681` completed the data build with policy-based Telecom inference.
  It kept 32,548 of 33,531 rows, including 24,120 double-success rows and
  8,428 rows with at least one failure label; the maximum retained length is
  16,379 tokens. [Run log](jobs/12681-build-data-v2-0903-094159170/run_*_20260903_094159170.log).
- Job `12684` completed the longest-32 two-update 8-GPU smoke. Steps 0 and 1
  had finite loss (`0.8333`, `0.8912`) and gradient norm (`17.051`, `20.571`),
  with zero truncation and no OOM or empty loss mask.
  [Run log](jobs/12684-sft-smoke-longest32-0903-095546104/run_*_20260903_095546104.log).
- Job `12686` completed the two-epoch, 4,068-update 8-GPU training in 1.78
  hours. Every logged loss and gradient norm is finite, all rollouts have zero
  truncation, and the final step has loss `0.1726` and gradient norm `2.3722`.
  The checkpoint root contains only `iter_0002033` and `iter_0004067` plus
  rollout metadata.
  [Run log](jobs/12686-sft-full-2epoch-0903-100154599/run_*_20260903_100154599.log).
- Job `12721` converted only `iter_0004067` to `final_hf` with one GPU, eight
  CPUs, and 128 GB memory. Both safetensors shards are readable and their 398
  tensors exactly match the index.
  [Run log](jobs/12721-convert-final-hf-0903-115138791/run_*_20260903_115138791.log).
- Jobs `12722` and `12723` completed final evaluation at seeds 300 and 301.
  Each has 100 tasks, 400 simulations, and zero infrastructure errors. Their
  overall pass@1/pass@4(any)/pass^4 scores are `53.25/82.00/25.00%` and
  `53.00/83.00/21.00%`, respectively.
  [Seed-300 run log](jobs/12722-eval-final-seed300-0903-115606051/run_*_20260903_115606051.log) and
  [seed-301 run log](jobs/12723-eval-final-seed301-0903-115611726/run_*_20260903_115611726.log).
- Job `12752` strictly validated both evaluations and built the
  [final report](FINAL_REPORT.md) and [machine-readable report](FINAL_REPORT.json).
  [Run log](jobs/12752-summarize-final-0903-122823526/run_*_20260903_122823526.log).

## Result

| Agent | Seed 300 pass@1 / pass@4(any) / pass^4 | Seed 301 | Two-seed mean |
|---|---:|---:|---:|
| Raw-all SFT | 53.25 / 82.00 / 25.00% | 53.00 / 83.00 / 21.00% | 53.12 / 82.50 / 23.00% |
| Processed SFT | 56.25 / 81.00 / 30.00% | 55.25 / 81.00 / 26.00% | 55.75 / 81.00 / 28.00% |
| Processed minus raw-all | +3.00 / -1.00 / +5.00pp | +2.25 / -2.00 / +5.00pp | +2.62 / -1.50 / +5.00pp |

On the two-seed mean, processed SFT also changes action accuracy by `+0.28pp`
and DB accuracy by `-1.91pp`. It improves pass@1 and pass^4 but not
pass@4(any); the detailed domain, behavior, and runtime breakdown is in the
final report.

Across the 800 matched trajectories, processed versus raw-all reduces
`max_steps` from 22 to 9, `too_many_errors` from 35 to 18, invalid-argument
calls from 31 to 7, and tool-result errors from 776 to 564. The aggregate
behavior gain is Telecom-led; Retail tool-result errors increase from 221 to
253 and `too_many_errors` from 2 to 11.

The [quality audit](QUALITY_AUDIT.md) finds that raw-all contains 25.89%
failure-labeled rows, 33.23% multi-call targets among tool-call rows, and 62.21%
of rows with a multi-call prefix; processed is zero on all three and all 30,376
rows pass its atomic target, current tool-schema, loss-mask, and call/result
pairing checks. A 100,000-draw paired task bootstrap over the two-seed means
gives overall pass@1 `+2.62pp [-2.12,+7.38]` and pass^4
`+5.00pp [-1.50,+11.00]`; Telecom pass@1 is the only positive domain-level
delta excluding zero at `+9.69pp [+2.81,+17.19]`.

## Fixed recipe

The sole source is `datasets/AReaL-tau2-data/tau2_sft_train.jsonl`. The builder
retains successful and failed labels and writes each selected source line
unchanged. It removes only rows whose native Qwen rendering of `messages +
answer` exceeds 16,384 tokens, after omitting `thinking` and `reasoning` from
the rendered training view. Historical Assistant turns are masked; only the
top-level `answer` content and tool calls receive loss.

Training starts from raw `Qwen3-4B-Instruct-2507` torch-dist weights on eight
GPUs with DP=8, TP/PP/CP=1, batch size 16, two epochs, 4,068 updates, cosine
`1e-5` to `1e-6` learning rate, 10% warmup, and `qwen3_full`. Only epoch one
`iter_0002033` and final `iter_0004067` are saved.

The final Agent is evaluated at seeds 300 and 301 with official-native
`llm_agent`, non-thinking Qwen3.6-27B User, Airline/Retail/Telecom quotas
20/40/40, four trials, temperature 0.6, top-p 1.0, 1,200 output tokens, and
200 steps. Banking is excluded.

## Data

| Domain | Source | Drop >16K | Keep |
|---|---:|---:|---:|
| Airline | 12,842 | 981 | 11,861 |
| Retail | 11,395 | 0 | 11,395 |
| Telecom | 9,294 | 2 | 9,292 |
| Overall | 33,531 | 983 | 32,548 |
