# tau2-agent-single-call-credit-core

## Purpose

Compare the current turn-credit-v1 recipe, outcome-only GRPO, and
turn-credit-v2 on the same strict-single-v1 Agent SFT. The experiment changes
only reward/advantage credit assignment; it does not start OPD. The consolidated
report is [`TAU2_SINGLE_CALL_CREDIT_TEST.md`](../../doc/TAU2_SINGLE_CALL_CREDIT_TEST.md).

## Configuration

All arms start from
`Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf`
with the v1 User SFT, train seed `1234`, rollout seed `42`, K=8, six prompt
groups per update, LR `2e-6`, K2 KL `0.01`, and 100 updates. The domain quotas
are `3/2/1` for updates 0-9, `3/1/2` for updates 10-19, and `2/2/2` for
updates 20-99 in Telecom/Airline/Retail order. Every arm keeps zero-signal
groups with `TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`. Sglang deterministic
inference is enabled so seed `42` is passed as each group sample's
`sampling_seed`.

| Arm | Credit configuration | Role |
|---|---|---|
| `v1-matched` | `turn-credit-v1` | Fresh matched control with field reward and v1 turn penalties; not raw GRPO. |
| `v2-l000` | `turn-credit-v2`, lambda `0` | Official-outcome-only GRPO control. |
| `v2-l010` | `turn-credit-v2`, lambda `0.1` | Fixed-budget, zero-sum turn-credit-v2 candidate. |

The Agent protocol is `strict-single-v1`: each Assistant message may contain
at most one native tool call, while a task may contain multiple Assistant
turns.

## Jobs

- `v1-matched` smoke: job `1474`,
  [`1474-tau2-credit-v1-matched-smoke-0814-133347670`](jobs/1474-tau2-credit-v1-matched-smoke-0814-133347670/run_20260814_133347670.log)
  (succeeded; runtime-only, sampling seed was not yet applied).
- `v2-l000` smoke: job `1472`,
  [`1472-tau2-credit-v2-l000-smoke-0814-133347370`](jobs/1472-tau2-credit-v2-l000-smoke-0814-133347370/run_20260814_133347370.log)
  (succeeded; runtime-only, sampling seed was not yet applied).
- `v2-l010` smoke: job `1473`,
  [`1473-tau2-credit-v2-l010-smoke-0814-133347522`](jobs/1473-tau2-credit-v2-l010-smoke-0814-133347522/run_20260814_133347522.log)
  (succeeded; runtime-only, sampling seed was not yet applied).
- `v1-matched` deterministic smoke: job `1483`,
  [`1483-tau2-credit-v1-matched-det-smoke-0814-135314295`](jobs/1483-tau2-credit-v1-matched-det-smoke-0814-135314295/run_20260814_135314295.log)
  (stopped before training; retry seed defect found).
- `v2-l000` deterministic smoke: job `1484`,
  [`1484-tau2-credit-v2-l000-det-smoke-0814-135314892`](jobs/1484-tau2-credit-v2-l000-det-smoke-0814-135314892/run_20260814_135314892.log)
  (stopped before training; retry seed defect found).
- `v2-l010` deterministic smoke: job `1485`,
  [`1485-tau2-credit-v2-l010-det-smoke-0814-135315508`](jobs/1485-tau2-credit-v2-l010-det-smoke-0814-135315508/run_20260814_135315508.log)
  (stopped before training; retry seed defect found).
- `v1-matched` corrected deterministic smoke: job `1487`,
  [`1487-tau2-credit-v1-matched-det2-smoke-0814-135610273`](jobs/1487-tau2-credit-v1-matched-det2-smoke-0814-135610273/run_20260814_135610273.log)
  (succeeded; Agent deterministic, User not yet deterministic).
- `v2-l000` corrected deterministic smoke: job `1488`,
  [`1488-tau2-credit-v2-l000-det2-smoke-0814-135610813`](jobs/1488-tau2-credit-v2-l000-det2-smoke-0814-135610813/run_20260814_135610813.log)
  (succeeded; Agent deterministic, User not yet deterministic).
- `v2-l010` corrected deterministic smoke: job `1489`,
  [`1489-tau2-credit-v2-l010-det2-smoke-0814-135611400`](jobs/1489-tau2-credit-v2-l010-det2-smoke-0814-135611400/run_20260814_135611400.log)
  (succeeded; Agent deterministic, User not yet deterministic).
- `v1-matched` Agent+User deterministic smoke: job `1490`,
  [`1490-tau2-credit-v1-matched-det3-smoke-0814-141510510`](jobs/1490-tau2-credit-v1-matched-det3-smoke-0814-141510510/run_20260814_141510510.log)
  (succeeded).
- `v2-l000` Agent+User deterministic smoke: job `1491`,
  [`1491-tau2-credit-v2-l000-det3-smoke-0814-141511072`](jobs/1491-tau2-credit-v2-l000-det3-smoke-0814-141511072/run_20260814_141511072.log)
  (succeeded).
- `v2-l010` Agent+User deterministic smoke: job `1492`,
  [`1492-tau2-credit-v2-l010-det3-smoke-0814-141511644`](jobs/1492-tau2-credit-v2-l010-det3-smoke-0814-141511644/run_20260814_141511644.log)
  (succeeded).
- `v1-matched` stage-a: job `1498`,
  [`1498-tau2-credit-v1-matched-stage-a-0814-143132887`](jobs/1498-tau2-credit-v1-matched-stage-a-0814-143132887/run_20260814_143132887.log)
  (succeeded; iter9).
- `v2-l000` stage-a: job `1499`,
  [`1499-tau2-credit-v2-l000-stage-a-0814-143133587`](jobs/1499-tau2-credit-v2-l000-stage-a-0814-143133587/run_20260814_143133587.log)
  (succeeded; iter9).
- `v2-l010` stage-a: job `1500`,
  [`1500-tau2-credit-v2-l010-stage-a-0814-143134237`](jobs/1500-tau2-credit-v2-l010-stage-a-0814-143134237/run_20260814_143134237.log)
  (succeeded; iter9).
- `v1-matched` stage-b: job `1515`,
  [`1515-tau2-credit-v1-matched-stage-b-0814-152053709`](jobs/1515-tau2-credit-v1-matched-stage-b-0814-152053709/run_20260814_152053709.log)
  (succeeded; iter19).
- `v2-l000` stage-b: job `1516`,
  [`1516-tau2-credit-v2-l000-stage-b-0814-152409950`](jobs/1516-tau2-credit-v2-l000-stage-b-0814-152409950/run_20260814_152409950.log)
  (succeeded; iter19).
- `v2-l010` stage-b: job `1517`,
  [`1517-tau2-credit-v2-l010-stage-b-0814-152410101`](jobs/1517-tau2-credit-v2-l010-stage-b-0814-152410101/run_20260814_152410101.log)
  (succeeded; iter19).
- `v2-l000` stage-final: job `1542`,
  [`1542-tau2-credit-v2-l000-stage-final-0814-160702062`](jobs/1542-tau2-credit-v2-l000-stage-final-0814-160702062/run_20260814_160702062.log)
  (succeeded; iter99).
- `v1-matched` stage-final: job `1543`,
  [`1543-tau2-credit-v1-matched-stage-final-0814-160702570`](jobs/1543-tau2-credit-v1-matched-stage-final-0814-160702570/run_20260814_160702570.log)
  (succeeded; iter99).
- `v2-l010` stage-final: job `1544`,
  [`1544-tau2-credit-v2-l010-stage-final-0814-160702795`](jobs/1544-tau2-credit-v2-l010-stage-final-0814-160702795/run_20260814_160702795.log)
  (succeeded; iter99).
- `v1-matched` iter99 conversion: job `1649`,
  [`1649-tau2-credit-convert-v1-matched-iter99-0814-201902466`](jobs/1649-tau2-credit-convert-v1-matched-iter99-0814-201902466/run_20260814_201902466.log)
  (succeeded).
- `v2-l010` iter99 conversion: job `1650`,
  [`1650-tau2-credit-convert-v2-l010-iter99-0814-201902627`](jobs/1650-tau2-credit-convert-v2-l010-iter99-0814-201902627/run_20260814_201902627.log)
  (succeeded).
- `v2-l000` iter99 conversion: job `1651`,
  [`1651-tau2-credit-convert-v2-l000-iter99-0814-201902780`](jobs/1651-tau2-credit-convert-v2-l000-iter99-0814-201902780/run_20260814_201902780.log)
  (succeeded).
- SFT candidate eval seed301: job `1652`,
  [`1652-tau2-credit-eval-sft-seed301-0814-201902935`](jobs/1652-tau2-credit-eval-sft-seed301-0814-201902935/run_20260814_201902935.log)
  (stopped; superseded because it used deterministic evaluation).
- SFT candidate eval seed300: job `1653`,
  [`1653-tau2-credit-eval-sft-seed300-0814-201903086`](jobs/1653-tau2-credit-eval-sft-seed300-0814-201903086/run_20260814_201903086.log)
  (stopped; superseded because it used deterministic evaluation).
- `v2-l010` candidate eval seed301: job `1663`,
  [`1663-tau2-credit-eval-v2-l010-seed301-0814-202516364`](jobs/1663-tau2-credit-eval-v2-l010-seed301-0814-202516364/run_20260814_202516364.log)
  (stopped; superseded because it used deterministic evaluation).
- `v1-matched` candidate eval seed301: job `1664`,
  [`1664-tau2-credit-eval-v1-matched-seed301-0814-202516531`](jobs/1664-tau2-credit-eval-v1-matched-seed301-0814-202516531/run_20260814_202516531.log)
  (stopped; superseded because it used deterministic evaluation).
- `v1-matched` candidate eval seed300: job `1665`,
  [`1665-tau2-credit-eval-v1-matched-seed300-0814-202516680`](jobs/1665-tau2-credit-eval-v1-matched-seed300-0814-202516680/run_20260814_202516680.log)
  (stopped; superseded because it used deterministic evaluation).
- `v2-l010` candidate eval seed300: job `1666`,
  [`1666-tau2-credit-eval-v2-l010-seed300-0814-202516867`](jobs/1666-tau2-credit-eval-v2-l010-seed300-0814-202516867/run_20260814_202516867.log)
  (stopped; superseded because it used deterministic evaluation).
- SFT historical-sampling rerun seed300: job `1671`,
  [`1671-tau2-credit-eval-v2-sft-seed300-0814-203258852`](jobs/1671-tau2-credit-eval-v2-sft-seed300-0814-203258852/run_20260814_203258852.log)
  (submitted).
- SFT historical-sampling rerun seed301: job `1672`,
  [`1672-tau2-credit-eval-v2-sft-seed301-0814-203259511`](jobs/1672-tau2-credit-eval-v2-sft-seed301-0814-203259511/run_20260814_203259511.log)
  (submitted).
- `v1-matched` historical-sampling rerun seed300: job `1673`,
  [`1673-tau2-credit-eval-v2-v1-matched-seed300-0814-203300169`](jobs/1673-tau2-credit-eval-v2-v1-matched-seed300-0814-203300169/run_20260814_203300169.log)
  (queued).
- `v1-matched` historical-sampling rerun seed301: job `1674`,
  [`1674-tau2-credit-eval-v2-v1-matched-seed301-0814-203300783`](jobs/1674-tau2-credit-eval-v2-v1-matched-seed301-0814-203300783/run_20260814_203300783.log)
  (queued).
- `v2-l000` historical-sampling rerun seed300: job `1675`,
  [`1675-tau2-credit-eval-v2-v2-l000-seed300-0814-203302097`](jobs/1675-tau2-credit-eval-v2-v2-l000-seed300-0814-203302097/run_20260814_203302097.log)
  (queued).
- `v2-l000` historical-sampling rerun seed301: job `1676`,
  [`1676-tau2-credit-eval-v2-v2-l000-seed301-0814-203302708`](jobs/1676-tau2-credit-eval-v2-v2-l000-seed301-0814-203302708/run_20260814_203302708.log)
  (queued).
- `v2-l010` historical-sampling rerun seed300: job `1677`,
  [`1677-tau2-credit-eval-v2-v2-l010-seed300-0814-203304103`](jobs/1677-tau2-credit-eval-v2-v2-l010-seed300-0814-203304103/run_20260814_203304103.log)
  (queued).
- `v2-l010` historical-sampling rerun seed301: job `1678`,
  [`1678-tau2-credit-eval-v2-v2-l010-seed301-0814-203304761`](jobs/1678-tau2-credit-eval-v2-v2-l010-seed301-0814-203304761/run_20260814_203304761.log)
  (queued).

## Evaluation

The SFT baseline and all three iter99 checkpoints use the same official
evaluation: all 100 three-domain test tasks, four trials, seeds 300/301,
temperature `0.6`, `max_steps=200`, strict-single-v1, and the v1 User SFT with
the Qwen parser enabled. Candidate mode follows the historical sampling
semantics: deterministic inference is disabled and the tau2 trial seed is not
forwarded as the Agent's SGLang `sampling_seed`. The SFT baseline is rerun
through the same candidate path; historical summaries are only a sanity check.

## Results

Training `rollout/raw_reward` below is the accepted official binary task
reward; lambda affects token advantages but is not added to this metric.

The Agent+User deterministic smoke passed with one finite update and an iter0
checkpoint for every arm. All 48 behavior trajectories matched exactly across
arms. `v2-l000` had zero weighted local-credit activation; `v2-l010` activated
on 28/48 trajectories with L1 budget `0.1` per active trajectory. All Stage A
runs completed iter9 without a numerical or runtime failure:

| Arm | Mean raw reward | First/last five | Final | Mean grad norm | Mean K2 loss |
|---|---:|---:|---:|---:|---:|
| `v1-matched` | 20.21% | 20.83% / 19.58% | 20.83% | 0.447 | 0.000428 |
| `v2-l000` | 21.25% | 20.00% / 22.50% | 39.58% | 0.303 | 0.000385 |
| `v2-l010` | 20.21% | 20.42% / 20.00% | 22.92% | 0.314 | 0.000475 |

Stage B also completed without a health failure:

| Arm | Stage B raw reward | First/last five | Cumulative 20-update reward | Final grad norm | Final K2 loss |
|---|---:|---:|---:|---:|---:|
| `v1-matched` | 23.96% | 25.83% / 22.08% | 22.08% | 0.342 | 0.00170 |
| `v2-l000` | 23.13% | 26.67% / 19.58% | 22.19% | 0.286 | 0.00164 |
| `v2-l010` | 20.42% | 22.92% / 17.92% | 20.31% | 0.236 | 0.00415 |

The trajectory-dump view is before dynamic filtering. For `v2-l010`, its
Stage A dump has reconstructed local per-token absolute advantage p99 `1.005`
and maximum `19.27`. In the accepted Stage A+B view, local/global total L1 is
`34.7/348.31` (`10.0%`); only `0.235%` of tokens exceed `2.5`, while the
reducer still caps each active trajectory's local L1 at `0.1`. The short-turn
tail is a scale diagnostic rather than evidence that lambda is inactive.

All Stage Final jobs completed 100 finite updates. Accepted training metrics
cover exactly 600 groups and 4,800 trajectories per arm:

| Arm | Raw reward | Last-20 reward | Binary zero-var | Truncation | Mean / max K2 |
|---|---:|---:|---:|---:|---:|
| `v1-matched` | 26.29% | 25.10% | 50.00% | 23.40% | .00898 / .05099 |
| `v2-l000` | 26.77% | 26.88% | 49.50% | 21.44% | .02558 / .19109 |
| `v2-l010` | 26.00% | 28.13% | 51.67% | 24.88% | .01053 / .03750 |

`v2-l010` is the redesigned reward/advantage arm. Its local channel activates
on 1,071/4,800 accepted trajectories and rescues 179/310 binary-zero-variance
groups, reducing truly signal-free groups to 131/600. Every active trajectory
has zero-sum modifier and local reducer L1 exactly `0.1`; local/global L1 is
`107.1/1806.95` (`5.93%`). Per-token absolute local advantage has p99 `.6644`
and max `19.27`, but only `.115%` of tokens exceed `2.5`; the maximum is a
nine-token malformed-JSON turn, while its trajectory budget remains `0.1`.

Training raw reward alone does not select an arm. `v2-l000` has the highest
full-run raw/action/DB metrics but severe rule and late-KL deterioration;
`v2-l010` sharply lowers wrong-namespace, nonexistent-tool, malformed-JSON,
repetition, and too-many-error rates without yet improving full-run raw reward.
The held-out official evaluation decides whether that behavioral gain transfers
to task success.

## Decision

An arm wins only when its two-seed mean pass@1 and pass@4(any) are both
strictly higher than the other two arms. Pass^4 is reported but is not a
selection condition. Exact pass ties use DB accuracy and then action accuracy;
otherwise a mixed result makes no recipe switch and keeps v1. Paired bootstrap
intervals are descriptive, not a hard gate.
