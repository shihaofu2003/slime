# tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl

## Purpose

Name: `tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl`.

Train one mixed-domain and three single-domain strict-single-v1 GRPO agents
from the same SFT with the raw Qwen3 User, turn-credit-v2 lambda `0.1`, and
training `max_steps=120`. Training precedes targeted official evaluation; this
experiment does not start OPD.

## Configuration

All four 8-GPU arms start from
`Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf` and
run 100 updates with LR `2e-6`, K=8, K2 KL `0.01`, train seed `1234`, rollout
seed `42`, the raw `Qwen3-4B-Instruct-2507` User, and strict-single-v1 Agent
protocol. Mixed uses the established `3/2/1`, `3/1/2`, then `2/2/2` quota
schedule; Airline, Retail, and Telecom use fixed `6/0/0`, `0/6/0`, and `0/0/6`
quotas respectively. Every arm keeps zero-signal groups and uses the existing
16,384-token cap and per-sample retry path.

## Jobs

- Mixed — job `1744` (`pt-gktgdi0c`), 8 GPUs:
  [run log](jobs/1744-tau2-sc-v2-raw120-mixed-0815-010031561/run_20260815_010031561.log)
- Airline — job `1742` (`pt-ybado7i5`), 8 GPUs:
  [run log](jobs/1742-tau2-sc-v2-raw120-airline-0815-010031243/run_20260815_010031243.log)
- Retail — job `1743` (`pt-0j1kns7d`), 8 GPUs:
  [run log](jobs/1743-tau2-sc-v2-raw120-retail-0815-010031409/run_20260815_010031409.log)
- Telecom — job `1746` (`pt-h736rzb4`), 8 GPUs:
  [run log](jobs/1746-tau2-sc-v2-raw120-telecom-0815-010059274/run_20260815_010059274.log).
  Submitted through elastic admission after the first three jobs occupied the
  initial 24-GPU allocation.
- Airline resume from iter49 — job `1894` (`pt-ky9jdzrw`), 8 GPUs:
  [run log](jobs/1894-tau2-sc-v2-raw120-airline-resume-0815-0827-0815-082650168/run_20260815_082650168.log)
- Retail resume from iter79 — job `1895` (`pt-35utw94q`), 8 GPUs:
  [run log](jobs/1895-tau2-sc-v2-raw120-retail-resume-0815-0827-0815-082650347/run_20260815_082650347.log)
- Mixed resume from iter49 — job `1896` (`pt-iyaeen1e`), 8 GPUs:
  [run log](jobs/1896-tau2-sc-v2-raw120-mixed-resume-0815-0827-0815-082650516/run_20260815_082650516.log)
- Telecom resume from iter39 — job `1945` (`pt-p9pxvie7`), 8 GPUs:
  [run log](jobs/1945-tau2-sc-v2-raw120-telecom-resume-0815-0931-0815-093038396/run_20260815_093038396.log)

The four original jobs stopped in the same second during rollout generation
and were reported as `REMOTE_FAILED`, with no NaN, OOM, or training exception.
Each resume uses the newest complete distributed checkpoint from its original
job; no completed update is rerun.

After all four iter99 HF outputs were validated, 26 superseded distributed
checkpoint directories and markers were removed, reclaiming about 1.2 TiB.
The four `iter_0000099_hf` directories and rollout state were retained.
- Airline iter99 HF conversion — job `2019` (`pt-iofb47rq`):
  [run log](jobs/2019-tau2-sc-v2-raw120-airline-iter99-convert-0815-1342-0815-134056841/run_20260815_134056841.log)
- Retail iter99 HF conversion — job `2017` (`pt-3ndrsjo0`):
  [run log](jobs/2017-tau2-sc-v2-raw120-retail-iter99-convert-0815-1342-0815-134056456/run_20260815_134056456.log)
- Mixed iter99 HF conversion — job `2018` (`pt-e7my0zce`):
  [run log](jobs/2018-tau2-sc-v2-raw120-mixed-iter99-convert-0815-1342-0815-134056650/run_20260815_134056650.log)
- Mixed iter99 evaluation — jobs `2024` seed300 and `2023` seed301:
  [seed300 log](jobs/2024-tau2-sc-v2-raw120-mixed-eval-s300-0815-1353-0815-135208565/run_20260815_135208565.log),
  [seed301 log](jobs/2023-tau2-sc-v2-raw120-mixed-eval-s301-0815-1353-0815-135208226/run_20260815_135208226.log);
  [seed300 summary](eval/mixed/seed300_summary.json),
  [seed301 summary](eval/mixed/seed301_summary.json)
- Airline target-only submissions — jobs `2020` seed300 and `2021` seed301,
  stopped immediately after the scope was expanded to all three domains:
  [seed300 log](jobs/2020-tau2-sc-v2-raw120-airline-eval-s300-0815-1353-0815-135207632/run_20260815_135207632.log),
  [seed301 log](jobs/2021-tau2-sc-v2-raw120-airline-eval-s301-0815-1353-0815-135207849/run_20260815_135207849.log)
- Retail target-only submissions — jobs `2025` seed300 and `2022` seed301,
  stopped immediately after the scope was expanded to all three domains:
  [seed300 log](jobs/2025-tau2-sc-v2-raw120-retail-eval-s300-0815-1353-0815-135208763/run_20260815_135208763.log),
  [seed301 log](jobs/2022-tau2-sc-v2-raw120-retail-eval-s301-0815-1353-0815-135208025/run_20260815_135208025.log)
- Airline iter99 all-domain evaluation — jobs `2029` seed300 and `2028`
  seed301:
  [seed300 log](jobs/2029-tau2-sc-v2-raw120-airline-all-eval-s300-0815-1356-0815-135423258/run_20260815_135423258.log),
  [seed301 log](jobs/2028-tau2-sc-v2-raw120-airline-all-eval-s301-0815-1356-0815-135423071/run_20260815_135423071.log);
  [seed300 summary](eval/airline/seed300_summary.json),
  [seed301 summary](eval/airline/seed301_summary.json)
- Retail iter99 all-domain evaluation — jobs `2026` seed300 and `2027`
  seed301:
  [seed300 log](jobs/2026-tau2-sc-v2-raw120-retail-all-eval-s300-0815-1356-0815-135422683/run_20260815_135422683.log),
  [seed301 log](jobs/2027-tau2-sc-v2-raw120-retail-all-eval-s301-0815-1356-0815-135422880/run_20260815_135422880.log);
  [seed300 summary](eval/retail/seed300_summary.json),
  [seed301 summary](eval/retail/seed301_summary.json)
- Telecom iter99 HF conversion — job `2129` (`pt-93ua28qx`):
  [run log](jobs/2129-tau2-sc-v2-raw120-telecom-iter99-convert-0815-163921633/run_20260815_163921633.log)
- Telecom iter99 all-domain evaluation — jobs `2140` seed300 and `2139`
  seed301:
  [seed300 log](jobs/2140-tau2-sc-v2-raw120-telecom-all-eval-s300-0815-164316908/run_20260815_164316908.log),
  [seed301 log](jobs/2139-tau2-sc-v2-raw120-telecom-all-eval-s301-0815-164316690/run_20260815_164316690.log)

## Evaluation plan

After training and conversion, evaluate the mixed checkpoint and every expert
on all three domains. Use the raw User, the same strict-single-v1 protocol,
seeds 300/301, four trials, Agent temperature `0.6`, and evaluation
`max_steps=200`. Compare each expert with the mixed checkpoint at the same
update and report cross-domain behavior directly from these full evaluations.

Hourly training snapshots are recorded in [MONITOR.md](MONITOR.md).
