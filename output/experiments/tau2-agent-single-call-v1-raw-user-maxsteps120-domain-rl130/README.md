# tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130

Purpose: record the invalid submissions that mistakenly restarted one mixed
and three single-domain strict-single-v1 agents from SFT instead of iter99.

Status: jobs `2736`, `2737`, and `2738` were stopped after the mismatch was
identified; job `2740` had already been cancelled while queued. No checkpoint
from this experiment is a valid iter99 continuation.

## Configuration

These jobs loaded
`Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf`.
That contradicts the required source
`Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099_hf`.

Mixed uses quota `3/2/1` for updates 0-9, `3/1/2` for updates 10-19, and
`2/2/2` afterward in Telecom/Airline/Retail order. Airline, Retail, and
Telecom use fixed `0/6/0`, `0/0/6`, and `6/0/0`. Stage A/B restore optimizer
at iter9/19; updates 20-129 then run in one process. Thus iter109, iter119, and
iter129 are checkpoints from one continuous optimizer trajectory. An
interrupted final stage resumes only from its latest complete checkpoint.

## Evaluation

No conversion or evaluation will use these stopped lineages.

## Jobs

- Mixed — stopped job `2737`, 8 GPUs:
  [run log](jobs/2737-tau2-sc-v1-raw120-rl130-mixed-0816-154251369/run_20260816_154251369.log).
- Retail — stopped job `2736` (`pt-dfppz6ui`), 8 GPUs:
  [run log](jobs/2736-tau2-sc-v1-raw120-rl130-retail-0816-154250840/run_20260816_154250840.log).
- Telecom — stopped job `2738`, 8 GPUs:
  [run log](jobs/2738-tau2-sc-v1-raw120-rl130-telecom-0816-154251722/run_20260816_154251722.log).
- Airline — job `2740`, cancelled while queued before training:
  [submit log](jobs/2740-tau2-sc-v1-raw120-rl130-airline-0816-154334779/submit_20260816_154334779.log).

All four use normal priority. Jobs waiting beyond current quota remain in the
job-manager FIFO queue.
