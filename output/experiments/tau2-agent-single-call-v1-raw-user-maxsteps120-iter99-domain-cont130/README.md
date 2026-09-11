# tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130

Purpose: continue the RAW-User maxsteps120 strict-single-v1 iter99 Agent for
30 updates under mixed, Retail, and Telecom sampling, then evaluate
iter109/119/129 with one matched official protocol.

## Configuration

All arms initialize from
`Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099_hf`.
The original iter99 distributed optimizer checkpoint no longer exists, so each
arm creates one fresh optimizer at update 100. A single `train.py` process runs
updates 100–129 continuously and only saves at iter109, iter119, and iter129;
saving does not reload model, optimizer, RNG, or scheduler state. An
infrastructure retry may resume from the latest complete saved checkpoint.

Fixed settings are the raw `Qwen3-4B-Instruct-2507` User,
strict-single-v1, turn-credit-v1, training `max_steps=120`, LR `2e-6`, K=8,
K2 KL `0.01`, train seed `1234`, rollout seed `42`, the established field
reward and penalties, zero-signal replacement, and non-deterministic SGLang
sampling. Mixed uses the established post-iter99 `2/2/2` quota; Retail and
Telecom use fixed `0/0/6` and `6/0/0` quotas in Telecom/Airline/Retail order.

## Jobs

- Retail — job `2796` (`pt-dps4mejw`), 8 GPUs, normal priority:
  [run log](jobs/2796-tau2-sc-v1-raw120-i99-i129-retail-0816-170031790/run_20260816_170031790.log).
- Mixed — job `2797`, 8 GPUs, normal priority:
  [run log](jobs/2797-tau2-sc-v1-raw120-i99-i129-mixed-0816-170032129/run_20260816_170032129.log).
- Telecom — job `2798`, 8 GPUs, normal priority:
  [run log](jobs/2798-tau2-sc-v1-raw120-i99-i129-telecom-0816-170032464/run_20260816_170032464.log).
- Airline remains excluded after job `2740` was cancelled.

## Evaluation

Convert iter109/119/129 and evaluate all three domains with the raw User,
seeds 300/301, four trials, Agent temperature `0.6`, and evaluation
`max_steps=200`. Compare each domain arm with the same-iteration mixed control.
