# tau2-agent-single-call-v1-rl-crossed-parser-on

Purpose: fill the five RL crossed cells in the strict-single-v1 User-model
matrix with the User tool-call parser explicitly enabled at seeds `300` and
`301`.

## Protocol

- Agent: strict-single-v1 checkpoints from the v1 SFT/RAW-User RL lineages.
- User: either the raw Qwen3-4B-Instruct-2507 model or v1 User SFT, as named by
  the cell; both are served with `USER_SGLANG_EXTRA_ARGS="--tool-call-parser qwen"`.
- Domains: airline, retail, telecom; test split; 100 tasks; four trials;
  Agent temperature `0.6`; evaluation `max_steps=200`.
- Seeds: `300` and `301`; each task uses two GPUs (one Agent server and one
  User server).

## Tasks

- Job `1284` / V1-RL iter99 with RAW User — [run log](jobs/1284-v1-rl99-raw-user-parser-on-0813-212625797/run_20260813_212625797.log)
  (succeeded; [summary](eval/v1rl99-raw/seed300_summary.json)).
- Job `1285` / RAW-RL iter99 (max60) with V1 User — [run log](jobs/1285-raw-rl99-v1-user-parser-on-0813-212626712/run_20260813_212626712.log)
  (succeeded; [summary](eval/rawrl99-v1/seed300_summary.json)).
- Job `1286` / RAW-RL iter99 (max120) with V1 User — [run log](jobs/1286-raw-rl99-max120-v1-user-parser-on-0813-212627857/run_20260813_212627857.log)
  (succeeded; [summary](eval/rawrl99max120-v1/seed300_summary.json)).
- Job `1287` / RAW-RL iter199 (max60) with V1 User — [run log](jobs/1287-raw-rl199-v1-user-parser-on-0813-212628570/run_20260813_212628570.log)
  (succeeded; [summary](eval/rawrl199-v1/seed300_summary.json)).
- Job `1288` / RAW-RL iter199 (max120) with V1 User — [run log](jobs/1288-raw-rl199-max120-v1-user-parser-on-0813-212629273/run_20260813_212629273.log)
  (succeeded; [summary](eval/rawrl199max120-v1/seed300_summary.json)).
- Job `1302` / V1-RL iter99 with RAW User, seed301 — [run log](jobs/1302-v1-rl99-raw-user-parser-on-s301-0814-011408356/run_20260814_011408356.log)
  (succeeded; [summary](eval/v1rl99-raw/seed301_summary.json)).
- Job `1303` / RAW-RL iter99 (max60) with V1 User, seed301 — [run log](jobs/1303-raw-rl99-v1-user-parser-on-s301-0814-011408895/run_20260814_011408895.log)
  (succeeded; [summary](eval/rawrl99-v1/seed301_summary.json)).
- Job `1304` / RAW-RL iter99 (max120) with V1 User, seed301 — [run log](jobs/1304-raw-rl99-max120-v1-user-parser-on-s301-0814-011409442/run_20260814_011409442.log)
  (succeeded; [summary](eval/rawrl99max120-v1/seed301_summary.json)).
- Job `1305` / RAW-RL iter199 (max60) with V1 User, seed301 — [run log](jobs/1305-raw-rl199-v1-user-parser-on-s301-0814-011410748/run_20260814_011410748.log)
  (succeeded; [summary](eval/rawrl199-v1/seed301_summary.json)).
- Job `1306` / RAW-RL iter199 (max120) with V1 User, seed301 — [run log](jobs/1306-raw-rl199-max120-v1-user-parser-on-s301-0814-011411339/run_20260814_011411339.log)
  (succeeded; [summary](eval/rawrl199max120-v1/seed301_summary.json)).

## Results

All ten jobs completed 400/400 simulations with zero infrastructure errors per
seed. Each cell is the seed300/301 mean pass@1 / pass@4(any) / pass^4; controls
use the same Agent with the evaluation User matching its rollout User.

| Agent | Crossed evaluation | Crossed result | Matched-User control | Crossed - control |
|---|---|---:|---:|---:|
| V1-RL iter99 | RAW User | 25.25 / 50.50 / 5.50% | 27.50 / 53.50 / 10.00% | -2.25 / -3.00 / -4.50pp |
| RAW-RL iter99 max60 | V1 User | 24.25 / 49.50 / 6.00% | 29.25 / 54.00 / 6.50% | -5.00 / -4.50 / -0.50pp |
| RAW-RL iter99 max120 | V1 User | 25.25 / 51.00 / 6.00% | 28.38 / 56.50 / 9.50% | -3.13 / -5.50 / -3.50pp |
| RAW-RL iter199 max60 | V1 User | 25.00 / 53.00 / 6.00% | 31.50 / 58.00 / 8.00% | -6.50 / -5.00 / -2.00pp |
| RAW-RL iter199 max120 | V1 User | 25.25 / 49.00 / 5.50% | 30.13 / 58.50 / 7.00% | -4.88 / -9.50 / -1.50pp |

Every crossed two-seed result is lower than its matched-User control on all
three aggregate metrics. This supports rollout/evaluation User-distribution
matching, not a User-independent ranking between the V1-trained and
RAW-trained Agents.
