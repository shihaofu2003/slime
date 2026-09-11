# tau2-agent-single-call-v1-user-ablation

Purpose: measure whether the strict-single SFT and selected RL iter199 Agents'
official tau2 results depend on User SFT by replacing only the v1 User with its
original `Qwen3-4B-Instruct-2507` base model.

Name: tau2-agent-single-call-v1-user-ablation.

## Design

For each Agent checkpoint, the strict-single-v1 protocol, test tasks, four
trials, seeds 300/301, Agent temperature 0.6, and all other evaluation settings
match `tau2-agent-single-call-v1`. The only changed variable is the User
checkpoint:

- control: `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`
- ablation: unmodified `models/Qwen3-4B-Instruct-2507`

The Agent checkpoints are the single-call SFT `final_hf` and RL iter199. These
A/B evaluations test the sensitivity of each single-call Agent result to User
SFT. The RL iter199 Agent itself was trained with v1 User SFT as its rollout
User; both raw-User jobs change only the evaluation User and do not retrain the
Agent. They therefore do not by themselves isolate the training-time User
effect.

## Tasks

- `pt-bjbsdscd` / RL iter199 seed 300 raw-User evaluation —
  [run log](jobs/fsh-tau2-single-v1-rl199-raw-user-300-0810-222359/run_20260810_222359.log).
- `pt-53d5281a` / RL iter199 seed 301 raw-User evaluation —
  [run log](jobs/fsh-tau2-single-v1-rl199-raw-user-301-0810-222359/run_20260810_222359.log).
- `pt-92eevhlr` / SFT seed 300 raw-User evaluation —
  [run log](jobs/fsh-tau2-single-v1-sft-raw-user-s300-0812-113758/run_20260812_113758.log).
- `pt-6ij802w8` / SFT seed 301 raw-User evaluation —
  [run log](jobs/fsh-tau2-single-v1-sft-raw-user-s301-0812-113804/run_20260812_113804.log).

## Results

All four raw-User jobs succeeded with 400/400 simulations, zero infrastructure
errors, and zero Agent strict-single protocol errors.

### SFT Agent

| User | seed | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| v1 User SFT | 300 | 27.25% | 50.00% | 10.00% |
| v1 User SFT | 301 | 24.50% | 49.00% | 7.00% |
| v1 User SFT mean | 300/301 | 25.88% | 49.50% | 8.50% |
| raw Instruct User | 300 | 29.00% | 53.00% | 9.00% |
| raw Instruct User | 301 | 27.00% | 49.00% | 9.00% |
| raw Instruct User mean | 300/301 | **28.00%** | **51.00%** | **9.00%** |

Each cell below is pass@1 / pass@4(any) / pass^4. Deltas are raw User minus
v1 User SFT.

| Scope | v1 User SFT mean | Raw User mean | Delta |
|---|---:|---:|---:|
| Overall | 25.88 / 49.50 / 8.50% | 28.00 / 51.00 / 9.00% | **+2.13 / +1.50 / +0.50pp** |
| Airline | 33.75 / 50.00 / 22.50% | 42.50 / 57.50 / 30.00% | +8.75 / +7.50 / +7.50pp |
| Retail | 27.50 / 53.75 / 7.50% | 35.62 / 61.25 / 7.50% | +8.12 / +7.50 / 0.00pp |
| Telecom | 20.31 / 45.00 / 2.50% | 13.12 / 37.50 / 0.00% | **-7.19 / -7.50 / -2.50pp** |

The overall raw-minus-v1 paired two-seed task-bootstrap 95% intervals are
`[-2.75,+7.00]pp`, `[-6.50,+10.00]pp`, and `[-4.00,+5.00]pp`; all include
zero. Telecom pass@1 is consistently lower at `-7.19pp`, interval
`[-13.12,-1.88]pp`. The positive Airline and Retail changes offset that
Telecom regression in the overall mean.

The User behavior also changes sharply. In Telecom, v1 User emitted multi-call
User-tool turns in 83/320 trajectories (180 turns, 390 calls), versus 1/320
for raw User (one turn, two calls). Across 800 simulations, v1/raw termination
counts were `user_stop=614/707`, `too_many_errors=121/20`, and
`max_steps=65/73`. Mean action/DB accuracy was `64.91/30.14%` with v1 User and
`63.53/30.69%` with raw User. The score change is therefore an interaction
shift, not an infrastructure failure or a uniform capability improvement.

### RL iter199 Agent

| User | seed | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| v1 User SFT | 300 | 30.75% | 58.00% | 11.00% |
| v1 User SFT | 301 | 31.50% | 65.00% | 10.00% |
| v1 User SFT mean | 300/301 | 31.13% | 61.50% | 10.50% |
| raw Instruct User | 300 | 28.00% | 52.00% | 8.00% |
| raw Instruct User | 301 | 25.75% | 53.00% | 8.00% |
| raw Instruct User mean | 300/301 | **26.88%** | **52.50%** | **8.00%** |

| Scope | v1 User SFT mean | Raw User mean | Raw - v1 |
|---|---:|---:|---:|
| Overall | 31.13 / 61.50 / 10.50% | 26.88 / 52.50 / 8.00% | **-4.25 / -9.00 / -2.50pp** |
| Airline | 30.00 / 50.00 / 12.50% | 41.88 / 62.50 / 25.00% | +11.88 / +12.50 / +12.50pp |
| Retail | 31.25 / 58.75 / 15.00% | 29.69 / 58.75 / 7.50% | -1.56 / 0.00 / -7.50pp |
| Telecom | 31.56 / 70.00 / 5.00% | 16.56 / 41.25 / 0.00% | **-15.00 / -28.75 / -5.00pp** |

The overall raw-minus-v1 paired intervals are `[-9.50,+1.25]pp`,
`[-19.00,+1.00]pp`, and `[-8.00,+3.00]pp`; all include zero because Airline
and Telecom move in opposite directions. Telecom intervals are
`[-21.25,-8.75]pp`, `[-43.75,-13.75]pp`, and `[-10.00,-1.25]pp`.

### Same-User RL comparison

Holding the evaluation User fixed changes the apparent RL conclusion. In both
rows, RL iter199 was trained with v1 User SFT:

| Evaluation User | SFT | RL iter199 | RL - SFT |
|---|---:|---:|---:|
| v1 User SFT | 25.88 / 49.50 / 8.50% | 31.13 / 61.50 / 10.50% | **+5.25 / +12.00 / +2.00pp** |
| Raw User | 28.00 / 51.00 / 9.00% | 26.88 / 52.50 / 8.00% | **-1.13 / +1.50 / -1.00pp** |

Under raw User, the paired two-seed RL-minus-SFT intervals are
`[-4.75,+2.50]pp`, `[-4.50,+7.50]pp`, and `[-5.00,+2.50]pp`; none excludes
zero. By domain, raw-User RL-minus-SFT is `-0.63/+5.00/-5.00pp` for Airline,
`-5.94/-2.50/0.00pp` for Retail, and `+3.44/+3.75/0.00pp` for Telecom. Retail
pass@1 is lower by `5.94pp`, interval `[-11.88,-0.31]pp`.

The v1 User is therefore not a neutral evaluator: replacing it can reverse the
sign of the measured RL pass@1 gain. The separate raw-User-trained RL iter99
evaluation has now completed under raw User at `29.25/54.00/6.50%`, versus the
same raw-User SFT control at `28.00/51.00/9.00%`. Its
`+1.25/+3.00/-2.50pp` delta has overall paired intervals
`[-2.88,+5.25]`, `[-4.00,+10.00]`, and `[-6.00,+0.50]pp`; none excludes zero.
The gain is concentrated in Telecom. The final two-seed crossed iter99 results
separate the two directions: V1-RL99 scores `27.50/53.50/10.00%` with V1
versus `25.25/50.50/5.50%` with RAW, while RAW-RL99 scores
`29.25/54.00/6.50%` with RAW versus `24.25/49.50/6.00%` with V1. Full evidence
is in the
[raw-User RL100 README](../tau2-agent-single-call-v1-raw-user-rl100/README.md).

The later RAW-RL iter199 result completes the same-RAW-evaluation training-User
comparison: RAW-trained `31.50/58.00/8.00%` versus V1-trained
`26.88/52.50/8.00%`. Jobs `1284`–`1288` and `1302`–`1306` subsequently
completed the five reverse/missing crossed cells at both seeds. The
consolidated table is in the
[single-call main README](../tau2-agent-single-call-v1/README.md#user-model-ablation).

### Complete two-axis matrix

The consolidated matrix below reports both experimental axes explicitly. Every
evaluation uses `max_steps=200` and the Qwen User tool-call parser; cells are
pass@1 / pass@4(any) / pass^4. Both User columns in a row use the same seed
coverage, and `★` marks the cell that is higher on all three metrics. Rows
follow raw Qwen3, single-call SFT, iter99, iter199, and external Qwen3.5.

| Agent checkpoint | Training User | Training max_steps | Comparison seeds | Evaluation User = V1, parser ON | Evaluation User = RAW, parser ON |
|---|---|---:|---:|---:|---:|
| [Raw Qwen3-4B-Instruct-2507](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 23.88 / 45.50 / 6.50% ★ | 17.38 / 35.50 / 2.50% |
| Single-call SFT final | N/A (offline SFT) | N/A | 300/301 mean | 25.88 / 49.50 / 8.50% | 28.00 / 51.00 / 9.00% ★ |
| V1-RL iter99 | V1 | 60 | 300/301 mean | 27.50 / 53.50 / 10.00% ★ | 25.25 / 50.50 / 5.50% |
| RAW-RL iter99 | RAW | 60 | 300/301 mean | 24.25 / 49.50 / 6.00% | 29.25 / 54.00 / 6.50% ★ |
| RAW-RL iter99 max120 | RAW | 120 | 300/301 mean | 25.25 / 51.00 / 6.00% | 28.38 / 56.50 / 9.50% ★ |
| V1-RL iter199 | V1 | 60 | 300/301 mean | 31.13 / 61.50 / 10.50% ★ | 26.88 / 52.50 / 8.00% |
| RAW-RL iter199 | RAW | 60 | 300/301 mean | 25.00 / 53.00 / 6.00% | 31.50 / 58.00 / 8.00% ★ |
| RAW-RL iter199 max120 | RAW | 120 | 300/301 mean | 25.25 / 49.00 / 5.50% | 30.13 / 58.50 / 7.00% ★ |
| [Raw Qwen3.5-4B thinking-on](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 48.50 / 76.00 / 22.00% | 53.25 / 78.00 / 25.00% ★ |
| [Raw Qwen3.5-4B non-thinking](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 32.75 / 61.00 / 9.00% ★ | 25.63 / 53.50 / 7.00% |

Thus the table distinguishes the training and evaluation User axes. All five
formerly blank RL cells now have final seed300/301 means, and every RL Agent
scores higher on all three metrics with the evaluation User family used for
its rollouts. The raw-User-trained iter199 results are documented in the
[RAW-RL200 README](../tau2-agent-single-call-v1-raw-user-rl200/README.md) and
[max120 continuation README](../tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md).
The raw-Agent rows are external references, not `strict-single-v1` controls;
both columns use parser ON and are complete at both seeds. Historical
parser-OFF RAW references remain documented separately and are not used as
cells in this matrix.
