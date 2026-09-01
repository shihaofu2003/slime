# tau2-rl-agent-user-boundary-v2

Purpose: train the `turn-credit-v1` three-domain GRPO policy from the v3-selected
Contract + boundary SFT checkpoint, reducing Agent/User namespace errors without
erasing turn-local negative signal during group normalization.

## Credit and configuration

For each K-sample prompt group,
`global_advantage=(global_score-group_mean)/(group_std+eps)` and every token in
Assistant turn `t` receives
`turn_advantage=global_advantage-turn_penalty[t]`; `returns=advantages`.
Task success and tool-name/argument/DB/environment/communication partial credit
remain trajectory-level positive reward. Local penalties are absolute and are
not group-centered: User namespace `min(2,1+0.1*(n-1))`; nonexistent and
malformed calls `0.15` each, cap `0.30`; wrong argument fields `0.10`, cap
`0.40`; Agent execution errors `0.25`, cap `0.50`; repeated identical calls
from the second occurrence `0.05`, cap `0.20`; final trainable turn on
`max_steps` `0.20`. `too_many_errors` adds no duplicate penalty.

The rollout stores one-render token IDs, loss mask, Assistant spans, global
score, per-turn errors/spans, and a response-length `token_penalties` vector.
DP carries that list per sample; CP uses the log-prob slice helper. Misaligned,
orphaned, mask-external, or length-mismatched credit rejects the sample. A
global-zero-variance group remains trainable when any local penalty is nonzero.
K=8, six prompts/update, global batch 48, LR `2e-6`, k2 KL coefficient `0.01`,
entropy 0, and full-conversation cap 16,384 are fixed.

## Stages and gates

- Isolated updates 0-1 smoke: quota telecom/airline/retail `3/2/1`; it cannot
  seed formal training and must prove finite training, quota, span alignment,
  and lower penalized-turn advantage.
- Formal Pilot A restarts from the selected SFT: updates 0-9, quota `3/2/1`.
  Iter9 is converted and evaluated on seeds 300/301, 100 tasks × 4 trials, plus
  namespace probes; Pilot B cannot start without `ITER9_SAFETY_GATE.json`.
- Pilot B: updates 10-19, quota `3/1/2`; the combined namespace limits are
  affected `<=87`, attempts `<=512`, and attributed terminations `<=13`.
- Final: updates 20-99, quota `2/2/2`; it cannot start without the iter19 gate.
  Iter99 namespace metrics must not regress from either SFT or iter19. Per seed,
  pass@1 and pass@4(any) must strictly exceed both; pass^4 may fall at most 2pp
  from SFT and each domain pass@1 at most 5pp.

All stages run as 8-GPU normal-priority jobs. Every launcher runs
`test_rollout_logic.py` before Ray/SGLang. A failed gate stops the sequence; no
RL threshold is relaxed automatically.

## Tasks and logs

- `pt-ktfha3vo`, `fsh-boundary-turn-credit-smoke-0804-172413`: initial isolated
  smoke submission; deleted while still `STARTING` with no replica or start
  time so the normalization update could be captured by a clean resubmission —
  [submit log](jobs/fsh-boundary-turn-credit-smoke-0804-172413/submit_20260804_172413.log).
- `pt-b3pxepff`, `fsh-boundary-turn-credit-smoke-r2-0804-173007`: replacement
  smoke failed before Ray/SGLang startup because the mandatory preflight's DP
  fixture treated Ray's remote `RolloutManager` wrapper as a normal class; no
  update or checkpoint was produced —
  [run log](jobs/fsh-boundary-turn-credit-smoke-r2-0804-173007/run_20260804_173007.log).
- `fsh-boundary-turn-credit-smoke-r3-0804-175757`: corrected DP fixture, but
  SCO DNS lookup timed out before a remote job was created —
  [submit log](jobs/fsh-boundary-turn-credit-smoke-r3-0804-175757/submit_20260804_175757.log).
- `pt-jqkzjwmw`, `fsh-boundary-turn-credit-smoke-r3-retry-0804-175909`:
  effective r3 2-update, 8-GPU smoke using a new isolated checkpoint root and
  unchanged `3/2/1` quota. The checkpoint reached iter1 and the smoke gate
  passed: both update quotas were exact, all 96 trajectories were span/token
  aligned, no NaN/Inf/OOM occurred, and the observed penalized-minus-clean
  turn advantage was `-0.05` —
  [run log](jobs/fsh-boundary-turn-credit-smoke-r3-retry-0804-175909/run_20260804_175909.log).
- `pt-20ual5tm`, `fsh-boundary-turn-credit-pilot-a-0804-181532`: formal
  updates 0-9 Pilot A, restarted from the selected SFT with quota `3/2/1`.
  Ray succeeded and saved iter9; all train metrics were finite, the final KL
  loss was `0.000623`, and 519 trajectories include 480 accepted samples plus
  rejected/replacement diagnostics —
  [run log](jobs/fsh-boundary-turn-credit-pilot-a-0804-181532/run_20260804_181532.log).
- `pt-i1evbhfk`, `fsh-boundary-turn-credit-iter9-convert-0804-185302`:
  iter9 torch-dist to HF conversion, 8-GPU normal priority; succeeded —
  [run log](jobs/fsh-boundary-turn-credit-iter9-convert-0804-185302/run_20260804_185302.log).
- `pt-513fatkk`, `fsh-boundary-turn-credit-iter9-eval-seed300-0804-185649`:
  iter9 seed 300 official 100-task × 4-trial eval plus namespace probes;
  succeeded —
  [run log](jobs/fsh-boundary-turn-credit-iter9-eval-seed300-0804-185649/run_20260804_185649.log).
- `pt-pcdo2nk2`, `fsh-boundary-turn-credit-iter9-eval-seed301-0804-185649`:
  iter9 seed 301 official 100-task × 4-trial eval; deleted before execution at
  user request to retain one seed, then replaced after the user restored the
  full two-seed gate —
  [submit log](jobs/fsh-boundary-turn-credit-iter9-eval-seed301-0804-185649/submit_20260804_185649.log).
- `pt-zr0h0vgp`, `fsh-boundary-turn-credit-iter9-eval-seed301-r2-0804-185924`:
  replacement iter9 seed 301 official 100-task × 4-trial eval; succeeded —
  [run log](jobs/fsh-boundary-turn-credit-iter9-eval-seed301-r2-0804-185924/run_20260804_185924.log).

## Artifacts

- SFT source gate: [turn-aware-rl-v3 promotion](../tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json).
- Smoke checkpoint root:
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_turn_credit_smoke_r3_20260804`.
- Smoke decision: [SMOKE_GATE.json](SMOKE_GATE.json), status `pass`.
- Formal checkpoint root:
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804`.
- Iter9 decision: [ITER9_SAFETY_GATE.json](ITER9_SAFETY_GATE.json), status
  `fail` (`25/29` checks passed). Namespace improved on both seeds: affected
  trajectories `54→36` / `55→43`, attempts `326→151` / `314→191`, and
  attributed terminations `4→3` / `9→2`. Overall pass@1 changed
  `23.75→26.25%` / `23.50→24.25%`; pass@4(any) `42→52%` / `45→48%`; pass^4
  `8→10%` / `9→8%`. Promotion stopped because training truncation was
  `30.63%` (`<1%` required), malformed trajectories `18.13%` (`<10%`), tool
  execution-error trajectories `15.63%` (`<10%`), and max_steps `29.58%`
  (`<2%`). Pilot B was not submitted and no RL gate was relaxed.

## Follow-up and corrected interpretation

The iter9 artifact remains `fail` and is not rewritten. The later diagnosis
separated training health from behavior/effectiveness: Pilot A had a valid
iter9 checkpoint, finite metrics, low KL, exact quota, and aligned Contract,
response spans, loss masks, and turn-credit vectors. The four failed checks
above measured behavior or termination frequency; framework `truncated` also
combined normal `max_steps` horizon endings with timeout, context-window, and
over-token-cap failures. They did not establish a training-health failure and
were too early to decide whether the stable `2e-6` recipe was under-trained.

The follow-up [long100 experiment](../tau2-rl-agent-user-boundary-v2-long100/README.md)
therefore started again from the selected SFT with a fresh optimizer and
checkpoint root, retained all model/reward/horizon settings, and used
health-only iter9/iter19 continuation gates. It completed updates 0-99 and the
final decision was **`win`**:

| Same-protocol two-seed mean | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| selected SFT | 23.63% | 43.50% | 8.50% |
| this historical iter9 | 25.25% | 50.00% | 9.00% |
| long100 iter99 | **29.50%** | **56.50%** | **10.50%** |
| iter99 - iter9 | **+4.25pp** | **+6.50pp** | **+1.50pp** |

This resolves the historical ambiguity: the iter9 policy was a useful control,
but stopping on the diagnostic rates prevented the intended training-length
test. Behavior errors remain fully reported at iter99; they no longer masquerade
as OOM, non-finite training, checkpoint corruption, or protocol misalignment.
