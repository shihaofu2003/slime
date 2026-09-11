# Banking independent synthetic v1

## Goal and isolation

Generate independent Banking scenarios under `banking_syn_v1_*`. Generation,
naturalization, autonomous evaluation, and teacher code cannot read the sealed
97 benchmark tasks. Only the post-generation isolation audit may compare against
them. Runtime data contains an empty Banking DB, per-task synthetic entities, and
the 480 documents never required by the benchmark; the other 218 are absent.
The existing 3,304 benchmark-derived diagnostic tasks and 2,492 successful
trajectories remain excluded from SFT, GRPO, and OPD.

## Data contract

Each `ScenarioSpec` records category, template, L1-L6 level, atomic capabilities,
initialization data, allowed documents, decisive facts, User knowledge and tools,
an action dependency graph, expected state changes, response facts, and split.
Its `TaskContract` has `data_origin: independent_synthetic` and no source task ID.
One root scenario creates at most four tasks and cannot cross splits. Synthetic
discoverable-action labels and case parameters are explicit User-known facts
because the 480-document allowlist does not contain those randomized tool
signatures; policy decisions and action ordering remain separate Agent work.
Applicable L3/L4 writes visibly start after a completed verification, and
transfer success compares the transfer action rather than non-semantic reason
wording, so equivalent correct action paths remain valid. The naturalized
request is the initial User message seen by both participants; private identity
details and User-tool timing remain only in simulator instructions. L4--L6
refusals are judged by the customer-owned review request's final DB state rather
than an exact sequence of policy reads.

## Curriculum

The pilot contains 1,500 tasks and exactly 100 per business category. The final
target is 2,000 train, 500 dev, and 500 challenge tasks. Train form quotas are
200/100/300/400/600/300/100 for L1 evidence, L2 retrieval, L3 decision, L3
single action, L4 two-skill, L5 workflow, and L6 workflow. The original 10,000
train candidates remain available only as a selection source and diagnostic
record; rejected quota cells are filled only with fresh scenario IDs.

## Validation and models

Every task must pass static/schema checks, foreign keys and User-tool boundaries,
real reference replay at reward 1, BM25 target top-10 when applicable, and the
isolation audit. Agent and User are both `/mnt/afs/models/Qwen3.8-27B`; Agent uses
xhigh thinking, temperature 1.0, top-p 0.95, top-k 20, 16,384 output tokens, and
262,144 context. User is non-thinking, temperature 0, and uses 1,024 output
tokens. `reasoning_effort` appears only in `chat_template_kwargs`.

## Evaluation and recovery

Pilot trials use seeds 300-303. Every retained task receives one valid autonomous
trial. The 1,400 action-bearing train tasks plus all 500 dev and 500 challenge
tasks receive four valid trials. The dialogue cap is 60 steps, leaving ample
room for the 14-action L6 ceiling. Missing final messages, length exhaustion,
service failures, and abnormal termination are protocol failures, not business
failures. Only affected tasks are recovered at 32,768 output tokens while
first-attempt statistics remain unchanged.

## Pools and acceptance

Outputs include successful trajectories and overlapping `sft_candidate`,
`grpo_frontier`, `behavior_anchor`, `hard_sft_only`, and `synthetic_challenge`
pools. Acceptance requires exact split counts, no root-scenario overlap, at least
600 train frontier tasks, a successful Qwen3.8 trajectory for every train task,
at least one valid autonomous trajectory for every task, four valid trials for
all stratified tasks, 100% schema/reference replay, zero benchmark leakage, and
no unresolved protocol failure.

## Deferred work

The sealed benchmark is evaluated once only after data, recipe, and model are
frozen. SFT, GRPO, OPD, Banking environment changes, and `prepare_rl_data.py`
integration are outside this experiment.
