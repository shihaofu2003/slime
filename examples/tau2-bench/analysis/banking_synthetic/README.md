# Banking independent synthetic v1

This package builds Banking `ScenarioSpec`, Tau2 `Task`, and `TaskContract`
artifacts without using benchmark instances as generation inputs.

## Isolation boundary

Only `isolation.py` reads the sealed 97-task directory. It creates the exact
480-document allowlist and audits already-generated artifacts. The generator,
naturalizer, evaluator staging, teacher, pool, and assembly modules consume only
synthetic artifacts and the allowlisted document corpus.

## Pilot build

Submit `run_build_pilot.sh` on a CPU node. It builds 1,500 tasks, runs the local
test suite, replays every gold action in the real Banking environment, checks
BM25 top-10 retrieval, and runs the post-generation isolation audit. It also
replays a separate 75-task template matrix containing every business category
crossed with positive, refusal, User-tool, dynamic-tool, and multi-entity cases.

## Qwen3.8 smoke

Run `run_qwen38_naturalize.sh smoke`, then submit `run_qwen38_eval.sh` once for
`bm25` and once for `golden_retrieval`. The selected 30 tasks cover all 15
categories, all seven task forms, all five case kinds, and a positive write path
at every action-bearing difficulty. Both Agent and User are hard-bound to
`/mnt/afs/models/Qwen3.8-27B`. Because the permitted 480-document corpus does
not expose randomized discoverable-tool suffixes, each applicable scenario
provides its synthetic action label and case parameters as a non-private User
clue, including the exact Agent wrapper invocation for discoverable actions.
User-owned calls also carry a private, dependency-aware execution protocol.
Applicable L3/L4 write cases start from a visible, already-logged identity
verification so an equivalent Agent path that rechecks identity cannot change
the evaluator's target state. The Qwen-naturalized customer request is installed
as the task's initial User message, so the Agent always receives every public
case-note detail; identity fields and User-tool timing remain private simulator
instructions. L4--L6 refusal workflows use the customer-owned human-review
request's DB change, so alternative policy-search paths remain valid. The
naturalizer masks immutable case literals during prose rewriting and restores
them before validation; after four rejected rewrites it retains the original
programmatic wording and reports the fallback. Decision tasks expose a protected
Yes/No response format that matches their stable communication fact, and
profile-change tasks define the requested new email in both the visible outcome
and the User's known information. Evidence and decision tasks choose the first
complete fact that directly matches their business category; L1 requests
identify it with a short opening phrase so another fact in the same document is
not an equivalent answer. Short FAQ answers retain their question context
instead of becoming bare numeric fragments.

## Pilot and scale evaluation

After smoke acceptance, naturalize all pilot tasks and run seeds 300-303. Scale
data is built with `run_build_batch.sh` in roughly 2,000-task batches. Each new
task receives one autonomous trial with a 60-step dialogue cap; frontier
candidates plus dev and challenge
receive four. After protocol recovery, `select_followup.py` balances successful
and unsuccessful first trials within each form/category while prioritizing the
1,400 action-bearing L3--L6 train tasks; it always adds dev and challenge. Its
output receives seeds 301-303. `select_recovery.py`
emits only missing/protocol-failed IDs for a 32,768-token recovery run. Use
`smoke_recovery` for smoke inputs and `recovery` for pilot inputs in
`run_qwen38_eval.sh`; scale recovery supplies its artifact paths through the
documented `SYNTHETIC_TASKS` and `SYNTHETIC_CONTRACTS` variables. On subsequent
recovery rounds, pass prior retry outputs through `--recovery-results`; only
the original `--results` files determine the required trial count.
Evaluation summarization compacts each simulation while reading its result file,
so full message payloads are not retained together in memory.
The 1,500-task pilot is split into four BM25 shards (207/206/206/206 tasks) and
three golden-retrieval shards (225 tasks each) for every seed.

## Teacher and pools

`teacher.py` gives Qwen3.8 one synthetic action-graph node, its allowed action,
and allowlisted documents at a time, with at most four attempts. Hidden coaching
is omitted from saved trajectories and reward 1 is required.
`select_teacher.py` limits this path to tasks with valid autonomous trials but
no success and can restrict selection to the train split. Teacher jobs accept
deterministic shard index/count arguments so
large zero-success sets can run concurrently. `pools.py` produces the overlapping SFT, GRPO-frontier,
behavior-anchor, hard-SFT, and challenge manifests. Only train-split
trajectories can enter the training output; dev and challenge results remain
evaluation-only.

## Assembly and acceptance

`assemble.py` fills the exact category/form quota cells and writes fresh
replacement requests for any gap. It prioritizes frontier tasks within each
quota cell and requests fresh candidates if fewer than 600 can be retained,
starting with replaceable L6/L5/L4/L3-action quota slots;
`resample.py` creates new scenario IDs for those requests. `merge.py` joins
disjoint shards without rewriting artifacts,
and `run_final_validation.sh` performs the combined 3,000-task replay,
isolation audit, and `finalize.py` acceptance gate. The final gate checks exact
counts and quotas, split isolation, model paths, autonomous-trial and teacher
coverage, document isolation, and unresolved failures.

## Non-goals

This package does not change the Banking environment, call
`prepare_rl_data.py`, submit SFT/GRPO/OPD, or run the sealed benchmark.
