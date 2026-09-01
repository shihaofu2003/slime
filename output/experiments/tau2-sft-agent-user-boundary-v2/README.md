# tau2-sft-agent-user-boundary-v2

Purpose: measure the effect of a signed Agent/User ownership contract separately
from restored telecom coverage and explicit boundary anchors, while preserving a
matched 19,318-row, two-epoch raw-Instruct SFT budget.

## Causal control

The old SFT's lower visible namespace rate was partly single-call throttling:
dependency-safe multi increased affected trajectories from 58/160 to 82/160,
User-tool calls from 349 to 500, and `too_many_errors` from 23 to 35. The old
data contained 1,940 telecom rows but only incidental handoff supervision; the
current local-first data contains 353 telecom rows and 38 handoff events from
five dialogs. Both lack an explicit ownership boundary.

## Contract and data

- Profile: `agent-owned-dependency-safe-multi`; native Agent-only `tools=`,
  structured Assistant calls, `role=tool` results, full policies/manual, and
  signed policy/schema/chat-template metadata.
- Contract-only: 8,873 airline, 10,092 retail, 353 telecom.
- Contract + boundary: 8,783 airline, 8,595 retail, 970 ordinary telecom, 580
  natural handoff anchors, and 390 ownership-repair anchors.
- Loss mask: `qwen3_full`; maximum complete sequence 16,384; no truncation.
- Data and contract hashes are written by `prepare_manifest.json` and
  `build_manifest.json`. Preparation produced 715 ordinary review targets and a
  762-row/291-dialog handoff pool covering all 25 observed User tools; 472 are
  directly from the historical inventory and 290 replace inventory rows whose
  full prefix lacks a required natural report. The longest final rows are
  11,517 tokens for contract-only and 16,259 for contract + boundary.
- Built artifact hashes: contract-only
  `fe48909c2923b64a4a0ce0bec6ff7bc3373186ad2a532be2294812018d3c1695`;
  contract + boundary
  `995c46b437a784176e8c62683d5220a180967873b4329195c4c4fd0669857a39`.
  The airline/retail/telecom contract signatures are respectively
  `8beac16c…7778`, `a492ea76…c868`, and `a5dfebbc…3144`.

## Tasks and logs

- `pt-4ty8ji01`, `fsh-boundary-v2-prepare-0804-023346`: generate and hard-check
  ordinary-review and natural-handoff candidate pools; failed because the
  initial extractor used consensus-success pre-operation turns and found
  107 rather than the sealed 762 post-report targets —
  [run log](jobs/fsh-boundary-v2-prepare-0804-023346/run_20260804_023346.log).
- `pt-4tll5cim`, `fsh-boundary-v2-prepare-r2-0804-025559`: rerun candidate
  preparation with the historical 1,940-row/497-dialog inventory and strict
  post-report Agent targets; succeeded —
  [run log](jobs/fsh-boundary-v2-prepare-r2-0804-025559/run_20260804_025559.log).
- `pt-o5at8zim`, `fsh-boundary-v2-review-qwen36-0804-030301`: local
  Qwen3.6-27B review/repair of all 715 ordinary telecom targets; 516 unique
  targets across 179 dialogs passed, including 40 repaired-and-re-reviewed —
  [run log](jobs/fsh-boundary-v2-review-qwen36-0804-030301/run_20260804_030301.log).
- `pt-m6i9364w`, `fsh-boundary-v2-build-0804-032940`: build and validate both
  19,318-row arms plus longest-32 smoke sets; failed because the first
  replacement selector only admitted the 287 prior `keep=true` telecom targets,
  all already present in contract-only —
  [run log](jobs/fsh-boundary-v2-build-0804-032940/run_20260804_032940.log).
- `pt-56td3lha`, `fsh-boundary-v2-build-r2-0804-033415`: rebuild using only the
  516 cryptographically validated boundary-v2 review rows as the strict
  same-domain replacement pool; succeeded with exact domain recipes, 49
  schedule rows replacing 37 rejected targets, 580 unique natural anchors
  across 260 dialogs/all 25 observed User tools, and 30 × 13 ownership anchors —
  [run log](jobs/fsh-boundary-v2-build-r2-0804-033415/run_20260804_033415.log).
- `pt-jhao8cj9`, `fsh-boundary-v2-eval-raw-s300-0804-032107`: Raw Instruct,
  signed-contract official eval, seed 300; completed all 80 airline simulations
  and then failed in post-processing because the namespace analyzer iterated the
  live tau2 `Results` container instead of `Results.simulations` —
  [run log](jobs/fsh-boundary-v2-eval-raw-s300-0804-032107/run_20260804_032107.log).
- `pt-b8712fhh`, `fsh-boundary-v2-eval-raw-s301-0804-032107`: Raw Instruct,
  signed-contract official eval, seed 301; failed before evaluation because the
  shared runner was patched while this slower-starting shell was reading it —
  [run log](jobs/fsh-boundary-v2-eval-raw-s301-0804-032107/run_20260804_032107.log).
- `pt-v0v14wfw`, `fsh-boundary-v2-eval-raw-s301-r2-0804-032715`: clean retry of
  Raw Instruct signed-contract official eval nominally for seed 301; canceled
  after 32/80 airline simulations because it had imported the defective
  live-`Results` analyzer. The wrapper also failed to export `SEED`, so the
  actual config seed was 300; partial results and the log are retained —
  [run log](jobs/fsh-boundary-v2-eval-raw-s301-r2-0804-032715/run_20260804_032715.log).
- `pt-1nkyd6b9`, `fsh-boundary-v2-eval-old-sft-s300-0804-032108`: old SFT,
  signed-contract official eval, seed 300; failed after airline on the same live
  `Results` analyzer adapter defect as raw seed 300 —
  [run log](jobs/fsh-boundary-v2-eval-old-sft-s300-0804-032108/run_20260804_032108.log).
- `pt-2tyeeozz`, `fsh-boundary-v2-eval-old-sft-s301-0804-032108`: old SFT,
  nominal seed-301 signed-contract eval; failed after airline on the same live
  `Results` analyzer defect and actually used seed 300 because `SEED` was not
  exported —
  [run log](jobs/fsh-boundary-v2-eval-old-sft-s301-0804-032108/run_20260804_032108.log).
- `pt-k4y49tzn`, `fsh-boundary-v2-eval-current-sft-s300-0804-032109`: current
  new SFT, signed-contract official eval, seed 300; failed after airline on the
  same live `Results` analyzer adapter defect —
  [run log](jobs/fsh-boundary-v2-eval-current-sft-s300-0804-032109/run_20260804_032109.log).
- `pt-js7gp18m`, `fsh-boundary-v2-eval-current-sft-s301-0804-032109`: current
  new SFT nominal seed-301 eval; failed after airline on the same analyzer
  defect and actually used seed 300 because `SEED` was not exported —
  [run log](jobs/fsh-boundary-v2-eval-current-sft-s301-0804-032109/run_20260804_032109.log).
- `pt-orxd4c12`, `fsh-boundary-v2-eval-raw-s300-r2-0804-034711`: clean
  analyzer-fixed retry of Raw Instruct, seed 300; succeeded, with telecom
  namespace counts 74 affected trajectories / 289 attempts / 6 attributed
  terminations —
  [run log](jobs/fsh-boundary-v2-eval-raw-s300-r2-0804-034711/run_20260804_034711.log).
- `pt-lkb5l00g`, `fsh-boundary-v2-eval-old-sft-s300-r2-0804-034711`: clean
  analyzer-fixed retry of old SFT, seed 300; succeeded, with telecom namespace
  counts 132 affected trajectories / 1,914 raw attempts / 72 parsed and
  executed calls / 4 attributed terminations. The compute job was deleted only
  after its validated summary was written because SGLang cleanup hung —
  [run log](jobs/fsh-boundary-v2-eval-old-sft-s300-r2-0804-034711/run_20260804_034711.log).
- `pt-ze8lfd5f`, `fsh-boundary-v2-eval-old-sft-s301-r2-0804-034711`: clean
  analyzer-fixed nominal seed-301 retry of old SFT; canceled after discovering
  that the unexported `SEED` still made the actual config seed 300 —
  [run log](jobs/fsh-boundary-v2-eval-old-sft-s301-r2-0804-034711/run_20260804_034711.log).
- `pt-wi9jb9mj`, `fsh-boundary-v2-eval-current-sft-s300-r2-0804-034711`:
  clean analyzer-fixed retry of current new SFT, seed 300; succeeded, with
  telecom namespace counts 108 / 662 / 4 —
  [run log](jobs/fsh-boundary-v2-eval-current-sft-s300-r2-0804-034711/run_20260804_034711.log).
- `pt-y8rz0u4k`, `fsh-boundary-v2-eval-current-sft-s301-r2-0804-034711`:
  analyzer-fixed nominal seed-301 retry of current new SFT; canceled because
  the unexported `SEED` made the actual config seed 300 —
  [run log](jobs/fsh-boundary-v2-eval-current-sft-s301-r2-0804-034711/run_20260804_034711.log).
- `pt-700z0t9k`, `fsh-boundary-v2-eval-raw-s301-r3-0804-034743`: clean
  analyzer-fixed nominal seed-301 retry of Raw Instruct; canceled because the
  unexported `SEED` made the actual config seed 300 —
  [run log](jobs/fsh-boundary-v2-eval-raw-s301-r3-0804-034743/run_20260804_034743.log).
- `pt-ip3ci5pp`, `fsh-boundary-v2-eval-raw-s301-r4-0804-035903`: Raw Instruct
  retry after exporting the true config seed 301; succeeded, with telecom
  namespace counts 73 / 279 / 6 —
  [run log](jobs/fsh-boundary-v2-eval-raw-s301-r4-0804-035903/run_20260804_035903.log).
- `pt-svrf810t`, `fsh-boundary-v2-eval-old-sft-s301-r3-0804-035903`: old SFT
  retry after exporting the true config seed 301; succeeded, with telecom
  namespace counts 118 affected trajectories / 1,577 raw attempts / 45 parsed
  and executed calls / 0 attributed terminations —
  [run log](jobs/fsh-boundary-v2-eval-old-sft-s301-r3-0804-035903/run_20260804_035903.log).
- `pt-w3xz3337`, `fsh-boundary-v2-eval-current-sft-s301-r3-0804-035903`:
  current new SFT retry after exporting the true config seed 301; succeeded,
  with telecom namespace counts 98 / 631 / 3 —
  [run log](jobs/fsh-boundary-v2-eval-current-sft-s301-r3-0804-035903/run_20260804_035903.log).
- `pt-2eoetsdx`, `fsh-boundary-v2-smoke-contract-only-0804-035217`: 8-GPU,
  context-parallel 1 smoke on the 32 longest contract-only rows; canceled before
  start after the copied wrapper's relative sibling path was proven invalid —
  [run log](jobs/fsh-boundary-v2-smoke-contract-only-0804-035217/run_20260804_035217.log).
- `pt-58dg6gmu`, `fsh-boundary-v2-smoke-contract-boundary-0804-035217`: 8-GPU,
  context-parallel 1 smoke on the 32 longest contract + boundary rows; failed
  before training because the copied wrapper looked for an uncopied sibling
  script in its job-log directory —
  [run log](jobs/fsh-boundary-v2-smoke-contract-boundary-0804-035217/run_20260804_035217.log).
- `pt-hsv7kydp`, `fsh-boundary-v2-smoke-contract-only-r2-0804-040044`: CP=1
  contract-only longest-32 retry using the absolute repo launcher; succeeded at
  iteration 1 without OOM —
  [run log](jobs/fsh-boundary-v2-smoke-contract-only-r2-0804-040044/run_20260804_040044.log).
- `pt-ns1g07xk`, `fsh-boundary-v2-smoke-contract-boundary-r2-0804-040044`: CP=1
  contract + boundary longest-32 retry using the absolute repo launcher;
  succeeded at iteration 1 without OOM, so CP=2 was not needed —
  [run log](jobs/fsh-boundary-v2-smoke-contract-boundary-r2-0804-040044/run_20260804_040044.log).
- `pt-xsokb0jp`, `fsh-boundary-v2-train-contract-only-0804-040748`: full
  contract-only arm, 19,318 rows × 2 epochs, batch 16, CP=1; succeeded at
  iter2413 with final loss 0.1901 and grad norm 2.3091 —
  [run log](jobs/fsh-boundary-v2-train-contract-only-0804-040748/run_20260804_040748.log).
- `pt-xr4pcmvy`, `fsh-boundary-v2-train-contract-boundary-0804-040748`: full
  contract + boundary arm with the matched 19,318-row/two-epoch budget, CP=1;
  succeeded at iter2413 with final loss 0.2432 and grad norm 2.4285 —
  [run log](jobs/fsh-boundary-v2-train-contract-boundary-0804-040748/run_20260804_040748.log).
- `pt-8absl9sh`, `fsh-boundary-v2-convert-contract-only-0804-054548`:
  converted contract-only iter2413 to a complete 7.6-GiB `final_hf` directory —
  [run log](jobs/fsh-boundary-v2-convert-contract-only-0804-054548/run_20260804_054548.log).
- `pt-ikilol50`, `fsh-boundary-v2-convert-contract-boundary-0804-054548`:
  converted contract + boundary iter2413 to a complete 7.6-GiB `final_hf`
  directory —
  [run log](jobs/fsh-boundary-v2-convert-contract-boundary-0804-054548/run_20260804_054548.log).
- `pt-1em61084`, `fsh-boundary-v2-eval-contract-only-s300-0804-054829`:
  contract-only signed official eval, seed 300, including the fixed namespace
  probes; succeeded, with telecom reward 18.125% and namespace counts 81
  affected trajectories / 639 raw attempts / 631 parsed and executed calls /
  52 attributed terminations —
  [run log](jobs/fsh-boundary-v2-eval-contract-only-s300-0804-054829/run_20260804_054829.log).
- `pt-qpi3j2b5`, `fsh-boundary-v2-eval-contract-only-s301-0804-054829`:
  contract-only signed official eval, seed 301; succeeded, with telecom reward
  21.875% and namespace counts 78 affected trajectories / 642 raw attempts /
  633 parsed and executed calls / 51 attributed terminations —
  [run log](jobs/fsh-boundary-v2-eval-contract-only-s301-0804-054829/run_20260804_054829.log).
- `pt-gazfj179`, `fsh-boundary-v2-eval-contract-boundary-s300-0804-054829`:
  contract + boundary signed official eval, seed 300, including the fixed
  namespace probes; succeeded, with telecom reward 18.75% and namespace counts
  54 affected trajectories / 326 raw attempts / 107 parsed and executed calls /
  4 attributed terminations —
  [run log](jobs/fsh-boundary-v2-eval-contract-boundary-s300-0804-054829/run_20260804_054829.log).
- `pt-zmh9ylbw`, `fsh-boundary-v2-eval-contract-boundary-s301-0804-054829`:
  contract + boundary signed official eval, seed 301; succeeded, with telecom
  reward 16.875% and namespace counts 55 affected trajectories / 314 raw
  attempts / 121 parsed and executed calls / 9 attributed terminations —
  [run log](jobs/fsh-boundary-v2-eval-contract-boundary-s301-0804-054829/run_20260804_054829.log).

## Promotion

Final status: **pass under `turn-aware-rl-v3`; Contract + boundary selected**.
The [gate artifact](SFT_PROMOTION.json) records the new thresholds and the
failed prior strict gate. v3 keeps probes at `0/60` and at most `2/240`, while
setting per-seed official namespace limits to 60 affected trajectories, 350
attempts, and 10 attributed terminations; per-domain pass@1 may trail the best
fixed baseline by at most 8pp, and the paired pass^4 CI lower bound remains
`-2pp`.

Contract-only remains ineligible: its probes were 11/60 and 45/240, and its two
official seeds totaled 159 affected trajectories / 1,281 attempts / 103
attributed terminations. Contract + boundary passed probes at 0/60 and 0/240;
seed 300 was 54 / 326 / 4 and seed 301 was 55 / 314 / 9, with pass^4 paired CIs
`[-1,+12]` and `[-1,+11]` pp. The old 8 / 16 / 0 and 5pp limits rejected both
completed arms; v3 admits only the measured Contract + boundary initialization
so turn-aware RL can reduce its remaining namespace errors under stricter
stage-by-stage non-regression gates.

## Downstream RL validation

The selected Contract + boundary checkpoint was subsequently used by the
[turn-aware Pilot A](../tau2-rl-agent-user-boundary-v2/README.md) and, from a
fresh optimizer lineage, the
[100-update follow-up](../tau2-rl-agent-user-boundary-v2-long100/README.md).
The historical Pilot A stopped at iter9 on behavior/truncation thresholds even
though its training state was healthy; the follow-up corrected continuation to
health-only gates without changing this SFT promotion artifact.

The long100 iter99 final decision was **`win`**. Under the same signed profile,
100 tasks × 4 trials, and seeds 300/301, its two-seed mean
pass@1/pass@4(any)/pass^4 was `29.50/56.50/10.50%`, versus
`23.63/43.50/8.50%` for this selected SFT and `25.25/50.00/9.00%` for the
historical iter9 control. This validates Contract + boundary as the successful
initialization for the completed longer-training experiment; it does not
retroactively make Contract-only eligible.
