# vitabench-qwen35-full-eval

Purpose: evaluate Qwen3.5-4B across all 400 Chinese VitaBench tasks with four
trials per task, the user-selected ChatAnywhere `gpt-4.1-ca` user simulator and
`gemini-2.5-flash` evaluator, strict result-completeness checks, and descriptive
comparison to the published four-domain average score of 22.0.

Name: `vitabench-qwen35-full-eval`.

## Protocol

- Four 100-task suites: delivery, in-store, OTA, and cross-domain.
- Four trials per task: 1,600 trajectories total; Chinese, temperature 0,
  seed 300, max 300 steps. Four suite workers use one GPU and concurrency 4
  each (16 aggregate), with remote GPT/Gemini requests capped at 8 aggregate.
- Qwen3.5-4B thinking is enabled. `gpt-4.1-ca` is only the user simulator;
  `gemini-2.5-flash` is only the trajectory evaluator.
- All Qwen/GPT/Gemini response-level requests and responses are atomically
  journaled and exported as provenance-linked, checksummed SFT JSONL files.
- User/evaluator: `gpt-4.1-ca` / `gemini-2.5-flash`; the 22.0 reference is not
  strictly comparable because the published protocol is incomplete and uses
  different service/model choices.

## Jobs

- `pt-upeo6j2i` — `fsh-vitabench-qwen35-full-gptca-gemini25-0801-003100`,
  reserved 1×N6lS-80GB, normal priority:
  [run log](jobs/fsh-vitabench-qwen35-full-gptca-gemini25-0801-003100/run_20260801_003100.log).
  Failed before manifest/result creation because Bash rejected a repeated
  assignment to a readonly exported endpoint variable.
- `pt-dr1sa3ea` — `fsh-vitabench-qwen35-full-gptca-gemini25-r1-0801-004004`,
  reserved 1×N6lS-80GB, normal priority, readonly-variable fix:
  [run log](jobs/fsh-vitabench-qwen35-full-gptca-gemini25-r1-0801-004004/run_20260801_004005.log).
  Failed before gate/result creation because the protocol inputs changed during
  startup; its manifest, run plan, and server logs are retained in
  `artifacts_failed_r1_protocol_drift_20260801_004504/`.
- `pt-6baygcys` — `fsh-vitabench-qwen35-full-gptca-gemini25-r2-0801-004635`,
  reserved 1×N6lS-80GB, normal priority, clean final protocol snapshot:
  [run log](jobs/fsh-vitabench-qwen35-full-gptca-gemini25-r2-0801-004635/run_20260801_004635.log).
  Passed all 29 then-current preflight checks and the 16-trajectory four-suite
  gate, then was
  deliberately stopped before any formal shard completed so the evaluation
  could switch to four GPUs. Its manifest, gate results, empty formal shard,
  and server log remain in
  `artifacts_single_gpu_r2_stopped_for_4gpu_20260801_011118/`. Its private
  `sft/` archive contains 115 reconstructed GPT user-simulator examples and 71
  exact Gemini evaluator examples; the pre-journal Qwen calls remain only in
  the raw gate result/server artifacts and are not mislabeled as exact SFT
  requests.
- `pt-uvrbej2r` — `fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-0801-015625`,
  submitted at normal priority for 4×N6lS-80GB with one suite per GPU:
  [run log](jobs/fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-0801-015625/run_20260801_015625.log).
  Failed after 5 hours because Gemini repeatedly returned only a subset of the
  required rubric IDs for `instore` task `10826032`, trial 3. All 6.1 GB of
  schema-3 artifacts are retained in
  `artifacts_failed_gemini_rubric_coverage_20260801_090133/`.
- `pt-jbha1xcn` —
  `fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-rubricretry-spot-0801-090316`,
  submitted at normal priority for spot 4×N6lS-80GB after reserved member quota
  rejected the first retry:
  [run log](jobs/fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-rubricretry-spot-0801-090316/run_20260801_090316.log).
  The evaluator prompt now requires exact full rubric coverage, and invalid
  semantic output is retried in the same window up to three times before the
  existing task-level retry policy applies. The job remained unscheduled with
  zero replicas for over two hours and was deleted before startup after a
  reserved retry acquired resources.
- `pt-ebisobi4` —
  `fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-rubricretry-reserved-0801-111249`,
  reserved 4×N6lS-80GB retry of the same corrected protocol:
  [run log](jobs/fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-rubricretry-reserved-0801-111249/run_20260801_111249.log).
  Submitted at 11:12 CST and allocated all four GPUs at 11:13 CST. All 34
  regression tests and the 16-trajectory four-suite gate passed; the four
  formal shard workers started at 11:22 CST. Stopped at 11:33 CST after the
  remote API key was rotated. Its 62 formal trajectories and all journals were
  excluded from the final run and retained in
  `artifacts_invalid_api_key_no_model_output_20260801_113327/`.
- `pt-1eppiued` —
  `fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-newkey-rubricretry-0801-113456`,
  reserved 4×N6lS-80GB clean restart after the API-key rotation:
  [run log](jobs/fsh-vitabench-qwen35-full-4gpu-gptca-gemini25-newkey-rubricretry-0801-113456/run_20260801_113456.log).
  The new key passed direct preflight requests for both `gpt-4.1-ca` and
  `gemini-2.5-flash` before evaluation. The prior journals independently show
  non-empty remote outputs (785 GPT and 434 Gemini responses), so the restart
  is an isolation choice rather than evidence that the prior APIs returned no
  content. The same probes passed inside the job; all 34 regression tests and
  the 16-trajectory four-suite gate passed before the four clean formal shard
  workers started. Stopped and deleted at 12:18 CST after 151 of 1,600 formal
  trajectories when remote API spend reached about CNY 50. Partial artifacts
  and journals remain in `artifacts/`; no further remote billing is occurring.
