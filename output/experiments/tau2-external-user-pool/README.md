# tau2-external-user-pool

Purpose: serve the unchanged Qwen3.6-27B User independently so each 8-GPU asynchronous RL job uses 2 Trainer GPUs and 6 Generator GPUs (three TP2 replicas). Training is paused at user request.

## Deployment

User job: 4 GPUs, two identical TP2 Qwen3.6-27B replicas on cards0/1 and2/3, worker ports30001/30002. SGLang router on port30000 distributes requests round-robin. Public API: `http://<User job compute IP>:30000/v1`; full address printed on readiness. Model name `Qwen3.6-27B-tau2-user-nonthinking`. Keep dtype BF16, context65536, memory fraction0.90, max-running-requests32 per replica, qwen3 reasoning parser and qwen3_coder tool parser. Clients retain temperature0, top_p1, max_tokens512 and enable_thinking=false.

- Job20873: [run log](jobs/20873-tau2-user-pool-2x-tp2-0918-104403882/run_0_20260918_104403882.log), 4 GPUs / 56 CPUs / 792GB, normal priority. Launcher: `scripts/serve_tau2_user_pool.sh`. Worker/router logs under `service/<timestamp>/`.

## Training configuration

The async launcher now defaults to `USER_SGLANG=0`, `TAU2_USER_API_BASE=http://10.119.96.116:30000/v1`, and `TAU2_USER_API_KEY=EMPTY`. Override the API base on submission if the User job is replaced; its compute IP is not permanent. The async launcher allocates all8 GPUs to Ray; Trainer uses2, rollout uses6 with2 GPUs per engine. No User model is launched inside the RL job. Local-User mode remains supported with the previous2+4+2 allocation. User service is shared across training jobs and is stopped separately.

Shell syntax checks passed. Executed configuration checks confirm external User -> Ray8/Generator6 and local User -> Ray6/Generator4. Service startup and API inference validation pending. This changes serving capacity and asynchronous timing; comparisons to old runs must note the deployment difference. No training job restarted.

## User parameter parity

| Setting | Previous training User / new User replicas |
|---|---|
| Weights | `ServiceAgent/models/Qwen3.6-27B` |
| Served model | `Qwen3.6-27B-tau2-user-nonthinking` |
| Tensor parallel / dtype | 2 / bfloat16 |
| Context / memory fraction | 65536 / 0.90 |
| Language-only / max running requests | enabled / 32 per replica |
| Reasoning / tool parser | qwen3 / qwen3_coder |
| Request temperature / top_p | 0.0 / 1.0 |
| Request max_tokens | 512 |
| Request extra body | `{"chat_template_kwargs":{"enable_thinking":false}}` |

Nonthinking is enforced by the request extra body, not the served-model name. The existing async launcher continues to export all four request settings through Ray runtime env to the User client. Only replica count increases from1 to2; per-replica settings stay unchanged.

Latest scheduler check: job20873 QUEUED, GPU quota remaining3 versus requested4. No compute IP assigned and no runtime/API verification yet. Unrelated serving job20858 was left running.

2026-09-18 10:53 CST verification: job20873 RUNNING (`pt-ctfok8v5`); both workers ready and registered healthy with round_robin router. API `http://10.119.96.116:30000/v1`, API key `EMPTY`. Actual ServerArgs for both replicas match the parameter table. From the current environment, `/v1/models` and four concurrent chat requests passed: two arithmetic text responses (17×23=391), two structured get_weather tool calls with valid JSON city arguments. All four used temperature0/top_p1/max_tokens512/enable_thinking=false and reported reasoning_tokens0. Both worker logs show completion HTTP200 responses. Initial diagnostic assertion mistakenly expected401; corrected to391 and reran all four successfully. Training remains stopped; no 8-GPU RL launch or full training smoke performed.

2026-09-18 deployment implementation: `run_tau2_areal_async.sh async ...` now defaults to external User and explicitly fixes Generator TP2. Ray sees cards0–7; the disjoint resource arguments allocate Trainer2 and Generator6, creating three inference engines. Startup prints the topology and User API endpoint. Explicit `USER_SGLANG=1` retains local2+4+2 mode; sync keeps its previous local default. Shell syntax and launcher execution with a stubbed training process passed for default async, explicit local async, default sync and an overridden endpoint. Request parameters were checked in the exported environment. These checks do not constitute a live RL optimizer-step validation. Training remains paused and the existing post-training official evaluation deployment is unchanged.

2026-09-18 12:22 CST: live8-GPU client20886 verified external User calls and completed batch0 optimizer update, weight sync and version1 sampling. Both User replicas served training requests; service20873 remains RUNNING.

2026-09-20 restart:old User20873 STOPPED at user request. Replacement User22031 submitted with unchanged serving/model parameters,4 GPUs,normal priority. [Run log](jobs/22031-tau2-user-pool-refresh-sft4505-0920-230958466/run_0_20260920_230958466.log). It will serve both8-GPU [SFT4505 expert jobs](../tau2-domain-experts-sft4505-b128/README.md);new compute IP is read from its readiness log,not the old service address.

2026-09-20 23:39 CST:replacement22031 RUNNING and ready at `http://10.119.97.164:30000/v1`. Both22033 and22034 connected and completed real smoke optimizer updates using this service.
