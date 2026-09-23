# tau2-sft-areal3-banking-simplified-full-20260920

Purpose: prepare a new four-domain SFT dataset using the existing processed AReaL three-domain rows and the full simplified Banking export. Fresh SFT training submitted as job21863.

## Dataset

[JSONL](data/agent_areal3_banking_simplified_full_sft_20260920.jsonl),[source paths and verification statistics](data/agent_areal3_banking_simplified_full_sft_20260920.stats.json).

Merge order:AReaL official-native expanded30376 rows,then Banking simplified full5672 rows. Preserve source lines verbatim,including messages,tools,metadata,and row order. No deduplication,resampling,or field rewriting. Banking source is the full20260911 simplified export used by Banking simplified full SFT,not the old7016-row Banking expert export or the completed408 subset.

| Domain | Old mixed dataset | New dataset |
|---|---:|---:|
| Airline |13783|13783|
| Retail |13614|13614|
| Telecom |2979|2979|
| Banking |7016|5672|
| Total |37392|36048|

## Verification

All36048 JSONL rows parsed;required messages/tools/metadata fields and final Assistant targets checked. New output was reread and compared line-by-line with both sources:exact concatenation. New and old `agent_banking_mixed_sft.jsonl` three-domain subsequences compared across30376 rows:zero different lines,identical order. This includes every message,tool schema,and metadata field. The old mixed file was not changed.

Target rows:19693 tool_call,16355 text. Existing source tokenization/export checks are reused;no new tokenizer validation was performed because source rows were not modified.

## Training

Submitted2026-09-20 16:55 CST:job21863,8 GPUs,normal priority. SUCCEEDED;initial queue delay was due to cluster quota/resource shortage. [Launcher](run_sft.sh);[run log](jobs/21863-sft-areal3-banking-simplified-full-0920-165544878/run_0_20260920_165544878.log).

Same recipe as historical full-domain SFT job16543. Current generic launcher is identical to its submitted snapshot. Raw Qwen3-4B-Instruct-2507 torch-dist initialization,fresh optimizer;2 epochs,batch16,LR1e-5 cosine to1e-6,warmup0.1,Adam betas0.9/0.95,weight decay0.1,seed1234,qwen3_full target-only per-token loss,shuffle,16384 tokens/GPU,TP/PP/CP1,full recompute,save every400 updates,start rollout0,W&B offline. Changes:data path,project checkout,and output/log names. No evaluation bundled.

Checkpoint directory:`/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920`. Directory did not exist before submission. Dataset36048 rows gives4506 rollout batches at2 epochs,batch16;training duration estimated around3 hours after scheduling,based on historical run.

Completed2026-09-20 around19:50 CST:job21863 SUCCEEDED,exit0,total runtime2.8159h (22.5269 GPU-hours). Completed4506 updates,step0–4505;final checkpoint `iter_0004505` saved. Final loss0.22577,grad_norm2.6782,LR1e-6. HF conversion and evaluation subsequently completed in job21925.

## Final four-domain evaluation

Submitted2026-09-20 19:54 CST:job21925 (`pt-0t0yrrj3`),SUCCEEDED,8 GPUs,normal priority. [Launcher](eval_final_four_domain.sh);[run log](jobs/21925-sft-iter4505-four-domain-eval-0920-195444259/run_0_20260920_195444259.log).

Converts final iter4505 to `iter_0004505_hf` using original Qwen3-4B-Instruct-2507 HF metadata,then reuses existing official four-domain full wrapper. Parameters unchanged:full test197 tasks ×4 trials=788 simulations,seed300,Agent temperature0.6/top_p1/max_tokens1200,max_steps200,max_errors10,Qwen3.6-27B non-thinking User temperature0/max_tokens512,Banking BM25;domain concurrency1/2/2/4,global9,slot borrowing enabled. Results:`eval/four-domain-sft-iter4505/`.

Final evaluation completed2026-09-20 around21:01 CST:788/788 simulations,zero infrastructure errors,total job runtime1.0101h. [Summary](eval/four-domain-sft-iter4505/seed300_20260920_sft_iter4505_four_domain_summary.json).

| Domain | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Airline |42.50%|65.00%|30.00%|
| Retail |58.75%|77.50%|35.00%|
| Telecom |47.50%|82.50%|12.50%|
| Banking |4.12%|8.25%|1.03%|
| Overall |27.92%|43.15%|13.20%|

Banking:1170 KB_search calls over388 trajectories,mean3.0155. Long-search behavior improved versus old full-domain SFT,while absolute task success remains low. Official evaluation does not impose the RL16384-token training cap.

## Banking SFT versus RL context lengths

Verified simplified export:5672 rows from1651 trajectories;stored token counts mean6427.75,max16368 (export metadata,not a new tokenizer run). BM25 contributes3811 rows from779 trajectories:mean1.498 KB_search calls/trajectory,max4. Golden retrieval contributes1861 rows from872 trajectories and has no KB_search tool. These are different retrieval conditions within the SFT mix.

The Qwen3.8 simplifier deletes selected Agent calls together with their results and unnecessary Assistant text;it does not summarize retained tool-return bodies. Acceptance requires the complete Agent-view candidate,including system prompt/tools,to fit16384 tokens. Export then creates Assistant-target prefix rows and checks their token cap. Source accounting:565 originally over-cap trajectories recovered after simplification;22 still over-cap excluded;other rejected/error candidates also excluded. SFT therefore sees selected,edited demonstrations,whereas RL generates fresh trajectories with actual accumulating tool results. A target-only loss mask does not remove context tokens or their memory cost.

User-reported RL batch10 observation (run identity/source log not supplied here;not independently recounted):4263 final-attempt over-cap trajectories,mean Agent turns4.90,median5,p90≤6,mean searches4.50. Separate sample130 over-cap trajectories:mean KB_search return approximately14400 characters. Tool-call Assistant messages count as turns. These are conditional statistics of failures,not the complete rollout population;without total attempts/groups they do not establish an overall over-cap rate.

Retry semantics reported by user and consistent with current rollout code:reject over-cap sampling,reset environment/database/User state and regenerate the same task,at most2 retries (3 attempts total). On exhaustion,the unusable sample causes its K8 group to be filtered and replaced. Retry changes the sampled route but does not reduce tool-result size. Four to five results of14400 characters each accumulate57600–72000 characters before system prompt/tools/dialogue;token count depends on actual text/tokenization. This explains overflow after few Agent turns without requiring dozens of repeated searches. Group filtering can magnify sample rejection and favor shorter/easier tasks.

Source code:[candidate acceptance](../../../examples/tau2-bench/sft/qwen_simplify_banking.py),[deletion and prefix export](../../../examples/tau2-bench/sft/simplify_banking_expert.py),[RL retries](../../../examples/tau2-bench/rl/rollout.py). Source data/statistics paths are recorded in the dataset stats JSON above. No RL/evaluation parameters or data were changed during this diagnosis.

Evaluation context audit:Agent servers report context_len262144. No context-limit rejection,HTTP400/500,CUDA OOM,or traceback matches in the five SGLang worker logs;summary reports zero infrastructure errors. Banking197/388 trajectories have at least one recorded Agent prompt above16384 tokens (50.77%);max121154. Of those197,188 terminate user_stop,9 max_steps,7 succeed. These prompt-based counts exclude the generated response and are not an exact RL full-trajectory tokenizer replay. Banking overall:378 user_stop,10 max_steps. Across other domains,too_many_errors occurs3 times in Retail (Product not found) and2 in Telecom (nonexistent ###TRANSFER### tool),not context-limit errors. Long context may affect quality/latency,but these results do not identify it as the causal reason for task failure.
