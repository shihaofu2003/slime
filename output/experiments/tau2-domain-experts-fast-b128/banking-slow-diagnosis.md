# Banking evaluation slowdown diagnosis

Checked2026-09-20 around16:14 CST,job21769. Read-only diagnosis;running evaluation unchanged.

The observed chain is repeated KB_search → accumulated long context → expensive Agent prefill and queueing,amplified by uneven cache-aware routing. The raw model usually stops quickly,including unsuccessful endings. This does not mean raw Banking capability is stronger.

| Evidence | Raw | Airline iter49 |
|---|---|---|
| Saved simulations |388 complete|6 complete;long-running simulations excluded|
| KB_search calls |482 total,1.24/simulation,max16|166 total,27.67/simulation,max61|
| Mean KB return characters |19892|16188|
| Agent prompt tokens,median/max over saved messages |7827/87043|148129/225046|
| Mean saved trajectory duration |25.43s|108.99s;excludes unfinished long trajectories|

Tasks and environment policy compare equal. Agent temperature0.6/top_p1/max_tokens1200,User temperature0/max_tokens512,4 trials,max_steps200,max_errors10 also match. Raw was a four-domain run;current Banking-only concurrency9 is not an identical serving-load control.

Raw task015 took12.2s with1 KB_search. Current task015 performed61 searches,reached201 messages/max_steps,with approximately970k content characters. Queries repeatedly seek referral-link tools and instructions. task006 performed42 searches and hit max_steps. Historical SFT4673 Banking trajectories already show43 searches on task006,41 on task008,and43 on task015;the behavior cannot be attributed solely to subsequent Airline RL.

Agent worker0 requests observed at16:13 CST:input236441 tokens,cached3655,output33,queue90.09s,forward53.16s;another input183189,cached3652,output27,queue105.71s,forward42.66s. Worker0 has7 queued requests and roughly1M pending tokens. Worker1 last completed a non-health request at15:41:05 and remains healthy but idle afterward. User workers are mostly idle;observed active User request took1.03s. Router uses cache_aware,cache_threshold0.3,balance_abs_threshold64,balance_rel_threshold1.5,with both workers registered and no warning/error. Observed load concentration is established;the exact routing decision mechanism was not traced.

Raw completed388 simulations in2506.54s,with380 user_stop and8 max_steps;success10/388 (2.58%). Agent mean0.995s/call,User1.284s/call. Faster termination does not imply correct task completion.

Sources: [current trajectories](eval/banking-airline-expert-iter49/trajectories/),[current inference logs](eval/banking-airline-expert-iter49/),[SFT trajectories](../tau2-airline-db-count-tuning/eval/four-domain-sft-iter4673/trajectories/),[raw results](/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/data/simulations/tau2_official_Qwen3-4B-Instruct-2507-raw-qwen36-fixed_raw-full-qwen36-bm25-fixed_seed300_0910_raw_fixed_banking_knowledge_test_4trials/results.json).
