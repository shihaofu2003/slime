# tau2-eval-official-native-full

Purpose: run the formal four-domain Tau2 evaluation through the official native
`llm_agent` path after the corresponding GPU smoke passed.

## Tasks

- Full evaluation `12259`: Airline, Retail, Telecom, and Banking Knowledge;
  all selected test tasks, four trials, seed 300, eight GPUs —
  [run log](jobs/12259-tau2-official-native-four-domain-full-seed300-0902-124312499/run_*_20260902_124312499.log).
  The evaluation completed 788/788 simulations with zero infrastructure errors;
  pass@1/pass@4(any)/pass^4 is `19.67/30.46/10.15%`. Job-manager recorded a
  non-evaluation wrapper EOF after the summary was written because that wrapper
  was edited while the job was running.
- Exact seed-300 repeat `12279`: same model, official-native protocol, User,
  task set, four trials, concurrency, retrieval, and eight-GPU topology as
  `12259` —
  [run log](jobs/12279-tau2-official-native-four-domain-full-seed300-repeat-0902-135700721/run_*_20260902_135700721.log).
  Completed 788/788 simulations with zero infrastructure errors;
  pass@1/pass@4(any)/pass^4 is `20.94/32.49/10.15%`.

## Protocol

- Agent: two TP1 Qwen3-4B-Instruct-2507 replicas, `official-native`, Tau2
  `llm_agent`, native `/v1/chat/completions`, temperature 0.6.
- User: three TP2 Qwen3.6-27B replicas, non-thinking, language-only,
  `qwen3_coder`, temperature 0.0.
- Initial domain concurrency: `1:2:2:4`, global concurrency 9, with completed
  slots redistributed; Banking retrieval is BM25.
