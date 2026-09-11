# tau2-eval-qwen35-official-native-four-domain

Purpose: compare Qwen3.5-4B Agent thinking and non-thinking modes under the
same Tau2 official-native four-domain evaluation protocol.

## Protocol

- Agent: `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3.5-4B`, Tau2
  `llm_agent`, SGLang `qwen3_coder` tool-call parser.
- User: Qwen3.6-27B, non-thinking, text-only, language-only simulation.
- Domains: Airline, Retail, Telecom, Banking Knowledge with BM25 retrieval.
- Budget: all test tasks, 4 trials, seed 300. The two arms differ only in the
  Agent `enable_thinking` request setting.

## Jobs

- Job `12267`: expected RED wrapper regression before implementation —
  [run log](jobs/12267-qwen35-wrapper-red-0902-133410847/run_*_20260902_133410847.log).
- Job `12270`: post-implementation cluster preflight —
  [run log](jobs/12270-qwen35-wrapper-green-0902-133704849/run_*_20260902_133704849.log).
- Job `12274`: Qwen3.5-4B Agent thinking enabled, full seed-300 evaluation —
  [run log](jobs/12274-qwen35-thinking-official-native-four-domain-full-seed300-0902-134042667/run_*_20260902_134042667.log).
- Job `12275`: Qwen3.5-4B Agent thinking disabled, full seed-300 evaluation —
  [run log](jobs/12275-qwen35-nonthinking-official-native-four-domain-full-seed300-0902-134049491/run_*_20260902_134049491.log).
