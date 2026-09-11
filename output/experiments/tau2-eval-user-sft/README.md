# tau2-eval-user-sft

Purpose: run tau2-bench official Pass@4 evaluations with the trained local user
SFT model served by sglang.

## Tasks

- `pt-414k5xde` / `fsh-tau2-user-sft-agent-sft-multitool-0714-211608` —
  full Pass@4 eval for the multitool agent SFT checkpoint with the trained user
  SFT simulator:
  [jobs/fsh-tau2-user-sft-agent-sft-multitool-0714-211608/run_20260714_211608.log](jobs/fsh-tau2-user-sft-agent-sft-multitool-0714-211608/run_20260714_211608.log).
- `pt-8rg5egt4` / `fsh-tau2-user-sft-agent-areal-sft-0714-211608` — full
  Pass@4 eval for the existing areal strict no-thinking SFT checkpoint with the
  trained user SFT simulator:
  [jobs/fsh-tau2-user-sft-agent-areal-sft-0714-211608/run_20260714_211608.log](jobs/fsh-tau2-user-sft-agent-areal-sft-0714-211608/run_20260714_211608.log).
- `pt-86hgvxiq` / `fsh-tau2-user-sft-agent-qwen3-4b-instruct-0714-211608` —
  full Pass@4 eval for Qwen3-4B-Instruct-2507 with the trained user SFT
  simulator:
  [jobs/fsh-tau2-user-sft-agent-qwen3-4b-instruct-0714-211608/run_20260714_211608.log](jobs/fsh-tau2-user-sft-agent-qwen3-4b-instruct-0714-211608/run_20260714_211608.log).
- `pt-2t8q4t45` / `fsh-tau2-user-sft-agent-qwen3-4b-grpo-v1-0714-211608` —
  full Pass@4 eval for Qwen3-4B-tau2-grpo-v1 with the trained user SFT
  simulator:
  [jobs/fsh-tau2-user-sft-agent-qwen3-4b-grpo-v1-0714-211608/run_20260714_211608.log](jobs/fsh-tau2-user-sft-agent-qwen3-4b-grpo-v1-0714-211608/run_20260714_211608.log).
- `pt-0k0ieknu` / `fsh-tau2-user-sft-agent-qwen3-5-4b-0714-211608` — full
  Pass@4 eval for Qwen3.5-4B with the trained user SFT simulator:
  [jobs/fsh-tau2-user-sft-agent-qwen3-5-4b-0714-211608/run_20260714_211608.log](jobs/fsh-tau2-user-sft-agent-qwen3-5-4b-0714-211608/run_20260714_211608.log).
