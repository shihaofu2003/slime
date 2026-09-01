# tau2-eval-qwen36-user-timing-tp2

Purpose: evaluate the raw Qwen3-4B-Instruct-2507 Agent against a local
non-thinking Qwen3.6-27B User while recording model-startup, request, trajectory,
and per-domain timing evidence.

Name: `tau2-eval-qwen36-user-timing-tp2`.

## Tasks

- Job `11728` / `pt-cqopqo59` — 4×H100 allocation; Agent TP1 on GPU 0,
  User TP2 on GPUs 1–2, GPU 3 reserved, 65,536-token User context, seed 300,
  all three test domains, 100 tasks × 4 trials:
  [run log](jobs/11728-qwen3-4b-agent-qwen36-user-tp2-timed-0901-095135970/run_*_20260901_095135970.log).
