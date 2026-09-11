# Experiment aliases

| Submission alias | Real experiment | Applies to |
|---|---|---|
| `audio-curriculum-v1` | `tau2-banking-independent-synthetic-v1` | Historical submissions after job 15754; new jobs use the real name directly. |
| `audio-01` | `tau2-sft-turn-quality-qwen38-xhigh-v1` | Turn-quality resume submissions after jobs 15737 and 15745 were stopped. |

The alias changes only queue-facing names and the automatic job-log directory.
Generated tasks, trajectories, metrics, and acceptance artifacts use the real
experiment directory. Existing alias jobs retain their original names; new
Banking submissions use `tau2-banking-independent-synthetic-v1` and keep total
Banking demand at or below 24 GPUs.
