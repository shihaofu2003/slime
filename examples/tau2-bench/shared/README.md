# Shared tau2-bench utilities

This directory is reserved for small helpers that are used by more than one
stage (`eval`, `sft`, `rft`, `rl`, or `opd`). Keep stage-specific code in that
stage's directory.

`protocol_profiles.py` is the single definition of the tau2 Agent
`current-single`, `strict-single-v1`, `dependency-safe-multi`, and
`agent-owned-dependency-safe-multi` profiles. `agent_contract.py` adds the
signed system prompt, Agent-only native schemas, User-tool ownership inventory,
and strict source-to-Agent view used by boundary-v2 SFT, official eval, and RL.
`strict-single-v1` reuses the native Agent view and ownership boundary but is
unsigned and rejects multi-block model output instead of batch-executing it.
