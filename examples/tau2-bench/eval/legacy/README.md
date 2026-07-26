# Legacy tau2-bench eval

This directory archives the first working tau2-bench eval test. It is kept for
historical comparison and should not be used as the default eval path.

Known differences from the official path:

- It drives `AgentGymEnv` directly instead of tau2's runner/orchestrator.
- It owns custom prompt construction, action parsing, and pass@k/pass^k
  aggregation.
- It is retained only as an archived eval test; new official eval runs default
  to the `test` split in slime.

Use `../official/` for new evaluation runs.
