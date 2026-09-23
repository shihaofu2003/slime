# two-node-gpu-probe

Purpose: test submission of a two-node GPU job with a short hostname/GPU probe.

- 2026-09-14: job `19005` / `pt-4irtqpzs`, spot, 2 nodes × 8 GPUs (16 total), priority normal. Stopped (`STOPPED`) at user request to switch to normal resources; execution not verified.
- [Submission log](jobs/19005-two-node-probe-8x2-spot-0914-093656633/submit_20260914_093656633.log).
- Run log pending creation: `jobs/19005-two-node-probe-8x2-spot-0914-093656633/run_*_20260914_093656633.log` (job-manager reported pattern).

- 2026-09-14: replacement job `19006`, normal priority, `spot=false`, 2 nodes × 8 GPUs (16 total). Cancelled (`CANCELLED`) at user request. Previously queued because remaining quota was 4 GPUs, below the requested 16.
- [Normal submission log](jobs/19006-two-node-probe-8x2-normal-0914-094203371/submit_20260914_094203371.log).
- Normal run log pending creation: `jobs/19006-two-node-probe-8x2-normal-0914-094203371/run_*_20260914_094203371.log` (job-manager reported pattern).
