# qwen38-27b-inference-smoke

Purpose: verify that the current compute-cluster image can load and serve the
BF16 Hugging Face checkpoint at `/mnt/afs/models/Qwen3.8-27B` with SGLang,
including `xhigh` reasoning, text generation, parsed tool calls, and image input.
Determine whether a Transformers update or checkpoint conversion is necessary.

Name: `qwen38-27b-inference-smoke`.

## Protocol

- One cluster GPU, TP1, BF16, 32,768-token context, and the standard slime
  container/setup path.
- Record GPU, CUDA, PyTorch, Transformers, and SGLang versions from the allocated
  compute node.
- Require successful AutoConfig/tokenizer/processor loading and an `xhigh`
  response with parsed reasoning content and the correct final answer.
- Use the model-card thinking sampling parameters. On SGLang 0.5.13, pass
  `reasoning_effort=xhigh` through `chat_template_kwargs`; its top-level OpenAI
  request schema does not include `xhigh`, but the serving path forwards the
  nested value to the Hugging Face chat template after validation.

## Result

- The allocated node provided one NVIDIA H100 80GB HBM3 GPU (compute capability
  9.0), CUDA 12.9, PyTorch 2.11.0, Transformers 5.8.1, and SGLang 0.5.13.
- Transformers loaded `Qwen3_5Config`, `Qwen2Tokenizer`, and `Qwen3VLProcessor`
  directly from the original 18-shard checkpoint. SGLang loaded
  `Qwen3_5ForConditionalGeneration`; the successful `xhigh` run loaded weights
  in 215.49 seconds. Model weights used
  51.05 GB of GPU memory and the BF16 KV cache allocated 5.20 GB each for K and
  V.
- The `xhigh` request returned 174 characters of parsed reasoning and final
  answer `42` (71 prompt tokens and 66 completion tokens). Only `xhigh` was
  tested among the reasoning-effort levels.
- Text generation returned `42`; the tool parser returned
  `get_temperature({"city": "Beijing"})`; image inference identified a red
  test image as `red`.
- The local model config records Transformers 5.8.0.dev0, while the cluster has
  Transformers 5.8.1. Direct loading and generation succeeded, so no
  Transformers change is needed for this inference path. The dependency warning
  from Megatron Bridge concerns the unused training stack.
- The original Hugging Face checkpoint is directly usable for inference. No
  conversion and no duplicate model under `../models` are needed. This smoke
  validates a 32,768-token serving context; it does not measure the native
  262,144-token maximum.

## Jobs

- `12707` / `pt-qkuc77p1` — one-GPU compatibility smoke, succeeded in 345
  seconds (0.0959 GPU-hours):
  [run log](jobs/12707-qwen38-27b-hf-sglang-smoke-0903-113605218/run_0_20260903_113605218.log).
- `12720` / `pt-28aobjt4` — top-level OpenAI-style
  `reasoning_effort=xhigh`, failed with HTTP 400 after the baseline text request
  because SGLang 0.5.13 rejects `xhigh` in that request field:
  [run log](jobs/12720-qwen38-27b-xhigh-smoke-0903-114949461/run_0_20260903_114949461.log).
- `12725` / `pt-tb9zkv2y` — SGLang 0.5.13-compatible nested `xhigh` request,
  succeeded in 392 seconds (0.1088 GPU-hours):
  [run log](jobs/12725-qwen38-27b-xhigh-compat-smoke-0903-115952222/run_0_20260903_115952222.log).
