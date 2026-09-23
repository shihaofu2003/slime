# serve: sglang 推理服务部署

Experiment name: `serve`
Purpose: 长驻 sglang 推理服务，把 `Qwen3-4B-Instruct-2507` 以一卡部署在计算集群上，
向集群外（登录节点 / 本地）暴露 OpenAI 兼容的 `ip:port`。

## 任务

- [sglang-serve-qwen3-4b-instruct](jobs/20858-sglang-serve-qwen3-4b-instruct-0918-102236523/run_0_20260918_102236523.log)
  — 1 GPU, `python3 -m sglang.launch_server`, serving
  `models/Qwen3-4B-Instruct-2507` (HF safetensors; the `_torch_dist` dir is
  Megatron `.distcp` training format and cannot be served directly),
  `tp=1`, `bf16`, `context-length 32768`, OpenAI-compatible API.
  Parser choices for this non-thinking 2507 model (this sglang build):
  **no** `--reasoning-parser` (plain `qwen3` misclassifies content into
  `reasoning_content`; `qwen3_non_thinking` does not exist here) and
  `--tool-call-parser qwen` (`qwen3_coder` fails to parse `<tool_call>` text,
  leaving `tool_calls` empty).

## 访问方式

- base_url: `http://10.119.96.115:30000/v1` (job id 20858, node `pt-8f208ea1…`)
- model: `qwen3-4b-instruct-2507`；无需 api_key（任意值即可）
- 从登录节点已验证：`/health` 200、中英文 chat、function calling
  （`tool_calls` 正确解析出 `get_weather(city=北京, date=明天)`）。
- 服务随 job 生命周期存活；停止用 `job stop <jobid>`。IP/端口随重新提交变化，
  以 run log 中 `SERVICE READY` 段为准。
