# cross-job-probe: 跨任务推理连通性测试

Experiment name: `cross-job-probe`
Purpose: 验证计算集群上一个任务（job）的容器能否通过 pod 网络调用**另一个任务**里
部署的 sglang 推理服务（job 20858，`http://10.119.96.115:30000/v1`）。

## 任务

- [cross-job-inference-probe](jobs/20868-cross-job-inference-probe-0918-103035734/run_0_20260918_103035734.log)
  — 1 GPU 探测容器（`pt-b4dc7262…`，ip `10.119.111.57`，与推理服务不同节点），
  依次执行 5 项检查：TCP `/dev/tcp` 连通、`/health`、`/v1/models`、真实 chat
  推理（17×23=401）、tool-call 往返（`get_weather` 参数解析）。

## 结果

全部通过（`PROBE_RESULT=PASS`，job state SUCCEEDED，exit 0）：

1. TCP 到 `10.119.96.115:30000` 可达
2. `/health` → 200
3. `/v1/models` → `qwen3-4b-instruct-2507`
4. chat 推理 → `content='401'`
5. tool call → `get_weather(city=北京, date=明天)`

结论：集群内跨任务、跨节点调用推理服务可行；推理服务保持 job 20858 运行即可被
其他训练/评测任务直接当作 OpenAI 兼容 endpoint 使用。
