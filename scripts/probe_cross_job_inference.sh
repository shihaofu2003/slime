#!/usr/bin/env bash
# Probe cross-job connectivity on the compute cluster: from inside THIS job's
# container, call the sglang server running in a DIFFERENT job's container
# (default: the serve job's endpoint) over the pod network, and verify a real
# inference round-trip. Exits non-zero if any check fails, so the job state
# itself reports the probe result.
#
# Submit (via scripts/submit.sh):
#   bash scripts/submit.sh --experiment cross-job-probe --gpus 1 \
#     [-e TARGET_BASE=http://10.x.y.z:30000] scripts/probe_cross_job_inference.sh

set -uo pipefail

TARGET_BASE="${TARGET_BASE:-http://10.119.96.115:30000}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b-instruct-2507}"
TIMEOUT="${TIMEOUT:-30}"

echo "[probe] local host=$(hostname) ip=$(hostname -I | awk '{print $1}')"
echo "[probe] target=${TARGET_BASE} model=${MODEL_NAME}"
echo "[probe] NOTE: this container is a DIFFERENT job from the sglang server."

fail=0

echo "--- [1/5] TCP reachability (${TARGET_BASE#http://} -> nc) ---"
HOSTPORT="${TARGET_BASE#http://}"
THOST="${HOSTPORT%%:*}"; TPORT="${HOSTPORT##*:}"
if timeout 10 bash -c "cat < /dev/null > /dev/tcp/${THOST}/${TPORT}" 2>/dev/null; then
  echo "TCP OK: ${THOST}:${TPORT} reachable"
else
  echo "TCP FAIL: cannot connect ${THOST}:${TPORT}"; fail=1
fi

echo "--- [2/5] HTTP GET /health ---"
code=$(curl -s -o /tmp/probe_health.out -w '%{http_code}' --max-time "${TIMEOUT}" "${TARGET_BASE}/health" 2>&1) || code="curl-error"
echo "status=${code} body=$(head -c 200 /tmp/probe_health.out 2>/dev/null)"
[[ "$code" == "200" ]] || fail=1

echo "--- [3/5] GET /v1/models ---"
curl -s --max-time "${TIMEOUT}" "${TARGET_BASE}/v1/models" -o /tmp/probe_models.json
python3 - "$MODEL_NAME" /tmp/probe_models.json <<'PY' || fail=1
import json, sys
model, path = sys.argv[1], sys.argv[2]
try:
    ids = [m["id"] for m in json.load(open(path))["data"]]
except Exception as e:
    print(f"models FAIL: bad payload: {e}"); sys.exit(1)
print("models OK:", ids)
sys.exit(0 if model in ids else 1)
PY

echo "--- [4/5] POST /v1/chat/completions (real inference) ---"
curl -s --max-time 120 "${TARGET_BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Answer with the number only.\"}], \"max_tokens\": 64, \"temperature\": 0}" \
  -o /tmp/probe_chat.json
python3 - <<'PY' || fail=1
import json
try:
    d = json.load(open("/tmp/probe_chat.json"))
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    usage = d.get("usage", {})
except Exception as e:
    print(f"chat FAIL: bad payload: {e}"); raise SystemExit(1)
ok = "401" in content
print(f"chat {'OK' if ok else 'FAIL'}: content={content!r} tokens={usage.get('completion_tokens')}")
raise SystemExit(0 if ok else 1)
PY

echo "--- [5/5] tool-call round-trip ---"
curl -s --max-time 120 "${TARGET_BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"查一下北京明天的天气\"}], \"max_tokens\": 512, \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"date\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}], \"tool_choice\": \"auto\"}" \
  -o /tmp/probe_tool.json
python3 - <<'PY' || fail=1
import json
try:
    d = json.load(open("/tmp/probe_tool.json"))
    tcs = d["choices"][0]["message"].get("tool_calls") or []
    name = tcs[0]["function"]["name"] if tcs else None
    args = json.loads(tcs[0]["function"]["arguments"]) if tcs else {}
except Exception as e:
    print(f"tool FAIL: bad payload: {e}"); raise SystemExit(1)
ok = name == "get_weather" and args.get("city") == "北京"
print(f"tool {'OK' if ok else 'FAIL'}: name={name} args={args}")
raise SystemExit(0 if ok else 1)
PY

echo "=============================================================="
if [[ "$fail" == "0" ]]; then
  echo "CROSS-JOB PROBE PASSED: job-to-job inference over the pod network works."
  echo "PROBE_RESULT=PASS"
else
  echo "CROSS-JOB PROBE FAILED: see items above."
  echo "PROBE_RESULT=FAIL"
fi
echo "=============================================================="
exit "$fail"
