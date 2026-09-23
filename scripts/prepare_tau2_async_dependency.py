"""Prepare experiment-local Tau2 with APN validation and pooled HTTP connections."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "tau2-bench/src"
DESTINATION = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src"


def main():
    shutil.copytree(SOURCE, DESTINATION, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    tool_file = DESTINATION / "tau2/domains/telecom/user_tools.py"
    text = tool_file.read_text()
    old = """        if isinstance(apn_settings, dict):
            apn_settings = APNSettings(**apn_settings)
"""
    if old not in text:
        raise RuntimeError("Inspect the current Tau2 APN setter before applying this fix")
    tool_file.write_text(text.replace(old, "        apn_settings = APNSettings.model_validate(apn_settings)\n", 1))
    llm_file = DESTINATION / "tau2/utils/llm_utils.py"
    text = llm_file.read_text()
    old = "httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)"
    if old not in text:
        raise RuntimeError("Inspect the current Tau2 HTTP client before changing its connection pool")
    # At most 80 dialogues share this client. Keep their connections reusable;
    # model execution concurrency is bounded separately by the User server.
    llm_file.write_text(text.replace(old, "httpx_limits = httpx.Limits(max_keepalive_connections=80, max_connections=80)", 1))
    print(DESTINATION)


if __name__ == "__main__":
    main()
