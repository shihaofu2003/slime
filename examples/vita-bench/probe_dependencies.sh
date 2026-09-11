#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
SETUP_SCRIPT="${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh"
VENV_DIR="/tmp/serviceagent-vitabench-venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  bash "${SETUP_SCRIPT}" "${VITABENCH_DIR}"
fi

echo "[vitabench-deps-probe] python=$(${VENV_PYTHON} --version 2>&1)"
echo "[vitabench-deps-probe] pip=$(${VENV_PYTHON} -m pip --version)"

VITABENCH_DIR="${VITABENCH_DIR}" VENV_DIR="${VENV_DIR}" "${VENV_PYTHON}" - <<'PY'
import importlib.util
import os
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

source = Path(os.environ["VITABENCH_DIR"]).resolve()
venv = Path(os.environ["VENV_DIR"]).resolve()
with (source / "pyproject.toml").open("rb") as handle:
    requirements = [Requirement(text) for text in tomllib.load(handle)["project"]["dependencies"]]

if len(requirements) != 29:
    raise RuntimeError(f"expected 29 Vita dependencies, found {len(requirements)}")

for requirement in requirements:
    installed = metadata.version(requirement.name)
    if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f"{requirement} is not satisfied by {installed}")
    distribution = metadata.distribution(requirement.name)
    location = Path(distribution.locate_file("")).resolve()
    layer = "venv" if location == venv or venv in location.parents else "system"
    print(
        "[vitabench-deps-probe] DEP",
        f"name={requirement.name}",
        f"required={requirement.specifier or '*'}",
        f"installed={installed}",
        f"layer={layer}",
    )

expected_shared = {
    "litellm": "1.81.11",
    "PyYAML": "6.0.3",
    "tenacity": "9.1.2",
}
for name, wanted in expected_shared.items():
    installed = metadata.version(name)
    if installed != wanted:
        raise RuntimeError(f"{name}=={installed}, expected tau2 version {wanted}")

spec = importlib.util.find_spec("vita")
if spec is None:
    raise RuntimeError("vita is not importable")
resolved = Path(next(iter(spec.submodule_search_locations))).resolve()
if resolved != source and source not in resolved.parents:
    raise RuntimeError(f"vita resolves to {resolved}, expected under {source}")

import slime
import tau2
import torch.utils.tensorboard
from vita.run import get_options, get_tasks

options = get_options()
tasks = get_tasks("delivery", language="english")
selected = get_tasks("delivery", ["10711001"], language="english")
if len(tasks) != 100 or len(selected) != 1 or selected[0].id != "10711001":
    raise RuntimeError("Vita delivery task loading failed")

print(
    "[vitabench-deps-probe] OK",
    f"dependencies={len(requirements)}",
    f"agents={len(options.agents)}",
    f"delivery_tasks={len(tasks)}",
    f"python={sys.version.split()[0]}",
)
PY

"${VENV_DIR}/bin/vita" --help >/dev/null
tau2 --help >/dev/null
echo "[vitabench-deps-probe] OK: vita/tau2 CLIs and all compatibility checks passed"
