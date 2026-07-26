#!/bin/bash

set -ex

# 该脚本用于在 conda 环境中构建与 docker/Dockerfile 基本一致的运行栈：
# CUDA 12.9 运行库、SGLang、Megatron-LM、slime，以及训练/rollout 需要的
# native kernels。原脚本会把 micromamba 安装到容器用户 home，并把源码 checkout
# 到 /root。当前服务器的 /root 会在容器重启后重置，因此需要长期保留的环境元数据、
# 源码、编译缓存、模型缓存和运行时临时目录都必须放在 /mnt/afs 下。
export PERSISTENT_ROOT="${PERSISTENT_ROOT:-/mnt/afs/users/fush}"

export CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/conda}"
# conda 环境名称使用用户前缀，避免和共享 conda 中其他人的 slime 环境冲突。
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-fsh-slime}"

# BASE_DIR 只用于存放构建/安装时需要的外部源码依赖，例如 sglang 和 Megatron-LM。
# 当前 slime 源码本体是本脚本所在的 slime-main 目录，不会默认 clone 到 BASE_DIR。
# 默认把外部源码依赖直接放在 ServiceAgent 下，目录结构为：
#   ServiceAgent/slime-main     当前 slime 源码
#   ServiceAgent/sglang         SGLang 源码依赖
#   ServiceAgent/Megatron-LM    Megatron-LM 源码依赖
export BASE_DIR="${BASE_DIR:-$PERSISTENT_ROOT/projects/ServiceAgent}"
export CACHE_ROOT="${CACHE_ROOT:-$BASE_DIR/.cache/slime}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 使用共享的持久化 conda 安装，不再在 /root 下重新安装 micromamba。
# 这样 conda 环境会保存在 /mnt/afs/conda/envs，容器重启后仍然存在。
export PATH="$CONDA_ROOT/bin:$CONDA_ROOT/condabin:$PATH"

# 所有包缓存、模型缓存和编译缓存都不要写入 /root。这里设置的默认路径也会影响后续
# 使用 HuggingFace、torch、triton、ray、ModelScope、pip 或 cargo 的下载/训练脚本。
# conda 程序和环境仍使用 /mnt/afs/conda，但下载包缓存放到用户目录，避免要求共享
# /mnt/afs/conda/pkgs 对当前容器用户可写。
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
export CARGO_HOME="${CARGO_HOME:-$CACHE_ROOT/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$CACHE_ROOT/rustup}"
export RAY_TMPDIR="${RAY_TMPDIR:-$BASE_DIR/tmp/ray}"
export TMPDIR="${TMPDIR:-$BASE_DIR/tmp}"
mkdir -p \
  "$BASE_DIR" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$MODELSCOPE_CACHE" \
  "$CARGO_HOME" \
  "$RUSTUP_HOME" \
  "$RAY_TMPDIR" \
  "$TMPDIR"

# 如果通过 CONDARC 复用了旧版 micromamba 生成的 condarc，移除 nodefaults 元通道。
# 部分新版 solver 会把它误当成真实 channel 并尝试访问，导致超时。
if [ -n "${CONDARC:-}" ] && [ -f "$CONDARC" ]; then
  sed -i '/^\s*-\s*nodefaults\s*$/d' "$CONDARC"
fi

if [ ! -x "$CONDA_ROOT/bin/conda" ] || [ ! -f "$CONDA_ROOT/bin/activate" ]; then
  echo "找不到可用的 conda 或 activate：$CONDA_ROOT" >&2
  exit 1
fi

if [ -d "$CONDA_ROOT/envs/$CONDA_ENV_NAME" ]; then
  if [ ! -w "$CONDA_ROOT/envs/$CONDA_ENV_NAME" ]; then
    echo "conda 环境已存在但不可写：$CONDA_ROOT/envs/$CONDA_ENV_NAME" >&2
    exit 1
  fi
elif [ ! -w "$CONDA_ROOT/envs" ]; then
  echo "conda 环境不存在且无法写入环境目录：$CONDA_ROOT/envs" >&2
  echo "请先确认当前容器用户有权限创建 $CONDA_ROOT/envs/$CONDA_ENV_NAME。" >&2
  exit 1
fi

if [ ! -d "$CONDA_ROOT/envs/$CONDA_ENV_NAME" ]; then
  conda create -n "$CONDA_ENV_NAME" python=3.12 pip -c conda-forge -y
fi
# 按当前服务器约定使用 activate 脚本激活环境，而不是依赖交互 shell 的 conda activate。
source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
export CUDA_HOME="$CONDA_PREFIX"

# 以下版本需要与 docker/Dockerfile 保持一致：
#   - SGLANG_IMAGE_TAG (ARG)                  -> 下方 SGLANG_VERSION
#   - MEGATRON_COMMIT (ARG)                   -> 下方 MEGATRON_COMMIT
#   - PATCH_VERSION (ARG, default "latest")   -> 下方 PATCH_VERSION
export SGLANG_VERSION="v0.5.12.post1"
export SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="latest"

cd "$BASE_DIR"

# 安装 CUDA 12.9，因为这是当前 torch wheel 使用的默认 CUDA 版本。
conda install -n "$CONDA_ENV_NAME" \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y

conda install -n "$CONDA_ENV_NAME" -c conda-forge cudnn -y
# sglang 的 editable install 会构建 Rust 扩展（通过 setuptools-rust 构建 sglang-grpc），
# 因此 conda 环境中需要可用的 rustc 和 cargo。
conda install -n "$CONDA_ENV_NAME" -c conda-forge rust -y

pip install cuda-python==12.9

# 安装 sglang。Dockerfile 的基础镜像是 slimerl/sglang:v0.5.12.post1-cu129，
# 镜像里已经包含带 cu129 native kernels 的 sglang；conda 构建时需要在这里手动安装。
# 后续两步用于清理 sglang 依赖带来的 cu13 包污染：
#   1. 强制重装 torch / sglang-kernel / sgl-deep-gemm 为 +cu129 wheel
#      （PyPI 默认可能解析到 cu13）；
#   2. 卸载被 sglang 拉入的 cu13 nvidia-* runtime libs，再安装对应的 cu12 包，
#      修复 site-packages/nvidia/* 下由 pip uninstall 破坏的共享库目录。
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd "$BASE_DIR"
  git clone https://github.com/sgl-project/sglang.git
fi
cd "$BASE_DIR/sglang"
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple


pip install cmake ninja

# 安装 flash-attn 2，与 Dockerfile 保持一致。
# Megatron 当前支持的最新版本是 v2.7.4.post1。
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install flash-linear-attention==0.4.1
# FlashQLA：Qwen3.5/Qwen3-Next 的可选 GDN backend
# （通过 --qwen-gdn-backend flashqla 启用，需要 SM90+ GPU）。
pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
# 安装 tilelang，与 Dockerfile 保持一致。
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# 使用 --no-build-isolation：TMS 的 setup.py 需要找到 nvcc、headers 和当前环境中
# 已安装的 torch，才能构建 cu${TMS_CUDA_MAJOR} native hook。pip 默认的 PEP 517
# 隔离构建 venv 会隐藏这些依赖，导致 wheel 只有 Python 文件（约 46KB），preload .so
# 没有被编译出来，运行 sglang 时会触发：
# `Only hook_mode=preload supports pauseable CUDA Graph`。
pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation
# 与 Dockerfile 保持一致；这里使用的 fork/branch 与旧版 build_conda.sh 不同。
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# 安装 Megatron-LM。
cd "$BASE_DIR"
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
# 由于后续使用 --no-build-isolation，这里显式预装 Megatron 的构建依赖。
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# 使用 --no-build-isolation：setup.py 会构建 C++ 扩展
# megatron.core.datasets.helpers_cpp，并通过子进程执行 `python3 -m pybind11`。
# 不隔离构建时，pip 使用当前 conda 环境中的 python，而该环境已经安装 pybind11。
# 否则该扩展会被标记为 optional 并静默跳过，后续 GPT dataset 加载会失败。
cd "$BASE_DIR/Megatron-LM" && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# 安装 slime 并应用补丁。

# 默认安装当前脚本所在的 checkout。如果希望安装另一个持久化 clone，可以通过 SLIME_DIR 覆盖。
export SLIME_DIR="${SLIME_DIR:-$SCRIPT_DIR}"
if [ ! -d "$SLIME_DIR/.git" ] && [ ! -f "$SLIME_DIR/setup.py" ]; then
  cd "$BASE_DIR"
  git clone https://github.com/THUDM/slime.git
  export SLIME_DIR="$BASE_DIR/slime"
fi
cd "$SLIME_DIR"
# 先从 requirements.txt 安装 slime 的纯 Python 运行依赖（wandb、ray、accelerate、
# transformers 等），再用 --no-deps 安装 slime 本体，避免 pip 重新解析依赖并覆盖前面
# 已经固定好的 native libs（torch+cu129、sglang-kernel+cu129 等）。
# Dockerfile 中也是通过两个 RUN layer 实现同样的安装顺序。
pip install -r requirements.txt
pip install -e . --no-deps

# 安装 int4_qat kernel，与 Dockerfile 保持一致。
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install . --no-build-isolation

# 对应 PyTorch 兼容问题：https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
# kernels 0.15.x 在导入 `transformers.integrations.hub_kernels` 时会触发
# ValueError("Either a revision or a version must be specified")。
# 因此固定到 <0.15，保证运行时 `import sglang` 正常。
pip install "kernels<0.15.0"

# 应用补丁，与 Dockerfile 行为一致：使用 --3way，并在出现冲突时失败退出。
cd "$BASE_DIR/sglang"
if git apply --check $SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch 2>/dev/null; then
  git update-index --refresh || true
  git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "sglang patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "sglang patch already applied or not applicable, skipping"
fi
cd "$BASE_DIR/Megatron-LM"
if git apply --check $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch 2>/dev/null; then
  git update-index --refresh || true
  git apply $SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "megatron patch already applied or not applicable, skipping"
fi
