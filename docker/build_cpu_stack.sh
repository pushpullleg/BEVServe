#!/usr/bin/env bash
# Build the CPU-only model stack for the public T1 smoke test (no GPU, no dataset, no checkpoint).
# Used identically by docker/Dockerfile.cpu, the CI workflow and local runs:
#
#   python3.10 -m venv .venv-cpu && PYTHON=.venv-cpu/bin/python bash docker/build_cpu_stack.sh
#
# Same pinned sources and Python package versions as the GPU image, except:
# - torch/torchvision CPU wheels (2.7.0+cpu / 0.22.0+cpu) instead of +cu128;
# - mmcv-full 1.7.2 compiled with its CPU ops only (FORCE_CUDA=0);
# - BEVDet's Python sources are used without its compiled bev_pool_v2 extension: BEVDet's
#   setup.py lists the .cu file as a regular source, so the extension cannot be built without
#   CUDA. The smoke test substitutes a pure-PyTorch implementation at test time.
set -euo pipefail

PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC_DIR:-$HERE/../.cache/src}
MMCV_REPO=${MMCV_REPO:-https://github.com/open-mmlab/mmcv.git}
MMCV_REF=${MMCV_REF:-4c01b026f0afa5a91a5f54aea313788da1e40f95}
BEVDET_REPO=${BEVDET_REPO:-https://github.com/pushpullleg/BEVDet.git}
BEVDET_REF=${BEVDET_REF:-a0142cc1074fefa0079b1b1f10e0367047913c86}
export MAX_JOBS=${MAX_JOBS:-8} PIP_DISABLE_PIP_VERSION_CHECK=1

fetch() {  # fetch <repo> <commit> <dir>
  rm -rf "$3" && git init -q "$3"
  git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD
}

"$PY" -m pip install -q "setuptools<70" wheel ninja==1.13.2
# torch from the PyTorch index without dependencies: every dependency is an exact pin in
# requirements-model.txt and comes from PyPI (one index per package, no index mixing).
"$PY" -m pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.0+cpu torchvision==0.22.0+cpu
mkdir -p "$SRC"
grep -v -E '^(nvidia-|triton==)' "$HERE/requirements-model.txt" > "$SRC/requirements-cpu.txt"
"$PY" -m pip install -q --no-deps -r "$SRC/requirements-cpu.txt"

fetch "$MMCV_REPO" "$MMCV_REF" "$SRC/mmcv"
git -C "$SRC/mmcv" apply "$HERE/patches/mmcv-1.7.2-torch27-get-stream.patch"
MMCV_WITH_OPS=1 FORCE_CUDA=0 CUDA_HOME= "$PY" -m pip install -q --no-build-isolation --no-deps "$SRC/mmcv"

fetch "$BEVDET_REPO" "$BEVDET_REF" "$SRC/bevdet"
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$(cd "$SRC/bevdet" && pwd)" > "$SITE/bevdet-sources.pth"

"$PY" -m pip install -q --no-deps -e "$HERE/.."
"$PY" -m pip install -q pytest==9.1.1 httpx==0.28.1 iniconfig==2.3.1 pluggy==1.7.0
"$PY" -m pip freeze > "$SRC/manifest.txt"
echo "CPU stack ready: $("$PY" -c 'import torch, mmcv; print("torch", torch.__version__, "mmcv", mmcv.__version__)')"
