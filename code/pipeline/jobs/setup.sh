#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/env.sh
unset HF_HOME
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1 MALLOC_ARENA_MAX=2
uv sync --python-platform x86_64-manylinux_2_28 --offline || uv sync --python-platform x86_64-manylinux_2_28
