#!/bin/bash
set -ex

exec > >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') [postStart] $line" | tee -a /tmp/setup_node.log /proc/1/fd/1; done) 2>&1

echo "[Setup] Installing Slime fork with PhaseCallback support..."
# NOTE: Once PhaseCallback is upstreamed to THUDM/slime, change this to:
#   pip install "slime @ git+https://github.com/THUDM/slime.git"
pip install --no-cache-dir "slime @ git+https://github.com/aishukamal/slime.git@feat/phase-callbacks"

echo "[Setup] Installing timeslice client and callback package..."
pip install --no-cache-dir "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"
pip install --no-cache-dir "timeslice-slime @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=guides/rl-frameworks/slime/package"
pip install --no-cache-dir "grpcio>=1.81.1" "protobuf>=7.35"

echo "[Setup] Ensuring HuggingFace assets exist in /tmp..."
python3 -c "
import os
from huggingface_hub import snapshot_download, hf_hub_download
if not os.path.exists('/tmp/Qwen2.5-0.5B-Instruct/config.json'):
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='/tmp/Qwen2.5-0.5B-Instruct')
if not os.path.exists('/tmp/dapo-math-17k/dapo-math-17k.jsonl'):
    hf_hub_download(repo_id='zhuzilin/dapo-math-17k', filename='dapo-math-17k.jsonl', repo_type='dataset', local_dir='/tmp/dapo-math-17k')
"

echo "[Setup] Node setup complete."
