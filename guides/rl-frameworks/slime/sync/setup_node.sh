#!/bin/bash
set -ex

# Redirect all stdout/stderr to both log file and container console with timestamps
exec > >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') [postStart] $line" | tee -a /tmp/setup_node.log /proc/1/fd/1; done) 2>&1

echo "[Setup] Updating Slime codebase to timeslice branch..."
git -C /root/slime remote add fork https://github.com/jessicaochen/slime.git || true
git -C /root/slime fetch fork
git -C /root/slime checkout timeslice
git -C /root/slime reset --hard fork/timeslice

echo "[Setup] Installing Python client library..."
pip install -U grpcio protobuf
pip install git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git@11a661f#subdirectory=pkg/client/python

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
