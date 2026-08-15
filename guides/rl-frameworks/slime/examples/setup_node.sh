#!/bin/bash
set -ex

exec > >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') [postStart] $line" | tee -a /tmp/setup_node.log /proc/1/fd/1; done) 2>&1

# All nodes (head + workers) need the Slime fork because Ray actors on workers
# deserialize classes defined in the fork (e.g. SGLangEngine.pull_weights).
echo "[Setup] Replacing base image Slime with fork (PhaseCallback support)..."
# NOTE: Once PhaseCallback is upstreamed to THUDM/slime, remove this block.
# Clone to temp dir first, then swap — avoids window where /root/slime is missing
git clone --depth 1 -b feat/phase-callbacks https://github.com/aishukamal/slime.git /root/slime-fork
rm -rf /root/slime
mv /root/slime-fork /root/slime
# --no-deps: all deps (torch, accelerate, etc.) are already in the image.
# Without --no-deps, pip pulls nvidia-cudnn which corrupts the already-running
# CUDA runtime and crashes Ray worker processes.
cd /root/slime && pip install --no-cache-dir --no-deps -e .
cd /

echo "[Setup] Patching placement_group.py: separate PGs for train/rollout node pinning..."
python3 -c "
import pathlib, os

p = pathlib.Path('/root/slime/slime/ray/placement_group.py')
src = p.read_text()

# 1. Add os import if not present
if 'import os' not in src:
    src = 'import os\n' + src
    print('  added: import os')

# 2. Replace _create_placement_group to accept bundle_resources
old_func = '''def _create_placement_group(num_gpus):
    \"\"\"Create a placement group with the specified number of GPUs.\"\"\"
    if num_gpus == 0:
        return None, [], []

    bundles = [{\"GPU\": 1, \"CPU\": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy=\"PACK\")'''

new_func = '''def _create_placement_group(num_gpus, bundle_resources=None):
    \"\"\"Create a placement group with the specified number of GPUs.\"\"\"
    if num_gpus == 0:
        return None, [], []

    bundles = []
    for i in range(num_gpus):
        bundle = {\"GPU\": 1, \"CPU\": 1}
        if bundle_resources and i < len(bundle_resources) and bundle_resources[i]:
            bundle.update(bundle_resources[i])
        bundles.append(bundle)
    pg = placement_group(bundles, strategy=\"SPREAD\")'''

if old_func in src:
    src = src.replace(old_func, new_func)
    print('  patched: _create_placement_group (custom resources + SPREAD)')
else:
    print('  SKIP (pattern not found): _create_placement_group')

# 3. Replace entire create_placement_groups to use separate PGs when custom resources are set.
#    One combined PG with IP-based reordering shuffles the actor/rollout assignment,
#    putting trainers on the samplers node and vice versa. Separate PGs avoid this entirely.
old_create = '''def create_placement_groups(args):
    \"\"\"Create placement groups for actor, critic, and rollout engines.\"\"\"

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f\"Creating placement group with {num_gpus} GPUs...\")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        \"actor\": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        \"rollout\": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result[\"critic\"] = result[\"actor\"] if args.use_critic else None

    return result'''

new_create = '''def create_placement_groups(args):
    \"\"\"Create placement groups for actor, critic, and rollout engines.\"\"\"

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    # When custom Ray resources are defined (time-slicing), create separate PGs
    # so that actor bundles land on trainer nodes and rollout bundles land on
    # sampler nodes.  A single combined PG suffers from IP-based reordering
    # that swaps the logical actor/rollout assignment.
    # Use RAY_RESOURCE vars (matching rayStartParams.resources) for placement,
    # not the GROUP vars (which may be per-job for the orchestrator).
    train_resource = os.environ.get(\"TIMESLICE_TRAINER_RAY_RESOURCE\") or os.environ.get(\"TIMESLICE_TRAINER_GROUP\")
    rollout_resource = os.environ.get(\"TIMESLICE_SAMPLER_RAY_RESOURCE\") or os.environ.get(\"TIMESLICE_SAMPLER_GROUP\")
    if (train_resource or rollout_resource) and rollout_offset > 0 and rollout_offset < num_gpus:
        train_num = rollout_offset
        rollout_num = num_gpus - rollout_offset
        train_extras = [{train_resource: 1}] * train_num if train_resource else None
        rollout_extras = [{rollout_resource: 1}] * rollout_num if rollout_resource else None
        logger.info(f\"Creating SEPARATE placement groups: train={train_num} GPUs ({train_resource}), rollout={rollout_num} GPUs ({rollout_resource})\")
        actor_pg = _create_placement_group(train_num, bundle_resources=train_extras)
        rollout_pg = _create_placement_group(rollout_num, bundle_resources=rollout_extras)
        result = {\"actor\": actor_pg, \"rollout\": rollout_pg}
        result[\"critic\"] = result[\"actor\"] if args.use_critic else None
        return result

    logger.info(f\"Creating placement group with {num_gpus} GPUs...\")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        \"actor\": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        \"rollout\": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result[\"critic\"] = result[\"actor\"] if args.use_critic else None

    return result'''

if old_create in src:
    src = src.replace(old_create, new_create)
    print('  patched: create_placement_groups (separate PGs for time-slicing)')
else:
    print('  SKIP (pattern not found): create_placement_groups')

p.write_text(src)
print('  placement_group.py patch complete')
"

# Head node (no GPU) also needs the timeslice client + callback package.
# Workers don't — the TimesliceCallback runs on the head (driver process).
if ! nvidia-smi &>/dev/null; then
    echo "[Setup] Head node — installing timeslice client and llm-d-timeslice-slime package..."
    # Use --no-deps to avoid upgrading protobuf to 7.x which breaks the running Ray process
    # (Ray's dashboard has protobuf 6.x loaded in memory; upgrading on disk causes version mismatch)
    pip install --no-cache-dir --no-deps "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"
    # TODO: Remove @feat/slime-package pin after PR merges to main
    pip install --no-cache-dir --no-deps "llm-d-timeslice-slime @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git@feat/slime-package#subdirectory=pkg/integrations/slime"
    pip install --no-cache-dir "grpcio>=1.81.1"

    echo "[Setup] Patching train.py/train_async.py: create placement groups BEFORE lock acquire..."
    # The orchestrator requires pods with timeslice.io/job-id labels to exist on
    # the GPU node before granting a lock.  Placement groups trigger worker pod
    # creation via the Ray autoscaler, so they must be created before the phase
    # callback (which calls Acquire).  Reorder the init sequence accordingly and
    # add a brief sleep so the orchestrator discovers the new pods.
    python3 -c "
import re, pathlib
for name in ('train.py', 'train_async.py'):
    p = pathlib.Path('/root/slime') / name
    if not p.exists():
        continue
    src = p.read_text()
    # Pattern: load_phase_callback → on_phase_begin → create_placement_groups
    # Reorder: create_placement_groups → (sleep) → load_phase_callback → on_phase_begin
    old = '''    phase_cb = load_phase_callback(args)

    # allocate the GPUs
    if phase_cb:
        phase_cb.on_phase_begin(\"init\", \"both\")

    pgs = create_placement_groups(args)'''
    new = '''    # Create placement groups FIRST so worker pods (with timeslice.io/job-id
    # labels) exist before the orchestrator Acquire call.
    pgs = create_placement_groups(args)
    import time as _time; _time.sleep(15)  # let orchestrator discover pods

    phase_cb = load_phase_callback(args)
    if phase_cb:
        phase_cb.on_phase_begin(\"init\", \"both\")'''
    if old in src:
        p.write_text(src.replace(old, new))
        print(f'  patched: {name}')
    else:
        print(f'  SKIP (pattern not found): {name}')
"

    echo "[Setup] Patching timeslice pb2 files for protobuf 6.x compatibility..."
    # The timeslice pb2 files were generated with protobuf 7.35.1 but Ray ships protobuf 6.x.
    # Remove the version validation calls so they work with the existing runtime.
    python3 -c "
import glob, re
for f in glob.glob('/usr/local/lib/python3.12/dist-packages/timeslice/**/*_pb2.py', recursive=True):
    with open(f) as fh:
        code = fh.read()
    code = re.sub(
        r'_runtime_version\.ValidateProtobufRuntimeVersion\([^)]+\)',
        'pass  # version check patched for Ray compat',
        code, flags=re.DOTALL)
    with open(f, 'w') as fh:
        fh.write(code)
    print(f'  patched: {f}')
"
fi

# Both head and workers need model weights (workers load them via Ray actors)
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
