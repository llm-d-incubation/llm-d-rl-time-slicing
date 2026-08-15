#!/bin/bash
set -ex

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 1. Time-slicing mode: "sync" (default) or "async"
MODE="${SLIME_MODE:-sync}"
DRIVER="train.py"
export TIMESLICE_JOB_ID="${JOB_NAME}"
export TIMESLICE_TRAINER_GROUP="trainers"
# Sampler group: shared by default (sync), per-job for async to avoid sampler contention.
# Can be overridden by TIMESLICE_SAMPLER_GROUP_OVERRIDE env var.
export TIMESLICE_SAMPLER_GROUP="${TIMESLICE_SAMPLER_GROUP_OVERRIDE:-samplers}"
# Ray resource names for placement group pinning (match rayStartParams.resources)
export TIMESLICE_TRAINER_RAY_RESOURCE="trainers"
export TIMESLICE_SAMPLER_RAY_RESOURCE="samplers"

if [ "$MODE" = "async" ]; then
    DRIVER="train_async.py"
    export TIMESLICE_SAMPLER_GROUP="${TIMESLICE_SAMPLER_GROUP_OVERRIDE:-samplers-${JOB_NAME}}"
fi

if [ "$MODE" = "sync" ]; then
    DRIVER="train.py"
fi

MODEL_NAME="Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_DIR="/tmp/${MODEL_NAME}"
LOCAL_PROMPT_DATA="/tmp/dapo-math-17k/dapo-math-17k.jsonl"
SAVE_DIR="${SAVE_DIR:-/tmp/slime_checkpoints}"

# 2. Qwen2.5-0.5B Megatron Arguments
MODEL_ARGS=(
    --hf-checkpoint ${LOCAL_MODEL_DIR}
    --ref-load ${LOCAL_MODEL_DIR}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen2.5-0.5B-Instruct
    --megatron-to-hf-mode bridge
    --save ${SAVE_DIR}
    --save-interval 5
    --swiglu
    --num-layers 24
    --hidden-size 896
    --ffn-hidden-size 4864
    --num-attention-heads 14
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization "RMSNorm"
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --group-query-attention
    --num-query-groups 2
    --vocab-size 151936
)

# 3. Resource Settings (Vanilla integer GPUs for Ray)
RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    --rollout-num-gpus 1
    --rollout-num-gpus-per-engine 1
)

# 4. Workload Settings (GBS=8, rollout-batch-size=1)
ROLLOUT_ARGS=(
    --prompt-data ${LOCAL_PROMPT_DATA}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 2
    --rollout-batch-size 36
    --n-samples-per-prompt 8
    --max-training-samples 288
    --num-steps-per-rollout 1
    --global-batch-size 288
    --micro-batch-size 4
    --rollout-max-response-len 1024
    --rollout-temperature 0.8
)

# 5. GRPO, Perf and SGLang Configuration
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 1024
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --eps-clip 0.2
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
)
SGLANG_ARGS=(
    --sglang-mem-fraction-static 0.4
    --sglang-disable-cuda-graph
    --sglang-disable-piecewise-cuda-graph
)

# 6. Time-slicing via PhaseCallback — zero Slime code changes
TIMESLICE_ARGS=(
    --phase-callback-path timeslice_slime.callback.TimesliceCallback
)

python3 /root/slime/${DRIVER} \
    "${RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TIMESLICE_ARGS[@]}"
