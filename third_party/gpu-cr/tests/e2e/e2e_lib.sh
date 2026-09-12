# Shared driver for the GPU e2e and perf-regression suites. Sourced by
# run_e2e.sh and perf_regression.sh.
#
# Required env: CR_CLIENT, WORKLOAD, STORE (dump/staging dir).
# Optional env: GPU_CR_CTL_PATH (tmpfs ctl dir), RUN (scratch dir).
# GPU_CR_CTL_PATH and GPU_CR_CUDA_CHECKPOINT are honored only by
# libraries/clients that support them; inert for the e9bbb52 baseline.

RUN=${RUN:-$(mktemp -d /tmp/gpu-cr-e2e.XXXXXX)}
SPEC="$RUN/spec"
CMD="$RUN/cmd"
RESP="$RUN/resp"
WL_PID=""
WL_REGIONS=""
SEQ=0

# start_workload <vgpu_so> [ENV=VAL ...]
start_workload() {
    local so=$1
    shift
    rm -f "$SPEC" "$CMD" "$RESP"
    env LD_PRELOAD="$so" \
        GPU_VENDOR="${GPU_VENDOR:-NVIDIA}" \
        EXPORT_FILE_PATH="$STORE" \
        ${GPU_CR_CTL_PATH:+GPU_CR_CTL_PATH="$GPU_CR_CTL_PATH"} \
        E2E_SPEC_FILE="$SPEC" E2E_CMD_FILE="$CMD" E2E_RESP_FILE="$RESP" \
        "$@" "$WORKLOAD" 2>> "$RUN/workload.stderr" &
    WL_PID=$!
    # CUDA init + hooked allocations can take a while on first touch.
    for _ in $(seq 1 1200); do
        [ -s "$SPEC" ] && break
        if ! kill -0 "$WL_PID" 2>/dev/null; then
            echo "FATAL: workload died during startup; tail of stderr:" >&2
            tail -30 "$RUN/workload.stderr" >&2
            return 1
        fi
        sleep 0.1
    done
    [ -s "$SPEC" ] || { echo "FATAL: workload spec never appeared" >&2; return 1; }
    WL_PID=$(sed -n 1p "$SPEC")
    WL_REGIONS=$(sed -n 2p "$SPEC")
    return 0
}

# wl_cmd <verify|exit> — returns 0 iff the workload answered "ok".
wl_cmd() {
    SEQ=$((SEQ + 1))
    printf '%s %s\n' "$SEQ" "$1" > "$CMD.tmp" && mv "$CMD.tmp" "$CMD"
    for _ in $(seq 1 3000); do
        local line
        line=$(grep "^$SEQ " "$RESP" 2>/dev/null | head -1)
        if [ -n "$line" ]; then
            echo "$line"
            case "$line" in "$SEQ ok"*) return 0 ;; *) return 1 ;; esac
        fi
        sleep 0.1
    done
    echo "$SEQ timeout"
    return 1
}

stop_workload() {
    [ -n "$WL_PID" ] || return 0
    wl_cmd exit > /dev/null 2>&1
    kill "$WL_PID" 2>/dev/null
    wait "$WL_PID" 2>/dev/null
    # hugetlbfs pages are freed only when the backing files go away: purge
    # the per-run buffers or back-to-back runs exhaust the pool. NOTE:
    # clears every ckpt-*/control*/pid_map_* under $STORE — point STORE
    # at a dedicated volume, not a shared store. $STORE/ctl is the
    # zero-config discovery dir (harmless no-op when absent).
    rm -f "$STORE"/ckpt-* "$STORE"/control* "$STORE"/pid_map_* \
          "$STORE"/ctl/control* "$STORE"/ctl/pid_map_* \
          "$STORE"/ctl/ctl-ready-* 2>/dev/null
    WL_PID=""
    return 0
}

# stub_cuda_checkpoint — puts a no-op cuda-checkpoint on PATH so full-restore
# data-plane runs can be driven without freezing the process. The perf suite
# measures the .so data plane only; the freeze/thaw half is NVIDIA's binary,
# identical across GPU-CR builds.
stub_cuda_checkpoint() {
    local dir="$RUN/stub-bin"
    mkdir -p "$dir"
    printf '#!/bin/sh\nexit 0\n' > "$dir/cuda-checkpoint"
    chmod +x "$dir/cuda-checkpoint"
    export PATH="$dir:$PATH"
    export GPU_CR_CUDA_CHECKPOINT="$dir/cuda-checkpoint"
}

# extract_ms <stderr-file> <prefix> — prints one time-in-ms per matching
# ".. size: X GB, time: Y ms, .." line (prefix: "ckpt" | "restore" |
# "selective ckpt" | "selective restore").
extract_ms() {
    sed -n "s/^$2 size: .* time: \([0-9][0-9]*\) ms.*/\1/p" "$1"
}

median() {
    sort -n | awk '{a[NR]=$1} END {if (NR==0) print -1; else print a[int((NR+1)/2)]}'
}
