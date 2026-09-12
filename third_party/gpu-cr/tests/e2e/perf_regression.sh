#!/usr/bin/env bash
# Performance-regression gate: the consolidated build must not regress the
# full checkpoint/restore data plane that upstream GPU-CR (e9bbb52) delivers.
#
# Runs the SAME pattern workload under the baseline .so and the candidate
# .so on the same node, measures the .so-reported full ckpt()/restore data-
# plane times over N iterations, and fails if the candidate's median is more
# than --threshold-pct slower on either direction. The cuda-checkpoint
# freeze/thaw half is NVIDIA's binary — identical for both builds — and is
# stubbed out so only GPU-CR code is measured.
#
# The candidate's selective-path timings are also recorded (informational:
# no baseline exists for them in e9bbb52).
#
# Usage:
#   perf_regression.sh --baseline-so SO --candidate-so SO \
#       [--baseline-client BIN] [--candidate-client BIN] \
#       [--iters 5] [--threshold-pct 15]
# Required env: CR_CLIENT (default client), WORKLOAD, STORE.
# Optional env: CR_TIMEOUT (seconds per cr_client call, default 120).
# Sizing env:  PERF_NUM_BUFFERS (default 8), PERF_BUFFER_MB (default 512).
set -u
HERE=$(dirname "$(readlink -f "$0")")
. "$HERE/e2e_lib.sh"

BASELINE_SO="" CANDIDATE_SO=""
BASELINE_CLIENT="${CR_CLIENT:-}" CANDIDATE_CLIENT="${CR_CLIENT:-}"
ITERS=5 THRESHOLD_PCT=15
while [ $# -gt 0 ]; do
    case $1 in
        --baseline-so) BASELINE_SO=$2; shift 2 ;;
        --candidate-so) CANDIDATE_SO=$2; shift 2 ;;
        --baseline-client) BASELINE_CLIENT=$2; shift 2 ;;
        --candidate-client) CANDIDATE_CLIENT=$2; shift 2 ;;
        --iters) ITERS=$2; shift 2 ;;
        --threshold-pct) THRESHOLD_PCT=$2; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
: "${BASELINE_SO:?--baseline-so required}" "${CANDIDATE_SO:?--candidate-so required}"
: "${WORKLOAD:?}" "${STORE:?}"
NUM_BUFFERS=${PERF_NUM_BUFFERS:-8}
BUFFER_MB=${PERF_BUFFER_MB:-512}
SHM_MB=$((NUM_BUFFERS * BUFFER_MB + 128))

trap 'stop_workload' EXIT
stub_cuda_checkpoint

# measure <label> <so> <client> [extra workload env...]
# Prints "<label> <ckpt_median_ms> <restore_median_ms>".
measure() {
    local label=$1 so=$2 client=$3
    shift 3
    : > "$RUN/workload.stderr"
    # The baseline .so has no env-based sizing: its dump buffer is the
    # compile-time SHM_SIZE_GB, which the caller must have built large
    # enough. GPU_CR_SHM_MB is passed for candidates that support env
    # sizing; libraries that predate it (the e9bbb52 baseline included)
    # ignore it.
    start_workload "$so" \
        E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
        GPU_CR_SHM_MB="$SHM_MB" "$@" || { stop_workload; return 1; }
    local crc="env EXPORT_FILE_PATH=$STORE"
    $crc timeout "${CR_TIMEOUT:-120}" "$client" -i -p "$WL_PID" > /dev/null 2>&1 \
        || { echo "$label: init failed" >&2; stop_workload; return 1; }

    for i in $(seq 1 "$ITERS"); do
        $crc timeout "${CR_TIMEOUT:-120}" "$client" -c -p "$WL_PID" > /dev/null 2>&1 \
            || { echo "$label: full ckpt iter $i failed" >&2; stop_workload; return 1; }
        $crc timeout "${CR_TIMEOUT:-120}" "$client" -r -p "$WL_PID" > /dev/null 2>&1 \
            || { echo "$label: full restore iter $i failed" >&2; stop_workload; return 1; }
        wl_cmd verify > /dev/null \
            || { echo "$label: verify failed after iter $i" >&2; stop_workload; return 1; }
    done
    local ckpt_ms restore_ms
    ckpt_ms=$(extract_ms "$RUN/workload.stderr" "ckpt" | median)
    restore_ms=$(extract_ms "$RUN/workload.stderr" "restore" | median)
    stop_workload
    echo "$label $ckpt_ms $restore_ms"
}

echo "perf regression: ${NUM_BUFFERS}x${BUFFER_MB}MiB buffers, $ITERS iters, threshold ${THRESHOLD_PCT}%"

# Perf tests run in the legacy (no-ctl) layout: the e9bbb52 baseline
# predates the ctl-path control plane and both variants must see identical
# plumbing. Unsetting the env is not enough on the nested-ctl pod layout —
# the candidate would zero-config-discover $STORE/ctl — so dumps move to a
# ctl-free subdirectory of the store (same hugetlbfs, no ctl/ inside).
unset GPU_CR_CTL_PATH
STORE="$STORE/perf-legacy"
mkdir -p "$STORE"

BASE=$(measure baseline "$BASELINE_SO" "$BASELINE_CLIENT") || exit 1
CAND=$(measure candidate "$CANDIDATE_SO" "$CANDIDATE_CLIENT") || exit 1
read -r _ BASE_CKPT BASE_RESTORE <<EOF
$BASE
EOF
read -r _ CAND_CKPT CAND_RESTORE <<EOF
$CAND
EOF

# Informational: candidate selective path (no e9bbb52 baseline exists).
# Forward hook: needs a cr_client with selective support (-s); with a
# baseline-only client the ops fail fast and both medians report -1.
: > "$RUN/workload.stderr"
if start_workload "$CANDIDATE_SO" \
        E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
        GPU_CR_SHM_MB="$SHM_MB"; then
    env EXPORT_FILE_PATH="$STORE" timeout "${CR_TIMEOUT:-120}" \
        "$CANDIDATE_CLIENT" -i -p "$WL_PID" > /dev/null 2>&1
    for i in $(seq 1 "$ITERS"); do
        env EXPORT_FILE_PATH="$STORE" timeout "${CR_TIMEOUT:-120}" \
            "$CANDIDATE_CLIENT" -c -p "$WL_PID" -s "$WL_REGIONS" > /dev/null 2>&1
        env EXPORT_FILE_PATH="$STORE" timeout "${CR_TIMEOUT:-120}" \
            "$CANDIDATE_CLIENT" -r -p "$WL_PID" -s "$WL_REGIONS" > /dev/null 2>&1
    done
    SEL_CKPT=$(extract_ms "$RUN/workload.stderr" "selective ckpt" | median)
    SEL_RESTORE=$(extract_ms "$RUN/workload.stderr" "selective restore" | median)
    stop_workload
else
    SEL_CKPT=-1; SEL_RESTORE=-1
fi

fail=0
judge() { # <name> <base_ms> <cand_ms>
    local limit=$(( $2 * (100 + THRESHOLD_PCT) / 100 ))
    if [ "$2" -le 0 ] || [ "$3" -le 0 ]; then
        echo "ERROR: $1 has no measurements (baseline=$2 candidate=$3)"; fail=1
    elif [ "$3" -gt "$limit" ]; then
        echo "REGRESSION: $1 baseline=${2}ms candidate=${3}ms (> ${limit}ms limit)"; fail=1
    else
        echo "OK: $1 baseline=${2}ms candidate=${3}ms (limit ${limit}ms)"
    fi
}
echo
judge "full ckpt data plane"    "$BASE_CKPT"    "$CAND_CKPT"
judge "full restore data plane" "$BASE_RESTORE" "$CAND_RESTORE"
echo "INFO: candidate selective ckpt median ${SEL_CKPT}ms, selective restore median ${SEL_RESTORE}ms"

cat > "$RUN/perf-results.json" <<EOF
{
  "buffers": {"count": $NUM_BUFFERS, "mb_each": $BUFFER_MB},
  "iters": $ITERS,
  "threshold_pct": $THRESHOLD_PCT,
  "full_ckpt_ms":    {"baseline": $BASE_CKPT,    "candidate": $CAND_CKPT},
  "full_restore_ms": {"baseline": $BASE_RESTORE, "candidate": $CAND_RESTORE},
  "selective_ms":    {"ckpt": $SEL_CKPT, "restore": $SEL_RESTORE},
  "pass": $((1 - fail))
}
EOF
echo "results: $RUN/perf-results.json"
exit "$fail"
