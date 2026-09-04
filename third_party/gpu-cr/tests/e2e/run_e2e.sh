#!/usr/bin/env bash
# End-to-end test for the GPU-CR stack on a GPU node.
#
# Drives a real CUDA workload under LD_PRELOAD through the checkpoint/
# restore surface, gating on byte-identical GPU memory after every restore:
#   G0  runtime buffer config honored: provenance line printed at library
#       load (before any CR signal exists) + the dump buffer's physical
#       extent matches the env-requested size
#   G1  baseline pattern verify
#   G2  full checkpoint/restore data plane + verify (stubbed toggle unless
#       E2E_FULL_TOGGLE=1 and a real cuda-checkpoint is on PATH)
#   G3  below-floor GPU_CR_SHM_MB warns at load and falls back to the
#       build default; the workload stays healthy
#
# Required env: CR_CLIENT, WORKLOAD, VGPU_SO, STORE.
# Optional env: E2E_NUM_BUFFERS, E2E_BUFFER_MB,
#               GPU_CR_SHM_MB (defaults to a size fitting the buffers),
#               CR_TIMEOUT (seconds per cr_client call, default 120).
set -u
HERE=$(dirname "$(readlink -f "$0")")
. "$HERE/e2e_lib.sh"

: "${CR_CLIENT:?}" "${WORKLOAD:?}" "${VGPU_SO:?}" "${STORE:?}"
NUM_BUFFERS=${E2E_NUM_BUFFERS:-4}
BUFFER_MB=${E2E_BUFFER_MB:-64}
# Dump buffer: extents + 2MiB header + slack.
SHM_MB=${GPU_CR_SHM_MB:-$((NUM_BUFFERS * BUFFER_MB + 128))}
# What the library must actually allocate: the env value rounded up to the
# 2MiB hugepage granule. G0b compares the dump file's real extent to this.
SHM_BYTES=$(( (SHM_MB * 1048576 + 2097151) / 2097152 * 2097152 ))
SHM_MIB=$((SHM_BYTES / 1048576))

PASS=0; FAIL=0
gate() {
    local name=$1; shift
    if "$@"; then echo "PASS: $name"; PASS=$((PASS+1));
    else echo "FAIL: $name"; FAIL=$((FAIL+1)); fi
}
cr() { env EXPORT_FILE_PATH="$STORE" timeout "${CR_TIMEOUT:-120}" "$CR_CLIENT" "$@"; }

# The dump buffer is the one ckpt-<id>.data under $STORE that is not the
# -host staging file; its extent is the ftruncate the workload performed
# from the cached env config. A shared/dirty $STORE may hold dump files
# from earlier runs, so the check is bound to THIS workload: snapshot the
# store before init and judge only files that appeared after — exactly one
# must, and its extent must match.
#
# The snapshot itself fails CLOSED: an empty store is a legitimate empty
# snapshot, but an unreadable $STORE or unwritable snapshot file aborts
# the run — with a missing snapshot every pre-existing file would look
# new to dump_extent_ok, and a stale file could satisfy (or wrongly
# fail) the extent gate.
record_store_files() {
    : > "$RUN/store-pre" || { echo "FATAL: cannot write $RUN/store-pre" >&2; exit 1; }
    [ -r "$STORE" ] && [ -x "$STORE" ] || { echo "FATAL: cannot read $STORE" >&2; exit 1; }
    local f
    for f in "$STORE"/ckpt-*.data; do
        [ -e "$f" ] || continue    # unmatched glob literal
        printf '%s\n' "$f" >> "$RUN/store-pre" \
            || { echo "FATAL: cannot write $RUN/store-pre" >&2; exit 1; }
    done
}
dump_extent_ok() {
    local f sz found=""
    for f in "$STORE"/ckpt-*.data; do
        case "$f" in *-host.data) continue ;; esac
        [ -e "$f" ] || continue
        grep -qxF "$f" "$RUN/store-pre" 2>/dev/null && continue
        if [ -n "$found" ]; then
            echo "multiple new dump files: $found, $f" >&2
            return 1
        fi
        found=$f
    done
    [ -n "$found" ] || { echo "no new dump-buffer file under $STORE" >&2; return 1; }
    sz=$(stat -c %s "$found" 2>/dev/null) || { echo "stat $found failed" >&2; return 1; }
    [ "$sz" -eq "$SHM_BYTES" ] && return 0
    echo "dump file $found: $sz bytes, expected $SHM_BYTES" >&2
    return 1
}

trap 'stop_workload' EXIT
if [ "${E2E_FULL_TOGGLE:-0}" != "1" ]; then stub_cuda_checkpoint; fi

start_workload "$VGPU_SO" \
    E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
    GPU_CR_SHM_MB="$SHM_MB" || exit 1
echo "workload up: pid=$WL_PID (run dir $RUN)"

# G0a runs BEFORE any cr_client call on purpose: the provenance line must
# come from library load, not from the first CR signal.
gate "G0a env config parsed at load" \
    grep -qF "[gpu-cr-config] dump buffer ${SHM_MIB} MiB (env GPU_CR_SHM_MB)" \
    "$RUN/workload.stderr"

record_store_files
cr -i -p "$WL_PID" || { echo "FATAL: init failed ($?)"; exit 1; }

gate "G0b dump buffer extent matches env" dump_extent_ok
gate "G1 baseline verify" wl_cmd verify

gate "G2a full ckpt" cr -c -p "$WL_PID"
gate "G2b full restore" cr -r -p "$WL_PID"
gate "G2c verify after full restore" wl_cmd verify

# G3: a below-floor value must warn and fall back at library load. No CR
# signal is ever sent to this workload, so a parse deferred to the signal
# path would produce neither line — and since init never runs, the
# build-default-sized buffer is never allocated (the banner alone is
# asserted; the hugepage pool stays untouched).
stop_workload
start_workload "$VGPU_SO" \
    E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
    GPU_CR_SHM_MB=10 || exit 1
gate "G3a below-floor value warns" \
    grep -qF "WARNING: GPU_CR_SHM below the 64MiB floor (10)" "$RUN/workload.stderr"
gate "G3b falls back to build default" \
    grep -qF "MiB (build default)" "$RUN/workload.stderr"
gate "G3c workload healthy after fallback" wl_cmd verify

stop_workload
echo
echo "=== e2e summary: $PASS passed, $FAIL failed ==="
echo "workload stderr: $RUN/workload.stderr"
[ "$FAIL" -eq 0 ]
