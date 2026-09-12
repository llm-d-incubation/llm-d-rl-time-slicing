#!/usr/bin/env bash
# End-to-end test for the consolidated GPU-CR stack on a GPU node.
#
# Drives a real CUDA workload under LD_PRELOAD through the full
# checkpoint/restore surface, gating on byte-identical GPU memory after
# every restore:
#   G0  runtime buffer config honored: provenance line printed at library
#       load (before any CR signal exists) + the dump buffer's physical
#       extent matches the env-requested size
#   G1  baseline pattern verify
#   G2  destination-path selective checkpoint (-o) succeeds
#   G3  destination-path selective restore succeeds
#   G4  pattern verify after dest-path restore (byte-identical)
#   G5  buffer-path selective checkpoint/restore + verify
#   G6  full checkpoint/restore data plane + verify (stubbed toggle unless
#       E2E_FULL_TOGGLE=1 and a real cuda-checkpoint is on PATH)
#   G7  below-floor GPU_CR_SHM_MB warns at load and falls back to the
#       build default; the workload stays healthy
#   G8  zero-config ctl discovery: with NO GPU_CR_CTL_PATH anywhere, a
#       tmpfs-backed $STORE/ctl is found by both sides and a dest-path
#       C/R round-trips byte-identically (skipped, not failed, when
#       $STORE/ctl is not a tmpfs — e.g. the baseline pod layout)
#
# Required env: CR_CLIENT, WORKLOAD, VGPU_SO, STORE.
# Optional env: GPU_CR_CTL_PATH, E2E_NUM_BUFFERS, E2E_BUFFER_MB,
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
cr() { env EXPORT_FILE_PATH="$STORE" \
          ${GPU_CR_CTL_PATH:+GPU_CR_CTL_PATH="$GPU_CR_CTL_PATH"} \
          timeout "${CR_TIMEOUT:-120}" "$CR_CLIENT" "$@"; }

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
echo "workload up: pid=$WL_PID regions=$WL_REGIONS (run dir $RUN)"

# G0a runs BEFORE any cr_client call on purpose: the provenance line must
# come from library load, not from the first CR signal.
gate "G0a env config parsed at load" \
    grep -qF "[gpu-cr-config] dump buffer ${SHM_MIB} MiB (env GPU_CR_SHM_MB)" \
    "$RUN/workload.stderr"

record_store_files
cr -i -p "$WL_PID" || { echo "FATAL: init failed ($?)"; exit 1; }

gate "G0b dump buffer extent matches env" dump_extent_ok
gate "G1 baseline verify" wl_cmd verify

DUMP="$STORE/e2e-dump.bin"
rm -f "$DUMP"
gate "G2 dest-path selective ckpt" cr -c -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP"
gate "G3 dest-path selective restore" cr -r -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP"
gate "G4 verify after dest-path restore" wl_cmd verify
# G4 was the dump's last consumer; on a hugetlbfs STORE the file pins its
# pages until removed.
rm -f "$DUMP"

gate "G5a buffer-path selective ckpt" cr -c -p "$WL_PID" -s "$WL_REGIONS"
gate "G5b buffer-path selective restore" cr -r -p "$WL_PID" -s "$WL_REGIONS"
gate "G5c verify after buffer-path restore" wl_cmd verify

gate "G6a full ckpt" cr -c -p "$WL_PID"
gate "G6b full restore" cr -r -p "$WL_PID"
gate "G6c verify after full restore" wl_cmd verify

# G7: a below-floor value must warn and fall back at library load. No CR
# signal is ever sent to this workload, so a parse deferred to the signal
# path would produce neither line — and since init never runs, the
# build-default-sized buffer is never allocated (the banner alone is
# asserted; the hugepage pool stays untouched).
stop_workload
start_workload "$VGPU_SO" \
    E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
    GPU_CR_SHM_MB=10 || exit 1
gate "G7a below-floor value warns" \
    grep -qF "WARNING: GPU_CR_SHM below the 64MiB floor (10)" "$RUN/workload.stderr"
gate "G7b falls back to build default" \
    grep -qF "MiB (build default)" "$RUN/workload.stderr"
gate "G7c workload healthy after fallback" wl_cmd verify

# G8: zero-config ctl discovery. Restart the workload with GPU_CR_CTL_PATH
# scrubbed on BOTH sides; the .so and cr_client must independently find the
# tmpfs at $STORE/ctl through EXPORT_FILE_PATH alone (the consumer-side
# contract of the discovery design). Environments without a nested ctl
# tmpfs (baseline pod layout, bare runs) skip rather than fail.
CTL_CAND="$STORE/ctl"
# stat -f -c %T is GNU coreutils (fine in the CUDA/Ubuntu e2e image); on
# busybox/BSD stat this prints nothing and G8 skips — adjust if the suite
# ever moves off a GNU userland.
if [ "$(stat -f -c %T "$CTL_CAND" 2>/dev/null)" = "tmpfs" ]; then
    stop_workload
    cr8() { env -u GPU_CR_CTL_PATH EXPORT_FILE_PATH="$STORE" \
                timeout "${CR_TIMEOUT:-120}" "$CR_CLIENT" "$@"; }
    GPU_CR_CTL_PATH="" start_workload "$VGPU_SO" \
        E2E_NUM_BUFFERS="$NUM_BUFFERS" E2E_BUFFER_MB="$BUFFER_MB" \
        GPU_CR_SHM_MB="$SHM_MB" || exit 1
    gate "G8a advert discovered under \$STORE/ctl" \
        test -f "$CTL_CAND/ctl-ready-$WL_PID"
    gate "G8b init (no ctl env)" cr8 -i -p "$WL_PID"
    DUMP8="$STORE/e2e-dump-disc.bin"
    rm -f "$DUMP8"
    gate "G8c dest-path selective ckpt (no ctl env)" \
        cr8 -c -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP8"
    gate "G8d dest-path selective restore (no ctl env)" \
        cr8 -r -p "$WL_PID" -s "$WL_REGIONS" -o "$DUMP8"
    gate "G8e verify after discovery restore" wl_cmd verify
    rm -f "$DUMP8"
else
    echo "SKIP: G8 zero-config discovery ($CTL_CAND is not tmpfs-backed)"
fi

stop_workload
echo
echo "=== e2e summary: $PASS passed, $FAIL failed ==="
echo "workload stderr: $RUN/workload.stderr"
[ "$FAIL" -eq 0 ]
