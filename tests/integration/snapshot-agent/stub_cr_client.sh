#!/bin/sh
# Stub cr_client for the cross-language integration test. Simulates GPU-CR
# for a single workload: "device memory" is the file $EXPORT_FILE_PATH/device;
# -c dumps it into the -o destination (destination-path checkpoints, exactly
# the invocation shape the memory-regions backend emits), -r loads the
# destination back into the device file.
set -eu

ctl="${EXPORT_FILE_PATH:?EXPORT_FILE_PATH must be set}"
echo "$@" >>"$ctl/cr_client_calls.log"

mode=""
dest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    dest="$arg"
  fi
  case "$arg" in
    -c) mode=checkpoint ;;
    -r) mode=restore ;;
  esac
  prev="$arg"
done

if [ -z "$dest" ]; then
  echo "stub cr_client: expected -o <destination> in: $*" >&2
  exit 1
fi

case "$mode" in
  checkpoint) cp "$ctl/device" "$dest" ;;
  restore) cp "$dest" "$ctl/device" ;;
  *)
    echo "stub cr_client: no -c/-r flag in: $*" >&2
    exit 1
    ;;
esac
