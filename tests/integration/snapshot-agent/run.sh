#!/usr/bin/env bash
# Launcher for the snapshot-agent integration tests.
#
# The test suite itself is written in Go (see *_test.go) and runs INSIDE the
# cluster: this script deploys the test-runner pod, copies the repo source
# into it, and executes `go test` there. The Go harness deploys the agent and
# engine pods itself, one engine at a time, so a single free GPU is enough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
K="${KUBECTL:-kubectl}"

IMAGE=""
PROJECT=""
CLUSTER=""
ZONE=""
MODEL=""
PHASE="both"
SKIP_CLEANUP=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --image)        IMAGE="$2"; shift 2 ;;
    --project)      PROJECT="$2"; shift 2 ;;
    --cluster)      CLUSTER="$2"; shift 2 ;;
    --zone)         ZONE="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --phase)        PHASE="$2"; shift 2 ;;
    --skip-cleanup) SKIP_CLEANUP=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  echo "Error: --image is required"
  echo "Usage: $0 --image IMAGE [--project PROJECT --cluster CLUSTER --zone ZONE] [--model MODEL] [--phase standalone|k8s|both]"
  exit 1
fi

case "$PHASE" in
  standalone) RUN_PATTERN='^TestStandalone$' ;;
  k8s)        RUN_PATTERN='^TestK8s$' ;;
  both)       RUN_PATTERN='^(TestStandalone|TestK8s)$' ;;
  *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

log() { echo "$(date +%H:%M:%S) $*"; }

if [[ -n "$CLUSTER" ]]; then
  ZONE_FLAG=""
  [[ -n "$ZONE" ]] && ZONE_FLAG="--zone ${ZONE}"
  PROJECT_FLAG=""
  [[ -n "$PROJECT" ]] && PROJECT_FLAG="--project ${PROJECT}"
  log "Getting credentials for cluster ${CLUSTER}..."
  # shellcheck disable=SC2086
  gcloud container clusters get-credentials "$CLUSTER" $ZONE_FLAG $PROJECT_FLAG 2>&1
fi

log "Deploying test runner..."
$K apply -f "${SCRIPT_DIR}/runner.yaml"
$K wait --for=condition=Ready pod/test-runner --timeout=300s

log "Copying repo source into test runner..."
tar -czf - -C "$REPO_ROOT" --exclude .git . \
  | $K exec -i test-runner -- sh -c "rm -rf /workspace && mkdir -p /workspace && tar -xzf - -C /workspace"

log "Installing the Python client in the test runner (tests drive the agent through it)..."
$K exec test-runner -- sh -c "apt-get update -qq > /dev/null && apt-get install -y -qq python3 python3-pip > /dev/null && pip3 install --break-system-packages -q /workspace/pkg/client/python"

log "Running Go test suite in-cluster (this deploys agent + engine pods)..."
EXIT=0
$K exec test-runner -- env "AGENT_IMAGE=${IMAGE}" "MODEL=${MODEL}" "TEST_NODE=${TEST_NODE:-}" \
  sh -c "cd /workspace && go test -tags=integration -count=1 -v -timeout 40m -run '${RUN_PATTERN}' ./tests/integration/snapshot-agent/" \
  || EXIT=$?

if [[ "$SKIP_CLEANUP" == "false" ]]; then
  log "Cleaning up..."
  $K delete pods -l test-suite=snapshot-agent-integration --force --grace-period=0 2>/dev/null || true
  $K delete -f "${SCRIPT_DIR}/runner.yaml" --force --grace-period=0 2>/dev/null || true
fi

exit "$EXIT"
