#!/usr/bin/env bash
# Launcher for the integration test suites — the single entrypoint.
#
# Phases and their fixtures — each one a REAL install path:
#   standalone    TestStandalone against the `make standalone` artifacts
#                 (built in-cluster in the test-runner pod, run from a plain
#                 Debian base — the chart has no standalone mode)
#   k8s           TestK8s against the OFFICIAL snapshot-agent Helm chart
#   orchestrator  TestOrchestrator against BOTH official Helm charts
#                 (snapshot-agent + timeslice-orchestrator), driving real
#                 scenario workloads through the orchestrator's gRPC API
#   all           all of the above
#
# The test suite itself is written in Go (see *_test.go) and runs INSIDE the
# cluster: this script installs the chart fixture(s), deploys the test-runner
# pod, copies the repo source into it, and executes `go test` there. The Go
# harness deploys engine pods itself, one engine at a time, so a single free
# GPU is enough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K="${KUBECTL:-kubectl}"
HELM="${HELM:-helm}"

AGENT_IMAGE=""
ORCH_IMAGE=""
PROJECT=""
CLUSTER=""
ZONE=""
MODEL=""
PHASE="both"
BUILD=false
SKIP_CLEANUP=false
NEED_STANDALONE=false
NEED_SA_CHART=false
NEED_ORCH_CHART=false
# CHART_AGENT_PORT lets the chart-deployed agent bind a non-default port so
# the suite can coexist with an unrelated agent on 9001 (hostNetwork).
CHART_AGENT_PORT="${CHART_AGENT_PORT:-9002}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --agent-image)  AGENT_IMAGE="$2"; shift 2 ;;
    --orch-image)   ORCH_IMAGE="$2"; shift 2 ;;
    --build)        BUILD=true; shift ;;
    --project)      PROJECT="$2"; shift 2 ;;
    --cluster)      CLUSTER="$2"; shift 2 ;;
    --zone)         ZONE="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --phase)        PHASE="$2"; shift 2 ;;
    --skip-cleanup) SKIP_CLEANUP=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

usage() {
  echo "Usage: $0 [--agent-image IMAGE] [--orch-image IMAGE] [--build --project PROJECT] [--cluster CLUSTER --zone ZONE] [--model MODEL] [--phase standalone|k8s|both|orchestrator|all]"
  echo ""
  echo "  --cluster/--zone/--project are optional; omit them to use your current kubectl context."
  echo "  --project is required with --build (Cloud Build needs it for the image registry)."
}

case "$PHASE" in
  standalone)   RUN_PATTERN='^TestStandalone$';         NEED_STANDALONE=true ;;
  k8s)          RUN_PATTERN='^TestK8s$';                NEED_SA_CHART=true ;;
  both)         RUN_PATTERN='^(TestStandalone|TestK8s)$'
                NEED_STANDALONE=true; NEED_SA_CHART=true ;;
  orchestrator) RUN_PATTERN='^TestOrchestrator$';       NEED_SA_CHART=true; NEED_ORCH_CHART=true ;;
  all)          RUN_PATTERN='^(TestStandalone|TestK8s|TestOrchestrator)$'
                NEED_STANDALONE=true; NEED_SA_CHART=true; NEED_ORCH_CHART=true ;;
  *) echo "Unknown phase: $PHASE"; usage; exit 1 ;;
esac

# --build produces the image(s) from the working directory via Cloud Build,
# tagged with the current commit so repeated runs don't fight IfNotPresent
# caching on the node.
if [[ "$BUILD" == "true" ]]; then
  if [[ -z "$PROJECT" ]]; then
    echo "Error: --build requires --project (images are built with Cloud Build and pushed to gcr.io/PROJECT)"
    usage
    exit 1
  fi
  BUILD_TAG="integ-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo dev)"
fi

# The standalone phase needs no image: it builds the agent from source in the
# test-runner pod (`make standalone`) and runs the artifacts directly.
if [[ "$NEED_SA_CHART" == "true" && -z "$AGENT_IMAGE" && "$BUILD" != "true" ]]; then
  echo "Error: --agent-image (or --build) is required for the k8s/orchestrator phase (the snapshot-agent chart installs it)"
  usage
  exit 1
fi
if [[ "$NEED_ORCH_CHART" == "true" && -z "$ORCH_IMAGE" && "$BUILD" != "true" ]]; then
  echo "Error: --orch-image (or --build) is required for the orchestrator phase (the orchestrator chart installs it)"
  usage
  exit 1
fi
if [[ "$NEED_SA_CHART" == "true" && -z "${TEST_NODE:-}" && "$NEED_ORCH_CHART" != "true" ]]; then
  echo "Error: TEST_NODE must be set for the k8s phase: the chart is pinned to" \
       "one node so the suite stays off other workloads on shared clusters"
  exit 1
fi
if [[ "$NEED_ORCH_CHART" == "true" ]]; then
  if [[ -z "${TEST_NODE_SAMPLERS:-}" || -z "${TEST_NODE_TRAINERS:-}" ]]; then
    echo "Error: TEST_NODE_SAMPLERS and TEST_NODE_TRAINERS must be set for the orchestrator phase" \
         "(one GPU node per group — the canonical deployment topology)"
    exit 1
  fi
fi

log() { echo "$(date +%H:%M:%S) $*"; }

cleanup() {
  [[ "$SKIP_CLEANUP" == "true" ]] && return 0
  log "Cleaning up..."
  $K delete pods -l test-suite=snapshot-agent-integration --force --grace-period=0 2>/dev/null || true
  $K delete -f "${SCRIPT_DIR}/runner.yaml" --force --grace-period=0 2>/dev/null || true
  if [[ "$NEED_ORCH_CHART" == "true" ]]; then
    $HELM uninstall orch-chart-test -n timeslice-system 2>/dev/null || true
    # Remove integration group labels (only the ones we added).
    if [[ -n "${TEST_NODE_SAMPLERS:-}" ]]; then
      $K label node "$TEST_NODE_SAMPLERS" "group.timeslice.io/samplers-" 2>/dev/null || true
    fi
    if [[ -n "${TEST_NODE_TRAINERS:-}" ]]; then
      $K label node "$TEST_NODE_TRAINERS" "group.timeslice.io/trainers-" 2>/dev/null || true
    fi
  fi
  if [[ "$NEED_SA_CHART" == "true" ]]; then
    $HELM uninstall sa-chart-test -n timeslice-system 2>/dev/null || true
  fi
}

# wait_chart_pods_ready LABEL_SELECTOR [NODE]: fail fast on fixture problems —
# a chart pod that never becomes Ready is a deployment failure and must be
# attributed to the chart install, not to the tests.
wait_chart_pods_ready() {
  local selector="$1" node="${2:-}" field=""
  [[ -n "$node" ]] && field="--field-selector=spec.nodeName=${node}"
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    local ready
    # shellcheck disable=SC2086
    ready=$($K get pods -n timeslice-system -l "$selector" $field \
      -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
    [[ "$ready" == *"True"* ]] && return 0
    sleep 3
  done
  return 1
}

chart_failure() { # NAME SELECTOR
  echo "ERROR: $1 chart install failed: its pod did not become Ready within 300s." >&2
  echo "This is a chart/deployment failure, not a test failure. Pod state:" >&2
  $K get pods -n timeslice-system -l "$2" -o wide >&2 || true
  $K logs -n timeslice-system -l "$2" --tail=20 >&2 || true
  cleanup
  exit 1
}

if [[ -n "$CLUSTER" ]]; then
  ZONE_FLAG=""
  [[ -n "$ZONE" ]] && ZONE_FLAG="--zone ${ZONE}"
  PROJECT_FLAG=""
  [[ -n "$PROJECT" ]] && PROJECT_FLAG="--project ${PROJECT}"
  log "Getting credentials for cluster ${CLUSTER}..."
  # shellcheck disable=SC2086
  gcloud container clusters get-credentials "$CLUSTER" $ZONE_FLAG $PROJECT_FLAG 2>&1
fi

if [[ "$BUILD" == "true" && "$NEED_SA_CHART" == "true" && -z "$AGENT_IMAGE" ]]; then
  AGENT_IMAGE="gcr.io/${PROJECT}/snapshot-agent:${BUILD_TAG}"
  log "Building ${AGENT_IMAGE} from the working directory (Cloud Build)..."
  gcloud builds submit --project "$PROJECT" --config="${REPO_ROOT}/cloudbuild-image.yaml" \
    --substitutions="_IMAGE=${AGENT_IMAGE},_DOCKERFILE=docker/snapshot-agent/Dockerfile" \
    "$REPO_ROOT"
fi

if [[ "$BUILD" == "true" && "$NEED_ORCH_CHART" == "true" && -z "$ORCH_IMAGE" ]]; then
  ORCH_IMAGE="gcr.io/${PROJECT}/timesliceorchestrator:${BUILD_TAG}"
  log "Building ${ORCH_IMAGE} from the working directory (Cloud Build)..."
  gcloud builds submit --project "$PROJECT" --config="${REPO_ROOT}/cloudbuild-image.yaml" \
    --substitutions="_IMAGE=${ORCH_IMAGE},_DOCKERFILE=docker/timesliceorchestrator/Dockerfile" \
    "$REPO_ROOT"
fi

if [[ "$NEED_SA_CHART" == "true" ]]; then
  # The chart templates pin their namespace to timeslice-system.
  $K create namespace timeslice-system --dry-run=client -o yaml | $K apply -f -
  SA_HELM_ARGS=(
    -n timeslice-system
    --set fullnameOverride=sa-chart-test
    --set image.repository="${AGENT_IMAGE%:*}"
    --set image.tag="${AGENT_IMAGE##*:}"
    --set image.pullPolicy=IfNotPresent
    --set port="${CHART_AGENT_PORT}"
  )
  if [[ "$NEED_ORCH_CHART" == "true" ]]; then
    # Orchestrator phase: SA DaemonSet must land on both group nodes (no nodeSelector pin).
    log "Installing the official snapshot-agent chart (both group nodes, port ${CHART_AGENT_PORT})..."
    $HELM upgrade --install sa-chart-test "${REPO_ROOT}/deploy/snapshot-agent" "${SA_HELM_ARGS[@]}"
    SA_SELECTOR="app.kubernetes.io/name=snapshot-agent,app.kubernetes.io/instance=sa-chart-test"
    for NODE in "$TEST_NODE_SAMPLERS" "$TEST_NODE_TRAINERS"; do
      log "Waiting for the chart's DaemonSet pod to become Ready on ${NODE}..."
      wait_chart_pods_ready "$SA_SELECTOR" "$NODE" || chart_failure "snapshot-agent" "$SA_SELECTOR"
    done
  else
    # k8s phase: single-node pin.
    log "Installing the official snapshot-agent chart (node ${TEST_NODE}, port ${CHART_AGENT_PORT})..."
    $HELM upgrade --install sa-chart-test "${REPO_ROOT}/deploy/snapshot-agent" "${SA_HELM_ARGS[@]}" \
      --set nodeSelector."kubernetes\.io/hostname"="${TEST_NODE}"
    SA_SELECTOR="app.kubernetes.io/name=snapshot-agent,app.kubernetes.io/instance=sa-chart-test"
    log "Waiting for the chart's DaemonSet pod to become Ready on ${TEST_NODE}..."
    wait_chart_pods_ready "$SA_SELECTOR" "$TEST_NODE" || chart_failure "snapshot-agent" "$SA_SELECTOR"
  fi
fi

if [[ "$NEED_ORCH_CHART" == "true" ]]; then
  $K create namespace timeslice-system --dry-run=client -o yaml | $K apply -f -
  log "Installing the official orchestrator chart (agent port=${CHART_AGENT_PORT})..."
  $HELM upgrade --install orch-chart-test "${REPO_ROOT}/deploy/timesliceorchestrator" \
    -n timeslice-system \
    --set fullnameOverride=orch-chart-test \
    --set image.repository="${ORCH_IMAGE%:*}" \
    --set image.tag="${ORCH_IMAGE##*:}" \
    --set image.pullPolicy=IfNotPresent \
    --set snapshotAgentPort="${CHART_AGENT_PORT}"
  ORCH_SELECTOR="app.kubernetes.io/name=timesliceorchestrator,app.kubernetes.io/instance=orch-chart-test"
  log "Waiting for the orchestrator Deployment pod to become Ready..."
  wait_chart_pods_ready "$ORCH_SELECTOR" || chart_failure "orchestrator" "$ORCH_SELECTOR"
fi

log "Deploying test runner..."
$K apply -f "${SCRIPT_DIR}/runner.yaml"
$K wait --for=condition=Ready pod/test-runner --timeout=300s

log "Copying repo source into test runner..."
tar -czf - -C "$REPO_ROOT" --exclude .git . \
  | $K exec -i test-runner -- sh -c "rm -rf /workspace && mkdir -p /workspace && tar -xzf - -C /workspace"

log "Installing the Python client in the test runner (tests drive the agent through it)..."
$K exec test-runner -- sh -c "apt-get update -qq > /dev/null && apt-get install -y -qq python3 python3-pip > /dev/null && pip3 install --break-system-packages -q /workspace/pkg/client/python"

if [[ "$NEED_STANDALONE" == "true" ]]; then
  log "Building the standalone artifacts in the test runner (make standalone)..."
  $K exec test-runner -- sh -c "cd /workspace && make standalone"
fi

log "Running Go test suite in-cluster (this deploys agent + engine pods)..."
SA_CHART_DEPLOYED=""
[[ "$NEED_SA_CHART" == "true" ]] && SA_CHART_DEPLOYED=1
ORCH_CHART_DEPLOYED=""
[[ "$NEED_ORCH_CHART" == "true" ]] && ORCH_CHART_DEPLOYED=1
EXIT=0
$K exec test-runner -- env "MODEL=${MODEL}" "TEST_NODE=${TEST_NODE:-}" \
  "TEST_NODE_SAMPLERS=${TEST_NODE_SAMPLERS:-}" \
  "TEST_NODE_TRAINERS=${TEST_NODE_TRAINERS:-}" \
  "SA_CHART_DEPLOYED=${SA_CHART_DEPLOYED}" \
  "ORCH_CHART_DEPLOYED=${ORCH_CHART_DEPLOYED}" \
  "CHART_AGENT_PORT=${CHART_AGENT_PORT}" \
  sh -c "cd /workspace && go test -tags=integration -count=1 -v -timeout 40m -run '${RUN_PATTERN}' ./tests/integration/..." \
  || EXIT=$?

cleanup

exit "$EXIT"
