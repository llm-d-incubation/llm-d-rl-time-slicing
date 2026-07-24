# Integration Tests

End-to-end tests for the snapshot-agent and the accelerator-orchestrator on real hardware, with `run.sh` as the single entrypoint. The standalone and k8s phases exercise the agent's Makefile and Helm chart deployment paths; the orchestrator phase exercises the orchestrator's Helm chart.

The orchestrator phase runs the shared E2E scenarios from `pkg/accelerator-orchestrator/scenarios` — the same code `rlts test orchestrator` runs for operator-side smoke testing, and that the scenario package's own in-process tests run against fakes in CI — so there is one implementation of the orchestration scenarios. The scenarios use fake RL jobs with pause pods, so the chart is pointed at an agent port with no listener (`ORCH_AGENT_PORT`): a real agent correctly reports jobs without GPU activity as idle-not-loaded, which would stall lock handoffs — agent-driven C/R scenarios need real GPU workloads and are future work.

The test suite is written in Go and runs inside the cluster: `run.sh` deploys a test-runner pod, copies the repo source into it, and executes `go test` there. The Go harness deploys the snapshot-agent and inference engine pods itself — one engine at a time, so a single free GPU is enough.

All snapshot/restore calls go through the **Python client** (`timeslice.snapshot_agent`, invoked via `agentctl.py`), so the entire client layer is covered.

- `run.sh` — the single entrypoint (build images, install chart fixtures, deploy runner, copy source, build `make standalone`, install the Python client, `go test`, cleanup)
- `runner.yaml` — test-runner pod + RBAC
- `harness/` — shared framework: in-cluster client, node selection, pod lifecycle, exec/HTTP/VRAM helpers
- `snapshot-agent/` — the agent suite: `standalone_test.go` / `k8s_test.go`, plus the agent specifics (`harness.go` agent deployment, `engines.go` engine specs, `agentctl.py` — a thin CLI over the Python client that builds `BackendConfig` protos from primitive flags)
- `orchestrator/` — the orchestrator suite: `orchestrator_test.go`, a thin wrapper over the shared scenario package

## Adding a test

Snapshot-agent tests: add a `t.Run(...)` inside the engine group that provides the pods it needs, using the harness helpers:

```go
h.WithEngine(t, VLLM, func(t *testing.T, e *Engine) {
    t.Run("MyNewTest", func(t *testing.T) {
        before := h.Inference(t, e)                            // deterministic completion
        h.SnapshotOK(t, "my-job", vllmSleepConfig(e.Endpoint(), 1))
        vram := h.VRAMMiB(t, e)                                // GPU memory in use
        h.RestoreOK(t, "my-job", vllmWakeConfig(e.Endpoint()))
        RequireFreedAndCorrect(t, vram, before, h.Inference(t, e))
    })
})
```

A new engine is an `EngineSpec` in `snapshot-agent/engines.go`.

Orchestrator tests: add a scenario function to `pkg/accelerator-orchestrator/scenarios` (that also makes it available to `rlts test orchestrator` and the package's in-process tests), then wire a thin `t.Run` in `orchestrator/orchestrator_test.go` that calls it with the gRPC client and clientset.

## Prerequisites

- A GKE cluster with at least 1 free GPU
- `gcloud` and `kubectl` on the machine running the tests
  (Go and everything else run inside the cluster)
- For the k8s phase: a snapshot-agent image built from the official
  Dockerfile (`--build` does this for you; `cloudbuild-image.yaml` defaults
  to `docker/snapshot-agent/Dockerfile`). The standalone phase needs no
  image — it builds the agent from source in the test runner.
- Cloud Build API enabled (`gcloud services enable cloudbuild.googleapis.com`)
  and permission to push to the project's registry; GKE nodes in the same
  project can pull from `gcr.io/<project>` by default.

## Testing your changes

Everything runs from your working directory — uncommitted changes included — so no commit or merge is needed at any layer:

```bash
TEST_NODE=<gpu-node> ./tests/integration/run.sh \
  --build --project <gcp-project>
```

`--build` has Cloud Build produce the images the selected phases need from the working directory (tagged `integ-<commit>` so repeated runs don't collide with node image caches) and runs all phases against them. `run.sh` then copies the local workspace into the cluster, so the standalone phase's `make standalone` build, the Helm charts (installed from local `deploy/`), the Python client, and the test code all come from the workspace too.

Alternative (pre-built images — any registry the cluster can pull from):

```bash
gcloud builds submit --config=cloudbuild-image.yaml \
  --substitutions=_IMAGE=gcr.io/<project>/snapshot-agent:dev .
gcloud builds submit --config=cloudbuild-image.yaml \
  --substitutions=_IMAGE=gcr.io/<project>/acceleratororchestrator:dev,_DOCKERFILE=docker/acceleratororchestrator/Dockerfile .

TEST_NODE=<gpu-node> ./tests/integration/run.sh \
  --agent-image gcr.io/<project>/snapshot-agent:dev \
  --orch-image gcr.io/<project>/acceleratororchestrator:dev
```

## Options

```text
--agent-image IMAGE  Snapshot-agent image the k8s phase installs via the
                     official chart (required for k8s/both/all unless
                     --build)
--orch-image IMAGE   Accelerator-orchestrator image the orchestrator phase
                     installs via the official chart (required for
                     orchestrator/all unless --build)
--build              Build the needed image(s) from the working directory
                     via Cloud Build (requires --project); explicit
                     --agent-image/--orch-image override
--project PROJECT    GCP project (image pushes with --build; also used by
                     gcloud get-credentials with --cluster)
--cluster CLUSTER    GKE cluster name
--zone ZONE          GKE cluster zone
--model MODEL        Model to load (default: Qwen/Qwen2.5-0.5B)
--phase PHASE        "standalone", "k8s", "orchestrator", "both"
                     (standalone + k8s), or "all" (default)
--skip-cleanup       Leave the test-runner pod and chart fixtures running
                     for debugging
```

Environment:

- `TEST_NODE=<node-name>` — required for the k8s and orchestrator phases
  (the fixtures pin to this node); for the standalone phase it pins the
  suite instead of the default pick (first node with a free GPU by
  requests). Use the pin when the cluster runs workloads that occupy GPUs
  without requesting them (time-slicing experiments), which the default
  pick cannot see.
- `CHART_AGENT_PORT=<port>` — port for the chart-deployed agent (default
  9001), so the suite can coexist with an unrelated agent on the default
  port (the chart runs on hostNetwork).
- `ORCH_AGENT_PORT=<port>` — agent port the chart-deployed orchestrator
  dials (default 9003, deliberately unused: see the orchestrator phase
  notes above).

## Exit code

`go test`'s exit code (0 = all passed).
