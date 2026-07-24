# Snapshot Agent Integration Tests

End-to-end tests for snapshot-agent backends on real GPU hardware, in k8s and standalone modes, exercising the Helm chart and Makefile deployment paths respectively.

The test suite is written in Go and runs inside the cluster: `run.sh` deploys a test-runner pod, copies the repo source into it, and executes `go test` there. The Go harness deploys the snapshot-agent and inference engine pods itself — one engine at a time, so a single free GPU is enough.

All snapshot/restore calls go through the **Python client** (`timeslice.snapshot_agent`, invoked via `agentctl.py`), so the entire client layer is covered.

- `run.sh` — launcher (build image, install chart fixture, deploy runner, copy source, build `make standalone`, install the Python client, `go test`, cleanup)
- `runner.yaml` — test-runner pod + RBAC
- `harness.go` / `engines.go` — harness: pod lifecycle, exec/HTTP helpers, pod specs
- `agentctl.py` — thin CLI over the Python client; builds `BackendConfig` protos in Python from primitive flags
- `standalone_test.go` / `k8s_test.go` — the test cases

## Adding a test

Add a `t.Run(...)` inside the engine group that provides the pods it needs, using the harness helpers:

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

A new engine is an `EngineSpec` in `engines.go`.

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

Everything runs from your working directory — uncommitted changes included —
so no commit or merge is needed at any layer:

```bash
TEST_NODE=<gpu-node> ./tests/integration/snapshot-agent/run.sh \
  --build --project <gcp-project>
```

`--build` has Cloud Build produce the agent image from the working directory
(tagged `integ-<commit>` so repeated runs don't collide with node image
caches) and runs both phases against it. `run.sh` then copies the local
workspace into the cluster, so the standalone phase's `make standalone`
build, the Helm chart (installed from local `deploy/`), the Python client,
and the test code all come from the workspace too.

Alternative (pre-built image — any registry the cluster can pull from):

```bash
gcloud builds submit --config=cloudbuild-image.yaml \
  --substitutions=_IMAGE=gcr.io/<project>/snapshot-agent:dev .

TEST_NODE=<gpu-node> ./tests/integration/snapshot-agent/run.sh \
  --agent-image gcr.io/<project>/snapshot-agent:dev
```

## Options

```text
--agent-image IMAGE  Snapshot-agent image the k8s phase installs via the
                     official chart (required for k8s/both unless --build)
--build              Build the agent image from the working directory via
                     Cloud Build (requires --project); explicit
                     --agent-image overrides
--project PROJECT    GCP project (image pushes with --build; also used by
                     gcloud get-credentials with --cluster)
--cluster CLUSTER    GKE cluster name
--zone ZONE          GKE cluster zone
--model MODEL        Model to load (default: Qwen/Qwen2.5-0.5B)
--phase PHASE        "standalone", "k8s", or "both" (default)
--skip-cleanup       Leave the test-runner pod and chart fixture running
                     for debugging
```

Environment:

- `TEST_NODE=<node-name>` — required for the k8s phase (the chart is pinned
  to this node); for the standalone phase it pins the suite instead of the
  default pick (first node with a free GPU by requests). Use the pin when
  the cluster runs workloads that occupy GPUs without requesting them
  (time-slicing experiments), which the default pick cannot see.
- `CHART_AGENT_PORT=<port>` — port for the chart-deployed agent (default
  9001), so the suite can coexist with an unrelated agent on the default
  port (the chart runs on hostNetwork).

## Exit code

`go test`'s exit code (0 = all passed).
