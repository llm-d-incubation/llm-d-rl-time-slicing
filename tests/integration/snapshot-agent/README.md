# Snapshot Agent Integration Tests

End-to-end tests for snapshot-agent backends on real GPU hardware, in both standalone and K8s deployment modes.

The test suite is written in Go and runs inside the cluster: `run.sh` deploys a test-runner pod, copies the repo source into it, and executes `go test` there. The Go harness deploys the snapshot-agent and inference engine pods itself — one engine at a time, so a single free GPU is enough.

All snapshot/restore calls go through the **Python client** (`timeslice.snapshot_agent`, invoked via `agentctl.py`), so the entire client layer is covered.

- `run.sh` — launcher (deploy runner, copy source, install the Python client, `go test`, cleanup)
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
- A snapshot-agent image. Build one from the repo root with:

```bash
gcloud builds submit --config=cloudbuild-image.yaml \
  --substitutions=_IMAGE=gcr.io/<your-project>/snapshot-agent:dev .
```

This builds from your working directory, so local modifications are included — no commit needed. Requires the Cloud Build API (`gcloud services enable cloudbuild.googleapis.com`) and permission to push to the project's registry; GKE nodes in the same project can pull from `gcr.io/<project>` by default.

## Running

```bash
./tests/integration/snapshot-agent/run.sh \
  --image gcr.io/<your-project>/snapshot-agent:dev \
  --project <your-project> \
  --cluster <your-cluster> \
  --zone <your-zone>
```

## Options

```
--image IMAGE        Snapshot-agent container image (required)
--project PROJECT    GCP project (runs gcloud get-credentials)
--cluster CLUSTER    GKE cluster name
--zone ZONE          GKE cluster zone
--model MODEL        Model to load (default: Qwen/Qwen2.5-0.5B)
--phase PHASE        "standalone", "k8s", or "both" (default: both)
--skip-cleanup       Leave the test-runner pod running for debugging
```

## Exit code

`go test`'s exit code (0 = all passed).
