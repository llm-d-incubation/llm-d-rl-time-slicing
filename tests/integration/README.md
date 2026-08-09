# Integration Tests

End-to-end tests for the time-slicing stack on real GPU hardware, exercising the official Helm chart deployment paths.

The test suite is written in Go and runs inside the cluster: `run.sh` deploys a test-runner pod, copies the repo source into it, and executes `go test` there.

## Suites

### snapshot-agent (phases: standalone, k8s)

Tests snapshot-agent backends in standalone and k8s modes. The Go harness deploys the snapshot-agent and inference engine pods itself -- one engine at a time, so a single free GPU is enough. All snapshot/restore calls go through the **Python client** (`timeslice.snapshot_agent`, invoked via `agentctl.py`), so the entire client layer is covered.

**How the standalone mode works:** since the test suite runs inside a GKE cluster, standalone mode is simulated by deploying a privileged pod with `hostPID` and `hostNetwork` on the test node. The `make standalone` artifacts are built in the test runner and copied into this pod, which then runs the agent binary with the same GPU and PID namespace access as a host process.

### orchestrator (phase: orchestrator)

Composed orchestrator integration suite. Installs BOTH official Helm charts (snapshot-agent + timeslice-orchestrator) and drives real orchestrator scenarios through the gRPC API. The orchestrator chart is configured with `snapshotAgentPort` matching `CHART_AGENT_PORT` so it commands the suite's own agent.

Uses a 2-node topology: `TEST_NODE_SAMPLERS` for the samplers group, `TEST_NODE_TRAINERS` for the trainers group (one GPU per node, separate nodes). `exclusiveLabel` temporarily removes group labels from other nodes so pods land only on the designated test nodes.

## Layout

- `run.sh` -- launcher (build images, install chart fixtures, deploy runner, copy source, build `make standalone`, install the Python client, `go test`, cleanup)
- `runner.yaml` -- test-runner pod + RBAC
- `harness/` -- shared framework: in-cluster client, node selection, pod lifecycle, exec/HTTP/VRAM helpers
- `snapshot-agent/` -- the agent suite: `standalone_test.go` / `k8s_test.go`, plus the agent specifics (`harness.go` agent deployment, `engines.go` engine specs, `agentctl.py`)
- `orchestrator/` -- the orchestrator suite: `orchestrator_test.go` / `harness.go`
- `orchestrator/scenarios/` -- scenario drivers shared by both the simulate tier (unit tests) and the composed suite
- `orchestrator/simulate/` -- fakes tier: in-process orchestrator with fake K8s, runs on every PR

## Adding a test

**snapshot-agent:** Add a `t.Run(...)` inside the engine group that provides the pods it needs, using the harness helpers:

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

**orchestrator:** Add a scenario to `orchestrator/scenarios/` and call it from the `TestOrchestrator` function in `orchestrator/orchestrator_test.go`.

## Prerequisites

- A GKE cluster with at least 1 free GPU
- `gcloud` and `kubectl` on the machine running the tests
  (Go and everything else run inside the cluster)
- For the k8s/orchestrator phases: images built from the official
  Dockerfiles (`--build` does this for you). The standalone phase needs no
  image -- it builds the agent from source in the test runner.
- Cloud Build API enabled (`gcloud services enable cloudbuild.googleapis.com`)
  and permission to push to the project's registry; GKE nodes in the same
  project can pull from `gcr.io/<project>` by default.

## Testing your changes

Everything runs from your working directory -- uncommitted changes included --
so no commit or merge is needed at any layer:

```bash
# snapshot-agent phases (standalone + k8s):
TEST_NODE=<gpu-node> ./tests/integration/run.sh \
  --build --project <gcp-project>

# orchestrator phase only (2 GPU nodes, one per group):
TEST_NODE_SAMPLERS=<gpu-node-1> TEST_NODE_TRAINERS=<gpu-node-2> \
  ./tests/integration/run.sh \
  --build --project <gcp-project> --phase orchestrator

# everything (standalone + k8s on TEST_NODE, orchestrator on the 2-node topology):
TEST_NODE=<gpu-node> TEST_NODE_SAMPLERS=<gpu-node-1> TEST_NODE_TRAINERS=<gpu-node-2> \
  ./tests/integration/run.sh \
  --build --project <gcp-project> --phase all
```

`--build` has Cloud Build produce the images from the working directory
(tagged `integ-<commit>` so repeated runs don't collide with node image
caches).

Alternative (pre-built images -- any registry the cluster can pull from):

```bash
TEST_NODE=<gpu-node> TEST_NODE_SAMPLERS=<gpu-node-1> TEST_NODE_TRAINERS=<gpu-node-2> \
  ./tests/integration/run.sh \
  --agent-image gcr.io/<project>/snapshot-agent:dev \
  --orch-image gcr.io/<project>/timesliceorchestrator:dev \
  --phase all
```

## Options

```text
--agent-image IMAGE  Snapshot-agent image (required for k8s/orchestrator
                     unless --build)
--orch-image IMAGE   Orchestrator image (required for orchestrator unless
                     --build)
--build              Build images from the working directory via Cloud Build
                     (requires --project); explicit --agent-image /
                     --orch-image overrides
--project PROJECT    GCP project (required with --build for image pushes;
                     also used by gcloud get-credentials with --cluster)
--cluster CLUSTER    GKE cluster name (optional; omit to use current
                     kubectl context)
--zone ZONE          GKE cluster zone (optional)
--model MODEL        Model to load (default: Qwen/Qwen2.5-0.5B)
--phase PHASE        "standalone", "k8s", "orchestrator", "both" (default,
                     = standalone+k8s), or "all" (= standalone+k8s+orch)
--skip-cleanup       Leave the test-runner pod and chart fixtures running
                     for debugging
```

Environment:

- `TEST_NODE=<node-name>` -- required for the standalone and k8s phases
  (charts are pinned to this node); for the standalone phase it pins the
  suite instead of the default pick (first node with a free GPU). Use the
  pin when the cluster runs workloads that occupy GPUs without requesting
  them (time-slicing experiments), which the default pick cannot see.
- `TEST_NODE_SAMPLERS=<node-name>` -- required for the orchestrator phase.
  GPU node for the samplers group (must be different from `TEST_NODE_TRAINERS`).
- `TEST_NODE_TRAINERS=<node-name>` -- required for the orchestrator phase.
  GPU node for the trainers group (must be different from `TEST_NODE_SAMPLERS`).
- `CHART_AGENT_PORT=<port>` -- port for the chart-deployed agent (default
  9002), so the suite can coexist with an unrelated agent on the default
  port (the chart runs on hostNetwork). The orchestrator chart is
  configured with the same port via `snapshotAgentPort`.

## Exit code

`go test`'s exit code (0 = all passed).
