# Time-Slicing Integration Guide for verl Workloads

This guide covers integrating **verl** (Volcano Engine Reinforcement Learning framework) with the **llm-d-rl-time-slicing** platform using the pre-packaged `timeslice-verl` integration (`pkg/integrations/verl/`).

### Motivation: Maximizing GPU Utilization

In verl's **fully-async** recipe (`fully_async_policy`), rollout generation runs continuously while the trainer sits idle whenever it waits for the next batch of samples — in our reference runs that idle fraction was ~55–70% of wall-clock. Cooperative time-slicing backfills those valleys: multiple jobs' trainers share one physical GPU pool, and the platform checkpoints the idle trainer's GPU state to host RAM (`cuda-checkpoint`) at every handoff. Rollout engines keep dedicated GPUs and never stop generating: while a job's trainer is checkpointed off the shared GPU, its rollout engine keeps generating samples into the staleness queue.

### Supported Modes

| verl mode | Status | Notes |
|---|---|---|
| **Fully-async** (`fully_async_policy`) | **Supported** | Via the `TimesliceFullyAsyncTrainer` subclass + fork lifecycle hooks. See the runnable example below. |
| v1 sync modes (e.g. `main_ppo` PPO/GRPO) | Not shipped in this PR | The v1 trainers expose the same `on_*` hook convention; integration is future work. |

For a runnable example, see:

* **[Fully-Async Example](examples/fully-async/README.md)** — two fully-async RL jobs (math RLVR + code RLVR) sharing one trainer GPU, each with a dedicated rollout GPU

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploying the Time-Slicing Platform](#2-deploying-the-time-slicing-platform)
3. [Integrating with verl](#3-integrating-with-verl)
4. [General Troubleshooting](#4-general-troubleshooting)

---

## 1. Cluster Prerequisites

Before deploying cooperative time-slicing for verl, ensure your environment meets the following requirements:

* A Kubernetes cluster (we validated on GKE, COS nodes) with 1-GPU nodes (H100-class recommended). The GPUs used for time-slicing must NOT be in use by anything else.
* Cluster-admin access; `kubectl`, `helm` (v3), `git`, and `envsubst` (from gettext; `brew install gettext` on macOS) on your workstation.
* Nodes can pull from Docker Hub (`verlai/verl` image, ~20 GB) and reach GitHub + HuggingFace from pods (model + dataset download at startup; the model is fetched by HF id per node, no shared volume).

### Node Labeling and Time-Slice Groups

The orchestrator discovers resource pools (*groups*) from node labels. Jobs in the same group take turns holding the group's accelerator lock. Label ONLY the nodes whose GPUs should be time-sliced into the group — the orchestrator discovers groups from this label:

```bash
kubectl label node <trainer-node> group.timeslice.io/trainers=true --overwrite
```

Nodes hosting rollout engines must NOT carry the group label, and rollout pods must not carry `timeslice.io/*` pod labels: only labeled pods on labeled nodes are snapshot candidates, so unlabeled rollout processes are never touched by the platform.

Verify the orchestrator synced the group after labeling:

```bash
kubectl -n timeslice-system logs deploy/timeslice-timesliceorchestrator --tail=50 \
  | grep -i trainers | tail -3
```

### Host-RAM Sizing Rule

A head pod's memory limit must fit its normal RSS **plus its entire GPU allocation** — the cuda-checkpoint dump of a trainer's full GPU state lands in pod memory. Size trainer-node host RAM to hold the GPU memory footprint of every job whose trainer can be checkpointed there.

---

## 2. Deploying the Time-Slicing Platform

We deploy the core platform components — **TimeSlice Orchestrator** (Deployment: the gRPC lock service) and **Snapshot Agent** (DaemonSet, one per GPU node: performs `cuda-checkpoint` snapshot/restore of a job's GPU state when the orchestrator hands the lock over) — using the parent Helm chart.

> If you previously installed the platform, do a clean
> `helm uninstall timeslice -n timeslice-system` first — `helm install` fails
> on an existing release name. (Note: uninstalling also
> removes the bundled NVIDIA DRA driver DaemonSet until you reinstall.)

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing

helm dependency update ./deploy
helm install timeslice ./deploy -n timeslice-system --create-namespace
```

> **Release pin:** the chart defaults pull the official `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` images at `latest`. Once the first tagged release is cut, pin it with `--set timesliceorchestrator.image.tag=<version> --set snapshot-agent.image.tag=<version>`.

Verify (orchestrator Running; one agent pod per GPU node):

```bash
kubectl -n timeslice-system get pods -o wide
```

---

## 3. Integrating with verl

### How It Works

The `timeslice-verl` package ships `TimesliceFullyAsyncTrainer`, a subclass of verl's `fully_async_policy` trainer that overrides the trainer's empty `on_*` lifecycle hooks to acquire/release the orchestrator lock at the trainer's natural wait points. It registers under the trainer name `timeslice` (verl's `register_trainer` registry; the package's `verl.plugins` entry point makes `import verl` load the registering module) and is selected with a single hydra override: `async_training.trainer_name=timeslice`. No monkey-patching and no verl source edits in your job — it requires a verl build with the fully-async lifecycle hooks + trainer registry and per-pool placement-group bundle resources (currently the `feat/fully-async-lifecycle-hooks` fork branch, until the commits land upstream).

Lock protocol:

| Trainer lifecycle point | Lock action |
|---|---|
| `init_workers` (model load) | ACQUIRE → load → YIELD |
| initial weight sync (pre-fit) | ACQUIRE → sync → YIELD |
| a batch of samples is ready | ACQUIRE (this is the resume point) |
| weight update + param sync done | YIELD |

### Placement-Group Pinning

Trainer/rollout placement is pinned with Ray **custom resources** — the head starts ray with `--resources='{"trainer_node": 100}'`, the worker with `--resources='{"rollout_node": 100}'`, and verl's per-pool placement-group bundle resources (hydra override `ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool: {rollout_node: 1}}`) make verl's trainer placement group request `{trainer_node: 1}` and the rollout PG `{rollout_node: 1}`.

### Installing verl (Pinned Fork)

Each job pod clones the [`feat/fully-async-lifecycle-hooks`](https://github.com/aishukamal/verl/tree/feat/fully-async-lifecycle-hooks) branch of `github.com/aishukamal/verl` — upstream `983cb0f24443f87b3d161fad318445130a620b07` + two feature commits (fully-async lifecycle hooks + trainer registry; per-pool PG bundle resources) — and installs it with `pip install -e ".[gpu]"`. This is a temporary fork until the commits land upstream; once they do, point the `VERL_REPO`/`VERL_REF` pod envs at mainline verl instead.

### Installing the `timeslice-verl` Package (ConfigMap)

Publish the integration package (`pkg/integrations/verl/`) into the cluster as ConfigMaps — the job pods install it from these, no GitHub fetch:

```bash
# Package source: the repo you cloned in §2
PKG=llm-d-rl-time-slicing/pkg/integrations/verl
kubectl create configmap timeslice-verl-root --from-file=$PKG/pyproject.toml
kubectl create configmap timeslice-verl-src  --from-file=$PKG/timeslice_verl/
```

ConfigMap keys cannot contain `/`, so the package tree is split into two maps (project root + module sources) and reassembled inside the pod; this also works in air-gapped clusters.

### Required NCCL Environment

Set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` on every job pod — the NVLS/cuMem NCCL transports don't survive cuda-checkpoint, and a trainer will die right after a restore without them (the example manifests already set both).

---

## 4. General Troubleshooting

Every entry below is a failure we actually hit while building the integration. For failures specific to the runnable demo's topology and manifests, see the [example's troubleshooting section](examples/fully-async/README.md#8-troubleshooting).

| Symptom | Cause | Fix |
|---|---|---|
| Trainer placement group lands on the rollout node (or rollout work on the trainer node) | PG pinning not active — `--resources` missing on a `ray start` line, or the `ray_pg_extra_resources` hydra override is missing/typo'd, or the verl build lacks the feature | Verify both `ray start` lines carry `--resources`, keep the `ray_pg_extra_resources={trainer_pool: ...}` override in the head launch script intact, and use the pinned verl branch (§3); check `ray status` custom resources |
| Rollout throughput collapses when the other job trains; agent log shows cuda-checkpoint activity on a ROLLOUT node | A rollout pod carries `timeslice.io/*` labels, or a rollout node is labeled into the group | Rollout pods/nodes must stay unlabeled — only head pods and the trainer node (§1). A rollout node must never show cuda-checkpoint activity |
| `Found multiple active Ray instances` warning on a shared node | With `hostPID`, Ray's `address='auto'` discovery scans `/proc` and sees the OTHER job's GCS — a driver can join the wrong ray cluster | `export RAY_ADDRESS="$(hostname -i):6379"` right after `ray start --head` (the example manifests do) — keep that line if you edit |
| Hydra error `Key 'use_trainer_do_validate' is not in struct trainer` | Wrong config key path | These flags live under `async_training.*`, not `trainer.*` (the example manifests are correct) |
| `save_freq` / checkpoint-save-skipped warning in log | Checkpoint saving would RPC a possibly-frozen worker | By design: the timeslice trainer vetoes saves when `save_freq > 0`; keep `trainer.save_freq=-1` |
| Steps complete but one job's staleness queue near `staleness_threshold × batch` | Very asymmetric job speeds | Raise `async_training.staleness_threshold` (we use 8) or budget shorter runs; queue overflow drops samples |
