# Time-Slicing Integration Guide for verl Workloads

This guide provides step-by-step instructions on how to time-slice any two **verl** (Volcano Engine Reinforcement Learning framework) async RL workloads with the **llm-d-rl-time-slicing** platform using the pre-packaged `llm-d-timeslice-verl` integration (`pkg/integrations/verl/`). A complete two-job runnable is provided as an [example](examples/fully-async/README.md); the sections below are the generic walkthrough it instantiates.

### Motivation: Maximizing GPU Utilization

In verl's **fully-async** recipe (`fully_async_policy`), rollout generation runs continuously while the trainer sits idle whenever it waits for the next batch of samples — in our reference runs that idle fraction was ~55–70% of wall-clock. Cooperative time-slicing backfills those valleys: multiple jobs' trainers share one physical GPU pool, and the platform checkpoints the idle trainer's GPU state to host RAM (`cuda-checkpoint`) at every handoff. Rollout engines keep dedicated GPUs and never stop generating: while a job's trainer is checkpointed off the shared GPU, its rollout engine keeps generating samples into the staleness queue.

This guide covers verl's fully-async trainer (`fully_async_policy`), integrated via the `TimesliceFullyAsyncTrainer` subclass and verl's fully-async lifecycle hooks.

For a runnable example, see:

* **[Fully-Async Example](examples/fully-async/README.md)** — the trainers of two fully-async RL jobs (math RLVR + code RLVR) time-sliced on one GPU, each job with a dedicated rollout GPU

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploying the Time-Slicing Platform](#2-deploying-the-time-slicing-platform)
3. [Integrating with verl](#3-integrating-with-verl)
4. [Deploying Time-Sliced verl Jobs](#4-deploying-time-sliced-verl-jobs)
5. [Submitting and Observing Time-Sliced Jobs](#5-submitting-and-observing-time-sliced-jobs)
6. [General Troubleshooting](#6-general-troubleshooting)

---

## 1. Cluster Prerequisites

Before deploying cooperative time-slicing for verl, ensure your environment meets the following requirements:

* Kubernetes **v1.34** or later (we validated on GKE, COS nodes) with 1-GPU nodes (H100-class recommended). The GPUs used for time-slicing must NOT be in use by anything else.
* GPU nodes must run **NVIDIA GPU Driver 565 or later**. This is a strict requirement to support **NVIDIA Dynamic Resource Allocation (DRA)**. The NVIDIA DRA driver itself ships with the platform Helm chart (§2).
* Cluster-admin access; `kubectl`, `helm` (v3), and `git` on your workstation.
* Nodes can pull from Docker Hub (`verlai/verl` image, ~20 GB) and reach GitHub + HuggingFace from pods (model + dataset download at startup; the model is fetched by HF id per node, no shared volume).

### Node Labeling and Time-Slice Groups

The orchestrator discovers resource pools (*groups*) from node labels. Jobs in the same group take turns holding the group's accelerator lock. The principle is that time-slicing machinery touches ONLY time-sliced nodes: the **trainer node** carries all the time-slicing markers — the group label (drives group discovery), the `timeslice.io/enabled` label (the platform-wide marker for nodes participating in time-slicing — the same convention the slime guide uses; in this recipe that is only the trainer node — and what the platform install in §2 targets snapshot-agent and DRA kubelet-plugin placement at), and the isolation taint (keeps unrelated workloads off the shared GPU; all pod specs in this guide tolerate it) — while **rollout nodes carry nothing timeslice-related**. Set up the trainer node with three commands:

```bash
kubectl label node <trainer-node> group.timeslice.io/trainers=true --overwrite
kubectl label node <trainer-node> timeslice.io/enabled=true --overwrite
kubectl taint node <trainer-node> timeslice.io/shared=true:NoSchedule --overwrite
```

Do NOT put either label on nodes hosting rollout engines, and do not put `timeslice.io/*` pod labels on rollout pods: only labeled pods on labeled nodes are snapshot candidates, so unlabeled rollout processes are never touched by the platform. Rollout nodes need **no labels or taints at all** — rollout pods request GPUs the ordinary way (§ Shared DRA Resource Claim below), which is GPU access, not time-slicing. Placement follows from three guarantees.

1. Both jobs' head pods co-locate on the trainer node because they consume the SAME shared `ResourceClaim`, and a claim's consumers must run where its device is. (If the trainer node lacks CPU/RAM headroom for the second head pod, that pod goes Pending; see the host-RAM sizing rule below.)
2. Rollout pods can never land on the trainer node: each rollout pod carries a required node anti-affinity on `group.timeslice.io/trainers`.
3. The device plugin allocates whole GPUs exclusively, so two rollout pods always land on distinct GPUs; on a 1-GPU-per-node topology that means distinct nodes. On multi-GPU nodes this becomes distinct GPUs, possibly on the same node.

On clusters with GPU nodes beyond the recipe's, rollouts may schedule onto any free GPU node; operators running amid other workloads should scope their GPU pool (e.g. with their own labels/taints).

### Shared DRA Resource Claim

Cooperative time-slicing leverages Kubernetes **Dynamic Resource Allocation (DRA)** so multiple jobs' pods can share physical GPU hardware without scheduler blocking — and without elevated pod permissions: the NVIDIA DRA driver injects the GPU and driver userspace into ordinary pods. The recipe contains exactly ONE `ResourceClaim` — the **shared** trainer claim — because sharing is what only DRA can express: reference it from every job's head pod, and each tenant gets the full, unpartitioned GPU during its turn. Rollout pods request their GPUs the ordinary way (`nvidia.com/gpu: 1` via the device plugin) — they need no claims. (On clusters that manage GPUs purely via DRA, with no device plugin, give each rollout pod its own `ExactCount: 1` claim instead — and since the DRA kubelet plugin must then run on the rollout nodes too, drop or widen the `kubeletPlugin` nodeSelector flag from the §2 install.)

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: shared-trainers-gpu-claim
  namespace: default
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
        allocationMode: ExactCount
        count: 1
```

Apply the claim before submitting jobs (the example ships this exact file):
```bash
kubectl apply -f guides/rl-frameworks/verl/examples/fully-async/resource-claims.yaml
```

Verify the orchestrator synced the group after labeling:

```bash
kubectl -n timeslice-system logs deploy/timeslice-timesliceorchestrator --tail=50 \
  | grep -i trainers | tail -3
```

### Host-RAM Sizing Rule

A head pod's memory limit must fit its normal RSS **plus its entire GPU allocation** — the cuda-checkpoint dump of a trainer's full GPU state lands in pod memory. Size trainer-node host RAM to hold the GPU memory footprint of every job whose trainer can be checkpointed there.

---

## 2. Deploying the Time-Slicing Platform

Deploy the core platform components — **TimeSlice Orchestrator** (Deployment: the gRPC lock service) and **Snapshot Agent** (DaemonSet on the time-sliced nodes: performs `cuda-checkpoint` snapshot/restore of a job's GPU state when the orchestrator hands the lock over) — using the parent Helm chart.

### Step 1: Clone the Repository

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing
```

> **Release pin:** the chart is not published to a registry — you install it from the clone. To pin a release, clone the repo at the release tag (`git clone --branch <version> https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git`) so the chart matches the pinned images, and pin the images with `--set timesliceorchestrator.image.tag=<version> --set snapshot-agent.image.tag=<version>` (the chart defaults pull the official `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` images at `latest`). Until the first tagged release is cut, `main` + `latest` is the only option.

### Step 2: Install the Helm Chart

Install into a dedicated namespace (`timeslice-system`), pinning the node-level platform components to `timeslice.io/enabled` nodes:

```bash
helm dependency update ./deploy
helm install timeslice ./deploy -n timeslice-system --create-namespace \
  --set-string "snapshot-agent.nodeSelector.timeslice\.io/enabled=true" \
  --set-string "nvidia-dra-driver-gpu.kubeletPlugin.nodeSelector.timeslice\.io/enabled=true"
```

The first flag pins the snapshot agent (a `hostPID` DaemonSet) to `timeslice.io/enabled` nodes — in this recipe, just the trainer node; agents on rollout nodes would perform no work anyway (all prior runs confirmed this). The second flag pins the NVIDIA DRA driver's kubelet plugin — the bundled driver's only node-level component, which publishes `ResourceSlice`s and injects the GPU into claim-consuming pods — to the same nodes: DRA is needed exactly where the shared claim binds, and gating it keeps other GPU nodes free of `ResourceSlice`s advertising their GPUs as claimable.

**Zero-impact principle:** you can share the cluster with non-timeslice workloads with zero impact to them. This install plus the labels + taint in §1 touch only the trainer node — the snapshot agent and the DRA kubelet plugin run only there, and every other node carries no platform components, labels, or taints, so workloads on those nodes are unaffected. The recipe's rollout pods compete for free GPUs like any ordinary workload: standard scheduling, no special machinery.

> If a `timeslice` release already exists, do a clean
> `helm uninstall timeslice -n timeslice-system` first — `helm install` fails
> on an existing release name. (Note: uninstalling also
> removes the bundled NVIDIA DRA driver DaemonSet until you reinstall.)

### Step 3: Verify Platform Health

Verify the orchestrator is Running, and that an agent pod and a `nvidia-dra-driver-gpu-kubelet-plugin` pod run on the trainer node — and only there; no other node should carry either:

```bash
kubectl -n timeslice-system get pods -o wide
```

Optionally verify with the `rlts` CLI (requires Go toolchain; from the repo root — the same binary is used to observe lock handoffs while jobs run):

```bash
go build -o bin/rlts ./cmd/rlts
./bin/rlts orchestrator list
```

---

## 3. Integrating with verl

### How It Works

The `llm-d-timeslice-verl` package ships `TimesliceFullyAsyncTrainer`, a subclass of verl's `fully_async_policy` trainer that overrides the trainer's empty `on_*` lifecycle hooks to acquire/release the orchestrator lock at the trainer's natural wait points. It registers under the trainer name `timeslice` (verl's `register_trainer` registry; the package's `verl.plugins` entry point makes `import verl` load the registering module) and is selected with a single hydra override: `async_training.trainer_name=timeslice`. No monkey-patching and no verl source edits in your job — it requires a verl build with the fully-async lifecycle hooks + trainer registry and per-pool placement-group bundle resources (currently the `feat/fully-async-lifecycle-hooks` fork branch, until the commits land upstream).

Lock protocol:

| Trainer lifecycle point | Lock action |
|---|---|
| `init_workers` (model load) | ACQUIRE → load → YIELD |
| initial weight sync (pre-fit) | ACQUIRE → sync → YIELD |
| a batch of samples is ready | ACQUIRE (this is the resume point) |
| weight update + param sync done | YIELD |

### Placement-Group Pinning

Pin trainer/rollout placement with Ray **custom resources** — start ray on the head with `--resources='{"trainer_node": 100}'`, on the worker with `--resources='{"rollout_node": 100}'`, and pass verl's per-pool placement-group bundle resources (hydra override `ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool: {rollout_node: 1}}`) so verl's trainer placement group requests `{trainer_node: 1}` and the rollout PG `{rollout_node: 1}`.

### Installing verl (Pinned Fork)

In each job pod (or your job image), clone the [`feat/fully-async-lifecycle-hooks`](https://github.com/aishukamal/verl/tree/feat/fully-async-lifecycle-hooks) branch of `github.com/aishukamal/verl` — upstream `983cb0f24443f87b3d161fad318445130a620b07` + two feature commits (fully-async lifecycle hooks + trainer registry; per-pool PG bundle resources) — and install it with `pip install -e ".[gpu]"`. This is a temporary fork until the commits land upstream; once they do, point the install at mainline verl instead (the example manifests parameterize this as the `VERL_REPO`/`VERL_REF` pod envs).

### Installing the `llm-d-timeslice-verl` Package

Install the timeslice client and the integration package in every job pod (or bake them into your job image):

```bash
# Install the timeslice client library
pip install "git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"

# Install the llm-d-timeslice-verl trainer package
pip install --no-cache-dir --no-deps "llm-d-timeslice-verl @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/integrations/verl"
```

> **Why `--no-deps`?** The package declares no dependencies of its own — the timeslice client it uses is installed separately (above), so there is nothing for pip to resolve.

### Usage

Every time-sliced verl job must carry this contract:

* **Hydra overrides** on the verl launch command:
  * `async_training.trainer_name=timeslice` — selects the registered timeslice trainer.
  * `"ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool: {rollout_node: 1}}"` — pins the placement groups (see [Placement-Group Pinning](#placement-group-pinning)).
  * `trainer.save_freq=-1` — checkpoint saving is disabled under time-slicing: a save RPCs the FSDP worker, which may be checkpointed off the GPU at that moment. The timeslice trainer vetoes saves (with a warning) if this is left `> 0`. Note the consequence: time-sliced runs write no training checkpoints.
  * `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true` — tears down and rebuilds the trainer↔rollout weight-sync NCCL communicator around each sync, so no communicator exists outside lock-held windows.
  * `async_training.trigger_parameter_sync_step=1` — yield the GPU once per training step; larger values hold the lock across sample-queue waits.
  * `async_training.use_dynamic_resource_scheduling=false` and `async_training.use_trainer_do_validate=false` — keep dynamic rescheduling and validation off the time-sliced trainer GPU (both would touch the GPU outside the lock protocol).
* **`ray start` custom resources**: head `--resources='{"trainer_node": 100}'`, rollout worker `--resources='{"rollout_node": 100}'`.
* **Environment variables** (set as container-level env vars on both pods, kept identical — the trainer driver actor is a CPU-only Ray actor that can be scheduled on either Ray node and inherits the raylet's env):
  * `TIMESLICE_FULLY_ASYNC=1` — activates the hooks (they are inert without it).
  * `TIMESLICE_JOB_ID` — unique per job; must equal the head pod's `timeslice.io/job-id` label.
  * `TIMESLICE_GROUP` — the group whose lock the trainer takes (e.g. `trainers`).
  * `TIMESLICE_ORCH_ADDR` — orchestrator gRPC address (`timeslice-timesliceorchestrator.timeslice-system.svc:50051`).
* **Pod labels** on the head pod ONLY (the trainer is the sole time-sliced process; rollout pods stay unlabeled): `timeslice.io/job-id: <job id>` and `timeslice.io/group: <group>` — the exact keys the platform selects snapshot candidates on.
* **NCCL environment**: set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` on every job pod (the example manifests set both) — precautionary settings around cuda-checkpoint; keep them as set.

---

## 4. Deploying Time-Sliced verl Jobs

Deploy each job as a **2-node Ray cluster**:

* **Head pod** — runs the verl driver + FSDP trainer. Schedule it on the shared, group-labeled trainer node (`nodeSelector: group.timeslice.io/trainers: "true"`), reference the **shared** trainer `ResourceClaim` (the same claim in every job's head pod), and carry the `timeslice.io/job-id` + `timeslice.io/group` pod labels.
* **Rollout pod** — runs the vLLM rollout engine as a ray worker. Request its GPU the ordinary way (`resources.limits: nvidia.com/gpu: "1"` — the device plugin, no claim), and keep it off the trainer node (whose GPU is claim-managed) with a required node anti-affinity on `group.timeslice.io/trainers` (`operator: DoesNotExist`): it can then land on any GPU node except the trainer node, and since the device plugin allocates whole GPUs exclusively, rollout pods always get distinct GPUs. Leave it unlabeled (no `timeslice.io/*` labels).
* **Headless Service** — pointing at the head pod, so the rollout pod can resolve the ray head address.
* **Tolerations** — give both pods tolerations for the GPU-node taints they must land on (the example tolerates `nvidia.com/gpu` and `timeslice.io/shared=true:NoSchedule`).

A complete two-job runnable is provided in [examples/fully-async/](examples/fully-async/).

---

## 5. Submitting and Observing Time-Sliced Jobs

### Step 1: Submit Multiple Jobs

Apply two (or more) independent verl jobs with unique `TIMESLICE_JOB_ID` values and the same `TIMESLICE_GROUP`:

```bash
kubectl apply -f <job-a>.yaml
kubectl apply -f <job-b>.yaml
```

### Step 2: Port-Forward the Orchestrator

```bash
kubectl port-forward svc/timeslice-timesliceorchestrator 50051:50051 -n timeslice-system
```

### Step 3: Observe Time-Slicing via the CLI

Watch the shared trainer pool with the `rlts` CLI (built in §2):

```bash
watch -n 0.5 ./bin/rlts orchestrator status trainers
```

**Expected output:** the `Active Job` and `Locking Job` alternate between the jobs' `TIMESLICE_JOB_ID`s. While one job's trainer holds the GPU, the other shows up in the `Waiter Queue Depth`. The rollout engines never appear here — they take no locks.

### Step 4: Observe Context Switches in the Logs

The integration package logs every acquire/release into the head pod's log:

```bash
kubectl logs <head-pod> --tail=2000 | grep -E "ACQUIRE|RELEASE" | tail -5
# [timeslice] job=<job id> ACQUIRE group=trainers waited=232000ms context_restored=True
# [timeslice] job=<job id> RELEASE group=trainers pending_waiters=1 snapshot_deferred=False
```

Snapshot/restore operations appear in the log of the snapshot-agent pod on the shared trainer node:

```bash
kubectl -n timeslice-system logs <agent-pod-on-trainer-node> --tail=100 | grep -i "cuda-checkpoint"
```

---

## 6. General Troubleshooting

For failures specific to the runnable demo's topology and manifests, see the [example's troubleshooting section](examples/fully-async/README.md#8-troubleshooting).

| Symptom | Cause | Fix |
|---|---|---|
| Trainer placement group lands on the rollout node (or rollout work on the trainer node) | PG pinning not active — `--resources` missing on a `ray start` line, or the `ray_pg_extra_resources` hydra override is missing/typo'd, or the verl build lacks the feature | Verify both `ray start` lines carry `--resources`, keep the `ray_pg_extra_resources={trainer_pool: ...}` override in the head launch script intact, and use the pinned verl branch (§3); check `ray status` custom resources |
| Rollout throughput collapses when the other job trains; agent log shows cuda-checkpoint activity on a ROLLOUT node | A rollout pod carries `timeslice.io/*` labels, or a rollout node is labeled into the group | Rollout pods/nodes must stay unlabeled — only head pods and the trainer node (§1). A rollout node must never show cuda-checkpoint activity |
| `Found multiple active Ray instances` warning on a shared node | Ray's `address='auto'` discovery found more than one GCS — a driver can join the wrong ray cluster when two jobs' head pods share the trainer node | `export RAY_ADDRESS="$(hostname -i):6379"` right after `ray start --head` (the example manifests do) — keep that line if you edit |
| Hydra error `Key 'use_trainer_do_validate' is not in struct trainer` | Wrong config key path | These flags live under `async_training.*`, not `trainer.*` (the example manifests are correct) |
| `save_freq` / checkpoint-save-skipped warning in log | Checkpoint saving would RPC a possibly-frozen worker | By design: the timeslice trainer vetoes saves when `save_freq > 0`; keep `trainer.save_freq=-1` |
| Steps complete but one job's staleness queue near `staleness_threshold × batch` | Very asymmetric job speeds | Raise `async_training.staleness_threshold` (we use 8) or budget shorter runs; queue overflow drops samples |
