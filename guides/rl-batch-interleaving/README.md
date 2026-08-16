# Interleaving a verl RL Trainer with a Batch-Inference Server on One GPU

This guide is a step-by-step walkthrough for time-slicing **one verl
`fully_async_policy` RL training job** with **a batch-inference server** on
the same GPU through the llm-d-rl-time-slicing platform. In verl's
fully-async recipe, rollout generation runs continuously while the trainer
sits idle whenever it waits for the next batch of samples — in our reference
runs that idle fraction was 40–70% of wall-clock. Instead of backfilling
those valleys with another RL trainer, this recipe backfills them with
latency-tolerant batch inference: the RL trainer has **absolute priority**,
and the batch server holds the GPU only while the trainer doesn't want it.
Neither workload's source is modified.

The two tenants yield the GPU through two different checkpoint/restore
mechanisms, chosen to match their duty cycles:

- **The trainer** is checkpointed with **`cuda-checkpoint`** (the platform's
  default backend): the snapshot agent dumps its entire GPU state to host RAM
  at each yield (~10–35 s per swap for small models), amortized against
  training bursts that run for minutes.
- **The batch server** yields through the snapshot agent's **workload
  channel**: it registers sleep/wake callbacks that map to the inference
  engine's native offload (vLLM sleep mode moves weights + KV cache to host
  RAM in ~1–2 s and back in under a second — no process restart, no model
  reload). Second-scale yields are what make it safe for the batch tenant to
  give the GPU back the moment the trainer queues.

The RL job keeps a dedicated 1-GPU node for its rollout engine, which never
stops generating; the FSDP trainer and the batch server take turns on the
shared node:

```
  ROLLOUT NODE (dedicated)             TRAINER NODE (SHARED, time-sliced)
  ┌──────────────────────────┐         ┌──────────────────────────────────┐
  │ RL rollout pod           │  ray    │ RL head pod      batch server pod│
  │ rollout engine (vLLM,    │◀───────▶│ FSDP trainer ⇄   inference engine│
  │ generates continuously)  │         │ cuda-checkpoint  + supervisor    │
  └──────────────────────────┘         │      C/R         sleep/wake      │
                                       └──────────────────────────────────┘
  lock owner over time on the trainer node's GPU:
  batch ███████░T██████████░T█████████░T████████   (T = training burst)
               ▲ trainer queues → batch yields in ≤ poll(0.5s) + drain(~3s) + sleep(~1-2s)
```

This README is the generic walkthrough for running **any** verl fully-async
RL job against **any** batch tenant that can sleep/wake (or be
cuda-checkpointed). A complete pinned runnable — concrete models, manifests,
and a load generator, launchable in a handful of commands — is in
**[examples/](examples/README.md)**.

---

## Table of Contents
1. [Cluster Prerequisites](#1-cluster-prerequisites)
2. [Deploy the Time-Slicing Platform](#2-deploy-the-time-slicing-platform)
3. [Integrate Your RL Workload](#3-integrate-your-rl-workload)
4. [The Batch Tenant and the Supervisor](#4-the-batch-tenant-and-the-supervisor)
5. [Deploy the Tenants](#5-deploy-the-tenants)
6. [Submit and Observe](#6-submit-and-observe)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Cluster Prerequisites

Before deploying, ensure your environment meets the following requirements:

* Kubernetes **v1.34** or later (we validated on GKE, COS nodes) with 1-GPU
  nodes (H100-class recommended). The GPUs used for time-slicing must NOT be
  in use by anything else.
* GPU nodes must run **NVIDIA GPU Driver 565 or later**. This is a strict
  requirement to support **NVIDIA Dynamic Resource Allocation (DRA)**. The
  NVIDIA DRA driver itself ships with the platform Helm chart (§2).
* Cluster-admin access; `kubectl`, `helm` (v3), and `git` on your
  workstation.
* Nodes can pull your job images and reach GitHub + HuggingFace from pods
  (verl jobs typically fetch the model by HF id per node at startup — no
  shared volume needed).
* **Two 1-GPU nodes minimum**: one **shared trainer node** (RL head pod +
  batch server pod, one GPU between them) and one dedicated RL rollout node.

### Node Labeling and Time-Slice Groups

The orchestrator discovers resource pools (*groups*) from node labels. Jobs
in the same group take turns holding the group's accelerator lock. The
principle is that time-slicing machinery touches ONLY time-sliced nodes:
the **shared trainer node** carries all the time-slicing markers — the
group label (drives group discovery), the `timeslice.io/enabled` label
(the platform-wide marker for nodes participating in time-slicing — the
same convention the slime guide uses; in this recipe that is only the
trainer node — and what the platform install in §2 targets snapshot-agent
and DRA kubelet-plugin placement at), and the isolation taint (keeps
unrelated workloads off the shared GPU; the pods in this guide already
carry the matching toleration) — while the **rollout node carries nothing
timeslice-related**. Set up the trainer node with three commands:

```bash
kubectl label node <trainer-node> group.timeslice.io/trainers=true --overwrite
kubectl label node <trainer-node> timeslice.io/enabled=true --overwrite
kubectl taint node <trainer-node> timeslice.io/shared=true:NoSchedule --overwrite
```

Do NOT put either label on any other node, and do not put
`timeslice.io/*` pod labels on the rollout pod: only labeled pods on
labeled nodes are snapshot candidates, so the unlabeled rollout process is
never touched by the platform — while the trainer is checkpointed off the
shared GPU, the rollout keeps generating samples into the staleness queue.
The rollout node needs **no labels or taints at all** — the rollout pod
requests its GPU the ordinary way (§ Shared DRA Resource Claim below),
which is GPU access, not time-slicing. Placement follows from three
guarantees.

1. The RL head pod and the batch server pod co-locate on the trainer node
   because they consume the SAME shared `ResourceClaim`, and a claim's
   consumers must run where its device is. (If the trainer node lacks
   CPU/RAM headroom for the second tenant, that pod goes Pending; see the
   host-RAM sizing rule below.)
2. The rollout pod can never land on the trainer node: it carries a
   required node anti-affinity on `group.timeslice.io/trainers`.
3. The device plugin allocates whole GPUs exclusively, so rollout pods
   always land on distinct GPUs; on a 1-GPU-per-node topology that means
   distinct nodes. On multi-GPU nodes this becomes distinct GPUs, possibly
   on the same node.

On clusters with GPU nodes beyond the recipe's, rollouts may schedule onto
any free GPU node; operators running amid other workloads should scope
their GPU pool (e.g. with their own labels/taints).

### Shared DRA Resource Claim

Cooperative time-slicing leverages Kubernetes **Dynamic Resource Allocation
(DRA)** so multiple tenants' pods can share physical GPU hardware without
scheduler blocking — and without elevated pod permissions: the NVIDIA DRA
driver injects the GPU and driver userspace into ordinary pods. The recipe
contains exactly ONE `ResourceClaim` — the **shared** trainer claim —
because sharing is what only DRA can express: reference it from BOTH
tenants on the trainer node (the RL head pod and the batch server pod —
that is what lets them co-locate on the one GPU, each holding the full,
unpartitioned GPU during its turn). The rollout pod requests its GPU the
ordinary way (`nvidia.com/gpu: 1` via the device plugin) — it needs no
claim. (On clusters that manage GPUs purely via DRA, with no device plugin,
give the rollout pod its own `ExactCount: 1` claim instead — and since the
DRA kubelet plugin must then run on the rollout node too, drop or widen the
`kubeletPlugin` nodeSelector flag from the §2 install.)

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

Apply the claim before submitting any tenant (the example ships this exact
file as `examples/resource-claims.yaml`):

```bash
kubectl apply -f resource-claims.yaml
```

### Host-RAM Sizing Rule

Everything yielded on the shared node lands in host RAM, so size the trainer
node for BOTH tenants' offloads:

* The RL head pod's memory limit must fit its normal RSS **plus its entire
  GPU allocation** — the cuda-checkpoint dump of the trainer's full GPU state
  lands in pod memory.
* The batch server pod's memory limit must additionally fit its sleep
  offload: the engine's weights + KV cache move to host RAM at every yield
  (for reference, the example demo budgets 80 Gi/110 Gi for a 1.5 B trainer
  head and 60 Gi for the batch server's sleep offload, on a 234 Gi node).

---

## 2. Deploy the Time-Slicing Platform

Deploy the core platform components — **TimeSlice Orchestrator** (Deployment:
the gRPC lock service) and **Snapshot Agent** (DaemonSet on the time-sliced
nodes: performs each tenant's snapshot/restore when the orchestrator hands
the lock over) — using the parent Helm chart.

### Step 1: Clone the Repository

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing
```

> **Release pin:** the chart is not published to a registry — you install it
> from the clone. To pin a release, clone the repo at the release tag
> (`git clone --branch <version> https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git`)
> so the chart matches the pinned images, and pin the images with
> `--set timesliceorchestrator.image.tag=<version> --set snapshot-agent.image.tag=<version>`
> (the chart defaults pull the official
> `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` images at `latest`).
> Until the first tagged release is cut, `main` + `latest` is the only
> option. One constraint specific to this recipe: whatever you pin, the
> snapshot agent must support **per-job backend resolution** — the workload
> channel is how the agent drives the batch engine's sleep/wake instead of
> cuda-checkpointing it (the official images at `latest`, the chart default,
> do).

### Step 2: Install the Helm Chart

Install into a dedicated namespace (`timeslice-system`), pinning the
node-level platform components to `timeslice.io/enabled` nodes:

```bash
helm dependency update ./deploy
helm install timeslice ./deploy -n timeslice-system --create-namespace \
  --set-string "snapshot-agent.nodeSelector.timeslice\.io/enabled=true" \
  --set-string "nvidia-dra-driver-gpu.kubeletPlugin.nodeSelector.timeslice\.io/enabled=true"
```

The first flag pins the snapshot agent (a `hostPID` DaemonSet) to
`timeslice.io/enabled` nodes — in this recipe, just the trainer node;
agents on rollout nodes would perform no work anyway (all prior runs
confirmed this). The second flag pins the NVIDIA DRA driver's kubelet
plugin — the bundled driver's only node-level component, which publishes
`ResourceSlice`s and injects the GPU into claim-consuming pods — to the
same nodes: DRA is needed exactly where the shared claim binds, and gating
it keeps other GPU nodes free of `ResourceSlice`s advertising their GPUs
as claimable.

**Zero-impact principle:** you can share the cluster with non-timeslice
workloads with zero impact to them. This install plus the labels + taint
in §1 touch only the trainer node — the snapshot agent and the DRA kubelet
plugin run only there, and every other node carries no platform
components, labels, or taints, so workloads on those nodes are unaffected.
The recipe's rollout pod competes for free GPUs like any ordinary
workload: standard scheduling, no special machinery.

> If a `timeslice` release already exists, do a clean
> `helm uninstall timeslice -n timeslice-system` first — `helm install` fails
> on an existing release name. (Note: uninstalling also removes the bundled
> NVIDIA DRA driver DaemonSet until you reinstall.)

### Step 3: Verify Platform Health

Verify the orchestrator is Running, and that an agent pod and a
`nvidia-dra-driver-gpu-kubelet-plugin` pod run on the trainer node — and
only there; no other node should carry either:

```bash
kubectl -n timeslice-system get pods -o wide
```

After labeling the trainer node (§1), verify the orchestrator synced the
group:

```bash
kubectl -n timeslice-system logs deploy/timeslice-timesliceorchestrator --tail=50 \
  | grep -i trainers | tail -3
```

Optionally verify with the `rlts` CLI (requires Go toolchain; from the repo
root — the same binary is used to observe lock handoffs while tenants run):

```bash
go build -o bin/rlts ./cmd/rlts
./bin/rlts orchestrator list
```

No snapshot-device filter is needed in this topology: every node has one GPU,
and only the labeled pods on the trainer node are snapshot targets — the
rollout pod is unlabeled on another node.

---

## 3. Integrate Your RL Workload

### How It Works

The `llm-d-timeslice-verl` package ships `TimesliceFullyAsyncTrainer`, a
subclass of verl's `fully_async_policy` trainer that overrides the trainer's
empty `on_*` lifecycle hooks to acquire/release the orchestrator lock at the
trainer's natural wait points. It registers under the trainer name
`timeslice` (verl's `register_trainer` registry; the package's `verl.plugins`
entry point makes `import verl` load the registering module) and is selected
with a single hydra override: `async_training.trainer_name=timeslice`. No
monkey-patching and no verl source edits in your job — it requires a verl
build with the fully-async lifecycle hooks + trainer registry and per-pool
placement-group bundle resources (currently the
`feat/fully-async-lifecycle-hooks` fork branch, until the commits land
upstream).

Lock protocol:

| Trainer lifecycle point | Lock action |
|---|---|
| `init_workers` (model load) | ACQUIRE → load → YIELD |
| initial weight sync (pre-fit) | ACQUIRE → sync → YIELD |
| a batch of samples is ready | ACQUIRE (this is the resume point) |
| weight update + param sync done | YIELD |

### Placement-Group Pinning

Pin trainer/rollout placement with Ray **custom resources** — start ray on
the head with `--resources='{"trainer_node": 100}'`, on the worker with
`--resources='{"rollout_node": 100}'`, and pass verl's per-pool
placement-group bundle resources (hydra override
`ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool: {rollout_node: 1}}`)
so verl's trainer placement group requests `{trainer_node: 1}` and the
rollout PG `{rollout_node: 1}`.

### Installing verl (Pinned Fork)

In each job pod (or your job image), clone the
[`feat/fully-async-lifecycle-hooks`](https://github.com/aishukamal/verl/tree/feat/fully-async-lifecycle-hooks)
branch of `github.com/aishukamal/verl` — upstream
`983cb0f24443f87b3d161fad318445130a620b07` + two feature commits (fully-async
lifecycle hooks + trainer registry; per-pool PG bundle resources) — and
install it with `pip install -e ".[gpu]"`. This is a temporary fork until the
commits land upstream; once they do, point the install at mainline verl
instead (the example manifests parameterize this as the `VERL_REPO`/`VERL_REF`
pod envs).

### Installing the `llm-d-timeslice-verl` Package

Install the timeslice client and the integration package in every job pod
(or bake them into your job image):

```bash
# Install the timeslice client library
pip install "git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"

# Install the llm-d-timeslice-verl trainer package
pip install --no-cache-dir --no-deps "llm-d-timeslice-verl @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/integrations/verl"
```

> **Why `--no-deps`?** The package declares no dependencies of its own — the
> timeslice client it uses is installed separately (above), so there is
> nothing for pip to resolve.

### The Job Contract

Every time-sliced verl job must carry this contract:

* **Hydra overrides** on the verl launch command:
  * `async_training.trainer_name=timeslice` — selects the registered
    timeslice trainer.
  * `"ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool: {rollout_node: 1}}"`
    — pins the placement groups (see
    [Placement-Group Pinning](#placement-group-pinning)).
  * `trainer.save_freq=-1` — checkpoint saving is disabled under
    time-slicing: a save RPCs the FSDP worker, which may be checkpointed off
    the GPU at that moment. The timeslice trainer vetoes saves (with a
    warning) if this is left `> 0`. Note the consequence: time-sliced runs
    write no training checkpoints.
  * `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true`
    — tears down and rebuilds the trainer↔rollout weight-sync NCCL
    communicator around each sync, so no communicator exists outside
    lock-held windows.
  * `async_training.trigger_parameter_sync_step=1` — yield the GPU once per
    training step; larger values hold the lock across sample-queue waits
    (and starve the batch tenant for no benefit).
  * `async_training.use_dynamic_resource_scheduling=false` and
    `async_training.use_trainer_do_validate=false` — keep dynamic
    rescheduling and validation off the time-sliced trainer GPU (both would
    touch the GPU outside the lock protocol).
* **`ray start` custom resources**: head
  `--resources='{"trainer_node": 100}'`, rollout worker
  `--resources='{"rollout_node": 100}'`.
* **Environment variables** (set as container-level env vars on both pods,
  kept identical — the trainer driver actor is a CPU-only Ray actor that can
  be scheduled on either Ray node and inherits the raylet's env):
  * `TIMESLICE_FULLY_ASYNC=1` — activates the hooks (they are inert without
    it).
  * `TIMESLICE_JOB_ID` — unique per tenant; must equal the head pod's
    `timeslice.io/job-id` label.
  * `TIMESLICE_GROUP` — the group whose lock the trainer takes (e.g.
    `trainers`).
  * `TIMESLICE_ORCH_ADDR` — orchestrator gRPC address
    (`timeslice-timesliceorchestrator.timeslice-system.svc:50051`).
* **Pod labels** on the head pod ONLY (the trainer is the sole time-sliced
  process in the RL job; the rollout pod stays unlabeled):
  `timeslice.io/job-id: <job id>` and `timeslice.io/group: <group>` — the
  exact keys the platform selects snapshot candidates on.
* **NCCL environment**: set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` on
  every job pod (the example manifests set both) — precautionary settings
  around cuda-checkpoint; keep them as set.

> **See also:** the [verl integration guide](../rl-frameworks/verl/README.md)
> walks through this same contract in the setting where the second tenant is
> another RL trainer.

---

## 4. The Batch Tenant and the Supervisor

Everything in §3 is the standard time-sliced verl setup. The ONE component
this recipe adds is the batch tenant's side of the lock protocol: an RL
trainer yields at its natural wait points, but a serving engine has no
natural wait points — something must decide when it yields and how it
freezes. Two requirements, and they generalize to any batch engine:

1. **A cheap freeze/thaw path.** The engine should offload weights + KV
   cache to host RAM and restore them without a process restart or model
   reload. In vLLM this is **sleep mode** (`--enable-sleep-mode`, with
   `VLLM_SERVER_DEV_MODE=1` exposing the `/sleep`, `/wake_up`, `/is_sleeping`
   dev endpoints); other engines have equivalents (e.g. SGLang's
   release/resume memory occupation). An engine with no such API can still
   participate — the platform simply cuda-checkpoints the whole process like
   a trainer — but yields then cost tens of seconds instead of ~1–2 s, which
   the trainer eats at every handoff.
2. **A drain path.** The engine must stop admitting requests and finish (or
   fail fast) in-flight work BEFORE freezing. vLLM's `/sleep` does NOT drain
   the scheduler, and sleeping with a request mid-decode is a fatal CUDA
   error (vllm#28714; requests sent to a sleeping engine crash it too,
   vllm#15483).

### The Supervisor (Reference Implementation)

`examples/shadow-vllm.yaml` (EXAMPLE-ONLY, demo-quality) wraps **stock
`vllm serve`** with a ~100-line supervisor script that adapts it to the
platform's lock protocol. Whatever engine you run, your supervisor needs the
same four behaviors:

- **Priority by construction.** The supervisor acquires the group lock and
  holds it only while nobody else waits: every `WAITER_POLL_SECONDS`
  (default 0.5 s) it polls the orchestrator's waiter queue and yields as soon
  as the trainer queues. The trainer therefore never waits behind batch work
  for more than poll + drain + sleep (seconds), while the batch engine soaks
  up every idle minute.
- **Workload-channel registration.** At startup the supervisor registers
  sleep/wake callbacks with the node-local snapshot agent
  (`TIMESLICE_AGENT_ADDR`, the node IP :9001). The agent resolves each job's
  snapshot backend per job (explicit config → live workload channel → pod
  annotation → default cuda), so for this job it invokes the registered
  callbacks — `POST /sleep` / `POST /wake_up` — instead of
  cuda-checkpointing the engine process.
- **Drain before yield — required, not a nicety** (see requirement 2 above).
  Before releasing the lock the supervisor closes its readiness gate (the pod
  leaves the Service endpoints, so new batch requests fail fast) and waits
  until the engine's `running+waiting == 0` (~3 s) — only then `release()`.
- **Reopen after wake.** After reacquiring the lock and waking the engine,
  the supervisor reopens its readiness gate so the pod rejoins the Service.

A handoff, end to end:

1. The trainer's sample batch becomes ready → the timeslice trainer's hook
   calls `acquire()` and queues.
2. The supervisor's 0.5 s poll sees `waiter_queue_depth > 0` → it drains
   (above) → calls `release()`.
3. The orchestrator snapshots the batch job — through the registered
   workload channel, so the agent triggers `POST /sleep`.
4. The orchestrator restores the trainer (cuda-checkpoint) and grants the
   lock. The trainer runs its burst (minutes), then yields; the platform
   snapshots it and wakes the batch engine the same way in reverse; the
   supervisor reopens its readiness gate after the wake.
5. Batch requests sent while the engine is draining or asleep fail fast at
   the Service (connection refused — the pod is NotReady). Production
   clients should queue and retry — see §7.

### Swapping In Your Engine or Model

- **Bigger batch model**: anything that fits in the engine's GPU-memory
  fraction (the example's `VLLM_GPU_FRAC`, default 0.7) of the shared GPU
  alongside zero trainer residency (the two tenants never co-reside in HBM).
  Sleep/wake time scales with weight size (~1 s per 15 GB to host RAM) —
  re-check host-RAM sizing (§1) when you grow it.
- **Another engine**: keep the supervisor's loop and swap the three engine
  touchpoints — the freeze/thaw calls, the drain check (queue depths), and
  the health probe. Register the same way on the workload channel.
- **Yield latency vs. poll cost**: `WAITER_POLL_SECONDS` (default 0.5) is
  the worst-case extra wait the trainer sees before the supervisor notices
  it.

---

## 5. Deploy the Tenants

### The RL Job

Deploy the RL job as a **2-node Ray cluster**:

* **Head pod** — runs the verl driver + FSDP trainer. Schedule it on the
  shared, group-labeled trainer node
  (`nodeSelector: group.timeslice.io/trainers: "true"`), reference the
  **shared** `ResourceClaim`, and carry the `timeslice.io/job-id` +
  `timeslice.io/group` pod labels.
* **Rollout pod** — runs the rollout engine as a ray worker. Request its
  GPU the ordinary way (`resources.limits: nvidia.com/gpu: "1"` — the
  device plugin, no claim), and keep it off the trainer node (whose GPU is
  claim-managed) with a required node anti-affinity on
  `group.timeslice.io/trainers` (`operator: DoesNotExist`): it can then
  land on any GPU node except the trainer node, and since the device plugin
  allocates whole GPUs exclusively, it always gets a GPU of its own. Leave
  it unlabeled (no `timeslice.io/*` labels).
* **Headless Service** — pointing at the head pod, so the rollout pod can
  resolve the ray head address.
* **Tolerations** — give both pods tolerations for the GPU-node taints they
  must land on (the example tolerates `nvidia.com/gpu` and
  `timeslice.io/shared=true:NoSchedule`).

### The Batch Server Pod

* Schedule it on the SAME group-labeled trainer node and reference the SAME
  shared `ResourceClaim` as the RL head pod — the shared claim is what lets
  the two tenants co-locate on the one GPU.
* Give it its own `timeslice.io/job-id` label (unique — it is a full lock
  participant) and the same `timeslice.io/group` label as the RL head pod.
* Set `TIMESLICE_AGENT_ADDR` to the node-local snapshot agent (node IP,
  port 9001 — downward-API `status.hostIP` works) so the supervisor can
  register its workload channel, plus the same `TIMESLICE_ORCH_ADDR` /
  `TIMESLICE_JOB_ID` / `TIMESLICE_GROUP` envs the supervisor reads.
* Front it with a regular Service; the supervisor's readiness gate controls
  endpoint membership during drains.

### Launch Order

Start the **batch server first**, so it owns the GPU (and serves traffic)
throughout the RL job's long CPU-side setup; then the RL job (both pods at
once — the rollout pod's installs run in parallel with the head's); then
your batch client/load. A verl job typically spends 20–45 minutes on
installs, ray join, and model + dataset download before it first requests
the GPU — all of it CPU-side, all of it batch-serving time.

Placement needs no rollout node labels: the trainer-node tenants land where
their shared claim binds, and the rollout pod lands on any GPU node except
the trainer node (required anti-affinity) with an ordinary device-plugin
GPU (`nvidia.com/gpu: 1`). Neither path needs elevated pod permissions.

---

## 6. Submit and Observe

Apply the batch tenant, the RL job, and your load in the order above
(§5; the [example](examples/README.md) is the literal command sequence).

Grab the agent pod on the trainer node — its log shows every
snapshot/restore:

```bash
AGENT_POD=$(kubectl -n timeslice-system get pods -l app.kubernetes.io/name=snapshot-agent \
  --field-selector spec.nodeName=<trainer-node> -o jsonpath='{.items[0].metadata.name}')
echo "$AGENT_POD"
```

### Watch the Lock

Port-forward the orchestrator gRPC service and watch the shared pool with
the `rlts` CLI (built in §2):

```bash
kubectl port-forward svc/timeslice-timesliceorchestrator 50051:50051 -n timeslice-system
watch -n 0.5 ./bin/rlts orchestrator status trainers
```

**Expected output:** the `Active Job` and `Locking Job` alternate between
your RL job id and your batch job id. Most of the time the batch tenant
holds the lock (it harvests idle time); when the trainer's batch is ready,
the RL job id appears in the `Waiter Queue Depth` (depth = 1), the
supervisor drains and yields within seconds, and the lock flips to the
trainer for the training burst. The RL rollout never appears here — it takes
no locks.

### Watch the Logs

```bash
# lock handoffs: the integration package logs every acquire/release
# ([timeslice] ... ACQUIRE/RELEASE lines, forwarded into the head pod's log):
kubectl logs <rl-head-pod> --tail=2000 | grep -E "ACQUIRE|RELEASE" | tail -4
# [timeslice] job=<rl job id> ACQUIRE group=trainers waited=8210ms context_restored=True
# [timeslice] job=<rl job id> RELEASE group=trainers pending_waiters=1 snapshot_deferred=False

# supervisor handoffs on the batch pod:
kubectl logs <batch-pod> --tail=10
# [supervisor] trainer is waiting - yielding GPU
# [supervisor] drained in 2.82s
# [supervisor] vLLM slept (HBM -> host RAM) in 0.97s
# [supervisor] lock reacquired (waited 214380 ms, context_restored=True)
# [supervisor] vLLM woke in 0.05s

# snapshot/restore operations on the shared trainer GPU:
kubectl -n timeslice-system logs "$AGENT_POD" --tail=100 | grep -i "cuda-checkpoint"
```

### What Healthy Looks Like

Batch throughput **alternates by design**: full throughput in every
trainer-idle window, then a few-minutes-long window of fast failures during
each training burst (the pod is NotReady while drained/asleep — that is the
priority contract working, not an outage; see §7 for the client posture).
Healthy steady state: supervisor drain ≈ 3 s, sleep ≤ 3 s, wake ≤ 1 s;
trainer `waited=` on ACQUIRE in the seconds range (poll latency + drain +
engine sleep + the trainer's own cuda-checkpoint context restore — the
trainer never queues behind batch work).

Success criteria to check after ~3 training steps:

- Trainer step time within ~10% of a solo run (run the RL job alone for the
  baseline — with no contention the trainer acquires instantly and the
  platform defers snapshots, so it behaves like an unshared run).
- Batch throughput > 0 in every trainer-idle window.
- No cuda-checkpoint operations targeting the rollout node in any agent log
  (the rollout node must never show cuda-checkpoint activity).

---

## 7. Troubleshooting

Failures specific to the example's manifests are in the
[example's troubleshooting section](examples/README.md#7-troubleshooting).
Generic to the recipe:

| Symptom | Cause | Fix |
|---|---|---|
| Trainer placement group lands on the rollout node (or rollout work on the trainer node) | PG pinning not active — `--resources` missing on a `ray start` line, or the `ray_pg_extra_resources` hydra override is missing/typo'd, or the verl build lacks the feature | Verify both `ray start` lines carry `--resources`, keep the `ray_pg_extra_resources={trainer_pool: ...}` override intact, and use the pinned verl branch (§3); check `ray status` custom resources |
| Rollout throughput collapses when the trainer runs; agent log shows cuda-checkpoint activity on the ROLLOUT node | The rollout pod carries `timeslice.io/*` labels, or the rollout node is labeled into the group | Rollout pods/nodes must stay unlabeled — only the head pod, the batch pod, and the trainer node (§1) |
| The batch tenant never yields, or the trainer's restore fails with OOM on the shared GPU | The batch job's workload-channel registration didn't reach the agent, so the platform can't drive its sleep | Check the supervisor log for the registration and the agent log for the accepted channel; `TIMESLICE_AGENT_ADDR` must be the node-local agent (node IP :9001), not the orchestrator |
| Batch engine crashes with a CUDA error seconds after a sleep | It was frozen with requests in flight — the engine does not drain itself (vLLM: vllm#28714, vllm#15483) | Your supervisor must drain (stop admitting + in-flight = 0) BEFORE `release()`; never call the engine's freeze endpoint directly while it serves |
| Hydra error `Key 'use_trainer_do_validate' is not in struct trainer` | Wrong config key path | These flags live under `async_training.*`, not `trainer.*` |
| `save_freq` / checkpoint-save-skipped warning in the trainer log | Checkpoint saving would RPC a possibly-frozen worker | By design: the timeslice trainer vetoes saves when `save_freq > 0`; keep `trainer.save_freq=-1` |
| RL staleness queue near `staleness_threshold × batch` | Long trainer lock-out or very slow steps | Raise `async_training.staleness_threshold` or shorten training bursts; queue overflow drops samples |
| `Found multiple active Ray instances` warning on the shared node | Ray's `address='auto'` discovery found more than one GCS on the node | `export RAY_ADDRESS="$(hostname -i):6379"` right after `ray start --head` |

**The fail-fast client caveat**, worth stating to any consumer of the batch
Service: while the engine is draining or asleep, new batch requests fail
fast at the Service (the pod is NotReady). A demo client can simply count
those errors and retry later; a production batch front-end should be
queue-based with retries — e.g.
[llm-d async processor](https://github.com/llm-d/llm-d-async) as the batch
dispatcher — so that trainer bursts show up as latency, not failures.

---

For a complete pinned runnable of everything above — concrete models,
manifests, load generator, and the exact command sequence — see
**[examples/README.md](examples/README.md)**.
