# Async RL and async inference: lend an idle RL trainer GPU to async inference

**Summary.** In our H100 runs, the trainer GPU was in use **92%** of the time
with this platform (training, or lent to async inference), against **41%** on a
dedicated trainer GPU: 2.28x the use of the same GPU. The RL job converged at the
same step as on the dedicated GPU (step 140), and took 3.8% longer in
wall-clock to get there.

- **RL job:** verl fully-async GRPO, DeepSeek-R1-Distill-Qwen-1.5B on DAPO-Math,
  16K-token responses, 8 samples per prompt. One H100 for the trainer and one
  for rollout.
- **Async inference:** Qwen2.5-7B-Instruct on vLLM, 3.5 requests/s of
  1024-token answers through llm-d-async and the llm-d router, served only on
  the lent trainer GPU.
- **Platform:** GKE, nodes with one H100 each, with
  [application-aware suspend](#application-aware-suspend) on both the trainer
  and the guest.

Asynchronous RL jobs (for example verl's fully-async mode) keep the rollout GPUs
busy all the time, but the trainer GPU often sits idle while it waits for the
next batch of samples. This is especially true for long chain-of-thought
models, where the trainer can be idle for minutes in every step.

This platform lends that idle trainer GPU to your async inference servers and
takes it back the moment the trainer needs it. The trainer always wins:

1. When the trainer starts waiting for samples, its GPU state is checkpointed to
   host memory and the GPU is lent to an async inference server (a "guest").
2. When the trainer is about to train again, the guest is frozen in place and
   the trainer's GPU state is restored. Requests still running on the guest are
   cut off just before the freeze, so the caller retries them on another server
   instead of waiting for the next lend.

You do not change your training code, your model server, or your async
inference pipeline. The whole setup is:

| Who | What |
|---|---|
| Platform team | `helm install timeslice` |
| RL team | the prebuilt verl image, one label and three settings in the RayJob |
| Async inference team | one label on the model server pods |

Each team does its part on its own. Once the platform team has installed the
platform, the RL team and the async inference team can do their parts in any
order, or at the same time; neither waits for the other.

Sections 1 to 4 are all you need to run. [Section 7](#7-optional-settings)
lists optional settings.

## Contents

1. [Requirements](#1-requirements)
2. [Platform team: install the platform](#2-platform-team-install-the-platform)
3. [RL team: change the RayJob](#3-rl-team-change-the-rayjob)
4. [Async inference team: label the model server](#4-async-inference-team-label-the-model-server)
5. [See the lending in action](#5-see-the-lending-in-action)
6. [What the platform adds to your pods](#6-what-the-platform-adds-to-your-pods)
7. [Optional settings](#7-optional-settings)
8. [Troubleshooting](#8-troubleshooting)
9. [Limits](#9-limits)

## 1. Requirements

This guide targets RL workloads that use verl's fully-async mode, with vLLM as
the model server for async inference.

- A Kubernetes cluster with GPUs.
- KubeRay, if your RL jobs are RayJobs.
- A disaggregated RL setup: trainers and rollout workers (samplers) on
  different nodes.
- Host memory: the trainer's GPU memory is checkpointed into its pod's memory.
  The platform raises the trainer pod's memory limit by one GPU's memory per
  GPU (requests are unchanged), so the node needs that much free memory on top
  of the trainer's normal use.
- The RL image: the prebuilt verl image
  (`ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl`, about 11 GB to
  pull). To keep your own verl image instead, it needs Python 3.10 or later
  with `grpcio` 1.66 or later and `protobuf` 6 or later (stock `verlai/verl`
  images have them); see [Keep your own verl image](#keep-your-own-verl-image).
- The async inference server: anything that serves over HTTP and tolerates
  pauses of a few minutes. Put a queue in front of it (for example
  [llm-d-async](https://github.com/llm-d/llm-d-async)) so requests wait instead
  of failing.

## 2. Platform team: install the platform

```bash
git clone --branch demo/batch https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing/deploy
helm dependency update .
helm install timeslice . -n timeslice-system --create-namespace
```

That is all. Optional settings (pinning the image tag, limiting namespaces,
keeping your own verl image) are in [section 7](#7-optional-settings).

To check that the platform is up:

```bash
kubectl -n timeslice-system get pods -o wide
```

You should see the orchestrator, the webhook (2 replicas), the donor
controller, and on GPU nodes the snapshot agent and the shadow device plugin.
The guest kubelet starts on a trainer node once a trainer pod runs there.

## 3. RL team: change the RayJob

Start from your normal RayJob. The trainer worker group and the rollout worker
group must be separate worker groups, with the trainer group on its own GPU
nodes. Switch the image to the prebuilt verl image
(`ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl:demo-batch`, which has
verl and its fully-async dependencies, so drop any `pip install` of verl at
start), then make these changes:

1. **Label the trainer worker group** `timeslice.io/donor: "true"`. This is the
   group whose GPUs are lent.
2. **Pin Ray placement** with one `resources` line per worker group, so verl's
   trainer and rollout placement groups land on the right pods.
3. **Two verl settings**: `async_training.trainer_name=timeslice` (the
   time-slicing trainer, a subclass of verl's fully-async trainer) and
   `ray_pg_extra_resources=...` (the custom resources from change 2).
4. **Only with the default `cuda-checkpoint` hand-over**: set
   `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` in every pod of the job, since
   `cuda-checkpoint` cannot save NCCL's cuMem and NVLS buffers. An
   [application-aware](#application-aware-suspend) trainer needs neither.

The diff on a typical fully-async RayJob (1 trainer GPU + 1 rollout GPU; your
group names, resources and node pools will differ):

```diff
@@ entrypoint @@
     async_training.staleness_threshold=8
     async_training.trigger_parameter_sync_step=1
+    async_training.trainer_name=timeslice
+    'ray_pg_extra_resources={trainer_pool:{trainer_node:1},rollout_pool:{rollout_node:1}}'
@@ headGroupSpec @@
           containers:
             - name: ray-head
-              image: verlai/verl:vllm020.dev2
+              image: ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl:demo-batch
@@ workerGroupSpecs: trainer @@
       - groupName: trainer
         replicas: 1
         rayStartParams:
           num-cpus: "20"
           num-gpus: "1"
+          resources: "'{\"trainer_node\": 100}'"
         template:
+          metadata:
+            labels: {timeslice.io/donor: "true"}
           spec:
             nodeSelector: {cloud.google.com/gke-nodepool: h100-trainer-pool}   # your trainer pool
             containers:
               - name: ray-worker
-                image: verlai/verl:vllm020.dev2
+                image: ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl:demo-batch
@@ workerGroupSpecs: rollout @@
       - groupName: rollout
         replicas: 1
         rayStartParams:
           num-cpus: "20"
           num-gpus: "1"
+          resources: "'{\"rollout_node\": 100}'"
         template:
           spec:
             nodeSelector: {cloud.google.com/gke-nodepool: h100-rollout-pool}   # your rollout pool
             containers:
               - name: ray-worker
-                image: verlai/verl:vllm020.dev2
+                image: ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/verl:demo-batch
```

Nothing else changes: same command, same data, same resources. Apply the
RayJob as usual.

To check that the job is wired:

```bash
kubectl get pod -l ray.io/group=trainer -o jsonpath='{.items[0].metadata.annotations}' | tr , '\n' | grep timeslice
kubectl get nodes -l type=virtual-kubelet
```

The trainer pod carries `timeslice.io/donor-memory-raised`, and a virtual node
appears for the trainer node. The job's log shows the time-slicing trainer:

```bash
kubectl logs <rayjob submitter pod> | grep -i timeslice | head
```

### Keep your own verl image

This is optional and off by default. If the RL team must keep its own verl
image (its own dependencies, a different CUDA or vLLM version, a private base
image), the webhook can inject the verl fork and the timeslice packages into
the Ray pods instead.

1. Install the platform with the overlay that turns the injection on:

   ```bash
   helm install timeslice . -n timeslice-system --create-namespace \
     -f values-inject-verl.yaml
   ```

2. Keep your image (and any start-up `pip install`) and make only the three
   changes above.

With the overlay, the webhook adds an init container to every pod of a Ray
cluster that has a donor group (head, trainer and rollout workers). The init
container copies these packages into a read-only volume, and `PYTHONPATH`
points at it:

- the timeslice client and the verl plugin (`trainer_name=timeslice`);
- **a verl fork**: verl at commit `983cb0f2` plus two commits that add
  lifecycle hooks to the fully-async trainer (source:
  `github.com/aishukamal/verl`, branch `feat/fully-async-lifecycle-hooks`,
  commit `976b5f57`), the same pinned commit as in the prebuilt image. It
  ships with its Apache-2.0 LICENSE and replaces the image's own verl in those
  pods only. verl's dependencies still come from your image. If your image
  already has a verl with these hooks, add
  `--set timeslice-webhook.rlIntegration.injectVerl=false` to inject only the
  timeslice packages.

If your RL pods are not created by KubeRay, label every non-trainer pod of the
job `timeslice.io/rl-integration: "true"` so they get the same packages (for
KubeRay the webhook does this for you).

## 4. Async inference team: label the model server

This step does not depend on the RL team. You can label the model server
before, during or after RL jobs run: guests use trainer GPUs whenever one is
lent.

Add `timeslice.io/guest: "true"` to the model server's pod labels. Keep asking
for `nvidia.com/gpu` as usual.

```diff
   template:
     metadata:
       labels:
         app: guest-vllm
+        timeslice.io/guest: "true"
     spec:
       containers:
         - name: vllm
           image: vllm/vllm-openai:v0.9.2
```

Guests prefer lent GPUs. When no trainer GPU is lent, a new guest pod runs on a
regular GPU node if one has room, and stays `Pending` otherwise. The node is
picked once, when the pod is scheduled: a guest on a regular node stays there,
and a guest on a lent GPU is frozen while the trainer has it back. If your
cluster autoscales GPU nodes, a `Pending` guest can add a node. To keep guests
on lent GPUs only, install with
`--set timeslice-webhook.flags.guest-steering=required`.

### If you use llm-d-async and the llm-d router

Install [llm-d-async](https://github.com/llm-d/llm-d-async) with its own Helm
chart, unchanged; it needs no time-slicing settings. Label only the model
server pods, not the async processor.

When a guest is reclaimed, the requests running on it are cut off before it is
frozen: the snapshot agent resets their connections. The router answers a
request cut off before its response started with `503` (Envoy's
`upstream_reset_before_response_started`), which the async processor retries.
The async processor sends non-streamed requests, so this covers it. A streamed
response that had already started ends early instead, still with status `200`
but without the final `data: [DONE]` line: a client that streams must treat a
stream without `[DONE]` as failed and retry it. No request waits for a frozen
guest, so neither timeout depends on how long a trainer update takes. Check
both against your longest single generation instead:

- **Router route timeout**: Envoy's default route timeout is 15 s, shorter
  than many generations. Set the route `timeout` above your longest single
  generation (or `0s`, no limit). The route `idle_timeout` can stay at its
  default (5 minutes) unless one non-streamed generation takes longer.
- **Async processor request timeout**: the default (5 minutes) is enough as
  long as it stays above your longest single generation.

## 5. See the lending in action

```bash
kubectl get pods -l timeslice.io/guest=true -o wide -w
kubectl -n timeslice-system logs ds/timeslice-guest-kubelet -f | grep 'host command'
```

Each trainer step shows one lend (the guest is resumed or starts, then becomes
Ready) and one reclaim (the guest is suspended, the trainer is restored). To see
the trainer GPU's utilization go up, use your cluster's usual GPU metrics (on
GKE, the GPU metrics in Cloud Monitoring).

## 6. What the platform adds to your pods

The admission webhook changes only pods and Ray clusters that opted in with the
labels above. It adds:

- **Into every pod of a Ray cluster that has a donor group** (head, trainer and
  rollout workers): `TIMESLICE_FULLY_ASYNC=1` and the job's identity and
  orchestrator address (`TIMESLICE_JOB_ID`, `TIMESLICE_GROUP`,
  `TIMESLICE_ORCH_ADDR`). Values you set yourself are kept.
- **Into the trainer pods**: a higher memory limit (one GPU's memory per GPU,
  for the checkpoint) and the annotation `timeslice.io/donor-memory-raised`.
- **Into the guest pods**: a preference for the virtual nodes that represent
  lent GPUs.
- **Into every pod of a Ray cluster whose donor group opted into
  application-aware parking** (see
  [Application-aware suspend](#application-aware-suspend)):
  `TIMESLICE_DONOR_BACKEND=app_channel`, `TIMESLICE_AGENT_PORT` and
  `TIMESLICE_NODE_IP` (the pod's host IP), so the trainer can reach the
  node's snapshot agent.

The verl plugin also defaults one verl setting the lend cycle needs
(`rebuild_group=True` for the NCCL weight sync, so the sync group is rebuilt
after each restore). An explicit setting of yours wins.

As soon as a guest is admitted, the platform starts pulling its images onto the
trainer node, so the first lend usually finds them there. A large image on a
node that has never pulled it can still delay the first lend (100 s for the
vLLM image in our H100 runs; 2 s once cached).

## 7. Optional settings

None of these is needed to run. Use them to pin versions, mirror images or
limit the platform's scope.

| Setting | Who | Use it to |
|---|---|---|
| `global.imageTag` (Helm value) | Platform team | Pin the images to the commit you cloned (`demo-<short sha>`) or a version. The default is `demo-batch`, the newest release of the `demo/batch` branch. |
| Image references (`<component>.image.repository`, and `timeslice-webhook.rlIntegration.image.repository`) | Platform team | Pull the images from your own registry or from images you built yourself. |
| `timeslice-webhook.namespaceSelector` (Helm value) | Platform team | Limit the platform to some namespaces. By default it watches every namespace except `kube-system` and `timeslice-system`. |
| `-f values-inject-verl.yaml` | Platform team | Keep the RL team's own verl image instead of the prebuilt one; see [Keep your own verl image](#keep-your-own-verl-image). |
| `timeslice.io/backend: app_channel` (trainer pod annotation) | RL team | Let the trainer park itself instead of being checkpointed; see [Application-aware suspend](#application-aware-suspend). |
| `timeslice.io/backend: app_endpoint` (guest pod annotation) | Async inference team | Put a vLLM guest to sleep through its own API instead of checkpointing it; see [Application-aware suspend](#application-aware-suspend). |
| The `timeslice` Python client in your trainer | RL team | Use an RL framework other than verl; see [Bring your own RL framework](#bring-your-own-rl-framework). |

### Application-aware suspend

By default the platform checkpoints the whole GPU state of the trainer and of
the guest with `cuda-checkpoint`. Either side can instead suspend through its
own application, which moves only what matters and is much faster: in our H100
runs, with both sides application-aware, the trainer got its GPU back about 4 s
after asking, instead of about 22 s with `cuda-checkpoint` on both sides. Each
side opts in on its own, with one annotation.

**Trainer (RL team).** Annotate the trainer worker group:

```diff
@@ workerGroupSpecs: trainer @@
         template:
           metadata:
             labels: {timeslice.io/donor: "true"}
+            annotations: {timeslice.io/backend: app_channel}
```

The verl plugin then registers with the node's snapshot agent. When the GPU is
lent, the trainer moves its model and optimizer state to host memory through
verl's offload control and back when it trains again. The GPU is lent only if
the trainer keeps at most 4 GiB of GPU memory. No other change is needed; the
trainer still needs host memory for its model and optimizer state.

**Guest (async inference team).** Annotate a vLLM guest and start vLLM with sleep mode:

```diff
   template:
     metadata:
       labels:
         app: guest-vllm
         timeslice.io/guest: "true"
+      annotations:
+        timeslice.io/backend: app_endpoint
     spec:
       containers:
         - name: vllm
           image: vllm/vllm-openai:v0.9.2
+          args: [..., "--enable-sleep-mode"]
+          env:
+            - {name: VLLM_SERVER_DEV_MODE, value: "1"}
```

At a reclaim the snapshot agent cuts off the guest's running requests as
before, then calls vLLM's `/sleep` (weights to host memory, KV cache dropped)
and freezes the guest (it may keep at most 4 GiB of GPU memory); at a lend it
thaws the guest and calls `/wake_up`. The agent finds vLLM on the pod IP and
its first serving port; to point it elsewhere, add the annotation
`timeslice.io/backend-config: '{"app": "APP_VLLM", "endpoints": ["http://localhost:8000"]}'`.
Only vLLM is supported for guests.

### Bring your own RL framework

The prebuilt image ships a verl plugin. Any other framework can use the platform
through the `timeslice` Python client
([`pkg/client/python`](../../pkg/client/python)), in two parts:

1. **Hold the GPU lock while the trainer uses the GPU** (required). This tells
   the platform when the GPU can be lent.
2. **Park the trainer yourself** (optional). This replaces `cuda-checkpoint`
   with [application-aware suspend](#application-aware-suspend).

The verl plugin is a full example of both parts. These fork commits show the
work:

| Commit | What it adds |
|---|---|
| [`9a69ffa`](https://github.com/Edwinhr716/llm-d-rl-time-slicing/commit/9a69ffabbefd2b389240e23cffe4b1334a2ef58d) | The verl plugin with the GPU lock: [`locks.py`](../../pkg/integrations/verl/timeslice_verl/locks.py) wraps the client, [`hooks.py`](../../pkg/integrations/verl/timeslice_verl/hooks.py) maps verl's trainer hooks to acquire and release, and [`trainer.py`](../../pkg/integrations/verl/timeslice_verl/trainer.py) registers the trainer. |
| [`cc29903`](https://github.com/Edwinhr716/llm-d-rl-time-slicing/commit/cc29903a4e99d1d8d8d2d63419dfa0b3dbba838e) | Application-aware parking of the verl trainer: [`app_aware.py`](../../pkg/integrations/verl/timeslice_verl/app_aware.py), and the webhook settings that point the trainer at the node's snapshot agent. |
| [`dd3fc55`](https://github.com/Edwinhr716/llm-d-rl-time-slicing/commit/dd3fc554c77f4968c40e25e6b2d895839bfd2d74) | The same GPU lock for slime, in about 60 lines ([`callback.py`](../../pkg/integrations/slime/timeslice_slime/callback.py)). |

**Install the client** in your trainer image:

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing@demo/batch#subdirectory=pkg/client/python"
```

**Label the trainer pods** `timeslice.io/donor: "true"`, as in
[section 3](#3-rl-team-change-the-rayjob). Outside KubeRay, also give them a
job identity: `timeslice.io/job-id` (unique per job) and `timeslice.io/group`
(the name of the job's GPU lock, unique per job too). The webhook then sets
`TIMESLICE_JOB_ID`, `TIMESLICE_GROUP` and `TIMESLICE_ORCH_ADDR` in the trainer
pods (in a Ray cluster, in every pod of the cluster).

**1. Hold the GPU lock.** Acquire before the trainer touches the GPU, and
release while it waits for samples:

```python
import os
from contextlib import contextmanager
from timeslice import TimeSliceOrchestratorClient

lock = TimeSliceOrchestratorClient(
    target=os.environ["TIMESLICE_ORCH_ADDR"],
    job_id=os.environ["TIMESLICE_JOB_ID"],
    group_id=os.environ["TIMESLICE_GROUP"],
)

@contextmanager
def gpu():
    lock.acquire(timeout_sec=600)  # returns once the trainer is back on the GPU
    try:
        yield
    finally:
        lock.release()             # the GPU can be lent from here, also after an error

with gpu():
    init_workers_and_load_model()

while training:
    batch = wait_for_samples()     # CPU only: the GPU is lent meanwhile
    with gpu():
        train_step(batch)
        sync_weights_to_rollout()
        save_checkpoint_if_due()   # save while you still hold the lock
```

- **Every GPU use of the trainer goes between `acquire` and `release`**,
  including weight syncs and checkpoint saves and loads. After `release` the
  trainer's GPU state may be checkpointed off the GPU or parked.
- **Wait for samples with the lock released.** That wait is the time the GPU is
  lent.
- **Let an `acquire` error stop the job.** A trainer must not train without the
  lock.
- **Release on every exit path,** errors included. A job that dies holding the
  lock keeps its GPU from being lent.
- **The default install lends on every `release`.** If the orchestrator runs
  with `--lend-policy=hint`, pass `expected_idle=<seconds>` to `release`; the
  GPU is lent only when that is at least `--min-bubble`.
- **With the default `cuda-checkpoint` hand-over**, set `NCCL_CUMEM_ENABLE=0`
  and `NCCL_NVLS_ENABLE=0` in every pod of the job (see
  [section 3](#3-rl-team-change-the-rayjob)).

[`locks.py`](../../pkg/integrations/verl/timeslice_verl/locks.py) wraps this in
an idempotent helper that does nothing when the `TIMESLICE_*` settings are
missing, so the same image runs with and without the platform.

**2. Park the trainer yourself (optional).** Register a park and restore
workload with the node's snapshot agent, and annotate the trainer pods
`timeslice.io/backend: app_channel`. The agent calls `snapshot` before it lends
the GPU and `restore` before it grants the trainer's next `acquire`:

```python
import os
from timeslice.snapshot_agent.workload import register_workload

class TrainerPark:
    def snapshot(self, mode, tags):    # before the GPU is lent; mode is "offload"
        if trainer_holds_lock():
            raise RuntimeError("park refused: the trainer is mid-step")
        move_model_and_optimizer_to("cpu")
        torch.cuda.empty_cache()

    def restore(self, tags):           # before the trainer's next acquire returns
        move_model_and_optimizer_to("cuda")

park = register_workload(
    agent=f"{os.environ['TIMESLICE_NODE_IP']}:{os.environ.get('TIMESLICE_AGENT_PORT', '9001')}",
    job_id=os.environ["TIMESLICE_JOB_ID"],
    group=os.environ["TIMESLICE_GROUP"],
    workload=TrainerPark(),
    supported_modes=["offload"],
)
```

- **Register once**, after the model is loaded and before the first `release`,
  from a process that can move the trainer's state. If the trainer spans several
  nodes, register with the agent of each of them. The client reconnects and
  registers again on its own if the agent restarts.
- **Keep at most 4 GiB of GPU memory after a park.** The agent checks it, and
  does not lend the GPU if the trainer keeps more.
- **Refuse a park while the trainer holds the lock.** The agent then does not
  lend the GPU.
- **After each `acquire`, make sure the state is back on the GPU,** and restore
  it yourself if not: a restore can be lost if the agent restarts while the
  trainer is parked.
- **The agent address:** the agent runs on the host network of each GPU node.
  In a Ray cluster, the webhook sets `TIMESLICE_NODE_IP` and
  `TIMESLICE_AGENT_PORT` for you. Outside KubeRay, set `TIMESLICE_NODE_IP` from
  the downward API (`status.hostIP`); the agent's port is 9001 unless the
  platform team changed it.

[`app_aware.py`](../../pkg/integrations/verl/timeslice_verl/app_aware.py) does
all of this for verl: it parks the optimizer and then the model, frees the
allocator caches, and checks that the trainer is resident after every
`acquire`.

## 8. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Platform pods in `ImagePullBackOff` or `ErrImagePull` | No image with that tag: CI has not published it yet (the first release of `demo/batch`, or a pinned commit whose release run has not finished or failed), or the image is not public yet | Wait for the release workflow run (Actions, "CI - Release"); if you pinned `global.imageTag`, remove the pin to use `demo-batch` |
| The RayJob's pods are not created; an admission error names the webhook | The webhook is not ready yet | Wait until both webhook replicas are Ready, then re-apply |
| No virtual node appears | More than one pod holds `nvidia.com/gpu` on the trainer node, or the trainer pod lacks the donor label | Keep only the trainer pod on that node; check the label on the pod template |
| The job fails at start with an import error for `timeslice_verl` | The pod was created before the platform was installed, or outside a labelled Ray cluster | Re-create the RayJob after installing the platform |
| Guest stays `Pending` | No trainer GPU is lent yet (the trainer is busy or still starting), and no regular GPU node has room | Wait for the first trainer wait; check the virtual node exists |
| The trainer log shows a park failing with `VERIFY_FAILED`, and its GPU is not lent | An application-aware trainer kept more than 4 GiB of GPU memory after parking; the error lists each process's GPU memory | Report it with the trainer and snapshot agent logs; remove the `app_channel` annotation to fall back to `cuda-checkpoint` |
| An `app_endpoint` guest fails to suspend | vLLM was started without `--enable-sleep-mode` or `VLLM_SERVER_DEV_MODE=1`, or the agent cannot reach it | Add both; check the snapshot agent log for the endpoint it called |
| Trainer pod OOM-killed | The node lacks host memory for the checkpoint | Use nodes with more memory, or fewer GPUs per trainer pod |
| Async inference requests fail with 504 or time out | The router's route timeout is shorter than a generation | Set the route timeout as in section 4 |
| The router logs `503` with `upstream_reset_before_response_started` around each reclaim | Requests running on a guest were cut off before its freeze | Expected: the async processor retries them; the snapshot agent logs `aborted in-flight connections` with the count |
| Steps get slower than without time-slicing | The reclaim is slow (large guest or checkpoint) | See the guest kubelet log for `agent call done` durations |
| The job fails at start with `The grpc package installed is at version ...` | The RL image's `grpcio` is older than 1.66 | Use an image with `grpcio` 1.66 or later |
| Leftover `NotReady` virtual nodes (`kubectl get nodes -l type=virtual-kubelet`) after RL jobs end | A virtual node is deleted about a minute after its era ends (the donor controller's era TTL, 15 minutes by default, after the trainer pod ends). One that stays was kept for a restart: its guest kubelet stopped while it could not read its DaemonSet (an older chart, or the RBAC was removed first during `helm uninstall`) | When no RL job runs on that trainer node: `kubectl delete node <name>`, then `kubectl patch node <name> --type=merge -p '{"metadata":{"finalizers":null}}'` |
| A guest pod stays `Terminating` on a `NotReady` virtual node | Its guest kubelet went away without releasing the node (a crash, or the case above). When it releases a node it force-deletes the guests left there | `kubectl delete pod <name> --grace-period=0 --force` |
| The first lend after a guest is admitted takes about a minute longer | The guest image is still being pulled onto the trainer node | Expected once per node; later lends find the image cached |
| A guest stays `Pending` on a virtual node whose trainer pod has ended | The guest kubelet cordons a virtual node as soon as its trainer pod ends, and 10 s later deletes the `Pending` guests there that have a controller, which re-creates them on another node. A bare pod is not deleted; it gets a `StrandedNoDonor` event | Delete the bare pod, or run guests from a Deployment |

## 9. Limits

- One trainer pod per trainer node, and that node runs no other GPU pods.
- While a GPU is lent, its guests have it to themselves (the trainer's state is
  in host memory), so a guest's GPU memory must fit in that GPU.
- Requests running on a guest when it is reclaimed are cut off and retried
  from the start; the tokens they had generated are lost. Long generations
  waste the most. A client that does not retry `503`, or a streaming client
  that accepts a stream without `[DONE]`, loses those requests.
- The verl plugin targets verl's fully-async mode.
- Application-aware suspend supports verl trainers and vLLM guests, and only
  offloads to host memory (no discard). The backend is chosen per pod by
  annotation.
- A guest on a lent GPU is not moved while the trainer has the GPU back: it
  stays frozen, and requests routed only to it wait. When the RL job ends, a
  guest that has not started yet is deleted, and its Deployment re-creates it on
  a regular GPU node if one has room; a guest that has already started stays on
  the virtual node until you delete it. Keep a few regular replicas behind the
  same queue for work that must always progress.
- A guest that is reclaimed before it has ever served is deleted and
  re-created, not frozen, so its next lend is a cold start.
