# Interleaving RL Training with Batch Inference on a Shared GPU

This guide time-slices **one verl fully-async RL training job** with a
**stock vLLM batch-inference server** on the same GPU. The RL trainer has
absolute priority; vLLM harvests the trainer's idle valleys (sample waits
between training bursts — 40–70% of wall-clock for async RL workloads) to
serve latency-tolerant batch traffic. Neither workload's code is modified:
verl is integrated through its fully-async lifecycle hooks: the
`timeslice-verl` package from this repo ships a FullyAsyncTrainer subclass
registered under the trainer name `timeslice`
(`async_training.trainer_name=timeslice`), and vLLM runs its normal
OpenAI-compatible server wrapped by a ~100-line supervisor (shipped in
`examples/shadow-vllm.yaml` — demo-quality, example-only).

What makes the batch side cheap to yield: vLLM's native **sleep mode** moves
weights + KV cache to host RAM in ~1–2 s and back in ~100 ms — no process
restart, no model reload. The RL trainer side uses `cuda-checkpoint`
(~10–35 s per swap for a 1.5 B trainer), which is amortized against training
bursts that run for minutes.

**Topology** (two 1-GPU nodes; H100-class):

```
  ROLLOUT NODE (dedicated)             TRAINER NODE (SHARED, time-sliced)
  ┌──────────────────────────┐         ┌──────────────────────────────────┐
  │ rl-rollout               │  ray    │ rl-head          shadow-vllm     │
  │ RL rollout engine (vLLM, │◀───────▶│ RL trainer   ⇄   vLLM server     │
  │ generates continuously)  │         │ cuda-checkpoint    sleep mode    │
  └──────────────────────────┘         │      C/R           (~1-2 s swaps)│
                                       └──────────────────────────────────┘
  lock owner over time on the trainer node's GPU:
  vLLM ████████░T██████████░T█████████░T████████   (T = training burst)
                ▲ trainer queues → vLLM yields in ≤ poll(0.5s) + drain(~3s) + sleep(~1-2s)
```

The RL job is a **2-node Ray cluster**: head pod `rl-head` (verl driver +
FSDP trainer) on the shared trainer node, rollout pod `rl-rollout` (vLLM
rollout engine) on a dedicated node. Placement is pinned with Ray custom
resources (`trainer_node` / `rollout_node`) through verl's per-pool
placement-group bundle resources (hydra override `ray_pg_extra_resources`).

## 1. Components

| Piece | What it is | Where it runs |
|---|---|---|
| TimeSlice Orchestrator | gRPC group-lock service | `timeslice-system` ns |
| Snapshot Agent | DaemonSet; executes cuda-checkpoint C/R AND relays sleep/wake to registered apps (workload channel) | every GPU node |
| RL job (`examples/rl-job.yaml`) | verl `fully_async_policy` code-RLVR training (Eurus-2 code split, rewards = live test execution), integrated via verl lifecycle hooks (trainer subclass registered as `timeslice`); head pod + rollout pod + headless Service | `default` ns |
| Shadow vLLM (`examples/shadow-vllm.yaml`) | EXAMPLE-ONLY: stock `vllm serve` + supervisor: holds the lock only while nobody waits; registers sleep/wake callbacks with the agent | `default` ns, trainer node |
| Load generator (`examples/load-generator.yaml`) | EXAMPLE-ONLY: continuous `/v1/completions` client + throughput log | `default` ns |

How a handoff works, end to end:

1. Trainer's sample batch becomes ready → the timeslice trainer's hook calls
   `acquire()`.
2. Supervisor's 0.5 s poll sees `waiter_queue_depth > 0` → it **drains**
   (closes its readiness gate, so the pod leaves the Service endpoints and
   new batch requests fail fast; waits ~3 s for in-flight requests to
   finish — vLLM's `/sleep` does NOT drain the scheduler, and sleeping with
   a request mid-decode is a fatal CUDA error, vllm#28714) → calls
   `release()`.
3. Orchestrator snapshots the vLLM job — the agent's per-job backend
   resolution (explicit config → live workload channel → pod annotation →
   default cuda) lands on its **registered workload channel** → the agent
   invokes the supervisor's callback → `POST /sleep` → HBM freed in ~1–2 s.
4. Orchestrator restores the trainer (cuda-checkpoint, trainer node only) and
   grants the lock. Trainer runs its burst (minutes), then yields; the
   platform snapshots it and wakes vLLM the same way in reverse; the
   supervisor reopens its readiness gate after the wake.
5. Batch requests sent while vLLM is draining or asleep fail fast at the
   Service (connection refused — the pod is NotReady); the demo client
   counts them (`errors_or_asleep`) and carries on. Production clients
   should queue and retry — see §8.

## 2. Prerequisites

- Kubernetes cluster (validated on GKE/COS) with **two free 1-GPU nodes**
  (H100-class recommended; reference shape GKE `a3-highgpu-1g`, 26 vCPU /
  234 Gi each): one shared trainer node (RL head pod + shadow vLLM pod) and
  one dedicated RL rollout node.
- Cluster-admin; `kubectl`, `helm` v3, `git`, `envsubst` on your workstation.
- Pods can reach Docker Hub, GitHub, HuggingFace. No HF token needed (models
  are fetched by HF id per node).
- Host RAM on the trainer node: the RL head pod requests 80 Gi (110 Gi
  limit; trainer checkpoint ≈ full GPU allocation) + 60 Gi for the shadow
  vLLM's sleep offload.

**Version pins** (keep on first run): verl branch
[`feat/fully-async-lifecycle-hooks`](https://github.com/aishukamal/verl/tree/feat/fully-async-lifecycle-hooks)
of `github.com/aishukamal/verl` — upstream
`983cb0f24443f87b3d161fad318445130a620b07` plus two feature commits
(fully-async lifecycle hooks + trainer registry; per-pool PG bundle
resources); temporary fork until the commits land upstream. Job image
`verlai/verl:vllm020.dev2`; shadow vLLM image `vllm/vllm-openai:v0.9.2`;
platform images: official `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*`
containing per-job backend resolution in the
agent (config-less requests resolve: explicit config → live workload
channel → `timeslice.io/backend` pod annotation → default cuda; the
workload-channel step is required by this guide — see the release-pin
note in §3);
integration package `timeslice-verl` from this repo,
`pkg/integrations/verl/` (ConfigMap install, see §5); trainer
model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; batch model
`Qwen/Qwen2.5-0.5B-Instruct`.

## 3. Step 1 — Install the platform

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing
helm dependency update ./deploy
# If a previous `timeslice` release exists, `helm uninstall` it first —
# helm install fails on an existing release name. NOTE: the chart owns the
# timeslice-system namespace, so uninstall deletes it; wait for the
# namespace to finish terminating (~30-60 s) or the reinstall fails with
# "namespace is being terminated".
helm install timeslice ./deploy -n timeslice-system --create-namespace

kubectl -n timeslice-system get pods -o wide     # orchestrator + agents Running
```

> **Release pin:** the chart defaults pull the official
> `ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/*` images at `latest`,
> which include per-job backend
> resolution in the snapshot agent (#159 — required here: the workload
> channel is how the agent drives vLLM's sleep/wake instead of
> cuda-checkpointing it). Once the first tagged release containing it is
> cut, pin it with `--set timesliceorchestrator.image.tag=<version>
> --set snapshot-agent.image.tag=<version>`.

No snapshot-device filter is needed in this topology: every node has one GPU,
and only the labeled pods on the trainer node (rl-head, shadow-vllm) are
snapshot targets — the rollout pod is unlabeled on another node.

## 4. Step 2 — Pick the nodes and label the trainer node

```bash
export TRAINER_NODE=<the shared 1-GPU node (RL trainer + shadow vLLM)>
export ROLLOUT_NODE=<the RL rollout's dedicated 1-GPU node>

kubectl label node "$TRAINER_NODE" group.timeslice.io/trainers=true --overwrite
```

Only the trainer node gets the label (the rollout node must NOT carry it).
Verify an agent runs on the trainer node and the orchestrator synced the
group:

```bash
AGENT_POD=$(kubectl -n timeslice-system get pods -l app.kubernetes.io/name=snapshot-agent \
  --field-selector spec.nodeName=$TRAINER_NODE -o jsonpath='{.items[0].metadata.name}')
echo "$AGENT_POD"
kubectl -n timeslice-system logs deploy/timeslice-timesliceorchestrator --tail=50 | grep -i trainers | tail -3
```

## 5. Step 3 — Launch (order matters)

First, publish the `timeslice-verl` integration package into the cluster as
ConfigMaps (the RL job pods install it from these — no GitHub fetch):

```bash
# Package source: the repo you cloned in §3
PKG=llm-d-rl-time-slicing/pkg/integrations/verl
kubectl create configmap timeslice-verl-root --from-file=$PKG/pyproject.toml
kubectl create configmap timeslice-verl-src  --from-file=$PKG/timeslice_verl/
```

ConfigMap keys cannot contain `/`, so the package tree is split into two maps
(project root + module sources) and reassembled inside the pod; this also
works in air-gapped clusters.

**verl source**: the RL pods clone the `feat/fully-async-lifecycle-hooks`
branch of `github.com/aishukamal/verl` and install it with
`pip install -e ".[gpu]"`. Once the commits land upstream, point the
`VERL_REPO`/`VERL_REF` pod envs at mainline verl instead.

Then start the **shadow vLLM first** so it owns the GPU during the trainer's
long CPU-side setup, then the RL job (both pods at once — the rollout pod's
installs run in parallel with the head's), then the load:

```bash
export RUN_SECONDS=5400     # RL training budget (~90 min)

envsubst '${TRAINER_NODE}' < examples/shadow-vllm.yaml | kubectl apply -f -
# wait for "[supervisor] vLLM is up" (first model download ~2-4 min):
kubectl logs shadow-vllm --tail=20

envsubst '${TRAINER_NODE} ${ROLLOUT_NODE} ${RUN_SECONDS}' < examples/rl-job.yaml | kubectl apply -f -
kubectl apply -f examples/load-generator.yaml
```

The RL job spends 20–45 min on setup (verl install on both pods, the head's
wait for the rollout pod to join the ray cluster, model + dataset prep —
**both pods prepare the dataset locally**: the `FullyAsyncRollouter` actor
runs on the rollout pod and reads the parquet there; no shared volume, same
seed ⇒ identical files) before it first requests the GPU — vLLM serves batch
traffic the whole time.

## 6. Step 4 — Watch it work

```bash
kubectl logs batch-load-generator --tail=6
# [12:01:05] last 5s: completed=41 errors_or_asleep=0     <- vLLM holds GPU
# [12:04:35] last 5s: completed=0  errors_or_asleep=12    <- trainer burst
# [12:08:10] last 5s: completed=38 errors_or_asleep=1     <- vLLM back

kubectl logs shadow-vllm --tail=10
# [supervisor] trainer is waiting - yielding GPU
# [supervisor] drained in 2.82s
# [supervisor] vLLM slept (HBM -> host RAM) in 0.97s
# [supervisor] lock reacquired (waited 214380 ms, context_restored=True)
# [supervisor] vLLM woke in 0.05s

# lock handoffs: the integration package logs every acquire/release
# ([timeslice] ... ACQUIRE/RELEASE lines, forwarded into the head pod's log):
kubectl logs rl-head --tail=2000 | grep -E "ACQUIRE|RELEASE" | tail -4
# [timeslice] job=rl-trainer ACQUIRE group=trainers waited=8210ms context_restored=True
# [timeslice] job=rl-trainer RELEASE group=trainers pending_waiters=1 snapshot_deferred=False
```

Healthy steady state: the load generator alternates between full-throughput
windows (trainer idle) and error windows a few minutes long (trainer burst);
supervisor drain ≈ 3 s, sleep ≤ 3 s, wake ≤ 1 s; trainer `waited=` on
ACQUIRE ≈ 5–11 s (poll latency + drain + vLLM sleep + the trainer's own
cuda-checkpoint context restore, ~5 s once training state is large — the
trainer never queues behind batch work).

Success criteria to check after ~3 training steps:

- Trainer step time within ~10% of a solo run (compare `train.log` timing to
  a run without the shadow vLLM, or to the reference: median 570 s/step).
- Batch throughput > 0 in every trainer-idle window.
- No cuda-checkpoint operations targeting the rollout node in any agent log
  (the rollout node must never show cuda-checkpoint activity).

## 7. Step 5 — Collect results

```bash
mkdir -p results
kubectl cp rl-head:/workspace/results/train.log results/train.log || true
kubectl logs batch-load-generator --timestamps > results/batch_throughput.log
kubectl logs shadow-vllm --timestamps > results/supervisor.log
```

`train.log` (trainer bursts, with the `[timeslice]` ACQUIRE/RELEASE
lock-handoff lines) + `batch_throughput.log` (harvested inference) together
give the shared-GPU timeline: every second is either a training burst, batch
serving, or a swap.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| vLLM never sleeps at handoff; trainer restore then fails with OOM on the shared GPU | Workload registration didn't reach the agent (check supervisor log for `workload registered`) | Verify `TIMESLICE_AGENT_ADDR` resolves to the node IP :9001 and agent logs show the registration; the supervisor's local wake fallback covers restores but sleep MUST go through the channel |
| `POST /sleep` returns 404 | vLLM started without dev endpoints | `VLLM_SERVER_DEV_MODE=1` and `--enable-sleep-mode` are both required (set in the manifest) |
| vLLM dies seconds after a sleep (`CUDA error: an illegal memory access` in EngineCore), then the group faults on the failed wake | `/sleep` was called with requests in flight — vLLM does not drain the scheduler on sleep (vllm#28714, also affects ≥0.10.x; requests sent to a sleeping engine crash it too, vllm#15483) | Keep the supervisor's readiness-gate drain (in the manifest): gate closed + `running+waiting==0` BEFORE `release()`. Never call `/sleep` on a serving engine directly |
| Trainer placement group lands on the rollout node (or rollout work on the trainer node) | PG pinning not active — `--resources` missing on a `ray start` line, or the `ray_pg_extra_resources` hydra override is missing/typo'd, or the verl build lacks the feature | Verify both `ray start` lines carry `--resources`, keep the override in `examples/rl-job.yaml`'s `run_head.sh` intact, and use the pinned verl branch (§2); check `ray status` custom resources |
| cuda-checkpoint activity on the rollout node's agent, or rollout throughput collapses during handoffs | The rollout pod/node got labeled into the group | The rollout node must never show cuda-checkpoint activity: keep `timeslice.io/*` labels off rl-rollout and the group label off `$ROLLOUT_NODE` |
| `rl-head` log stuck at `Phase 2b: wait for the rollout pod to join`, then `FATAL: rollout pod did not join` | rl-rollout Pending/crashed, or headless Service `rl-head` missing | `kubectl get pod rl-rollout; kubectl logs rl-rollout --tail=50`; apply the whole rendered `examples/rl-job.yaml` |
| Load generator: 100% errors even when trainer is idle | vLLM crashed (supervisor exits, pod restarts) or Service selector mismatch | `kubectl logs shadow-vllm --previous`; check `kubectl get endpoints shadow-vllm` |
| RL head or shadow vLLM pod Pending | Trainer node CPU/RAM too small for 8 CPU + 80 Gi (head) alongside 6 CPU + 60 Gi (vLLM) | Use an a3-highgpu-1g-class node (26 vCPU / 234 Gi), or shrink requests |

Demo-client caveat, worth stating to any customer: while vLLM is draining or
asleep, new batch requests fail fast at the Service (the pod is NotReady).
The demo load generator just counts those errors. A production batch
front-end should be queue-based with retries — this composes naturally with
[llm-d async processor](https://github.com/llm-d/llm-d-async) as the batch
dispatcher, which is the planned productization path.

## 9. Adapting

- **Bigger batch model**: anything that fits in `VLLM_GPU_FRAC` of the shared
  GPU alongside zero trainer residency (they never co-reside). Sleep/wake
  time scales with weight size (~1 s per 15 GB to host RAM).
- **Your RL workload**: swap model/dataset in `examples/rl-job.yaml` exactly as in the
  two-RL-jobs guide (§9 there); everything under "Required" comments stays.
- **Yield latency vs. poll cost**: `WAITER_POLL_SECONDS` (default 0.5) is the
  worst-case extra wait the trainer sees.
- **Trainer solo baseline** (for the ≤10% overhead check): run `examples/rl-job.yaml`
  alone — without the shadow vLLM the trainer acquires instantly every step
  and the platform defers snapshots (nothing contends), so it behaves like an
  unshared run.

## 10. Teardown

```bash
kubectl delete pod shadow-vllm batch-load-generator --ignore-not-found
kubectl delete pod rl-head rl-rollout --ignore-not-found
kubectl delete service shadow-vllm rl-head --ignore-not-found
kubectl delete configmap shadow-vllm-scripts rlbatch-trainer --ignore-not-found
kubectl delete configmap timeslice-verl-root timeslice-verl-src --ignore-not-found
helm uninstall timeslice -n timeslice-system
kubectl label node "$TRAINER_NODE" group.timeslice.io/trainers- || true
```
