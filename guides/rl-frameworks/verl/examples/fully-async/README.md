# Time-Slicing Two verl Fully-Async RL Jobs on One Trainer GPU

This guide runs **two independent verl `fully_async_policy` RL training jobs**
whose trainers **share a single GPU** through the llm-d-rl-time-slicing
platform. Each job keeps a dedicated 1-GPU node for its vLLM rollout engine;
the two FSDP trainers take turns on a third, shared 1-GPU node. The platform
checkpoints the idle trainer's GPU memory to host RAM (`cuda-checkpoint`) at
every handoff — no verl source changes, no manual coordination between the
jobs.

> How the integration works (the `TimesliceFullyAsyncTrainer` subclass, lock
> protocol, placement-group pinning), the platform install, node-labeling
> concept, and the `timeslice-verl` package install are covered in the
> **[verl integration guide](../../README.md)** — this README covers only
> what is specific to this two-job demo.

**Measured results** (2× DeepSeek-R1-Distill-Qwen-1.5B, GRPO, H100s — one math
job + one code job, ~2.5 h window):

| Metric | Solo baseline | Time-sliced (2 jobs) |
|---|---|---|
| Shared trainer GPU duty cycle | 41.6% | **86.6% (2.08×)** |
| Training steps completed | 10 | **22 (13 + 9)** |
| Median step time | 627 s | 554 s (A) / 570 s (B) |
| Median swap cost | — | 21 s snapshot + 9 s restore |

The two jobs in this guide are real, distinct workloads:

- **Job A — math RLVR**: DAPO-Math-17k, rule-based answer matching.
- **Job B — code RLVR**: Eurus-2 code split, rewards from live in-pod
  execution of each problem's test cases (`prime_code`).

Both use GRPO on DeepSeek-R1-Distill-Qwen-1.5B (MIT-licensed, ungated — no
HuggingFace token needed).

---

## 1. Architecture

```
   three 1-GPU nodes (we use GKE a3-highgpu-1g, 1× H100 each)

  ROLLOUT NODE A            ROLLOUT NODE B            TRAINER NODE (SHARED)
  ┌──────────────────┐      ┌──────────────────┐      ┌───────────────────────┐
  │ job-a-rollout    │      │ job-b-rollout    │      │ job-a-head  job-b-head│
  │ vLLM, always on  │      │ vLLM, always on  │      │ trainer A ⇄ trainer B │
  │ (ray worker)     │      │ (ray worker)     │      │ alternating via       │
  └────────┬─────────┘      └────────┬─────────┘      │ cuda-checkpoint C/R   │
           │ ray (job A)             │ ray (job B)    └──────────┬────────────┘
           │                         │                           │ snapshot/restore
           ▼                         ▼                           │ (snapshot agent
  ┌──────────────────┐      ┌──────────────────┐                 │  DaemonSet)
  │ job-a-head       │      │ job-b-head       │◀────────────────┘
  │ ray head + verl  │      │ ray head + verl  │
  │ driver + plugin  │      │ driver + plugin  │
  └────────┬─────────┘      └────────┬─────────┘
           │ gRPC lock               │ gRPC lock
           └────────────┬────────────┘
                        ▼
             ┌─────────────────────┐
             │ TimeSlice           │
             │ Orchestrator        │
             │ (group lock)        │
             └─────────────────────┘
```

Each job is a **2-node Ray cluster**: a head pod on the shared trainer node
(runs the verl driver + FSDP trainer) and a rollout pod on a dedicated node
(runs the vLLM rollout engine). Trainer/rollout placement is pinned exactly
as described in the integration guide's
[placement-group pinning section](../../README.md#placement-group-pinning) —
the head starts ray with `--resources='{"trainer_node": 100}'`, the worker
with `--resources='{"rollout_node": 100}'`, and both manifests carry the
`ray_pg_extra_resources={trainer_pool: {trainer_node: 1}, rollout_pool:
{rollout_node: 1}}` hydra override.

The platform pieces at work — the **TimeSlice Orchestrator** (group lock),
the **Snapshot Agent** DaemonSet (cuda-checkpoint snapshot/restore at each
handoff), and the **`timeslice-verl` integration package** (installed inside
each pod, selected via `async_training.trainer_name=timeslice`) — are
described in the [integration guide](../../README.md#3-integrating-with-verl).
In this topology, only the **head pods** carry the `timeslice.io/*` labels the
platform selects on; the rollout pods are unlabeled and live on other nodes,
so rollout processes are never touched. While a job's trainer is checkpointed
off the shared GPU, its rollout node keeps generating samples into the
staleness queue.

## 2. Prerequisites

Cluster-wide prerequisites (workstation tools, image/network reachability,
node labeling, host-RAM sizing rule) are in the
[integration guide §1](../../README.md#1-cluster-prerequisites). Specific to
this demo:

- **Three 1-GPU nodes** (H100-class recommended; the reference shape is GKE
  `a3-highgpu-1g`, 26 vCPU / 234 Gi each): one shared trainer node + one
  dedicated rollout node per job. The GPUs must NOT be in use by anything
  else.
- No HuggingFace token required (model and datasets are ungated).
- Host RAM on the trainer node: the two head pods request 80 Gi each
  (110 Gi limit) — for this 1.5 B model the trainer's shared-GPU footprint
  reaches ~50 GB, and the cuda-checkpoint dump lands in pod memory (see the
  [host-RAM sizing rule](../../README.md#host-ram-sizing-rule)).

**Version pins used throughout** (change nothing on the first run) — the verl
fork branch, platform images, and `timeslice-verl` package install are pinned
in the [integration guide](../../README.md) (§2–3); demo-specific pins:

| Component | Pin |
|---|---|
| Job container image | `verlai/verl:vllm020.dev2` |
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (HF id, downloaded per node) |

Files in this guide directory:

| File | What it is |
|---|---|
| `job-a-math.yaml` | Job A (math RLVR): ConfigMap + head pod + rollout pod + headless Service |
| `job-b-code.yaml` | Job B (code RLVR): ConfigMap + head pod + rollout pod + headless Service |

## 3. Step 1 — Install the time-slicing platform

Follow the [integration guide §2](../../README.md#2-deploying-the-time-slicing-platform)
(including the clean-uninstall note if you previously installed the platform
and the verify step):

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git
cd llm-d-rl-time-slicing

helm dependency update ./deploy
helm install timeslice ./deploy -n timeslice-system --create-namespace
```

The chart defaults pull the official platform images at `latest`; see the
integration guide's release-pin note for pinning a tagged release once one
is cut.

No snapshot-device filter is needed in this topology: every node has exactly
one GPU, and only the labeled head pods on the trainer node are snapshot
candidates.

## 4. Step 2 — Pick the nodes and label the trainer node

Pick three 1-GPU nodes: one **trainer node** (shared by both jobs' head pods)
and one **rollout node per job**. Label ONLY the trainer node into the
`trainers` group (see the
[labeling concept](../../README.md#node-labeling-and-time-slice-groups);
the rollout nodes must NOT carry it):

```bash
export TRAINER_NODE=<the shared 1-GPU trainer node>
export ROLLOUT_NODE_A=<job A's dedicated 1-GPU node>
export ROLLOUT_NODE_B=<job B's dedicated 1-GPU node>

kubectl label node "$TRAINER_NODE" group.timeslice.io/trainers=true --overwrite
```

Verify the platform sees it (the orchestrator group-sync check is in the
[integration guide §1](../../README.md#node-labeling-and-time-slice-groups)),
and grab the agent pod on the trainer node — you'll grep its logs in §6:

```bash
# a snapshot-agent pod runs on the trainer node:
AGENT_POD=$(kubectl -n timeslice-system get pods -l app.kubernetes.io/name=snapshot-agent \
  --field-selector spec.nodeName=$TRAINER_NODE -o jsonpath='{.items[0].metadata.name}')
echo "$AGENT_POD"
```

## 5. Step 3 — Launch the two jobs

First, publish the `timeslice-verl` package into the cluster as the
`timeslice-verl-root`/`timeslice-verl-src` ConfigMaps — commands and details
in the [integration guide §3](../../README.md#installing-the-timeslice-verl-package-configmap).
The job pods install the package from these (no GitHub fetch), and each pod
clones and installs the
[pinned verl fork branch](../../README.md#installing-verl-pinned-fork) at
startup (once the commits land upstream, point the `VERL_REPO`/`VERL_REF` pod
envs at mainline verl instead).

Then launch:

```bash
export RUN_SECONDS=5400        # training budget per job (~90 min ≈ 8–12 steps)

# IMPORTANT: envsubst MUST get exactly the variable list shown per file — the
# manifests embed shell scripts whose other $VARs must pass through untouched.
envsubst '${TRAINER_NODE} ${ROLLOUT_NODE_A} ${RUN_SECONDS}' < job-a-math.yaml | kubectl apply -f -
envsubst '${TRAINER_NODE} ${ROLLOUT_NODE_B} ${RUN_SECONDS}' < job-b-code.yaml | kubectl apply -f -
```

Each job creates 4 objects in namespace `default`: a ConfigMap, a head pod
(`job-a-head` / `job-b-head`, on the trainer node), a rollout pod
(`job-a-rollout` / `job-b-rollout`, on its dedicated node) and a headless
Service the rollout pod uses to find the ray head. Startup takes **20–45
minutes** before the first training step (much less when the image and HF
caches are warm on the nodes): every pod installs verl from the pinned
branch and the integration package; **both pods prepare the dataset locally** (the
`FullyAsyncRollouter` actor runs on the rollout pod and reads the parquet
there — no shared volume, same seed ⇒ identical files); the head waits (up
to 40 min) for its rollout pod to join the ray cluster and downloads the
model by HF id. This is all logged; slow ≠ stuck (see §6 for what to
expect when).

## 6. Step 4 — Watch it work

```bash
kubectl logs job-a-head --tail=30        # never use -f in scripts; poll instead
kubectl logs job-a-rollout --tail=10
```

The sequence you should see per job (grep-able markers):

1. `Phase 1: install verl` (both pods, ~20 min) → head: `Phase 2b: wait for
   the rollout pod to join` (a progress line every 20 s poll) → `Phase 3:
   prepare dataset`.
2. `[timeslice] job=job-a-trainer ACQUIRE group=trainers waited=...ms` — first
   acquisition (model load onto the shared GPU). Grep for `"ACQUIRE"`; note
   the job scripts run under `set -x`, so echoed lines appear twice (once
   with a `+` trace prefix).
3. Alternation, visible in both job logs and the agent's:

```bash
# lock handoffs: the integration package logs every acquire/release
# ([timeslice] ... ACQUIRE/RELEASE lines, forwarded into the head pod's log):
kubectl logs job-a-head --tail=2000 | grep -E "ACQUIRE|RELEASE" | tail -5
# [timeslice] job=job-a-trainer ACQUIRE group=trainers waited=232000ms context_restored=True
# [timeslice] job=job-a-trainer RELEASE group=trainers pending_waiters=1 snapshot_deferred=False

# snapshot/restore operations on the shared trainer GPU:
kubectl -n timeslice-system logs "$AGENT_POD" --tail=100 | grep -i "cuda-checkpoint"
```

Healthy steady state (after ~2 warm-up steps): demand-driven A↔B alternation
(mostly A→B→A→B; a job takes back-to-back bursts when the other's sample
queue isn't ready yet — that backfill is the point, not a fault);
`waited=` on ACQUIRE ≈ the other job's training burst (2–6 min for this
model); snapshots 12–35 s, restores 5–14 s;
`count/dropped_stale_samples` stays 0 in the verl stat lines (the key
appears periodically — check its value, not the line count).

The run ends when `RUN_SECONDS` expires (`timeout` sends INT — the log line
`Training stopped by planned ... timeout (expected end state)` is success, and
the head pod stays alive 2 h for result collection).

## 7. Step 5 — Collect results

```bash
for POD in job-a-head job-b-head; do
  mkdir -p "results/$POD"
  kubectl cp "$POD:/workspace/results/train.log" "results/$POD/train.log" || true
done
```

- `train.log` — full verl output (steps, rewards, timing breakdown, and the
  `[timeslice]` ACQUIRE/RELEASE lock-handoff lines; head pod).

## 8. Troubleshooting

General failures — unpinned placement groups,
ray `address='auto'` cross-talk, hydra key
paths, the `save_freq` veto, staleness-queue overflow — are covered in the
[integration guide §4](../../README.md#4-general-troubleshooting). The rows
below are specific to this demo's manifests and topology:

| Symptom | Cause | Fix |
|---|---|---|
| Head log stuck at `Phase 2b: wait for the rollout pod to join` past 40 min, then `FATAL: rollout pod did not join` | Rollout pod Pending/crashed, or it can't resolve the headless Service (`<job>-head.default.svc.cluster.local:6379`) | `kubectl get pod job-a-rollout; kubectl logs job-a-rollout --tail=50`; both pods and the Service must come from the same rendered manifest |
| Head pod Pending forever | Trainer node lacks free CPU/RAM for the second 8-CPU/80 Gi head request | The two head pods need ~17 CPU / 161 Gi on the trainer node (a3-highgpu-1g: 26 vCPU / 234 Gi) |
| Node reclaimed mid-run (spot/preemptible) | — | Results live in the pods (emptyDir): collect every ~10 min on spot capacity, or use on-demand nodes |

## 9. Adapting to your workload

Swap either job's model/data with three overrides in its manifest (and keep
everything under "Required" comments intact):

- `actor_rollout_ref.model.path` — any HF model id (fits on one GPU;
  TP > 1 is not yet supported by the snapshot path)
- `data_prep.py` + `data.train_files`/`val_files` — your dataset in verl
  RLHFDataset format
- `trainer.experiment_name`

Rules of thumb when scaling up: snapshot time grows with allocated trainer
memory (~13 s at 3 GB → ~35 s at 50 GB on H100 + PCIe host RAM); swap overhead
is amortized by your step time (keep bursts ≥ 10× swap cost);
`async_training.trigger_parameter_sync_step=1` yields once per step — raise it
to trade staleness for fewer swaps.

## 10. Teardown

```bash
kubectl delete pod job-a-head job-a-rollout job-b-head job-b-rollout --ignore-not-found
kubectl delete service job-a-head job-b-head --ignore-not-found
kubectl delete configmap ts-job-a ts-job-b --ignore-not-found
kubectl delete configmap timeslice-verl-root timeslice-verl-src --ignore-not-found
helm uninstall timeslice -n timeslice-system
kubectl label node "$TRAINER_NODE" group.timeslice.io/trainers- || true
```
