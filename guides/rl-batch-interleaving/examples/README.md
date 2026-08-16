# Time-Sliced verl RL Trainer + vLLM Batch Inference Example

Runnable demo of the [RL + batch interleaving guide](../README.md): one verl
fully-async code-RLVR job (GRPO on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`,
Eurus-2 code split) and a stock `vllm serve` of `Qwen/Qwen2.5-0.5B-Instruct`
share one H100 — the trainer has absolute priority, vLLM harvests its idle
valleys. Both models are ungated (no HuggingFace token).

**Reference-run behavior** (2× GKE `a3-highgpu-1g`, H100):

| Metric | Value |
|---|---|
| Trainer ACQUIRE wait at each handoff | 5–11 s |
| Trainer step time vs. solo run | within ~10% (reference median 570 s/step) |
| vLLM yield (drain + sleep) / wake | ~3 s + ~1–2 s / ≤ 1 s |
| Batch throughput | full throughput in every trainer-idle window |

---

## 1. Architecture

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

The RL job is a 2-node Ray cluster: the head pod (verl driver + FSDP trainer)
shares the trainer node's GPU with the batch server pod, while the rollout pod
keeps its own dedicated node and never stops generating. At each handoff the
trainer is checkpointed with **`cuda-checkpoint`** (its entire GPU state dumps
to host RAM), while vLLM yields through the snapshot agent's workload channel:
a supervisor drains the engine, puts it to sleep (weights + KV cache to host
RAM, no process restart), and wakes it when the trainer releases. How the
supervisor works is covered in
[root guide §4](../README.md#4-the-batch-tenant-and-the-supervisor).

## 2. Prerequisites

The time-slicing platform must already be deployed and healthy, with the
cluster requirements, node-labeling concept, shared DRA claim, and host-RAM
sizing rule from [root guide §1](../README.md#1-cluster-prerequisites) and the
platform install + verify steps from
[root guide §2](../README.md#2-deploy-the-time-slicing-platform). Specific to
this demo:

- **Two otherwise-idle 1-GPU nodes** (reference shape GKE `a3-highgpu-1g`,
  26 vCPU / 234 Gi, H100): a **trainer node** (shared: RL head pod + vLLM pod)
  and a **rollout node** (dedicated to the RL rollout engine).
- **Host RAM on the trainer node** must fit both tenants' offloads (see the
  [host-RAM sizing rule](../README.md#host-ram-sizing-rule)): the RL head pod
  requests 8 CPU + 80 Gi (110 Gi limit — the cuda-checkpoint dump lands in pod
  memory) alongside the vLLM pod's 6 CPU + 60 Gi sleep offload.

**Version pins** (change nothing on the first run) — all pins live in the
manifests:

| Component | Pin |
|---|---|
| RL job image | `verlai/verl:vllm020.dev2` |
| Batch image | `vllm/vllm-openai:v0.9.2` |
| verl | fork branch, via the `VERL_REPO`/`VERL_REF` pod envs |

Files in this guide directory:

| File | What it is |
|---|---|
| `resource-claims.yaml` | The single DRA claim: `shared-trainers-gpu-claim` (rl-head + shadow-vllm; rl-rollout uses an ordinary `nvidia.com/gpu` request instead) |
| `rl-job.yaml` | The RL job: ConfigMap + head pod + rollout pod + headless Service |
| `shadow-vllm.yaml` | EXAMPLE-ONLY: stock `vllm serve` + the supervisor |
| `load-generator.yaml` | EXAMPLE-ONLY: continuous `/v1/completions` client + throughput log |

## 3. Step 1 — Apply the resource claim and label the trainer node

Apply the shared GPU claim (the recipe's only `ResourceClaim` — rl-rollout
requests its GPU via the ordinary device plugin instead), then set up the
trainer node. The trainer node carries all the time-slicing markers — the
group label (group discovery), the `timeslice.io/enabled` label (the
platform-wide marker for time-slicing nodes, which the root guide §2
install targets snapshot-agent and DRA kubelet-plugin placement at) and
the isolation taint (keeps unrelated workloads off the shared GPU; the
pods in this guide already carry the matching toleration) — while the
rollout node carries nothing timeslice-related:

```bash
kubectl apply -f resource-claims.yaml
kubectl label node <trainer-node> group.timeslice.io/trainers=true --overwrite
kubectl label node <trainer-node> timeslice.io/enabled=true --overwrite
kubectl taint node <trainer-node> timeslice.io/shared=true:NoSchedule --overwrite
```

After labeling, the snapshot-agent and DRA kubelet-plugin pods (installed
per the root guide §2) appear on the trainer node — and only there:

```bash
kubectl -n timeslice-system get pods -o wide
```

The rollout node needs no setup: rl-head and shadow-vllm co-locate where
their shared claim binds, and rl-rollout lands on any GPU node except the
trainer node (required anti-affinity; the guide's
[labeling section](../README.md#node-labeling-and-time-slice-groups) spells
out the placement guarantees).

## 4. Step 2 — Launch the tenants

Order matters. Launch the batch tenant FIRST — it owns the GPU during the RL
job's long CPU-side setup — and wait for the supervisor to report the engine
up:

```bash
kubectl apply -f shadow-vllm.yaml
kubectl logs shadow-vllm --tail=20    # wait for "[supervisor] vLLM is up" (first model download ~2-4 min)
```

Then launch the RL job and the load (training budget: edit the `RUN_SECONDS`
env in `rl-job.yaml`, default 5400 s):

```bash
kubectl apply -f rl-job.yaml
kubectl apply -f load-generator.yaml
```

The RL job spends **20–45 minutes** on setup (verl install on both pods, ray
join, model + dataset download) before it first requests the GPU; vLLM serves
batch traffic the whole time. Slow ≠ stuck — the phases are logged in
`kubectl logs rl-head`.

## 5. Step 3 — Watch it work

```bash
# Batch throughput alternates with trainer bursts:
kubectl logs batch-load-generator --tail=6
# [12:01:05] last 5s: completed=41 errors_or_asleep=0     <- vLLM holds GPU
# [12:04:35] last 5s: completed=0  errors_or_asleep=12    <- trainer burst
# [12:08:10] last 5s: completed=38 errors_or_asleep=1     <- vLLM back

# Supervisor handoffs (drain -> sleep -> reacquire -> wake):
kubectl logs shadow-vllm --tail=10
# [supervisor] trainer is waiting - yielding GPU
# [supervisor] drained in 2.82s
# [supervisor] vLLM slept (HBM -> host RAM) in 0.97s
# [supervisor] lock reacquired (waited 214380 ms, context_restored=True)
# [supervisor] vLLM woke in 0.05s

# Trainer-side lock handoffs:
kubectl logs rl-head --tail=2000 | grep -E "ACQUIRE|RELEASE" | tail -4
```

Or watch the orchestrator's group state directly (`rlts` built per the
[root guide §2](../README.md#2-deploy-the-time-slicing-platform)):

```bash
kubectl port-forward svc/timeslice-timesliceorchestrator 50051:50051 -n timeslice-system
watch -n 0.5 ./bin/rlts orchestrator status trainers
```

`Active Job` / `Locking Job` alternate between `rl-trainer` and `shadow-vllm`;
`rl-trainer` appears briefly at `Waiter Queue Depth` = 1 at each handoff. The
run ends when `RUN_SECONDS` expires — the log line `Training stopped by
planned ... timeout (expected end state)` is success, and the head pod stays
alive 2 h for collection.

## 6. Step 4 — Collect results

```bash
mkdir -p results
kubectl cp rl-head:/workspace/results/train.log results/train.log || true
kubectl logs batch-load-generator --timestamps > results/batch_throughput.log
kubectl logs shadow-vllm --timestamps > results/supervisor.log
```

## 7. Troubleshooting

Generic failures (placement, labels, hydra keys, drain rationale, client
retry posture) are in the [root guide §7](../README.md#7-troubleshooting).
Demo-specific:

| Symptom | Cause | Fix |
|---|---|---|
| vLLM never sleeps at handoff; trainer restore then fails with OOM on the shared GPU | Workload registration didn't reach the agent (check supervisor log for `workload registered`) | Verify `TIMESLICE_AGENT_ADDR` resolves to the node IP :9001 and agent logs show the registration; the supervisor's local wake fallback covers restores but sleep MUST go through the channel |
| `POST /sleep` returns 404 | vLLM started without dev endpoints | `VLLM_SERVER_DEV_MODE=1` and `--enable-sleep-mode` are both required (set in the manifest) |
| vLLM dies seconds after a sleep (`CUDA error: an illegal memory access` in EngineCore), then the group faults on the failed wake | `/sleep` was called with requests in flight — vLLM does not drain the scheduler on sleep (vllm#28714, also affects ≥0.10.x; requests sent to a sleeping engine crash it too, vllm#15483) | Keep the supervisor's readiness-gate drain (in the manifest): gate closed + `running+waiting==0` BEFORE `release()`. Never call `/sleep` on a serving engine directly |
| `rl-head` log stuck at `Phase 2b: wait for the rollout pod to join`, then `FATAL: rollout pod did not join` | rl-rollout Pending/crashed, or headless Service `rl-head` missing | `kubectl get pod rl-rollout; kubectl logs rl-rollout --tail=50`; apply the whole rendered `rl-job.yaml` |
| Load generator: 100% errors even when trainer is idle | vLLM crashed (supervisor exits, pod restarts) or Service selector mismatch | `kubectl logs shadow-vllm --previous`; check `kubectl get endpoints shadow-vllm` |
| RL head or shadow vLLM pod Pending | Trainer node CPU/RAM too small for 8 CPU + 80 Gi (head) alongside 6 CPU + 60 Gi (vLLM) | Use an a3-highgpu-1g-class node (26 vCPU / 234 Gi), or shrink requests |

## 8. Adapting to your workload

The model/engine/poll knobs are in the root guide's
[Swapping In Your Engine or Model](../README.md#swapping-in-your-engine-or-model)
section: a bigger batch model must fit the engine's GPU-memory fraction
(`VLLM_GPU_FRAC`, default 0.7) of the shared GPU, and sleep/wake time scales
with weight size (~1 s per 15 GB to host RAM); another engine keeps the
supervisor's loop and swaps the freeze/thaw calls, the drain check, and the
health probe; `WAITER_POLL_SECONDS` (default 0.5) is the worst-case extra
wait the trainer sees before the supervisor notices it.

For the solo baseline (the reference table's step-time comparison), run the
RL job alone: with no contention the trainer acquires instantly and the
platform defers snapshots, so it behaves like an unshared run.

## 9. Teardown

```bash
kubectl delete pod shadow-vllm batch-load-generator --ignore-not-found
kubectl delete pod rl-head rl-rollout --ignore-not-found
kubectl delete service shadow-vllm rl-head --ignore-not-found
kubectl delete configmap shadow-vllm-scripts rlbatch-trainer --ignore-not-found
kubectl delete -f resource-claims.yaml --ignore-not-found
kubectl label node <trainer-node> group.timeslice.io/trainers- || true
kubectl label node <trainer-node> timeslice.io/enabled- || true
kubectl taint node <trainer-node> timeslice.io/shared- || true
```

(Platform teardown, if wanted: `helm uninstall timeslice -n timeslice-system`.)
