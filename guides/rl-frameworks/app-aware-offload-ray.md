# Application-Aware GPU Offload for Ray-Based RL Frameworks

This guide is for **framework owners** who want a time-sliced trainer to move its own GPU state to host memory and back, through the snapshot-agent's workload channel (`app_channel` backend), instead of having the platform freeze the whole process with `cuda-checkpoint`. It describes the generic relay shipped in the `timeslice` client, the contract a framework binding must satisfy, and the pod and node requirements. verl is the shipped example; the same relay serves any framework whose trainer ranks are Ray actors.

If you only want to run the verl example, start at the [verl guide's app-channel section](verl/README.md#offloading-via-the-workload-channel-app_channel).

## 1. When to use this instead of cuda-checkpoint

`cuda-checkpoint` needs nothing from the application: it locks the CUDA context, copies all device memory to the process's pageable host RAM, and frees the GPU completely. Its cost is the full copy on every handoff (21 s snapshot / 9 s restore on the 1.5 B fully-async demo) and a pod memory limit that must hold the entire GPU footprint.

An application-aware offload moves only the tensors the trainer owns (model shards, optimizer state, gradients) through the engine's own offload path, which is pinned-memory and asynchronous. It is an order of magnitude faster, selective, and keeps the CUDA context and NCCL communicators alive. It does **not** empty the GPU: the context, communicator buffers and allocator residue (roughly 1–2 GiB per GPU on H100-class hardware, more with large NCCL groups) stay resident. Use it when the co-tenant tolerates that residue, or compose it with a transparent step (section 7).

Scope: the relay acts on the **trainer ranks** only. Rollout or sampler engines (vLLM, SGLang) keep whatever sleep/wake handling the framework or the `app_endpoint` backend gives them; nothing here touches them.

## 2. Architecture: the per-node proxy actor

The agent's workload channel is inverted: the workload opens a stream to the node-local agent, registers under its job id, and receives `snapshot` / `restore` commands. Three facts shape the design for a Ray trainer:

- The agent accepts **one registration per job per node**; a later one replaces the earlier.
- Trainer ranks are one actor per GPU, each executing long blocking tasks. The channel runs its callbacks on a background thread, which must not touch GPU state while the rank's main thread may be using it. The safe way to make a rank offload is to send it a Ray task, so the rank runs the offload on its own thread between tasks.
- The driver that owns the worker group runs on one node, but the orchestrator swaps node by node, and each node's agent only knows the job through a stream opened to *that* agent.

So the relay spawns one zero-CPU **proxy actor per trainer node**, pinned to that node. Each proxy registers the job with its node-local agent and, on a command, calls the engine's offload RPC on the ranks of its node, waits for all of them, and acknowledges.

```
trainer driver actor (CPU, any node)                  trainer node N
  after ranks exist, before first yield:                ┌─────────────────────────────┐
  spawn_offload_proxies(handles, spec, job_id=...) ───> │ OffloadProxy (num_cpus=0)   │──WorkloadChannel──> snapshot-agent :9001
                                                        │  RankFanout(ranks on N)     │<── snapshot / restore ──┘
                                                        │    -> h.<method>.remote(..) │
                                                        └──────────────┬──────────────┘
                                                                       ▼ ray.get(refs, timeout)
                                                        rank actors on N
```

Under the orchestrator's lock protocol the timing is safe by construction: a snapshot is only requested after the job yielded the group lock, and a restore only inside the job's next acquire, before it is granted. The ranks are idle at both moments.

## 3. The relay API (`timeslice.snapshot_agent.rayworker`)

```python
from timeslice.snapshot_agent.rayworker import OffloadSpec, spawn_offload_proxies

spec = OffloadSpec(
    method="actor_to",                                                  # remote method on every rank handle
    snapshot_kwargs={"device": "cpu", "model": True, "optimizer": True, "grad": True},
    restore_kwargs={"device": "device", "model": True, "optimizer": True, "grad": False},
    # restore_method=None,            # defaults to `method`
    # supported_modes=("offload",),   # trainers cannot rebuild discarded state
    # call_timeout_sec=100.0,         # per command; the agent allows 120 s
)

proxies = spawn_offload_proxies(
    rank_handles,                     # raw Ray actor handles, any nodes, rank order
    spec,
    job_id="job-a-trainer",           # == the timeslice.io/job-id pod label
    group="trainers",
    agent=None,                       # each proxy resolves TIMESLICE_AGENT_ADDR on its own node
    ready_timeout=60.0,
)
# ... training ...
proxies.close()                       # crash paths only; proxies die with their owner otherwise
```

What it does:

- `group_handles_by_node` asks each handle for its Ray node id (via `__ray_call__`) and groups the handles per node, preserving rank order.
- One `OffloadProxy` actor per node: `num_cpus=0`, `NodeAffinitySchedulingStrategy(node_id, soft=False)`, named `timeslice-offload-proxy-<job_id>-<node_id[:8]>` (a name collision fails the spawn rather than adopting a stale proxy), non-detached (owned by the caller, so it disappears with it and the dropped stream deregisters it at the agent).
- Each proxy calls `register_workload(agent, job_id, group, workload=RankFanout(...))`, advertising `supported_modes=["offload"]`. The agent rejects a `discard` request for the job before any command is sent.
- `spawn_offload_proxies` returns only after **every** proxy has its channel to its agent open, or raises `TimeoutError` after closing what it started.
- On a command, `RankFanout` calls `handle.<method>.remote(**kwargs)` on every local rank and `ray.get`s the results with `call_timeout_sec`. Any rank error or timeout fails the command with an attributable message (the agent surfaces it as `workload <job> failed command: ...`). Each command logs one line: `job=<id> OFFLOAD|RELOAD node=<id8> ranks=<n> method=<m> took=<s>s`.
- `proxies.info()` returns, per node, the connection state and the counters `snapshots` / `restores`. A binding can poll it to learn that the job has been parked, for example to hold a finishing driver until a pending park completed (section 8).

Requirements: the relay module needs `ray` (not a dependency of the `timeslice` client, so it lives in its own module and is not re-exported from `timeslice.snapshot_agent`; `pip install "timeslice[ray]"` pulls it in), and `timeslice` must be importable on every node that hosts rank actors, because the proxy class and the node-id probe are pickled by reference.

## 4. Writing a binding for your framework

A binding is four decisions, all on the framework side. Nothing in the agent or the relay changes.

1. **The spec.** Which remote method on a rank handle moves state to host and back, with which arguments. For verl this is the engine workers' manual offload RPC, `to(device, model, optimizer, grad)`, exposed on raw colocated handles as `f"{role}_to"` (see `pkg/integrations/verl/timeslice_verl/offload.py`). A framework whose actors expose separate `offload()` / `onload()` methods uses `method="offload", restore_method="onload"`.
2. **When to spawn.** After the rank actors exist and **before the job first yields the accelerator lock**, because a snapshot can arrive immediately after the yield and must find a registered channel. Wait for `spawn_offload_proxies` to return before yielding. In verl this is the end of `init_workers`.
3. **Which backend to ask for.** Once the proxies are registered, yield with `backend_config=app_channel` (section 6) so the orchestrator parks and restores this job through the channel; until then, or if spawning was skipped, yield without one and stay on the agent default. In verl the lock helper carries this as `PhaseLocks.backend_config`, set right after the spawn.
4. **When to close.** On crash paths, so a dying job does not leave a registered channel behind. Normal exit needs nothing. In verl this is the trainer's `on_exception` hook.

Keep the binding inert unless explicitly enabled (verl: `TIMESLICE_APP_OFFLOAD=1`), degrade to a warning when configuration is missing, and let a spawn *failure* fail the job loudly while it still holds the lock: yielding without a channel would make every subsequent snapshot of the job fail.

**slime (sketch, not implemented).** slime's Megatron trainer actors already carry offload/onload methods used by its `--offload-train` option; a binding would be an `OffloadSpec` naming those methods, spawned from the `PhaseCallback` after the actor placement groups are created and before the first `on_phase_end` releases the trainer lock, and closed in `close()`.

## 5. Pod and node contract

- **Labels on every pod that hosts trainer ranks**: `timeslice.io/job-id: <job_id>` and `timeslice.io/group: <group>`. Each node's agent only accepts a snapshot once it has seen GPU activity from a labeled pod on that node, and the orchestrator discovers job membership from the same labels. For a single-node trainer that is the head pod; a multi-node trainer labels every Ray worker pod on a trainer node too.
- **Agent on every trainer node** (the snapshot-agent DaemonSet runs with host networking on port 9001).
- **`TIMESLICE_AGENT_ADDR` on every pod**, node-relative, resolved by the proxy on its own node:
  ```yaml
  - name: NODE_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: TIMESLICE_AGENT_ADDR
    value: "$(NODE_IP):9001"
  ```
  Do not override it through a Ray `runtime_env`, which would pin every proxy to one node's address.
- **`timeslice` installed on every node** that hosts rank actors (the proxy actor runs there).
- **Host memory**: the offloaded state lands in pinned host memory of the rank processes, so the pod memory limit must hold model + optimizer (+ grads) per local rank. This is smaller than the full-VRAM rule of `cuda-checkpoint`, but not zero.

## 6. How the agent selects this backend

The job tells the orchestrator. A `Yield` (`release` in the Python client) may carry a `backend_config`; the orchestrator stores it per job and forwards it to the node-local agents on that job's snapshot and on its next restore, so both use the same backend. For this relay the job passes `app_channel` (`TimeSliceOrchestratorClient(..., backend_config="app_channel")`, `release(backend_config="app_channel")`, or `app_channel_config(mode=..., tags=...)`). A job that never passes one is served by the agent's default backend (`cuda`). The rules, and the known issue that the stored config does not survive an orchestrator restart, are in the orchestrator guide, [Choosing the snapshot backend per job](../timeslice-orchestrator/README.md#choosing-the-snapshot-backend-per-job). No pod label or annotation is involved.

Send it only after the proxies are registered: an `app_channel` request for a job with no channel fails and marks the job FAULTED (section 8). The verl binding does this automatically; the first release after the spawn is the init yield, so the first parking of the job already goes through the channel.

**Debugging: drive the agent directly.** The agent can also be asked without the orchestrator, from a pod on the trainer node, which is useful to check a binding before wiring it into the lock protocol:

```bash
kubectl exec <head-pod> -- python3 - <<'EOF'
import os
from timeslice.snapshot_agent import SnapshotAgentClient, app_channel_config
with SnapshotAgentClient(os.environ["TIMESLICE_AGENT_ADDR"]) as c:
    print(c.snapshot_and_wait(os.environ["TIMESLICE_JOB_ID"], backend_config=app_channel_config()).status)
    print(c.restore_and_wait(os.environ["TIMESLICE_JOB_ID"], backend_config=app_channel_config()).status)
EOF
```

Only do this while the job is between a lock release and its next acquire (ranks idle). A command issued mid-step queues behind the running actor task and exceeds the fan-out timeout.

## 7. Composing with cuda-checkpoint

The proxy actor holds no CUDA context, so it survives a `cuda-checkpoint` of the rank processes. To empty the GPU completely: app-level snapshot first (through the channel), then the transparent checkpoint of the ranks; restore in the reverse order. Today the agent's state machine is per job id, so the two steps use distinct job ids (see the snapshot-agent guide, "Composing Backends"); the orchestrator does not drive this composition.

## 8. Observing and troubleshooting

Job log (`[timeslice]` lines; the proxies print to stdout, which Ray forwards into the driver's log):

```
[timeslice] job=job-a-trainer OFFLOAD-PROXIES ready nodes=1 ranks=1 method=actor_to backend=app_channel took=1.3s
[timeslice] job=job-a-trainer OFFLOAD node=3f2a9c1e ranks=1 method=actor_to took=1.9s
[timeslice] job=job-a-trainer RELOAD node=3f2a9c1e ranks=1 method=actor_to took=1.6s
```

Agent log: `Dispatching snapshot command to workload`, `Workload completed snapshot command ... duration=...`, and the same for restore. `ray list actors` shows one `timeslice-offload-proxy-<job>-<node>` per trainer node.

| Symptom | Cause | Fix |
|---|---|---|
| `no workload channel registered for job "<id>"` | No proxy registered on that node: the binding is disabled, the proxy died, or `TIMESLICE_AGENT_ADDR` points at the wrong agent | Check the job log for `OFFLOAD-PROXIES ready`; verify the env on every pod; `ray list actors` |
| `offload proxies ... did not connect to the snapshot-agent within 60s` | Agent not on that node, not on host networking, or `NODE_IP` missing | `kubectl -n timeslice-system get pods -o wide`; compare with the pod's `TIMESLICE_AGENT_ADDR` |
| `timed out waiting for workload ... to complete command` / `GetTimeoutError` | A rank did not return within `call_timeout_sec`: command issued while the rank was busy, or the state is too large for the window | Only trigger between release and acquire; raise `call_timeout_sec` (stay under the agent's 120 s) |
| `workload does not support suspend mode SUSPEND_MODE_DISCARD` | Trainers are offload-only | Request `offload` or leave the mode unspecified |
| GPU memory does not drop to zero after offload | Expected: context, NCCL and allocator residue stay | Compose with `cuda-checkpoint` (section 7) if the co-tenant needs the whole device |
| A job that just **finished** is marked FAULTED and the whole group stops acquiring | The driver exited (taking the proxies and the channel with it) before the orchestrator parked the job for a peer; the `app_channel` request found no channel. Upstream issue [#210](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/issues/210) (open) | Keep the relay alive after the last yield until the park completed: poll `proxies.info()` for `snapshots` to advance (or hold for a grace period) before exiting. Recovery today: restart the snapshot-agent DaemonSet |
