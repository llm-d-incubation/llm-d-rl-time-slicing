# rlts CLI

`rlts` is a command-line interface for interacting with the LLM-D RL Time Slicing components, primarily the Accelerator Orchestrator.

## Building the CLI

To build the CLI binary, run the following command from the root of the repository:

```bash
go build -o bin/rlts ./cmd/rlts
```

This will create the `rlts` binary in the `bin` directory.

## Connecting to a Deployed Orchestrator

If the Accelerator Orchestrator is deployed inside a Kubernetes cluster, you need to establish a connection to it. The easiest way to do this from your local machine is using `kubectl port-forward`.

Assuming the orchestrator is deployed as a service named `acceleratororchestrator` in the default namespace:

```bash
kubectl port-forward svc/acceleratororchestrator 50051:50051
```

This forwards traffic from your local port `50051` to the orchestrator service in the cluster. Keep this command running in a separate terminal.

## Usage

By default, the CLI attempts to connect to `localhost:50051`. You can override this using the global `--addr` flag.

```bash
rlts [command] [--addr <address>]
```

### Global Flags

*   `--addr string`: Address of the accelerator orchestrator gRPC server (default "localhost:50051")
*   `-h`, `--help`: Help for `rlts`

---

## Orchestrator Commands

All commands for interacting with the Accelerator Orchestrator are nested under `rlts orchestrator`.

### List Active Groups

List all active time-slice groups managed by the orchestrator.

```bash
./bin/rlts orchestrator list
```

### Get Group Status

Get the detailed status of a specific time-slice group, including the current locking job, waiter queue depth, and individual job context states on active agents.

```bash
./bin/rlts orchestrator status <group-id>
```

Example:
```bash
./bin/rlts orchestrator status my-time-slice-group
```

### Acquire Access (Manual/Debug)

Manually request exclusive access to a time-slice group for a specific job. **Note:** This call is blocking and will wait until access is granted by the orchestrator.

```bash
./bin/rlts orchestrator acquire <group-id> <job-id>
```

Example:
```bash
./bin/rlts orchestrator acquire my-time-slice-group job-123
```

### Yield Access (Manual/Debug)

Manually release exclusive access to a time-slice group for a specific job.

```bash
./bin/rlts orchestrator yield <group-id> <job-id>
```

Example:
```bash
./bin/rlts orchestrator yield my-time-slice-group job-123
```
