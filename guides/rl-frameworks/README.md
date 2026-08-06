# RL Framework Integrations

Time-slicing integration packages for RL training frameworks. Each package lets multiple jobs share GPU hardware with zero changes to framework source code.

## Supported Frameworks

| Framework | Modes | Integration | Package | Status |
|-----------|-------|-------------|---------|--------|
| [Slime](slime/) | Sync disagg, Async disagg | PhaseCallback (`--phase-callback-path`) | `timeslice-slime` | Validated on H100 |
| [verl](verl/) | Sync colocated, Sync disagg, Fully-async | Subclass + `@register_trainer` | `timeslice-verl` | Validated on H100 |

## How It Works

Each integration wraps the framework's GPU phase boundaries (init, generate, train, weight sync) with orchestrator lock acquire/release calls. The platform's snapshot agent handles GPU state checkpoint/restore transparently — the framework runs unmodified.

For implementation details, see the [Framework Integration Guide](../framework-integration/).

## Adding a New Framework

1. Read the [Framework Integration Guide](../framework-integration/) to understand the phase boundary and lock protocol patterns
2. Identify your framework's driver script or lifecycle hooks
3. Choose an integration pattern (PhaseCallback, subclass, or method patch)
4. Build a `timeslice-<framework>` package using the shared lock primitives
5. Add a guide and runnable example under `rl-frameworks/<framework>/`
