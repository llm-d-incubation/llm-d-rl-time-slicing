# RL Framework Integrations

Time-slicing has been validated with the following RL training frameworks. Each integration lets multiple jobs share GPU hardware at phase boundaries.

## Supported Frameworks

| Framework | Modes Validated | Integration Approach | Guide |
|-----------|----------------|---------------------|-------|
| [Slime](slime/) | Sync disagg, Async disagg | Fork-based (current) / PhaseCallback (in development) | [Slime guide](slime/) |
| [verl](verl/) | Sync colocated, Sync disagg, Fully-async | Trainer subclass + `@register_trainer` | [verl guide](verl/) |

## Adding a New Framework

See the [Framework Integration Guide](../framework-integration/) for the generic pattern: identify phase boundaries, choose an integration approach (PhaseCallback, subclass, or method patch), implement the lock protocol.
