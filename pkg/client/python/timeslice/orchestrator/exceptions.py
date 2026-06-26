import grpc


class OrchestratorError(Exception):
    """Base exception for all Orchestrator client errors."""

    pass


class OrchestratorConnectionError(OrchestratorError):
    """Raised when the client cannot connect to the Orchestrator service."""

    pass


class OrchestratorTimeoutError(OrchestratorError):
    """Raised when an operation times out."""

    pass


class OrchestratorInvalidArgumentError(OrchestratorError):
    """Raised when invalid arguments are passed to the Orchestrator."""

    pass


class OrchestratorInternalError(OrchestratorError):
    """Raised when the Orchestrator encounters an internal error."""

    pass


def wrap_grpc_error(error: grpc.RpcError) -> OrchestratorError:
    """Helper to map grpc.RpcError to OrchestratorError."""
    code = error.code()
    details = error.details()

    if code == grpc.StatusCode.UNAVAILABLE:
        return OrchestratorConnectionError(
            f"Failed to connect to Orchestrator: {details}"
        )
    elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return OrchestratorTimeoutError(f"Orchestrator operation timed out: {details}")
    elif code == grpc.StatusCode.INVALID_ARGUMENT:
        return OrchestratorInvalidArgumentError(f"Invalid argument: {details}")
    elif code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.UNKNOWN):
        return OrchestratorInternalError(f"Orchestrator internal error: {details}")
    else:
        return OrchestratorError(
            f"Orchestrator RPC failed with status {code}: {details}"
        )
