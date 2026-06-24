import contextlib
from typing import Any, List, Optional, Tuple

import grpc

from timeslice.orchestrator._generated import pb2, pb2_grpc
from timeslice.orchestrator.exceptions import wrap_grpc_error
from timeslice.orchestrator.types import (
    AgentJobState,
    AcquireResult,
    GroupLockState,
    GroupStatus,
    OrchestratorGroupStatus,
    SnapshotAgentJobState,
    YieldResult,
)


class OrchestratorClient:
    """Convenient UX wrapper client for the Accelerator Orchestrator Service."""

    def __init__(
        self,
        target: str,
        job_id: Optional[str] = None,
        group_id: Optional[str] = None,
        channel_options: Optional[List[Tuple[str, Any]]] = None,
    ):
        """Initializes the Orchestrator Client.

        Args:
            target: gRPC server address (e.g., 'localhost:50051').
            job_id: Optional unique identifier of the job.
            group_id: Optional unique identifier of the time-slice group.
            channel_options: Optional gRPC channel options.
        """
        self.target = target
        self.job_id = job_id
        self.group_id = group_id

        # Initialize insecure channel. (Can be extended to secure if needed in future)
        self._channel = grpc.insecure_channel(target, options=channel_options)
        self._stub = pb2_grpc.AcceleratorOrchestratorServiceStub(self._channel)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes the underlying gRPC channel."""
        if hasattr(self, "_channel") and self._channel:
            self._channel.close()
            self._channel = None

    def __del__(self):
        self.close()

    def acquire(
        self,
        job_id: Optional[str] = None,
        group_id: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> AcquireResult:
        """Acquires exclusive access to the time-slice group.

        This call blocks until access is granted or timeout is reached.

        Args:
            job_id: Optional job_id to override the constructor value.
            group_id: Optional group_id to override the constructor value.
            timeout_sec: Optional timeout in seconds for the RPC call.

        Returns:
            AcquireResult containing success, waited_ms, and context_restored.

        Raises:
            ValueError: If job_id or group_id is not provided either here or in the constructor.
            OrchestratorError: If the RPC fails.
        """
        resolved_job_id = job_id or self.job_id
        resolved_group_id = group_id or self.group_id

        if not resolved_job_id:
            raise ValueError(
                "job_id must be provided either in the constructor or in the method call."
            )
        if not resolved_group_id:
            raise ValueError(
                "group_id must be provided either in the constructor or in the method call."
            )

        request = pb2.AcquireRequest(job_id=resolved_job_id, group_id=resolved_group_id)
        try:
            # Note: timeout in grpc is passed as 'timeout' keyword argument in seconds
            response = self._stub.Acquire(request, timeout=timeout_sec)
            return AcquireResult(
                success=response.success,
                waited_ms=response.waited_ms,
                context_restored=response.context_restored,
            )
        except grpc.RpcError as e:
            raise wrap_grpc_error(e) from e

    def release(
        self,
        job_id: Optional[str] = None,
        group_id: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> YieldResult:
        """Releases exclusive access to the time-slice group.

        Returns immediately once recorded.

        Args:
            job_id: Optional job_id to override the constructor value.
            group_id: Optional group_id to override the constructor value.
            timeout_sec: Optional timeout in seconds for the RPC call.

        Returns:
            YieldResult containing success, pending_waiters, and snapshot_deferred.

        Raises:
            ValueError: If job_id or group_id is not provided either here or in the constructor.
            OrchestratorError: If the RPC fails.
        """
        resolved_job_id = job_id or self.job_id
        resolved_group_id = group_id or self.group_id

        if not resolved_job_id:
            raise ValueError(
                "job_id must be provided either in the constructor or in the method call."
            )
        if not resolved_group_id:
            raise ValueError(
                "group_id must be provided either in the constructor or in the method call."
            )

        request = pb2.YieldRequest(job_id=resolved_job_id, group_id=resolved_group_id)
        try:
            response = self._stub.Yield(request, timeout=timeout_sec)
            return YieldResult(
                success=response.success,
                pending_waiters=response.pending_waiters,
                snapshot_deferred=response.snapshot_deferred,
            )
        except grpc.RpcError as e:
            raise wrap_grpc_error(e) from e

    def get_status(
        self, group_id: Optional[str] = None, timeout_sec: Optional[float] = None
    ) -> OrchestratorGroupStatus:
        """Returns the detailed status of the time-slice group.

        Args:
            group_id: Optional group_id to override the constructor value.
            timeout_sec: Optional timeout in seconds for the RPC call.

        Returns:
            OrchestratorGroupStatus containing group status and agent states.

        Raises:
            ValueError: If group_id is not provided either here or in the constructor.
            OrchestratorError: If the RPC fails.
        """
        resolved_group_id = group_id or self.group_id
        if not resolved_group_id:
            raise ValueError(
                "group_id must be provided either in the constructor or in the method call."
            )

        request = pb2.GetGroupStatusRequest(group_id=resolved_group_id)
        try:
            response = self._stub.GetGroupStatus(request, timeout=timeout_sec)

            # Map GroupStatus
            proto_group = response.group
            group_state = self._map_group_state(proto_group.group_state)

            # Convert protobuf timestamp to python datetime
            dt = (
                proto_group.state_timestamp.ToDatetime()
                if proto_group.HasField("state_timestamp")
                else None
            )

            group_status = GroupStatus(
                group_id=proto_group.group_id,
                group_state=group_state,
                state_timestamp=dt,
                locking_job=proto_group.locking_job,
                active_job=proto_group.active_job,
                waiter_queue_depth=proto_group.waiter_queue_depth,
                loaded_job=proto_group.loaded_job,
            )

            # Map Agent States
            agent_states = []
            for proto_agent_state in response.agent_job_states:
                agent_states.append(
                    SnapshotAgentJobState(
                        agent=proto_agent_state.agent,
                        job_id=proto_agent_state.job_id,
                        job_state=self._map_agent_state(proto_agent_state.job_state),
                    )
                )

            return OrchestratorGroupStatus(
                group=group_status,
                agent_job_states=agent_states,
            )
        except grpc.RpcError as e:
            raise wrap_grpc_error(e) from e

    def list_groups(self, timeout_sec: Optional[float] = None) -> List[str]:
        """Lists all active time-slice group IDs.

        Args:
            timeout_sec: Optional timeout in seconds for the RPC call.

        Returns:
            List of group ID strings.

        Raises:
            OrchestratorError: If the RPC fails.
        """
        request = pb2.ListGroupsRequest()
        try:
            response = self._stub.ListGroups(request, timeout=timeout_sec)
            return list(response.group_ids)
        except grpc.RpcError as e:
            raise wrap_grpc_error(e) from e

    @contextlib.contextmanager
    def lock(
        self,
        job_id: Optional[str] = None,
        group_id: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ):
        """Context manager for safe acquire/release flow.

        Usage:
            with client.lock():
                # exclusive access here
        """
        self.acquire(job_id=job_id, group_id=group_id, timeout_sec=timeout_sec)
        try:
            yield
        finally:
            self.release(job_id=job_id, group_id=group_id)

    def _map_group_state(self, proto_state: int) -> GroupLockState:
        try:
            name = pb2.GroupStatus.State.Name(proto_state)
            friendly_name = name.replace("STATE_", "")
            return GroupLockState[friendly_name]
        except (ValueError, KeyError):
            return GroupLockState.UNSPECIFIED

    def _map_agent_state(self, proto_state: int) -> AgentJobState:
        try:
            name = pb2.SnapshotAgentJobState.State.Name(proto_state)
            friendly_name = name.replace("STATE_", "")
            return AgentJobState[friendly_name]
        except (ValueError, KeyError):
            return AgentJobState.UNSPECIFIED
