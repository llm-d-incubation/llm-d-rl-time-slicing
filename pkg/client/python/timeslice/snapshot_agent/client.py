import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

import grpc

from . import health_pb2, health_pb2_grpc, snapshot_agent_pb2, snapshot_agent_pb2_grpc
from .types import (
    AcceleratorStatus,
    GetOperationResponse,
    HealthResponse,
    JobStatus,
    RestoreResponse,
    SnapshotResponse,
    StatusResponse,
)

logger = logging.getLogger(__name__)


class SnapshotAgentError(Exception):
    """Base exception for SnapshotAgentClient."""

    def __init__(
        self,
        message: str,
        code: Optional[grpc.StatusCode] = None,
        details: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details


class SnapshotAgentInterface(ABC):
    """Abstract base class for SnapshotAgentService client."""

    @abstractmethod
    def snapshot(
        self, job_id: str, group: str, backend: Union[str, int] = 0
    ) -> SnapshotResponse:
        """Triggers an asynchronous snapshot."""
        pass

    @abstractmethod
    def restore(
        self, job_id: str, group: str, backend: Union[str, int] = 0
    ) -> RestoreResponse:
        """Triggers an asynchronous restoration."""
        pass

    @abstractmethod
    def get_operation(self, operation_id: str) -> GetOperationResponse:
        """Polls the status of a long-running operation."""
        pass

    @abstractmethod
    def status(self) -> StatusResponse:
        """Returns the current state of jobs and accelerators."""
        pass

    @abstractmethod
    def check_health(self, service: str = "") -> HealthResponse:
        """Checks the health of the service using gRPC Health Checking Protocol."""
        pass


class SnapshotAgentClient(SnapshotAgentInterface):
    """
    Client for SnapshotAgentService following gRPC wrapper patterns.
    Provides an idiomatic Python interface to the gRPC service.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        channel: Optional[grpc.Channel] = None,
    ):
        """
        Initialize the client.
        Args:
            endpoint: gRPC endpoint (e.g., 'localhost:9001'). If provided, a new channel is created.
            channel: An existing gRPC channel to use. If provided, 'endpoint' is ignored.
        """
        if channel:
            self.channel = channel
            self._own_channel = False
        elif endpoint:
            self.channel = grpc.insecure_channel(endpoint)
            self._own_channel = True
        else:
            raise ValueError("Either 'endpoint' or 'channel' must be provided.")

        self.stub = snapshot_agent_pb2_grpc.SnapshotAgentServiceStub(self.channel)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)

    def close(self):
        """Close the gRPC channel if it was created by this client."""
        if self._own_channel and self.channel:
            self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _handle_rpc_error(self, e: grpc.RpcError, method_name: str):
        """Maps gRPC errors to SnapshotAgentError."""
        message = f"gRPC error calling {method_name}: {e.code()} - {e.details()}"
        logger.error(message)
        raise SnapshotAgentError(message, code=e.code(), details=e.details()) from e

    def _get_backend_enum(self, backend: Union[str, int]) -> int:
        """Maps backend string or int to the Backend enum value."""
        if isinstance(backend, int):
            return backend
        try:
            # Try exact match first
            return snapshot_agent_pb2.Backend.Value(backend)
        except ValueError:
            # Try with prefix
            try:
                return snapshot_agent_pb2.Backend.Value(f"BACKEND_{backend.upper()}")
            except ValueError:
                logger.warning(
                    f"Unknown backend '{backend}', using BACKEND_UNSPECIFIED"
                )
                return snapshot_agent_pb2.BACKEND_UNSPECIFIED

    def snapshot(
        self, job_id: str, group: str, backend: Union[str, int] = 0
    ) -> SnapshotResponse:
        """
        Triggers an asynchronous snapshot.
        Args:
            job_id: ID of the job to snapshot.
            group: Group for the snapshot.
            backend: Backend to use (e.g., 'CUDA' or snapshot_agent_pb2.BACKEND_CUDA).
        Returns:
            SnapshotResponse containing the operation_id.
        Raises:
            SnapshotAgentError on gRPC or unexpected errors.
        """
        try:
            backend_enum = self._get_backend_enum(backend)
            request = snapshot_agent_pb2.SnapshotRequest(
                job_id=job_id, group=group, backend=backend_enum
            )
            logger.info(
                f"Calling Snapshot with job_id={job_id}, group={group}, backend={backend_enum}..."
            )
            response = self.stub.Snapshot(request)
            return SnapshotResponse(operation_id=response.operation_id)
        except grpc.RpcError as e:
            self._handle_rpc_error(e, "Snapshot")
        except Exception as e:
            logger.error(f"Unexpected error in Snapshot: {e}")
            raise SnapshotAgentError(f"Unexpected error: {e}") from e

    def restore(
        self, job_id: str, group: str, backend: Union[str, int] = 0
    ) -> RestoreResponse:
        """
        Triggers an asynchronous restoration.
        Args:
            job_id: ID of the job to restore.
            group: Group for the restoration.
            backend: Backend to use.
        Returns:
            RestoreResponse containing the operation_id.
        Raises:
            SnapshotAgentError on gRPC or unexpected errors.
        """
        try:
            backend_enum = self._get_backend_enum(backend)
            request = snapshot_agent_pb2.RestoreRequest(
                job_id=job_id, group=group, backend=backend_enum
            )
            logger.info(
                f"Calling Restore with job_id={job_id}, group={group}, backend={backend_enum}..."
            )
            response = self.stub.Restore(request)
            return RestoreResponse(operation_id=response.operation_id)
        except grpc.RpcError as e:
            self._handle_rpc_error(e, "Restore")
        except Exception as e:
            logger.error(f"Unexpected error in Restore: {e}")
            raise SnapshotAgentError(f"Unexpected error: {e}") from e

    def get_operation(self, operation_id: str) -> GetOperationResponse:
        """
        Polls the status of a long-running operation.
        Returns:
            GetOperationResponse containing operation status and metadata.
        """
        try:
            request = snapshot_agent_pb2.GetOperationRequest(operation_id=operation_id)
            response = self.stub.GetOperation(request)

            return GetOperationResponse(
                status=snapshot_agent_pb2.OperationStatus.Name(response.status),
                elapsed_ms=response.elapsed_ms,
                storage_bytes=response.storage_bytes
                if response.HasField("storage_bytes")
                else None,
                snapshot_device_bytes=response.snapshot_device_bytes
                if response.HasField("snapshot_device_bytes")
                else None,
                error=response.error if response.HasField("error") else None,
            )
        except grpc.RpcError as e:
            self._handle_rpc_error(e, "GetOperation")
        except Exception as e:
            logger.error(f"Unexpected error in GetOperation: {e}")
            raise SnapshotAgentError(f"Unexpected error: {e}") from e

    def status(self) -> StatusResponse:
        """
        Returns the current state of jobs and accelerators.
        Returns:
            StatusResponse with job_statuses and accelerator_statuses lists.
        """
        try:
            request = snapshot_agent_pb2.StatusRequest()
            response = self.stub.Status(request)

            job_statuses = [
                JobStatus(
                    job_id=js.job_id,
                    state=snapshot_agent_pb2.JobState.Name(js.state),
                )
                for js in response.job_statuses
            ]

            accelerator_statuses = [
                AcceleratorStatus(
                    id=as_.id,
                    memory_used_bytes=as_.memory_used_bytes,
                    memory_total_bytes=as_.memory_total_bytes,
                )
                for as_ in response.accelerator_statuses
            ]

            return StatusResponse(
                job_statuses=job_statuses,
                accelerator_statuses=accelerator_statuses,
            )
        except grpc.RpcError as e:
            self._handle_rpc_error(e, "Status")
        except Exception as e:
            logger.error(f"Unexpected error in Status: {e}")
            raise SnapshotAgentError(f"Unexpected error: {e}") from e

    def check_health(self, service: str = "") -> HealthResponse:
        """
        Checks the health of the service using gRPC Health Checking Protocol.
        Args:
            service: The service name to check. For snapshot-agent, this can be
                     a backend type (e.g., 'cuda') or empty for the default backend.
        Returns:
            HealthResponse containing the serving status.
        """
        try:
            request = health_pb2.HealthCheckRequest(service=service)
            response = self.health_stub.Check(request)
            return HealthResponse(
                status=health_pb2.HealthCheckResponse.ServingStatus.Name(
                    response.status
                )
            )
        except grpc.RpcError as e:
            self._handle_rpc_error(e, "CheckHealth")
        except Exception as e:
            logger.error(f"Unexpected error in CheckHealth: {e}")
            raise SnapshotAgentError(f"Unexpected error: {e}") from e

    # Facade helper methods for convenience

    def wait_for_operation(
        self, operation_id: str, poll_interval_sec: float = 1.0
    ) -> GetOperationResponse:
        """
        Wait for an operation to complete or fail.
        Args:
            operation_id: ID of the operation to poll.
            poll_interval_sec: Time to wait between polls.
        Returns:
            The final GetOperationResponse.
        """
        while True:
            response = self.get_operation(operation_id)
            if response.status in [
                "OPERATION_STATUS_COMPLETE",
                "OPERATION_STATUS_FAILED",
            ]:
                return response
            time.sleep(poll_interval_sec)

    def snapshot_and_wait(
        self,
        job_id: str,
        group: str,
        backend: Union[str, int] = 0,
        poll_interval_sec: float = 1.0,
    ) -> GetOperationResponse:
        """Calls snapshot and waits for completion."""
        response = self.snapshot(job_id, group, backend)
        return self.wait_for_operation(response.operation_id, poll_interval_sec)

    def restore_and_wait(
        self,
        job_id: str,
        group: str,
        backend: Union[str, int] = 0,
        poll_interval_sec: float = 1.0,
    ) -> GetOperationResponse:
        """Calls restore and waits for completion."""
        response = self.restore(job_id, group, backend)
        return self.wait_for_operation(response.operation_id, poll_interval_sec)
