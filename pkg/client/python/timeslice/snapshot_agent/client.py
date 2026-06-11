import logging
import time

import grpc

from . import snapshot_agent_pb2, snapshot_agent_pb2_grpc

logger = logging.getLogger(__name__)


class SnapshotAgentClient:
    """Client for SnapshotAgentService."""

    def __init__(self, endpoint: str):
        """
        Initialize the client.
        Args:
            endpoint: gRPC endpoint (e.g., 'localhost:9001')
        """
        self.endpoint = endpoint
        self.channel = grpc.insecure_channel(endpoint)
        self.stub = snapshot_agent_pb2_grpc.SnapshotAgentServiceStub(self.channel)

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def snapshot(self, job_id: str, group: str):
        """
        Calls the Snapshot endpoint of the SnapshotAgentService.
        Args:
            job_id: ID of the job to snapshot.
            group: Group for the snapshot.
        Returns:
            SnapshotResponse or None on error.
        """
        try:
            request = snapshot_agent_pb2.SnapshotRequest(
                job_id=job_id, group=group
            )
            logger.info(f"Calling Snapshot with job_id={job_id}, group={group}...")
            response = self.stub.Snapshot(request)
            logger.info(f"Snapshot Response: operation_id={response.operation_id}")
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error calling Snapshot: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error in Snapshot: {e}")
        return None

    def get_operation(self, operation_id: str):
        """
        Calls the GetOperation endpoint of the SnapshotAgentService.
        Args:
            operation_id: ID of the operation to poll.
        Returns:
            GetOperationResponse or None on error.
        """
        try:
            request = snapshot_agent_pb2.GetOperationRequest(operation_id=operation_id)
            logger.info(f"Calling GetOperation with operation_id={operation_id}...")
            response = self.stub.GetOperation(request)
            logger.info(
                f"GetOperation Response: status={response.status}, elapsed={response.elapsed_ms}ms"
            )
            if response.error:
                logger.error(f"Operation error: {response.error}")
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error calling GetOperation: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error in GetOperation: {e}")
        return None

    def _wait_for_operation(self, operation_id: str, poll_interval_sec: float = 1.0):
        """
        Wait for an operation to complete or fail.
        Args:
            operation_id: ID of the operation to poll.
            poll_interval_sec: Time to wait between polls.
        Returns:
            GetOperationResponse or None on error.
        """
        while True:
            response = self.get_operation(operation_id)
            if not response:
                return None
            if response.status in [
                snapshot_agent_pb2.OPERATION_STATUS_COMPLETE,
                snapshot_agent_pb2.OPERATION_STATUS_FAILED,
            ]:
                return response
            time.sleep(poll_interval_sec)

    def snapshot_and_wait(
        self, job_id: str, group: str, poll_interval_sec: float = 1.0
    ):
        """
        Calls Snapshot and waits for the operation to complete or fail.
        Args:
            job_id: ID of the job to snapshot.
            group: Group for the snapshot.
            poll_interval_sec: Time to wait between polls.
        Returns:
            GetOperationResponse or None on error.
        """
        response = self.snapshot(job_id, group)
        if not response:
            return None
        return self._wait_for_operation(response.operation_id, poll_interval_sec)

    def restore(self, job_id: str, group: str):
        """
        Calls the Restore endpoint of the SnapshotAgentService.
        Args:
            job_id: ID of the job to restore.
            group: Group for the restoration.
        Returns:
            RestoreResponse or None on error.
        """
        try:
            request = snapshot_agent_pb2.RestoreRequest(
                job_id=job_id, group=group
            )
            logger.info(f"Calling Restore with job_id={job_id}, group={group}...")
            response = self.stub.Restore(request)
            logger.info(f"Restore Response: operation_id={response.operation_id}")
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error calling Restore: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error in Restore: {e}")
        return None

    def restore_and_wait(
        self, job_id: str, group: str, poll_interval_sec: float = 1.0
    ):
        """
        Calls Restore and waits for the operation to complete or fail.
        Args:
            job_id: ID of the job to restore.
            group: Group for the restoration.
            poll_interval_sec: Time to wait between polls.
        Returns:
            GetOperationResponse or None on error.
        """
        response = self.restore(job_id, group)
        if not response:
            return None
        return self._wait_for_operation(response.operation_id, poll_interval_sec)

    def status(self):
        """
        Calls the Status endpoint of the SnapshotAgentService.
        Returns:
            StatusResponse or None on error.
        """
        try:
            request = snapshot_agent_pb2.StatusRequest()
            logger.info("Calling Status...")
            response = self.stub.Status(request)
            logger.info(
                f"Status Response: {len(response.job_statuses)} jobs, "
                f"{len(response.accelerator_statuses)} accelerators"
            )
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error calling Status: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error in Status: {e}")
        return None

    def health(self):
        """
        Calls the Health endpoint of the SnapshotAgentService.
        Returns:
            HealthResponse or None on error.
        """
        try:
            request = snapshot_agent_pb2.HealthRequest()
            logger.info("Calling Health...")
            response = self.stub.Health(request)
            logger.info(f"Health Response: healthy={response.healthy}")
            return response
        except grpc.RpcError as e:
            logger.error(f"gRPC error calling Health: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error in Health: {e}")
        return None
