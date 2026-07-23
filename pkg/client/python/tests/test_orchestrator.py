import datetime
import json
import unittest
from unittest.mock import MagicMock, patch

import grpc

from timeslice import TimeSliceOrchestratorClient
from timeslice.orchestrator.client import DEFAULT_CHANNEL_OPTIONS
from timeslice.orchestrator.exceptions import (
    OrchestratorConnectionError,
    OrchestratorTimeoutError,
)
from timeslice.orchestrator.types import AgentJobState, GroupLockState

from timeslice.orchestrator._generated import pb2


class TestTimeSliceOrchestratorClient(unittest.TestCase):
    def setUp(self):
        self.target = "localhost:50051"
        self.job_id = "test-job"
        self.group_id = "test-group"

        # Patch the gRPC channel and stub
        self.mock_channel_patcher = patch(
            "timeslice.orchestrator.client.grpc.insecure_channel"
        )
        self.mock_stub_patcher = patch(
            "timeslice.orchestrator.client.pb2_grpc.AcceleratorOrchestratorServiceStub"
        )

        self.mock_insecure_channel = self.mock_channel_patcher.start()
        self.mock_stub_class = self.mock_stub_patcher.start()

        self.mock_stub = MagicMock()
        self.mock_stub_class.return_value = self.mock_stub

        self.client = TimeSliceOrchestratorClient(
            self.target, self.job_id, self.group_id
        )

    def tearDown(self):
        self.client.close()
        self.mock_channel_patcher.stop()
        self.mock_stub_patcher.stop()

    def test_init(self):
        self.mock_insecure_channel.assert_called_once_with(
            self.target, options=DEFAULT_CHANNEL_OPTIONS
        )
        self.mock_stub_class.assert_called_once_with(
            self.mock_insecure_channel.return_value
        )
        self.assertEqual(self.client.job_id, self.job_id)
        self.assertEqual(self.client.group_id, self.group_id)

    def test_default_retry_status_codes(self):
        # Verify DEFAULT_CHANNEL_OPTIONS configures retryable status codes
        self.assertEqual(len(DEFAULT_CHANNEL_OPTIONS), 1)
        key, service_config_str = DEFAULT_CHANNEL_OPTIONS[0]
        self.assertEqual(key, "grpc.service_config")
        config = json.loads(service_config_str)
        retry_policy = config["methodConfig"][0]["retryPolicy"]
        self.assertEqual(
            retry_policy["retryableStatusCodes"],
            ["UNAVAILABLE", "RESOURCE_EXHAUSTED", "ABORTED", "DEADLINE_EXCEEDED"],
        )

    def test_init_custom_channel_options_override(self):
        custom_options = [("grpc.max_receive_message_length", 1024)]
        client = TimeSliceOrchestratorClient(
            self.target, channel_options=custom_options
        )
        self.mock_insecure_channel.assert_called_with(
            self.target, options=custom_options
        )
        client.close()

    def test_acquire_success(self):
        # Mock Acquire response
        mock_response = pb2.AcquireResponse(
            success=True, waited_ms=150, context_restored=True
        )
        self.mock_stub.Acquire.return_value = mock_response

        result = self.client.acquire(timeout_sec=10.0)

        # Verify call
        self.mock_stub.Acquire.assert_called_once()
        args, kwargs = self.mock_stub.Acquire.call_args
        request = args[0]
        self.assertEqual(request.job_id, self.job_id)
        self.assertEqual(request.group_id, self.group_id)
        self.assertEqual(kwargs.get("timeout"), 10.0)

        # Verify result mapping
        self.assertTrue(result.success)
        self.assertEqual(result.waited_ms, 150)
        self.assertTrue(result.context_restored)

    def test_release_success(self):
        # Mock Yield response
        mock_response = pb2.YieldResponse(
            success=True, pending_waiters=2, snapshot_deferred=False
        )
        self.mock_stub.Yield.return_value = mock_response

        result = self.client.release()

        # Verify call
        self.mock_stub.Yield.assert_called_once()
        args, _ = self.mock_stub.Yield.call_args
        request = args[0]
        self.assertEqual(request.job_id, self.job_id)
        self.assertEqual(request.group_id, self.group_id)

        # Verify result mapping
        self.assertTrue(result.success)
        self.assertEqual(result.pending_waiters, 2)
        self.assertFalse(result.snapshot_deferred)

    def test_get_status_success(self):
        # Mock GetGroupStatus response
        from google.protobuf.timestamp_pb2 import Timestamp

        proto_ts = Timestamp()
        proto_ts.FromDatetime(
            datetime.datetime(2026, 6, 24, 12, 0, 0, tzinfo=datetime.timezone.utc)
        )

        mock_response = pb2.GetGroupStatusResponse(
            group=pb2.GroupStatus(
                group_id=self.group_id,
                group_state=pb2.GroupStatus.State.STATE_LOCKED,
                state_timestamp=proto_ts,
                locking_job="some-other-job",
                active_job="some-other-job",
                waiter_queue_depth=3,
                loaded_job="some-other-job",
            ),
            agent_job_states=[
                pb2.SnapshotAgentJobState(
                    agent="node-1",
                    job_id="some-other-job",
                    job_state=pb2.SnapshotAgentJobState.State.STATE_RUNNING,
                ),
                pb2.SnapshotAgentJobState(
                    agent="node-2",
                    job_id="some-other-job",
                    job_state=pb2.SnapshotAgentJobState.State.STATE_SAVED,
                ),
            ],
        )
        self.mock_stub.GetGroupStatus.return_value = mock_response

        status = self.client.get_status()

        # Verify call
        self.mock_stub.GetGroupStatus.assert_called_once()
        args, _ = self.mock_stub.GetGroupStatus.call_args
        request = args[0]
        self.assertEqual(request.group_id, self.group_id)

        # Verify group mapping
        self.assertEqual(status.group.group_id, self.group_id)
        self.assertEqual(status.group.group_state, GroupLockState.LOCKED)
        self.assertEqual(status.group.locking_job, "some-other-job")
        self.assertEqual(status.group.waiter_queue_depth, 3)
        self.assertEqual(status.group.state_timestamp.year, 2026)

        # Verify agent states mapping
        self.assertEqual(len(status.agent_job_states), 2)
        self.assertEqual(status.agent_job_states[0].agent, "node-1")
        self.assertEqual(status.agent_job_states[0].job_state, AgentJobState.RUNNING)
        self.assertEqual(status.agent_job_states[1].agent, "node-2")
        self.assertEqual(status.agent_job_states[1].job_state, AgentJobState.SAVED)

    def test_list_groups_success(self):
        mock_response = pb2.ListGroupsResponse(group_ids=["group-1", "group-2"])
        self.mock_stub.ListGroups.return_value = mock_response

        groups = self.client.list_groups()

        self.mock_stub.ListGroups.assert_called_once()
        self.assertEqual(groups, ["group-1", "group-2"])

    def test_on_accelerators_context_manager(self):
        # Mock acquire and release
        self.client.acquire = MagicMock()
        self.client.release = MagicMock()

        with self.client.on_accelerators(timeout_sec=5.0) as result:
            self.assertEqual(result, self.client.acquire.return_value)
            self.client.acquire.assert_called_once_with(
                job_id=None, group_id=None, timeout_sec=5.0
            )
            self.client.release.assert_not_called()

        self.client.release.assert_called_once_with(job_id=None, group_id=None)

    def test_on_accelerators_context_manager_exception(self):
        # Verify release is called even if block raises error
        self.client.acquire = MagicMock()
        self.client.release = MagicMock()

        try:
            with self.client.on_accelerators():
                raise ValueError("Something went wrong")
        except ValueError:
            pass

        self.client.acquire.assert_called_once_with(
            job_id=None, group_id=None, timeout_sec=None
        )
        self.client.release.assert_called_once_with(job_id=None, group_id=None)

    def test_grpc_error_mapping_unavailable(self):
        # Mock gRPC error
        mock_error = grpc.RpcError()
        # We need to mock the code and details methods
        mock_error.code = MagicMock(return_value=grpc.StatusCode.UNAVAILABLE)
        mock_error.details = MagicMock(return_value="Server down")

        self.mock_stub.Acquire.side_effect = mock_error

        with self.assertRaises(OrchestratorConnectionError):
            self.client.acquire()

    def test_grpc_error_mapping_timeout(self):
        mock_error = grpc.RpcError()
        mock_error.code = MagicMock(return_value=grpc.StatusCode.DEADLINE_EXCEEDED)
        mock_error.details = MagicMock(return_value="Timed out")

        self.mock_stub.Acquire.side_effect = mock_error

        with self.assertRaises(OrchestratorTimeoutError):
            self.client.acquire()

    def test_init_optional_args(self):
        # We can create a client with only target
        client = TimeSliceOrchestratorClient(self.target)
        self.assertIsNone(client.job_id)
        self.assertIsNone(client.group_id)
        client.close()

    def test_acquire_missing_args_raises_value_error(self):
        client = TimeSliceOrchestratorClient(self.target)

        # Missing both
        with self.assertRaises(ValueError) as ctx:
            client.acquire()
        self.assertIn("job_id must be provided", str(ctx.exception))

        # Missing group_id
        with self.assertRaises(ValueError) as ctx:
            client.acquire(job_id="some-job")
        self.assertIn("group_id must be provided", str(ctx.exception))

        client.close()

    def test_acquire_override_args(self):
        # Mock Acquire response
        mock_response = pb2.AcquireResponse(
            success=True, waited_ms=50, context_restored=False
        )
        self.mock_stub.Acquire.return_value = mock_response

        # Call acquire with overrides
        result = self.client.acquire(job_id="override-job", group_id="override-group")

        # Verify call used overrides instead of constructor values ("test-job", "test-group")
        self.mock_stub.Acquire.assert_called_once()
        args, _ = self.mock_stub.Acquire.call_args
        request = args[0]
        self.assertEqual(request.job_id, "override-job")
        self.assertEqual(request.group_id, "override-group")
        self.assertTrue(result.success)

    def test_release_override_args(self):
        mock_response = pb2.YieldResponse(
            success=True, pending_waiters=0, snapshot_deferred=True
        )
        self.mock_stub.Yield.return_value = mock_response

        result = self.client.release(job_id="override-job", group_id="override-group")

        self.mock_stub.Yield.assert_called_once()
        args, _ = self.mock_stub.Yield.call_args
        request = args[0]
        self.assertEqual(request.job_id, "override-job")
        self.assertEqual(request.group_id, "override-group")
        self.assertTrue(result.success)
        self.assertEqual(result.pending_waiters, 0)
        self.assertTrue(result.snapshot_deferred)

    def test_get_status_override_args(self):
        mock_response = pb2.GetGroupStatusResponse(
            group=pb2.GroupStatus(
                group_id="override-group", group_state=pb2.GroupStatus.State.STATE_IDLE
            )
        )
        self.mock_stub.GetGroupStatus.return_value = mock_response

        status = self.client.get_status(group_id="override-group")

        self.mock_stub.GetGroupStatus.assert_called_once()
        args, _ = self.mock_stub.GetGroupStatus.call_args
        request = args[0]
        self.assertEqual(request.group_id, "override-group")
        self.assertEqual(status.group.group_id, "override-group")

    def test_on_accelerators_override_args(self):
        self.client.acquire = MagicMock()
        self.client.release = MagicMock()

        with self.client.on_accelerators(
            job_id="override-job", group_id="override-group", timeout_sec=5.0
        ) as result:
            self.assertEqual(result, self.client.acquire.return_value)
            self.client.acquire.assert_called_once_with(
                job_id="override-job", group_id="override-group", timeout_sec=5.0
            )
            self.client.release.assert_not_called()

        self.client.release.assert_called_once_with(
            job_id="override-job", group_id="override-group"
        )


if __name__ == "__main__":
    unittest.main()
