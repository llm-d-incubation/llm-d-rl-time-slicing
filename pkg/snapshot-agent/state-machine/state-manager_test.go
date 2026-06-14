package statemachine

import (
	"errors"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestNewStateManager(t *testing.T) {
	sm := NewStateManager()
	if sm == nil {
		t.Fatal("NewStateManager returned nil")
	}
	if sm.jobs == nil {
		t.Error("sm.jobs map not initialized")
	}
	if sm.operations == nil {
		t.Error("sm.operations map not initialized")
	}
}

func TestGetOrCreateJob(t *testing.T) {
	sm := NewStateManager()
	jobID := "test-job"
	group := "test-group"
	groupTwo := "test-group-2"

	sm.mu.Lock()
	job1 := sm.getOrCreateJob(jobID, group)
	sm.mu.Unlock()

	if job1 == nil {
		t.Fatal("getOrCreateJob returned nil")
	}
	if job1.State != pb.JobState_JOB_STATE_IDLE {
		t.Errorf("Unexpected job fields: %+v", job1)
	}

	sm.mu.Lock()
	job2 := sm.getOrCreateJob(jobID, groupTwo) // Group should not be updated if already exists
	sm.mu.Unlock()

	if job2 != job1 {
		t.Error("getOrCreateJob returned a different instance for existing jobID")
	}
	if job2.Group != group {
		t.Errorf("Expected group %s, got %s", group, job2.Group)
	}
}

func TestStartSnapshot(t *testing.T) {
	tests := []struct {
		name          string
		initialState  pb.JobState
		workerErr     error
		expectErrCode codes.Code
		expectOp      bool
		finalState    pb.JobState
	}{
		{
			name:         "Success from IDLE",
			initialState: pb.JobState_JOB_STATE_IDLE,
			expectOp:     true,
			finalState:   pb.JobState_JOB_STATE_SAVED,
		},
		{
			name:          "Fails when TRANSITIONING",
			initialState:  pb.JobState_JOB_STATE_TRANSITIONING,
			expectErrCode: codes.Aborted,
		},
		{
			name:          "Fails when FAULTED",
			initialState:  pb.JobState_JOB_STATE_FAULTED,
			expectErrCode: codes.FailedPrecondition,
		},
		{
			name:         "Worker failure leads to FAULTED",
			initialState: pb.JobState_JOB_STATE_IDLE,
			workerErr:    errors.New("worker failed"),
			expectOp:     true,
			finalState:   pb.JobState_JOB_STATE_FAULTED,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sm := NewStateManager()
			jobID := "job-1"
			group := "group-1"

			sm.mu.Lock()
			job := sm.getOrCreateJob(jobID, group)
			job.State = tc.initialState
			sm.mu.Unlock()

			worker := func() error {
				return tc.workerErr
			}

			opID, err := sm.StartSnapshot(jobID, group, worker)

			if tc.expectErrCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok || st.Code() != tc.expectErrCode {
					t.Fatalf("Expected error code %v, got %v", tc.expectErrCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if tc.expectOp {
				if opID == "" {
					t.Error("Expected opID, got empty string")
				}

				op := waitForOperation(t, sm, opID)

				expectedStatus := pb.OperationStatus_OPERATION_STATUS_COMPLETE
				expectedErr := ""
				if tc.workerErr != nil {
					expectedStatus = pb.OperationStatus_OPERATION_STATUS_FAILED
					expectedErr = tc.workerErr.Error()
				}
				checkOperationStatus(t, op, expectedStatus, expectedErr)

				if tc.workerErr == nil && op.StorageBytes != 1024 {
					t.Errorf("Expected StorageBytes 1024, got %d", op.StorageBytes)
				}

				checkJobState(t, sm, jobID, tc.finalState)
			}
		})
	}
}

func TestStartRestore(t *testing.T) {
	tests := []struct {
		name          string
		initialState  pb.JobState
		workerErr     error
		expectErrCode codes.Code
		expectOpID    string // special case for "already-running"
		finalState    pb.JobState
	}{
		{
			name:         "Success from SAVED",
			initialState: pb.JobState_JOB_STATE_SAVED,
			finalState:   pb.JobState_JOB_STATE_RUNNING,
		},
		{
			name:         "Already running",
			initialState: pb.JobState_JOB_STATE_RUNNING,
			expectOpID:   "already-running",
			finalState:   pb.JobState_JOB_STATE_RUNNING,
		},
		{
			name:          "Fails when TRANSITIONING",
			initialState:  pb.JobState_JOB_STATE_TRANSITIONING,
			expectErrCode: codes.Aborted,
		},
		{
			name:          "Fails when FAULTED",
			initialState:  pb.JobState_JOB_STATE_FAULTED,
			expectErrCode: codes.FailedPrecondition,
		},
		{
			name:         "Worker failure leads to FAULTED",
			initialState: pb.JobState_JOB_STATE_SAVED,
			workerErr:    errors.New("restore failed"),
			finalState:   pb.JobState_JOB_STATE_FAULTED,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sm := NewStateManager()
			jobID := "job-1"
			group := "group-1"

			sm.mu.Lock()
			job := sm.getOrCreateJob(jobID, group)
			job.State = tc.initialState
			sm.mu.Unlock()

			worker := func() error {
				return tc.workerErr
			}

			opID, err := sm.StartRestore(jobID, group, worker)

			if tc.expectErrCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok || st.Code() != tc.expectErrCode {
					t.Fatalf("Expected error code %v, got %v", tc.expectErrCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if tc.expectOpID != "" {
				if opID != tc.expectOpID {
					t.Errorf("Expected opID %s, got %s", tc.expectOpID, opID)
				}
				return
			}

			op := waitForOperation(t, sm, opID)

			expectedStatus := pb.OperationStatus_OPERATION_STATUS_COMPLETE
			expectedErr := ""
			if tc.workerErr != nil {
				expectedStatus = pb.OperationStatus_OPERATION_STATUS_FAILED
				expectedErr = tc.workerErr.Error()
			}
			checkOperationStatus(t, op, expectedStatus, expectedErr)

			if tc.workerErr == nil && op.SnapshotDeviceBytes != 1024 {
				t.Errorf("Expected SnapshotDeviceBytes 1024, got %d", op.SnapshotDeviceBytes)
			}

			checkJobState(t, sm, jobID, tc.finalState)
		})
	}
}

func TestGetJobStatus(t *testing.T) {
	sm := NewStateManager()

	if len(sm.GetJobStatus()) != 0 {
		t.Error("Expected 0 jobs")
	}

	sm.mu.Lock()
	sm.getOrCreateJob("job1", "group1").State = pb.JobState_JOB_STATE_RUNNING
	sm.getOrCreateJob("job2", "group1").State = pb.JobState_JOB_STATE_SAVED
	sm.mu.Unlock()

	statuses := sm.GetJobStatus()
	if len(statuses) != 2 {
		t.Errorf("Expected 2 jobs, got %d", len(statuses))
	}

	checkJobState(t, sm, "job1", pb.JobState_JOB_STATE_RUNNING)
	checkJobState(t, sm, "job2", pb.JobState_JOB_STATE_SAVED)
}

func TestJobPIDs(t *testing.T) {
	sm := NewStateManager()
	jobID := "job1"
	pids := []int{100, 200, 300}

	sm.UpdateJobPIDs(jobID, pids)
	_, err := sm.GetJobPIDs(jobID)
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound for non-existent job, got %v", err)
	}

	sm.mu.Lock()
	sm.getOrCreateJob(jobID, "group1")
	sm.mu.Unlock()

	sm.UpdateJobPIDs(jobID, pids)
	gotPids, err := sm.GetJobPIDs(jobID)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(gotPids) != len(pids) {
		t.Errorf("Expected %d pids, got %d", len(pids), len(gotPids))
	}
	for i := range pids {
		if gotPids[i] != pids[i] {
			t.Errorf("PID mismatch at %d: expected %d, got %d", i, pids[i], gotPids[i])
		}
	}

	gotPids[0] = 999
	gotPids2, _ := sm.GetJobPIDs(jobID)
	if gotPids2[0] == 999 {
		t.Error("GetJobPIDs returned a pointer to internal slice, not a copy")
	}

	sm.UpdateJobPIDs(jobID, []int{})
	_, err = sm.GetJobPIDs(jobID)
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound for empty PIDs, got %v", err)
	}
}

func TestConcurrencyControl(t *testing.T) {
	sm := NewStateManager()
	jobID := "concurrent-job"
	group := "group-1"

	// 1. Start a "slow" worker that blocks until we tell it to finish
	blockWorker := make(chan struct{})
	workerStarted := make(chan struct{})
	
	slowWorker := func() error {
		close(workerStarted)
		<-blockWorker
		return nil
	}

	opID1, err := sm.StartSnapshot(jobID, group, slowWorker)
	if err != nil {
		t.Fatalf("Failed to start first operation: %v", err)
	}

	// Wait for the first worker to actually start and transition the state
	select {
	case <-workerStarted:
	case <-time.After(1 * time.Second):
		t.Fatal("Slow worker failed to start")
	}

	// 2. Attempt to start multiple other operations simultaneously
	const concurrentAttempts = 10
	errs := make(chan error, concurrentAttempts*2)

	for i := 0; i < concurrentAttempts; i++ {
		go func() {
			_, err := sm.StartSnapshot(jobID, group, func() error { return nil })
			errs <- err
		}()
		go func() {
			_, err := sm.StartRestore(jobID, group, func() error { return nil })
			errs <- err
		}()
	}

	// 3. Verify that all attempts failed with Aborted (since job is TRANSITIONING)
	for i := 0; i < concurrentAttempts*2; i++ {
		err := <-errs
		if err == nil {
			t.Errorf("Attempt %d should have failed, but succeeded", i)
			continue
		}
		if status.Code(err) != codes.Aborted {
			t.Errorf("Attempt %d: expected Aborted, got %v", i, err)
		}
	}

	// 4. Release the first worker and verify completion
	close(blockWorker)

	op := waitForOperation(t, sm, opID1)
	if op.Status != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
		t.Errorf("First operation failed: %v", op.Error)
	}
}

func TestGetOperation(t *testing.T) {
	sm := NewStateManager()
	opID := "op-1"

	_, ok := sm.GetOperation(opID)
	if ok {
		t.Error("Expected ok=false for non-existent operation")
	}

	op := &Operation{ID: opID, Status: pb.OperationStatus_OPERATION_STATUS_PENDING}
	sm.mu.Lock()
	sm.operations[opID] = op
	sm.mu.Unlock()

	gotOp, ok := sm.GetOperation(opID)
	if !ok {
		t.Fatal("Expected ok=true for existing operation")
	}
	if gotOp.ID != opID {
		t.Errorf("Expected opID %s, got %s", opID, gotOp.ID)
	}
}

func waitForOperation(t *testing.T, sm *StateManager, opID string) *Operation {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	var op *Operation
	var ok bool
	for time.Now().Before(deadline) {
		op, ok = sm.GetOperation(opID)
		if ok && !op.FinishedAt.IsZero() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if !ok || op.FinishedAt.IsZero() {
		t.Fatalf("Timeout waiting for operation %s completion", opID)
	}

	if op.StartedAt.IsZero() {
		t.Errorf("Operation %s StartedAt not set", opID)
	}
	return op
}

func checkJobState(t *testing.T, sm *StateManager, jobID string, expectedState pb.JobState) {
	t.Helper()
	statuses := sm.GetJobStatus()
	for _, s := range statuses {
		if s.JobId == jobID {
			if s.State != expectedState {
				t.Errorf("Expected job %s state %v, got %v", jobID, expectedState, s.State)
			}
			return
		}
	}
	t.Errorf("Job %s status not found", jobID)
}

func checkOperationStatus(t *testing.T, op *Operation, expected pb.OperationStatus, expectedErr string) {
	t.Helper()
	if op.Status != expected {
		t.Errorf("Operation %s: expected status %v, got %v", op.ID, expected, op.Status)
	}
	if op.Error != expectedErr {
		t.Errorf("Operation %s: expected error %q, got %q", op.ID, expectedErr, op.Error)
	}
}
