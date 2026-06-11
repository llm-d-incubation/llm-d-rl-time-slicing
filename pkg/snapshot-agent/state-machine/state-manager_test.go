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

	// Test creation
	sm.mu.Lock()
	job1 := sm.getOrCreateJob(jobID, group)
	sm.mu.Unlock()

	if job1 == nil {
		t.Fatal("getOrCreateJob returned nil")
	}
	if job1.ID != jobID || job1.Group != group || job1.State != pb.JobState_JOB_STATE_IDLE {
		t.Errorf("Unexpected job fields: %+v", job1)
	}

	// Test retrieval
	sm.mu.Lock()
	job2 := sm.getOrCreateJob(jobID, "different-group") // Group should not be updated if already exists
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
			workerErr:    nil,
			expectOp:     true,
			finalState:   pb.JobState_JOB_STATE_SAVED,
		},
		{
			name:          "Fails when TRANSITIONING",
			initialState:  pb.JobState_JOB_STATE_TRANSITIONING,
			expectErrCode: codes.Aborted,
			expectOp:      false,
		},
		{
			name:          "Fails when FAULTED",
			initialState:  pb.JobState_JOB_STATE_FAULTED,
			expectErrCode: codes.FailedPrecondition,
			expectOp:      false,
		},
		{
			name:         "Worker failure leads to FAULTED",
			initialState: pb.JobState_JOB_STATE_IDLE,
			workerErr:    errors.New("worker failed"),
			expectOp:     true,
			finalState:   pb.JobState_JOB_STATE_FAULTED,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := NewStateManager()
			jobID := "job-1"
			group := "group-1"

			// Setup initial state
			sm.mu.Lock()
			job := sm.getOrCreateJob(jobID, group)
			job.State = tt.initialState
			sm.mu.Unlock()

			done := make(chan struct{})
			worker := func() error {
				defer close(done)
				return tt.workerErr
			}

			opID, err := sm.StartSnapshot(jobID, group, worker)

			if tt.expectErrCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok || st.Code() != tt.expectErrCode {
					t.Fatalf("Expected error code %v, got %v", tt.expectErrCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if tt.expectOp {
				if opID == "" {
					t.Error("Expected opID, got empty string")
				}

				// Wait for worker to finish
				select {
				case <-done:
				case <-time.After(1 * time.Second):
					t.Fatal("Timeout waiting for worker")
				}

				// Small delay to allow async update to finish (though done is closed inside worker, 
				// the mutex unlock in StartSnapshot's goroutine happens AFTER worker returns)
				// Wait, the goroutine is:
				/*
					go func() {
						err := worker()
						job.mu.Lock()
						defer job.mu.Unlock()
						sm.mu.Lock()
						defer sm.mu.Unlock()
						...
					}()
				*/
				// So we should wait a bit or use another sync mechanism.
				// Let's polling for the state change.
				
				deadline := time.Now().Add(1 * time.Second)
				for time.Now().Before(deadline) {
					sm.mu.RLock()
					op, ok := sm.operations[opID]
					sm.mu.RUnlock()
					
					if ok && op.Status != pb.OperationStatus_OPERATION_STATUS_PENDING {
						break
					}
					time.Sleep(10 * time.Millisecond)
				}

				op, ok := sm.GetOperation(opID)
				if !ok {
					t.Fatalf("Operation %s not found", opID)
				}

				if op.StartedAt.IsZero() {
					t.Error("Operation StartedAt not set")
				}
				if op.FinishedAt.IsZero() {
					t.Error("Operation FinishedAt not set")
				}

				if tt.workerErr != nil {
					if op.Status != pb.OperationStatus_OPERATION_STATUS_FAILED {
						t.Errorf("Expected FAILED status, got %v", op.Status)
					}
					if op.Error != tt.workerErr.Error() {
						t.Errorf("Expected error %v, got %s", tt.workerErr, op.Error)
					}
				} else {
					if op.Status != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
						t.Errorf("Expected COMPLETE status, got %v", op.Status)
					}
					if op.StorageBytes != 1024 {
						t.Errorf("Expected StorageBytes 1024, got %d", op.StorageBytes)
					}
				}

				statuses := sm.GetJobStatus()
				var found bool
				for _, s := range statuses {
					if s.JobId == jobID {
						found = true
						if s.State != tt.finalState {
							t.Errorf("Expected job state %v, got %v", tt.finalState, s.State)
						}
					}
				}
				if !found {
					t.Error("Job status not found")
				}
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
			workerErr:    nil,
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

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := NewStateManager()
			jobID := "job-1"
			group := "group-1"

			sm.mu.Lock()
			job := sm.getOrCreateJob(jobID, group)
			job.State = tt.initialState
			sm.mu.Unlock()

			done := make(chan struct{})
			worker := func() error {
				defer close(done)
				return tt.workerErr
			}

			opID, err := sm.StartRestore(jobID, group, worker)

			if tt.expectErrCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok || st.Code() != tt.expectErrCode {
					t.Fatalf("Expected error code %v, got %v", tt.expectErrCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if tt.expectOpID != "" {
				if opID != tt.expectOpID {
					t.Errorf("Expected opID %s, got %s", tt.expectOpID, opID)
				}
				return
			}

			// Wait for worker
			select {
			case <-done:
			case <-time.After(1 * time.Second):
				t.Fatal("Timeout waiting for worker")
			}

			// Polling for completion
			deadline := time.Now().Add(1 * time.Second)
			for time.Now().Before(deadline) {
				op, ok := sm.GetOperation(opID)
				if ok && op.Status != pb.OperationStatus_OPERATION_STATUS_PENDING {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}

			op, ok := sm.GetOperation(opID)
			if !ok {
				t.Fatalf("Operation %s not found", opID)
			}

			if op.StartedAt.IsZero() {
				t.Error("Operation StartedAt not set")
			}
			if op.FinishedAt.IsZero() {
				t.Error("Operation FinishedAt not set")
			}

			if tt.workerErr != nil {
				if op.Status != pb.OperationStatus_OPERATION_STATUS_FAILED {
					t.Errorf("Expected FAILED status, got %v", op.Status)
				}
			} else {
				if op.Status != pb.OperationStatus_OPERATION_STATUS_COMPLETE {
					t.Errorf("Expected COMPLETE status, got %v", op.Status)
				}
				if op.SnapshotDeviceBytes != 1024 {
					t.Errorf("Expected SnapshotDeviceBytes 1024, got %d", op.SnapshotDeviceBytes)
				}
			}

			statuses := sm.GetJobStatus()
			for _, s := range statuses {
				if s.JobId == jobID && s.State != tt.finalState {
					t.Errorf("Expected job state %v, got %v", tt.finalState, s.State)
				}
			}
		})
	}
}

func TestGetJobStatus(t *testing.T) {
	sm := NewStateManager()
	
	// Empty case
	if len(sm.GetJobStatus()) != 0 {
		t.Error("Expected 0 jobs")
	}

	// Multiple jobs
	sm.mu.Lock()
	sm.getOrCreateJob("job1", "group1").State = pb.JobState_JOB_STATE_RUNNING
	sm.getOrCreateJob("job2", "group1").State = pb.JobState_JOB_STATE_SAVED
	sm.mu.Unlock()

	statuses := sm.GetJobStatus()
	if len(statuses) != 2 {
		t.Errorf("Expected 2 jobs, got %d", len(statuses))
	}

	found := make(map[string]pb.JobState)
	for _, s := range statuses {
		found[s.JobId] = s.State
	}

	if found["job1"] != pb.JobState_JOB_STATE_RUNNING {
		t.Errorf("job1 wrong state: %v", found["job1"])
	}
	if found["job2"] != pb.JobState_JOB_STATE_SAVED {
		t.Errorf("job2 wrong state: %v", found["job2"])
	}
}

func TestJobPIDs(t *testing.T) {
	sm := NewStateManager()
	jobID := "job1"
	pids := []int{100, 200, 300}

	// Test Update/Get on non-existent job
	sm.UpdateJobPIDs(jobID, pids)
	_, err := sm.GetJobPIDs(jobID)
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound for non-existent job, got %v", err)
	}

	// Create job and update PIDs
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

	// Test copy behavior (modifying returned slice shouldn't affect internal state)
	gotPids[0] = 999
	gotPids2, _ := sm.GetJobPIDs(jobID)
	if gotPids2[0] == 999 {
		t.Error("GetJobPIDs returned a pointer to internal slice, not a copy")
	}

	// Test no PIDs case
	sm.UpdateJobPIDs(jobID, []int{})
	_, err = sm.GetJobPIDs(jobID)
	if status.Code(err) != codes.NotFound {
		t.Errorf("Expected NotFound for empty PIDs, got %v", err)
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
