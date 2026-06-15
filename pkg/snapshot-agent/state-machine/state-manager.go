package statemachine

import (
	"sync"
	"time"

	"github.com/google/uuid"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// OpType represents the type of operation (Snapshot or Restore).
type OpType string

const (
	OpTypeSnapshot OpType = "Snapshot"
	OpTypeRestore  OpType = "Restore"
)

// Job represents a per-workload state.
type Job struct {
	ID    string
	Group string
	State pb.JobState
	PIDs  []int
	mu    sync.Mutex
}

// Operation represents a long-running snapshot or restore task.
type Operation struct {
	ID                  string
	JobID               string
	Status              pb.OperationStatus
	Type                OpType
	StartedAt           time.Time
	FinishedAt          time.Time
	Error               string
	StorageBytes        int64
	SnapshotDeviceBytes int64
}

// StateManager handles thread-safe job transitions and operation tracking.
type StateManager struct {
	jobs       map[string]*Job
	operations map[string]*Operation
	mu         sync.RWMutex
}

// NewStateManager creates a new StateManager instance.
func NewStateManager() *StateManager {
	return &StateManager{
		jobs:       make(map[string]*Job),
		operations: make(map[string]*Operation),
	}
}

// getOrCreateJob returns an existing job or creates a new one.
// Must be called with sm.mu held.
func (sm *StateManager) getOrCreateJob(jobID, group string) *Job {
	job, ok := sm.jobs[jobID]
	if !ok {
		job = &Job{
			ID:    jobID,
			Group: group,
			State: pb.JobState_JOB_STATE_IDLE,
		}
		sm.jobs[jobID] = job
	}
	return job
}

// StartSnapshot initiates a snapshot operation if the job state allows it.
func (sm *StateManager) StartSnapshot(jobID, group string, worker func() error) (string, error) {
	sm.mu.Lock()
	job := sm.getOrCreateJob(jobID, group)
	sm.mu.Unlock()

	job.mu.Lock()
	defer job.mu.Unlock()

	// 1. Concurrency Guard
	if job.State == pb.JobState_JOB_STATE_TRANSITIONING {
		return "", status.Errorf(codes.Aborted, "job %s is already transitioning", jobID)
	}

	// 2. Fault Protection
	if job.State == pb.JobState_JOB_STATE_FAULTED {
		return "", status.Errorf(codes.FailedPrecondition, "job %s is in FAULTED state", jobID)
	}

	opID := uuid.New().String()
	op := &Operation{
		ID:        opID,
		JobID:     jobID,
		Status:    pb.OperationStatus_OPERATION_STATUS_PENDING,
		Type:      OpTypeSnapshot,
		StartedAt: time.Now(),
	}

	sm.mu.Lock()
	sm.operations[opID] = op
	sm.mu.Unlock()

	// Update job state to TRANSITIONING
	job.State = pb.JobState_JOB_STATE_TRANSITIONING

	// 3. Asynchronous Workflow
	go func() {
		err := worker()

		sm.mu.Lock()
		defer sm.mu.Unlock()

		job.mu.Lock()
		defer job.mu.Unlock()

		op.FinishedAt = time.Now()
		if err != nil {
			op.Status = pb.OperationStatus_OPERATION_STATUS_FAILED
			op.Error = err.Error()
			job.State = pb.JobState_JOB_STATE_FAULTED
		} else {
			op.Status = pb.OperationStatus_OPERATION_STATUS_COMPLETE
			op.StorageBytes = 1024
			job.State = pb.JobState_JOB_STATE_SAVED
		}
	}()

	return opID, nil
}

// StartRestore initiates a restore operation if the job state allows it.
func (sm *StateManager) StartRestore(jobID, group string, worker func() error) (string, error) {
	sm.mu.Lock()
	job := sm.getOrCreateJob(jobID, group)
	sm.mu.Unlock()

	job.mu.Lock()
	defer job.mu.Unlock()

	// 1. Redundancy Optimization
	if job.State == pb.JobState_JOB_STATE_RUNNING {
		return "already-running", nil
	}

	// 2. Concurrency Guard
	if job.State == pb.JobState_JOB_STATE_TRANSITIONING {
		return "", status.Errorf(codes.Aborted, "job %s is already transitioning", jobID)
	}

	// 3. Fault Protection
	if job.State == pb.JobState_JOB_STATE_FAULTED {
		return "", status.Errorf(codes.FailedPrecondition, "job %s is in FAULTED state", jobID)
	}

	opID := uuid.New().String()
	op := &Operation{
		ID:        opID,
		JobID:     jobID,
		Status:    pb.OperationStatus_OPERATION_STATUS_PENDING,
		Type:      OpTypeRestore,
		StartedAt: time.Now(),
	}

	sm.mu.Lock()
	sm.operations[opID] = op
	sm.mu.Unlock()

	// Update job state to TRANSITIONING
	job.State = pb.JobState_JOB_STATE_TRANSITIONING

	// 4. Asynchronous Workflow
	go func() {
		err := worker()

		sm.mu.Lock()
		defer sm.mu.Unlock()

		job.mu.Lock()
		defer job.mu.Unlock()

		op.FinishedAt = time.Now()
		if err != nil {
			op.Status = pb.OperationStatus_OPERATION_STATUS_FAILED
			op.Error = err.Error()
			job.State = pb.JobState_JOB_STATE_FAULTED
		} else {
			op.Status = pb.OperationStatus_OPERATION_STATUS_COMPLETE
			job.State = pb.JobState_JOB_STATE_RUNNING
			op.SnapshotDeviceBytes = 1024
		}
	}()

	return opID, nil
}

// GetOperation returns the status of a specific operation.
func (sm *StateManager) GetOperation(opID string) (*Operation, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	op, ok := sm.operations[opID]
	if !ok {
		return nil, false
	}
	// Return a copy to avoid race conditions
	copyOp := *op
	return &copyOp, true
}

// GetJobStatus returns the current status of all jobs.
func (sm *StateManager) GetJobStatus() []*pb.JobStatus {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	statuses := make([]*pb.JobStatus, 0, len(sm.jobs))
	for id, job := range sm.jobs {
		job.mu.Lock()
		statuses = append(statuses, &pb.JobStatus{
			JobId: id,
			State: job.State,
		})
		job.mu.Unlock()
	}
	return statuses
}

// UpdateJobPIDs updates the PIDs associated with a job.
func (sm *StateManager) UpdateJobPIDs(jobID string, pids []int) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	job, ok := sm.jobs[jobID]
	if !ok {
		return
	}

	job.mu.Lock()
	defer job.mu.Unlock()
	job.PIDs = pids
}

// GetJobPIDs returns the PIDs associated with a job.
func (sm *StateManager) GetJobPIDs(jobID string) ([]int, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	job, ok := sm.jobs[jobID]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "job %s not found", jobID)
	}

	job.mu.Lock()
	defer job.mu.Unlock()
	if len(job.PIDs) == 0 {
		return nil, status.Errorf(codes.NotFound, "no PIDs found for job %s", jobID)
	}

	// Return a copy to avoid race conditions
	pids := make([]int, len(job.PIDs))
	copy(pids, job.PIDs)
	return pids, nil
}
