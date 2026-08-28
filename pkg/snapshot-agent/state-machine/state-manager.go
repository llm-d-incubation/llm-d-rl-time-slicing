package statemachine

import (
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ErrNoLiveProcesses marks a snapshot worker failure that means the job's
// workload processes no longer exist (e.g. the job completed and released its
// lock with the snapshot deferred, then exited before lazy eviction ran).
// There is nothing left to checkpoint and the accelerator is already free, so
// the snapshot operation completes and the job returns to IDLE instead of
// FAULTED — a FAULTED ghost would block every other job in the group.
var ErrNoLiveProcesses = errors.New("no live workload processes for job")

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
	// Slot is the snapshot slot currently loaded on the device, for backends
	// with named snapshot slots (memory-regions). Empty for backends without
	// slot semantics.
	Slot string
	mu   sync.Mutex
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

	// mu guards jobs and operations.
	// Lock order: mu → Job.mu. The reverse order deadlocks.
	mu sync.RWMutex
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

// RegisterJob registers a new job with IDLE state if it doesn't already exist.
func (sm *StateManager) RegisterJob(jobID, group string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	_ = sm.getOrCreateJob(jobID, group)
}

// StartSnapshot initiates a snapshot operation if the job state allows it.
func (sm *StateManager) StartSnapshot(jobID, group string, worker func() error) (string, error) {
	return sm.StartSnapshotSlot(jobID, group, "", worker)
}

// StartSnapshotSlot is StartSnapshot for backends with named snapshot slots
// (memory-regions): slot names the snapshot being taken and is recorded as
// the job's loaded slot on success. With slot == "" the behavior is exactly
// StartSnapshot's. With a non-empty slot, a FAULTED job may also be
// snapshotted: faults are typically transient (dead workload PID, timed-out
// cr_client) and a fresh attempt should reset the job rather than requiring
// an agent redeploy.
func (sm *StateManager) StartSnapshotSlot(jobID, group, slot string, worker func() error) (string, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	job := sm.getOrCreateJob(jobID, group)

	job.mu.Lock()
	defer job.mu.Unlock()

	// 1. Concurrency Guard
	if job.State == pb.JobState_JOB_STATE_TRANSITIONING {
		return "", status.Errorf(codes.Aborted, "job %s is already transitioning", jobID)
	}

	// 2. Fault Recovery (slot-aware backends only)
	if slot != "" && job.State == pb.JobState_JOB_STATE_FAULTED {
		slog.Warn("Job is FAULTED; allowing new snapshot to reset it", "jobID", jobID, "slot", slot)
	} else if job.State != pb.JobState_JOB_STATE_RUNNING {
		// 3. State Validation: Only allow snapshotting of RUNNING jobs
		return "", status.Errorf(codes.FailedPrecondition, "cannot snapshot job %s in state %s (must be RUNNING)", jobID, job.State)
	}

	opID := uuid.New().String()
	op := &Operation{
		ID:        opID,
		JobID:     jobID,
		Status:    pb.OperationStatus_OPERATION_STATUS_PENDING,
		Type:      OpTypeSnapshot,
		StartedAt: time.Now(),
	}

	sm.operations[opID] = op

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
		switch {
		case errors.Is(err, ErrNoLiveProcesses):
			// The workload exited before this (lazily deferred) snapshot ran.
			// The eviction goal — a free accelerator — is already met, so the
			// operation completes and the job returns to IDLE with no context.
			slog.Warn("Snapshot found no live processes; treating job as exited",
				"jobID", jobID, "error", err)
			op.Status = pb.OperationStatus_OPERATION_STATUS_COMPLETE
			job.State = pb.JobState_JOB_STATE_IDLE
			job.PIDs = nil
			job.Slot = ""
		case err != nil:
			op.Status = pb.OperationStatus_OPERATION_STATUS_FAILED
			op.Error = err.Error()
			job.State = pb.JobState_JOB_STATE_FAULTED
		default:
			op.Status = pb.OperationStatus_OPERATION_STATUS_COMPLETE
			op.StorageBytes = 1024
			job.State = pb.JobState_JOB_STATE_SAVED
			job.Slot = slot
		}
	}()

	return opID, nil
}

// StartRestore initiates a restore operation if the job state allows it.
func (sm *StateManager) StartRestore(jobID, group string, worker func() error) (string, error) {
	return sm.StartRestoreSlot(jobID, group, "", worker)
}

// StartRestoreSlot is StartRestore for backends with named snapshot slots
// (memory-regions). With slot == "" the behavior is exactly StartRestore's.
// With a non-empty slot:
//   - a RUNNING job only short-circuits to "already-running" when the
//     requested slot is already the loaded one; restoring a different slot
//     proceeds (live slot swap, the core memory-regions use case);
//   - a FAULTED job may be restored (fault recovery; see StartSnapshotSlot).
func (sm *StateManager) StartRestoreSlot(jobID, group, slot string, worker func() error) (string, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	job := sm.getOrCreateJob(jobID, group)

	job.mu.Lock()
	defer job.mu.Unlock()

	// 1. Redundancy Optimization: the requested state is already live.
	if job.State == pb.JobState_JOB_STATE_RUNNING && job.Slot == slot {
		return "already-running", nil
	}

	// 2. Concurrency Guard
	if job.State == pb.JobState_JOB_STATE_TRANSITIONING {
		return "", status.Errorf(codes.Aborted, "job %s is already transitioning", jobID)
	}

	// 3. Fault Recovery (slot-aware backends only)
	if slot != "" && job.State == pb.JobState_JOB_STATE_FAULTED {
		slog.Warn("Job is FAULTED; allowing new restore to reset it", "jobID", jobID, "slot", slot)
	} else if job.State != pb.JobState_JOB_STATE_SAVED && (slot == "" || job.State != pb.JobState_JOB_STATE_RUNNING) {
		// 4. State Validation: restores need a SAVED job — or, for
		// slot-aware backends, a RUNNING job swapping to a different slot.
		return "", status.Errorf(codes.FailedPrecondition, "cannot restore job %s in state %s (must be SAVED)", jobID, job.State)
	}

	opID := uuid.New().String()
	op := &Operation{
		ID:        opID,
		JobID:     jobID,
		Status:    pb.OperationStatus_OPERATION_STATUS_PENDING,
		Type:      OpTypeRestore,
		StartedAt: time.Now(),
	}

	sm.operations[opID] = op

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
			job.Slot = slot
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
	job, ok := sm.jobs[jobID]
	sm.mu.Unlock()
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
	job, ok := sm.jobs[jobID]
	sm.mu.RUnlock()
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

// TransitionToRunning transitions a job from IDLE to RUNNING and associates PIDs.
func (sm *StateManager) TransitionToRunning(jobID string, pids []int) error {
	sm.mu.Lock()
	job, ok := sm.jobs[jobID]
	sm.mu.Unlock()
	if !ok {
		return status.Errorf(codes.NotFound, "job %s not found", jobID)
	}

	job.mu.Lock()
	defer job.mu.Unlock()

	if job.State != pb.JobState_JOB_STATE_IDLE {
		return status.Errorf(codes.FailedPrecondition, "job %s is not in IDLE state (current: %s)", jobID, job.State)
	}

	job.State = pb.JobState_JOB_STATE_RUNNING
	job.PIDs = pids
	return nil
}
