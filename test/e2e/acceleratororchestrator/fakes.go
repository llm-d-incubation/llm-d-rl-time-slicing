package acceleratororchestrator

import (
	"context"
	"fmt"
	"sync"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"k8s.io/client-go/util/workqueue"
)

// trackQueue wraps a rate limiting queue and tracks Done() and AddRateLimited() calls.
type trackQueue struct {
	workqueue.TypedRateLimitingInterface[string]
	mu                  sync.Mutex
	doneCount           int
	addRateLimitedCount int
}

func (t *trackQueue) Done(item string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.doneCount++
	t.TypedRateLimitingInterface.Done(item)
}

func (t *trackQueue) AddRateLimited(item string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.addRateLimitedCount++
	t.TypedRateLimitingInterface.AddRateLimited(item)
}

func (t *trackQueue) getDoneCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.doneCount
}

func (t *trackQueue) getAddRateLimitedCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.addRateLimitedCount
}

type pendingOp struct {
	node        string
	job         string
	opType      string // "snapshot" or "restore"
	targetState agentpb.JobState
}

// FakeSnapshotAgentStore simulates the behavior of snapshot agents on nodes.
type FakeSnapshotAgentStore struct {
	mu                sync.Mutex
	jobStates         map[string]map[string]agentpb.JobState // node -> job -> state
	pendingOperations map[string]pendingOp
	opCounter         int

	// Optional hooks for tests to observe events
	OnSnapshot func(node, jobID string)
	OnRestore  func(node, jobID string)
}

func NewFakeSnapshotAgentStore() *FakeSnapshotAgentStore {
	return &FakeSnapshotAgentStore{
		jobStates:         make(map[string]map[string]agentpb.JobState),
		pendingOperations: make(map[string]pendingOp),
	}
}

// SetJobState allows setting the initial state of a job on a node.
func (f *FakeSnapshotAgentStore) SetJobState(node, job string, state agentpb.JobState) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.jobStates[node]; !ok {
		f.jobStates[node] = make(map[string]agentpb.JobState)
	}
	f.jobStates[node][job] = state
}

// GetJobState allows reading the state of a job on a node.
func (f *FakeSnapshotAgentStore) GetJobState(node, job string) agentpb.JobState {
	f.mu.Lock()
	defer f.mu.Unlock()
	if nodes, ok := f.jobStates[node]; ok {
		return nodes[job]
	}
	return agentpb.JobState_JOB_STATE_UNSPECIFIED
}

func (f *FakeSnapshotAgentStore) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	var statuses []*agentpb.JobStatus
	if jobs, ok := f.jobStates[nodeName]; ok {
		for jobID, state := range jobs {
			statuses = append(statuses, &agentpb.JobStatus{
				JobId: jobID,
				State: state,
			})
		}
	}
	return &agentpb.StatusResponse{JobStatuses: statuses}, nil
}

func (f *FakeSnapshotAgentStore) CloseClient(nodeName string) error {
	return nil
}

func (f *FakeSnapshotAgentStore) Snapshot(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.opCounter++
	opID := fmt.Sprintf("op-snap-%d", f.opCounter)
	f.pendingOperations[opID] = pendingOp{
		node:        nodeName,
		job:         jobID,
		opType:      "snapshot",
		targetState: agentpb.JobState_JOB_STATE_SAVED,
	}

	// Transition to transitioning
	if _, ok := f.jobStates[nodeName]; !ok {
		f.jobStates[nodeName] = make(map[string]agentpb.JobState)
	}
	f.jobStates[nodeName][jobID] = agentpb.JobState_JOB_STATE_TRANSITIONING

	if f.OnSnapshot != nil {
		go f.OnSnapshot(nodeName, jobID)
	}

	return &agentpb.SnapshotResponse{OperationId: opID}, nil
}

func (f *FakeSnapshotAgentStore) Restore(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.RestoreResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.opCounter++
	opID := fmt.Sprintf("op-restore-%d", f.opCounter)
	f.pendingOperations[opID] = pendingOp{
		node:        nodeName,
		job:         jobID,
		opType:      "restore",
		targetState: agentpb.JobState_JOB_STATE_RUNNING,
	}

	// Transition to transitioning
	if _, ok := f.jobStates[nodeName]; !ok {
		f.jobStates[nodeName] = make(map[string]agentpb.JobState)
	}
	f.jobStates[nodeName][jobID] = agentpb.JobState_JOB_STATE_TRANSITIONING

	if f.OnRestore != nil {
		go f.OnRestore(nodeName, jobID)
	}

	return &agentpb.RestoreResponse{OperationId: opID}, nil
}

func (f *FakeSnapshotAgentStore) GetOperation(ctx context.Context, nodeName, operationID string) (*agentpb.GetOperationResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	op, ok := f.pendingOperations[operationID]
	if !ok {
		return &agentpb.GetOperationResponse{Status: agentpb.OperationStatus_OPERATION_STATUS_FAILED}, nil
	}

	// Apply the transition
	f.jobStates[op.node][op.job] = op.targetState
	delete(f.pendingOperations, operationID)

	return &agentpb.GetOperationResponse{
		Status:    agentpb.OperationStatus_OPERATION_STATUS_COMPLETE,
		ElapsedMs: 10,
	}, nil
}
