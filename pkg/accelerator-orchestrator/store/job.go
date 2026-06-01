package store

import (
	"sync"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

// Job represents the state of a job within a group.
type Job struct {
	mu           sync.RWMutex
	jobID        string
	groupID      string
	pods         []string // pod UUIDs
	contextState map[string]pb.SnapshotAgentJobState_State
}

// NewJob creates a new Job with default values.
func NewJob(jobID, groupID string) *Job {
	return &Job{
		jobID:        jobID,
		groupID:      groupID,
		contextState: make(map[string]pb.SnapshotAgentJobState_State),
	}
}

func (j *Job) JobID() string {
	return j.jobID
}

func (j *Job) GroupID() string {
	return j.groupID
}

func (j *Job) Pods() []string {
	j.mu.RLock()
	defer j.mu.RUnlock()
	if j.pods == nil {
		return nil
	}
	return append([]string(nil), j.pods...)
}

func (j *Job) SetPods(pods []string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.pods = append([]string(nil), pods...)
}

func (j *Job) ContextState() map[string]pb.SnapshotAgentJobState_State {
	j.mu.RLock()
	defer j.mu.RUnlock()
	if j.contextState == nil {
		return nil
	}
	res := make(map[string]pb.SnapshotAgentJobState_State)
	for k, v := range j.contextState {
		res[k] = v
	}
	return res
}

func (j *Job) SetContextState(cs map[string]pb.SnapshotAgentJobState_State) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if cs == nil {
		j.contextState = nil
		return
	}
	j.contextState = make(map[string]pb.SnapshotAgentJobState_State)
	for k, v := range cs {
		j.contextState[k] = v
	}
}

func (j *Job) UpdateContextState(nodeName string, state pb.SnapshotAgentJobState_State) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.contextState == nil {
		j.contextState = make(map[string]pb.SnapshotAgentJobState_State)
	}
	j.contextState[nodeName] = state
}
