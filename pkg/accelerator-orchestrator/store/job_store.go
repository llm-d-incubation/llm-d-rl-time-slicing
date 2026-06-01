package store

import (
	"context"
	"sync"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

// JobStore manages Job state in memory.
type JobStore struct {
	mu   sync.RWMutex
	jobs map[string]*Job // key: jobID + ":" + groupID
}

// NewJobStore creates a new JobStore.
func NewJobStore() *JobStore {
	return &JobStore{
		jobs: make(map[string]*Job),
	}
}

func makeJobKey(jobID, groupID string) string {
	return jobID + ":" + groupID
}

// Get returns the job with the given ID and group ID.
func (s *JobStore) Get(ctx context.Context, jobID, groupID string) (*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	key := makeJobKey(jobID, groupID)
	j, ok := s.jobs[key]
	if !ok {
		return nil, ErrNotFound
	}
	return j, nil
}

// ListByGroup returns all jobs belonging to the given group.
func (s *JobStore) ListByGroup(ctx context.Context, groupID string) ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var list []*Job
	for _, j := range s.jobs {
		if j.GroupID() == groupID {
			list = append(list, j)
		}
	}
	return list, nil
}

// Put stores or updates the job.
func (s *JobStore) Put(ctx context.Context, job *Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := makeJobKey(job.JobID(), job.GroupID())
	s.jobs[key] = job
	return nil
}

// Delete removes the job.
func (s *JobStore) Delete(ctx context.Context, jobID, groupID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := makeJobKey(jobID, groupID)
	delete(s.jobs, key)
	return nil
}

// UpdateContextState updates the last-known context state of a job on a specific node.
func (s *JobStore) UpdateContextState(
	ctx context.Context,
	jobID, groupID, nodeName string,
	state pb.SnapshotAgentJobState_State,
) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	key := makeJobKey(jobID, groupID)
	j, ok := s.jobs[key]
	if !ok {
		return ErrNotFound
	}
	j.UpdateContextState(nodeName, state)
	return nil
}
