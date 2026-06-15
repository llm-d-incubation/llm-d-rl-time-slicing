package statemachine

import (
	"sync"
)

// Exported for testing purposes only.
type (
	InternalJob       = Job
	InternalOperation = Operation
)

func (sm *StateManager) InternalJobs() map[string]*Job {
	return sm.jobs
}

func (sm *StateManager) InternalOperations() map[string]*Operation {
	return sm.operations
}

func (sm *StateManager) InternalMu() *sync.RWMutex {
	return &sm.mu
}

func (sm *StateManager) InternalGetOrCreateJob(jobID, group string) *Job {
	return sm.getOrCreateJob(jobID, group)
}
