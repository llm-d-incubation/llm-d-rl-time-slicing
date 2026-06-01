package store

import (
	"context"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

// Group represents the in-memory and persistent state of a time-slice group.
type Group struct {
	mu             sync.RWMutex
	id             string
	nodes          []string
	lockingJob     string // job_id holding the lock, empty if none
	activeJob      string // job_id whose context is active, empty if none
	state          pb.GroupStatus_State
	stateTimestamp time.Time
	queue          *WaitingJobQueue
	lockStore      LockStore
}

// NewGroup creates a new Group with default values.
func NewGroup(id string) *Group {
	return &Group{
		id:             id,
		state:          pb.GroupStatus_STATE_UNSPECIFIED,
		stateTimestamp: time.Now(),
		queue:          NewWaitingJobQueue(),
	}
}

func (g *Group) ID() string {
	return g.id
}

func (g *Group) Nodes() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.nodes == nil {
		return nil
	}
	return append([]string(nil), g.nodes...)
}

func (g *Group) SetNodes(nodes []string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nodes = append([]string(nil), nodes...)
}

func (g *Group) LockingJob() string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.lockingJob
}

func (g *Group) SetLockingJob(jobID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.lockingJob = jobID
}

func (g *Group) ActiveJob() string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.activeJob
}

func (g *Group) SetActiveJob(jobID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.activeJob = jobID
}

func (g *Group) State() (pb.GroupStatus_State, time.Time) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.state, g.stateTimestamp
}

func (g *Group) SetState(state pb.GroupStatus_State) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.state = state
	g.stateTimestamp = time.Now()
}

// GetWaitingJobQueue returns a reference to the waiter queue for the group.
func (g *Group) GetWaitingJobQueue() *WaitingJobQueue {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.queue
}

func (g *Group) Delete(ctx context.Context) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.lockStore != nil {
		currentLock, err := g.lockStore.GetLock(ctx, g.id)
		if err != nil {
			return err
		}
		if currentLock != "" {
			if err := g.lockStore.Unlock(ctx, g.id, currentLock); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *Group) Lock(ctx context.Context, jobID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.lockStore != nil {
		if err := g.lockStore.Lock(ctx, g.id, jobID); err != nil {
			return err
		}
	}
	g.lockingJob = jobID
	g.stateTimestamp = time.Now()
	return nil
}

func (g *Group) Unlock(ctx context.Context, jobID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.lockStore != nil {
		if err := g.lockStore.Unlock(ctx, g.id, jobID); err != nil {
			return err
		}
	}
	g.lockingJob = ""
	g.stateTimestamp = time.Now()
	return nil
}
