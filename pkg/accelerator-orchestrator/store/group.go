package store

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

// Group represents the in-memory and persistent state of a time-slice group.
type Group struct {
	mu    sync.RWMutex
	id    string
	nodes []string
	// lockingJob is the job_id holding the lock, empty if none.
	// Acts like the cached value for value stored in lockStore.
	lockingJob string
	// activeJob is the job_id whose context is active, empty if none.
	// The primary situation where the active job is not locking is the
	// STATE_IDLE_YIELDED state where the job has yielded, but snapshotting
	// is delayed because no other job has requested to lock the group.
	activeJob      string
	state          pb.GroupStatus_State
	stateTimestamp time.Time
	queue          *WaitingJobQueue
	lockStore      LockStore
}

// NewGroup creates a new Group and initializes its locking state from the lockStore if available.
func NewGroup(ctx context.Context, id string, lockStore LockStore) (*Group, error) {
	g := &Group{
		id:             id,
		state:          pb.GroupStatus_STATE_UNSPECIFIED,
		stateTimestamp: time.Now(),
		queue:          NewWaitingJobQueue(),
		lockStore:      lockStore,
	}

	if lockStore != nil {
		lockingJob, err := lockStore.GetLock(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("failed to get lock status for group %s: %w", id, err)
		}
		g.lockingJob = lockingJob
	}

	return g, nil
}

func (g *Group) ID() string {
	return g.id
}

func (g *Group) Nodes() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
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
	if g.state == state {
		return
	}
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
	lj := g.LockingJob()
	if lj == "" {
		return nil
	}

	return g.Unlock(ctx, lj)
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
	return nil
}

// Clone returns a deep, atomic copy of the Group.
// The returned Group is safe to read from without further synchronization with the original Group.
func (g *Group) Clone() *Group {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Deep copy nodes slice
	nodes := make([]string, len(g.nodes))
	copy(nodes, g.nodes)

	// Clone the queue
	var clonedQueue *WaitingJobQueue
	if g.queue != nil {
		clonedQueue = g.queue.Clone()
	}

	return &Group{
		id:             g.id,
		nodes:          nodes,
		lockingJob:     g.lockingJob,
		activeJob:      g.activeJob,
		state:          g.state,
		stateTimestamp: g.stateTimestamp,
		queue:          clonedQueue,
		lockStore:      g.lockStore,
	}
}

