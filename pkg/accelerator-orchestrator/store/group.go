package store

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
)

// GroupLockStore defines the interface for persisting lock state for a specific group.
type GroupLockStore interface {
	GetLock(ctx context.Context) (string, error)
	Lock(ctx context.Context, jobID string) error
	Unlock(ctx context.Context, jobID string) error
}

// GroupSpec defines the specification of a time-slice group.
type GroupSpec struct {
	mu        sync.RWMutex
	lockStore GroupLockStore
	// lockingJob is job we want to hold the lock.
	lockingJob string
	queue      *WaitingJobQueue
	// activeJob is job for which we want context loaded on nodes.
	// This can be non-empty when lockingJob is empty as an optimization
	// when there is no one waiting to be the locking job.
	activeJob string
}

// Group represents the in-memory and persistent state of a time-slice group.
type Group struct {
	mu             sync.RWMutex
	id             string
	nodes          []string
	spec           *GroupSpec
	state          pb.GroupStatus_State
	stateTimestamp time.Time
}

// NewGroup creates a new Group and initializes its locking state from the lockStore if available.
func NewGroup(ctx context.Context, id string, lockStore GroupLockStore) (*Group, error) {
	g := &Group{
		id: id,
		spec: &GroupSpec{
			lockStore: lockStore,
			queue:     NewWaitingJobQueue(),
		},
		state:          pb.GroupStatus_STATE_UNSPECIFIED,
		stateTimestamp: time.Now(),
	}

	if lockStore != nil {
		lockingJob, err := lockStore.GetLock(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get lock status for group %s: %w", id, err)
		}
		g.spec.lockingJob = lockingJob
		g.spec.activeJob = lockingJob
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

// Spec returns the GroupSpec for the group.
func (g *Group) Spec() *GroupSpec {
	// No lock is safe here because the pointer is not
	// modifiable after the Group is created. Fields
	// on the spec itself are controlled by the internal mutex.
	return g.spec
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

// Delete deletes the group by releasing its lock if it is currently held.
func (g *Group) Delete(ctx context.Context) error {
	return g.spec.Delete(ctx)
}

// Methods on GroupSpec

// ActiveJob returns the job ID for which the context should be loaded on nodes.
func (s *GroupSpec) ActiveJob() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.activeJob
}

// LockingJob returns the job ID that currently holds the lock, or empty if none.
func (s *GroupSpec) LockingJob() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lockingJob
}

// GetWaitingJobQueue returns the queue of jobs waiting for the lock.
func (s *GroupSpec) GetWaitingJobQueue() *WaitingJobQueue {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.queue
}

// Delete cleans up the spec information.
func (s *GroupSpec) Delete(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lockingJob == "" {
		return nil
	}
	return s.unlock(ctx, s.lockingJob)
}

// lock assumes s.mu is held.
func (s *GroupSpec) lock(ctx context.Context, jobID string) error {
	if s.lockStore != nil {
		if err := s.lockStore.Lock(ctx, jobID); err != nil {
			return err
		}
	}
	s.lockingJob = jobID
	s.activeJob = jobID
	return nil
}

// unlock assumes s.mu is held.
func (s *GroupSpec) unlock(ctx context.Context, jobID string) error {
	if s.lockStore != nil {
		if err := s.lockStore.Unlock(ctx, jobID); err != nil {
			return err
		}
	}
	s.lockingJob = ""
	// Notice we do not clear the active job. This is because
	// we actually want to leave the context on the machines until
	// there is a new job that wants to lock the group.
	return nil
}

// GroupSnapshot represents an immutable, point-in-time copy of a Group's state.
type GroupSnapshot struct {
	ID               string
	Nodes            []string
	State            pb.GroupStatus_State
	StateTimestamp   time.Time
	LockingJob       string
	ActiveJob        string
	WaiterQueueDepth int
}

// Snapshot returns a consistent, point-in-time snapshot of the group's state.
func (g *Group) Snapshot() *GroupSnapshot {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Deep copy nodes slice
	nodes := make([]string, len(g.nodes))
	copy(nodes, g.nodes)

	g.spec.mu.RLock()
	defer g.spec.mu.RUnlock()

	return &GroupSnapshot{
		ID:               g.id,
		Nodes:            nodes,
		State:            g.state,
		StateTimestamp:   g.stateTimestamp,
		LockingJob:       g.spec.lockingJob,
		ActiveJob:        g.spec.activeJob,
		WaiterQueueDepth: g.spec.queue.Len(),
	}
}

// Yield releases the lock for the current job and grants it to the next job in the queue.
// If the unlock is successful, it peeks at the next job, attempts to lock it,
// and only dequeues it if the lock operation succeeds.
func (s *GroupSpec) Yield(ctx context.Context, jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.unlock(ctx, jobID); err != nil {
		return err
	}

	nextJobID, ok := s.queue.Peek()
	if !ok {
		return nil
	}

	if err := s.lock(ctx, nextJobID); err != nil {
		return fmt.Errorf("yield succeeded in unlocking but failed to lock next job %s: %w", nextJobID, err)
	}

	s.queue.Dequeue()
	return nil
}

// RequestLock requests a lock for the given job.
// If the job already holds the lock, it returns immediately.
// Otherwise, it enqueues the job in the waiting queue.
func (s *GroupSpec) RequestLock(jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.lockingJob == jobID {
		return
	}

	s.queue.Enqueue(jobID)
}

// SetActiveJob sets the active job.
// Primarily used for testing.
func (s *GroupSpec) SetActiveJob(jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeJob = jobID
}
