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

// GroupSpec defines the specification of desired end state of a time-slice group.
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

// GroupStatus represents the current status of a time-slice group.
// Updated by the controller reconcile loop.
type GroupStatus struct {
	mu             sync.RWMutex
	nodes          []string
	state          pb.GroupStatus_State
	stateTimestamp time.Time
	// loadedJob is a job that the controller has loaded all
	// the snapshotted context for on the nodes. Context for all
	// other jobs will have been offloaded as well.
	loadedJob string
}

func (s *GroupStatus) Nodes() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]string(nil), s.nodes...)
}

func (s *GroupStatus) SetNodes(nodes []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nodes = append([]string(nil), nodes...)
}

func (s *GroupStatus) State() (pb.GroupStatus_State, time.Time) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state, s.stateTimestamp
}

func (s *GroupStatus) SetState(state pb.GroupStatus_State) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == state {
		return
	}
	s.state = state
	s.stateTimestamp = time.Now()
}

func (s *GroupStatus) LoadedJob() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.loadedJob
}

func (s *GroupStatus) SetLoadedJob(jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loadedJob = jobID
}

// Group represents the in-memory and persistent state of a time-slice group.
type Group struct {
	id     string
	spec   *GroupSpec
	status *GroupStatus
}

// NewGroup creates a new Group and initializes its locking state from the lockStore if available.
func NewGroup(ctx context.Context, id string, lockStore GroupLockStore) (*Group, error) {
	group := &Group{
		id: id,
		spec: &GroupSpec{
			lockStore: lockStore,
			queue:     NewWaitingJobQueue(),
		},
		status: &GroupStatus{
			state:          pb.GroupStatus_STATE_UNSPECIFIED,
			stateTimestamp: time.Now(),
		},
	}

	if lockStore != nil {
		lockingJob, err := lockStore.GetLock(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get lock status for group %s: %w", id, err)
		}
		group.spec.lockingJob = lockingJob
		group.spec.activeJob = lockingJob
	}

	return group, nil
}

func (g *Group) ID() string {
	return g.id
}

// Spec returns the GroupSpec for the group.
func (g *Group) Spec() *GroupSpec {
	// No lock is safe here because the pointer is not
	// modifiable after the Group is created. Fields
	// on the spec itself are controlled by the internal mutex.
	return g.spec
}

// Status returns the GroupStatus.
func (g *Group) Status() *GroupStatus {
	return g.status
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
	LoadedJob        string
}

// Snapshot returns a consistent, point-in-time snapshot of the group's state.
func (g *Group) Snapshot() *GroupSnapshot {
	g.status.mu.RLock()
	defer g.status.mu.RUnlock()

	// Deep copy nodes slice
	nodes := make([]string, len(g.status.nodes))
	copy(nodes, g.status.nodes)

	loadedJob := g.status.loadedJob

	g.spec.mu.RLock()
	defer g.spec.mu.RUnlock()

	return &GroupSnapshot{
		ID:               g.id,
		Nodes:            nodes,
		State:            g.status.state,
		StateTimestamp:   g.status.stateTimestamp,
		LockingJob:       g.spec.lockingJob,
		ActiveJob:        g.spec.activeJob,
		WaiterQueueDepth: g.spec.queue.Len(),
		LoadedJob:        loadedJob,
	}
}

// TryPromote attempts to promote the next waiting job in the queue to be the locking job
// if the group is currently unlocked.
func (s *GroupSpec) TryPromote(ctx context.Context) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.lockingJob != "" {
		return false, nil
	}

	nextJobID, ok := s.queue.Peek()
	if !ok {
		return false, nil
	}

	if err := s.lock(ctx, nextJobID); err != nil {
		return false, fmt.Errorf("failed to lock promoted job %s: %w", nextJobID, err)
	}

	s.queue.Dequeue()
	return true, nil
}

// Yield releases the lock for the current job.
func (s *GroupSpec) Yield(ctx context.Context, jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.unlock(ctx, jobID)

	// Note that promoting the next job to take the lock is
	// done in the reconiliation loop to not block a RL job trying
	// to yield their job if there is a temporary issue locking for next job.
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
