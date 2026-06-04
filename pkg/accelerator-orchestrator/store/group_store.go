package store

import (
	"context"
	"errors"
	"sync"
)

var (
	// ErrNotFound is returned when a resource is not found in the store.
	ErrNotFound = errors.New("resource not found")
	// ErrAlreadyLocked is returned when trying to lock a group that is already locked.
	ErrAlreadyLocked = errors.New("group is already locked")
	// ErrNotLockHolder is returned when trying to unlock a group locked by another job.
	ErrNotLockHolder = errors.New("group is locked by another job")
	// ErrAlreadyExists is returned when trying to add a resource that already exists.
	ErrAlreadyExists = errors.New("resource already exists")
)

// LockStore defines the interface for persisting lock state.
type LockStore interface {
	// GetLock returns the job_id currently holding the lock for the group.
	// Returns empty string if the group is unlocked.
	GetLock(ctx context.Context, groupID string) (string, error)

	// Lock persistently sets the job_id holding the lock for the group.
	// Returns ErrAlreadyLocked if the group is already locked by another job.
	Lock(ctx context.Context, groupID string, jobID string) error

	// Unlock persistently releases the lock for the group.
	// Returns ErrNotLockHolder if the group is locked by another job.
	Unlock(ctx context.Context, groupID string, jobID string) error
}

type GroupStore struct {
	mu        sync.RWMutex
	groups    map[string]*Group
	lockStore LockStore
}

// NewGroupStore creates a new GroupStore.
func NewGroupStore(lockStore LockStore) *GroupStore {
	return &GroupStore{
		groups:    make(map[string]*Group),
		lockStore: lockStore,
	}
}

// Get returns the group with the given ID.
func (s *GroupStore) Get(ctx context.Context, id string) (*Group, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	g, ok := s.groups[id]
	if !ok {
		return nil, ErrNotFound
	}
	return g, nil
}

// GetOrCreate returns the group with the given ID. If it does not exist, it creates
// a new group, adds it to the store, and returns it.
// Returns the group, a boolean indicating if a new group was created, and any error.
func (s *GroupStore) GetOrCreate(ctx context.Context, id string) (*Group, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	g, ok := s.groups[id]
	if ok {
		return g, false, nil
	}

	g = NewGroup(id, s.lockStore)
	s.groups[id] = g
	return g, true, nil
}

// List returns all known groups.
func (s *GroupStore) List(ctx context.Context) ([]*Group, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	list := make([]*Group, 0, len(s.groups))
	for _, g := range s.groups {
		list = append(list, g)
	}
	return list, nil
}

// Delete removes the group.
func (s *GroupStore) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	g, ok := s.groups[id]
	if ok {
		if err := g.Delete(ctx); err != nil {
			return err
		}
		delete(s.groups, id)
	}
	return nil
}
