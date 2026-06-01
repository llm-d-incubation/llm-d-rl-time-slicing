package store

import (
	"context"
	"sync"
)

// MemLockStore is an in-memory implementation of LockStore.
type MemLockStore struct {
	mu    sync.RWMutex
	locks map[string]string // groupID -> jobID
}

// NewMemLockStore creates a new MemLockStore.
func NewMemLockStore() *MemLockStore {
	return &MemLockStore{
		locks: make(map[string]string),
	}
}

// GetLock returns the job_id currently holding the lock for the group.
func (s *MemLockStore) GetLock(ctx context.Context, groupID string) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.locks[groupID], nil
}

// Lock persistently sets the job_id holding the lock for the group.
func (s *MemLockStore) Lock(ctx context.Context, groupID, jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.locks[groupID]
	if ok && current != "" && current != jobID {
		return ErrAlreadyLocked
	}
	s.locks[groupID] = jobID
	return nil
}

// Unlock persistently releases the lock for the group.
func (s *MemLockStore) Unlock(ctx context.Context, groupID, jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.locks[groupID]
	if !ok || current == "" {
		return nil
	}
	if current != jobID {
		return ErrNotLockHolder
	}
	delete(s.locks, groupID)
	return nil
}
