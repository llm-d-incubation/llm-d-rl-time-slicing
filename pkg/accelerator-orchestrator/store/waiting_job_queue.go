package store

import (
	"sync"
	"time"
)

// WaitingJob represents a job waiting in the queue.
type WaitingJob struct {
	JobID       string
	QueuedSince time.Time
}

// WaitingJobQueue is a thread-safe FIFO queue for WaitingJobs that prevents duplicates.
type WaitingJobQueue struct {
	mu    sync.RWMutex
	jobs  []WaitingJob
	exist map[string]struct{}
}

// NewWaitingJobQueue creates a new WaitingJobQueue.
func NewWaitingJobQueue() *WaitingJobQueue {
	return &WaitingJobQueue{
		exist: make(map[string]struct{}),
	}
}

// Enqueue adds a job to the end of the queue if it's not already present.
// Returns true if the job was added, false if it was a duplicate (noop).
func (q *WaitingJobQueue) Enqueue(jobID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	if _, ok := q.exist[jobID]; ok {
		return false
	}

	q.exist[jobID] = struct{}{}
	q.jobs = append(q.jobs, WaitingJob{
		JobID:       jobID,
		QueuedSince: time.Now(),
	})
	return true
}

// Dequeue removes and returns the next job from the front of the queue.
// Returns the jobID and true if successful, or empty string and false if the queue is empty.
func (q *WaitingJobQueue) Dequeue() (string, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.jobs) == 0 {
		return "", false
	}

	job := q.jobs[0]
	q.jobs = q.jobs[1:]
	delete(q.exist, job.JobID)
	return job.JobID, true
}

// Len returns the number of jobs currently in the queue.
func (q *WaitingJobQueue) Len() int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	return len(q.jobs)
}

// Exists checks if a job is currently in the queue.
func (q *WaitingJobQueue) Exists(jobID string) bool {
	q.mu.RLock()
	defer q.mu.RUnlock()
	_, ok := q.exist[jobID]
	return ok
}

// List returns a deep copy of the jobs currently in the queue, maintaining FIFO order.
func (q *WaitingJobQueue) List() []WaitingJob {
	q.mu.RLock()
	defer q.mu.RUnlock()

	if len(q.jobs) == 0 {
		return nil
	}
	res := make([]WaitingJob, len(q.jobs))
	copy(res, q.jobs)
	return res
}

// Reset clears the queue and re-initializes it with the provided jobs, preserving their order.
// It automatically filters out any duplicates in the input slice (keeping the first occurrence).
func (q *WaitingJobQueue) Reset(jobs []WaitingJob) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.jobs = nil
	q.exist = make(map[string]struct{})

	for _, j := range jobs {
		if _, ok := q.exist[j.JobID]; ok {
			continue // skip duplicates in input
		}
		q.exist[j.JobID] = struct{}{}
		q.jobs = append(q.jobs, j)
	}
}
