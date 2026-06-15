package store

import (
	"container/list"
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
	jobs  list.List
	exist map[string]*list.Element
}

// NewWaitingJobQueue creates a new WaitingJobQueue.
func NewWaitingJobQueue() *WaitingJobQueue {
	return &WaitingJobQueue{
		exist: make(map[string]*list.Element),
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

	elem := q.jobs.PushBack(WaitingJob{
		JobID:       jobID,
		QueuedSince: time.Now(),
	})
	q.exist[jobID] = elem
	return true
}

// Dequeue removes and returns the next job from the front of the queue.
// Returns the jobID and true if successful, or empty string and false if the queue is empty.
func (q *WaitingJobQueue) Dequeue() (string, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	front := q.jobs.Front()
	if front == nil {
		return "", false
	}

	_ = q.jobs.Remove(front)
	job, ok := front.Value.(WaitingJob)
	if !ok {
		panic("invalid type in queue")
	}
	delete(q.exist, job.JobID)
	return job.JobID, true
}

// Len returns the number of jobs currently in the queue.
func (q *WaitingJobQueue) Len() int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	return q.jobs.Len()
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

	res := make([]WaitingJob, 0, q.jobs.Len())
	for e := q.jobs.Front(); e != nil; e = e.Next() {
		job, ok := e.Value.(WaitingJob)
		if !ok {
			panic("invalid type in queue")
		}
		res = append(res, job)
	}
	return res
}
