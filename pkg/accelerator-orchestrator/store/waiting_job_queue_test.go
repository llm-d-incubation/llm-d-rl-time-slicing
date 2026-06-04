package store_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
)

func TestWaitingJobQueue_BasicAndDuplicates(t *testing.T) {
	type op struct {
		action   string // "enqueue", "dequeue", "exists", "len"
		val      string // input for action
		wantBool bool   // expected result for enqueue/dequeue/exists
		wantInt  int    // expected result for len
		wantVal  string // expected output for dequeue
	}

	tests := []struct {
		name  string
		steps []op
	}{
		{
			name: "basic enqueue and dequeue FIFO",
			steps: []op{
				{action: "len", wantInt: 0},
				{action: "enqueue", val: "job-a", wantBool: true},
				{action: "enqueue", val: "job-b", wantBool: true},
				{action: "len", wantInt: 2},
				{action: "exists", val: "job-a", wantBool: true},
				{action: "exists", val: "job-c", wantBool: false},
				{action: "dequeue", wantBool: true, wantVal: "job-a"},
				{action: "len", wantInt: 1},
				{action: "dequeue", wantBool: true, wantVal: "job-b"},
				{action: "len", wantInt: 0},
				{action: "dequeue", wantBool: false, wantVal: ""},
			},
		},
		{
			name: "prevent duplicates",
			steps: []op{
				{action: "enqueue", val: "job-a", wantBool: true},
				{action: "enqueue", val: "job-a", wantBool: false}, // duplicate
				{action: "len", wantInt: 1},
				{action: "dequeue", wantBool: true, wantVal: "job-a"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			queue := store.NewWaitingJobQueue()
			for i, step := range tc.steps {
				switch step.action {
				case "enqueue":
					got := queue.Enqueue(step.val)
					if got != step.wantBool {
						t.Fatalf("step %d: Enqueue(%q) = %t, want %t", i, step.val, got, step.wantBool)
					}
				case "dequeue":
					gotVal, ok := queue.Dequeue()
					if ok != step.wantBool || gotVal != step.wantVal {
						t.Fatalf("step %d: Dequeue() = (%q, %t), want (%q, %t)", i, gotVal, ok, step.wantVal, step.wantBool)
					}
				case "exists":
					got := queue.Exists(step.val)
					if got != step.wantBool {
						t.Fatalf("step %d: Exists(%q) = %t, want %t", i, step.val, got, step.wantBool)
					}
				case "len":
					got := queue.Len()
					if got != step.wantInt {
						t.Fatalf("step %d: Len() = %d, want %d", i, got, step.wantInt)
					}
				}
			}
		})
	}
}

func TestWaitingJobQueue_Concurrency(t *testing.T) {
	queue := store.NewWaitingJobQueue()
	var wg sync.WaitGroup

	numWorkers := 100
	jobsPerWorker := 10

	// 1. Concurrent enqueue of unique jobs
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < jobsPerWorker; j++ {
				jobID := fmt.Sprintf("worker-%d-job-%d", workerID, j)
				queue.Enqueue(jobID)
			}
		}(i)
	}
	wg.Wait()

	expectedLen := numWorkers * jobsPerWorker
	if queue.Len() != expectedLen {
		t.Errorf("Concurrent unique enqueue got len %d, want %d", queue.Len(), expectedLen)
	}

	// 2. Concurrent enqueue of the SAME job (only 1 should succeed)
	queue = store.NewWaitingJobQueue()
	wg.Add(numWorkers)
	successCount := 0
	var countMu sync.Mutex

	for i := 0; i < numWorkers; i++ {
		go func() {
			defer wg.Done()
			if queue.Enqueue("single-job") {
				countMu.Lock()
				successCount++
				countMu.Unlock()
			}
		}()
	}
	wg.Wait()

	if queue.Len() != 1 {
		t.Errorf("Concurrent duplicate enqueue got len %d, want 1", queue.Len())
	}
	if successCount != 1 {
		t.Errorf("Concurrent duplicate enqueue reported %d successes, want 1", successCount)
	}
}
