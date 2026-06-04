package store_test

import (
	"context"
	"errors"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
)

func TestJobStore_Get(t *testing.T) {
	tests := []struct {
		name    string
		initial []*store.Job
		jobID   string
		groupID string
		wantJob *store.Job
		wantErr error
	}{
		{
			name:    "empty store",
			initial: nil,
			jobID:   "job-a",
			groupID: "group-1",
			wantErr: store.ErrNotFound,
		},
		{
			name: "job exists",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			jobID:   "job-a",
			groupID: "group-1",
			wantJob: store.NewJob("group-1", "job-a"),
			wantErr: nil,
		},
		{
			name: "different group, same jobID",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			jobID:   "job-a",
			groupID: "group-2",
			wantErr: store.ErrNotFound,
		},
		{
			name: "different jobID, same group",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			jobID:   "job-b",
			groupID: "group-1",
			wantErr: store.ErrNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			jobStore := store.NewJobStore()
			for _, j := range tc.initial {
				if err := jobStore.Put(ctx, j); err != nil {
					t.Fatalf("failed to put initial job: %v", err)
				}
			}

			got, err := jobStore.Get(ctx, tc.groupID, tc.jobID)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Get() error = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr == nil {
				if got.JobID() != tc.wantJob.JobID() || got.GroupID() != tc.wantJob.GroupID() {
					t.Errorf("Get() got job with ID %q and GroupID %q, want ID %q and GroupID %q",
						got.JobID(), got.GroupID(), tc.wantJob.JobID(), tc.wantJob.GroupID())
				}
			}
		})
	}
}

func TestJobStore_Put(t *testing.T) {
	tests := []struct {
		name        string
		initial     []*store.Job
		putJob      *store.Job
		expectedLen int
	}{
		{
			name:        "put into empty store",
			initial:     nil,
			putJob:      store.NewJob("group-1", "job-a"),
			expectedLen: 1,
		},
		{
			name: "overwrite existing job",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			putJob:      store.NewJob("group-1", "job-a"),
			expectedLen: 1,
		},
		{
			name: "put another job",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			putJob:      store.NewJob("group-1", "job-b"),
			expectedLen: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			jobStore := store.NewJobStore()
			for _, j := range tc.initial {
				if err := jobStore.Put(ctx, j); err != nil {
					t.Fatalf("failed to put initial job: %v", err)
				}
			}

			err := jobStore.Put(ctx, tc.putJob)
			if err != nil {
				t.Fatalf("Put() returned error: %v", err)
			}

			// Verify internal length via ListByGroup
			var count int
			// We know we only use group-1 for tests here
			list1, err := jobStore.ListByGroup(ctx, "group-1")
			if err != nil {
				t.Fatalf("ListByGroup() failed: %v", err)
			}
			count += len(list1)

			if count != tc.expectedLen {
				t.Errorf("expected store size %d, got %d", tc.expectedLen, count)
			}

			// Retrieve and verify
			got, err := jobStore.Get(ctx, tc.putJob.GroupID(), tc.putJob.JobID())
			if err != nil {
				t.Fatalf("failed to retrieve put job: %v", err)
			}
			if got.JobID() != tc.putJob.JobID() || got.GroupID() != tc.putJob.GroupID() {
				t.Errorf("retrieved job does not match put job")
			}
		})
	}
}

func TestJobStore_ListByGroup(t *testing.T) {
	tests := []struct {
		name        string
		initial     []*store.Job
		listGroup   string
		expectedIDs []string
	}{
		{
			name:        "list empty store",
			initial:     nil,
			listGroup:   "group-1",
			expectedIDs: nil,
		},
		{
			name: "list group with single job",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			listGroup:   "group-1",
			expectedIDs: []string{"job-a"},
		},
		{
			name: "list group with multiple jobs",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
				store.NewJob("group-1", "job-b"),
				store.NewJob("group-2", "job-c"),
			},
			listGroup:   "group-1",
			expectedIDs: []string{"job-a", "job-b"},
		},
		{
			name: "list group with no matching jobs",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			listGroup:   "group-2",
			expectedIDs: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			jobStore := store.NewJobStore()
			for _, j := range tc.initial {
				if err := jobStore.Put(ctx, j); err != nil {
					t.Fatalf("failed to put initial job: %v", err)
				}
			}

			list, err := jobStore.ListByGroup(ctx, tc.listGroup)
			if err != nil {
				t.Fatalf("ListByGroup() returned error: %v", err)
			}

			gotIDs := make([]string, 0, len(list))
			for _, j := range list {
				gotIDs = append(gotIDs, j.JobID())
			}

			// Order check is optional but let's ensure elements match
			if len(gotIDs) != len(tc.expectedIDs) {
				t.Fatalf("ListByGroup() returned %d items, want %d", len(gotIDs), len(tc.expectedIDs))
			}

			// Simple subset validation (since order in map iteration is randomized in Go)
			for _, id := range tc.expectedIDs {
				found := false
				for _, gotID := range gotIDs {
					if gotID == id {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("ListByGroup() did not contain expected job ID %q", id)
				}
			}
		})
	}
}

func TestJobStore_Delete(t *testing.T) {
	tests := []struct {
		name          string
		initial       []*store.Job
		deleteJobID   string
		deleteGroupID string
		expectedErr   error
	}{
		{
			name:          "delete from empty store",
			initial:       nil,
			deleteJobID:   "job-a",
			deleteGroupID: "group-1",
			expectedErr:   nil, // Store Delete of non-existent is generally a no-op/nil err
		},
		{
			name: "delete existing job",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			deleteJobID:   "job-a",
			deleteGroupID: "group-1",
			expectedErr:   nil,
		},
		{
			name: "delete with wrong groupID",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			deleteJobID:   "job-a",
			deleteGroupID: "group-2",
			expectedErr:   nil, // will be no-op
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			jobStore := store.NewJobStore()
			for _, j := range tc.initial {
				if err := jobStore.Put(ctx, j); err != nil {
					t.Fatalf("failed to put initial job: %v", err)
				}
			}

			err := jobStore.Delete(ctx, tc.deleteGroupID, tc.deleteJobID)
			if !errors.Is(err, tc.expectedErr) {
				t.Fatalf("Delete() error = %v, want %v", err, tc.expectedErr)
			}

			// If we deleted a valid job, make sure it's gone
			if len(tc.initial) > 0 && tc.deleteGroupID == "group-1" && tc.deleteJobID == "job-a" {
				_, err := jobStore.Get(ctx, tc.deleteGroupID, tc.deleteJobID)
				if !errors.Is(err, store.ErrNotFound) {
					t.Errorf("Get() after Delete returned error %v, want ErrNotFound", err)
				}
			}
		})
	}
}

func TestJobStore_UpdateContextState(t *testing.T) {
	tests := []struct {
		name          string
		initial       []*store.Job
		updateJobID   string
		updateGroupID string
		updateNode    string
		updateVal     pb.SnapshotAgentJobState_State
		expectedErr   error
	}{
		{
			name:          "update non-existent job",
			initial:       nil,
			updateJobID:   "job-a",
			updateGroupID: "group-1",
			updateNode:    "node-1",
			updateVal:     pb.SnapshotAgentJobState_STATE_RUNNING,
			expectedErr:   store.ErrNotFound,
		},
		{
			name: "update existing job",
			initial: []*store.Job{
				store.NewJob("group-1", "job-a"),
			},
			updateJobID:   "job-a",
			updateGroupID: "group-1",
			updateNode:    "node-1",
			updateVal:     pb.SnapshotAgentJobState_STATE_RUNNING,
			expectedErr:   nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			jobStore := store.NewJobStore()
			for _, j := range tc.initial {
				if err := jobStore.Put(ctx, j); err != nil {
					t.Fatalf("failed to put initial job: %v", err)
				}
			}

			err := jobStore.UpdateContextState(ctx, tc.updateGroupID, tc.updateJobID, tc.updateNode, tc.updateVal)
			if !errors.Is(err, tc.expectedErr) {
				t.Fatalf("UpdateContextState() error = %v, want %v", err, tc.expectedErr)
			}

			if tc.expectedErr == nil {
				got, err := jobStore.Get(ctx, tc.updateGroupID, tc.updateJobID)
				if err != nil {
					t.Fatalf("failed to get job after update: %v", err)
				}
				if got.ContextState()[tc.updateNode] != tc.updateVal {
					t.Errorf("ContextState for node %q = %v, want %v", tc.updateNode, got.ContextState()[tc.updateNode], tc.updateVal)
				}
			}
		})
	}
}
