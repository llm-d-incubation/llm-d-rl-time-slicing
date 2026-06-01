package store_test

import (
	"reflect"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
)

func TestJob_GettersAndSetters(t *testing.T) {
	tests := []struct {
		name         string
		jobID        string
		groupID      string
		pods         []string
		contextState map[string]pb.SnapshotAgentJobState_State
	}{
		{
			name:         "empty/nil values",
			jobID:        "",
			groupID:      "",
			pods:         nil,
			contextState: nil,
		},
		{
			name:    "populated values",
			jobID:   "job-123",
			groupID: "group-abc",
			pods:    []string{"pod-a", "pod-b"},
			contextState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
				"node-2": pb.SnapshotAgentJobState_STATE_SAVED,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			job := store.NewJob(tc.jobID, tc.groupID)

			if job.JobID() != tc.jobID {
				t.Errorf("JobID() = %q, want %q", job.JobID(), tc.jobID)
			}
			if job.GroupID() != tc.groupID {
				t.Errorf("GroupID() = %q, want %q", job.GroupID(), tc.groupID)
			}

			job.SetPods(tc.pods)
			if !reflect.DeepEqual(job.Pods(), tc.pods) {
				t.Errorf("Pods() = %+v, want %+v", job.Pods(), tc.pods)
			}

			job.SetContextState(tc.contextState)
			if !reflect.DeepEqual(job.ContextState(), tc.contextState) {
				t.Errorf("ContextState() = %+v, want %+v", job.ContextState(), tc.contextState)
			}

			// Verify deep copy/mutation protection for Pods
			if len(tc.pods) > 0 {
				mutatedPods := job.Pods()
				mutatedPods[0] = "mutated"
				if reflect.DeepEqual(job.Pods(), mutatedPods) {
					t.Errorf("mutating returned Pods slice modified the internal state")
				}
			}

			// Verify deep copy/mutation protection for ContextState
			if len(tc.contextState) > 0 {
				mutatedState := job.ContextState()
				for k := range mutatedState {
					mutatedState[k] = pb.SnapshotAgentJobState_STATE_UNSPECIFIED
				}
				if reflect.DeepEqual(job.ContextState(), mutatedState) {
					t.Errorf("mutating returned ContextState map modified the internal state")
				}
			}
		})
	}
}

func TestJob_UpdateContextState(t *testing.T) {
	tests := []struct {
		name          string
		initialState  map[string]pb.SnapshotAgentJobState_State
		updateNode    string
		updateVal     pb.SnapshotAgentJobState_State
		expectedState map[string]pb.SnapshotAgentJobState_State
	}{
		{
			name:         "update nil state",
			initialState: nil,
			updateNode:   "node-1",
			updateVal:    pb.SnapshotAgentJobState_STATE_RUNNING,
			expectedState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
			},
		},
		{
			name: "add new node to existing state",
			initialState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
			},
			updateNode: "node-2",
			updateVal:  pb.SnapshotAgentJobState_STATE_SAVED,
			expectedState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
				"node-2": pb.SnapshotAgentJobState_STATE_SAVED,
			},
		},
		{
			name: "overwrite existing node state",
			initialState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
			},
			updateNode: "node-1",
			updateVal:  pb.SnapshotAgentJobState_STATE_SAVED,
			expectedState: map[string]pb.SnapshotAgentJobState_State{
				"node-1": pb.SnapshotAgentJobState_STATE_SAVED,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			job := store.NewJob("job-1", "group-1")
			job.SetContextState(tc.initialState)

			job.UpdateContextState(tc.updateNode, tc.updateVal)

			if !reflect.DeepEqual(job.ContextState(), tc.expectedState) {
				t.Errorf("ContextState() after UpdateContextState = %+v, want %+v", job.ContextState(), tc.expectedState)
			}
		})
	}
}
