package store_test

import (
	"context"
	"errors"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
)

func TestGroupStore_Get(t *testing.T) {
	tests := []struct {
		name      string
		initial   []*store.Group
		groupID   string
		wantGroup *store.Group
		wantErr   error
	}{
		{
			name:    "empty store",
			initial: nil,
			groupID: "group-1",
			wantErr: store.ErrNotFound,
		},
		{
			name: "group exists",
			initial: []*store.Group{
				newTestGroup("group-1", []string{"node-a"}),
			},
			groupID:   "group-1",
			wantGroup: newTestGroup("group-1", []string{"node-a"}),
			wantErr:   nil,
		},
		{
			name: "different groupID",
			initial: []*store.Group{
				newTestGroup("group-1", []string{"node-a"}),
			},
			groupID: "group-2",
			wantErr: store.ErrNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			groupStore := store.NewGroupStore(store.NewMemLockStore())
			for _, g := range tc.initial {
				addGroupToStore(t, ctx, groupStore, g)
			}

			got, err := groupStore.Get(ctx, tc.groupID)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Get() error = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr == nil {
				if got.ID() != tc.wantGroup.ID() {
					t.Errorf("Get() got group ID %q, want %q", got.ID(), tc.wantGroup.ID())
				}
			}
		})
	}
}

func TestGroupStore_List(t *testing.T) {
	tests := []struct {
		name        string
		initial     []*store.Group
		expectedIDs []string
	}{
		{
			name:        "list empty store",
			initial:     nil,
			expectedIDs: nil,
		},
		{
			name: "list store with single group",
			initial: []*store.Group{
				newTestGroup("group-1", nil),
			},
			expectedIDs: []string{"group-1"},
		},
		{
			name: "list store with multiple groups",
			initial: []*store.Group{
				newTestGroup("group-1", nil),
				newTestGroup("group-2", nil),
			},
			expectedIDs: []string{"group-1", "group-2"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			groupStore := store.NewGroupStore(store.NewMemLockStore())
			for _, g := range tc.initial {
				addGroupToStore(t, ctx, groupStore, g)
			}

			list, err := groupStore.List(ctx)
			if err != nil {
				t.Fatalf("List() returned error: %v", err)
			}

			if len(list) != len(tc.expectedIDs) {
				t.Fatalf("List() returned %d items, want %d", len(list), len(tc.expectedIDs))
			}

			for _, id := range tc.expectedIDs {
				found := false
				for _, g := range list {
					if g.ID() == id {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("List() did not contain expected group ID %q", id)
				}
			}
		})
	}
}

func TestGroupStore_Delete(t *testing.T) {
	tests := []struct {
		name        string
		initial     []*store.Group
		deleteID    string
		expectedErr error
	}{
		{
			name:        "delete from empty store",
			initial:     nil,
			deleteID:    "group-1",
			expectedErr: nil,
		},
		{
			name: "delete existing group",
			initial: []*store.Group{
				newTestGroup("group-1", nil),
			},
			deleteID:    "group-1",
			expectedErr: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			groupStore := store.NewGroupStore(store.NewMemLockStore())
			for _, g := range tc.initial {
				addGroupToStore(t, ctx, groupStore, g)
			}

			err := groupStore.Delete(ctx, tc.deleteID)
			if !errors.Is(err, tc.expectedErr) {
				t.Fatalf("Delete() error = %v, want %v", err, tc.expectedErr)
			}

			if len(tc.initial) > 0 && tc.deleteID == "group-1" {
				_, err := groupStore.Get(ctx, tc.deleteID)
				if !errors.Is(err, store.ErrNotFound) {
					t.Errorf("Get() after Delete returned error %v, want ErrNotFound", err)
				}
			}
		})
	}
}

func TestGroupStateEnum(t *testing.T) {
	tests := []struct {
		state pb.GroupStatus_State
		want  string
	}{
		{pb.GroupStatus_STATE_UNSPECIFIED, "STATE_UNSPECIFIED"},
		{pb.GroupStatus_STATE_IDLE, "STATE_IDLE"},
		{pb.GroupStatus_STATE_IDLE_YIELDED, "STATE_IDLE_YIELDED"},
		{pb.GroupStatus_STATE_LOCKED, "STATE_LOCKED"},
		{pb.GroupStatus_STATE_SWITCHING, "STATE_SWITCHING"},
		{pb.GroupStatus_STATE_UNKNOWN, "STATE_UNKNOWN"},
		{pb.GroupStatus_State(999), "999"},
	}

	for _, tt := range tests {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("%d.String() = %q, want %q", int(tt.state), got, tt.want)
		}
	}

	// Verify default value is GroupStatus_STATE_UNSPECIFIED (0)
	var g store.Group
	if state, _ := g.State(); state != pb.GroupStatus_STATE_UNSPECIFIED {
		t.Errorf("default Group.State = %v, want GroupStatus_STATE_UNSPECIFIED", state)
	}
}

func newTestGroup(id string, nodes []string) *store.Group {
	g := store.NewGroup(id, nil)
	if nodes != nil {
		g.SetNodes(nodes)
	}
	return g
}

func TestGroupStore_GetOrCreate(t *testing.T) {
	tests := []struct {
		name        string
		initial     []*store.Group
		groupID     string
		wantCreated bool
		expectedLen int
	}{
		{
			name:        "group does not exist",
			initial:     nil,
			groupID:     "group-1",
			wantCreated: true,
			expectedLen: 1,
		},
		{
			name: "group already exists",
			initial: []*store.Group{
				newTestGroup("group-1", nil),
			},
			groupID:     "group-1",
			wantCreated: false,
			expectedLen: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			groupStore := store.NewGroupStore(store.NewMemLockStore())
			for _, g := range tc.initial {
				addGroupToStore(t, ctx, groupStore, g)
			}

			got, created, err := groupStore.GetOrCreate(ctx, tc.groupID)
			if err != nil {
				t.Fatalf("GetOrCreate() returned error: %v", err)
			}

			if created != tc.wantCreated {
				t.Errorf("GetOrCreate() created = %t, want %t", created, tc.wantCreated)
			}

			if got.ID() != tc.groupID {
				t.Errorf("GetOrCreate() returned group with ID %q, want %q", got.ID(), tc.groupID)
			}

			list, err := groupStore.List(ctx)
			if err != nil {
				t.Fatalf("List() returned error: %v", err)
			}
			if len(list) != tc.expectedLen {
				t.Errorf("expected store size %d, got %d", tc.expectedLen, len(list))
			}

			// Verify it is in the store
			stored, err := groupStore.Get(ctx, tc.groupID)
			if err != nil {
				t.Fatalf("failed to retrieve group from store: %v", err)
			}
			if stored != got {
				t.Errorf("retrieved group does not match returned group pointer")
			}
		})
	}
}

func addGroupToStore(t *testing.T, ctx context.Context, s *store.GroupStore, g *store.Group) {
	t.Helper()
	got, created, err := s.GetOrCreate(ctx, g.ID())
	if err != nil {
		t.Fatalf("failed to add group to store: %v", err)
	}
	if !created {
		t.Fatalf("failed to add initial group: group %q already exists", g.ID())
	}
	nodes := g.Nodes()
	if len(nodes) > 0 {
		got.SetNodes(nodes)
	}
}
