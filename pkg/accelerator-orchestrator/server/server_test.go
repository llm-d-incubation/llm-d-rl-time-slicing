package server_test

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"sync"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

const bufSize = 1024 * 1024

var lis *bufconn.Listener

//nolint:gocritic // Conflict with nonamedreturns linter
func initGRPCServer(groupStore server.GroupStore, jobStore server.JobStore) (*mockWorkQueue, func()) {
	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()
	mq := &mockWorkQueue{}
	ctrl := controller.NewController(nil, nil, mq, nil, nil)
	pb.RegisterAcceleratorOrchestratorServiceServer(s, server.NewServer(ctrl, groupStore, jobStore))
	go func() {
		if err := s.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			slog.Error("Server exited with error", "error", err)
			panic(err)
		}
	}()
	return mq, func() {
		s.GracefulStop()
		lis.Close()
	}
}

type mockGroupStore struct {
	getFunc  func(ctx context.Context, id string) (*store.Group, error)
	listFunc func(ctx context.Context) ([]*store.Group, error)
}

func (m *mockGroupStore) Get(ctx context.Context, id string) (*store.Group, error) {
	if m.getFunc != nil {
		return m.getFunc(ctx, id)
	}
	return nil, store.ErrNotFound
}

func (m *mockGroupStore) List(ctx context.Context) ([]*store.Group, error) {
	if m.listFunc != nil {
		return m.listFunc(ctx)
	}
	return nil, nil
}

type mockJobStore struct {
	getFunc         func(ctx context.Context, groupID, jobID string) (*store.Job, error)
	listByGroupFunc func(ctx context.Context, groupID string) ([]*store.Job, error)
}

func (m *mockJobStore) Get(ctx context.Context, groupID, jobID string) (*store.Job, error) {
	if m.getFunc != nil {
		return m.getFunc(ctx, groupID, jobID)
	}
	return nil, store.ErrNotFound
}

func (m *mockJobStore) ListByGroup(ctx context.Context, groupID string) ([]*store.Job, error) {
	if m.listByGroupFunc != nil {
		return m.listByGroupFunc(ctx, groupID)
	}
	return nil, nil
}

func bufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

func TestServer_Acquire(t *testing.T) {
	tests := []struct {
		name          string
		setupStores   func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore)
		groupID       string
		jobID         string
		timeout       time.Duration
		expectedCode  codes.Code
		verify        func(t *testing.T, resp *pb.AcquireResponse, err error)
		expectEnqueue bool
	}{
		{
			name: "group not found",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				return store.NewGroupStore(store.NewMemLockStore()), store.NewJobStore()
			},
			groupID:      "unknown-group",
			jobID:        "job-1",
			expectedCode: codes.NotFound,
		},
		{
			name: "group faulted",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				js := store.NewJobStore()
				job := store.NewJob("group-1", "job-2")
				job.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_FAULTED)
				if err := js.Put(ctx, job); err != nil {
					t.Fatalf("failed to put job: %v", err)
				}
				return gs, js
			},
			groupID:       "group-1",
			jobID:         "job-1",
			expectedCode:  codes.Unavailable,
			expectEnqueue: true,
		},
		{
			name: "timeout waiting for load",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-1",
			timeout:       100 * time.Millisecond,
			expectedCode:  codes.DeadlineExceeded,
			expectEnqueue: true,
		},
		{
			name: "success after wait",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}

				// Simulate controller loading the job after a short delay
				go func() {
					time.Sleep(50 * time.Millisecond)
					g.Status().SetLoadedJob("job-1")
				}()

				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-1",
			expectedCode:  codes.OK,
			expectEnqueue: true,
			verify: func(t *testing.T, resp *pb.AcquireResponse, err error) {
				t.Helper()
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !resp.Success {
					t.Errorf("Expected success to be true")
				}
				if resp.WaitedMs < 50 {
					t.Errorf("Expected WaitedMs to be at least 50ms, got %d", resp.WaitedMs)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			serverCtx := context.Background()
			clientCtx := context.Background()
			if tc.timeout > 0 {
				var cancel context.CancelFunc
				clientCtx, cancel = context.WithTimeout(clientCtx, tc.timeout)
				defer cancel()
			}
			gs, js := tc.setupStores(t, serverCtx)

			mq, cleanup := initGRPCServer(gs, js)
			defer cleanup()
			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(bufDialer),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial bufnet: %v", err)
			}
			defer conn.Close()
			client := pb.NewAcceleratorOrchestratorServiceClient(conn)

			resp, err := client.Acquire(clientCtx, &pb.AcquireRequest{
				GroupId: tc.groupID,
				JobId:   tc.jobID,
			})

			added := mq.GetAdded()
			if tc.expectEnqueue {
				if len(added) != 1 || added[0] != tc.groupID {
					t.Errorf("expected group %s to be enqueued, got %v", tc.groupID, added)
				}
			} else {
				if len(added) != 0 {
					t.Errorf("expected no enqueue, got %v", added)
				}
			}

			if tc.expectedCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok {
					t.Fatalf("Expected gRPC status error, got: %v", err)
				}
				if st.Code() != tc.expectedCode {
					t.Errorf("Expected code %v, got %v", tc.expectedCode, st.Code())
				}
				return
			}

			if tc.verify != nil {
				tc.verify(t, resp, err)
			}
		})
	}
}

func TestServer_Yield(t *testing.T) {
	tests := []struct {
		name          string
		setupStores   func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore)
		groupID       string
		jobID         string
		expectedCode  codes.Code
		verify        func(t *testing.T, resp *pb.YieldResponse, err error)
		expectEnqueue bool
	}{
		{
			name: "group not found",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				return store.NewGroupStore(store.NewMemLockStore()), store.NewJobStore()
			},
			groupID:      "unknown-group",
			jobID:        "job-1",
			expectedCode: codes.NotFound,
		},
		{
			name: "job does not hold lock",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				lockStore := store.NewMemLockStore()
				// Lock it for job-2
				if err := lockStore.Lock(ctx, "group-1", "job-2"); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
				gs := store.NewGroupStore(lockStore)
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				return gs, store.NewJobStore()
			},
			groupID:      "group-1",
			jobID:        "job-1", // job-1 tries to yield, but job-2 holds lock
			expectedCode: codes.PermissionDenied,
		},
		{
			name: "success with no waiters (snapshot deferred)",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				lockStore := store.NewMemLockStore()
				if err := lockStore.Lock(ctx, "group-1", "job-1"); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
				gs := store.NewGroupStore(lockStore)
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-1",
			expectedCode:  codes.OK,
			expectEnqueue: true,
			verify: func(t *testing.T, resp *pb.YieldResponse, err error) {
				t.Helper()
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !resp.Success {
					t.Errorf("Expected success to be true")
				}
				if resp.PendingWaiters != 0 {
					t.Errorf("Expected 0 pending waiters, got %d", resp.PendingWaiters)
				}
				if !resp.SnapshotDeferred {
					t.Errorf("Expected snapshot to be deferred")
				}
			},
		},
		{
			name: "success with waiters (snapshot not deferred)",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				lockStore := store.NewMemLockStore()
				if err := lockStore.Lock(ctx, "group-1", "job-1"); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
				gs := store.NewGroupStore(lockStore)
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				// Add a waiter
				g.Spec().RequestLock("job-2")
				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-1",
			expectedCode:  codes.OK,
			expectEnqueue: true,
			verify: func(t *testing.T, resp *pb.YieldResponse, err error) {
				t.Helper()
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !resp.Success {
					t.Errorf("Expected success to be true")
				}
				if resp.PendingWaiters != 1 {
					t.Errorf("Expected 1 pending waiter, got %d", resp.PendingWaiters)
				}
				if resp.SnapshotDeferred {
					t.Errorf("Expected snapshot NOT to be deferred")
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			gs, js := tc.setupStores(t, ctx)

			mq, cleanup := initGRPCServer(gs, js)
			defer cleanup()
			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(bufDialer),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial bufnet: %v", err)
			}
			defer conn.Close()
			client := pb.NewAcceleratorOrchestratorServiceClient(conn)

			resp, err := client.Yield(ctx, &pb.YieldRequest{
				GroupId: tc.groupID,
				JobId:   tc.jobID,
			})

			added := mq.GetAdded()
			if tc.expectEnqueue {
				if len(added) != 1 || added[0] != tc.groupID {
					t.Errorf("expected group %s to be enqueued, got %v", tc.groupID, added)
				}
			} else {
				if len(added) != 0 {
					t.Errorf("expected no enqueue, got %v", added)
				}
			}

			if tc.expectedCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok {
					t.Fatalf("Expected gRPC status error, got: %v", err)
				}
				if st.Code() != tc.expectedCode {
					t.Errorf("Expected code %v, got %v", tc.expectedCode, st.Code())
				}
				return
			}

			if tc.verify != nil {
				tc.verify(t, resp, err)
			}
		})
	}
}

func TestServer_ListGroups(t *testing.T) {
	tests := []struct {
		name         string
		setupStore   func(t *testing.T, ctx context.Context) server.GroupStore
		expectedIDs  []string
		expectedCode codes.Code
	}{
		{
			name: "no groups",
			setupStore: func(t *testing.T, ctx context.Context) server.GroupStore {
				t.Helper()
				return store.NewGroupStore(store.NewMemLockStore())
			},
			expectedIDs:  nil,
			expectedCode: codes.OK,
		},
		{
			name: "multiple groups",
			setupStore: func(t *testing.T, ctx context.Context) server.GroupStore {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group-1: %v", err)
				}
				_, _, err = gs.GetOrCreate(ctx, "group-2")
				if err != nil {
					t.Fatalf("failed to create group-2: %v", err)
				}
				return gs
			},
			expectedIDs:  []string{"group-1", "group-2"},
			expectedCode: codes.OK,
		},
		{
			name: "error path",
			setupStore: func(t *testing.T, ctx context.Context) server.GroupStore {
				t.Helper()
				return &mockGroupStore{
					listFunc: func(ctx context.Context) ([]*store.Group, error) {
						return nil, errors.New("database error")
					},
				}
			},
			expectedIDs:  nil,
			expectedCode: codes.Internal,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			gs := tc.setupStore(t, ctx)

			_, cleanup := initGRPCServer(gs, nil)
			defer cleanup()
			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(bufDialer),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial bufnet: %v", err)
			}
			defer conn.Close()
			client := pb.NewAcceleratorOrchestratorServiceClient(conn)

			resp, err := client.ListGroups(ctx, &pb.ListGroupsRequest{})

			if tc.expectedCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok {
					t.Fatalf("Expected gRPC status error, got: %v", err)
				}
				if st.Code() != tc.expectedCode {
					t.Errorf("Expected code %v, got %v", tc.expectedCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			expectedMap := make(map[string]bool)
			for _, id := range tc.expectedIDs {
				expectedMap[id] = true
			}

			if len(resp.GroupIds) != len(tc.expectedIDs) {
				t.Errorf("Expected %d groups, got %d", len(tc.expectedIDs), len(resp.GroupIds))
			}

			for _, id := range resp.GroupIds {
				if !expectedMap[id] {
					t.Errorf("Unexpected group ID: %s", id)
				}
			}
		})
	}
}

func TestServer_GetGroupStatus(t *testing.T) {
	tests := []struct {
		name         string
		setupStores  func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore)
		groupID      string
		expectedCode codes.Code
		verify       func(t *testing.T, resp *pb.GetGroupStatusResponse)
	}{
		{
			name: "group not found",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				return store.NewGroupStore(store.NewMemLockStore()), store.NewJobStore()
			},
			groupID:      "unknown-group",
			expectedCode: codes.NotFound,
		},
		{
			name: "failed to get group (internal)",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := &mockGroupStore{
					getFunc: func(ctx context.Context, id string) (*store.Group, error) {
						return nil, errors.New("database error")
					},
				}
				return gs, store.NewJobStore()
			},
			groupID:      "group-1",
			expectedCode: codes.Internal,
		},
		{
			name: "group state unknown",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				g.Status().SetState(pb.GroupStatus_STATE_UNKNOWN)
				return gs, store.NewJobStore()
			},
			groupID:      "group-1",
			expectedCode: codes.Unavailable,
		},
		{
			name: "failed to list jobs",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				_, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				js := &mockJobStore{
					listByGroupFunc: func(ctx context.Context, groupID string) ([]*store.Job, error) {
						return nil, errors.New("database error")
					},
				}
				return gs, js
			},
			groupID:      "group-1",
			expectedCode: codes.Internal,
		},
		{
			name: "success no active job",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				gs := store.NewGroupStore(store.NewMemLockStore())
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				g.Status().SetState(pb.GroupStatus_STATE_IDLE)
				return gs, store.NewJobStore()
			},
			groupID:      "group-1",
			expectedCode: codes.OK,
			verify: func(t *testing.T, resp *pb.GetGroupStatusResponse) {
				t.Helper()
				if resp.Group.GroupId != "group-1" {
					t.Errorf("Expected group ID group-1, got %s", resp.Group.GroupId)
				}
				if resp.Group.GroupState != pb.GroupStatus_STATE_IDLE {
					t.Errorf("Expected state IDLE, got %s", resp.Group.GroupState)
				}
				if len(resp.AgentJobStates) != 0 {
					t.Errorf("Expected no agent job states, got %d", len(resp.AgentJobStates))
				}
			},
		},
		{
			name: "success with multiple jobs and agent states",
			setupStores: func(t *testing.T, ctx context.Context) (server.GroupStore, server.JobStore) {
				t.Helper()
				lockStore := store.NewMemLockStore()
				if err := lockStore.Lock(ctx, "group-1", "job-1"); err != nil {
					t.Fatalf("failed to setup lock in lockStore: %v", err)
				}
				gs := store.NewGroupStore(lockStore)
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				g.Status().SetState(pb.GroupStatus_STATE_LOCKED)

				js := store.NewJobStore()

				// Job 1 (Running on all nodes)
				job1 := store.NewJob("group-1", "job-1")
				job1.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_RUNNING)
				job1.UpdateContextState("node-2", pb.SnapshotAgentJobState_STATE_RUNNING)
				if err := js.Put(ctx, job1); err != nil {
					t.Fatalf("failed to put job1: %v", err)
				}

				// Job 2 - smaller and only needs one node (Runs only on node-1, unspecified on node-2)
				job2 := store.NewJob("group-1", "job-2")
				job2.UpdateContextState("node-1", pb.SnapshotAgentJobState_STATE_SAVED)
				job2.UpdateContextState("node-2", pb.SnapshotAgentJobState_STATE_UNSPECIFIED)
				if err := js.Put(ctx, job2); err != nil {
					t.Fatalf("failed to put job2: %v", err)
				}

				return gs, js
			},
			groupID:      "group-1",
			expectedCode: codes.OK,
			verify: func(t *testing.T, resp *pb.GetGroupStatusResponse) {
				t.Helper()
				if resp.Group.GroupId != "group-1" {
					t.Errorf("Expected group ID group-1, got %s", resp.Group.GroupId)
				}
				if resp.Group.GroupState != pb.GroupStatus_STATE_LOCKED {
					t.Errorf("Expected state LOCKED, got %s", resp.Group.GroupState)
				}
				if resp.Group.ActiveJob != "job-1" {
					t.Errorf("Expected active job job-1, got %s", resp.Group.ActiveJob)
				}
				if resp.Group.LockingJob != "job-1" {
					t.Errorf("Expected locking job job-1, got %s", resp.Group.LockingJob)
				}

				// We expect 4 agent job states in total (2 jobs * 2 nodes)
				if len(resp.AgentJobStates) != 4 {
					t.Errorf("Expected 4 agent job states, got %d", len(resp.AgentJobStates))
				}

				// Helper map to verify
				expected := map[string]map[string]pb.SnapshotAgentJobState_State{
					"job-1": {
						"node-1": pb.SnapshotAgentJobState_STATE_RUNNING,
						"node-2": pb.SnapshotAgentJobState_STATE_RUNNING,
					},
					"job-2": {
						"node-1": pb.SnapshotAgentJobState_STATE_SAVED,
						"node-2": pb.SnapshotAgentJobState_STATE_UNSPECIFIED,
					},
				}

				for _, s := range resp.AgentJobStates {
					jobExpected, ok := expected[s.JobId]
					if !ok {
						t.Errorf("Unexpected job ID: %s", s.JobId)
						continue
					}
					nodeExpectedState, ok := jobExpected[s.Agent]
					if !ok {
						t.Errorf("Unexpected agent %s for job %s", s.Agent, s.JobId)
						continue
					}
					if s.JobState != nodeExpectedState {
						t.Errorf("Expected state %v for job %s on agent %s, got %v", nodeExpectedState, s.JobId, s.Agent, s.JobState)
					}
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			gs, js := tc.setupStores(t, ctx)

			_, cleanup := initGRPCServer(gs, js)
			defer cleanup()
			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(bufDialer),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("Failed to dial bufnet: %v", err)
			}
			defer conn.Close()
			client := pb.NewAcceleratorOrchestratorServiceClient(conn)

			resp, err := client.GetGroupStatus(ctx, &pb.GetGroupStatusRequest{GroupId: tc.groupID})

			if tc.expectedCode != codes.OK {
				if err == nil {
					t.Fatalf("Expected error, got nil")
				}
				st, ok := status.FromError(err)
				if !ok {
					t.Fatalf("Expected gRPC status error, got: %v", err)
				}
				if st.Code() != tc.expectedCode {
					t.Errorf("Expected code %v, got %v", tc.expectedCode, st.Code())
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if tc.verify != nil {
				tc.verify(t, resp)
			}
		})
	}
}

type mockWorkQueue struct {
	controller.WorkQueue
	mu    sync.Mutex
	added []string
}

func (m *mockWorkQueue) Add(groupID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.added = append(m.added, groupID)
}

func (m *mockWorkQueue) GetAdded() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]string, len(m.added))
	copy(cp, m.added)
	return cp
}

func (m *mockWorkQueue) AddRateLimited(groupID string) {}
func (m *mockWorkQueue) Forget(groupID string)         {}
func (m *mockWorkQueue) Done(groupID string)           {}
func (m *mockWorkQueue) Get() (string, bool)           { return "", false }
func (m *mockWorkQueue) ShutDown()                     {}
