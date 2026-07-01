package server

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
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

const bufSize = 1024 * 1024

var lis *bufconn.Listener

// BufDialer is exported for external tests.
func BufDialer(context.Context, string) (net.Conn, error) {
	return lis.Dial()
}

// InitGRPCServer is exported for external tests.
// It sets the acquirePollInterval to 1ms for fast testing.
func InitGRPCServer(groupStore GroupStore, jobStore JobStore) (*Server, *MockWorkQueue, func()) {
	lis = bufconn.Listen(bufSize)
	s := grpc.NewServer()
	mq := &MockWorkQueue{}
	ctrl := controller.NewController(nil, nil, mq, nil, nil)
	srv := NewServer(ctrl, groupStore, jobStore)
	srv.acquirePollInterval = 1 * time.Millisecond // Set to 1ms for testing
	pb.RegisterAcceleratorOrchestratorServiceServer(s, srv)
	go func() {
		if err := s.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			slog.Error("Server exited with error", "error", err)
			panic(err)
		}
	}()
	return srv, mq, func() {
		s.GracefulStop()
		lis.Close()
	}
}

// MockWorkQueue is exported for external tests.
type MockWorkQueue struct {
	controller.WorkQueue
	mu    sync.Mutex
	added []string
}

func (m *MockWorkQueue) Add(groupID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.added = append(m.added, groupID)
}

func (m *MockWorkQueue) GetAdded() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]string, len(m.added))
	copy(cp, m.added)
	return cp
}

func (m *MockWorkQueue) AddRateLimited(groupID string) {}
func (m *MockWorkQueue) Forget(groupID string)         {}
func (m *MockWorkQueue) Done(groupID string)           {}
func (m *MockWorkQueue) Get() (string, bool)           { return "", false }
func (m *MockWorkQueue) ShutDown()                     {}

// MockGroupStore is exported for external tests.
type MockGroupStore struct {
	GetFunc  func(ctx context.Context, id string) (*store.Group, error)
	ListFunc func(ctx context.Context) ([]*store.Group, error)
	SetGroup func(g *store.Group)
}

func (m *MockGroupStore) Get(ctx context.Context, id string) (*store.Group, error) {
	if m.GetFunc != nil {
		return m.GetFunc(ctx, id)
	}
	return nil, store.ErrNotFound
}

func (m *MockGroupStore) List(ctx context.Context) ([]*store.Group, error) {
	if m.ListFunc != nil {
		return m.ListFunc(ctx)
	}
	return nil, nil
}

// MockJobStore is exported for external tests.
type MockJobStore struct {
	GetFunc         func(ctx context.Context, groupID, jobID string) (*store.Job, error)
	ListByGroupFunc func(ctx context.Context, groupID string) ([]*store.Job, error)
}

func (m *MockJobStore) Get(ctx context.Context, groupID, jobID string) (*store.Job, error) {
	if m.GetFunc != nil {
		return m.GetFunc(ctx, groupID, jobID)
	}
	return nil, store.ErrNotFound
}

func (m *MockJobStore) ListByGroup(ctx context.Context, groupID string) ([]*store.Job, error) {
	if m.ListByGroupFunc != nil {
		return m.ListByGroupFunc(ctx, groupID)
	}
	return nil, nil
}

// MockGroupLockStore is exported for external tests.
type MockGroupLockStore struct {
	lock string
}

func (m *MockGroupLockStore) GetLock(ctx context.Context) (string, error) {
	return m.lock, nil
}

func (m *MockGroupLockStore) Lock(ctx context.Context, jobID string) error {
	m.lock = jobID
	return nil
}

func (m *MockGroupLockStore) Unlock(ctx context.Context, jobID string) error {
	m.lock = ""
	return nil
}

func TestServer_Acquire_Whitebox(t *testing.T) {
	tests := []struct {
		name          string
		setupStores   func(t *testing.T, ctx context.Context) (GroupStore, JobStore)
		groupID       string
		jobID         string
		expectedCode  codes.Code
		verify        func(t *testing.T, resp *pb.AcquireResponse, err error)
		expectEnqueue bool
		hook          func(t *testing.T, srv *Server, gs GroupStore, js JobStore, cancel context.CancelFunc)
	}{
		{
			name: "block when it has not yet reconciled but another job requesting lock",
			setupStores: func(t *testing.T, ctx context.Context) (GroupStore, JobStore) {
				t.Helper()
				lockStore := store.NewMemLockStore()
				// Imagine job-1 got the lock then yielded.
				// Then job-2 asks for the lock. That acquire is now in waiting while job-1 unloads.
				// Lock it for job-2 (LockingJob = job-2)
				if err := lockStore.Lock(ctx, "group-1", "job-2"); err != nil {
					t.Fatalf("failed to lock: %v", err)
				}
				gs := store.NewGroupStore(lockStore)
				g, _, err := gs.GetOrCreate(ctx, "group-1")
				if err != nil {
					t.Fatalf("failed to create group: %v", err)
				}
				// Simulate job-1 has not yet unloaded (LoadedJob = job-1)
				g.Status().SetLoadedJob("job-1")
				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-1",        // job-1 tries to acquire, should block because lock is for job-2
			expectedCode:  codes.Canceled, // We expect Canceled because we will cancel it manually
			expectEnqueue: true,
			hook: func(t *testing.T, srv *Server, gs GroupStore, js JobStore, cancel context.CancelFunc) {
				t.Helper()
				tickCalled := make(chan struct{}, 1)
				origCheck := srv.checkAcquire
				srv.checkAcquire = func(ctx context.Context, groupID, jobID string, startTime time.Time) (*pb.AcquireResponse, error, bool) {
					resp, err, done := origCheck(ctx, groupID, jobID, startTime)
					select {
					case tickCalled <- struct{}{}:
					default:
					}
					return resp, err, done
				}
				// Wait for 5 ticks to be sure it is blocked, then cancel
				go func() {
					for i := 0; i < 5; i++ {
						<-tickCalled
					}
					cancel()
				}()
			},
		},
		{
			name: "succeed when job loads later",
			setupStores: func(t *testing.T, ctx context.Context) (GroupStore, JobStore) {
				t.Helper()
				// Create group 1: locked for job-2, but job-1 is loaded
				m1 := &MockGroupLockStore{lock: "job-2"}
				g1, err := store.NewGroup(ctx, "group-1", m1)
				if err != nil {
					t.Fatalf("failed to create group1: %v", err)
				}
				g1.Status().SetLoadedJob("job-1")

				var mu sync.Mutex
				currentGroup := g1

				gs := &MockGroupStore{
					GetFunc: func(ctx context.Context, id string) (*store.Group, error) {
						mu.Lock()
						defer mu.Unlock()
						return currentGroup, nil
					},
					SetGroup: func(g *store.Group) {
						mu.Lock()
						currentGroup = g
						mu.Unlock()
					},
				}
				return gs, store.NewJobStore()
			},
			groupID:       "group-1",
			jobID:         "job-2", // job-2 tries to acquire, should succeed after job-2 is loaded
			expectedCode:  codes.OK,
			expectEnqueue: true,
			hook: func(t *testing.T, srv *Server, gs GroupStore, js JobStore, cancel context.CancelFunc) {
				t.Helper()
				mGS, ok := gs.(*MockGroupStore)
				if !ok {
					t.Fatalf("expected *MockGroupStore, got %T", gs)
				}
				tickCalled := make(chan struct{}, 1)
				origCheck := srv.checkAcquire
				srv.checkAcquire = func(ctx context.Context, groupID, jobID string, startTime time.Time) (*pb.AcquireResponse, error, bool) {
					resp, err, done := origCheck(ctx, groupID, jobID, startTime)
					select {
					case tickCalled <- struct{}{}:
					default:
					}
					return resp, err, done
				}
				// Wait for first tick, then update group to g2
				go func() {
					<-tickCalled
					m2 := &MockGroupLockStore{lock: "job-2"}
					g2, err := store.NewGroup(context.Background(), "group-1", m2)
					if err != nil {
						t.Errorf("failed to create group2: %v", err)
						return
					}
					g2.Status().SetLoadedJob("job-2")
					mGS.SetGroup(g2)
				}()
			},
			verify: func(t *testing.T, resp *pb.AcquireResponse, err error) {
				t.Helper()
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !resp.Success {
					t.Errorf("Expected success to be true")
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			serverCtx := context.Background()
			clientCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			gs, js := tc.setupStores(t, serverCtx)

			srv, mq, cleanup := InitGRPCServer(gs, js)
			defer cleanup()

			if tc.hook != nil {
				tc.hook(t, srv, gs, js, cancel)
			}

			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(BufDialer),
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
