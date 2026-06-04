package store_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func TestConfigMapLockStore_LockAndUnlock(t *testing.T) {
	type step struct {
		op           string // "lock", "unlock", "get"
		group        string
		job          string
		expectedErr  error
		expectedLock string
	}

	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "basic lock, get, conflict, unlock sequence",
			steps: []step{
				{op: "get", group: "group-1", expectedLock: ""},
				{op: "lock", group: "group-1", job: "job-a", expectedErr: nil},
				{op: "get", group: "group-1", expectedLock: "job-a"},
				{op: "get", group: "group-2", expectedLock: ""}, // group-2 shouldn't have lock
				{op: "lock", group: "group-1", job: "job-b", expectedErr: store.ErrAlreadyLocked},
				{op: "lock", group: "group-1", job: "job-a", expectedErr: nil}, // idempotent lock
				{op: "unlock", group: "group-1", job: "job-b", expectedErr: store.ErrNotLockHolder},
				{op: "get", group: "group-1", expectedLock: "job-a"},
				{op: "unlock", group: "group-1", job: "job-a", expectedErr: nil},
				{op: "get", group: "group-1", expectedLock: ""},
				{op: "unlock", group: "group-1", job: "job-a", expectedErr: nil}, // idempotent unlock
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			client := fake.NewSimpleClientset()
			s := store.NewConfigMapLockStore(client)

			for i, step := range tc.steps {
				var err error
				switch step.op {
				case "lock":
					err = s.Lock(ctx, step.group, step.job)
					if !errors.Is(err, step.expectedErr) {
						t.Fatalf("step %d: Lock(%q, %q) error = %v, want %v", i, step.group, step.job, err, step.expectedErr)
					}
				case "unlock":
					err = s.Unlock(ctx, step.group, step.job)
					if !errors.Is(err, step.expectedErr) {
						t.Fatalf("step %d: Unlock(%q, %q) error = %v, want %v", i, step.group, step.job, err, step.expectedErr)
					}
				case "get":
					got, err := s.GetLock(ctx, step.group)
					if err != nil {
						t.Fatalf("step %d: GetLock(%q) failed: %v", i, step.group, err)
					}
					if got != step.expectedLock {
						t.Fatalf("step %d: GetLock(%q) = %q, want %q", i, step.group, got, step.expectedLock)
					}
				}
			}
		})
	}
}

func TestConfigMapLockStore_GetLockScenarios(t *testing.T) {
	tests := []struct {
		name        string
		configMap   *corev1.ConfigMap
		groupName   string
		expected    string
		expectedErr error
	}{
		{
			name:        "non-existent ConfigMap",
			configMap:   nil,
			groupName:   "group-1",
			expected:    "",
			expectedErr: nil,
		},
		{
			name: "ConfigMap with nil Data",
			configMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      store.ConfigMapName,
					Namespace: store.Namespace,
				},
				Data: nil,
			},
			groupName:   "group-1",
			expected:    "",
			expectedErr: nil,
		},
		{
			name: "ConfigMap with existing lock",
			configMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      store.ConfigMapName,
					Namespace: store.Namespace,
				},
				Data: map[string]string{
					"group-1": "job-a",
				},
			},
			groupName:   "group-1",
			expected:    "job-a",
			expectedErr: nil,
		},
		{
			name: "ConfigMap exists but group key is missing",
			configMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      store.ConfigMapName,
					Namespace: store.Namespace,
				},
				Data: map[string]string{
					"group-1": "job-a",
				},
			},
			groupName:   "group-2",
			expected:    "",
			expectedErr: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			client := fake.NewSimpleClientset()
			s := store.NewConfigMapLockStore(client)

			if tc.configMap != nil {
				_, err := client.CoreV1().ConfigMaps(store.Namespace).Create(ctx, tc.configMap, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to pre-create ConfigMap: %v", err)
				}
			}

			got, err := s.GetLock(ctx, tc.groupName)
			if !errors.Is(err, tc.expectedErr) {
				t.Fatalf("GetLock() error = %v, want %v", err, tc.expectedErr)
			}
			if got != tc.expected {
				t.Errorf("GetLock() = %q, want %q", got, tc.expected)
			}
		})
	}
}

func TestConfigMapLockStore_Lock_RetryOnConflict(t *testing.T) {
	ctx := context.Background()
	groupID := "group-1"
	jobID := "job-a"

	tests := []struct {
		name          string
		conflictCount int
		shouldSucceed bool
	}{
		{
			name:          "succeeds after 2 conflicts",
			conflictCount: 2,
			shouldSucceed: true,
		},
		{
			name:          "fails after too many conflicts (6 conflicts)",
			conflictCount: 6,
			shouldSucceed: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			lockStore := store.NewConfigMapLockStore(client)

			// Pre-create the configmap so Lock doesn't have to create it (keeps reactor simpler)
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      store.ConfigMapName,
					Namespace: store.Namespace,
				},
				Data: make(map[string]string),
			}
			_, err := client.CoreV1().ConfigMaps(store.Namespace).Create(ctx, cm, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to pre-create ConfigMap: %v", err)
			}

			attempts := 0
			client.PrependReactor("update", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
				attempts++
				if attempts <= tc.conflictCount {
					return true, nil, apierrors.NewConflict(
						schema.GroupResource{Resource: "configmaps"},
						store.ConfigMapName,
						errors.New("simulated conflict"),
					)
				}
				return false, nil, nil // Let the fake client handle it normally
			})

			err = lockStore.Lock(ctx, groupID, jobID)
			if tc.shouldSucceed {
				verifyLockSuccess(t, ctx, lockStore, groupID, jobID, err)
			} else {
				verifyConflict(t, "Lock()", err)
			}
		})
	}
}

func TestConfigMapLockStore_Unlock_RetryOnConflict(t *testing.T) {
	ctx := context.Background()
	groupID := "group-1"
	jobID := "job-a"

	tests := []struct {
		name          string
		conflictCount int
		shouldSucceed bool
	}{
		{
			name:          "succeeds after 2 conflicts",
			conflictCount: 2,
			shouldSucceed: true,
		},
		{
			name:          "fails after too many conflicts (6 conflicts)",
			conflictCount: 6,
			shouldSucceed: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			lockStore := store.NewConfigMapLockStore(client)

			// Pre-create the configmap with lock
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      store.ConfigMapName,
					Namespace: store.Namespace,
				},
				Data: map[string]string{
					groupID: jobID,
				},
			}
			_, err := client.CoreV1().ConfigMaps(store.Namespace).Create(ctx, cm, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to pre-create ConfigMap: %v", err)
			}

			attempts := 0
			client.PrependReactor("update", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
				attempts++
				if attempts <= tc.conflictCount {
					return true, nil, apierrors.NewConflict(
						schema.GroupResource{Resource: "configmaps"},
						store.ConfigMapName,
						errors.New("simulated conflict"),
					)
				}
				return false, nil, nil
			})

			err = lockStore.Unlock(ctx, groupID, jobID)
			if tc.shouldSucceed {
				verifyUnlockSuccess(t, ctx, lockStore, groupID, err)
			} else {
				verifyConflict(t, "Unlock()", err)
			}
		})
	}
}

func verifyLockSuccess(t *testing.T, ctx context.Context, lockStore *store.ConfigMapLockStore, groupID, jobID string, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("Lock() failed: %v, want success", err)
	}
	got, err := lockStore.GetLock(ctx, groupID)
	if err != nil {
		t.Fatalf("GetLock() failed: %v", err)
	}
	if got != jobID {
		t.Errorf("GetLock() = %q, want %q", got, jobID)
	}
}

func verifyUnlockSuccess(t *testing.T, ctx context.Context, lockStore *store.ConfigMapLockStore, groupID string, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("Unlock() failed: %v, want success", err)
	}
	got, err := lockStore.GetLock(ctx, groupID)
	if err != nil {
		t.Fatalf("GetLock() failed: %v", err)
	}
	if got != "" {
		t.Errorf("GetLock() = %q, want empty (unlocked)", got)
	}
}

func verifyConflict(t *testing.T, op string, err error) {
	t.Helper()
	if err == nil {
		t.Errorf("%s succeeded, want failure due to conflict", op)
	}
	if !apierrors.IsConflict(err) {
		t.Errorf("%s error = %v, want Conflict error", op, err)
	}
}

func TestConfigMapLockStore_Lock_GetOrCreateRetry(t *testing.T) {
	ctx := context.Background()
	groupID := "group-1"
	jobID := "job-a"

	tests := []struct {
		name               string
		setupReactor       func(client *fake.Clientset)
		shouldSucceed      bool
		expectedErrMessage string
	}{
		{
			name: "success on first try",
			setupReactor: func(client *fake.Clientset) {
				// No special reactor needed, fake client handles it
			},
			shouldSucceed: true,
		},
		{
			name: "succeeds after 1 retry (concurrent create)",
			setupReactor: func(client *fake.Clientset) {
				attempts := 0
				client.PrependReactor("create", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
					attempts++
					if attempts == 1 {
						// Simulate concurrent creation by adding it to tracker
						createAction, ok := action.(k8stesting.CreateAction)
						if !ok {
							return true, nil, errors.New("expected CreateAction")
						}
						obj := createAction.GetObject()
						if err := client.Tracker().Add(obj); err != nil {
							return true, nil, err
						}
						return true, nil, apierrors.NewAlreadyExists(
							schema.GroupResource{Resource: "configmaps"},
							store.ConfigMapName,
						)
					}
					return false, nil, nil
				})
			},
			shouldSucceed: true,
		},
		{
			name: "fails after too many attempts",
			setupReactor: func(client *fake.Clientset) {
				// Always fail Create with AlreadyExists, but never add to tracker
				// So Get will always return NotFound
				client.PrependReactor("create", "configmaps", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(
						schema.GroupResource{Resource: "configmaps"},
						store.ConfigMapName,
					)
				})
			},
			shouldSucceed:      false,
			expectedErrMessage: "failed to get or create configmap after 5 attempts",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			tc.setupReactor(client)
			lockStore := store.NewConfigMapLockStore(client)

			err := lockStore.Lock(ctx, groupID, jobID)
			if tc.shouldSucceed {
				verifyLockSuccess(t, ctx, lockStore, groupID, jobID, err)
			} else {
				if err == nil {
					t.Errorf("Lock() succeeded, want failure")
				} else if !strings.Contains(err.Error(), tc.expectedErrMessage) {
					t.Errorf("Lock() error = %v, want error containing %q", err, tc.expectedErrMessage)
				}
			}
		})
	}
}
