package store_test

import (
	"context"
	"errors"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
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
