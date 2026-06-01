package store

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
)

const (
	// Namespace is the namespace where the locks configmap resides.
	Namespace = "timeslice-system"
	// ConfigMapName is the name of the configmap storing the locks.
	ConfigMapName = "accelerator-orchestrator-locks"
)

// ConfigMapLockStore implements LockStore using a Kubernetes ConfigMap.
type ConfigMapLockStore struct {
	client kubernetes.Interface
}

// NewConfigMapLockStore creates a new ConfigMapLockStore.
func NewConfigMapLockStore(client kubernetes.Interface) *ConfigMapLockStore {
	return &ConfigMapLockStore{
		client: client,
	}
}

func (s *ConfigMapLockStore) getOrCreateConfigMap(ctx context.Context) (*corev1.ConfigMap, error) {
	cm, err := s.client.CoreV1().ConfigMaps(Namespace).Get(ctx, ConfigMapName, metav1.GetOptions{})
	if err == nil {
		return cm, nil
	}
	if !apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("failed to get configmap: %w", err)
	}

	// Create it
	cm = &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: Namespace,
		},
		Data: make(map[string]string),
	}
	created, err := s.client.CoreV1().ConfigMaps(Namespace).Create(ctx, cm, metav1.CreateOptions{})
	if err == nil {
		return created, nil
	}
	// If it was created concurrently, try to get it again
	if apierrors.IsAlreadyExists(err) {
		return s.client.CoreV1().ConfigMaps(Namespace).Get(ctx, ConfigMapName, metav1.GetOptions{})
	}
	return nil, fmt.Errorf("failed to create configmap: %w", err)
}

// GetLock returns the job_id currently holding the lock for the group.
func (s *ConfigMapLockStore) GetLock(ctx context.Context, groupID string) (string, error) {
	cm, err := s.client.CoreV1().ConfigMaps(Namespace).Get(ctx, ConfigMapName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return "", nil // ConfigMap doesn't exist, so no locks
		}
		return "", fmt.Errorf("failed to get configmap: %w", err)
	}
	if cm.Data == nil {
		return "", nil
	}
	return cm.Data[groupID], nil
}

// Lock persistently sets the job_id holding the lock for the group.
func (s *ConfigMapLockStore) Lock(ctx context.Context, groupID, jobID string) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		cm, err := s.getOrCreateConfigMap(ctx)
		if err != nil {
			return err
		}

		if cm.Data == nil {
			cm.Data = make(map[string]string)
		}

		current, ok := cm.Data[groupID]
		if ok && current != "" && current != jobID {
			return ErrAlreadyLocked
		}

		cm.Data[groupID] = jobID
		_, err = s.client.CoreV1().ConfigMaps(Namespace).Update(ctx, cm, metav1.UpdateOptions{})
		return err
	})
}

// Unlock persistently releases the lock for the group.
func (s *ConfigMapLockStore) Unlock(ctx context.Context, groupID, jobID string) error {
	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		cm, err := s.client.CoreV1().ConfigMaps(Namespace).Get(ctx, ConfigMapName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil // ConfigMap doesn't exist, so already unlocked
			}
			return fmt.Errorf("failed to get configmap: %w", err)
		}

		if cm.Data == nil {
			return nil // No data, so already unlocked
		}

		current := cm.Data[groupID]
		if current == "" {
			return nil // already unlocked
		}
		if current != jobID {
			return ErrNotLockHolder
		}

		delete(cm.Data, groupID)
		_, err = s.client.CoreV1().ConfigMaps(Namespace).Update(ctx, cm, metav1.UpdateOptions{})
		return err
	})
}
