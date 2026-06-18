//nolint:testpackage // MockSnapshotAgentStore needs to be in package controller to be shared with internal tests.
package controller

import (
	"context"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

// MockSnapshotAgentStore is a shared mock for testing.
// It is exported so it can be used by external tests (package controller_test).
type MockSnapshotAgentStore struct {
	store.SnapshotAgentStore
	GetStatusFunc func(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	SnapshotFunc  func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error)
	OperationFunc func(ctx context.Context, nodeName, operationID string) (*agentpb.GetOperationResponse, error)
	RestoreFunc   func(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.RestoreResponse, error)

	// Queued responses for GetOperation
	OperationResponses []*agentpb.GetOperationResponse
	OperationIndex     int
}

func (m *MockSnapshotAgentStore) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	if m.GetStatusFunc != nil {
		return m.GetStatusFunc(ctx, nodeName)
	}
	return &agentpb.StatusResponse{}, nil
}

func (m *MockSnapshotAgentStore) CloseClient(nodeName string) error {
	return nil
}

func (m *MockSnapshotAgentStore) Snapshot(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.SnapshotResponse, error) {
	if m.SnapshotFunc != nil {
		return m.SnapshotFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.SnapshotResponse{}, nil
}

func (m *MockSnapshotAgentStore) GetOperation(
	ctx context.Context, nodeName, operationID string,
) (*agentpb.GetOperationResponse, error) {
	if m.OperationFunc != nil {
		return m.OperationFunc(ctx, nodeName, operationID)
	}
	if len(m.OperationResponses) > 0 {
		if m.OperationIndex < len(m.OperationResponses) {
			resp := m.OperationResponses[m.OperationIndex]
			m.OperationIndex++
			return resp, nil
		}
		return m.OperationResponses[len(m.OperationResponses)-1], nil
	}
	return &agentpb.GetOperationResponse{}, nil
}

func (m *MockSnapshotAgentStore) Restore(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.RestoreResponse, error) {
	if m.RestoreFunc != nil {
		return m.RestoreFunc(ctx, nodeName, jobID, groupID)
	}
	return &agentpb.RestoreResponse{}, nil
}
