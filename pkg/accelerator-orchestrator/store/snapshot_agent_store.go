package store

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// SnapshotAgentStore defines the interface for communicating with snapshot agents.
type SnapshotAgentStore interface {
	GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
	CloseClient(nodeName string) error
	Snapshot(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.SnapshotResponse, error)
	GetOperation(ctx context.Context, nodeName, operationID string) (*agentpb.GetOperationResponse, error)
	Restore(ctx context.Context, nodeName, jobID, groupID string) (*agentpb.RestoreResponse, error)
}

type clientEntry struct {
	client agentpb.SnapshotAgentServiceClient
	conn   *grpc.ClientConn
}

type cacheEntry struct {
	response  *agentpb.StatusResponse
	timestamp time.Time
}

// GRPCSnapshotAgentStore implements SnapshotAgentStore using gRPC.
type GRPCSnapshotAgentStore struct {
	mu          sync.Mutex
	clients     map[string]*clientEntry
	cache       map[string]*cacheEntry
	cacheTTL    time.Duration
	defaultPort int
}

// NewGRPCSnapshotAgentStore creates a new GRPCSnapshotAgentStore.
// If ttl is <= 0, caching is disabled.
func NewGRPCSnapshotAgentStore(ttl time.Duration, defaultPort int) *GRPCSnapshotAgentStore {
	return &GRPCSnapshotAgentStore{
		clients:     make(map[string]*clientEntry),
		cache:       make(map[string]*cacheEntry),
		cacheTTL:    ttl,
		defaultPort: defaultPort,
	}
}

func (s *GRPCSnapshotAgentStore) getClient(address string) (agentpb.SnapshotAgentServiceClient, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry, ok := s.clients[address]; ok {
		return entry.client, nil
	}

	// Dial the agent using NewClient (preferred in newer gRPC versions)
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to dial agent at %s: %w", address, err)
	}

	client := agentpb.NewSnapshotAgentServiceClient(conn)
	s.clients[address] = &clientEntry{
		client: client,
		conn:   conn,
	}
	return client, nil
}

// GetStatus queries the snapshot agent for the given node for its status.
// It returns cached results if they are within the TTL.
func (s *GRPCSnapshotAgentStore) GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error) {
	address := s.resolveNodeAddress(nodeName)

	s.mu.Lock()
	if s.cacheTTL > 0 {
		if entry, ok := s.cache[address]; ok {
			if time.Since(entry.timestamp) < s.cacheTTL {
				s.mu.Unlock()
				return entry.response, nil
			}
		}
	}
	s.mu.Unlock()

	client, err := s.getClient(address)
	if err != nil {
		return nil, err
	}

	resp, err := client.Status(ctx, &agentpb.StatusRequest{})
	if err != nil {
		return nil, fmt.Errorf("failed to get status from agent at %s: %w", address, err)
	}

	if s.cacheTTL > 0 {
		s.mu.Lock()
		s.cache[address] = &cacheEntry{
			response:  resp,
			timestamp: time.Now(),
		}
		s.mu.Unlock()
	}

	return resp, nil
}

// Snapshot triggers a snapshot on the agent for the given node.
func (s *GRPCSnapshotAgentStore) Snapshot(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.SnapshotResponse, error) {
	address := s.resolveNodeAddress(nodeName)
	client, err := s.getClient(address)
	if err != nil {
		return nil, err
	}

	resp, err := client.Snapshot(ctx, &agentpb.SnapshotRequest{
		JobId: jobID,
		Group: groupID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to trigger snapshot on agent at %s: %w", address, err)
	}

	return resp, nil
}

// GetOperation queries the snapshot agent for the status of a long-running operation.
func (s *GRPCSnapshotAgentStore) GetOperation(
	ctx context.Context, nodeName, operationID string,
) (*agentpb.GetOperationResponse, error) {
	address := s.resolveNodeAddress(nodeName)
	client, err := s.getClient(address)
	if err != nil {
		return nil, err
	}

	resp, err := client.GetOperation(ctx, &agentpb.GetOperationRequest{
		OperationId: operationID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get operation %s from agent at %s: %w", operationID, address, err)
	}

	return resp, nil
}

// Restore triggers an asynchronous restoration of the accelerator context for a job on the node.
func (s *GRPCSnapshotAgentStore) Restore(
	ctx context.Context, nodeName, jobID, groupID string,
) (*agentpb.RestoreResponse, error) {
	address := s.resolveNodeAddress(nodeName)
	client, err := s.getClient(address)
	if err != nil {
		return nil, err
	}

	resp, err := client.Restore(ctx, &agentpb.RestoreRequest{
		JobId: jobID,
		Group: groupID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to trigger restore for job %s on agent at %s: %w", jobID, address, err)
	}

	return resp, nil
}

// CloseClient closes the gRPC connection and clears the cache for the given node.
func (s *GRPCSnapshotAgentStore) CloseClient(nodeName string) error {
	address := s.resolveNodeAddress(nodeName)

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.clients[address]
	if !ok {
		return nil // Already closed or never opened
	}

	var closeErr error
	if entry.conn != nil {
		closeErr = entry.conn.Close()
	}

	delete(s.clients, address)
	delete(s.cache, address)

	if closeErr != nil {
		return fmt.Errorf("failed to close connection for node %s: %w", nodeName, closeErr)
	}
	return nil
}

func (s *GRPCSnapshotAgentStore) resolveNodeAddress(nodeName string) string {
	// If the nodeName already contains a port, use it as-is.
	if _, _, err := net.SplitHostPort(nodeName); err == nil {
		return nodeName
	}
	// Otherwise, append the default snapshot-agent port.
	return net.JoinHostPort(nodeName, strconv.Itoa(s.defaultPort))
}
