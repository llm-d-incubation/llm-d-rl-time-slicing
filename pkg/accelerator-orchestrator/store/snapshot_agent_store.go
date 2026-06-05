package store

import (
	"context"
	"fmt"
	"sync"
	"time"

	agentpb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// SnapshotAgentStore defines the interface for communicating with snapshot agents.
type SnapshotAgentStore interface {
	GetStatus(ctx context.Context, nodeName string) (*agentpb.StatusResponse, error)
}

type cacheEntry struct {
	response  *agentpb.StatusResponse
	timestamp time.Time
}

// GRPCSnapshotAgentStore implements SnapshotAgentStore using gRPC.
type GRPCSnapshotAgentStore struct {
	mu       sync.Mutex
	clients  map[string]agentpb.SnapshotAgentServiceClient
	cache    map[string]*cacheEntry
	cacheTTL time.Duration
}

// NewGRPCSnapshotAgentStore creates a new GRPCSnapshotAgentStore.
// If ttl is <= 0, caching is disabled.
func NewGRPCSnapshotAgentStore(ttl time.Duration) *GRPCSnapshotAgentStore {
	return &GRPCSnapshotAgentStore{
		clients:  make(map[string]agentpb.SnapshotAgentServiceClient),
		cache:    make(map[string]*cacheEntry),
		cacheTTL: ttl,
	}
}

func (s *GRPCSnapshotAgentStore) getClient(address string) (agentpb.SnapshotAgentServiceClient, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if client, ok := s.clients[address]; ok {
		return client, nil
	}

	// Dial the agent using NewClient (preferred in newer gRPC versions)
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to dial agent at %s: %w", address, err)
	}

	client := agentpb.NewSnapshotAgentServiceClient(conn)
	s.clients[address] = client
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

func (s *GRPCSnapshotAgentStore) resolveNodeAddress(nodeName string) string {
	// TODO: Implement actual node name to address translation once we know the port and DNS choices.
	// For now, assume they are the same for unit tests.
	return nodeName
}
