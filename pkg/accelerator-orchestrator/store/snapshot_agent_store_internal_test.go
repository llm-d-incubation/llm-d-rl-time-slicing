// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package store

import (
	"testing"
)

func TestResolveNodeAddress(t *testing.T) {
	tests := []struct {
		name        string
		nodeName    string
		defaultPort int
		want        string
	}{
		{
			name:        "without port",
			nodeName:    "node-1",
			defaultPort: 9001,
			want:        "node-1:9001",
		},
		{
			name:        "with port",
			nodeName:    "node-1:12345",
			defaultPort: 9001,
			want:        "node-1:12345",
		},
		{
			name:        "IP without port",
			nodeName:    "127.0.0.1",
			defaultPort: 9001,
			want:        "127.0.0.1:9001",
		},
		{
			name:        "IP with port",
			nodeName:    "127.0.0.1:12345",
			defaultPort: 9001,
			want:        "127.0.0.1:12345",
		},
		{
			name:        "different default port",
			nodeName:    "node-2",
			defaultPort: 8080,
			want:        "node-2:8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewGRPCSnapshotAgentStore(0, tt.defaultPort)
			got := s.resolveNodeAddress(tt.nodeName)
			if got != tt.want {
				t.Errorf("resolveNodeAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}
