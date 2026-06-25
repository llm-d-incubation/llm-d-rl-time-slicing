// Copyright 2026 The llm-d Authors.
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

package acceleratororchestrator

import (
	"sync"

	corev1 "k8s.io/api/core/v1"
)

// PodFactory manages pod templates for different groups.
type PodFactory struct {
	mu        sync.RWMutex
	templates map[string]*corev1.Pod
}

// NewPodFactory creates a new PodFactory.
func NewPodFactory() *PodFactory {
	return &PodFactory{
		templates: make(map[string]*corev1.Pod),
	}
}

// Register registers a pod template for a specific group.
func (p *PodFactory) Register(groupID string, pod *corev1.Pod) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.templates[groupID] = pod
}

// GetPod returns a copy of the registered pod template for the group,
// or a default vanilla pause pod if none is registered.
func (p *PodFactory) GetPod(groupID string) *corev1.Pod {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if template, ok := p.templates[groupID]; ok {
		return template.DeepCopy()
	}

	// Default vanilla pod (pause container)
	return &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "dummy",
					Image: "registry.k8s.io/pause:3.9",
				},
			},
		},
	}
}
