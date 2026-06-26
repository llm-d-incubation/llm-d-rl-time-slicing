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
	"k8s.io/apimachinery/pkg/util/intstr"
)

// PodFactory manages pod templates keyed by a template name/key.
type PodFactory struct {
	mu        sync.RWMutex
	templates map[string]*corev1.Pod
}

// NewPodFactory creates a new PodFactory and registers the default pause pod.
func NewPodFactory() *PodFactory {
	factory := &PodFactory{
		templates: make(map[string]*corev1.Pod),
	}
	factory.Register("default", &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "dummy",
					Image: "registry.k8s.io/pause:3.9",
				},
			},
		},
	})
	factory.Register("vllm", &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "vllm-container",
					Image: "vllm/vllm-openai:latest",
					Command: []string{
						"python3",
						"-m",
						"vllm.entrypoints.openai.api_server",
					},
					Args: []string{
						"--model", "Qwen/Qwen2-0.5B-Instruct",
						"--port", "8000",
					},
					Ports: []corev1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: 8000,
						},
					},
					ReadinessProbe: &corev1.Probe{
						ProbeHandler: corev1.ProbeHandler{
							HTTPGet: &corev1.HTTPGetAction{
								Path: "/health",
								Port: intstr.FromInt(8000),
							},
						},
						InitialDelaySeconds: 15,
						PeriodSeconds:       5,
						FailureThreshold:    60,
					},
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "dshm",
							MountPath: "/dev/shm",
						},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "dshm",
					VolumeSource: corev1.VolumeSource{
						EmptyDir: &corev1.EmptyDirVolumeSource{
							Medium: corev1.StorageMediumMemory,
						},
					},
				},
			},
		},
	})
	return factory
}

// Register registers a pod template for a specific key.
func (p *PodFactory) Register(key string, pod *corev1.Pod) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.templates[key] = pod
}

// GetPod returns a copy of the registered pod template for the key.
// If key is blank or not found, it falls back to the "default" template.
func (p *PodFactory) GetPod(key string) *corev1.Pod {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if key == "" {
		key = "default"
	}

	if template, ok := p.templates[key]; ok {
		return template.DeepCopy()
	}

	// Fallback to default template if the key is not found
	if template, ok := p.templates["default"]; ok {
		return template.DeepCopy()
	}

	// Ultimate fallback (should not be reached as "default" is registered in NewPodFactory)
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
