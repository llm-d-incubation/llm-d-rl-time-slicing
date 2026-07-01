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
	gracePeriodSec := int64(2)
	factory.Register("default", &corev1.Pod{
		Spec: corev1.PodSpec{
			TerminationGracePeriodSeconds: &gracePeriodSec,
			Containers: []corev1.Container{
				{
					Name:  "dummy",
					Image: "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
					Command: []string{
						"python3",
						"-c",
						"import torch, time, sys\n" +
							"try:\n" +
							"    if not torch.cuda.is_available():\n" +
							"        print('CUDA not available', file=sys.stderr, flush=True)\n" +
							"        sys.exit(1)\n" +
							"    x = torch.randn(1000, 1000, device='cuda')\n" +
							"    print('CUDA context created and tensor allocated', flush=True)\n" +
							"    while True:\n" +
							"        x = x + 0.0001\n" +
							"        time.sleep(1)\n" +
							"except Exception as e:\n" +
							"    print('Error:', e, file=sys.stderr, flush=True)\n" +
							"    sys.exit(1)",
					},
					ImagePullPolicy: corev1.PullIfNotPresent,
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

	// Fallback to default template if the key is not found.
	// "default" is guaranteed to be registered in NewPodFactory.
	return p.templates["default"].DeepCopy()
}
