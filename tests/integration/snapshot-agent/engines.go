//go:build integration

package integration

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EngineSpec describes how to deploy an inference engine for testing.
type EngineSpec struct {
	Name       string // pod name is <Name>-test
	Port       int
	PIDPattern string // pattern to locate the engine process in /proc
	BuildPod   func(h *Harness) *corev1.Pod
}

// VLLM runs vLLM with sleep mode enabled.
var VLLM = EngineSpec{
	Name:       "vllm",
	Port:       8000,
	PIDPattern: "vllm.entrypoints",
	BuildPod: func(h *Harness) *corev1.Pod {
		return enginePod(h, "vllm", "vllm/vllm-openai:latest",
			[]string{"python3", "-m", "vllm.entrypoints.openai.api_server"},
			[]string{
				"--model=" + h.Model,
				"--enable-sleep-mode",
				"--host=0.0.0.0",
				"--port=8000",
				"--gpu-memory-utilization=0.5",
			},
			[]corev1.EnvVar{{Name: "VLLM_SERVER_DEV_MODE", Value: "1"}},
		)
	},
}

// SGLang runs SGLang with the memory saver and CPU weight backup enabled.
var SGLang = EngineSpec{
	Name:       "sglang",
	Port:       30000,
	PIDPattern: "sglang.launch_server",
	BuildPod: func(h *Harness) *corev1.Pod {
		return enginePod(h, "sglang", "lmsysorg/sglang:latest",
			[]string{"python3", "-m", "sglang.launch_server"},
			[]string{
				"--model-path=" + h.Model,
				"--enable-memory-saver",
				"--enable-weights-cpu-backup",
				"--host=0.0.0.0",
				"--port=30000",
				"--mem-fraction-static=0.5",
			},
			nil,
		)
	},
}

func enginePod(h *Harness, name, image string, command, args []string, env []corev1.EnvVar) *corev1.Pod {
	labels := map[string]string{
		"app":        name + "-test",
		"test-suite": "snapshot-agent-integration",
	}
	// In k8s mode the watcher discovers jobs from these labels.
	if h.Mode == "k8s" {
		labels["timeslice.io/job-id"] = name + "-k8s"
		labels["timeslice.io/group"] = "test"
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name + "-test",
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "snapshot-agent-test",
			RestartPolicy:      corev1.RestartPolicyNever,
			NodeName:           h.Node,
			Tolerations:        gpuTolerations(),
			Containers: []corev1.Container{{
				Name:    name,
				Image:   image,
				Command: command,
				Args:    args,
				Env:     env,
				Resources: corev1.ResourceRequirements{
					Limits:   corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
					Requests: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
				},
			}},
		},
	}
}

func agentPod(image, node, mode string) *corev1.Pod {
	privileged := true
	hostPathDir := corev1.HostPathDirectory
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      agentPodName,
			Namespace: namespace,
			Labels: map[string]string{
				"app":        agentPodName,
				"test-suite": "snapshot-agent-integration",
			},
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "snapshot-agent-test",
			RestartPolicy:      corev1.RestartPolicyNever,
			NodeName:           node,
			HostPID:            true,
			Tolerations:        gpuTolerations(),
			InitContainers: []corev1.Container{{
				Name:    "install-cuda-checkpoint",
				Image:   "alpine:latest",
				Command: []string{"sh", "-c", "apk add --no-cache wget && wget -qO /opt/bin/cuda-checkpoint https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint && chmod +x /opt/bin/cuda-checkpoint"},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "bin-dir", MountPath: "/opt/bin"},
				},
			}},
			Containers: []corev1.Container{{
				Name:  "snapshot-agent",
				Image: image,
				Args:  []string{"--port=9001", "--deployment-mode=" + mode},
				Env: []corev1.EnvVar{
					{Name: "PATH", Value: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/bin"},
					{Name: "NODE_NAME", ValueFrom: &corev1.EnvVarSource{
						FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"},
					}},
					{Name: "NVIDIA_VISIBLE_DEVICES", Value: "all"},
					{Name: "NVIDIA_DRIVER_CAPABILITIES", Value: "compute,utility"},
					{Name: "LD_LIBRARY_PATH", Value: "/usr/local/nvidia/lib64"},
				},
				Ports:           []corev1.ContainerPort{{ContainerPort: agentPort}},
				SecurityContext: &corev1.SecurityContext{Privileged: &privileged},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "bin-dir", MountPath: "/opt/bin"},
					{Name: "nvidia-driver", MountPath: "/usr/local/nvidia", ReadOnly: true},
				},
			}},
			Volumes: []corev1.Volume{
				{Name: "bin-dir", VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{},
				}},
				{Name: "nvidia-driver", VolumeSource: corev1.VolumeSource{
					HostPath: &corev1.HostPathVolumeSource{
						Path: "/home/kubernetes/bin/nvidia",
						Type: &hostPathDir,
					},
				}},
			},
		},
	}
}

func gpuTolerations() []corev1.Toleration {
	return []corev1.Toleration{{
		Key:      "nvidia.com/gpu",
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	}}
}
