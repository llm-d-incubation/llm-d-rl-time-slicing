//go:build integration

package integration

import (
	"fmt"

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
		return enginePod(h, "vllm", "vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089",
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
		return enginePod(h, "sglang", "lmsysorg/sglang:v0.5.15@sha256:af911a303a12516adf23ab8bb89c8fdf161ec0ceafc278a436fa111f5c118988",
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
				Name:  "install-cuda-checkpoint",
				Image: "alpine:latest",
				// Pinned to an immutable commit rather than the mutable main ref.
				Command: []string{"sh", "-c", "apk add --no-cache wget && wget -qO /opt/bin/cuda-checkpoint https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/00d5cce84c628088d6caa203fc4af40c1538b6f7/bin/x86_64_Linux/cuda-checkpoint && chmod +x /opt/bin/cuda-checkpoint"},
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

// channelWorkloadPod runs vLLM through its Python API (no HTTP server) and
// registers with the agent over the workload channel. The client library and
// the workload script are mounted from a ConfigMap built by the harness; the
// readiness probe fires once the model is loaded and the workload registered.
func channelWorkloadPod(h *Harness, jobID string) *corev1.Pod {
	labels := map[string]string{
		"app":        channelPodName,
		"test-suite": "snapshot-agent-integration",
	}
	if h.Mode == "k8s" {
		labels["timeslice.io/job-id"] = jobID
		labels["timeslice.io/group"] = "test"
	}
	// The script must not run from /opt/src: the client package's types.py
	// would shadow the stdlib types module via the script-dir sys.path entry.
	startup := "pip install -q --upgrade grpcio protobuf && " +
		"mkdir -p /opt/pkg/timeslice/snapshot_agent && " +
		"cp /opt/src/*.py /opt/pkg/timeslice/snapshot_agent/ && " +
		": > /opt/pkg/timeslice/__init__.py && " +
		"cp /opt/src/channel_workload.py /opt/channel_workload.py && " +
		"PYTHONPATH=/opt/pkg exec python3 /opt/channel_workload.py"
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      channelPodName,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "snapshot-agent-test",
			RestartPolicy:      corev1.RestartPolicyNever,
			NodeName:           h.Node,
			Tolerations:        gpuTolerations(),
			Volumes: []corev1.Volume{{
				Name: "src",
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: channelConfigMapName},
					},
				},
			}},
			Containers: []corev1.Container{{
				Name:    channelContainer,
				Image:   "vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089",
				Command: []string{"sh", "-c", startup},
				Env: []corev1.EnvVar{
					{Name: "MODEL", Value: h.Model},
					{Name: "SNAPSHOT_AGENT_ADDR", Value: fmt.Sprintf("%s:%d", h.AgentIP, agentPort)},
					{Name: "TIME_SLICE_JOB_ID", Value: jobID},
					{Name: "TIME_SLICE_GROUP", Value: "test"},
				},
				VolumeMounts: []corev1.VolumeMount{{Name: "src", MountPath: "/opt/src"}},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						Exec: &corev1.ExecAction{Command: []string{"test", "-f", "/workload-state/ready"}},
					},
					InitialDelaySeconds: 20,
					PeriodSeconds:       5,
					FailureThreshold:    60,
				},
				Resources: corev1.ResourceRequirements{
					Limits:   corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
					Requests: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
				},
			}},
		},
	}
}
