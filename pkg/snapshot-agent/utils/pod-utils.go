package utils

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

const (
	// JobIDLabel is the label used to identify pods by their job ID.
	JobIDLabel = "timeslice.io/job-id"
)

var (
	// For mocking in tests.
	GetK8sClient = func() (kubernetes.Interface, error) {
		config, err := rest.InClusterConfig()
		if err != nil {
			return nil, err
		}
		return kubernetes.NewForConfig(config)
	}

	NvmlInit                   = nvml.Init
	NvmlShutdown               = nvml.Shutdown
	NvmlDeviceGetCount         = nvml.DeviceGetCount
	NvmlDeviceGetHandleByIndex = func(index int) (DeviceInterface, nvml.Return) {
		return nvml.DeviceGetHandleByIndex(index)
	}

	IsPIDInPodCgroupFunc = isPIDInPodCgroup
)

type DeviceInterface interface {
	GetComputeRunningProcesses() ([]nvml.ProcessInfo, nvml.Return)
	GetGraphicsRunningProcesses() ([]nvml.ProcessInfo, nvml.Return)
}

// GetLocalPods returns a list of pods running on the same node as the current pod.
// It uses the NODE_NAME environment variable (populated via the Downward API)
// to filter pods by node and the snapshot-agent label to filter by managed pods.
func GetLocalPods(ctx context.Context, jobID string) ([]corev1.Pod, error) {
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		return nil, fmt.Errorf("NODE_NAME environment variable not set")
	}

	// Create the clientset
	clientset, err := GetK8sClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	// List pods on the current node that have the snapshot-agent label
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
		LabelSelector: fmt.Sprintf("%s=%s", JobIDLabel, jobID),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods on node %s: %w", nodeName, err)
	}

	return podList.Items, nil
}

// GetPodPIDs returns the host-namespace PIDs of all CUDA-context-holding processes belonging to the specified pod.
func GetPodPIDs(ctx context.Context, podName, namespace string) ([]int, error) {
	logger := klog.FromContext(ctx)
	// 1. Get the pod UID
	podUID, err := getPodUID(ctx, podName, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod UID: %w", err)
	}

	// 2. Initialize NVML
	logger.Info("Initializing NVML")
	ret := NvmlInit()
	if ret != nvml.SUCCESS {
		logger.Error(fmt.Errorf("%s", nvml.ErrorString(ret)), "Failed to initialize NVML")
		return nil, fmt.Errorf("failed to initialize NVML: %v", nvml.ErrorString(ret))
	}
	logger.Info("NVML initialized successfully")
	defer func() {
		if ret := NvmlShutdown(); ret != nvml.SUCCESS {
			logger.Error(fmt.Errorf("%s", nvml.ErrorString(ret)), "Failed to shutdown NVML")
		}
	}()

	// 3. Discover PIDs
	count, ret := NvmlDeviceGetCount()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to get device count: %v", nvml.ErrorString(ret))
	}

	var pids []int
	seenPIDs := make(map[int]bool)

	for i := 0; i < count; i++ {
		device, ret := NvmlDeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			continue
		}

		// Query compute processes
		procs, ret := device.GetComputeRunningProcesses()
		if ret != nvml.SUCCESS {
			continue
		}

		for _, proc := range procs {
			pid := int(proc.Pid)
			if seenPIDs[pid] {
				continue
			}

			inCgroup, err := IsPIDInPodCgroupFunc(pid, podUID)
			if err != nil {
				continue
			}

			if inCgroup {
				pids = append(pids, pid)
				seenPIDs[pid] = true
			}
		}

		// Query graphics processes as well, as some ML workloads might use them
		graphicsProcs, ret := device.GetGraphicsRunningProcesses()
		if ret == nvml.SUCCESS {
			for _, proc := range graphicsProcs {
				pid := int(proc.Pid)
				if seenPIDs[pid] {
					continue
				}

				inCgroup, err := IsPIDInPodCgroupFunc(pid, podUID)
				if err != nil {
					continue
				}

				if inCgroup {
					pids = append(pids, pid)
					seenPIDs[pid] = true
				}
			}
		}
	}

	return pids, nil
}

func getPodUID(ctx context.Context, podName, namespace string) (string, error) {
	clientset, err := GetK8sClient()
	if err != nil {
		return "", err
	}
	pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return string(pod.UID), nil
}

func isPIDInPodCgroup(pid int, podUID string) (bool, error) {
	return IsPIDInPodCgroupInternal(fmt.Sprintf("/proc/%d/cgroup", pid), podUID)
}

func IsPIDInPodCgroupInternal(cgroupPath, podUID string) (bool, error) {
	f, err := os.Open(cgroupPath)
	if err != nil {
		return false, err
	}
	defer f.Close()

	// Pod UID in cgroup paths can have dashes replaced by underscores in some K8s versions/runtimes
	podUIDUnderscores := strings.ReplaceAll(podUID, "-", "_")

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, podUID) || strings.Contains(line, podUIDUnderscores) {
			return true, nil
		}
	}
	return false, scanner.Err()
}
