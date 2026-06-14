package utils_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

type mockDevice struct {
	computeProcs  []nvml.ProcessInfo
	graphicsProcs []nvml.ProcessInfo
}

func (m *mockDevice) GetComputeRunningProcesses() ([]nvml.ProcessInfo, nvml.Return) {
	return m.computeProcs, nvml.SUCCESS
}

func (m *mockDevice) GetGraphicsRunningProcesses() ([]nvml.ProcessInfo, nvml.Return) {
	return m.graphicsProcs, nvml.SUCCESS
}

func TestGetPodPIDs(t *testing.T) {
	origGetK8sClient := utils.GetK8sClient
	origNvmlInit := utils.NvmlInit
	origNvmlShutdown := utils.NvmlShutdown
	origNvmlDeviceGetCount := utils.NvmlDeviceGetCount
	origNvmlDeviceGetHandleByIndex := utils.NvmlDeviceGetHandleByIndex
	origIsPIDInPodCgroupFunc := utils.IsPIDInPodCgroupFunc

	defer func() {
		utils.GetK8sClient = origGetK8sClient
		utils.NvmlInit = origNvmlInit
		utils.NvmlShutdown = origNvmlShutdown
		utils.NvmlDeviceGetCount = origNvmlDeviceGetCount
		utils.NvmlDeviceGetHandleByIndex = origNvmlDeviceGetHandleByIndex
		utils.IsPIDInPodCgroupFunc = origIsPIDInPodCgroupFunc
	}()

	utils.NvmlShutdown = func() nvml.Return { return nvml.SUCCESS }

	tests := []struct {
		name          string
		podName       string
		namespace     string
		setupMocks    func()
		expectedPIDs  []int
		expectError   bool
	}{
		{
			name:      "Success",
			podName:   "test-pod",
			namespace: "test-ns",
			setupMocks: func() {
				podUID := "test-uid"
				pod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "test-ns",
						UID:       types.UID(podUID),
					},
				}
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(pod), nil
				}
				utils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				utils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 1, nvml.SUCCESS }
				device := &mockDevice{
					computeProcs:  []nvml.ProcessInfo{{Pid: 100}, {Pid: 200}},
					graphicsProcs: []nvml.ProcessInfo{{Pid: 200}, {Pid: 300}},
				}
				utils.NvmlDeviceGetHandleByIndex = func(index int) (utils.DeviceInterface, nvml.Return) {
					return device, nvml.SUCCESS
				}
				utils.IsPIDInPodCgroupFunc = func(pid int, uid string) (bool, error) {
					return pid == 100 || pid == 300, nil
				}
			},
			expectedPIDs: []int{100, 300},
			expectError:  false,
		},
		{
			name:      "GetPodUID Failure",
			podName:   "pod",
			namespace: "ns",
			setupMocks: func() {
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return nil, fmt.Errorf("k8s error")
				}
			},
			expectError: true,
		},
		{
			name:      "NVML Init Failure",
			podName:   "pod",
			namespace: "ns",
			setupMocks: func() {
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"},
					}), nil
				}
				utils.NvmlInit = func() nvml.Return { return nvml.ERROR_UNKNOWN }
			},
			expectError: true,
		},
		{
			name:      "NVML Device Count Failure",
			podName:   "pod",
			namespace: "ns",
			setupMocks: func() {
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"},
					}), nil
				}
				utils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				utils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 0, nvml.ERROR_UNKNOWN }
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupMocks()
			pids, err := utils.GetPodPIDs(context.Background(), tt.podName, tt.namespace)
			if (err != nil) != tt.expectError {
				t.Errorf("GetPodPIDs() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && !reflect.DeepEqual(pids, tt.expectedPIDs) {
				t.Errorf("GetPodPIDs() = %v, expected %v", pids, tt.expectedPIDs)
			}
		})
	}
}

func TestGetLocalPods(t *testing.T) {
	origGetK8sClient := utils.GetK8sClient
	defer func() { utils.GetK8sClient = origGetK8sClient }()

	nodeName := "test-node"
	jobID := "test-job"

	tests := []struct {
		name        string
		nodeEnv     string
		setupMocks  func()
		expectError bool
		expectedLen int
	}{
		{
			name:    "Success",
			nodeEnv: nodeName,
			setupMocks: func() {
				pod1 := corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						Labels: map[string]string{
							utils.SnapshotAgentLabel: utils.SnapshotAgentValue,
							utils.JobIDLabel:         jobID,
						},
					},
					Spec: corev1.PodSpec{NodeName: nodeName},
				}
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(&pod1), nil
				}
			},
			expectError: false,
			expectedLen: 1,
		},
		{
			name:        "Missing NODE_NAME",
			nodeEnv:     "",
			setupMocks:  func() {},
			expectError: true,
		},
		{
			name:    "K8s Client Failure",
			nodeEnv: nodeName,
			setupMocks: func() {
				utils.GetK8sClient = func() (kubernetes.Interface, error) {
					return nil, fmt.Errorf("k8s error")
				}
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.nodeEnv != "" {
				os.Setenv("NODE_NAME", tt.nodeEnv)
				defer os.Setenv("NODE_NAME", "")
			} else {
				os.Unsetenv("NODE_NAME")
			}

			tt.setupMocks()
			pods, err := utils.GetLocalPods(context.Background(), jobID)
			if (err != nil) != tt.expectError {
				t.Errorf("GetLocalPods() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && len(pods) != tt.expectedLen {
				t.Errorf("GetLocalPods() len = %d, expected %d", len(pods), tt.expectedLen)
			}
		})
	}
}

func TestIsPIDInPodCgroupInternal(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cgroup-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	podUID := "1234-5678-90ab-cdef"
	podUIDUnderscores := "1234_5678_90ab_cdef"

	tests := []struct {
		name      string
		cgroupLog string
		podUID    string
		filePath  string
		want      bool
		wantError bool
	}{
		{
			name:      "Match with dashes",
			cgroupLog: "12:pids:/kubepods.slice/kubepods-pod" + podUID + ".slice/docker-123.scope\n",
			podUID:    podUID,
			want:      true,
		},
		{
			name:      "Match with underscores",
			cgroupLog: "12:pids:/kubepods.slice/kubepods-pod" + podUIDUnderscores + ".slice/docker-123.scope\n",
			podUID:    podUID,
			want:      true,
		},
		{
			name:      "No match",
			cgroupLog: "12:pids:/kubepods.slice/kubepods-pod-other-uid.slice/docker-123.scope\n",
			podUID:    podUID,
			want:      false,
		},
		{
			name:      "Empty cgroup",
			cgroupLog: "",
			podUID:    podUID,
			want:      false,
		},
		{
			name:      "File not found",
			filePath:  "/non/existent/path",
			podUID:    podUID,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cgroupPath := tt.filePath
			if cgroupPath == "" {
				cgroupPath = filepath.Join(tempDir, "cgroup-"+tt.name)
				err := os.WriteFile(cgroupPath, []byte(tt.cgroupLog), 0644)
				if err != nil {
					t.Fatalf("Failed to write cgroup file: %v", err)
				}
			}

			got, err := utils.IsPIDInPodCgroupInternal(cgroupPath, tt.podUID)
			if (err != nil) != tt.wantError {
				t.Errorf("IsPIDInPodCgroupInternal() error = %v, wantError %v", err, tt.wantError)
				return
			}
			if got != tt.want {
				t.Errorf("IsPIDInPodCgroupInternal() = %v, want %v", got, tt.want)
			}
		})
	}
}
