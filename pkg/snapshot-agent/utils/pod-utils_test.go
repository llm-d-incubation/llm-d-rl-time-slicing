package utils_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	snapshotutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	origGetK8sClient := snapshotutils.GetK8sClient
	origNvmlInit := snapshotutils.NvmlInit
	origNvmlShutdown := snapshotutils.NvmlShutdown
	origNvmlDeviceGetCount := snapshotutils.NvmlDeviceGetCount
	origNvmlDeviceGetHandleByIndex := snapshotutils.NvmlDeviceGetHandleByIndex
	origIsPIDInPodCgroupFunc := snapshotutils.IsPIDInPodCgroupFunc

	defer func() {
		snapshotutils.GetK8sClient = origGetK8sClient
		snapshotutils.NvmlInit = origNvmlInit
		snapshotutils.NvmlShutdown = origNvmlShutdown
		snapshotutils.NvmlDeviceGetCount = origNvmlDeviceGetCount
		snapshotutils.NvmlDeviceGetHandleByIndex = origNvmlDeviceGetHandleByIndex
		snapshotutils.IsPIDInPodCgroupFunc = origIsPIDInPodCgroupFunc
	}()

	snapshotutils.NvmlShutdown = func() nvml.Return { return nvml.SUCCESS }

	tests := []struct {
		name         string
		podName      string
		namespace    string
		setupMocks   func()
		expectedPIDs []int
		expectError  bool
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(pod), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				snapshotutils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 1, nvml.SUCCESS }
				device := &mockDevice{
					computeProcs:  []nvml.ProcessInfo{{Pid: 100, UsedGpuMemory: 1024}, {Pid: 200, UsedGpuMemory: 1024}},
					graphicsProcs: []nvml.ProcessInfo{{Pid: 200, UsedGpuMemory: 1024}, {Pid: 300, UsedGpuMemory: 1024}},
				}
				snapshotutils.NvmlDeviceGetHandleByIndex = func(index int) (snapshotutils.DeviceInterface, nvml.Return) {
					return device, nvml.SUCCESS
				}
				snapshotutils.IsPIDInPodCgroupFunc = func(pid int, uid string) (bool, error) {
					return pid == 100 || pid == 300, nil
				}
			},
			expectedPIDs: []int{100, 300},
			expectError:  false,
		},
		{
			name:      "Ignore Zero VRAM Processes",
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(pod), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				snapshotutils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 1, nvml.SUCCESS }
				device := &mockDevice{
					computeProcs: []nvml.ProcessInfo{{Pid: 100, UsedGpuMemory: 1024}, {Pid: 200, UsedGpuMemory: 0}},
					// ^uint64(0) is NVML_VALUE_NOT_AVAILABLE (-1) as reported in the uint64 field.
					graphicsProcs: []nvml.ProcessInfo{{Pid: 300, UsedGpuMemory: ^uint64(0)}},
				}
				snapshotutils.NvmlDeviceGetHandleByIndex = func(index int) (snapshotutils.DeviceInterface, nvml.Return) {
					return device, nvml.SUCCESS
				}
				snapshotutils.IsPIDInPodCgroupFunc = func(pid int, uid string) (bool, error) {
					return true, nil
				}
			},
			expectedPIDs: []int{100},
			expectError:  false,
		},
		{
			name:      "GetPodUID Failure",
			podName:   "pod",
			namespace: "ns",
			setupMocks: func() {
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"},
					}), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.ERROR_UNKNOWN }
			},
			expectError: true,
		},
		{
			name:      "NVML Device Count Failure",
			podName:   "pod",
			namespace: "ns",
			setupMocks: func() {
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(&corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"},
					}), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				snapshotutils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 0, nvml.ERROR_UNKNOWN }
			},
			expectError: true,
		},
		{
			name:      "PIDs not in pod cgroup",
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(pod), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				snapshotutils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 1, nvml.SUCCESS }
				device := &mockDevice{
					computeProcs: []nvml.ProcessInfo{{Pid: 400, UsedGpuMemory: 1024}, {Pid: 500, UsedGpuMemory: 1024}},
				}
				snapshotutils.NvmlDeviceGetHandleByIndex = func(index int) (snapshotutils.DeviceInterface, nvml.Return) {
					return device, nvml.SUCCESS
				}
				snapshotutils.IsPIDInPodCgroupFunc = func(pid int, uid string) (bool, error) {
					return false, nil // None of the PIDs belong to this pod
				}
			},
			expectedPIDs: nil,
			expectError:  false,
		},
		{
			name:      "IsPIDInPodCgroupFunc error",
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
					return fake.NewSimpleClientset(pod), nil
				}
				snapshotutils.NvmlInit = func() nvml.Return { return nvml.SUCCESS }
				snapshotutils.NvmlDeviceGetCount = func() (int, nvml.Return) { return 1, nvml.SUCCESS }
				device := &mockDevice{
					computeProcs: []nvml.ProcessInfo{{Pid: 600, UsedGpuMemory: 1024}},
				}
				snapshotutils.NvmlDeviceGetHandleByIndex = func(index int) (snapshotutils.DeviceInterface, nvml.Return) {
					return device, nvml.SUCCESS
				}
				snapshotutils.IsPIDInPodCgroupFunc = func(pid int, uid string) (bool, error) {
					return false, fmt.Errorf("cgroup error")
				}
			},
			expectedPIDs: nil,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupMocks()
			pids, err := snapshotutils.GetPodPIDs(context.Background(), tt.podName, tt.namespace)
			if (err != nil) != tt.expectError {
				t.Errorf("GetPodPIDs() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError {
				sort.Ints(pids)
				sort.Ints(tt.expectedPIDs)
				if !reflect.DeepEqual(pids, tt.expectedPIDs) {
					t.Errorf("GetPodPIDs() = %v, expected %v", pids, tt.expectedPIDs)
				}
			}
		})
	}
}

func TestGetLocalPods(t *testing.T) {
	origGetK8sClient := snapshotutils.GetK8sClient
	defer func() { snapshotutils.GetK8sClient = origGetK8sClient }()

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
							snapshotutils.JobIDLabel: jobID,
						},
					},
					Spec: corev1.PodSpec{NodeName: nodeName},
				}
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
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
				snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
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
			pods, err := snapshotutils.GetLocalPods(context.Background(), jobID)
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

func TestGetJobBackendAnnotation(t *testing.T) {
	origGetK8sClient := snapshotutils.GetK8sClient
	defer func() { snapshotutils.GetK8sClient = origGetK8sClient }()

	nodeName := "test-node"
	jobID := "test-job"

	newPod := func(name string, annotations map[string]string) *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        name,
				Labels:      map[string]string{snapshotutils.JobIDLabel: jobID},
				Annotations: annotations,
			},
			Spec: corev1.PodSpec{NodeName: nodeName},
		}
	}

	tests := []struct {
		name        string
		nodeEnv     string
		pods        []*corev1.Pod
		want        *snapshotutils.JobBackendAnnotation
		expectError bool
	}{
		{
			name:    "Backend with config declared",
			nodeEnv: nodeName,
			pods: []*corev1.Pod{
				newPod("pod1", map[string]string{
					snapshotutils.BackendAnnotation:       "app_endpoint",
					snapshotutils.BackendConfigAnnotation: `{"endpoints": ["http://localhost:8000"]}`,
				}),
			},
			want: &snapshotutils.JobBackendAnnotation{
				Backend: "app_endpoint",
				Config:  `{"endpoints": ["http://localhost:8000"]}`,
			},
		},
		{
			name:    "Backend without config declared",
			nodeEnv: nodeName,
			pods:    []*corev1.Pod{newPod("pod1", map[string]string{snapshotutils.BackendAnnotation: "cuda"})},
			want:    &snapshotutils.JobBackendAnnotation{Backend: "cuda"},
		},
		{
			name:    "No declaration",
			nodeEnv: nodeName,
			pods:    []*corev1.Pod{newPod("pod1", nil)},
			want:    nil,
		},
		{
			name:    "Agreeing pods",
			nodeEnv: nodeName,
			pods: []*corev1.Pod{
				newPod("pod1", map[string]string{snapshotutils.BackendAnnotation: "cuda"}),
				newPod("pod2", map[string]string{snapshotutils.BackendAnnotation: "cuda"}),
				newPod("pod3", nil), // silent pods do not conflict
			},
			want: &snapshotutils.JobBackendAnnotation{Backend: "cuda"},
		},
		{
			name:    "Conflicting backends",
			nodeEnv: nodeName,
			pods: []*corev1.Pod{
				newPod("pod1", map[string]string{snapshotutils.BackendAnnotation: "cuda"}),
				newPod("pod2", map[string]string{snapshotutils.BackendAnnotation: "app_endpoint"}),
			},
			expectError: true,
		},
		{
			name:    "Conflicting configs",
			nodeEnv: nodeName,
			pods: []*corev1.Pod{
				newPod("pod1", map[string]string{
					snapshotutils.BackendAnnotation:       "app_endpoint",
					snapshotutils.BackendConfigAnnotation: `{"endpoints": ["http://localhost:8000"]}`,
				}),
				newPod("pod2", map[string]string{
					snapshotutils.BackendAnnotation:       "app_endpoint",
					snapshotutils.BackendConfigAnnotation: `{"endpoints": ["http://localhost:9000"]}`,
				}),
			},
			expectError: true,
		},
		{
			name:        "Missing NODE_NAME",
			nodeEnv:     "",
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

			objects := make([]runtime.Object, 0, len(tt.pods))
			for _, pod := range tt.pods {
				objects = append(objects, pod)
			}
			snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
				return fake.NewSimpleClientset(objects...), nil
			}

			got, err := snapshotutils.GetJobBackendAnnotation(context.Background(), jobID)
			if (err != nil) != tt.expectError {
				t.Errorf("GetJobBackendAnnotation() error = %v, expectError %v", err, tt.expectError)
				return
			}
			if !tt.expectError && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetJobBackendAnnotation() = %v, expected %v", got, tt.want)
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
				err := os.WriteFile(cgroupPath, []byte(tt.cgroupLog), 0o600)
				if err != nil {
					t.Fatalf("Failed to write cgroup file: %v", err)
				}
			}

			got, err := snapshotutils.IsPIDInPodCgroupInternal(cgroupPath, tt.podUID)
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
