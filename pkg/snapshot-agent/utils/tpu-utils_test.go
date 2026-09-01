package utils_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	snapshotutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// writeTpuProc creates a fixture process under root: a libtpu control thread
// (if libtpuThread), a cgroup file bound to podUID, and optionally a dangling
// symlink fd pointing at a /dev/vfio group.
func writeTpuProc(t *testing.T, root string, pid int, libtpuThread bool, podUID string, vfio bool) {
	t.Helper()
	base := filepath.Join(root, fmt.Sprint(pid))
	taskDir := filepath.Join(base, "task", fmt.Sprint(pid+1))
	if err := os.MkdirAll(taskDir, 0o755); err != nil {
		t.Fatal(err)
	}
	comm := "python3\n"
	if libtpuThread {
		comm = "libtpu00030004\n"
	}
	if err := os.WriteFile(filepath.Join(taskDir, "comm"), []byte(comm), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "comm"), []byte(comm), 0o600); err != nil {
		t.Fatal(err)
	}
	cgroup := fmt.Sprintf("0::/kubepods/burstable/pod%s/cont\n", podUID)
	if err := os.WriteFile(filepath.Join(base, "cgroup"), []byte(cgroup), 0o600); err != nil {
		t.Fatal(err)
	}
	fdDir := filepath.Join(base, "fd")
	if err := os.MkdirAll(fdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/dev/null", filepath.Join(fdDir, "0")); err != nil {
		t.Fatal(err)
	}
	if vfio {
		if err := os.Symlink("/dev/vfio/0", filepath.Join(fdDir, "7")); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGetPodTpuPIDs(t *testing.T) {
	root := t.TempDir()
	restoreProc := snapshotutils.SetProcRootForTest(root)
	defer restoreProc()

	const podUID = "tpu-pod-uid"
	writeTpuProc(t, root, 100, true, podUID, true)      // running TPU proc of our pod
	writeTpuProc(t, root, 200, true, podUID, false)     // checkpointed: thread but no vfio fd
	writeTpuProc(t, root, 300, true, "other-uid", true) // other pod's TPU proc
	writeTpuProc(t, root, 400, false, podUID, false)    // plain process of our pod

	origGetK8sClient := snapshotutils.GetK8sClient
	defer func() { snapshotutils.GetK8sClient = origGetK8sClient }()
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

	pids, err := snapshotutils.GetPodTpuPIDs(context.Background(), "test-pod", "test-ns")
	if err != nil {
		t.Fatalf("GetPodTpuPIDs() error = %v", err)
	}
	if want := []int{100}; !reflect.DeepEqual(pids, want) {
		t.Errorf("GetPodTpuPIDs() = %v, want %v", pids, want)
	}
}

func TestHasTpuProcesses(t *testing.T) {
	t.Run("RunningProc", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, true, "uid", true)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || !got {
			t.Errorf("HasTpuProcesses() = %v, %v; want true, nil", got, err)
		}
	})

	t.Run("OnlyCheckpointedProc", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, true, "uid", false)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || got {
			t.Errorf("HasTpuProcesses() = %v, %v; want false, nil (checkpointed proc holds no vfio fd)", got, err)
		}
	})

	t.Run("NoTpuProcs", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, false, "uid", false)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || got {
			t.Errorf("HasTpuProcesses() = %v, %v; want false, nil", got, err)
		}
	})
}

func TestVfioGroupHolders(t *testing.T) {
	root := t.TempDir()
	defer snapshotutils.SetProcRootForTest(root)()
	writeTpuProc(t, root, 100, true, "uid", true)  // holds /dev/vfio/0
	writeTpuProc(t, root, 200, true, "uid", false) // holds nothing

	holders := snapshotutils.VfioGroupHolders()
	want := map[string][]string{"/dev/vfio/0": {"libtpu00030004"}}
	if len(holders) != 1 || len(holders["/dev/vfio/0"]) != 1 {
		t.Fatalf("VfioGroupHolders() = %v, want one holder of /dev/vfio/0 (%v)", holders, want)
	}
	if got := holders["/dev/vfio/0"][0]; got != "100(libtpu00030004)" {
		t.Errorf("VfioGroupHolders() holder = %q, want %q", got, "100(libtpu00030004)")
	}
}

// writeVfioGroups creates numeric group fixture files under a vfio root.
func writeVfioGroups(t *testing.T, root string, groups ...string) {
	t.Helper()
	for _, g := range groups {
		if err := os.WriteFile(filepath.Join(root, g), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestWaitVfioFree(t *testing.T) {
	t.Run("NoGroups", func(t *testing.T) {
		defer snapshotutils.SetVfioRootForTest(t.TempDir())()
		if err := snapshotutils.WaitVfioFree(context.Background(), time.Second); err != nil {
			t.Errorf("WaitVfioFree() with no groups = %v, want nil (skip)", err)
		}
	})

	t.Run("AllFree", func(t *testing.T) {
		root := t.TempDir()
		writeVfioGroups(t, root, "0", "1")
		defer snapshotutils.SetVfioRootForTest(root)()
		defer snapshotutils.SetOpenVfioGroupForTest(func(string) error { return nil })()

		if err := snapshotutils.WaitVfioFree(context.Background(), time.Second); err != nil {
			t.Errorf("WaitVfioFree() = %v, want nil", err)
		}
	})

	t.Run("BusyThenFree", func(t *testing.T) {
		root := t.TempDir()
		writeVfioGroups(t, root, "0")
		defer snapshotutils.SetVfioRootForTest(root)()
		var calls atomic.Int32
		defer snapshotutils.SetOpenVfioGroupForTest(func(string) error {
			if calls.Add(1) <= 2 {
				return fmt.Errorf("EBUSY")
			}
			return nil
		})()

		if err := snapshotutils.WaitVfioFree(context.Background(), 10*time.Second); err != nil {
			t.Errorf("WaitVfioFree() = %v, want nil once the group frees", err)
		}
		if calls.Load() < 3 {
			t.Errorf("openVfioGroup called %d times, want >= 3 (poll until free)", calls.Load())
		}
	})

	t.Run("Timeout", func(t *testing.T) {
		root := t.TempDir()
		writeVfioGroups(t, root, "0")
		defer snapshotutils.SetVfioRootForTest(root)()
		defer snapshotutils.SetOpenVfioGroupForTest(func(string) error { return fmt.Errorf("EBUSY") })()

		if err := snapshotutils.WaitVfioFree(context.Background(), 300*time.Millisecond); err == nil {
			t.Error("WaitVfioFree() = nil, want error when groups stay busy past the timeout")
		}
	})

	t.Run("ContextCanceled", func(t *testing.T) {
		root := t.TempDir()
		writeVfioGroups(t, root, "0")
		defer snapshotutils.SetVfioRootForTest(root)()
		defer snapshotutils.SetOpenVfioGroupForTest(func(string) error { return fmt.Errorf("EBUSY") })()

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		if err := snapshotutils.WaitVfioFree(ctx, time.Minute); err == nil {
			t.Error("WaitVfioFree() = nil, want error when the context is canceled")
		}
	})
}

func TestClearTpuLockfiles(t *testing.T) {
	root := t.TempDir()
	defer snapshotutils.SetProcRootForTest(root)()

	lockDir := filepath.Join(root, "100", "root", "tmp")
	if err := os.MkdirAll(lockDir, 0o755); err != nil {
		t.Fatal(err)
	}
	lockfile := filepath.Join(lockDir, "libtpu_lockfile")
	if err := os.WriteFile(lockfile, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	// PID 200 has no lockfile; must not error.
	snapshotutils.ClearTpuLockfiles(context.Background(), []string{"100", "200"})

	if _, err := os.Stat(lockfile); !os.IsNotExist(err) {
		t.Errorf("lockfile still present after ClearTpuLockfiles: stat err = %v", err)
	}
}
