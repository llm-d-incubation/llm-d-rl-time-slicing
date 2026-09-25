//go:build integration

package integration

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

// Memory-regions test pieces (see the MemoryRegionsSlotSwap run in
// standalone_test.go, and memory_regions_test.go for the hermetic
// cross-language tier that runs on every PR).
//
// The backend shells out to GPU-CR's cr_client at a pinned path; real GPU-CR
// needs a preloaded workload and a hugetlbfs store, so this suite substitutes
// the ONE process boundary — cr_client — with stub_cr_client.sh and keeps
// everything else real: the chart-shaped standalone agent, the state machine,
// the backend's slot/store bookkeeping, and the Python client. "Device
// memory" is a file in the store the stub dumps/loads, so slot swaps are
// verified bitwise end to end.
//
// The stand-in workload must be a live process in the agent's PID namespace:
// the backend re-reads the owner pid's /proc starttime on every operation to
// name the slot's owner directory (<pid>-<starttime>), so a fabricated pid
// cannot work. The agent pod runs with hostPID, so a process started inside
// it is visible at the same pid the backend resolves.
const (
	// mrStoreDir is the agent's EXPORT_FILE_PATH (set in agentPod): the
	// backend derives the destination group store (<store>/groups) and the
	// control-file dir from it. It lives on the agent pod's emptyDir.
	mrStoreDir = "/opt/rlts/mr-store"
	// mrCrClientPath mirrors the backend's pinned cr_client location: the
	// fixture installs the stub there because the backend does not consult
	// PATH.
	mrCrClientPath = "/usr/local/bin/cr_client"
	// mrRegionSpec is the single region every call uses, in the client's
	// "pid" prefix-less part filled per fixture: address:size.
	mrAddress = "0x7f0000000000"
	mrSize    = 64
)

// memoryRegionsConfig builds the agentctl.py flags for a memory-regions
// config: one region on the fixture's workload pid, saved to / restored from
// the named snapshot slot.
func memoryRegionsConfig(pid int, slot string) BackendArgs {
	spec := fmt.Sprintf("%d:%s:%d", pid, mrAddress, mrSize)
	return BackendArgs{"--backend", "memory-regions", "--regions", spec, "--snapshot-name", slot}
}

// MemoryRegionsFixture is the in-agent-pod test rig: the stub cr_client, the
// store directory, and a live stand-in workload process.
type MemoryRegionsFixture struct {
	PID int
}

// WithMemoryRegionsFixture prepares the agent pod for the memory-regions
// test (stub cr_client at the pinned path, store dir, live stand-in
// workload + its pid_map), runs fn, and tears everything down.
func (h *Harness) WithMemoryRegionsFixture(t *testing.T, fn func(t *testing.T, f *MemoryRegionsFixture)) {
	t.Helper()

	// Refuse to overwrite a real cr_client: this fixture must never turn a
	// node with a real GPU-CR install into a stubbed one.
	if out, _ := h.ExecPod(agentPodName, "snapshot-agent", opTimeout,
		"sh", "-c", "test -e "+mrCrClientPath+" && echo exists || true"); strings.Contains(out, "exists") {
		t.Skipf("%s already exists in the agent pod; refusing to overwrite", mrCrClientPath)
	}

	stub, err := os.ReadFile("stub_cr_client.sh")
	if err != nil {
		t.Fatalf("reading stub_cr_client.sh: %v", err)
	}
	if _, err := h.ExecPodStdin(agentPodName, "snapshot-agent", bytes.NewReader(stub), opTimeout,
		"sh", "-c", "cat >"+mrCrClientPath+" && chmod 755 "+mrCrClientPath); err != nil {
		t.Fatalf("installing stub cr_client: %v", err)
	}
	t.Cleanup(func() {
		_, _ = h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "rm", "-f", mrCrClientPath)
	})

	if _, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "mkdir", "-p", mrStoreDir); err != nil {
		t.Fatalf("creating memory-regions store dir: %v", err)
	}
	t.Cleanup(func() {
		_, _ = h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "rm", "-rf", mrStoreDir)
	})

	// Live stand-in workload inside the agent pod (hostPID: its pid is the
	// node pid the backend resolves through /proc).
	out, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout,
		"sh", "-c", "sleep 1800 >/dev/null 2>&1 & echo $!")
	if err != nil {
		t.Fatalf("starting stand-in workload: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(out))
	if err != nil {
		t.Fatalf("parsing stand-in workload pid from %q: %v", out, err)
	}
	t.Cleanup(func() {
		_, _ = h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "sh", "-c",
			fmt.Sprintf("kill %d 2>/dev/null || true", pid))
	})

	// The workload's preloader would write the pid map; simulate it.
	if _, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "sh", "-c",
		fmt.Sprintf("echo 42 > %s/pid_map_%d", mrStoreDir, pid)); err != nil {
		t.Fatalf("writing pid map: %v", err)
	}

	fn(t, &MemoryRegionsFixture{PID: pid})
}

// WriteMRDevice sets the stub's "device memory" contents (a repeated byte,
// mrSize long) inside the agent pod.
func (h *Harness) WriteMRDevice(t *testing.T, fill string) {
	t.Helper()
	script := fmt.Sprintf(`i=0; : > %[1]s/device; while [ $i -lt %[2]d ]; do printf '%%s' '%[3]s' >> %[1]s/device; i=$((i+1)); done`,
		mrStoreDir, mrSize, fill)
	if _, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "sh", "-c", script); err != nil {
		t.Fatalf("writing device file: %v", err)
	}
}

// ReadMRDevice returns the stub's "device memory" contents from the agent pod.
func (h *Harness) ReadMRDevice(t *testing.T) string {
	t.Helper()
	out, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "cat", mrStoreDir+"/device")
	if err != nil {
		t.Fatalf("reading device file: %v", err)
	}
	return out
}

// AssertMRCallShapes verifies the stub was invoked with the destination-path
// shape the backend promises: -c/-r -p <pid> -s <spec> -o
// <store>/groups/<slot>/<pid>-<starttime>/<id>, for every slot listed.
func (h *Harness) AssertMRCallShapes(t *testing.T, pid int, slots []string) {
	t.Helper()
	log, err := h.ExecPod(agentPodName, "snapshot-agent", opTimeout, "cat", mrStoreDir+"/cr_client_calls.log")
	if err != nil {
		t.Fatalf("reading stub call log: %v", err)
	}
	for _, slot := range slots {
		for _, op := range []string{"-c", "-r"} {
			want := fmt.Sprintf("%s -p %d -s %s:%d -o %s/groups/%s/%d-",
				op, pid, mrAddress, mrSize, mrStoreDir, slot, pid)
			if !strings.Contains(log, want) {
				t.Errorf("stub call log missing %q:\n%s", want, log)
			}
		}
	}
}
