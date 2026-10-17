// This file is the HERMETIC tier of the memory-regions coverage: a real
// gRPC server with the real backend exec'ing stub_cr_client.sh, driven end
// to end by the Python client library — in process, no cluster, no GPU.
// This is the test that guards Go<->Python codegen drift (it would have
// caught the prototype's field-4 wire collision). It has no build tag, so
// `make test` compiles it on every PR; it runs when CROSS_LANG_TEST=1 (it
// needs python3 with grpcio+protobuf on PATH and write access to
// /usr/local/bin, where the backend's pinned cr_client path lives — both
// hold in the containerized test step).
//
// The in-cluster tier lives in standalone_test.go (MemoryRegionsSlotSwap,
// fixtures in memory_regions.go): the same stub and slot-swap semantics
// through a deployed standalone agent and agentctl.py, following the same
// pattern as the other backends' suites.
package integration_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/features"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

// crClientPath mirrors the backend's pinned cr_client location: the test
// installs the stub there because the backend does not consult PATH.
const crClientPath = "/usr/local/bin/cr_client"

func mustNil(t *testing.T, err error, what string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))
}

// installStub places the stub at the backend's pinned cr_client path. It
// refuses to overwrite a real binary and skips when the path is not
// writable (i.e. when run outside the containerized test step).
func installStub(t *testing.T, root string) {
	t.Helper()
	if _, err := os.Stat(crClientPath); err == nil {
		t.Skipf("%s already exists; refusing to overwrite", crClientPath)
	}
	stubSrc, err := os.ReadFile(filepath.Join(root, "tests", "integration", "snapshot-agent", "stub_cr_client.sh"))
	mustNil(t, err, "read stub source")
	//nolint:gosec // the stub must be executable
	if err := os.WriteFile(crClientPath, stubSrc, 0o755); err != nil {
		t.Skipf("cannot install stub at %s (%v); run inside the containerized test step", crClientPath, err)
	}
	t.Cleanup(func() { os.Remove(crClientPath) })
}

func TestCrossLanguageMemoryRegions(t *testing.T) {
	if os.Getenv("CROSS_LANG_TEST") != "1" {
		t.Skip("set CROSS_LANG_TEST=1 (needs python3 with grpcio+protobuf and a writable /usr/local/bin)")
	}
	root := repoRoot(t)
	installStub(t, root)

	// Shared store: the backend derives the destination group store and the
	// control-file dir from EXPORT_FILE_PATH; the stub reads it from env.
	ctlDir := t.TempDir()
	t.Setenv("EXPORT_FILE_PATH", ctlDir)

	// A live stand-in workload: the backend re-reads the owner pid's /proc
	// starttime on every operation to name the slot's owner directory, so
	// the pid must belong to a real running process.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	workload := exec.CommandContext(ctx, "sleep", "180")
	mustNil(t, workload.Start(), "start stand-in workload")
	t.Cleanup(func() { workload.Process.Kill() }) //nolint:errcheck // best-effort cleanup; the process may already be gone
	pid := workload.Process.Pid
	pidStr := strconv.Itoa(pid)

	// The workload's preloader would write the pid map; simulate it.
	mustNil(t, os.WriteFile(filepath.Join(ctlDir, "pid_map_"+pidStr), []byte("42\n"), 0o600),
		"write pid map")

	// Standalone-mode auto-transition needs "GPU occupied"; no NVML here.
	origHasGPU := utils.HasGPUProcesses
	utils.HasGPUProcesses = func(_ context.Context) (bool, error) { return true, nil }
	t.Cleanup(func() { utils.HasGPUProcesses = origHasGPU })

	// Real server + real MemoryRegions backend (real exec) on localhost.
	backendsMap := map[backends.BackendType]backends.Backend{
		backends.BackendMemoryRegions: backends.NewMemoryRegions(),
		backends.BackendNoop:          backends.NewNoopBackend(),
	}
	lis, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	mustNil(t, err, "listen")
	s := grpc.NewServer()
	srv := server.NewServer(backendsMap, backends.BackendNoop, "standalone", backends.NewChannelRegistry(),
		features.Gates{features.MemoryRegionsBackend: true})
	pb.RegisterSnapshotAgentServiceServer(s, srv)
	grpc_health_v1.RegisterHealthServer(s, server.NewHealthServer(backendsMap, backends.BackendNoop))
	go func() {
		if err := s.Serve(lis); err != nil {
			return
		}
	}()
	t.Cleanup(s.GracefulStop)

	endpoint := lis.Addr().String()

	// Drive it with the real Python client.
	python := os.Getenv("CROSS_LANG_PYTHON")
	if python == "" {
		python = "python3"
	}
	driver := filepath.Join(root, "tests", "integration", "snapshot-agent", "memory_regions_driver.py")
	cmd := exec.CommandContext(ctx, python, driver)
	cmd.Env = append(os.Environ(),
		"AGENT_ENDPOINT="+endpoint,
		"CTL_DIR="+ctlDir,
		"WORKLOAD_PID="+pidStr,
		"PYTHONPATH="+filepath.Join(root, "pkg", "client", "python"),
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("python driver failed: %v\n%s", err, string(out))
	}
	if !strings.Contains(string(out), "CROSS-LANG-OK") {
		t.Fatalf("driver did not report CROSS-LANG-OK:\n%s", string(out))
	}

	// The stub must have been invoked with the region spec and a
	// destination inside the slot's owner dir: .../groups/<slot>/<pid>-<starttime>/<id>.
	log, err := os.ReadFile(filepath.Join(ctlDir, "cr_client_calls.log"))
	mustNil(t, err, "read stub call log")
	for _, want := range []string{
		fmt.Sprintf("-c -p %s -s 0x7f0000000000:64 -o %s", pidStr, filepath.Join(ctlDir, "groups", "slot-a", pidStr+"-")),
		fmt.Sprintf("-c -p %s -s 0x7f0000000000:64 -o %s", pidStr, filepath.Join(ctlDir, "groups", "slot-b", pidStr+"-")),
		fmt.Sprintf("-r -p %s -s 0x7f0000000000:64 -o %s", pidStr, filepath.Join(ctlDir, "groups", "slot-a", pidStr+"-")),
		fmt.Sprintf("-r -p %s -s 0x7f0000000000:64 -o %s", pidStr, filepath.Join(ctlDir, "groups", "slot-b", pidStr+"-")),
	} {
		if !strings.Contains(string(log), want) {
			t.Fatalf("stub call log missing %q:\n%s", want, string(log))
		}
	}
	fmt.Println("cross-language driver output:", strings.TrimSpace(string(out)))
}
