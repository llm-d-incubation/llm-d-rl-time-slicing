//go:build integration

package integration

import (
	"os"
	"testing"
)

// TestK8s exercises all backends in k8s mode against the agent deployed by
// the OFFICIAL snapshot-agent Helm chart — the deployment path a user takes.
// run.sh installs the chart for this phase, pinning its DaemonSet to
// TEST_NODE, so every backend runs through the chart's wiring (privileged,
// hostPID, hostNetwork, driver mounts, bundled cuda-checkpoint): the watcher
// registers jobs from pod labels, CUDA PIDs are discovered from pods (no
// explicit target), and inference engine configs pass through to the backend.
func TestK8s(t *testing.T) {
	if os.Getenv("SA_CHART_DEPLOYED") == "" {
		t.Skip("requires the chart-deployed snapshot-agent (run.sh --phase k8s)")
	}

	h := NewChartHarness(t)

	h.WithEngine(t, VLLM, func(t *testing.T, e *Engine) {
		// All k8s-mode job IDs match the pod's timeslice.io/job-id label: the
		// watcher is the only source of job registration in k8s mode. The empty
		// CUDA config exercises PID discovery.
		t.Run("CUDAWatcherDiscoveredPIDs", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "vllm-k8s", cudaConfig())
			h.RestoreOK(t, "vllm-k8s", cudaConfig())
			after := h.Inference(t, e)
			if before != after {
				t.Errorf("inference changed after restore: before=%q after=%q", before, after)
			}
		})

		t.Run("VLLMSleepWake", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "vllm-k8s", appConfig("vllm", e.Endpoint(), ""))
			vramAsleep := h.VRAMMiB(t, e)
			t.Logf("VRAM after sleep: %d MiB", vramAsleep)
			h.RestoreOK(t, "vllm-k8s", appConfig("vllm", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramAsleep, before, after)
		})
	})

	h.WithEngine(t, SGLang, func(t *testing.T, e *Engine) {
		t.Run("SGLangReleaseResume", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "sglang-k8s", appConfig("sglang", e.Endpoint(), ""))
			vramReleased := h.VRAMMiB(t, e)
			t.Logf("VRAM after release: %d MiB", vramReleased)
			h.RestoreOK(t, "sglang-k8s", appConfig("sglang", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramReleased, before, after)
		})
	})
}
