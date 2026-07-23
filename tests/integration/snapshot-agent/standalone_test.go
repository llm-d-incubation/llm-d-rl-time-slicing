//go:build integration

package integration

import "testing"

// TestStandalone exercises all backends in standalone mode: the caller
// provides the full BackendConfig (explicit PIDs for CUDA) and the agent's
// NVML check bootstraps jobs on first snapshot.
func TestStandalone(t *testing.T) {
	h := NewHarness(t, "standalone")

	h.WithEngine(t, VLLM, func(t *testing.T, e *Engine) {
		t.Run("CUDACheckpointRestore", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "s-cuda", cudaConfig(e.PID))
			h.RestoreOK(t, "s-cuda", cudaConfig(e.PID))
			after := h.Inference(t, e)
			if before != after {
				t.Errorf("inference changed after restore: before=%q after=%q", before, after)
			}
		})

		t.Run("VLLMSleepWake", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "s-vllm", appConfig("vllm", e.Endpoint(), ""))
			vramAsleep := h.VRAMMiB(t, e)
			t.Logf("VRAM after sleep: %d MiB", vramAsleep)
			h.RestoreOK(t, "s-vllm", appConfig("vllm", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramAsleep, before, after)
		})

		// Compound: app-level sleep, then CUDA checkpoint of the slept
		// process, restored in reverse order.
		t.Run("VLLMCompound", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "s-vllm-c", appConfig("vllm", e.Endpoint(), "offload"))
			vramAsleep := h.VRAMMiB(t, e)
			h.SnapshotOK(t, "s-cuda-c", cudaConfig(e.PID))
			h.RestoreOK(t, "s-cuda-c", cudaConfig(e.PID))
			h.RestoreOK(t, "s-vllm-c", appConfig("vllm", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramAsleep, before, after)
		})

		// DISCARD drops the weights: the engine cannot serve correct inference
		// afterwards without an application-level weight push, so this test
		// verifies the suspend/resume operations and VRAM only. It runs last
		// in the group — the engine's weights are garbage after it.
		t.Run("VLLMSuspendDiscard", func(t *testing.T) {
			h.SnapshotOK(t, "s-vllm-d", appConfig("vllm", e.Endpoint(), "discard"))
			vramSuspended := h.VRAMMiB(t, e)
			t.Logf("VRAM after discard suspend: %d MiB", vramSuspended)
			if vramSuspended >= vramFreedMiB {
				t.Errorf("VRAM not freed: %d MiB (want < %d)", vramSuspended, vramFreedMiB)
			}
			h.RestoreOK(t, "s-vllm-d", appConfig("vllm", e.Endpoint(), ""))
		})
	})

	h.WithEngine(t, SGLang, func(t *testing.T, e *Engine) {
		t.Run("SGLangReleaseResume", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "s-sgl", appConfig("sglang", e.Endpoint(), ""))
			vramReleased := h.VRAMMiB(t, e)
			t.Logf("VRAM after release: %d MiB", vramReleased)
			h.RestoreOK(t, "s-sgl", appConfig("sglang", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramReleased, before, after)
		})

		t.Run("SGLangCompound", func(t *testing.T) {
			before := h.Inference(t, e)
			h.SnapshotOK(t, "s-sgl-c", appConfig("sglang", e.Endpoint(), ""))
			vramReleased := h.VRAMMiB(t, e)
			h.SnapshotOK(t, "s-cuda-sc", cudaConfig(e.PID))
			h.RestoreOK(t, "s-cuda-sc", cudaConfig(e.PID))
			h.RestoreOK(t, "s-sgl-c", appConfig("sglang", e.Endpoint(), ""))
			after := h.Inference(t, e)
			RequireFreedAndCorrect(t, vramReleased, before, after)
		})
	})
}
