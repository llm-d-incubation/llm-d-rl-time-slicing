//go:build integration

package integration

import (
	"strings"
	"testing"
)

// TestStandalone exercises all backends in standalone mode against an agent
// running from the `make standalone` artifacts (bin/snapshot-agent +
// bin/cuda-checkpoint) — the install path a standalone user takes. The caller
// provides the full BackendConfig (explicit PIDs for CUDA) and the agent's
// NVML check bootstraps jobs on first snapshot.
func TestStandalone(t *testing.T) {
	h := NewHarness(t)

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

		// Memory-regions slot swap: real agent + state machine + Python
		// client, with stub_cr_client.sh standing in for GPU-CR (see
		// memory_regions.go). Runs inside the engine group because
		// standalone-mode job bootstrap requires an occupied GPU; the
		// engine itself is not touched.
		t.Run("MemoryRegionsSlotSwap", func(t *testing.T) {
			h.WithMemoryRegionsFixture(t, func(t *testing.T, f *MemoryRegionsFixture) {
				const jobID = "s-mr"
				contentA := strings.Repeat("A", 64)
				contentB := strings.Repeat("B", 64)

				// Snapshot state A into slot-a; restore returns the job
				// to RUNNING.
				h.WriteMRDevice(t, "A")
				h.SnapshotOK(t, jobID, memoryRegionsConfig(f.PID, "slot-a"))
				h.RestoreOK(t, jobID, memoryRegionsConfig(f.PID, "slot-a"))

				// Snapshot state B into slot-b.
				h.WriteMRDevice(t, "B")
				h.SnapshotOK(t, jobID, memoryRegionsConfig(f.PID, "slot-b"))

				// Live slot swap while RUNNING, verified bitwise.
				for _, step := range []struct{ slot, want string }{
					{"slot-a", contentA},
					{"slot-b", contentB},
					{"slot-a", contentA},
					{"slot-b", contentB},
				} {
					h.RestoreOK(t, jobID, memoryRegionsConfig(f.PID, step.slot))
					if got := h.ReadMRDevice(t); got != step.want {
						t.Fatalf("device bytes after restoring %s: got %q, want %q",
							step.slot, got[:8], step.want[:8])
					}
				}

				// Every call must carry the destination-path shape with
				// the slot's <pid>-<starttime> owner directory.
				h.AssertMRCallShapes(t, f.PID, []string{"slot-a", "slot-b"})
			})
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

	// The channel workload embeds vLLM through the Python API (no HTTP
	// server) and registers with the agent via the client library; snapshot
	// requests address it by job ID alone.
	h.WithChannelWorkload(t, func(t *testing.T, w *ChannelWorkload) {
		t.Run("VLLMChannelSleepWake", func(t *testing.T) {
			before := h.TriggerGenerate(t, w)
			h.SnapshotOK(t, w.JobID, channelConfig(""))
			vramAsleep := h.WorkloadVRAMMiB(t, w)
			t.Logf("VRAM after channel snapshot: %d MiB", vramAsleep)
			h.RestoreOK(t, w.JobID, channelConfig(""))
			after := h.TriggerGenerate(t, w)
			RequireFreedAndCorrect(t, vramAsleep, before, after)
		})

		// Compound: channel-level suspend, then CUDA checkpoint of the
		// suspended process, restored in reverse order.
		t.Run("VLLMChannelCompound", func(t *testing.T) {
			before := h.TriggerGenerate(t, w)
			h.SnapshotOK(t, w.JobID, channelConfig(""))
			h.SnapshotOK(t, "chan-cuda", cudaConfig(w.PID))
			h.RestoreOK(t, "chan-cuda", cudaConfig(w.PID))
			h.RestoreOK(t, w.JobID, channelConfig(""))
			after := h.TriggerGenerate(t, w)
			if before != after {
				t.Errorf("generation changed after restore: before=%q after=%q", before, after)
			}
		})
	})
}
