//go:build integration

package integration

import (
	"os"
	"testing"
)

// TestTpu exercises the TPU backend end to end on a real TPU node: the
// standalone agent (plus the gVisor tpucheckpoint CLI) drives a batched
// checkpoint/restore of a multi-process libtpu workload through the Python
// client — one CLI invocation carrying all of the job's PIDs, per the
// backend's contract.
//
// Fixture requirements (run.sh --phase tpu wires these up; the test skips
// when they are missing):
//   - TEST_NODE: a TPU node (e.g. a single-host v5e); the workload requests
//     TPU_CHIPS chips (default 8) and runs one worker process per chip.
//   - TPU_WORKLOAD_IMAGE: an image with python3 + JAX for the node's TPUs.
//   - bin/tpucheckpoint: the gVisor tpucheckpoint CLI (run.sh fetches it
//     from TPU_CHECKPOINT_URI).
//   - a checkpointing-enabled libtpu: either baked into the image or
//     fetched from TPU_LIBTPU_URI; workloads run with
//     LIBTPU_CHECKPOINTING_ENABLED=true, which spawns the libtpu control
//     threads the CLI drives.
func TestTpu(t *testing.T) {
	image := os.Getenv("TPU_WORKLOAD_IMAGE")
	if image == "" {
		t.Skip("TPU_WORKLOAD_IMAGE not set (see run.sh --phase tpu)")
	}
	if _, err := os.Stat(standaloneBinDir + "/tpucheckpoint"); err != nil {
		t.Skip("bin/tpucheckpoint missing (run.sh fetches it from TPU_CHECKPOINT_URI)")
	}

	h := NewTpuHarness(t)

	h.WithTpuWorkload(t, image, func(t *testing.T, w *TpuWorkload) {
		t.Run("CheckpointRestore", func(t *testing.T) {
			// The workload must issue no TPU ops while checkpointed
			// (libtpu contract) — park every worker first, exactly like a
			// production driver quiesces before yielding its time slice.
			h.TpuQuiesce(t, w)
			parked := h.TpuSteps(t, w)

			h.SnapshotOK(t, "tpu-standalone", tpuConfig(w.PIDs...))
			h.RestoreOK(t, "tpu-standalone", tpuConfig(w.PIDs...))

			// Workers computing past their parked step counters is the
			// proof the chips came back.
			h.TpuResume(t, w)
			h.WaitTpuAdvancing(t, w, parked)
		})
	})
}
