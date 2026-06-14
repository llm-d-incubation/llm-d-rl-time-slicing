package backends

import "context"

// Exported for testing
var (
	GetCudaCheckpointPath = (*CudaCheckpoint).getCudaCheckpointPath
	RunSudoCommand        = (*CudaCheckpoint).runSudoCommand
	CheckpointPIDs        = (*CudaCheckpoint).checkpointPIDs
	RestorePIDs           = (*CudaCheckpoint).restorePIDs
)

func (c *CudaCheckpoint) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	c.execCommand = f
}

func (c *CudaCheckpoint) SetNvmlClient(n nvmlClient) {
	c.nvml = n
}

func (c *CudaCheckpoint) SetLookPath(f func(string) (string, error)) {
	c.lookPath = f
}
