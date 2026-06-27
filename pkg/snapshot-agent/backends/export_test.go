package backends

import "context"

func (c *CudaCheckpoint) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	c.execCommand = f
}

func (c *CudaCheckpoint) SetNvmlClient(n nvmlClient) {
	c.nvml = n
}

func (c *CudaCheckpoint) SetLookPath(f func(string) (string, error)) {
	c.lookPath = f
}

func (g *GpuCr) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	g.execCommand = f
}

func (g *GpuCr) SetLookPath(f func(string) (string, error)) {
	g.lookPath = f
}

