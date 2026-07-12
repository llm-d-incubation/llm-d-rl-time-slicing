package backends

import (
	"context"
	"time"
)

// ResolveSuspendMode exposes resolveSuspendMode for tests.
var ResolveSuspendMode = resolveSuspendMode

func (b *AppChannelBackend) SetCommandTimeout(d time.Duration) {
	b.commandTimeout = d
}

func (c *CudaCheckpoint) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	c.execCommand = f
}

func (c *CudaCheckpoint) SetNvmlClient(n nvmlClient) {
	c.nvml = n
}

func (c *CudaCheckpoint) SetLookPath(f func(string) (string, error)) {
	c.lookPath = f
}
