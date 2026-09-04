package backends

import (
	"context"
	"os"
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

func (d *DirectMemory) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	d.execCommand = f
}

func (d *DirectMemory) SetStatFunc(f func(string) (os.FileInfo, error)) {
	d.statFunc = f
}

// CrClientPath exposes the fixed cr_client install location for tests.
const CrClientPath = crClientPath

func (g *MemoryRegions) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	g.execCommand = f
}

func (g *MemoryRegions) SetStatFunc(f func(string) (os.FileInfo, error)) {
	g.statFunc = f
}

func (g *MemoryRegions) SetProcRoot(dir string) {
	g.procRoot = dir
}

// SetStarttimeFunc replaces the pid-starttime reader, so tests can use pids
// that have no procfs entry (and pin the starttime half of owner dirnames).
func (g *MemoryRegions) SetStarttimeFunc(f func(pid string) (int64, error)) {
	g.starttime = f
}
