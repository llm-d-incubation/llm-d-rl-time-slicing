#ifndef COMMON_H
#define COMMON_H

#include <unistd.h>   // for many thing
#include <stdlib.h>   // for standard library
#include <stdio.h>    // for file dump
#include <time.h>     // for timing
#include <dlfcn.h>    // for loading real shared library
#include <stdint.h>   // for uint64_t defn
#include <stdbool.h>  // for true false
#include <elf.h>      // for ELF Header
#include <sys/wait.h> // for waiting subprocess
#include <sys/stat.h> // for directory
#include <sys/mman.h>  // for mmap
#include <pthread.h>  // for mutex lock
#include <signal.h>   // for signal handling
#include <cstring>    // for memset
#include <map>
#include <utility>
#include <atomic>
#include <mutex>

#include "gpu_cr_config.h"

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ROUND_UP_2MB(x) (((x) + (2 * 1024 * 1024 - 1)) & ~(2 * 1024 * 1024 - 1))

namespace gpu_cr {
// CUDA VMM mapping granularity: every hooked allocation is reserved/mapped in
// units of this size (see ROUND_UP_2MB uses in the cudaMalloc hook).
inline constexpr size_t kVmmGranuleSize = 2UL * 1024 * 1024;

// The driver rejects a cudaMemcpyAsync on a hooked VMM mapping
// when the copy crosses a 2MB granule boundary with an unrounded length.
// Clamps a device copy so it ends at the next granule boundary or at the
// region end: every issued copy is <=2MB and granule-aligned at its start
// or its end — the copy shapes validated at scale. Applies to the selective
// (unrounded) paths only; the full-checkpoint paths copy rounded extents
// and never hit this boundary.
constexpr size_t GranuleClampLen(uintptr_t dev_addr, size_t len) {
  size_t to_boundary = kVmmGranuleSize - (dev_addr & (kVmmGranuleSize - 1));
  return len < to_boundary ? len : to_boundary;
}
}  // namespace gpu_cr

// SHM_SIZE: Per-GPU checkpoint buffer on hugepages.
// Each GPU process allocates SHM_SIZE + STAGING_BUF_SIZE*STAGING_BUF_NUM.
// For TP=N, total hugepage needed = N * (SHM_SIZE + 2*staging) + overhead.
//
// These are runtime values now. The compile-time SHM_SIZE_GB
// (cmake -DSHM_SIZE_GB=40) sets the DEFAULT; GPU_CR_SHM_GB / GPU_CR_SHM_MB
// / GPU_CR_STAGING_MB env vars override it at library load. The macros
// below keep every historical use site source-compatible while reading
// the cached runtime config.
#ifndef SHM_SIZE_GB
#define SHM_SIZE_GB 25
#endif

namespace gpu_cr {
inline constexpr size_t kShmDefaultBytes =
    static_cast<size_t>(SHM_SIZE_GB) << 30;
inline constexpr size_t kStagingDefaultBytes = 1UL << 30;
}  // namespace gpu_cr

#define SHM_SIZE (gpu_cr::Config().shm_size)

#define MAX_FILE_NUM 4096
#define COPY_THRESHOLD (1UL << 29) // 0.5GB, when to copy from host_buf to shm
#define NUM_COPY_THREADS 4
#define CR_INIT_SIGNAL     SIGRTMAX
#define CR_CKPT_SIGNAL     SIGUSR1
#define CR_RESTORE_SIGNAL  SIGUSR2

// Multi-GPU: IPC teardown/rebuild signals (real-time signals)
// These replace the old NCCL suspend/resume signals.
#define CR_IPC_TEARDOWN_SIGNAL  (SIGRTMAX - 1)
#define CR_IPC_REBUILD_SIGNAL   (SIGRTMAX - 2)
#define CR_IPC_VALIDATE_SIGNAL  (SIGRTMAX - 3)

// Legacy aliases (for backward compatibility during transition)
#define CR_NCCL_SUSPEND_SIGNAL  CR_IPC_TEARDOWN_SIGNAL
#define CR_NCCL_RESUME_SIGNAL   CR_IPC_REBUILD_SIGNAL

// Maximum number of processes in multi-GPU checkpoint
#define MAX_MULTI_GPU_PROCS 32

#define STAGING_BUF_SIZE (gpu_cr::Config().staging_size) // default 1GB, env-overridable
#define STAGING_BUF_NUM 2

typedef void (*sighandler_t)(int);
typedef sighandler_t (*signal_func_t)(int, sighandler_t);

// Global memory tracking map: ptr -> size
extern std::map<void*, size_t> allocated_memory;

// Global memory type tracking: ptr -> type (0=runtime Malloc, 1=VMM)
extern std::map<void*, int> allocated_memory_type;

extern std::mutex gpu_mem_mutex;

// Helper function declarations
void memcpy_multi(void* dest, void* src, size_t size);

struct shared_mem_file {
    void* ptr;
    uint64_t start_offset;
    uint64_t size;
};

struct shared_mem_fs {
    uint64_t file_num;
    uint64_t current_offset;
    struct shared_mem_file files[MAX_FILE_NUM];
};

namespace gpu_cr {
// Exclusive bound on regions per selective request: a request must carry
// FEWER than this many regions. Each region becomes one shared_mem_fs
// extent-table entry, so this equals MAX_FILE_NUM (the pristine GPU-CR
// table size); at 16 bytes/region it also keeps the request at 64KiB in
// the control mapping. Observed swaps use ~1000 regions, so the 4095
// usable slots leave ~4x headroom. The dump writer refuses to fill the
// last extent-table slot (file_num < MAX_FILE_NUM, matching
// DumpHeaderPlausible), so a request gated at num_regions <
// kMaxSelectiveRegions can never fail for capacity after the
// device-to-host copy work has already been done.
inline constexpr uint32_t kMaxSelectiveRegions = 4096;
}  // namespace gpu_cr

struct SelectiveCrRegion {
    void* ptr;
    uint64_t size;
};

// Selective checkpoint/restore request: the region list names the
// allocations to save/restore, and dest_path routes the dump to a
// caller-named file instead of the per-PID buffer. Fields are only ever
// APPENDED so the region-list prefix keeps its exact offsets across
// separately-built cr_client and .so binaries sharing the mapping.
namespace gpu_cr {
// Destination paths are agent store files (<store>/<group>/ckpt-<id>.data,
// well under 100 chars); 256 gives ample headroom while keeping the
// request struct small. Not PATH_MAX: the client rejects longer paths
// before signaling, so the cap fails crisply rather than by truncation.
inline constexpr size_t kSelectiveCrMaxPath = 256;
}  // namespace gpu_cr

struct SelectiveCrRequest {
    uint32_t num_regions;
    SelectiveCrRegion regions[gpu_cr::kMaxSelectiveRegions];
    /* dest_path: empty = per-PID buffer; non-empty = destination-file
     * routing. Must be NUL-terminated: the client rejects paths
     * >= kSelectiveCrMaxPath-1 chars before writing, and the .so refuses
     * a request with dest_path[kSelectiveCrMaxPath-1] != '\0' (-EINVAL). */
    char     dest_path[gpu_cr::kSelectiveCrMaxPath];
};

namespace gpu_cr {
// The .so stores kSelectiveReady in signal_controls.selective_ready at
// init_CR and re-asserts it at every FINISH (the consume-once zeroing of
// the request deliberately leaves this word alone). Zero means no .so has
// published yet — the mapping zero-initializes — so a client that races
// init_CR, or one attached to a mismatched build, fails fast instead of
// signaling a library whose handlers are not armed.
inline constexpr uint32_t kSelectiveReady = 1;

// op_status encoding: 0 = no result reported (fresh mapping, or the op
// never reached FINISH), kOpStatusOk = success, negative = -errno.
// FinishOpControls is the only writer: op sites record positive errno
// values and FINISH negates, so 0 can never be mistaken for success.
inline constexpr int32_t kOpStatusOk = 1;

// Trailing commit marker for destination-file dumps: written at
// DumpCommitOffset(fs->current_offset) (see dump_format.h) only after
// the last extent has landed, so a torn dump is detectable. Restores from a
// destination file refuse dumps whose marker is absent or has the wrong
// magic; generation is recorded per dump and reserved for future
// staleness checks (not yet compared).
inline constexpr uint64_t kDumpCommitMagic = 0x31524347u; /* "GCR1" */
}  // namespace gpu_cr

struct DumpCommit {
    uint64_t magic;
    uint64_t generation;
};

struct signal_controls {
    uint32_t signal;
    SelectiveCrRequest selective_req;
    uint32_t selective_ready; /* 0 until the .so publishes kSelectiveReady */
    int32_t  op_status;       /* 0 = not reported, kOpStatusOk, or -errno */
};

namespace gpu_cr {
// Post-op bookkeeping: publish the op result, then consume the request's
// dest_path so a stale path can never redirect a later op — zeroing the
// path IS the consumption, since a non-empty dest_path is what selects
// destination-file routing. The region-list prefix is left alone: the
// client rewrites it under flock for every op. Clients gate
// cuda-checkpoint --toggle on a positive op_status — never freeze a
// process whose state was not saved. selective_ready is re-asserted so a
// client that attaches after init_CR still sees it.
inline void FinishOpControls(signal_controls* c, int32_t rc) {
  c->op_status = rc == 0 ? kOpStatusOk : -rc;
  std::memset(c->selective_req.dest_path, 0, kSelectiveCrMaxPath);
  c->selective_ready = kSelectiveReady;
}
}  // namespace gpu_cr

#endif