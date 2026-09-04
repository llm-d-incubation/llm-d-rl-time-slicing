#ifndef GPU_CR_SRC_GPU_CR_CONFIG_H_
#define GPU_CR_SRC_GPU_CR_CONFIG_H_

// Runtime-sizable checkpoint buffers.
//
// Buffer sizes come from the environment, read once and cached; the
// compile-time macros (SHM_SIZE_GB build arg, 1GiB staging) are the
// DEFAULTS, not literals — an env-unset SHM8-built image keeps the same
// buffer footprint as today's SHM8 image (the fleet-default rule: rolling
// this code out must never change a deployment's footprint by itself).
//
//   GPU_CR_SHM_MB / GPU_CR_SHM_GB   dump buffer (MB wins if both set;
//                                   an empty value counts as unset)
//       0  = deferred: no dump-buffer mapping until a buffer-path op
//            first needs it; it then materializes at the 64MiB floor
//            (covers the shared_mem_fs header + IPC scratch blocks).
//            For -o-only deployments; pool rule: 2×staging + 64MiB +
//            Σ destination files.
//   GPU_CR_STAGING_MB               each of the two DMA staging buffers
//
// The coordinator maps worker buffers at its own SHM_SIZE, so the
// coordinator and its workers must run with identical GPU_CR_* env.
//
// No upper clamp: the hugepage pool is the real bound and mmap reports
// ENOMEM honestly. The only "too big" that is rejected at parse time is
// a size the platform cannot represent at all — beyond signed off_t
// (ftruncate's extent type), or a staging size whose x2 buffer total
// would wrap size_t; such values could never be honored on any pool.
// Floors: 64MiB dump (2MiB-aligned), 128MiB staging.
// Unparsable, out-of-range, or below-floor values warn and fall back to
// the default — never a silent clamp.
//
// This is a function-local-static singleton, NOT a second ELF
// constructor: cross-TU constructor order vs init() is unspecified, so
// init() warms the cache before registering any signal handler, and the
// handlers only ever consume cached values.

#include <cstddef>

namespace gpu_cr {

inline constexpr size_t kShmFloorBytes = 64UL << 20;
inline constexpr size_t kStagingFloorBytes = 128UL << 20;

struct BufConfig {
  size_t shm_size;      // dump-buffer size when (or if) materialized
  size_t staging_size;  // per staging buffer (two are allocated)
  bool shm_deferred;    // GPU_CR_SHM_*=0: create only on first use
};

namespace internal {

// strtoll semantics on the front of the string: leading whitespace and an
// optional '+' are tolerated (" 8" parses); anything after the digits is
// rejected ("8 " is not ok). Out-of-range values (ERANGE) are not ok.
long long ParseNonNegative(const char* value, bool* ok);

// Exposed for tests: a pure function of (defaults, environment), so each
// test case gets a fresh parse. Production reads the cached Config().
BufConfig Load(size_t shm_default, size_t staging_default);

}  // namespace internal

// Cached singleton; defaults come from the including translation unit's
// build configuration (common.h) so this header stays macro-order
// independent.
const BufConfig& Config();

}  // namespace gpu_cr

#endif  // GPU_CR_SRC_GPU_CR_CONFIG_H_
