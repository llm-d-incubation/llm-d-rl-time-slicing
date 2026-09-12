#include <atomic>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <thread>
#include <set>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <unistd.h>

#include "common.h"
#include "cr_signal_guard.h"
#include "ctl_path.h"
#include "dump_format.h"
#include "comm/comm.h"
#include "backend/backend.h"
#include "GPUs/GPU.h"
#include "nccl_hooks.h"

// IPC hooks for cuMem IPC management (defined in ipc_hooks.cpp).
// On NVIDIA this header also declares ipc_disable_all_peer_access /
// ipc_reenable_all_peer_access, the canonical P2P teardown helpers.
#include "ipc_hooks.h"
// UDS fd exchange for cross-process CUDA handle transfer (defined in ipc_fd_exchange.cpp)
#include "ipc_fd_exchange.h"

// Buffer for saving IPC export GPU data between teardown and rebuild phases
static void*  g_ipc_export_data_buf  = nullptr;
static size_t g_ipc_export_data_size = 0;
static void*  g_local_alloc_data_buf  = nullptr;
static size_t g_local_alloc_data_size = 0;

std::mutex fs_mutex;
Comm *comm;
Backend *backend;
GPU *gpu;

void* staging_buf[STAGING_BUF_NUM];

bool CR_initialized = false;

// Positive errno recorded by op sites (0 = success); FinishOpControls
// negates it into control->op_status at FINISH so clients can
// distinguish a served op's outcome from an unreported one.
int g_op_status = 0;

// memcpy_multi lives in src/memcpy_multi.cpp (declared in common.h) so the
// GPU-free unit suite can link it.

// Pointer-typed convenience wrapper over the granule clamp (the math
// lives in common.h as gpu_cr::GranuleClampLen so it is unit-testable).
static inline size_t GranuleClamp(const void* dev_ptr, size_t len) {
    return gpu_cr::GranuleClampLen(reinterpret_cast<uintptr_t>(dev_ptr), len);
}

// gpu_mem_mutex is shared with the allocation interposers (cudaMalloc/
// cudaFree in the GPU layer). The CR signal can land on a thread that is
// INSIDE one of those critical sections; a blocking lock here would
// self-deadlock the process. Ops therefore acquire with a bounded
// try-lock and fail cleanly (EBUSY) if the lock never frees — contention
// from OTHER threads resolves well within the bound, so the common case
// behaves exactly like the blocking lock it replaces.
static bool LockGpuMem(std::unique_lock<std::mutex>* lk) {
    constexpr int kAttempts = 2000;  // ~2s at 1ms per attempt
    for (int i = 0; i < kAttempts; i++) {
        if (lk->try_lock()) return true;
        struct timespec ts = {0, 1000000};  // 1ms; nanosleep is async-signal-safe
        nanosleep(&ts, nullptr);
    }
    return false;
}


double ckpt() {
    fprintf(stderr, "[vGPU-CKPT] ckpt() entered, PID=%d\n", getpid());
    fflush(stderr);

    double tot_size = 0;

    auto time_start = std::chrono::high_resolution_clock::now();
    long sync_time = 0, cpu_copy_time = 0, release_time = 0;

    void* tmp_buf = backend->get_tmp_buf();
    fprintf(stderr, "[vGPU-CKPT] tmp_buf=%p\n", tmp_buf);
    fflush(stderr);
    if (!tmp_buf) {
        fprintf(stderr, "[vGPU-CKPT] Error: dump buffer unavailable (deferred-mode materialization failed?)\n");
        g_op_status = ENOMEM;
        return -1;
    }
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
    int current_buf = 0;
    size_t buf_offset = 0;
    size_t des_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    fs_mutex.lock();
    std::unique_lock<std::mutex> lock(gpu_mem_mutex, std::defer_lock);
    if (!LockGpuMem(&lock)) {
        fprintf(stderr, "[vGPU-CKPT] Error: gpu_mem_mutex busy (allocation in flight on the "
                        "signaled thread?); failing cleanly\n");
        g_op_status = EBUSY;
        fs_mutex.unlock();
        return -1;
    }

    // With env-shrinkable buffers an oversized full checkpoint
    // is an operator-config away, so validate BEFORE touching fs or GPU
    // state — nothing dumped, nothing released, clean op_status failure
    // instead of the historical mid-loop exit(-1).
    {
        size_t required = ROUND_UP_2MB(sizeof(shared_mem_fs));
        for (const auto& entry : allocated_memory)
            required += ROUND_UP_2MB(entry.second);
        if (required > SHM_SIZE) {
            fprintf(stderr, "[vGPU-CKPT] Error: checkpoint needs %zu MiB but the dump buffer is "
                            "%zu MiB (raise GPU_CR_SHM_MB / GPU_CR_SHM_GB); failing cleanly\n",
                    required >> 20, static_cast<size_t>(SHM_SIZE) >> 20);
            g_op_status = ENOSPC;
            fs_mutex.unlock();
            return -1;
        }
    }

    fs->file_num = 0;
    fs->current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        fs_mutex.unlock();
        exit(-1);
    }
    gpu->recordEvent(event, stream);
    
    fprintf(stderr, "[vGPU-CKPT] ckpt %ld ptrs\n", allocated_memory.size());
    fflush(stderr);

    int ptr_idx = 0;
    for (const auto& entry : allocated_memory) {
        fprintf(stderr, "[vGPU-CKPT] Processing ptr #%d: %p\n", ++ptr_idx, entry.first);
        fflush(stderr);
        void* d = entry.first;
        uint64_t size = ROUND_UP_2MB(entry.second);
        tot_size += size;

        // Record file info
        fs->files[fs->file_num].ptr = d;
        fs->files[fs->file_num].start_offset = fs->current_offset;
        fs->files[fs->file_num].size = size;
        fs->current_offset += size;
        if (fs->current_offset > SHM_SIZE) {
            fprintf(stderr, "[vGPU-CKPT] Error: Not enough space in shared memory\n");
            fs_mutex.unlock();
            exit(-1);
        }
        fs->file_num++;
        if (fs->file_num >= MAX_FILE_NUM) {
            fprintf(stderr, "[vGPU-CKPT] Error: Too many files in shared memory fs\n");
            fs_mutex.unlock();
            exit(-1);
        }
        
        // Copy data from GPU to staging buffer
        while(size > 0) {
            size_t cur_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            void* start_addr = (char*)staging_buf[current_buf & 1] + buf_offset;
            
            if (gpu->memcpyAsync(start_addr, d, cur_size, GPUMemcpyKind::DeviceToHost, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                fs_mutex.unlock();
                exit(-1);
            }
            
            buf_offset += cur_size;
            d = (char*)d + cur_size;
            size -= cur_size;
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                if(current_buf > 0) {
                    auto t3 = std::chrono::high_resolution_clock::now();
                    gpu->synchronizeEvent(event);
                    auto t4 = std::chrono::high_resolution_clock::now();
                    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
                    
                    auto t5 = std::chrono::high_resolution_clock::now();
                    memcpy_multi((char*)fs + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
                    auto t6 = std::chrono::high_resolution_clock::now();
                    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
                    
                    des_offset += STAGING_BUF_SIZE;
                }
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    if(current_buf > 0) {
        auto t3 = std::chrono::high_resolution_clock::now();
        gpu->synchronizeEvent(event);
        auto t4 = std::chrono::high_resolution_clock::now();
        sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        
        auto t5 = std::chrono::high_resolution_clock::now();
        memcpy_multi((char*)fs + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
        auto t6 = std::chrono::high_resolution_clock::now();
        cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        
        des_offset += STAGING_BUF_SIZE;
    }
    
    auto t7 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto t8 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
    
    auto t9 = std::chrono::high_resolution_clock::now();
    memcpy_multi((char*)fs + des_offset, staging_buf[current_buf & 1], buf_offset);
    auto t10 = std::chrono::high_resolution_clock::now();
    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
    
    assert(des_offset + buf_offset == fs->current_offset);
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);

    // Release physical GPU memory after checkpoint (but keep virtual addresses)
    fprintf(stderr, "Releasing physical GPU memory for %ld pointers...\n", allocated_memory.size());
    auto t11 = std::chrono::high_resolution_clock::now();
    for (const auto& entry : allocated_memory) {
        void* ptr = entry.first;
        if (gpu->releasePhysicalMemory(ptr) != 0) {
            fprintf(stderr, "Error: Failed to release physical memory for ptr %p\n", ptr);
            fs_mutex.unlock();
            exit(-1);
        }
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    release_time = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();
    fprintf(stderr, "Physical GPU memory released, virtual addresses preserved\n");
    

    fprintf(stderr, "=== Checkpoint Timing Breakdown ===\n");
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n", 
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  Release memory:   %6ld ms\n", release_time / 1000);
    long data_transfer_time = sync_time + cpu_copy_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "===================================\n");
    
    fs_mutex.unlock();
    return tot_size;
}

static bool FindContainingAllocation(void* ptr, void** base_ptr, size_t* alloc_size) {
    for (auto const& [alloc_ptr, size] : allocated_memory) {
        if (ptr >= alloc_ptr &&
            static_cast<char*>(ptr) < static_cast<char*>(alloc_ptr) + size) {
            *base_ptr = alloc_ptr;
            *alloc_size = size;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Destination-path selective checkpoints.
// A caller-chosen file replaces the per-PID staging buffer for one op.
// The caller pre-creates the file (O_CREAT only); sizing is done HERE,
// because only the .so knows the containing-allocation totals. All
// dest-path failures set g_op_status and return instead of exit(-1) —
// the workload must survive a bad path or a full store.
// ---------------------------------------------------------------------------
constexpr unsigned long kHugetlbfsMagic = 0x958458f6;

static uint64_t g_dump_generation = 0;

struct DestMap {
    void*  addr = nullptr;
    size_t map_size = 0; // bytes mapped
    size_t capacity = 0; // header + extents + commit marker
    int    fd = -1;
    bool   hugetlb = false;
};

static void DestClose(DestMap* dm) {
    if (dm->addr && dm->addr != MAP_FAILED) munmap(dm->addr, dm->map_size);
    if (dm->fd >= 0) close(dm->fd);
    dm->addr = nullptr;
    dm->fd = -1;
}

static bool DestOpenCommon(const char* path, int prot, DestMap* dm) {
    // Never O_CREAT: existence is the caller's responsibility, and creating
    // here would let a stale dest_path materialize files.
    dm->fd = open(path, (prot & PROT_WRITE) ? O_RDWR : O_RDONLY);
    if (dm->fd < 0) {
        g_op_status = errno;
        fprintf(stderr, "[vGPU-DEST] open(%s) failed: %s\n", path, strerror(errno));
        return false;
    }
    struct statfs sfs;
    if (fstatfs(dm->fd, &sfs) == 0 && static_cast<unsigned long>(sfs.f_type) == kHugetlbfsMagic)
        dm->hugetlb = true;
    return true;
}

static bool DestOpenForCkpt(const char* path, size_t total, DestMap* dm) {
    if (!DestOpenCommon(path, PROT_READ | PROT_WRITE, dm)) return false;
    // tmpfs/disk reserve nothing at ftruncate or mmap: without fallocate a
    // full filesystem is a SIGBUS mid-store, not an error. hugetlbfs
    // reserves at mmap, so ENOMEM already surfaces there.
    if (!dm->hugetlb) {
        int rc = posix_fallocate(dm->fd, 0, static_cast<off_t>(total));
        if (rc != 0) {
            g_op_status = rc;
            fprintf(stderr, "[vGPU-DEST] fallocate(%s, %zu) failed: %s\n", path, total, strerror(rc));
            DestClose(dm);
            return false;
        }
    }
    size_t fsize = dm->hugetlb ? ROUND_UP_2MB(total) : total;
    if (ftruncate(dm->fd, static_cast<off_t>(fsize)) < 0) {
        g_op_status = errno;
        fprintf(stderr, "[vGPU-DEST] ftruncate(%s, %zu) failed: %s\n", path, fsize, strerror(errno));
        DestClose(dm);
        return false;
    }
    dm->addr = mmap(nullptr, fsize, PROT_READ | PROT_WRITE, MAP_SHARED, dm->fd, 0);
    if (dm->addr == MAP_FAILED) {
        g_op_status = errno;
        fprintf(stderr, "[vGPU-DEST] mmap(%s, %zu) failed: %s\n", path, fsize, strerror(errno));
        DestClose(dm);
        return false;
    }
    dm->map_size = fsize;
    dm->capacity = total;
    return true;
}

// Restore side: refuse anything without a valid header and commit marker —
// a torn dump must fail the op, not feed garbage to the GPU.
static bool DestOpenForRestore(const char* path, DestMap* dm) {
    if (!DestOpenCommon(path, PROT_READ, dm)) return false;
    struct stat st;
    if (fstat(dm->fd, &st) < 0) {
        g_op_status = errno;
        DestClose(dm);
        return false;
    }
    size_t min_size = ROUND_UP_2MB(sizeof(shared_mem_fs)) + sizeof(DumpCommit);
    if (static_cast<size_t>(st.st_size) < min_size) {
        g_op_status = EINVAL;
        fprintf(stderr, "[vGPU-DEST] %s too small (%lld bytes) to hold a dump\n", path,
                static_cast<long long>(st.st_size));
        DestClose(dm);
        return false;
    }
    dm->addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, dm->fd, 0);
    if (dm->addr == MAP_FAILED) {
        g_op_status = errno;
        fprintf(stderr, "[vGPU-DEST] mmap(%s) failed: %s\n", path, strerror(errno));
        DestClose(dm);
        return false;
    }
    dm->map_size = st.st_size;
    dm->capacity = st.st_size;

    shared_mem_fs* fs = static_cast<shared_mem_fs*>(dm->addr);
    if (!gpu_cr::DumpHeaderPlausible(fs->file_num, fs->current_offset, dm->map_size)) {
        g_op_status = EINVAL;
        fprintf(stderr, "[vGPU-DEST] %s has implausible header (file_num=%llu current_offset=%llu)\n",
                path, static_cast<unsigned long long>(fs->file_num),
                static_cast<unsigned long long>(fs->current_offset));
        DestClose(dm);
        return false;
    }
    const DumpCommit* dc = reinterpret_cast<const DumpCommit*>(
        static_cast<const char*>(dm->addr) +
        gpu_cr::DumpCommitOffset(fs->current_offset));
    if (dc->magic != gpu_cr::kDumpCommitMagic) {
        g_op_status = EINVAL;
        fprintf(stderr, "[vGPU-DEST] %s has no commit marker: torn or foreign dump, refusing restore\n", path);
        DestClose(dm);
        return false;
    }
    return true;
}

double ckpt_selective(const SelectiveCrRequest* req) {
    // A non-empty dest_path selects destination-file routing; it must be
    // NUL-terminated within the array (the request lives in a
    // caller-writable mapping, so treat it as untrusted).
    if (req->dest_path[gpu_cr::kSelectiveCrMaxPath - 1] != '\0') {
        fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: dest_path is not NUL-terminated; rejecting\n");
        g_op_status = EINVAL;
        return -1;
    }
    const char* dest_path = req->dest_path[0] != '\0' ? req->dest_path : nullptr;
    fprintf(stderr, "[vGPU-SELECTIVE-CKPT] ckpt_selective() entered, %u regions, PID=%d, dest=%s\n",
            req->num_regions, getpid(), dest_path ? dest_path : "(per-PID buffer)");
    fflush(stderr);

    // The request lives in a caller-writable mapping: bound it like the
    // dump header, before any lock or open.
    if (req->num_regions >= gpu_cr::kMaxSelectiveRegions) {
        fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: num_regions %u not below bound %u; rejecting\n",
                req->num_regions, gpu_cr::kMaxSelectiveRegions);
        g_op_status = EINVAL;
        return -1;
    }

    double tot_size = 0;

    auto time_start = std::chrono::high_resolution_clock::now();
    long sync_time = 0, cpu_copy_time = 0, release_time = 0;

    int current_buf = 0;
    size_t buf_offset = 0;
    size_t des_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    fs_mutex.lock();
    std::unique_lock<std::mutex> lock(gpu_mem_mutex, std::defer_lock);
    if (!LockGpuMem(&lock)) {
        fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: gpu_mem_mutex busy (allocation in flight "
                        "on the signaled thread?); failing cleanly\n");
        g_op_status = EBUSY;
        fs_mutex.unlock();
        return -1;
    }

    // Resolve target blocks BEFORE picking the output: with a destination
    // file the exact dump size must be known up front (only the .so knows
    // the containing-allocation totals — preloader-authoritative
    // sizing). Both mutexes are held, so the sizes cannot shift under us.
    std::set<void*> blocks_to_snapshot;
    for (uint32_t ri = 0; ri < req->num_regions; ri++) {
        void* d = req->regions[ri].ptr;
        void* base_ptr = nullptr;
        size_t alloc_size = 0;
        if (FindContainingAllocation(d, &base_ptr, &alloc_size)) {
            blocks_to_snapshot.insert(base_ptr);
        } else {
            // Reject, as the restore path does for the same condition: the
            // region list is caller input, so an unresolvable pointer means
            // the caller's view of the process is stale. Skipping it would
            // release less than the caller asked for -- or nothing at all --
            // and still report success, which reads at the scheduler as VRAM
            // that is free when it is not. Nothing is open or released yet.
            fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: region ptr %p not in any live "
                            "allocation; rejecting\n", d);
            g_op_status = EINVAL;
            fs_mutex.unlock();
            return -1;
        }
    }

    size_t dump_total = ROUND_UP_2MB(sizeof(shared_mem_fs));
    for (void* base_ptr : blocks_to_snapshot)
        dump_total += allocated_memory.find(base_ptr)->second;

    DestMap dm;
    shared_mem_fs* fs;
    size_t fs_capacity; // extent bound: header + extents, excluding the marker
    if (dest_path) {
        if (!DestOpenForCkpt(dest_path, gpu_cr::DumpCommitOffset(dump_total) + sizeof(DumpCommit), &dm)) {
            fs_mutex.unlock();
            return -1;
        }
        fs = static_cast<shared_mem_fs*>(dm.addr);
        fs_capacity = dump_total;
    } else {
        fs = static_cast<shared_mem_fs*>(backend->get_tmp_buf());
        if (!fs) {
            fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: dump buffer unavailable (deferred-mode "
                            "materialization failed?)\n");
            g_op_status = ENOMEM;
            fs_mutex.unlock();
            return -1;
        }
        fs_capacity = SHM_SIZE;
        // The legacy buffer is env-shrinkable now — same clean
        // pre-op failure as the dest path instead of the historical
        // mid-loop exit(-1).
        if (dump_total > fs_capacity) {
            fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: selective checkpoint needs %zu MiB but the "
                            "dump buffer is %zu MiB (raise GPU_CR_SHM_MB / GPU_CR_SHM_GB); failing cleanly\n",
                    dump_total >> 20, fs_capacity >> 20);
            g_op_status = ENOSPC;
            fs_mutex.unlock();
            return -1;
        }
    }

    fs->file_num = 0;
    fs->current_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));

    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\n");
        if (dest_path) {
            g_op_status = EIO;
            DestClose(&dm);
            fs_mutex.unlock();
            return -1;
        }
        fs_mutex.unlock();
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\n");
        gpu->destroyStream(stream);
        if (dest_path) {
            g_op_status = EIO;
            DestClose(&dm);
            fs_mutex.unlock();
            return -1;
        }
        fs_mutex.unlock();
        exit(-1);
    }
    gpu->recordEvent(event, stream);

    for (void* base_ptr : blocks_to_snapshot) {
        auto it = allocated_memory.find(base_ptr);
        assert(it != allocated_memory.end());
        size_t alloc_size = it->second;
        // Dump only the caller-requested allocation size, not the 2MB-rounded
        // VMM block: the rounding padding was never handed to the application,
        // and release/remap re-round internally (releasePhysicalMemory,
        // remapPhysicalMemory), so physical block handling is unchanged.
        uint64_t size = alloc_size;
        tot_size += size;

        fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Saving VMM block: base_ptr=%p size=%lu (aligned=%lu)\n",
                base_ptr, alloc_size, size);

        fs->files[fs->file_num].ptr = base_ptr;
        fs->files[fs->file_num].start_offset = fs->current_offset;
        fs->files[fs->file_num].size = size;
        fs->current_offset += size;
        // Bound against the actual output: destination files are sized
        // exactly (this cannot trip there — defensive only); the per-PID
        // buffer keeps its historical SHM_SIZE check.
        if (fs->current_offset > fs_capacity) {
            fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: Not enough space in %s\n",
                    dest_path ? dest_path : "shared memory");
            gpu->destroyStream(stream);
            gpu->destroyEvent(event);
            if (dest_path) {
                g_op_status = ENOSPC;
                DestClose(&dm);
                fs_mutex.unlock();
                return -1;
            }
            fs_mutex.unlock();
            exit(-1);
        }
        fs->file_num++;
        if (fs->file_num >= MAX_FILE_NUM) {
            fprintf(stderr, "[vGPU-SELECTIVE-CKPT] Error: Too many files in shared memory fs\n");
            gpu->destroyStream(stream);
            gpu->destroyEvent(event);
            if (dest_path) {
                g_op_status = E2BIG;
                DestClose(&dm);
                fs_mutex.unlock();
                return -1;
            }
            fs_mutex.unlock();
            exit(-1);
        }

        void* d = base_ptr;
        while (size > 0) {
            size_t cur_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            cur_size = GranuleClamp(d, cur_size);  // granule-boundary clamp
            void* start_addr = static_cast<char*>(staging_buf[current_buf & 1]) + buf_offset;

            if (gpu->memcpyAsync(start_addr, d, cur_size, GPUMemcpyKind::DeviceToHost, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\n");
                if (dest_path) {
                    // Nothing has been released yet, and without a commit
                    // marker the partial dump is detectably torn.
                    gpu->destroyStream(stream);
                    gpu->destroyEvent(event);
                    g_op_status = EIO;
                    DestClose(&dm);
                    fs_mutex.unlock();
                    return -1;
                }
                fs_mutex.unlock();
                exit(-1);
            }

            buf_offset += cur_size;
            d = static_cast<char*>(d) + cur_size;
            size -= cur_size;
            if (buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                if (current_buf > 0) {
                    auto t3 = std::chrono::high_resolution_clock::now();
                    gpu->synchronizeEvent(event);
                    auto t4 = std::chrono::high_resolution_clock::now();
                    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

                    auto t5 = std::chrono::high_resolution_clock::now();
                    memcpy_multi(reinterpret_cast<char*>(fs) + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
                    auto t6 = std::chrono::high_resolution_clock::now();
                    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

                    des_offset += STAGING_BUF_SIZE;
                }
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }

    if (current_buf > 0) {
        auto t3 = std::chrono::high_resolution_clock::now();
        gpu->synchronizeEvent(event);
        auto t4 = std::chrono::high_resolution_clock::now();
        sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

        auto t5 = std::chrono::high_resolution_clock::now();
        memcpy_multi(reinterpret_cast<char*>(fs) + des_offset, staging_buf[(current_buf - 1) & 1], STAGING_BUF_SIZE);
        auto t6 = std::chrono::high_resolution_clock::now();
        cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

        des_offset += STAGING_BUF_SIZE;
    }

    auto t7 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto t8 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();

    auto t9 = std::chrono::high_resolution_clock::now();
    memcpy_multi(reinterpret_cast<char*>(fs) + des_offset, staging_buf[current_buf & 1], buf_offset);
    auto t10 = std::chrono::high_resolution_clock::now();
    cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();

    assert(des_offset + buf_offset == fs->current_offset);
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);

    // Commit marker AFTER the last extent landed: restores refuse dumps
    // without it, so a crash anywhere above leaves a detectably-torn file.
    // The magic store is release-ordered so it lands last by construction
    // (not just in program order) — a marker is never observed with a
    // half-written generation.
    const uint64_t marker_off = gpu_cr::DumpCommitOffset(fs->current_offset);
    if (marker_off + sizeof(DumpCommit) <=
            (dest_path ? dm.capacity : static_cast<size_t>(SHM_SIZE))) {
        // Extents are unpadded, so current_offset can be misaligned; the
        // marker sits at the next alignof(DumpCommit) boundary so this
        // release store (and the restore-side load) is naturally aligned.
        DumpCommit* dc = reinterpret_cast<DumpCommit*>(
            reinterpret_cast<char*>(fs) + marker_off);
        dc->generation = ++g_dump_generation;
        __atomic_store_n(&dc->magic, gpu_cr::kDumpCommitMagic, __ATOMIC_RELEASE);
    }

    fprintf(stderr, "Releasing physical GPU memory for %lu selective regions...\n", (unsigned long)fs->file_num);
    auto t11 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < fs->file_num; i++) {
        void* ptr = fs->files[i].ptr;
        if (gpu->releasePhysicalMemory(ptr) != 0) {
            fprintf(stderr, "Error: Failed to release physical memory for ptr %p\n", ptr);
            if (dest_path) {
                g_op_status = EIO;
                DestClose(&dm);
                fs_mutex.unlock();
                return -1;
            }
            fs_mutex.unlock();
            exit(-1);
        }
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    release_time = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();
    fprintf(stderr, "Physical GPU memory released for selective regions, virtual addresses preserved\n");

    fprintf(stderr, "=== Selective Checkpoint Timing Breakdown ===\n");
    fprintf(stderr, "  Regions:          %6lu\n", (unsigned long)fs->file_num);
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n",
            cpu_copy_time / 1000,
            tot_size > 0 ? (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0) : 0.0);
    fprintf(stderr, "  Release memory:   %6ld ms\n", release_time / 1000);
    long data_transfer_time = sync_time + cpu_copy_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            tot_size > 0 ? (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0) : 0.0);
    fprintf(stderr, "===============================================\n");

    if (dest_path) DestClose(&dm);
    fs_mutex.unlock();
    return tot_size;
}

double restore_ptr_and_content() {
    std::unique_lock<std::mutex> lock(gpu_mem_mutex, std::defer_lock);
    if (!LockGpuMem(&lock)) {
        fprintf(stderr, "[vGPU-restore] Error: gpu_mem_mutex busy (allocation in flight on the "
                        "signaled thread?); failing cleanly\n");
        g_op_status = EBUSY;
        return -1;
    }
    double tot_size = 0;

    long remap_time = 0, cpu_copy_time = 0, sync_time = 0;

    void* tmp_buf = backend->get_tmp_buf();
    if (!tmp_buf) {
        fprintf(stderr, "[vGPU-restore] Error: dump buffer unavailable (deferred-mode materialization failed?)\n");
        g_op_status = ENOMEM;
        return -1;
    }
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;

    uint64_t file_num = fs->file_num;
    fprintf(stderr, "[vGPU-restore] restore %lu ptrs\n", file_num);
    
    // Remap physical memory for all pointers before copying data
    fprintf(stderr, "[vGPU-restore] Remapping physical GPU memory for %lu pointers...\n", file_num);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < file_num; i++) {
        void* ptr = fs->files[i].ptr;
        uint64_t size = fs->files[i].size;
        if (gpu->remapPhysicalMemory(ptr, size) != 0) {
            fprintf(stderr, "Error: Failed to remap physical memory for ptr %p\n", ptr);
            exit(-1);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    remap_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    fprintf(stderr, "[vGPU-restore] Physical GPU memory remapped\n");
    
    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\\n");
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\\n");
        exit(-1);
    }
    gpu->recordEvent(event, stream);

    int current_buf = 0;
    size_t buf_offset = 0;
    size_t src_offset = 0;
    
    for (uint64_t i = 0; i < file_num; i++) {
        void* requestedAddr = fs->files[i].ptr;
        uint64_t offset = fs->files[i].start_offset;
        uint64_t size = fs->files[i].size;
        tot_size += size;
        
        if(i == 0) {
            src_offset = fs->files[i].start_offset;
            size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
            auto tc1 = std::chrono::high_resolution_clock::now();
            memcpy_multi(staging_buf[current_buf & 1], (char*)fs + src_offset, cpu_copy_size);
            auto tc2 = std::chrono::high_resolution_clock::now();
            cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1).count();
            buf_offset = 0;
        }
        
        while(size > 0) {
            size_t this_copy_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            assert(buf_offset == offset - src_offset);
            
            if (gpu->memcpyAsync(requestedAddr, (char*)staging_buf[current_buf & 1] + (offset - src_offset),
                               this_copy_size, GPUMemcpyKind::HostToDevice, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\\n");
                exit(-1);
            }
            
            buf_offset += this_copy_size;
            offset += this_copy_size;
            requestedAddr = (char*)requestedAddr + this_copy_size;
            size -= this_copy_size;
            
            if(buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                src_offset += STAGING_BUF_SIZE;
                size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
                
                auto ts1 = std::chrono::high_resolution_clock::now();
                gpu->synchronizeEvent(event);
                auto ts2 = std::chrono::high_resolution_clock::now();
                sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts1).count();
                
                auto tc3 = std::chrono::high_resolution_clock::now();
                memcpy_multi(staging_buf[(current_buf + 1) & 1], (char*)fs + src_offset, cpu_copy_size);
                auto tc4 = std::chrono::high_resolution_clock::now();
                cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc4 - tc3).count();
                
                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }
    
    auto ts3 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto ts4 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts3).count();
    
    gpu->destroyStream(stream);
    gpu->destroyEvent(event);
    
    fprintf(stderr, "=== Restore Timing Breakdown ===\n");
    fprintf(stderr, "  Remap memory:     %6ld ms\n", remap_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n",
            cpu_copy_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0));
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    long data_transfer_time = cpu_copy_time + sync_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0));
    fprintf(stderr, "================================\n");
    
    return tot_size;
}

// Validates a selective dump's whole extent table before anything touches GPU
// state or reads a byte of payload.
//
// ckpt_selective lays the extents out back to back from the end of the header,
// one per live allocation, so a dump it wrote satisfies every check here by
// construction; one that does not is torn or forged. That matters because the
// offsets below are what index the mapping and the staging buffer later:
// `fs->current_offset - start_offset` is unsigned, so a first extent claiming
// to start past the end of the dump underflows to a huge length and the staged
// copy reads far outside the mapping.
//
// Validating up front, rather than per entry inside the remap loop, also keeps
// a bad entry from being caught only after earlier entries were already
// remapped — which would leave the allocation set half-restored with no way
// back. DumpHeaderPlausible covers only file_num and current_offset.
static bool SelectiveDumpExtentsValid(const shared_mem_fs* fs, uint64_t file_num) {
    // Inductive bound: expected <= fs->current_offset holds on entry and is
    // re-established by the size check below, so every comparison is a
    // subtraction against a known-larger value and no addition can overflow.
    uint64_t expected = ROUND_UP_2MB(sizeof(shared_mem_fs));
    if (fs->current_offset < expected) {
        fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] dump ends at %lu, before its own header; rejecting\n",
                (unsigned long)fs->current_offset);
        return false;
    }
    for (uint64_t i = 0; i < file_num; i++) {
        void* const ptr = fs->files[i].ptr;
        const uint64_t size = fs->files[i].size;
        if (fs->files[i].start_offset != expected) {
            fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] dump entry %lu starts at %lu, expected %lu; "
                            "extents must be contiguous\n",
                    (unsigned long)i, (unsigned long)fs->files[i].start_offset,
                    (unsigned long)expected);
            return false;
        }
        if (size == 0 || size > fs->current_offset - expected) {
            fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] dump entry %lu (%p) claims %lu bytes, which do "
                            "not fit before the dump's end offset %lu; rejecting\n",
                    (unsigned long)i, ptr, (unsigned long)size,
                    (unsigned long)fs->current_offset);
            return false;
        }
        // Exactly the live allocation, not merely inside one: ckpt_selective
        // resolves every requested pointer to its whole containing allocation,
        // so a partial extent means this dump did not come from that path.
        auto it = allocated_memory.find(ptr);
        if (it == allocated_memory.end() || it->second != size) {
            fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] dump entry %lu (%p, size %lu) is not a live "
                            "allocation of exactly that size; rejecting\n",
                    (unsigned long)i, ptr, (unsigned long)size);
            return false;
        }
        // Duplicate pointers would remap and overwrite the same allocation
        // twice. Quadratic, but file_num is capped at MAX_FILE_NUM and this
        // runs on the signal-handler path, where a hash set's allocation is
        // the worse trade.
        for (uint64_t j = 0; j < i; j++) {
            if (fs->files[j].ptr == ptr) {
                fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] dump entry %lu repeats pointer %p from "
                                "entry %lu; rejecting\n", (unsigned long)i, ptr, (unsigned long)j);
                return false;
            }
        }
        expected += size;
    }
    // The staged copy runs from the first extent to current_offset, so a table
    // that stops short would drag in bytes no entry claims.
    if (expected != fs->current_offset) {
        fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] extents end at %lu but the dump declares %lu; "
                        "rejecting\n", (unsigned long)expected, (unsigned long)fs->current_offset);
        return false;
    }
    return true;
}

double restore_ptr_and_content_selective(const SelectiveCrRequest* req) {
    // Same untrusted-request handling as ckpt_selective: NUL-termination
    // first, then a non-empty dest_path selects destination-file routing.
    if (req->dest_path[gpu_cr::kSelectiveCrMaxPath - 1] != '\0') {
        fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Error: dest_path is not NUL-terminated; rejecting\n");
        g_op_status = EINVAL;
        return -1;
    }
    const char* dest_path = req->dest_path[0] != '\0' ? req->dest_path : nullptr;
    // The request lives in a caller-writable mapping: bound it like the
    // dump header, before any lock or open.
    if (req->num_regions >= gpu_cr::kMaxSelectiveRegions) {
        fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Error: num_regions %u not below bound %u; rejecting\n",
                req->num_regions, gpu_cr::kMaxSelectiveRegions);
        g_op_status = EINVAL;
        return -1;
    }
    std::unique_lock<std::mutex> lock(gpu_mem_mutex, std::defer_lock);
    if (!LockGpuMem(&lock)) {
        fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Error: gpu_mem_mutex busy (allocation in "
                        "flight on the signaled thread?); failing cleanly\n");
        g_op_status = EBUSY;
        return -1;
    }
    double tot_size = 0;

    long remap_time = 0, cpu_copy_time = 0, sync_time = 0;

    DestMap dm;
    shared_mem_fs* fs;
    if (dest_path) {
        if (!DestOpenForRestore(dest_path, &dm)) return -1;
        fs = static_cast<shared_mem_fs*>(dm.addr);
        // Cross-check the request against the dump: every
        // requested region must resolve to a live allocation whose base is
        // one of the dump's files — the header is otherwise trusted input
        // from a caller-writable directory.
        for (uint32_t ri = 0; ri < req->num_regions; ri++) {
            void* base_ptr = nullptr;
            size_t alloc_size = 0;
            if (!FindContainingAllocation(req->regions[ri].ptr, &base_ptr, &alloc_size)) {
                fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] region ptr %p not in any live allocation\n",
                        req->regions[ri].ptr);
                g_op_status = EINVAL;
                DestClose(&dm);
                return -1;
            }
            bool in_dump = false;
            for (uint64_t i = 0; i < fs->file_num && !in_dump; i++)
                in_dump = (fs->files[i].ptr == base_ptr);
            if (!in_dump) {
                fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] region ptr %p (block %p) absent from dump %s\n",
                        req->regions[ri].ptr, base_ptr, dest_path);
                g_op_status = EINVAL;
                DestClose(&dm);
                return -1;
            }
        }
    } else {
        fs = static_cast<shared_mem_fs*>(backend->get_tmp_buf());
        if (!fs) {
            fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Error: dump buffer unavailable (deferred-mode "
                            "materialization failed?)\n");
            g_op_status = ENOMEM;
            return -1;
        }
        // DestOpenForRestore bounds the header on the dest path; the per-PID
        // buffer needs the same before anything indexes files[], which the
        // extent walk below does using the header's own file_num. SHM_SIZE is
        // the runtime buffer size, so this also catches an offset past a
        // buffer shrunk by GPU_CR_SHM_MB.
        if (!gpu_cr::DumpHeaderPlausible(fs->file_num, fs->current_offset,
                                         static_cast<uint64_t>(SHM_SIZE))) {
            fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Error: implausible dump header in the per-PID "
                            "buffer (file_num=%lu current_offset=%lu); rejecting\n",
                    (unsigned long)fs->file_num, (unsigned long)fs->current_offset);
            g_op_status = EINVAL;
            return -1;
        }
    }

    uint64_t file_num = fs->file_num;
    fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] restore %lu selective regions from %s\n",
            file_num, dest_path ? dest_path : "(per-PID buffer)");

    if (!SelectiveDumpExtentsValid(fs, file_num)) {
        g_op_status = EINVAL;
        if (dest_path) DestClose(&dm);
        return -1;
    }

    fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Remapping physical GPU memory...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < file_num; i++) {
        void* ptr = fs->files[i].ptr;
        uint64_t size = fs->files[i].size;
        if (gpu->remapPhysicalMemory(ptr, size) != 0) {
            fprintf(stderr, "Error: Failed to remap physical memory for ptr %p\n", ptr);
            if (dest_path) {
                g_op_status = EIO;
                DestClose(&dm);
                return -1;
            }
            exit(-1);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    remap_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    fprintf(stderr, "[vGPU-SELECTIVE-RESTORE] Physical GPU memory remapped\n");

    GPUStream stream;
    GPUEvent event;
    if (gpu->createStream(&stream) != 0) {
        fprintf(stderr, "Error: Failed to create stream\n");
        if (dest_path) {
            g_op_status = EIO;
            DestClose(&dm);
            return -1;
        }
        exit(-1);
    }
    if (gpu->createEvent(&event) != 0) {
        fprintf(stderr, "Error: Failed to create event\n");
        gpu->destroyStream(stream);
        if (dest_path) {
            g_op_status = EIO;
            DestClose(&dm);
            return -1;
        }
        exit(-1);
    }
    gpu->recordEvent(event, stream);

    int current_buf = 0;
    size_t buf_offset = 0;
    size_t src_offset = 0;

    for (uint64_t i = 0; i < file_num; i++) {
        void* requested_addr = fs->files[i].ptr;
        uint64_t offset = fs->files[i].start_offset;
        uint64_t size = fs->files[i].size;
        tot_size += size;

        if (i == 0) {
            src_offset = fs->files[i].start_offset;
            size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);
            auto tc1 = std::chrono::high_resolution_clock::now();
            memcpy_multi(staging_buf[current_buf & 1], reinterpret_cast<char*>(fs) + src_offset, cpu_copy_size);
            auto tc2 = std::chrono::high_resolution_clock::now();
            cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1).count();
            buf_offset = 0;
        }

        while (size > 0) {
            size_t this_copy_size = std::min(size, (size_t)STAGING_BUF_SIZE - buf_offset);
            this_copy_size = GranuleClamp(requested_addr, this_copy_size);  // granule-boundary clamp
            assert(buf_offset == offset - src_offset);

            if (gpu->memcpyAsync(requested_addr, static_cast<char*>(staging_buf[current_buf & 1]) + (offset - src_offset),
                               this_copy_size, GPUMemcpyKind::HostToDevice, stream) != 0) {
                fprintf(stderr, "Error: memcpyAsync failed\n");
                if (dest_path) {
                    gpu->destroyStream(stream);
                    gpu->destroyEvent(event);
                    g_op_status = EIO;
                    DestClose(&dm);
                    return -1;
                }
                exit(-1);
            }

            buf_offset += this_copy_size;
            offset += this_copy_size;
            requested_addr = static_cast<char*>(requested_addr) + this_copy_size;
            size -= this_copy_size;

            if (buf_offset >= STAGING_BUF_SIZE) {
                assert(buf_offset == STAGING_BUF_SIZE);
                src_offset += STAGING_BUF_SIZE;
                size_t cpu_copy_size = std::min((size_t)(fs->current_offset - src_offset), (size_t)STAGING_BUF_SIZE);

                auto ts1 = std::chrono::high_resolution_clock::now();
                gpu->synchronizeEvent(event);
                auto ts2 = std::chrono::high_resolution_clock::now();
                sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts1).count();

                auto tc3 = std::chrono::high_resolution_clock::now();
                memcpy_multi(staging_buf[(current_buf + 1) & 1], reinterpret_cast<char*>(fs) + src_offset, cpu_copy_size);
                auto tc4 = std::chrono::high_resolution_clock::now();
                cpu_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(tc4 - tc3).count();

                buf_offset = 0;
                current_buf++;
                gpu->recordEvent(event, stream);
            }
        }
    }

    auto ts3 = std::chrono::high_resolution_clock::now();
    gpu->synchronizeStream(stream);
    auto ts4 = std::chrono::high_resolution_clock::now();
    sync_time += std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts3).count();

    gpu->destroyStream(stream);
    gpu->destroyEvent(event);

    fprintf(stderr, "=== Selective Restore Timing Breakdown ===\n");
    fprintf(stderr, "  Regions:          %6lu\n", file_num);
    fprintf(stderr, "  Remap memory:     %6ld ms\n", remap_time / 1000);
    fprintf(stderr, "  CPU memcpy:       %6ld ms (%.2f GB/s)\n",
            cpu_copy_time / 1000,
            tot_size > 0 ? (tot_size / (1024.0*1024*1024)) / (cpu_copy_time / 1000000.0) : 0.0);
    fprintf(stderr, "  GPU sync:         %6ld ms\n", sync_time / 1000);
    long data_transfer_time = cpu_copy_time + sync_time;
    fprintf(stderr, "  Data transfer:    %6ld ms (%.2f GB/s)\n",
            data_transfer_time / 1000,
            tot_size > 0 ? (tot_size / (1024.0*1024*1024)) / (data_transfer_time / 1000000.0) : 0.0);
    fprintf(stderr, "============================================\n");

    if (dest_path) DestClose(&dm);
    return tot_size;
}

int get_id() {
    char id_name[512];
    bool ctl_mode = false;
    const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
    snprintf(id_name, sizeof(id_name), "%s/control", ctl_dir);
    // Same cross-UID policy as the per-PID control files: the counter is
    // shared by every workload on this ctl dir, and a 0755 first-creator
    // mode would refuse O_RDWR to a later workload under a different uid.
    // fchmod bypasses the umask; best-effort (non-owners cannot chmod).
    int fd_id = open(id_name, O_CREAT | O_RDWR, 0777);
    if (fd_id < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    fchmod(fd_id, 0777);
    // The counter is a single atomic int: on the ctl tmpfs one 4KiB page
    // suffices (and frees the hugepage the legacy layout pinned); on
    // hugetlbfs the historical 2MiB sizing is kept.
    size_t id_size = ctl_mode ? 4096 : HUGE_PAGE_SIZE;
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_id, id_size) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
    }
    if (ctl_mode) {
        int rc = posix_fallocate(fd_id, 0, static_cast<off_t>(id_size));
        if (rc != 0) {
            fprintf(stderr, "posix_fallocate(%s): %s (ctl tmpfs full?)\n", id_name, strerror(rc));
            exit(EXIT_FAILURE);
        }
    }
    std::atomic<int>* id_ptr = static_cast<std::atomic<int>*>(
        mmap(nullptr, id_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_id, 0));
    if (id_ptr == MAP_FAILED) {
        perror("mmap()");
        exit(EXIT_FAILURE);
    }
    int id = id_ptr->fetch_add(1);
    fprintf(stderr, "Process ID: %d, assigned CR ID: %d\n", getpid(), id);
    return id;
}


void init_CR() {
    if (CR_initialized) {
        fprintf(stderr, "[init_CR] CR already initialized\n");
        return;
    }

    fprintf(stderr, "[init_CR] Starting CR initialization...\n");
    int id = get_id();

    // Write PID -> ID mapping file. write(2), not stdio: buffered stdio
    // silently produces an EMPTY file on hugetlbfs (the historical pid_map
    // bug); on the ctl tmpfs plain writes just work.
    bool ctl_mode = false;
    const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
    char map_name[512];
    snprintf(map_name, sizeof(map_name), "%s/pid_map_%d", ctl_dir, getpid());
    int fd_map = open(map_name, O_CREAT | O_TRUNC | O_WRONLY, 0666);
    if (fd_map >= 0) {
        char id_buf[32];
        int id_len = snprintf(id_buf, sizeof(id_buf), "%d\n", id);
        ssize_t written = write(fd_map, id_buf, id_len);
        int write_errno = errno;  // saved before close()/chmod() can clobber it
        close(fd_map);
        chmod(map_name, 0666);
        if (written == id_len)
            fprintf(stderr, "[init_CR] Written PID map: %s -> %d\n", map_name, id);
        else
            fprintf(stderr, "[init_CR] PID map write to %s failed (%s) — callers fall back to /proc/<pid>/maps\n",
                    map_name, written < 0 ? strerror(write_errno) : "short write");
    } else {
        perror("[init_CR] Failed to open PID map file for writing");
    }

    comm = new ShareMemComm(getpid());
    comm->setup();
    // Publish selective support. Persistent across ops: the consume-once
    // zeroing at FINISH deliberately leaves this word alone (FINISH
    // re-asserts it for clients that attach later). Until this store a
    // fresh mapping reads 0, so a client that races init_CR refuses
    // selective ops instead of signaling unarmed handlers.
    (static_cast<ShareMemComm*>(comm))->control->selective_ready = gpu_cr::kSelectiveReady;
    backend = new ShareMem(id);
    backend->setup();
    gpu = createGPU();  // createGPU() will detect the GPU vendor and return the appropriate GPU object
    fprintf(stderr, "[init_CR] GPU vendor: %s\n", gpu->getVendorName().c_str());
    fprintf(stderr, "[init_CR] Allocating staging buffer (%zu MB)...\n", 
            (STAGING_BUF_SIZE * STAGING_BUF_NUM) / (1024 * 1024));
    
    void* tmp_buf_host = backend->get_host_buffer();
    if (!tmp_buf_host) {
        fprintf(stderr, "[init_CR] Error: Backend host buffer is null\n");
        exit(EXIT_FAILURE);
    }
    
    // Try to register as pinned memory
    size_t total_size = STAGING_BUF_SIZE * STAGING_BUF_NUM;
    if (gpu->registerHostMemory(tmp_buf_host, total_size) == 0) {
        fprintf(stderr, "[init_CR] Successfully registered as pinned memory\n");
    } else {
        fprintf(stderr, "[init_CR] Note: Could not register as pinned (will use regular memory)\n");
    }
    
    for (int i = 0; i < STAGING_BUF_NUM; i++) {
        staging_buf[i] = (char*)tmp_buf_host + i * STAGING_BUF_SIZE;
    }

    CR_initialized = true;
    fprintf(stderr, "[init_CR] Initialization complete, setting CR_initialized = true\n");
}

// Post-op bookkeeping: publish the op result, then consume the request's
// dest_path so a stale path can never redirect a later op. Clients gate
// cuda-checkpoint --toggle on a positive op_status — never freeze a
// process whose state was not saved.
static void FinishOp(ShareMemComm* scomm) {
    gpu_cr::FinishOpControls(scomm->control, g_op_status);
}

void cr_signal_handler(int signum) {
    fprintf(stderr, "[vGPU] Received signal %d from process %d\n", signum, getpid());
    fflush(stderr);
    
    // Only handle our specific signals
    if (signum != CR_INIT_SIGNAL && signum != CR_CKPT_SIGNAL && signum != CR_RESTORE_SIGNAL) {
        fprintf(stderr, "[vGPU] Ignoring unknown signal %d (not a CR signal)\n", signum);
        return;
    }
    
    if(signum == CR_INIT_SIGNAL) {
        if (!CR_initialized) {
            fprintf(stderr, "[vGPU] Starting init_CR()...\n");
            init_CR();
            fprintf(stderr, "[vGPU] CR initialization complete\n");
        } else {
            fprintf(stderr, "[vGPU] CR already initialized, skipping\n");
        }
        comm->send_msg(FINISH_MSG);
        fprintf(stderr, "[vGPU] FINISH_MSG sent, returning from signal handler\n");
        fflush(stderr);
        return;
    }

    if(!CR_initialized) {
        fprintf(stderr, "CR not initialized, initializing now...\n");
        init_CR();
    }

    uint32_t msg = comm->recv_msg();
    // Mask every CR signal for the rest of the operation. A handler auto-blocks
    // only its own signum, so a *different* CR signal could otherwise land on
    // this thread while the op below holds gpu_mem_mutex; the nested handler's
    // pushContext would then block forever on a non-recursive mutex its own
    // thread already owns. pushContext masks only around its snapshot, which is
    // too narrow — the mutex is held for the whole ckpt/restore. Holding the
    // mask through popContext and FINISH_MSG means no nesting is possible until
    // the op has been reported; a signal sent meanwhile stays pending and is
    // delivered on return.
    gpu_cr::ScopedBlockCrSignals op_signal_block;
    // Pair the pop below with this handler's own push: pushContext leaves the
    // stack untouched when it fails, and an unconditional pop would then
    // consume a context an earlier handler pushed and failed to pop.
    const bool context_pushed = gpu->pushContext() == 0;
    if(msg == SELECTIVE_CKPT_MSG) {
        ShareMemComm* scomm = static_cast<ShareMemComm*>(comm);
        const SelectiveCrRequest* req = &scomm->control->selective_req;
        g_op_status = 0;
        fprintf(stderr, "waiting for kernels to finish...\n");
        gpu->syncAllKernels();
        fprintf(stderr, "start selective ckpt (%u regions)...\n", req->num_regions);
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = ckpt_selective(req);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (tot_size < 0)
            fprintf(stderr, "selective ckpt FAILED, status=%d (%s)\n", g_op_status, strerror(g_op_status));
        else
            fprintf(stderr, "selective ckpt size: %f GB, time: %ld ms, bw: %f GB/s\n",
                   tot_size / 1024 / 1024 / 1024, duration.count(),
                   duration.count() > 0 ? tot_size / duration.count() * 1000 / 1024 / 1024 / 1024 : 0.0);
        FinishOp(scomm);
    } else if(msg == SELECTIVE_RESTORE_MSG) {
        ShareMemComm* scomm = static_cast<ShareMemComm*>(comm);
        const SelectiveCrRequest* req = &scomm->control->selective_req;
        g_op_status = 0;
        fprintf(stderr, "start selective restore...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = restore_ptr_and_content_selective(req);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (tot_size < 0)
            fprintf(stderr, "selective restore FAILED, status=%d (%s)\n", g_op_status, strerror(g_op_status));
        else
            fprintf(stderr, "selective restore size: %f GB, time: %ld ms, bw: %f GB/s\n",
                   tot_size / 1024 / 1024 / 1024, duration.count(),
                   duration.count() > 0 ? tot_size / duration.count() * 1000 / 1024 / 1024 / 1024 : 0.0);
        FinishOp(scomm);
    } else if(msg == CKPT_MSG) {
        g_op_status = 0;
        fprintf(stderr, "waiting for kernels to finish...\n");
        gpu->syncAllKernels();
        fprintf(stderr, "start ckpt...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = ckpt();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (tot_size < 0) {
            // Clean pre-op failure: nothing dumped, nothing released —
            // the workload keeps running. P2P must NOT be disabled and
            // cr_client must NOT proceed to cuda-checkpoint --toggle.
            fprintf(stderr, "ckpt FAILED, status=%d (%s)\n", g_op_status, strerror(g_op_status));
        } else {
            fprintf(stderr, "ckpt size: %f GB, time: %ld ms, bw: %f GB/s\n",
                   tot_size / 1024 / 1024 / 1024, duration.count(),
                   tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);

            // Disable P2P peer access before cuda-checkpoint freeze.
            // P2P access creates driver-level state that cuda-checkpoint cannot restore.
            // This must happen AFTER ckpt() (data is saved) and BEFORE cuda-checkpoint runs.
#if !defined(__HIP_PLATFORM_AMD__)
            fprintf(stderr, "[vGPU] Disabling P2P peer access for cuda-checkpoint...\n");
            ipc_disable_all_peer_access();
#endif
            // Note: External checkpoint (cuda-checkpoint for NVIDIA, CRIU for AMD)
            // is called from cr_client, not here
        }
        FinishOp(static_cast<ShareMemComm*>(comm));
    } else if (msg == RESTORE_MSG) {
        g_op_status = 0;
        // Note: cuda-checkpoint restore was already called by cr_client before this signal
        fprintf(stderr, "start restore...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = restore_ptr_and_content();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (tot_size < 0) {
            fprintf(stderr, "restore FAILED, status=%d (%s)\n", g_op_status, strerror(g_op_status));
        } else {
            fprintf(stderr, "restore size: %f GB, time: %ld ms, bw: %f GB/s\n",
                   tot_size / 1024 / 1024 / 1024, duration.count(),
                   tot_size / duration.count() * 1000 / 1024 / 1024 / 1024);

            // Re-enable P2P peer access after data restore
#if !defined(__HIP_PLATFORM_AMD__)
            fprintf(stderr, "[vGPU] Re-enabling P2P peer access after restore...\n");
            ipc_reenable_all_peer_access();
#endif
            fprintf(stderr, "finish restore\n");
        }
        FinishOp(static_cast<ShareMemComm*>(comm));
    }
    if (context_pushed) gpu->popContext();
    comm->send_msg(FINISH_MSG);
}

// ---------------------------------------------------------------------------
// IPC teardown/rebuild signal handler (for multi-GPU checkpoint/restore)
// Replaces the old NCCL suspend/resume handler — no NCCL source mods needed.
// ---------------------------------------------------------------------------
void cr_ipc_signal_handler(int signum) {
    fprintf(stderr, "[vGPU-IPC] Received signal %d (PID=%d)\n", signum, getpid());
    fflush(stderr);

    if (!CR_initialized) {
        fprintf(stderr, "[vGPU-IPC] CR not initialized, initializing now...\n");
        init_CR();
    }

    uint32_t msg = comm->recv_msg();

    if (msg == IPC_TEARDOWN_MSG) {
        // === Checkpoint Phase 1: Teardown IPC state ===
        fprintf(stderr, "[vGPU-IPC] === IPC Teardown Phase === (imports=%d, exports=%d, events=%d)\n",
                ipc_get_import_count(), ipc_get_export_count(), ipc_get_event_count());
        fflush(stderr);

        auto t_phase_start = std::chrono::high_resolution_clock::now();

        // GPU sync
        fprintf(stderr, "[vGPU-IPC] Synchronizing GPU (waiting for in-flight kernels)...\n");
        gpu->syncAllKernels();
        auto t_sync = std::chrono::high_resolution_clock::now();
        auto sync_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_sync - t_phase_start).count();
        fprintf(stderr, "[vGPU-IPC] GPU synchronized (%ld ms)\n", sync_ms);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Diagnostic: dump IPC state and nvidia fds BEFORE teardown
        ipc_dump_state();
        ipc_dump_nvidia_fds("BEFORE teardown");

        // Teardown all imported IPC mappings (cuMemUnmap + cuMemRelease)
        auto t_imports = std::chrono::high_resolution_clock::now();
        int torn = ipc_teardown_all_imports();
        auto t_imports_end = std::chrono::high_resolution_clock::now();
        auto imports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_imports_end - t_imports).count();
        fprintf(stderr, "[vGPU-IPC] Torn down %d IPC imports (%ld ms)\n", torn, imports_ms);

        // Save export GPU data to host buffer, then fully teardown exports
        size_t export_data_needed = ipc_get_export_data_size();
        fprintf(stderr, "[vGPU-IPC] Export data size needed: %zu bytes\n", export_data_needed);

        if (export_data_needed > 0) {
            if (g_ipc_export_data_buf) {
                munmap(g_ipc_export_data_buf, g_ipc_export_data_size);
                g_ipc_export_data_buf = nullptr;
            }
            g_ipc_export_data_size = export_data_needed;
            g_ipc_export_data_buf = mmap(nullptr, g_ipc_export_data_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_ipc_export_data_buf == MAP_FAILED) {
                fprintf(stderr, "[vGPU-IPC] ERROR: mmap for export data buffer failed\n");
                g_ipc_export_data_buf = nullptr;
                g_ipc_export_data_size = 0;
            }
        }

        auto t_exports = std::chrono::high_resolution_clock::now();
        int export_torn = 0;
        if (g_ipc_export_data_buf && g_ipc_export_data_size > 0) {
            export_torn = ipc_save_and_teardown_all_exports(
                g_ipc_export_data_buf, g_ipc_export_data_size);
            fprintf(stderr, "[vGPU-IPC] Export save+teardown: %d exports processed\n", export_torn);
        } else if (export_data_needed == 0) {
            fprintf(stderr, "[vGPU-IPC] No export data to save (0 mapped exports)\n");
        }
        auto t_exports_end = std::chrono::high_resolution_clock::now();
        auto exports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_exports_end - t_exports).count();
        fprintf(stderr, "[vGPU-IPC] Export teardown total: %ld ms\n", exports_ms);

        // Teardown IPC events
        auto t_events = std::chrono::high_resolution_clock::now();
        int events_torn = ipc_teardown_all_events();
        auto t_events_end = std::chrono::high_resolution_clock::now();
        auto events_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_events_end - t_events).count();
        fprintf(stderr, "[vGPU-IPC] IPC events torn down: %d (%ld ms)\n", events_torn, events_ms);

        // Teardown non-exported cuMem allocs
        size_t local_alloc_needed = ipc_get_local_alloc_data_size();
        fprintf(stderr, "[vGPU-IPC] Local cuMem alloc data size: %zu bytes\n", local_alloc_needed);

        if (local_alloc_needed > 0) {
            if (g_local_alloc_data_buf) {
                munmap(g_local_alloc_data_buf, g_local_alloc_data_size);
                g_local_alloc_data_buf = nullptr;
            }
            g_local_alloc_data_size = local_alloc_needed;
            g_local_alloc_data_buf = mmap(nullptr, g_local_alloc_data_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_local_alloc_data_buf == MAP_FAILED) {
                fprintf(stderr, "[vGPU-IPC] ERROR: mmap for local alloc buffer failed\n");
                g_local_alloc_data_buf = nullptr;
                g_local_alloc_data_size = 0;
            }
        }

        auto t_local = std::chrono::high_resolution_clock::now();
        if (g_local_alloc_data_buf && g_local_alloc_data_size > 0) {
            int local_torn = ipc_save_and_teardown_local_allocs(
                g_local_alloc_data_buf, g_local_alloc_data_size);
            fprintf(stderr, "[vGPU-IPC] Local alloc save+teardown: %d allocs processed\n", local_torn);
        } else if (local_alloc_needed == 0) {
            fprintf(stderr, "[vGPU-IPC] No local cuMem allocs to teardown\n");
        }
        auto t_local_end = std::chrono::high_resolution_clock::now();
        auto local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local).count();
        fprintf(stderr, "[vGPU-IPC] Local alloc teardown total: %ld ms\n", local_ms);

        // Diagnostic: dump nvidia fds AFTER teardown
        ipc_dump_nvidia_fds("AFTER teardown");

        // Disable P2P peer access
#if !defined(__HIP_PLATFORM_AMD__)
        auto t_p2p = std::chrono::high_resolution_clock::now();
        ipc_disable_all_peer_access();
        auto t_p2p_end = std::chrono::high_resolution_clock::now();
        auto p2p_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_p2p_end - t_p2p).count();
        fprintf(stderr, "[vGPU-IPC] P2P peer access disabled (%ld ms)\n", p2p_ms);
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_phase_start).count();
        fprintf(stderr, "[vGPU-IPC] IPC teardown completed in %ld ms (excl. GPU sync)\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Teardown Timing Summary: GPU-sync=%ld, Imports=%ld, Exports=%ld, Events=%ld, LocalAllocs=%ld, Total=%ld ms ===\n",
                sync_ms, imports_ms, exports_ms, events_ms, local_ms, total_ms);

    } else if (msg == IPC_EXPORT_MSG) {
        // === Restore Phase 3a: Re-export and publish handle info ===
        fprintf(stderr, "[vGPU-IPC] === IPC Re-export Phase ===\n");
        fflush(stderr);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Rebuild export allocations at original VAs, restore GPU data, re-export
        auto t_exports = std::chrono::high_resolution_clock::now();
        int rebuilt = 0;
        if (g_ipc_export_data_buf && g_ipc_export_data_size > 0) {
            rebuilt = ipc_rebuild_and_restore_all_exports(
                g_ipc_export_data_buf, g_ipc_export_data_size);
            fprintf(stderr, "[vGPU-IPC] Rebuilt+restored %d export allocations\n", rebuilt);

            munmap(g_ipc_export_data_buf, g_ipc_export_data_size);
            g_ipc_export_data_buf = nullptr;
            g_ipc_export_data_size = 0;
        } else {
            fprintf(stderr, "[vGPU-IPC] No export data to restore (buffer empty)\n");
        }
        auto t_exports_end = std::chrono::high_resolution_clock::now();
        auto exports_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_exports_end - t_exports).count();

        // Rebuild non-exported cuMem allocs
        auto t_local = std::chrono::high_resolution_clock::now();
        if (g_local_alloc_data_buf && g_local_alloc_data_size > 0) {
            int local_rebuilt = ipc_rebuild_local_allocs(
                g_local_alloc_data_buf, g_local_alloc_data_size);
            fprintf(stderr, "[vGPU-IPC] Rebuilt %d local cuMem allocs\n", local_rebuilt);

            munmap(g_local_alloc_data_buf, g_local_alloc_data_size);
            g_local_alloc_data_buf = nullptr;
            g_local_alloc_data_size = 0;
        }
        auto t_local_end = std::chrono::high_resolution_clock::now();
        auto local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local).count();

        // Write export info to shared memory for peers to read.
        // Deferred mode: this materializes the buffer at the floor size
        // (IPC verbs are buffer-path ops).
        auto t_shm = std::chrono::high_resolution_clock::now();
        void* tmp_buf = backend->get_tmp_buf();
        if (!tmp_buf) {
            fprintf(stderr, "[vGPU-IPC] ERROR: dump buffer unavailable for export blocks; aborting phase\n");
            comm->send_msg(FINISH_MSG);
            fflush(stderr);
            return;
        }
        shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
        IpcRebuildShmBlock* my_block = (IpcRebuildShmBlock*)((char*)tmp_buf +
            ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock));
        ipc_write_export_info_to_shm(my_block);
        auto t_shm_end = std::chrono::high_resolution_clock::now();
        auto shm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_shm_end - t_shm).count();

        // Start UDS fd server
        auto t_uds = std::chrono::high_resolution_clock::now();
        uds_fd_server_start();
        auto t_uds_end = std::chrono::high_resolution_clock::now();
        auto uds_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_uds_end - t_uds).count();

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        fprintf(stderr, "[vGPU-IPC] IPC re-export completed in %ld ms\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Re-export Timing Summary: Exports=%ld, LocalAllocs=%ld, SHM-write=%ld, UDS-server=%ld, Total=%ld ms ===\n",
                exports_ms, local_ms, shm_ms, uds_ms, ms);

    } else if (msg == IPC_IMPORT_MSG) {
        // === Restore Phase 3b: Import from peers ===
        fprintf(stderr, "[vGPU-IPC] === IPC Re-import Phase ===\n");
        fflush(stderr);

        auto t0 = std::chrono::high_resolution_clock::now();

        void* tmp_buf = backend->get_tmp_buf();
        if (!tmp_buf) {
            fprintf(stderr, "[vGPU-IPC] ERROR: dump buffer unavailable for import blocks; aborting phase\n");
            comm->send_msg(FINISH_MSG);
            fflush(stderr);
            return;
        }
        IpcRebuildShmBlock* peer_block = (IpcRebuildShmBlock*)((char*)tmp_buf +
            ROUND_UP_2MB(sizeof(shared_mem_fs)) - sizeof(IpcRebuildShmBlock) * 2);

        auto t_import = std::chrono::high_resolution_clock::now();
        if (peer_block->num_exports > 0) {
            int imported = ipc_import_from_shm_block(peer_block);
            fprintf(stderr, "[vGPU-IPC] Imported %d mappings from peers\n", imported);
        } else {
            fprintf(stderr, "[vGPU-IPC] No peer exports to import\n");
        }
        auto t_import_end = std::chrono::high_resolution_clock::now();
        auto import_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_import_end - t_import).count();

        // Stop UDS fd server
        auto t_uds_stop = std::chrono::high_resolution_clock::now();
        uds_fd_server_stop();
        auto t_uds_stop_end = std::chrono::high_resolution_clock::now();
        auto uds_stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_uds_stop_end - t_uds_stop).count();

        // Validate all IPC mappings after rebuild
        auto t_validate = std::chrono::high_resolution_clock::now();
        ipc_validate_all_mappings("AFTER import rebuild");
        auto t_validate_end = std::chrono::high_resolution_clock::now();
        auto validate_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_validate_end - t_validate).count();

        // Re-enable P2P peer access
#if !defined(__HIP_PLATFORM_AMD__)
        auto t_p2p = std::chrono::high_resolution_clock::now();
        ipc_reenable_all_peer_access();
        auto t_p2p_end = std::chrono::high_resolution_clock::now();
        auto p2p_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_p2p_end - t_p2p).count();
#else
        long p2p_ms = 0;
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        fprintf(stderr, "[vGPU-IPC] IPC re-import completed in %ld ms\n", ms);
        fprintf(stderr, "[vGPU-IPC] === Re-import Timing Summary: Import=%ld, UDS-stop=%ld, Validate=%ld, P2P=%ld, Total=%ld ms ===\n",
                import_ms, uds_stop_ms, validate_ms, p2p_ms, ms);

    } else {
        fprintf(stderr, "[vGPU-IPC] WARNING: unexpected message %u\n", msg);
    }

    comm->send_msg(FINISH_MSG);
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// Library constructor: register all signal handlers
// ---------------------------------------------------------------------------
__attribute__((constructor)) void init() {
    // Resolve buffer config FIRST — a function-local-static
    // singleton invoked here (not a second ELF constructor, whose order vs
    // this one would be unspecified). Signal handlers only read the cache.
    const gpu_cr::BufConfig& buf_cfg = gpu_cr::Config();

    fprintf(stderr, "[vGPU] Library loaded! Registering signal handlers...\n");
    fprintf(stderr, "[vGPU] Multi-GPU CR support enabled (IPC hook mode)\n");
    fflush(stderr);

    // Original single-GPU signals
    signal(CR_INIT_SIGNAL, cr_signal_handler);
    signal(CR_CKPT_SIGNAL, cr_signal_handler);
    signal(CR_RESTORE_SIGNAL, cr_signal_handler);

    // Multi-GPU IPC teardown/rebuild signals (replaces NCCL suspend/resume)
    signal(CR_IPC_TEARDOWN_SIGNAL, cr_ipc_signal_handler);
    signal(CR_IPC_REBUILD_SIGNAL, cr_ipc_signal_handler);

    // Diagnostic: validate all IPC mappings on demand
    signal(CR_IPC_VALIDATE_SIGNAL, [](int) {
        ipc_validate_all_mappings("ON-DEMAND");
        fflush(stderr);
    });

    // Readiness advertisement: written LAST, in the same constructor that
    // installs the handlers, so its existence proves every handler above
    // is in place before any coordinator can see the file — cr_client
    // refuses to kill() without it. starttime makes the file
    // self-invalidating across PID reuse. Never written in legacy mode (a
    // disk-backed dir could be shadowed by a later tmpfs mount, stranding
    // a stale advertisement). Failures are logged, never fatal: this must
    // not take down the workload.
    {
        bool ctl_mode = false;
        const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
        if (ctl_mode) {
            // shm_mb/staging_mb/deferred: additive keys so the
            // agent can OBSERVE workload buffer sizing and cross-check it
            // against pod hugepage requests — closing the observability
            // corner of the three-way consistency triangle.
            if (gpu_cr::WriteAdvertisement(ctl_dir, getpid(), buf_cfg.shm_size >> 20,
                                           buf_cfg.staging_size >> 20, buf_cfg.shm_deferred))
                fprintf(stderr, "[vGPU] ctl-ready advertisement written: %s/ctl-ready-%d\n",
                        ctl_dir, getpid());
        }
    }
}
