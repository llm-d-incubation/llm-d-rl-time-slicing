#include <atomic>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
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

// Helper function: multi-threaded memcpy
void memcpy_multi(void* dest, void* src, size_t size) {
    std::vector<std::thread> threads;
    size_t chunk_size = (size + NUM_COPY_THREADS - 1) / NUM_COPY_THREADS;
    for (int i = 0; i < NUM_COPY_THREADS; i++) {
        size_t offset = i * chunk_size;
        if (offset >= size) break;
        size_t this_chunk_size = std::min(chunk_size, size - offset);
        threads.emplace_back([=]() {
            memcpy((char*)dest + offset, (char*)src + offset, this_chunk_size);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
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
    shared_mem_fs* fs = (shared_mem_fs*)tmp_buf;
    int current_buf = 0;
    size_t buf_offset = 0;
    size_t des_offset = ROUND_UP_2MB(sizeof(shared_mem_fs));
    
    fs_mutex.lock();
    
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

double restore_ptr_and_content() {
    double tot_size = 0;
    
    long remap_time = 0, cpu_copy_time = 0, sync_time = 0;
    
    void* tmp_buf = backend->get_tmp_buf();
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

int get_id() {
    char id_name[512];
    const char* ctl_dir = std::getenv("EXPORT_FILE_PATH");
    if (!ctl_dir) ctl_dir = "/mnt/huge-ckpt";
    snprintf(id_name, sizeof(id_name), "%s/control", ctl_dir);
    int fd_id = open(id_name, O_CREAT | O_RDWR, 0755);
    if (fd_id < 0) {
        perror("open()");
        exit(EXIT_FAILURE);
    }
    // Set file size before mmap to avoid Bus error
    if (ftruncate(fd_id, HUGE_PAGE_SIZE) < 0) {
        perror("ftruncate()");
        exit(EXIT_FAILURE);
    }
    std::atomic<int>* id_ptr = (std::atomic<int>*)mmap(NULL, HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_id, 0);
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
    comm = new ShareMemComm(getpid());
    comm->setup();
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
    if(msg == CKPT_MSG) {
        fprintf(stderr, "waiting for kernels to finish...\n");
        gpu->syncAllKernels();
        fprintf(stderr, "start ckpt...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = ckpt();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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
    } else if (msg == RESTORE_MSG) {
        // Note: cuda-checkpoint restore was already called by cr_client before this signal
        fprintf(stderr, "start restore...\n");
        auto start = std::chrono::high_resolution_clock::now();
        double tot_size = restore_ptr_and_content();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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

        // Write export info to shared memory for peers to read
        auto t_shm = std::chrono::high_resolution_clock::now();
        void* tmp_buf = backend->get_tmp_buf();
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
    gpu_cr::Config();

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
}
