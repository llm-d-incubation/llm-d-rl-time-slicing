#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <assert.h>
#include <chrono>
#include <limits.h>
#include <libgen.h>
#include <string.h>
#include <string>

#ifdef __HIP_PLATFORM_AMD__
// AMD platform
#else
#include <cuda.h>
#endif

#include "common.h"
#include "ctl_path.h"
#include "dump_format.h"
#include "selective_spec.h"
#include "comm/comm.h"

namespace {

// Exit codes (consumed by the snapshot agent):
//   0 OK, 1 usage/parse, 2 op failed (op_status or invalid dump),
//   3 refused pre-signal (.so not ready / no advertisement, or target
//   dead), 4 timeout.
// Exit 4 poisons the channel: the flock releases at exit while the
// wedged handler may still be mid-op, so the workload must be restarted
// (or proven idle) before another op targets this PID.
constexpr int kExitOpFailed = 2;
constexpr int kExitRefused = 3;
constexpr int kExitTimeout = 4;

}  // namespace

std::string get_cuda_checkpoint_path() {
    // Deployment override (also used by the GPU-free integration tests):
    // point at the cuda-checkpoint binary explicitly instead of relying on
    // the layout-relative lookup below.
    const char* env_path = getenv("GPU_CR_CUDA_CHECKPOINT");
    if (env_path && env_path[0]) return env_path;

    char exe_path[1024];
    ssize_t count = readlink("/proc/self/exe", exe_path, 1024);
    if (count == -1) {
        perror("readlink");
        return "cuda-checkpoint";
    }
    exe_path[count] = '\0';

    char* dir = dirname(exe_path);
    std::string full_path = std::string(dir) + "/../cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint";

    if (access(full_path.c_str(), X_OK) != 0) {
        fprintf(stderr, "WARNING: helper binary not found at: %s\n", full_path.c_str());
        return "cuda-checkpoint";
    }

    return full_path;
}

namespace {

// Runs "<bin> --toggle --pid <pid>" without a shell: cr_client ships in
// distroless agent images where system() has no /bin/sh and always fails
// with 127. execlp keeps the PATH lookup for a bare "cuda-checkpoint".
// Returns the command's exit status (127 if it could not be executed,
// 128+signal if it died on one), or -1 if it could not be spawned/reaped —
// mirroring system()'s <0 / status split the callers already handle.
int RunCudaCheckpointToggle(const std::string& bin_path, pid_t target_pid) {
    std::string pid_str = std::to_string(target_pid);
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        execlp(bin_path.c_str(), bin_path.c_str(), "--toggle", "--pid",
               pid_str.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

// kill() with the result checked: a target that died between the pre-op
// gates and the signal should fail crisply here, not burn the whole
// FINISH timeout on a signal that never landed.
void SignalOrDie(pid_t target_pid, int sig) {
    if (kill(target_pid, sig) != 0) {
        fprintf(stderr, "Error: kill(%d, %d): %s\n", target_pid, sig, strerror(errno));
        exit(kExitRefused);
    }
}

// Pre-creates the destination file empty (O_CREAT|O_TRUNC), never sized —
// the .so alone can compute the dump total (preloader-authoritative
// sizing), and an inode costs no hugepages/blocks in this cgroup. The
// truncate matters: a recycled path holding an older valid dump could
// otherwise false-validate a torn write whenever the new commit-marker
// offset landed on old payload bytes that happen to spell the magic.
//
// Ownership: the dump is written by the TARGET, not by this client — the
// .so opens dest_path O_RDWR, never O_CREAT. cr_client runs as root in
// the agent pod while the workload is commonly non-root, so a root-owned
// file would turn every checkpoint into EACCES inside the target. The
// inode is chown'd to the target's uid/gid (stat of /proc/<pid>, which
// needs no cross-pod UID contract) rather than opened 0666: the store
// dir is shared, and dump validation proves a commit marker, not
// authorship — world-writable would let any co-mounted uid plant a
// marker-valid dump that a later restore trusts. (A non-dumpable target
// stats as root:root and keeps today's root-owned behavior.)
bool SecurePrecreate(const char* path, pid_t target_pid) {
    if (path[0] != '/') {
        fprintf(stderr, "Error: -o path must be absolute: %s\n", path);
        return false;
    }
    char dir_buf[gpu_cr::kSelectiveCrMaxPath];
    strncpy(dir_buf, path, sizeof(dir_buf) - 1);
    dir_buf[sizeof(dir_buf) - 1] = '\0';
    char* slash = strrchr(dir_buf, '/');
    const char* base = slash + 1;
    if (*base == '\0') {
        fprintf(stderr, "Error: -o path is a directory: %s\n", path);
        return false;
    }
    *slash = '\0';
    const char* dir = dir_buf[0] ? dir_buf : "/";

    int dfd = open(dir, O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        fprintf(stderr, "Error: cannot open destination dir %s: %s\n", dir, strerror(errno));
        return false;
    }
    // Symlink hardening, trailing component only: cr_client runs as root
    // in the agent pod against a workload-writable store, so a symlink
    // swap of the final component must not let it create/truncate
    // arbitrary host files. `base` is a single slash-free component
    // resolved relative to the already-opened parent, so O_NOFOLLOW makes
    // a trailing symlink fail with ELOOP (dangling included; O_CREAT does
    // not create through one). Intermediate components of `dir` are NOT
    // defended — the agent passes store paths whose parents it owns; a
    // workload-writable intermediate would need
    // openat2(RESOLVE_NO_SYMLINKS) to close.
    int fd = openat(dfd, base,
                    O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0644);
    close(dfd);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot create destination %s: %s\n", path, strerror(errno));
        return false;
    }
    char proc_path[64];
    snprintf(proc_path, sizeof(proc_path), "/proc/%d", target_pid);
    struct stat pst;
    if (stat(proc_path, &pst) != 0) {
        fprintf(stderr, "Error: cannot stat %s: %s\n", proc_path, strerror(errno));
        close(fd);
        return false;
    }
    // fchmod after the fact, not via the open mode: the mode must not be
    // umask-filtered, and 0600 is enough — the writer owns the file, and
    // the validating reopen runs as root (test rigs run both sides as
    // one uid, which owner-rw also covers).
    if (fchown(fd, pst.st_uid, pst.st_gid) != 0 || fchmod(fd, 0600) != 0) {
        fprintf(stderr, "Error: cannot hand %s to uid %u: %s\n", path,
                (unsigned)pst.st_uid, strerror(errno));
        close(fd);
        return false;
    }
    close(fd);
    return true;
}

// Post-checkpoint validation of a destination dump: header
// plausibility + trailing commit marker, via read(2) — no mmap, so no
// hugetlb reservation or fault lands in this process's cgroup.
bool ValidateDestDump(const char* path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot reopen destination %s: %s\n", path, strerror(errno));
        return false;
    }
    bool ok = gpu_cr::ValidateDumpFd(fd);
    close(fd);
    if (!ok)
        fprintf(stderr, "Error: destination %s failed dump validation (torn or unserved dump)\n", path);
    return ok;
}

// Pre-signal gate: refuse to kill() unless the target's .so wrote
// a readiness advertisement into OUR ctl dir. Presence in this dir proves
// shared backing (the only way the file got here); proto proves the
// library level; starttime kills the PID-reuse case, where the signal
// would terminate an innocent recycled PID; the file's owner uid must
// match the target process's uid — on a shared ctl dir a co-tenant under
// another uid cannot vouch for a victim PID (and the sticky bit keeps it
// from replacing the victim's real advert). Path equality is a warning
// only (same tmpfs may be mounted at different container paths).
bool CheckAdvertisement(const char* ctl_dir, int pid) {
    char path[512];
    snprintf(path, sizeof(path), "%s/ctl-ready-%d", ctl_dir, pid);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no readiness advertisement at %s — vGPU.so not loaded in PID %d, "
                        "GPU_CR_CTL_PATH mismatch, or a library without ctl support\n", path, pid);
        return false;
    }
    // fstat the OPEN file (no rename/replace TOCTOU) against /proc/<pid>,
    // whose st_uid is the target's effective uid.
    struct stat adv_st, proc_st;
    char proc_path[64];
    snprintf(proc_path, sizeof(proc_path), "/proc/%d", pid);
    if (fstat(fileno(f), &adv_st) != 0 || stat(proc_path, &proc_st) != 0) {
        fclose(f);
        fprintf(stderr, "Error: cannot stat advertisement %s or %s (%s)\n",
                path, proc_path, strerror(errno));
        return false;
    }
    if (adv_st.st_uid != proc_st.st_uid) {
        fclose(f);
        fprintf(stderr, "Error: advertisement %s owned by uid %u but PID %d runs as uid %u — "
                        "forged or stale advert, refusing to signal\n",
                path, static_cast<unsigned>(adv_st.st_uid), pid,
                static_cast<unsigned>(proc_st.st_uid));
        return false;
    }
    char buf[600] = "";
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        fprintf(stderr, "Error: empty advertisement %s\n", path);
        return false;
    }
    fclose(f);
    gpu_cr::Advertisement adv;
    if (!gpu_cr::ParseAdvertisement(buf, &adv)) {
        fprintf(stderr, "Error: unparsable advertisement %s\n", path);
        return false;
    }
    if (adv.proto < gpu_cr::kCtlProto) {
        fprintf(stderr, "Error: advertisement proto=%d too low (need >=%d)\n", adv.proto, gpu_cr::kCtlProto);
        return false;
    }
    long long cur_start = gpu_cr::ProcStarttime(pid);
    if (cur_start < 0) {
        fprintf(stderr, "Error: PID %d no longer exists\n", pid);
        return false;
    }
    if (adv.starttime != cur_start) {
        fprintf(stderr, "Error: starttime mismatch for PID %d (advertised %lld, current %lld) — "
                        "PID reuse, refusing to signal\n", pid, adv.starttime, cur_start);
        return false;
    }
    if (adv.ctl[0] && strcmp(adv.ctl, ctl_dir) != 0)
        fprintf(stderr, "Warning: advertisement names ctl=%s, ours is %s (same backing store, "
                        "different mount points?)\n", adv.ctl, ctl_dir);
    return true;
}

// Polls for FINISH with a deadline: a dead or wedged workload must fail this
// invocation, not hang it (and the agent behind it) forever.
bool WaitFinished(ShareMemComm* comm) {
    long timeout_sec = 120;
    const char* t = getenv("GPU_CR_OP_TIMEOUT_SEC");
    if (t) {
        long v = atol(t);
        if (v > 0) timeout_sec = v;
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (!comm->is_finished()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fprintf(stderr, "Error: timed out after %lds waiting for FINISH\n", timeout_sec);
            return false;
        }
        usleep(1000);
    }
    return true;
}

// After a FULL ckpt/restore: the .so reports op_status at FINISH (clean
// failures: oversized checkpoint, deferred-buffer ENOMEM). On checkpoint
// failure we must NOT proceed to cuda-checkpoint --toggle — freezing a
// process whose state was never saved is unrecoverable; on restore
// failure we must not unfreeze onto stale weights. op_status == 0 means
// no result was reported (a .so that predates status reporting, or a
// FINISH raced by a crash): full ops keep historical tolerance.
void CheckFullResult(ShareMemComm* comm, const char* what) {
    int32_t st = comm->control->op_status;
    if (st < 0) {
        fprintf(stderr, "Error: vGPU.so reported %s failure: %s (status=%d); NOT toggling cuda-checkpoint\n",
                what, strerror(-st), st);
        exit(kExitOpFailed);
    }
}

// After a selective op: surface the .so-reported status. Negative is a
// clean op failure. Zero means the op never reached FINISH bookkeeping
// (the client zeroed the word before signaling) — for selective ops that
// is out of contract, so fail rather than trust the dump.
void CheckSelectiveResult(ShareMemComm* comm) {
    int32_t st = comm->control->op_status;
    if (st < 0) {
        fprintf(stderr, "Error: vGPU.so reported op failure: %s (status=%d)\n", strerror(-st), st);
        exit(kExitOpFailed);
    }
    if (st == 0) {
        fprintf(stderr, "Error: vGPU.so reported no op result\n");
        exit(kExitOpFailed);
    }
}

}  // namespace


int main(int argc, char* argv[]) {
    int opt;
    int init = 0;
    int ckpt = 0;
    int restore = 0;
    int dump = 0;
    int pid = 0;
    int criu_pid = 0;
    int buffer_only = 0;
    const char* selective_spec = nullptr;
    const char* dest_path = nullptr;
    while ((opt = getopt(argc, argv, "icrdbp:m:s:o:")) != -1) {
        switch (opt) {
            case 'i':
                init = 1;
                break;
            case 'c':
                ckpt = 1;
                break;
            case 'r':
                restore = 1;
                break;
            case 'p':
                pid = atoi(optarg);
                break;
            case 'm':
                criu_pid = atoi(optarg);
                break;
            case 'b':
                buffer_only = 1;
                break;
            case 's':
                selective_spec = optarg;
                break;
            case 'o':
                dest_path = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-i|-c|-r] -p pid [-m criu_pid] [-b] [-s ptr:size,...] [-o /path/to/dump]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if(!ckpt && !restore && !dump && !init) {
        fprintf(stderr, "Either -i, -c, or -r must be specified\n");
        exit(EXIT_FAILURE);
    }
    if(ckpt + restore + dump + init > 1) {
        fprintf(stderr, "Only one of -i, -c, or -r  can be specified\n");
        exit(EXIT_FAILURE);
    }
    if (selective_spec && !ckpt && !restore) {
        fprintf(stderr, "Error: -s requires -c or -r\n");
        exit(EXIT_FAILURE);
    }
    if (dest_path) {
        if (!selective_spec) {
            fprintf(stderr, "Error: -o requires -c or -r together with -s\n");
            exit(EXIT_FAILURE);
        }
        if (strlen(dest_path) >= gpu_cr::kSelectiveCrMaxPath) {
            fprintf(stderr, "Error: -o path exceeds %zu bytes\n", gpu_cr::kSelectiveCrMaxPath - 1);
            exit(EXIT_FAILURE);
        }
    }
    assert(pid != 0);
    if (criu_pid == 0) criu_pid = pid;

    // A configured-but-broken ctl path is refused HERE, loudly.
    // (The .so falls back to legacy instead — it must not die — so the
    // coordinator is where the misconfiguration surfaces.)
    bool ctl_mode = false;
    const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
    const char* ctl_env = getenv("GPU_CR_CTL_PATH");
    if (ctl_env && ctl_env[0] && !ctl_mode) {
        fprintf(stderr, "Error: GPU_CR_CTL_PATH=%s is missing or not tmpfs-backed "
                        "(mount-order shadowing?) — refusing to operate\n", ctl_env);
        exit(kExitRefused);
    }

    // Pre-signal gate: never kill() a PID whose .so hasn't
    // advertised readiness in our ctl dir — covers not-loaded, env
    // mismatch, older libraries and PID reuse in one check. The
    // gate-to-signal window is a narrow, accepted TOCTOU: the starttime
    // check excludes reuse, and SignalOrDie fails crisply if the target
    // vanishes inside it.
    if (ctl_mode && !CheckAdvertisement(ctl_dir, pid))
        exit(kExitRefused);

    ShareMemComm *comm = new ShareMemComm(pid);
    comm->setup();

    // Serialize concurrent cr_clients against the same PID for the whole
    // op: nothing else orders two writers of selective_req.
    // Fixed order, legacy lock first: the legacy file is
    // taken with open(O_CREAT)+flock only — an inode lock faults no
    // hugetlb pages, so it stays safe with a zero hugepages request.
    if (ctl_mode) {
        char legacy_lock_path[512];
        snprintf(legacy_lock_path, sizeof(legacy_lock_path), "%s/control-%d", gpu_cr::DataDir(), pid);
        int legacy_lock_fd = open(legacy_lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0644);
        if (legacy_lock_fd >= 0) {
            if (flock(legacy_lock_fd, LOCK_EX) != 0)
                fprintf(stderr, "Warning: flock(%s) failed: %s\n", legacy_lock_path, strerror(errno));
            // held (with the fd) until process exit
        }
    }
    if (flock(comm->fd_control, LOCK_EX) != 0)
        fprintf(stderr, "Warning: flock(control-%d) failed: %s\n", pid, strerror(errno));

    // Selective-op gate, BEFORE any signal: a .so that never published
    // selective_ready (init_CR not yet run, or a mismatched build) has no
    // armed handlers — a signaled op would just burn the FINISH timeout,
    // and a dest-path restore served from the per-PID buffer would replay
    // stale bytes into GPU memory, which no post-op check can undo.
    // Full-process ops stay ungated (historical behavior). In ctl mode
    // the advertisement already proved the library is armed, without
    // needing a prior -i.
    if (selective_spec && !ctl_mode &&
        comm->control->selective_ready != gpu_cr::kSelectiveReady) {
        fprintf(stderr, "Error: vGPU.so has not published selective support "
                        "(run -i first, or fix the preloaded library)\n");
        exit(kExitRefused);
    }

    // Consume any earlier op result before signaling: op_status semantics
    // are "0 = this op never reported", which is only true if the word
    // cannot carry a stale success from a previous op across a crash.
    // Control-word accesses here and in WaitFinished are plain loads and
    // stores, matching the pre-existing channel style: safe on the pinned
    // x86-64 target (TSO) with the opaque call boundaries as compiler
    // barriers, not portable to weaker memory models.
    comm->control->op_status = 0;

    int ret = 0;

    if(init) {
        comm->send_msg(INIT_MSG);
        SignalOrDie(pid, CR_INIT_SIGNAL);
        if (!WaitFinished(comm)) exit(kExitTimeout);
    } else if (ckpt && selective_spec) {
        SelectiveCrRequest req;
        memset(&req, 0, sizeof(req));
        if (!gpu_cr::ParseSelectiveRegions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective checkpoint: %u regions\n", req.num_regions);
        for (uint32_t i = 0; i < req.num_regions; i++) {
            printf("  region %u: ptr=%p size=%lu\n", i, req.regions[i].ptr, req.regions[i].size);
        }
        // Always write the whole request (zeroed dest when -o absent) so a
        // stale dest_path from an earlier op can never survive this write.
        if (dest_path) {
            if (!SecurePrecreate(dest_path, pid)) exit(kExitOpFailed);
            strncpy(req.dest_path, dest_path, gpu_cr::kSelectiveCrMaxPath - 1);
        }
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_CKPT_MSG);
        SignalOrDie(pid, CR_CKPT_SIGNAL);
        printf("Selective dump signal sent.\n");
        if (!WaitFinished(comm)) exit(kExitTimeout);
        CheckSelectiveResult(comm);
        if (dest_path && !ValidateDestDump(dest_path)) exit(kExitOpFailed);
        printf("Selective checkpointing done\n");
    } else if (restore && selective_spec) {
        SelectiveCrRequest req;
        memset(&req, 0, sizeof(req));
        if (!gpu_cr::ParseSelectiveRegions(selective_spec, &req)) {
            fprintf(stderr, "Error: failed to parse selective regions\n");
            exit(EXIT_FAILURE);
        }
        printf("Selective restore: %u regions\n", req.num_regions);
        if (dest_path) {
            struct stat st;
            if (stat(dest_path, &st) != 0) {
                fprintf(stderr, "Error: restore source %s: %s\n", dest_path, strerror(errno));
                exit(kExitOpFailed);
            }
            strncpy(req.dest_path, dest_path, gpu_cr::kSelectiveCrMaxPath - 1);
        }
        comm->control->selective_req = req;
        comm->send_msg(SELECTIVE_RESTORE_MSG);
        SignalOrDie(pid, CR_RESTORE_SIGNAL);
        printf("Selective restore signal sent.\n");
        if (!WaitFinished(comm)) exit(kExitTimeout);
        CheckSelectiveResult(comm);
        printf("Selective restore done\n");
    } else if(ckpt) {
        comm->send_msg(CKPT_MSG);
        SignalOrDie(pid, CR_CKPT_SIGNAL);
        printf("Dump signal sent.\n");
        if (!WaitFinished(comm)) exit(kExitTimeout);
        CheckFullResult(comm, "checkpoint");
        printf("Dumping done.\n");
#ifdef __HIP_PLATFORM_AMD__
        // For AMD: call CRIU to dump the process
        const char* ckpt_dir = getenv("AMD_CKPT_DIR");
        if (!ckpt_dir) {
            fprintf(stderr, "ERROR: AMD_CKPT_DIR environment variable not set!\n");
            fprintf(stderr, "Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
            exit(EXIT_FAILURE);
        }
        
        printf("AMD: Calling CRIU to checkpoint process %d\n", criu_pid);
        printf("Checkpoint directory: %s\n", ckpt_dir);
        
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
            "sudo env LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
            "criu dump --link-remap --tcp-established -t %d -D %s -j -v4 -o %s/dump.log --ghost-limit 50M --ext-unix-sk -L /usr/local/lib/criu",
            criu_pid, ckpt_dir, ckpt_dir);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd);
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("CRIU checkpoint time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
#else
        std::string bin_path = get_cuda_checkpoint_path();
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only) {
            ret = RunCudaCheckpointToggle(bin_path, pid);
            if (ret < 0) {
                perror("toggle spawn");
                exit(EXIT_FAILURE);
            }
            // A nonzero freeze exit means the process is still running on
            // the GPU: report op failure (the dump itself is valid) rather
            // than let the agent believe it parked.
            if (ret != 0) {
                fprintf(stderr, "Error: '%s --toggle --pid %d' exited with status %d — "
                        "dump is valid but the process was NOT frozen\n",
                        bin_path.c_str(), pid, ret);
                exit(kExitOpFailed);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("cuda-checkpoint checkpoint time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
#endif
        printf("Checkpointing done\n");
    } else if(restore) {
#ifdef __HIP_PLATFORM_AMD__
        const char* ckpt_dir = getenv("AMD_CKPT_DIR");
        if (!ckpt_dir) {
            fprintf(stderr, "ERROR: AMD_CKPT_DIR environment variable not set!\n");
            fprintf(stderr, "Please set: export AMD_CKPT_DIR=/path/to/checkpoint/dir\n");
            exit(EXIT_FAILURE);
        }
        
        printf("AMD: Calling CRIU to restore process\n");
        printf("Restore directory: %s\n", ckpt_dir);
        
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "sudo env  LD_LIBRARY_PATH=/opt/amdgpu/lib/x86_64-linux-gnu "
            "criu restore --tcp-established -D %s -j -v4 -o %s/restore.log -L /usr/local/lib/criu --pidfile %s/restored.pid --ghost-limit 50M --ext-unix-sk --restore-detached ",
            ckpt_dir, ckpt_dir, ckpt_dir);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only)
            ret = system(cmd);
        if (ret < 0) {
            perror("system()");
            exit(EXIT_FAILURE);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("CRIU restore time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
        printf("Calling GPU-CR\n");

        comm->send_msg(RESTORE_MSG);
        SignalOrDie(pid, CR_RESTORE_SIGNAL);

        if (!WaitFinished(comm)) exit(kExitTimeout);
        auto t2 = std::chrono::high_resolution_clock::now();
        printf("GPUos restore time: %.3f s\n", std::chrono::duration<double>(t2 - t1).count());

        printf("Process internal restoration finished.\n");
        printf("Restoring done\n");
#else
        // Upstream ordering: thaw first, then drive the in-process restore.
        // The restore handler runs inside the target and needs a live CUDA
        // context; signaling a still-checkpointed process would leave the
        // handler blocked on its first driver call.
        std::string bin_path = get_cuda_checkpoint_path();
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!buffer_only) {
            ret = RunCudaCheckpointToggle(bin_path, pid);
            if (ret < 0) {
                perror("toggle spawn");
                exit(EXIT_FAILURE);
            }
            // A failed thaw leaves the process checkpointed — abort before
            // kill() rather than wedge it mid-restore.
            if (ret != 0) {
                fprintf(stderr, "Error: '%s --toggle --pid %d' exited with status %d\n",
                        bin_path.c_str(), pid, ret);
                exit(kExitOpFailed);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("cuda-checkpoint restore time: %.3f s\n", std::chrono::duration<double>(t1 - t0).count());
        comm->send_msg(RESTORE_MSG);
        SignalOrDie(pid, CR_RESTORE_SIGNAL);
        if (!WaitFinished(comm)) exit(kExitTimeout);
        CheckFullResult(comm, "restore");
        printf("Restoring done\n");
#endif
    }
    return 0;
}
