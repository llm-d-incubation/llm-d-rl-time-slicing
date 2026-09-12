// GPU-free stand-in for the vGPU.so side of the control channel, used by
// the cr_client integration tests. It reuses the REAL product pieces —
// ShareMemComm, gpu_cr::Config, WriteAdvertisement, FinishOpControls,
// ValidateDumpFd — and fakes only the GPU work (dump extents are pattern
// bytes instead of VRAM).
//
// Behavior knobs (env):
//   FAKE_OP_STATUS   errno-style status to report at FINISH (default 0)
//   FAKE_SKIP_COMMIT write destination dumps WITHOUT the commit marker
//   FAKE_NO_FINISH   never send FINISH (drives the cr_client timeout path)
//   FAKE_NOT_READY   behave like a .so that never armed selective
//                    support: no selective_ready, no op_status, no
//                    consume-once, no dest-path
//   FAKE_READY_FILE  file created once the control channel is up
//   FAKE_LOG         per-op log: "op=<msg> dest=<path-or-empty>" lines

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "ctl_path.h"
#include "dump_format.h"
#include "comm/comm.h"

namespace {

ShareMemComm* g_comm = nullptr;
bool g_not_ready = false;
int g_fake_status = 0;

void LogOp(uint32_t msg, const char* dest) {
  const char* log = getenv("FAKE_LOG");
  if (!log) return;
  FILE* f = fopen(log, "a");
  if (!f) return;
  fprintf(f, "op=%u dest=%s\n", msg, dest ? dest : "");
  fclose(f);
}

// Writes a valid destination dump for the requested regions: header, one
// extent per region (pattern bytes), and the commit marker unless
// FAKE_SKIP_COMMIT is set.
bool WriteFakeDump(const SelectiveCrRequest* req, const char* dest_path) {
  FILE* f = fopen(dest_path, "r+");
  if (!f) {
    fprintf(stderr, "[fake-workload] cannot open %s: %s\n", dest_path,
            strerror(errno));
    return false;
  }
  uint64_t header_size = ROUND_UP_2MB(sizeof(shared_mem_fs));
  uint64_t current_offset = header_size;
  uint64_t hdr[2];
  hdr[0] = req->num_regions;
  for (uint32_t i = 0; i < req->num_regions; i++)
    current_offset += req->regions[i].size;
  hdr[1] = current_offset;

  fseek(f, 0, SEEK_SET);
  fwrite(hdr, sizeof(hdr), 1, f);
  uint64_t offset = header_size;
  for (uint32_t i = 0; i < req->num_regions; i++) {
    fseek(f, offset, SEEK_SET);
    for (uint64_t b = 0; b < req->regions[i].size; b++) fputc(0xAB, f);
    offset += req->regions[i].size;
  }
  uint64_t marker = gpu_cr::DumpCommitOffset(current_offset);
  if (!getenv("FAKE_SKIP_COMMIT")) {
    DumpCommit commit = {gpu_cr::kDumpCommitMagic, 1};
    fseek(f, marker, SEEK_SET);
    fwrite(&commit, sizeof(commit), 1, f);
  } else {
    // Still size the file past the marker so only the magic is missing.
    fseek(f, marker + sizeof(DumpCommit) - 1, SEEK_SET);
    fputc(0, f);
  }
  fclose(f);
  return true;
}

void Finish(int op_status) {
  if (!g_not_ready) gpu_cr::FinishOpControls(g_comm->control, op_status);
  if (!getenv("FAKE_NO_FINISH")) g_comm->send_msg(FINISH_MSG);
}

void SignalHandler(int signum) {
  uint32_t msg = g_comm->recv_msg();
  const SelectiveCrRequest* req = &g_comm->control->selective_req;
  const char* dest =
      (!g_not_ready && req->dest_path[0] != '\0') ? req->dest_path : nullptr;
  fprintf(stderr, "[fake-workload] signal %d msg %u dest %s\n", signum, msg,
          dest ? dest : "(buffer)");
  LogOp(msg, dest);

  int status = g_fake_status;
  switch (msg) {
    case INIT_MSG:
      if (!g_not_ready) g_comm->control->selective_ready = gpu_cr::kSelectiveReady;
      if (!getenv("FAKE_NO_FINISH")) g_comm->send_msg(FINISH_MSG);
      return;  // real init path predates op_status reporting
    case SELECTIVE_CKPT_MSG:
      if (dest && status == 0 && !WriteFakeDump(req, dest)) status = EIO;
      break;
    case SELECTIVE_RESTORE_MSG:
      // Mimic the .so's pre-restore gate with the REAL validator.
      if (dest && status == 0) {
        int fd = open(dest, O_RDONLY);
        if (fd < 0 || !gpu_cr::ValidateDumpFd(fd)) status = EINVAL;
        if (fd >= 0) close(fd);
      }
      break;
    case CKPT_MSG:
    case RESTORE_MSG:
      break;
    default:
      fprintf(stderr, "[fake-workload] unexpected msg %u\n", msg);
      status = EPROTO;
      break;
  }
  Finish(status);
}

}  // namespace

int main() {
  g_not_ready = getenv("FAKE_NOT_READY") != nullptr;
  const char* st = getenv("FAKE_OP_STATUS");
  if (st) g_fake_status = atoi(st);

  const gpu_cr::BufConfig& cfg = gpu_cr::Config();

  g_comm = new ShareMemComm(getpid());
  g_comm->setup();
  if (!g_not_ready) g_comm->control->selective_ready = gpu_cr::kSelectiveReady;

  bool ctl_mode = false;
  const char* ctl_dir = gpu_cr::CtlDir(&ctl_mode);
  if (ctl_mode && !g_not_ready) {
    gpu_cr::WriteAdvertisement(ctl_dir, getpid(), cfg.shm_size >> 20,
                               cfg.staging_size >> 20, cfg.shm_deferred);
  }

  signal(CR_INIT_SIGNAL, SignalHandler);
  signal(CR_CKPT_SIGNAL, SignalHandler);
  signal(CR_RESTORE_SIGNAL, SignalHandler);

  const char* ready = getenv("FAKE_READY_FILE");
  if (ready) {
    FILE* f = fopen(ready, "w");
    if (f) {
      fprintf(f, "%d\n", getpid());
      fclose(f);
    }
  }

  fprintf(stderr, "[fake-workload] ready, pid=%d ctl_mode=%d not_ready=%d\n",
          getpid(), ctl_mode ? 1 : 0, g_not_ready ? 1 : 0);
  for (;;) pause();
}
