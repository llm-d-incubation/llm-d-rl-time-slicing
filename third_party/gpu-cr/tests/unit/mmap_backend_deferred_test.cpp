// Deferred dump-buffer mode (GPU_CR_SHM_MB=0) of the ShareMem backend:
// setup() skips the dump buffer entirely, the first get_tmp_buf()
// materializes it at the 64MiB floor, and a failed materialization
// returns nullptr instead of killing the process. Own test binary:
// gpu_cr::Config() is a process-cached singleton, so deferred mode
// cannot coexist with the eager-mode suite in gpu_cr_unit_tests.
// Exercised through the file backend (EXPORT_FILE_PATH). Linux-only.

#include "backend/backend.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "common.h"
#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

// Pin deferred mode (and a small staging size — setup() still maps the
// host buffer eagerly) into the cached Config() singleton at static-init
// time, before any test-time Config() call could cache other values.
const bool kDeferredEnvPinned = [] {
  setenv("GPU_CR_SHM_MB", "0", 1);
  setenv("GPU_CR_STAGING_MB", "128", 1);
  return Config().shm_deferred && Config().shm_size == kShmFloorBytes &&
         Config().staging_size == (128UL << 20);
}();

class MmapBackendDeferredTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(kDeferredEnvPinned);
    strcpy(dir_, "/tmp/gpu-cr-mmap-deferred-XXXXXX");
    ASSERT_NE(mkdtemp(dir_), nullptr);
    setenv("EXPORT_FILE_PATH", dir_, 1);
  }

  void TearDown() override {
    unlink(DumpPath(kId).c_str());
    unlink(HostPath(kId).c_str());
    rmdir(dir_);
    unsetenv("EXPORT_FILE_PATH");
  }

  std::string DumpPath(int id) const {
    return std::string(dir_) + "/ckpt-" + std::to_string(id) + ".data";
  }
  std::string HostPath(int id) const {
    return std::string(dir_) + "/ckpt-" + std::to_string(id) + "-host.data";
  }

  static constexpr int kId = 21;
  char dir_[64];
};

// setup() must neither create nor map the dump buffer; the host staging
// buffer is still mapped eagerly.
TEST_F(MmapBackendDeferredTest, SetupSkipsDumpBuffer) {
  ShareMem backend(kId);
  backend.setup();

  EXPECT_EQ(backend.tmp_buf, nullptr);
  struct stat st;
  EXPECT_NE(stat(DumpPath(kId).c_str(), &st), 0);

  ASSERT_EQ(stat(HostPath(kId).c_str(), &st), 0);
  EXPECT_EQ(static_cast<size_t>(st.st_size),
            Config().staging_size * STAGING_BUF_NUM);
  EXPECT_NE(backend.get_host_buffer(), nullptr);
}

// The first buffer-path op materializes the buffer at the floor size
// with a freshly reset header; later calls hand out the same mapping.
TEST_F(MmapBackendDeferredTest, FirstGetTmpBufMaterializesAtFloor) {
  ShareMem backend(kId);
  backend.setup();

  void* buf = backend.get_tmp_buf();
  ASSERT_NE(buf, nullptr);

  struct stat st;
  ASSERT_EQ(stat(DumpPath(kId).c_str(), &st), 0);
  EXPECT_EQ(static_cast<size_t>(st.st_size), kShmFloorBytes);

  shared_mem_fs* fs = static_cast<shared_mem_fs*>(buf);
  EXPECT_EQ(fs->file_num, 0u);
  EXPECT_EQ(fs->current_offset, ROUND_UP_2MB(sizeof(shared_mem_fs)));

  EXPECT_EQ(backend.get_tmp_buf(), buf);
}

// The try_lock contract: if buf_mutex is already held (a materialization
// in flight on another thread — or, in the real library, on the very
// thread a CR signal interrupted), the op fails cleanly with nullptr
// instead of blocking; once the lock is free the next op succeeds.
TEST_F(MmapBackendDeferredTest, ContendedMaterializationFailsCleanly) {
  ShareMem backend(kId);
  backend.setup();

  backend.buf_mutex.lock();
  EXPECT_EQ(backend.get_tmp_buf(), nullptr);
  backend.buf_mutex.unlock();

  EXPECT_NE(backend.get_tmp_buf(), nullptr);
}

// The non-fatal contract: a failed materialization reports nullptr and
// leaves the process alive; once the data dir is usable again the next
// buffer-path op retries and succeeds.
TEST_F(MmapBackendDeferredTest, FailedMaterializationIsNonFatalAndRetries) {
  ShareMem backend(kId);
  backend.setup();  // The host buffer needs the good dir (fatal path).

  setenv("EXPORT_FILE_PATH", "/nonexistent-gpu-cr-unit-dir", 1);
  EXPECT_EQ(backend.get_tmp_buf(), nullptr);

  setenv("EXPORT_FILE_PATH", dir_, 1);
  EXPECT_NE(backend.get_tmp_buf(), nullptr);
}

}  // namespace
}  // namespace gpu_cr
