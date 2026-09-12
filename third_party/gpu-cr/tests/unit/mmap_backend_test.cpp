// Upstream-baseline ShareMem backend (mmap_backend.cpp): setup() maps the
// dump buffer and the two-slot host staging buffer, get_tmp_buf() /
// get_host_buffer() hand them out. Exercised through the file backend
// (EXPORT_FILE_PATH) so no hugepages are needed. Linux-only.

#include "backend/backend.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <set>
#include <string>

#include "common.h"
#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

// Pin small buffer sizes (floors: 64MiB dump, 128MiB staging) and
// capture them into the cached gpu_cr::Config() singleton at static-init
// time — gpu_cr_config_test's fixture scrubs these env vars, so waiting
// until the first test-time Config() call would fall back to the 25GiB
// build default and fallocate that per test.
const bool kBufEnvPinned = [] {
  setenv("GPU_CR_SHM_MB", "64", 1);
  setenv("GPU_CR_STAGING_MB", "128", 1);
  return Config().shm_size == (64UL << 20) &&
         Config().staging_size == (128UL << 20);
}();

class MmapBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(kBufEnvPinned);
    strcpy(dir_, "/tmp/gpu-cr-mmap-backend-XXXXXX");
    ASSERT_NE(mkdtemp(dir_), nullptr);
    setenv("EXPORT_FILE_PATH", dir_, 1);
  }

  void TearDown() override {
    for (int id : used_ids_) {
      unlink(DumpPath(id).c_str());
      unlink(HostPath(id).c_str());
    }
    rmdir(dir_);
    unsetenv("EXPORT_FILE_PATH");
  }

  // Pure path helpers; Track(id) registers an id's backing files for
  // TearDown cleanup and passes the id through, so tests can register at
  // the point of construction: ShareMem backend(Track(7)).
  std::string DumpPath(int id) const {
    return std::string(dir_) + "/ckpt-" + std::to_string(id) + ".data";
  }
  std::string HostPath(int id) const {
    return std::string(dir_) + "/ckpt-" + std::to_string(id) + "-host.data";
  }
  int Track(int id) {
    used_ids_.insert(id);
    return id;
  }

  char dir_[64];
  std::set<int> used_ids_;
};

TEST_F(MmapBackendTest, SetupInitializesDumpBufferHeader) {
  ShareMem backend(Track(7));
  backend.setup();

  void* buf = backend.get_tmp_buf();
  ASSERT_NE(buf, nullptr);
  shared_mem_fs* fs = static_cast<shared_mem_fs*>(buf);
  EXPECT_EQ(fs->file_num, 0u);
  EXPECT_EQ(fs->current_offset, ROUND_UP_2MB(sizeof(shared_mem_fs)));
}

TEST_F(MmapBackendTest, FilesCreatedAtConfiguredSizes) {
  ShareMem backend(Track(3));
  backend.setup();
  ASSERT_NE(backend.get_tmp_buf(), nullptr);

  struct stat st;
  ASSERT_EQ(stat(DumpPath(3).c_str(), &st), 0);
  EXPECT_EQ(static_cast<size_t>(st.st_size), Config().shm_size);

  ASSERT_EQ(stat(HostPath(3).c_str(), &st), 0);
  EXPECT_EQ(static_cast<size_t>(st.st_size),
            Config().staging_size * STAGING_BUF_NUM);
}

TEST_F(MmapBackendTest, BuffersAreWritable) {
  ShareMem backend(Track(5));
  backend.setup();

  char* dump = static_cast<char*>(backend.get_tmp_buf());
  ASSERT_NE(dump, nullptr);
  shared_mem_fs* fs = reinterpret_cast<shared_mem_fs*>(dump);
  // First data byte after the reserved header extent.
  dump[fs->current_offset] = 0x5a;
  EXPECT_EQ(dump[fs->current_offset], 0x5a);

  char* host = static_cast<char*>(backend.get_host_buffer());
  ASSERT_NE(host, nullptr);
  // Touch both staging slots.
  host[0] = 0x11;
  host[Config().staging_size] = 0x22;
  EXPECT_EQ(host[0], 0x11);
  EXPECT_EQ(host[Config().staging_size], 0x22);
}

TEST_F(MmapBackendTest, GettersAreStable) {
  ShareMem backend(Track(6));
  backend.setup();
  void* tmp = backend.get_tmp_buf();
  ASSERT_NE(tmp, nullptr);
  EXPECT_EQ(backend.get_tmp_buf(), tmp);
  void* host = backend.get_host_buffer();
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(backend.get_host_buffer(), host);
}

TEST_F(MmapBackendTest, DistinctIdsGetDistinctFiles) {
  ShareMem a(Track(1));
  ShareMem b(Track(2));
  a.setup();
  b.setup();
  ASSERT_NE(a.get_tmp_buf(), nullptr);
  ASSERT_NE(b.get_tmp_buf(), nullptr);
  EXPECT_NE(a.get_tmp_buf(), b.get_tmp_buf());

  struct stat st;
  EXPECT_EQ(stat(DumpPath(1).c_str(), &st), 0);
  EXPECT_EQ(stat(DumpPath(2).c_str(), &st), 0);
}

// Reusing an id whose backing file holds a stale (or garbage) header must
// re-initialize it: setup() rewrites file_num and current_offset after
// mapping, which the restore path depends on to not read leftovers.
TEST_F(MmapBackendTest, SetupResetsStaleHeaderOnExistingFile) {
  int fd = open(DumpPath(Track(8)).c_str(), O_CREAT | O_RDWR, 0644);
  ASSERT_GE(fd, 0);
  shared_mem_fs garbage;
  memset(&garbage, 0xa5, sizeof(garbage));
  ASSERT_EQ(write(fd, &garbage, sizeof(garbage)),
            static_cast<ssize_t>(sizeof(garbage)));
  close(fd);

  ShareMem backend(8);
  backend.setup();
  shared_mem_fs* fs = static_cast<shared_mem_fs*>(backend.get_tmp_buf());
  ASSERT_NE(fs, nullptr);
  EXPECT_EQ(fs->file_num, 0u);
  EXPECT_EQ(fs->current_offset, ROUND_UP_2MB(sizeof(shared_mem_fs)));
}

// Eager setup keeps the historical fatal-exit contract when the data dir
// is unusable.
TEST_F(MmapBackendTest, MissingDataDirExitsFatally) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  setenv("EXPORT_FILE_PATH", "/nonexistent-gpu-cr-unit-dir", 1);
  ShareMem backend(9);
  EXPECT_EXIT(backend.setup(), ::testing::ExitedWithCode(EXIT_FAILURE),
              "dump buffer");
}

}  // namespace
}  // namespace gpu_cr
