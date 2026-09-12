// Upstream-baseline control channel (ShareMemComm): the coordinator and the
// .so each mmap control-<pid> in the data dir and exchange single-word
// messages through it. Linux-only (mmap of a disk-backed file).

#include "comm/comm.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <set>
#include <string>

#include "ctl_path.h"
#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

class ShareMemCommTest : public ::testing::Test {
 protected:
  void SetUp() override {
    unsetenv("GPU_CR_CTL_PATH");
    // CtlDir memoizes per process; each test must re-resolve under its
    // own layout.
    ResetCtlDirCacheForTests();
    strcpy(dir_, "/tmp/gpu-cr-shm-comm-XXXXXX");
    ASSERT_NE(mkdtemp(dir_), nullptr);
    setenv("EXPORT_FILE_PATH", dir_, 1);
  }

  void TearDown() override {
    for (pid_t pid : used_pids_) unlink(ControlPath(pid).c_str());
    rmdir(dir_);
    unsetenv("EXPORT_FILE_PATH");
  }

  std::string ControlPath(pid_t pid) {
    used_pids_.insert(pid);
    return std::string(dir_) + "/control-" + std::to_string(pid);
  }

  char dir_[64];
  std::set<pid_t> used_pids_;
};

// The Comm base class is the inert null channel.
TEST_F(ShareMemCommTest, BaseCommIsInert) {
  Comm comm(1234);
  comm.setup();
  comm.send_msg(CKPT_MSG);
  EXPECT_EQ(comm.recv_msg(), 0u);
  EXPECT_FALSE(comm.is_finished());
}

TEST_F(ShareMemCommTest, SetupCreatesControlFile) {
  ShareMemComm comm(111);
  comm.setup();
  EXPECT_GE(comm.fd_control, 0);  // kept open so callers can flock the op

  struct stat st;
  ASSERT_EQ(stat(ControlPath(111).c_str(), &st), 0);
  EXPECT_EQ(st.st_size, static_cast<off_t>(HUGE_PAGE_SIZE));
  // fchmod(0777): agent container and workload run as different users.
  EXPECT_EQ(st.st_mode & 07777, 0777u);
}

// Ctl mode: a tmpfs GPU_CR_CTL_PATH moves the control file off the data
// dir, and setup() preallocates the stored extent so a full ctl tmpfs
// fails here instead of as SIGBUS at the first store.
TEST_F(ShareMemCommTest, SetupUsesCtlDirWhenTmpfs) {
  if (!DirIsTmpfs("/dev/shm")) GTEST_SKIP() << "/dev/shm is not tmpfs here";
  char ctl_dir[] = "/dev/shm/gpu-cr-ctl-XXXXXX";
  ASSERT_NE(mkdtemp(ctl_dir), nullptr);
  setenv("GPU_CR_CTL_PATH", ctl_dir, 1);

  ShareMemComm comm(666);
  comm.setup();

  std::string ctl_control = std::string(ctl_dir) + "/control-666";
  struct stat st;
  ASSERT_EQ(stat(ctl_control.c_str(), &st), 0);
  EXPECT_EQ(st.st_size, static_cast<off_t>(HUGE_PAGE_SIZE));
  // posix_fallocate pinned: at least the stored extent is backed.
  EXPECT_GE(static_cast<size_t>(st.st_blocks) * 512,
            RoundUp4K(sizeof(signal_controls)));
  // The data dir must NOT grow a control file in ctl mode.
  EXPECT_NE(stat(ControlPath(666).c_str(), &st), 0);

  unlink(ctl_control.c_str());
  rmdir(ctl_dir);
  unsetenv("GPU_CR_CTL_PATH");
}

// Zero-config discovery: with NO env at all, a tmpfs-backed <data>/ctl
// subdir is found through the data dir the workload already has, and
// setup() puts the control file there — the consumer-side contract stays
// LD_PRELOAD + EXPORT_FILE_PATH only.
TEST_F(ShareMemCommTest, SetupDiscoversNestedCtlDir) {
  if (!DirIsTmpfs("/dev/shm")) GTEST_SKIP() << "/dev/shm is not tmpfs here";
  char data_dir[] = "/dev/shm/gpu-cr-data-XXXXXX";
  ASSERT_NE(mkdtemp(data_dir), nullptr);
  setenv("EXPORT_FILE_PATH", data_dir, 1);
  std::string nested = std::string(data_dir) + "/" + kCtlSubdir;
  ASSERT_EQ(mkdir(nested.c_str(), 0777), 0);
  ResetCtlDirCacheForTests();  // EXPORT_FILE_PATH changed after SetUp

  ShareMemComm comm(777);
  comm.setup();

  std::string ctl_control = nested + "/control-777";
  struct stat st;
  ASSERT_EQ(stat(ctl_control.c_str(), &st), 0);
  EXPECT_EQ(st.st_size, static_cast<off_t>(HUGE_PAGE_SIZE));
  // posix_fallocate pinned: at least the stored extent is backed.
  EXPECT_GE(static_cast<size_t>(st.st_blocks) * 512,
            RoundUp4K(sizeof(signal_controls)));
  // The data dir root must NOT grow a control file in ctl mode.
  EXPECT_NE(stat((std::string(data_dir) + "/control-777").c_str(), &st), 0);

  unlink(ctl_control.c_str());
  rmdir(nested.c_str());
  rmdir(data_dir);
}

// A fresh (zero-filled) mapping reads FINISH: cr_client relies on an idle
// channel being indistinguishable from a completed op.
TEST_F(ShareMemCommTest, FreshChannelReadsFinish) {
  ShareMemComm comm(222);
  comm.setup();
  EXPECT_EQ(comm.recv_msg(), static_cast<uint32_t>(FINISH_MSG));
  EXPECT_TRUE(comm.is_finished());
}

TEST_F(ShareMemCommTest, SendRecvRoundTrip) {
  ShareMemComm comm(333);
  comm.setup();
  for (uint32_t msg : {INIT_MSG, CKPT_MSG, RESTORE_MSG, IPC_TEARDOWN_MSG}) {
    comm.send_msg(msg);
    EXPECT_EQ(comm.recv_msg(), msg);
    EXPECT_FALSE(comm.is_finished());
  }
  comm.send_msg(FINISH_MSG);
  EXPECT_TRUE(comm.is_finished());
}

// Coordinator and workload sides are two ShareMemComm instances over the
// same pid: writes on one end must be visible on the other.
TEST_F(ShareMemCommTest, TwoEndpointsShareTheChannel) {
  ShareMemComm coordinator(444);
  ShareMemComm workload(444);
  coordinator.setup();
  workload.setup();

  coordinator.send_msg(CKPT_MSG);
  EXPECT_EQ(workload.recv_msg(), static_cast<uint32_t>(CKPT_MSG));

  workload.send_msg(FINISH_MSG);
  EXPECT_TRUE(coordinator.is_finished());
}

// Every protocol verb must be distinct — the channel is a single word.
TEST_F(ShareMemCommTest, MessageConstantsDistinct) {
  std::set<uint32_t> msgs = {INIT_MSG,          CKPT_MSG,
                             RESTORE_MSG,       FINISH_MSG,
                             IPC_TEARDOWN_MSG,  IPC_EXPORT_MSG,
                             IPC_IMPORT_MSG,    NCCL_SUSPEND_MSG,
                             NCCL_RESUME_MSG,   SELECTIVE_CKPT_MSG,
                             SELECTIVE_RESTORE_MSG};
  EXPECT_EQ(msgs.size(), 11u);
}

// setup() keeps the historical fatal-exit contract when the data dir is
// unusable (the coordinator runs it first, so this fails fast and clean).
TEST_F(ShareMemCommTest, MissingDataDirExitsFatally) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  setenv("EXPORT_FILE_PATH", "/nonexistent-gpu-cr-unit-dir", 1);
  ShareMemComm comm(555);
  EXPECT_EXIT(comm.setup(), ::testing::ExitedWithCode(EXIT_FAILURE), "open");
}

}  // namespace
}  // namespace gpu_cr
