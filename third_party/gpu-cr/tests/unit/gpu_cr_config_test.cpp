// Buffer-config unit matrix: env parsing, bounds, defaults, legacy warnings.
// Host-only (no CUDA); exercises gpu_cr::internal::Load() directly so each
// case gets a fresh parse (the production singleton caches, by design).

#include "gpu_cr_config.h"

#include <stdlib.h>

#include "gtest/gtest.h"

namespace gpu_cr {
namespace {

constexpr size_t kShm8 = 8UL << 30;  // an "SHM8 fleet build" default
constexpr size_t kStg1 = 1UL << 30;

class ConfigLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    unsetenv("GPU_CR_SHM_GB");
    unsetenv("GPU_CR_SHM_MB");
    unsetenv("GPU_CR_STAGING_MB");
    unsetenv("GPUCR_SHM_GB");
    unsetenv("GPUCR_STAGING_MB");
  }
  BufConfig Load() { return internal::Load(kShm8, kStg1); }
};

// Fleet-default rule: env unset => the BUILD's default, exactly.
TEST_F(ConfigLoadTest, UnsetEnvUsesBuildDefaults) {
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_EQ(c.staging_size, kStg1);
  EXPECT_FALSE(c.shm_deferred);
}

TEST_F(ConfigLoadTest, ShmGbHonored) {
  setenv("GPU_CR_SHM_GB", "12", 1);
  EXPECT_EQ(Load().shm_size, 12UL << 30);
}

TEST_F(ConfigLoadTest, ShmMbWinsOverGb) {
  setenv("GPU_CR_SHM_MB", "300", 1);
  setenv("GPU_CR_SHM_GB", "12", 1);
  EXPECT_EQ(Load().shm_size, 300UL << 20);
}

// An empty value counts as unset: MB="" must not shadow GB.
TEST_F(ConfigLoadTest, EmptyMbFallsThroughToGb) {
  setenv("GPU_CR_SHM_MB", "", 1);
  setenv("GPU_CR_SHM_GB", "12", 1);
  EXPECT_EQ(Load().shm_size, 12UL << 30);
}

// No upper clamp: values above the old 25GiB literal are honored.
TEST_F(ConfigLoadTest, NoUpperClamp) {
  setenv("GPU_CR_SHM_GB", "40", 1);
  EXPECT_EQ(Load().shm_size, 40UL << 30);
}

// Floor: below 64MiB -> default with warning, never a clamp.
TEST_F(ConfigLoadTest, BelowFloorFallsBackToDefault) {
  setenv("GPU_CR_SHM_MB", "10", 1);
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_FALSE(c.shm_deferred);
}

// The floor itself is accepted: 64MiB is valid, not "below 64MiB".
TEST_F(ConfigLoadTest, ExactShmFloorAccepted) {
  setenv("GPU_CR_SHM_MB", "64", 1);
  EXPECT_EQ(Load().shm_size, kShmFloorBytes);
}

// Deferred mode (=0) stays special: not subsumed by the floor.
TEST_F(ConfigLoadTest, ZeroMeansDeferredAtFloor) {
  setenv("GPU_CR_SHM_GB", "0", 1);
  BufConfig c = Load();
  EXPECT_TRUE(c.shm_deferred);
  EXPECT_EQ(c.shm_size, kShmFloorBytes);
}

// The MB spelling takes the other arm of the var selection; 0 must mean
// deferred there too.
TEST_F(ConfigLoadTest, ShmMbZeroAlsoDeferred) {
  setenv("GPU_CR_SHM_MB", "0", 1);
  BufConfig c = Load();
  EXPECT_TRUE(c.shm_deferred);
  EXPECT_EQ(c.shm_size, kShmFloorBytes);
}

TEST_F(ConfigLoadTest, UnparsableFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "8x", 1);
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_FALSE(c.shm_deferred);
}

TEST_F(ConfigLoadTest, NegativeFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "-3", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

// Overflow is never silent. 2^34+1 GB would wrap the size_t shift to
// exactly 1GiB and sail past the floor if unchecked.
TEST_F(ConfigLoadTest, ShmGbShiftOverflowFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "17179869185", 1);
  BufConfig c = Load();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_FALSE(c.shm_deferred);
}

// Beyond LLONG_MAX: strtoll saturates with ERANGE; must be rejected, not
// accepted as LLONG_MAX.
TEST_F(ConfigLoadTest, ShmGbErangeFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "99999999999999999999", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

// MB path wraps at n >= 2^44.
TEST_F(ConfigLoadTest, ShmMbShiftOverflowFallsBackToDefault) {
  setenv("GPU_CR_SHM_MB", "17592186044416", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

// MB within 2MiB-1 of SIZE_MAX>>20: the shift itself fits but the 2MiB
// round-up would wrap. Must reject, not silently produce a tiny buffer.
TEST_F(ConfigLoadTest, ShmMbAlignmentWrapFallsBackToDefault) {
  setenv("GPU_CR_SHM_MB", "17592186044415", 1);
  EXPECT_EQ(Load().shm_size, kShm8);
}

// Fits size_t but not signed off_t: 2^33 GB resolves to 2^63 bytes, which
// ftruncate() cannot represent. Must be rejected at parse, not crash the
// workload at init.
TEST_F(ConfigLoadTest, ShmGbOffTOverflowFallsBackToDefault) {
  setenv("GPU_CR_SHM_GB", "8589934592", 1);  // 2^33
  EXPECT_EQ(Load().shm_size, kShm8);
}

// The exact off_t boundary: 2^33-1 GB resolves to 2^63-2^30, the largest
// 2MiB-aligned extent a signed 64-bit off_t can carry after round-up.
TEST_F(ConfigLoadTest, ShmGbAtOffTBoundAccepted) {
  setenv("GPU_CR_SHM_GB", "8589934591", 1);  // 2^33 - 1
  EXPECT_EQ(Load().shm_size, (8589934591ULL << 30));
}

TEST_F(ConfigLoadTest, ShmAlignsUpTo2MiB) {
  setenv("GPU_CR_SHM_MB", "129", 1);
  EXPECT_EQ(Load().shm_size, 130UL << 20);
}

TEST_F(ConfigLoadTest, StagingHonored) {
  setenv("GPU_CR_STAGING_MB", "256", 1);
  EXPECT_EQ(Load().staging_size, 256UL << 20);
}

TEST_F(ConfigLoadTest, StagingBelowFloorFallsBackToDefault) {
  setenv("GPU_CR_STAGING_MB", "64", 1);
  EXPECT_EQ(Load().staging_size, kStg1);
}

TEST_F(ConfigLoadTest, ExactStagingFloorAccepted) {
  setenv("GPU_CR_STAGING_MB", "128", 1);
  EXPECT_EQ(Load().staging_size, kStagingFloorBytes);
}

TEST_F(ConfigLoadTest, StagingAlignsUpTo2MiB) {
  setenv("GPU_CR_STAGING_MB", "129", 1);
  EXPECT_EQ(Load().staging_size, 130UL << 20);
}

TEST_F(ConfigLoadTest, StagingNoUpperBound) {
  setenv("GPU_CR_STAGING_MB", "2048", 1);
  EXPECT_EQ(Load().staging_size, 2048UL << 20);
}

TEST_F(ConfigLoadTest, StagingOverflowFallsBackToDefault) {
  setenv("GPU_CR_STAGING_MB", "17592186044416", 1);
  EXPECT_EQ(Load().staging_size, kStg1);
}

// The two staging buffers are ftruncate'd/mmap'd as ONE file of
// staging x 2 bytes: 2^43 MB resolves to 2^63 per buffer, so the total
// wraps size_t to 0 (and each buffer alone already exceeds off_t). Must
// be rejected at parse, not discovered as a zero-extent mapping.
TEST_F(ConfigLoadTest, StagingTotalWrapFallsBackToDefault) {
  setenv("GPU_CR_STAGING_MB", "8796093022208", 1);  // 2^43
  EXPECT_EQ(Load().staging_size, kStg1);
}

// The exact total-representability boundary: 2^42-2 MB rounds to
// 2^62-2^21 per buffer; the x2 total is 2^63-2^22, still within off_t.
// One MB more would round the total past the off_t ceiling.
TEST_F(ConfigLoadTest, StagingAtTotalBoundAccepted) {
  setenv("GPU_CR_STAGING_MB", "4398046511102", 1);  // 2^42 - 2
  EXPECT_EQ(Load().staging_size, (4398046511102ULL << 20));
}

TEST_F(ConfigLoadTest, StagingJustPastTotalBoundFallsBackToDefault) {
  setenv("GPU_CR_STAGING_MB", "4398046511103", 1);  // 2^42 - 1
  EXPECT_EQ(Load().staging_size, kStg1);
}

// Legacy names: never honored (values ignored) for either field, and the
// warning that says so actually fires.
TEST_F(ConfigLoadTest, LegacyNamesIgnoredAndWarned) {
  setenv("GPUCR_SHM_GB", "60", 1);
  setenv("GPUCR_STAGING_MB", "999", 1);
  testing::internal::CaptureStderr();
  BufConfig c = Load();
  std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(c.shm_size, kShm8);
  EXPECT_EQ(c.staging_size, kStg1);
  EXPECT_NE(err.find("never read by any GPU-CR version"), std::string::npos);
}

TEST_F(ConfigLoadTest, ParseNonNegativeRejectsTrailingGarbage) {
  bool ok = true;
  internal::ParseNonNegative("12abc", &ok);
  EXPECT_FALSE(ok);
  EXPECT_EQ(internal::ParseNonNegative("12", &ok), 12);
  EXPECT_TRUE(ok);
  internal::ParseNonNegative("", &ok);
  EXPECT_FALSE(ok);
  internal::ParseNonNegative("-1", &ok);
  EXPECT_FALSE(ok);
}

// Pins the documented strtoll asymmetry: leading whitespace parses,
// trailing whitespace does not.
TEST_F(ConfigLoadTest, ParseNonNegativeStrtollFrontTolerance) {
  bool ok = false;
  EXPECT_EQ(internal::ParseNonNegative(" 8", &ok), 8);
  EXPECT_TRUE(ok);
  internal::ParseNonNegative("8 ", &ok);
  EXPECT_FALSE(ok);
}

}  // namespace
}  // namespace gpu_cr
