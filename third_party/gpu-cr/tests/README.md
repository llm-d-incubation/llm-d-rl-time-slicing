# GPU-CR test suites

Four tiers, from GPU-free unit tests to on-node performance regression.
Throughout, "upstream" means the original GPU-CR project this tree builds
on — <https://github.com/gpu-os/GPU-CR/tree/main> — pinned at commit
`e9bbb52`. The first two tiers run automatically inside every
`Dockerfile.build` image build (`RUN_TESTS=1`, the default) — a red suite
fails the build.

## What you need

- **Unit tier — no GPU, no cloud, no cluster, no CUDA.** Any Linux
  machine with CMake and a C++ compiler works; GoogleTest downloads at
  configure time, so network access is the only external dependency.
  Build only the `gpu_cr_unit_tests` target — the default targets
  (vGPU.so, cr_client) additionally need the CUDA toolkit.
- **Integration tier — no GPU or driver, but the CUDA toolkit to
  compile.** The `cr_client_integration` ctest entry drives the real
  `cr_client` binary, which includes `<cuda.h>` and links the CUDA
  stubs, so a plain C++ box cannot build it. Nothing at run time touches
  a GPU. `docker build -f Dockerfile.build .` runs both GPU-free tiers
  hermetically if you'd rather not install the toolchain.
- **Google Cloud Build is not required.** Every image is a plain
  `docker build`; `gcloud builds submit` flows are an optional
  convenience for building off your machine. The one exception as
  written: `tests/e2e/build_baseline_so.sh` submits the baseline image
  build to Cloud Build — if you don't use Google Cloud, run the
  Dockerfile it generates with local `docker build` and push the image
  to your own registry.
- **GPU tiers — a Linux host with an NVIDIA GPU and 2MB hugepages.**
  `run_e2e.sh` and `perf_regression.sh` are plain scripts and run on any
  such host (the checkpoint store lives on hugepage-backed memory). The
  one-shot `tests/e2e/e2e-pod.yaml` flow additionally expects a
  pre-existing Kubernetes cluster (GKE or otherwise) with a node
  exposing one GPU plus 2Mi hugepages, and a registry the cluster can
  pull the images from — nothing provisions a cluster for you.

## How the unit tier works

The unit tier uses three standard C++ tools. If you have not worked with
a C++ test suite before, this is the map:

- **CMake** is the build system: it reads `CMakeLists.txt` and generates
  the actual build scripts. The tests sit behind the
  `-DGPU_CR_BUILD_TESTS=ON` switch (off by default, so ordinary builds
  never pay for them). When the switch is on, CMake downloads
  **GoogleTest** at configure time (`FetchContent`) — there is nothing to
  install by hand.
- **GoogleTest** is the test framework. Unlike interpreted languages,
  where a runner discovers test files at run time, C++ tests are
  *compiled into a program*: every `tests/unit/*_test.cpp` file, plus the
  GPU-free production sources under test, build into one executable,
  `gpu_cr_unit_tests`. Running that program runs every test case, prints
  an `[  OK  ]` or `[  FAILED  ]` line per case, and exits nonzero if
  anything failed.
- **ctest** is the test runner that ships with CMake. The build registers
  the executable under the name `unit` (the `add_test` line in
  `CMakeLists.txt`), and `ctest -R unit` runs every registered test whose
  name matches `unit`, reporting pass/fail from the exit code.

Inside a `*_test.cpp` file, each case is a `TEST` block:

```cpp
// TEST(<suite name>, <case name>). It registers itself; no main() is
// needed — the GTest::gtest_main library supplies one.
TEST(RoundUp2MBTest, ZeroStaysZero) {
  EXPECT_EQ(ROUND_UP_2MB(0UL), 0UL);  // check, record failure, continue
}
```

`EXPECT_*` assertions record a failure and keep going, so a single run
reports every broken expectation. The `ASSERT_*` variants abort the
current test case on failure — used when the lines after them would
crash on the bad value (e.g. dereferencing the result of a failed
`mmap`).

Useful invocations:

```sh
# Configure, build, and run the whole unit tier:
cmake -DGPU_VENDOR=NVIDIA -DGPU_CR_BUILD_TESTS=ON .. && make gpu_cr_unit_tests && ctest -R unit

# Show the failing assertions, not just the red summary line:
ctest -R unit --output-on-failure

# Run the test program directly, filtered to one suite:
./gpu_cr_unit_tests --gtest_filter='RoundUp2MBTest.*'
```

To add a test, add a `TEST(...)` block to the matching `*_test.cpp` file
and rebuild — registration is automatic. A brand-new test file must also
be added to the `gpu_cr_unit_tests` source list in the top-level
`CMakeLists.txt`, or it will not be compiled in.

The remaining tiers are bash scripts, not GoogleTest: each prints a
`PASS`/`FAIL` line per scenario or gate and exits nonzero if anything
failed. The integration script is registered with ctest too, so it
reports through the same `ctest` front end as the unit tier.

## 1. Unit tests — `tests/unit/` (GoogleTest, no GPU, Linux)

Function-level coverage of two layers:

- **Code added on top of upstream**: buffer-size config parsing,
  control-path resolution and advertisement round-trip, dump-format
  validation, granule clamping, consume-once FINISH bookkeeping,
  region-spec parsing, and wire-layout guards.
- **The upstream-baseline functions themselves**: the 2MB rounding macro,
  signal numbers and wire structs (`common_baseline_test`), the
  ShareMemComm control channel (`share_mem_comm_test`), the ShareMem
  dump/staging buffer mapping via the file backend (`mmap_backend_test`),
  the UDS SCM_RIGHTS fd exchange (`ipc_fd_exchange_test`), and
  `memcpy_multi` (`memcpy_multi_test`). `createGPU()` and the CUDA/HIP
  hook layers need a driver link, so they stay covered by the
  integration and e2e tiers.

## 2. Integration tests — `tests/integration/` (no GPU, Linux)

`cr_client_integration_test.sh` drives the **real `cr_client` binary**
against `fake_workload`, a GPU-free stand-in for the vGPU.so side that
reuses the production control-channel code (ShareMemComm, advertisement
writer, FINISH bookkeeping, dump validator). Covers ctl (env-configured
AND zero-config `<data>/ctl` discovery) and legacy modes,
destination-path checkpoint/restore, torn-dump refusal, op_status
propagation (including the "never freeze after a failed checkpoint" gate),
timeouts, PID-reuse refusal, not-ready refusals, and the documented exit
codes (0 OK / 1 usage / 2 op failed / 3 refused pre-signal — not-ready
gate, missing advertisement, or dead target / 4 timeout).

Two contract points the suite pins, worth knowing when driving
`cr_client` from the agent:

- **Every op now carries a deadline** (`GPU_CR_OP_TIMEOUT_SEC`, default
  120s) — including full-process ops, which historically waited forever.
- **Exit 4 poisons the control channel.** The client's flock releases at
  exit while the wedged in-process handler may still be mid-op, and the
  next invocation rewrites the shared request words. After a timeout,
  restart the workload (or prove the handler finished) before issuing
  another op against that PID; a dest-path dump interrupted by a timeout
  carries no commit marker and is refused on restore.

```sh
ctest -R cr_client_integration --output-on-failure
```

The full per-case matrix (setup, invocation, expected exit) lives in
[`tests/integration/README.md`](integration/README.md).

Optionally, both GPU-free tiers in one shot on Google Cloud Build (not
required — see "What you need"; no image published):

```sh
gcloud builds submit --config cloudbuild-test.yaml .
```

## 3. End-to-end — `tests/e2e/run_e2e.sh` (GPU node)

A real CUDA workload (`pattern_workload`) under `LD_PRELOAD=vGPU-NVIDIA.so`
goes through destination-path selective C/R, buffer-path selective C/R, and
full C/R, gating on **byte-identical GPU memory after every restore**. The
final gate (G8) reruns a dest-path round trip with `GPU_CR_CTL_PATH` unset
on both sides, proving the zero-config `<store>/ctl` discovery path on the
production nested-tmpfs layout (skipped when `$STORE/ctl` is not tmpfs).

## 4. Performance regression — `tests/e2e/perf_regression.sh` (GPU node)

Verifies the consolidated build has not regressed the full checkpoint/
restore data plane that upstream (`e9bbb52`) delivers: same workload,
same node, baseline .so vs candidate .so, median-of-N compared against a
threshold (default 15%). Candidate selective-path timings are recorded as
informational (no upstream baseline exists for them).

Build the baseline once with `tests/e2e/build_baseline_so.sh` (run from
a checkout containing the pinned upstream commit — a vendored copy of
this tree does not carry upstream history), then run
both GPU tiers as a one-shot pod: `tests/e2e/e2e-pod.yaml` (exits 0 only if
every e2e gate and the perf gate pass).
