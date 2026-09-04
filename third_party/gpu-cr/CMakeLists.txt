cmake_minimum_required(VERSION 3.18)
project(GPU-CR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# GPU vendor selection
# Usage:
#   cmake -DGPU_VENDOR=NVIDIA ..        (default)
#   cmake -DGPU_VENDOR=AMD ..
#   GPU_VENDOR=AMD cmake ..             (env override)
# ---------------------------------------------------------------------------
if(NOT DEFINED GPU_VENDOR)
    if(DEFINED ENV{GPU_VENDOR})
        set(GPU_VENDOR $ENV{GPU_VENDOR})
    else()
        set(GPU_VENDOR "NVIDIA")
    endif()
endif()

message(STATUS "Building for GPU vendor: ${GPU_VENDOR}")

if(GPU_VENDOR STREQUAL "NVIDIA")
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
    set(GPU_INCLUDE_DIR "${CUDA_ROOT}/include")
    set(GPU_LIB_DIR "${CUDA_ROOT}/lib64")
    set(GPU_LIBS cudart cuda)
elseif(GPU_VENDOR STREQUAL "AMD")
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation directory")
    set(GPU_INCLUDE_DIR "${ROCM_ROOT}/include")
    set(GPU_LIB_DIR "${ROCM_ROOT}/lib")
    set(GPU_LIBS amdhip64)
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    message(FATAL_ERROR "Unknown GPU_VENDOR: ${GPU_VENDOR}. Must be NVIDIA or AMD")
endif()

# ---------------------------------------------------------------------------
# Per-worker staging buffer size on hugepages (compile-time constant; consumed
# by vGPU.so and multi_cr_client via src/common.h). Increase when running
# vLLM/larger workloads at gpu_memory_utilization > ~0.5.
# Override with `cmake -DSHM_SIZE_GB=40 ..`
# ---------------------------------------------------------------------------
if(DEFINED SHM_SIZE_GB)
    add_compile_definitions(SHM_SIZE_GB=${SHM_SIZE_GB})
    message(STATUS "Per-worker SHM_SIZE_GB = ${SHM_SIZE_GB}")
endif()

# ---------------------------------------------------------------------------
# Optional adapter build flags
# ---------------------------------------------------------------------------
option(GPU_CR_BUILD_NCCL_ADAPTER    "Build the NCCL checkpoint adapter (plugin + runtime)" OFF)
option(GPU_CR_BUILD_NVSHMEM_ADAPTER "Build the NVSHMEM checkpoint examples"                OFF)

# ---------------------------------------------------------------------------
# Core LD_PRELOAD library: vGPU-${GPU_VENDOR}.so
# ---------------------------------------------------------------------------
file(GLOB_RECURSE VGPU_SOURCE_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Drop scratch sources.
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "-pre\\.cpp$")

# vGPU.cpp still carries its own memcpy_multi definition; the standalone
# TU exists so the GPU-free unit suite can exercise it, and joins this
# library when the preloader refactor drops the in-file copy.
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/memcpy_multi\\.cpp$")

# Per-vendor backend exclusions.
if(GPU_VENDOR STREQUAL "NVIDIA")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/AMD/")
    message(STATUS "Excluding AMD backend sources")
elseif(GPU_VENDOR STREQUAL "AMD")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/NVIDIA/")
    # cuMem IPC + NCCL hooks are CUDA-specific; not built for AMD.
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/ipc_hooks\\.cpp$")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/ipc_fd_exchange\\.cpp$")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/nccl_hooks\\.cpp$")
    message(STATUS "Excluding NVIDIA backend sources and CUDA-only IPC/NCCL hooks")
endif()

file(GLOB_RECURSE VGPU_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_library(vGPU SHARED ${VGPU_SOURCE_FILES} ${VGPU_HEADER_FILES})

set_target_properties(vGPU PROPERTIES
    OUTPUT_NAME "vGPU-${GPU_VENDOR}"
    PREFIX ""
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

target_include_directories(vGPU PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(vGPU PRIVATE ${GPU_LIB_DIR})
target_link_libraries(vGPU PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(vGPU PRIVATE -fPIC -O3 -g)
target_link_options(vGPU PRIVATE -shared -fPIC)

# ---------------------------------------------------------------------------
# Coordinator CLIs
#   cr_client       — single-GPU checkpoint trigger
#   multi_cr_client — multi-GPU phased orchestrator
# Both reuse src/comm/share_mem.cpp for their SHM control channel.
# ---------------------------------------------------------------------------
set(_COORD_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
    # multi_cr_client maps worker buffers at SHM_SIZE, a runtime value
    # read from gpu_cr::Config().
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_cr_config.cpp
)

add_executable(cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/cr_client.cpp
    ${_COORD_COMMON_SOURCES}
)
target_include_directories(cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(cr_client PRIVATE ${GPU_LIB_DIR})
target_link_libraries(cr_client PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(cr_client PRIVATE -O3 -g)

add_executable(multi_cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/multi_cr_client.cpp
    ${_COORD_COMMON_SOURCES}
)
target_include_directories(multi_cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(multi_cr_client PRIVATE ${GPU_LIB_DIR})
target_link_libraries(multi_cr_client PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(multi_cr_client PRIVATE -O3 -g)

# ---------------------------------------------------------------------------
# Tests (GPU-free; run in Dockerfile.build)
#   gpu_cr_unit_tests        — GoogleTest unit suite for the ctl-path,
#                              dump-format, granule-clamp, region-spec and
#                              runtime-config code, plus the
#                              upstream-baseline control channel, buffer
#                              mapping, fd exchange and memcpy_multi
#   fake_workload            — GPU-free stand-in for the vGPU.so side of the
#                              control channel
#   cr_client_integration    — drives the real cr_client binary against
#                              fake_workload, asserting protocol behavior and
#                              the documented exit codes
# ---------------------------------------------------------------------------
option(GPU_CR_BUILD_TESTS "Build the GPU-free unit and integration tests" OFF)

if(GPU_CR_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(gpu_cr_unit_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/common_baseline_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/common_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/ctl_path_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/dump_format_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_cr_config_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/ipc_fd_exchange_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/memcpy_multi_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/mmap_backend_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/selective_spec_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/share_mem_comm_test.cpp
        # GPU-free production sources under test (upstream-baseline functions)
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/mmap_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_cr_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc_fd_exchange.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/memcpy_multi.cpp
    )
    target_include_directories(gpu_cr_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # selective_spec_test includes coordinator/selective_spec.h
        ${CMAKE_CURRENT_SOURCE_DIR}/coordinator
    )
    # The file backend reserves the full dump-buffer size up front
    # (posix_fallocate); pin the suite to the 1GiB minimum so the
    # mmap-backend tests don't reserve the 25GiB production default per
    # checkpoint id. Expressed as compile *options* (-U then -D) so the
    # override lands after any directory-level -DSHM_SIZE_GB from the
    # build harness in the generated compile command.
    target_compile_options(gpu_cr_unit_tests PRIVATE -USHM_SIZE_GB -DSHM_SIZE_GB=1)
    target_link_libraries(gpu_cr_unit_tests PRIVATE GTest::gtest_main pthread)
    add_test(NAME unit COMMAND gpu_cr_unit_tests)

    add_executable(fake_workload
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/fake_workload.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
    )
    target_include_directories(fake_workload PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    add_test(NAME cr_client_integration
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/cr_client_integration_test.sh
                $<TARGET_FILE:cr_client> $<TARGET_FILE:fake_workload>
    )
    set_tests_properties(cr_client_integration PROPERTIES TIMEOUT 300)
endif()

# ---------------------------------------------------------------------------
# E2E / perf-regression workload (needs the CUDA runtime; run on a GPU node
# under LD_PRELOAD via tests/e2e/run_e2e.sh and tests/e2e/perf_regression.sh)
# ---------------------------------------------------------------------------
option(GPU_CR_BUILD_E2E "Build the CUDA pattern workload for the e2e suites" OFF)

if(GPU_CR_BUILD_E2E)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "GPU_CR_BUILD_E2E requires GPU_VENDOR=NVIDIA")
    endif()
    add_executable(pattern_workload
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/pattern_workload.cpp
    )
    target_include_directories(pattern_workload PRIVATE ${GPU_INCLUDE_DIR})
    target_link_directories(pattern_workload PRIVATE ${GPU_LIB_DIR})
    target_link_libraries(pattern_workload PRIVATE cudart)
    target_compile_options(pattern_workload PRIVATE -O2 -g)
endif()


# ---------------------------------------------------------------------------
# Optional adapters (only meaningful on NVIDIA)
# ---------------------------------------------------------------------------
if(GPU_CR_BUILD_NCCL_ADAPTER)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "GPU_CR_BUILD_NCCL_ADAPTER requires GPU_VENDOR=NVIDIA")
    endif()
    add_subdirectory(adapters/nccl)
endif()

if(GPU_CR_BUILD_NVSHMEM_ADAPTER)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "GPU_CR_BUILD_NVSHMEM_ADAPTER requires GPU_VENDOR=NVIDIA")
    endif()
    add_subdirectory(adapters/nvshmem)
endif()
