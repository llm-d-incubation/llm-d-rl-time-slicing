# --- Build stage ---
FROM golang:1.26.2 AS builder

WORKDIR /workspace

# Copy the entire project to resolve internal dependencies
COPY . .

# Recreate go.work to ensure clean multi-module build
RUN rm -f go.work && \
    go work init . && \
    go work use ./pkg/timeslice-orchestrator && \
    go work use ./pkg/snapshot-agent

# Build the snapshot-agent
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o /workspace/snapshot-agent ./cmd/snapshot-agent

# Fetch cuda-checkpoint (pinned to a specific NVIDIA/cuda-checkpoint commit) so the
# image is self-contained and builds from a fresh clone without a pre-populated bin/
ADD https://github.com/NVIDIA/cuda-checkpoint/raw/00d5cce84c628088d6caa203fc4af40c1538b6f7/bin/x86_64_Linux/cuda-checkpoint /workspace/bin/cuda-checkpoint
RUN chmod +x /workspace/bin/cuda-checkpoint

# --- GPU-CR cr_client build stage ---
# cr_client and the workload's vGPU-NVIDIA.so preloader are version-locked
# through the GPU-CR control protocol, so the client is built from the SAME
# in-tree source (third_party/gpu-cr) the deployed .so ships from — never
# from a stale checked-in binary or an out-of-tree checkout. CPU-only:
# compiles against CUDA headers, no GPU needed.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS gpucr-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential && rm -rf /var/lib/apt/lists/*
COPY third_party/gpu-cr /tmp/GPU-CR
# Build-time patch, kept byte-in-sync with third_party/gpu-cr/Dockerfile.build
# (which builds the deployed .so): remap checkpoint/restore signals off
# SIGUSR1/2 (taken by NCCL/vLLM). The constants compile into both binaries;
# a mismatch makes checkpoints silently no-op. The sed no-ops once the remap
# becomes source truth in-tree (then this block and Dockerfile.build's twin
# can both be deleted); the greps assert the end state either way, and the
# 0777 grep pins the control-file mode that is already source truth in
# share_mem.cpp.
RUN cd /tmp/GPU-CR && \
    sed -i 's/#define CR_CKPT_SIGNAL     SIGUSR1/#define CR_CKPT_SIGNAL     (SIGRTMAX - 8)/' src/common.h && \
    sed -i 's/#define CR_RESTORE_SIGNAL  SIGUSR2/#define CR_RESTORE_SIGNAL  (SIGRTMAX - 7)/' src/common.h && \
    grep -q "SIGRTMAX - 8" src/common.h && \
    grep -q "SIGRTMAX - 7" src/common.h && \
    grep -q "0777" src/comm/share_mem.cpp && \
    mkdir build && cd build && \
    cmake -DGPU_VENDOR=NVIDIA .. && \
    make -j"$(nproc)" cr_client

# --- Runtime stage ---
# distroless/cc (base + libgcc/libstdc++): cr_client is a dynamically linked
# C++ binary.
FROM gcr.io/distroless/cc

WORKDIR /
COPY --from=builder /workspace/snapshot-agent .
# Copy bundled binaries (cuda-checkpoint) from the builder's bin directory
COPY --from=builder /workspace/bin ./bin
# cr_client at the GPU-CR backends' default fallback path (also on PATH),
# built from the same tree as the deployed vGPU-NVIDIA.so.
COPY --from=gpucr-builder /tmp/GPU-CR/build/cr_client /usr/local/bin/cr_client
COPY --from=gpucr-builder /tmp/GPU-CR/build/cr_client /bin/cr_client

ENTRYPOINT ["/snapshot-agent"]
