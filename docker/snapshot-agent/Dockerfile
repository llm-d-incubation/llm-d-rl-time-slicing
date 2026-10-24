# Released GPU-CR artifact package (see docker/gpu-cr/Dockerfile and
# .github/workflows/ci-release.yaml): carries cr_client and vGPU-NVIDIA.so
# built as a version-locked pair from third_party/gpu-cr. The agent takes
# cr_client from it; workload images must take vGPU-NVIDIA.so from the SAME
# tag — the two share compiled-in constants and the control-channel
# protocol, and a mismatch makes checkpoints silently no-op. Override for
# dev builds with --build-arg GPU_CR_IMAGE=<your gpu-cr image>.
ARG GPU_CR_IMAGE=ghcr.io/llm-d-incubation/llm-d-rl-time-slicing/gpu-cr:latest
FROM ${GPU_CR_IMAGE} AS gpu-cr

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

# --- Runtime stage ---
# distroless/cc (base + libgcc/libstdc++): cr_client is a dynamically linked
# C++ binary.
FROM gcr.io/distroless/cc

WORKDIR /
COPY --from=builder /workspace/snapshot-agent .
# Copy bundled binaries (cuda-checkpoint) from the builder's bin directory
COPY --from=builder /workspace/bin ./bin
# cr_client at the GPU-CR backends' default fallback path (also on PATH),
# taken from the released gpu-cr package (same image workloads take the
# preloader from).
COPY --from=gpu-cr /opt/gpu-cr/cr_client /usr/local/bin/cr_client
COPY --from=gpu-cr /opt/gpu-cr/cr_client /bin/cr_client

ENTRYPOINT ["/snapshot-agent"]
