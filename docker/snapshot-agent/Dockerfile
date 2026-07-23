# --- Build stage ---
FROM golang:1.26.2 AS builder

WORKDIR /workspace

# Copy the entire project to resolve internal dependencies
COPY . .

# Recreate go.work to ensure clean multi-module build
RUN rm -f go.work && \
    go work init . && \
    go work use ./pkg/accelerator-orchestrator && \
    go work use ./pkg/snapshot-agent

# Build the snapshot-agent
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o /workspace/snapshot-agent ./cmd/snapshot-agent

# Fetch cuda-checkpoint (pinned to a specific NVIDIA/cuda-checkpoint commit) so the
# image is self-contained and builds from a fresh clone without a pre-populated bin/
ADD https://github.com/NVIDIA/cuda-checkpoint/raw/00d5cce84c628088d6caa203fc4af40c1538b6f7/bin/x86_64_Linux/cuda-checkpoint /workspace/bin/cuda-checkpoint
RUN chmod +x /workspace/bin/cuda-checkpoint

# --- Runtime stage ---
FROM gcr.io/distroless/base

WORKDIR /
COPY --from=builder /workspace/snapshot-agent .
# Copy bundled binaries (cuda-checkpoint) from the builder's bin directory
COPY --from=builder /workspace/bin ./bin


ENTRYPOINT ["/snapshot-agent"]