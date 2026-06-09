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

# --- Runtime stage ---
FROM gcr.io/distroless/base

WORKDIR /
COPY --from=builder /workspace/snapshot-agent .
# Copy pre-built binaries (like cuda-checkpoint) from the bin directory
COPY --from=builder /workspace/bin ./bin


ENTRYPOINT ["/snapshot-agent"]