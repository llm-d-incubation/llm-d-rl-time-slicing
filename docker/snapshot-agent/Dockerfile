# Multi-stage build for snapshot-agent
# Supports multi-arch: linux/amd64, linux/arm64

# --- Build stage ---
FROM golang:1.26 AS builder

WORKDIR /workspace

# Copy the entire project to resolve internal dependencies
COPY . .

# Recreate go.work to ensure clean multi-module build
RUN rm -f go.work && \
    go work init . && \
    go work use ./pkg/accelerator-orchestrator && \
    go work use ./pkg/snapshot-agent

# Build the snapshot-agent
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /workspace/snapshot-agent ./cmd/snapshot-agent

# --- Runtime stage ---
FROM gcr.io/distroless/static:nonroot

WORKDIR /
COPY --from=builder /workspace/snapshot-agent .

USER 65532:65532

ENTRYPOINT ["/snapshot-agent"]
