# --- Build stage ---
FROM golang:1.26.2 AS builder

WORKDIR /workspace

# Copy the entire repository
COPY . .

# Recreate go.work to ensure clean multi-module build
RUN rm -f go.work && \
    go work init . && \
    go work use ./pkg/accelerator-orchestrator && \
    go work use ./pkg/snapshot-agent

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /workspace/acceleratororchestrator ./cmd/acceleratororchestrator

# --- Runtime stage ---
FROM gcr.io/distroless/static:nonroot

WORKDIR /
COPY --from=builder /workspace/acceleratororchestrator .

USER 65532:65532

# Expose gRPC port
EXPOSE 50051

ENTRYPOINT ["/acceleratororchestrator"]