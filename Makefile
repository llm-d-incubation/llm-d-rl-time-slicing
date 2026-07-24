# Project configuration
PROJECT_NAME ?= llm-d-rl-time-slicing
# Matches where CI publishes (ghcr.io/<owner>/<repo>/<component>); override
# REGISTRY for dev pushes to your own registry.
REGISTRY ?= ghcr.io/llm-d-incubation
ORCHESTRATOR_IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)/acceleratororchestrator
ORCHESTRATOR_DOCKERFILE ?= docker/acceleratororchestrator/Dockerfile
SNAPSHOT_AGENT_IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)/snapshot-agent
SNAPSHOT_AGENT_DOCKERFILE ?= docker/snapshot-agent/Dockerfile
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
# amd64-only, same as CI: the snapshot-agent needs the x86_64 cuda-checkpoint
# binary and a CGO build; the platform is adopted as a unit.
PLATFORMS ?= linux/amd64

# Go configuration
GOFLAGS ?=
LDFLAGS ?= -s -w -X main.version=$(VERSION)

# Tools
GOLANGCI_LINT_VERSION ?= v2.8.0
# Pinned to the same commit the container images bundle.
CUDA_CHECKPOINT_COMMIT ?= 00d5cce84c628088d6caa203fc4af40c1538b6f7

.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: build
build: ## Build the Go binary
	go build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o bin/$(PROJECT_NAME) ./cmd
	go build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o bin/snapshot-agent ./cmd/snapshot-agent

.PHONY: cuda-checkpoint
cuda-checkpoint: bin/cuda-checkpoint ## Fetch the pinned cuda-checkpoint binary into bin/ (x86_64)

bin/cuda-checkpoint:
	mkdir -p bin
	curl -fsSL -o bin/cuda-checkpoint https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/$(CUDA_CHECKPOINT_COMMIT)/bin/x86_64_Linux/cuda-checkpoint
	chmod +x bin/cuda-checkpoint

.PHONY: standalone
standalone: build cuda-checkpoint ## Build everything needed to run the agent in standalone mode
.PHONY: test
test: ## Run tests with race detection
	go test -race -count=1 ./...

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	go test -race -coverprofile=coverage.out -covermode=atomic ./...
	go tool cover -html=coverage.out -o coverage.html

.PHONY: lint
lint: lint-go lint-python ## Run all linters

.PHONY: lint-go
lint-go: ## Run Go linter (golangci-lint v2)
	golangci-lint run

.PHONY: lint-python
lint-python: ## Run Python linter (ruff) — skipped if no Python files found
	@if ls *.py **/*.py 2>/dev/null | head -1 > /dev/null 2>&1; then \
		ruff check . && ruff format --check .; \
	else \
		echo "No Python files found, skipping Python lint"; \
	fi

.PHONY: fmt
fmt: ## Format Go and Python code
	gofmt -w .
	@if ls *.py **/*.py 2>/dev/null | head -1 > /dev/null 2>&1; then \
		ruff format .; \
	fi

.PHONY: generate
generate: ## Run go generate
	go generate ./...

.PHONY: vet
vet: ## Run go vet
	go vet ./...

.PHONY: tidy
tidy: ## Run go mod tidy
	go mod tidy

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

##@ Container

.PHONY: image-build-orchestrator
image-build-orchestrator: ## Build acceleratororchestrator container image (local only)
	docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(ORCHESTRATOR_IMAGE):$(VERSION) \
		--tag $(ORCHESTRATOR_IMAGE):latest \
		-f $(ORCHESTRATOR_DOCKERFILE) \
		.

.PHONY: image-push-orchestrator
image-push-orchestrator: ## Build and push acceleratororchestrator container image
	docker buildx build \
		--platform $(PLATFORMS) \
		--push \
		--annotation "index:org.opencontainers.image.source=https://github.com/llm-d-incubation/$(PROJECT_NAME)" \
		--annotation "index:org.opencontainers.image.licenses=Apache-2.0" \
		--tag $(ORCHESTRATOR_IMAGE):$(VERSION) \
		--tag $(ORCHESTRATOR_IMAGE):latest \
		-f $(ORCHESTRATOR_DOCKERFILE) \
		.

.PHONY: snapshot-agent-image-build
snapshot-agent-image-build: ## Build snapshot-agent container image (local only)
	docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(SNAPSHOT_AGENT_IMAGE):$(VERSION) \
		--tag $(SNAPSHOT_AGENT_IMAGE):latest \
		-f $(SNAPSHOT_AGENT_DOCKERFILE) \
		.

.PHONY: snapshot-agent-image-push
snapshot-agent-image-push: ## Build and push snapshot-agent container image
	docker buildx build \
		--platform $(PLATFORMS) \
		--push \
		--annotation "index:org.opencontainers.image.source=https://github.com/llm-d-incubation/$(PROJECT_NAME)" \
		--annotation "index:org.opencontainers.image.licenses=Apache-2.0" \
		--tag $(SNAPSHOT_AGENT_IMAGE):$(VERSION) \
		--tag $(SNAPSHOT_AGENT_IMAGE):latest \
		-f $(SNAPSHOT_AGENT_DOCKERFILE) \
		.


##@ CI Helpers

.PHONY: ci-lint
ci-lint: ## CI: install and run golangci-lint
	@which golangci-lint > /dev/null 2>&1 || go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
	golangci-lint run

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf bin/ coverage.out coverage.html
