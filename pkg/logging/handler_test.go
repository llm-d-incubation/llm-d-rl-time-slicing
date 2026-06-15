package logging_test

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
)

func TestContextHandler(t *testing.T) {
	var buf bytes.Buffer
	jsonHandler := slog.NewJSONHandler(&buf, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	logger := slog.New(ctxHandler)

	ctx := context.Background()
	ctx = logging.WithServerMethod(ctx, "TestRoute")
	ctx = logging.WithJobID(ctx, "job-999")
	ctx = logging.WithGroupID(ctx, "group-888")
	ctx = logging.WithWorkerID(ctx, 42)

	logger.InfoContext(ctx, "Hello World")

	var data map[string]any
	if err := json.Unmarshal(buf.Bytes(), &data); err != nil {
		t.Fatalf("Failed to parse log output: %v", err)
	}

	if data["msg"] != "Hello World" {
		t.Errorf("Expected msg 'Hello World', got %v", data["msg"])
	}
	if data["ServerMethod"] != "TestRoute" {
		t.Errorf("Expected ServerMethod 'TestRoute', got %v", data["ServerMethod"])
	}
	if data["JobID"] != "job-999" {
		t.Errorf("Expected JobID 'job-999', got %v", data["JobID"])
	}
	if data["GroupID"] != "group-888" {
		t.Errorf("Expected GroupID 'group-888', got %v", data["GroupID"])
	}
	if data["WorkerID"] != float64(42) {
		t.Errorf("Expected WorkerID 42, got %v", data["WorkerID"])
	}
}
