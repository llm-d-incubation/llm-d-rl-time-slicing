package logging

import (
	"context"
	"log/slog"
)

type (
	serverMethodKeyType struct{}
	jobIDKeyType        struct{}
	groupIDKeyType      struct{}
	workerIDKeyType     struct{}
	nodeNameKeyType     struct{}
	operationIDKeyType  struct{}
)

// WithServerMethod returns a new context with the server method name.
func WithServerMethod(ctx context.Context, method string) context.Context {
	return context.WithValue(ctx, serverMethodKeyType{}, method)
}

// WithJobID returns a new context with the Job ID.
func WithJobID(ctx context.Context, jobID string) context.Context {
	return context.WithValue(ctx, jobIDKeyType{}, jobID)
}

// WithGroupID returns a new context with the Group ID.
func WithGroupID(ctx context.Context, groupID string) context.Context {
	return context.WithValue(ctx, groupIDKeyType{}, groupID)
}

// WithWorkerID returns a new context with the Worker ID.
func WithWorkerID(ctx context.Context, workerID int) context.Context {
	return context.WithValue(ctx, workerIDKeyType{}, workerID)
}

// WithNodeName returns a new context with the Node Name.
func WithNodeName(ctx context.Context, nodeName string) context.Context {
	return context.WithValue(ctx, nodeNameKeyType{}, nodeName)
}

// WithOperationID returns a new context with the Operation ID.
func WithOperationID(ctx context.Context, operationID string) context.Context {
	return context.WithValue(ctx, operationIDKeyType{}, operationID)
}

// ContextHandler is a slog.Handler that extracts logging metadata from context
// and adds it to the log record.
type ContextHandler struct {
	slog.Handler
}

// NewContextHandler creates a new ContextHandler wrapping the provided handler.
func NewContextHandler(h slog.Handler) *ContextHandler {
	return &ContextHandler{Handler: h}
}

//nolint:gocritic // slog.Handler.Handle signature requires passing Record by value
func (h *ContextHandler) Handle(ctx context.Context, r slog.Record) error {
	if method, ok := ctx.Value(serverMethodKeyType{}).(string); ok && method != "" {
		r.AddAttrs(slog.String("ServerMethod", method))
	}
	if jobID, ok := ctx.Value(jobIDKeyType{}).(string); ok && jobID != "" {
		r.AddAttrs(slog.String("JobID", jobID))
	}
	if groupID, ok := ctx.Value(groupIDKeyType{}).(string); ok && groupID != "" {
		r.AddAttrs(slog.String("GroupID", groupID))
	}
	if workerID, ok := ctx.Value(workerIDKeyType{}).(int); ok {
		r.AddAttrs(slog.Int("WorkerID", workerID))
	}
	if nodeName, ok := ctx.Value(nodeNameKeyType{}).(string); ok && nodeName != "" {
		r.AddAttrs(slog.String("NodeName", nodeName))
	}
	if operationID, ok := ctx.Value(operationIDKeyType{}).(string); ok && operationID != "" {
		r.AddAttrs(slog.String("OperationID", operationID))
	}
	return h.Handler.Handle(ctx, r)
}

// WithAttrs returns a new ContextHandler with the provided attributes.
func (h *ContextHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &ContextHandler{Handler: h.Handler.WithAttrs(attrs)}
}

// WithGroup returns a new ContextHandler with the provided group.
func (h *ContextHandler) WithGroup(name string) slog.Handler {
	return &ContextHandler{Handler: h.Handler.WithGroup(name)}
}
