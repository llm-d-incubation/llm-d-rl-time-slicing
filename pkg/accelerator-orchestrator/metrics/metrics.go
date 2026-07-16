package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

// durationBuckets spans 1ms to 10min: operation times range from near-instant
// (uncontended lock acquisition) to minutes (snapshot/restore of large models).
var durationBuckets = []float64{0.001, 0.01, 0.1, 1, 10, 60, 300, 600}

var (
	// QueueDepth tracks the current number of jobs waiting in the queue for a group lock.
	QueueDepth = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "accelerator_orchestrator_queue_depth",
			Help: "Current number of jobs waiting in the queue for a group lock.",
		},
		[]string{"group_id"},
	)

	// AcquireWaitDuration tracks the time spent by a job waiting in Acquire() until lock acquisition & context restoration.
	AcquireWaitDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "accelerator_orchestrator_acquire_wait_duration_seconds",
			Help:    "Time spent by a job waiting in Acquire() until lock acquisition and context restoration.",
			Buckets: durationBuckets,
		},
		[]string{"group_id"},
	)

	// AgentOperationDuration tracks the duration of snapshot and restore operations as reported by snapshot agents.
	AgentOperationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "accelerator_orchestrator_agent_operation_duration_seconds",
			Help:    "Duration of snapshot and restore operations as reported by snapshot agents.",
			Buckets: durationBuckets,
		},
		[]string{"group_id", "job_id", "node", "operation"},
	)

	// DeferredSnapshotsTotal tracks the number of times a snapshot was deferred during Yield() due to zero waiters.
	DeferredSnapshotsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "accelerator_orchestrator_deferred_snapshots_total",
			Help: "Number of times a snapshot was deferred during Yield() due to zero waiters.",
		},
		[]string{"group_id"},
	)
)

// CleanupGroup removes gauge series labeled with the given group so stale
// values don't persist in /metrics after the group is deleted. Cumulative
// metrics (histograms, counters) are left intact so group ID reuse doesn't
// appear as a counter reset.
func CleanupGroup(groupID string) {
	QueueDepth.DeletePartialMatch(prometheus.Labels{"group_id": groupID})
}

// Register registers all accelerator orchestrator Prometheus metrics with the default registry.
func Register() {
	prometheus.MustRegister(
		QueueDepth,
		AcquireWaitDuration,
		AgentOperationDuration,
		DeferredSnapshotsTotal,
	)
}
