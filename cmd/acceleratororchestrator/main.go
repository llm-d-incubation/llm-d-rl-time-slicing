package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/infrastructure"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
)

func main() {
	if err := run(); err != nil {
		slog.Error("Failed to run server", "error", err)
		os.Exit(1)
	}
}

func run() error {
	// Initialize slog with ContextHandler
	jsonHandler := slog.NewJSONHandler(os.Stdout, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	slog.SetDefault(slog.New(ctxHandler))

	port := flag.Int("port", 50051, "The server port")
	kubeconfig := flag.String("kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	controllerWorkers := flag.Int("controller-workers", 1, "The number of workers for the controller")
	snapshotAgentPort := flag.Int("snapshot-agent-port", 9001, "The default port for snapshot agents")
	resyncPeriod := flag.Duration("resync-period", 30*time.Second, "The period for periodic resync of agent states")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	config, err := buildKubeConfig(*kubeconfig)
	if err != nil {
		return fmt.Errorf("failed to load kubernetes config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	nodeInformerFactory := informers.NewSharedInformerFactory(clientset, time.Minute*30)
	podInformerFactory := informers.NewSharedInformerFactoryWithOptions(clientset, time.Minute*30,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.LabelSelector = "timeslice.io/group"
		}),
	)

	lockStore := store.NewConfigMapLockStore(clientset)
	groupStore := store.NewGroupStore(lockStore)
	jobStore := store.NewJobStore()
	snapshotAgentStore := store.NewGRPCSnapshotAgentStore(0, *snapshotAgentPort)
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{
			Name: "groups",
		},
	)

	infraOrch := infrastructure.NewKubernetesOrchestrator(
		nodeInformerFactory.Core().V1().Nodes(),
		podInformerFactory.Core().V1().Pods(),
		groupStore,
		jobStore,
		snapshotAgentStore,
	)
	if err := infraOrch.Start(ctx, queue); err != nil {
		return fmt.Errorf("failed to start infrastructure orchestrator: %w", err)
	}

	ctrl := controller.NewController(
		groupStore,
		jobStore,
		queue,
		infraOrch,
		snapshotAgentStore,
	)
	ctrl.ResyncPeriod = *resyncPeriod

	// Start informers
	nodeInformerFactory.Start(ctx.Done())
	podInformerFactory.Start(ctx.Done())

	slog.InfoContext(ctx, "Starting Accelerator Orchestrator server")
	return server.StartServer(ctx, *port, ctrl, groupStore, jobStore, *controllerWorkers)
}

func buildKubeConfig(kubeconfigPath string) (*rest.Config, error) {
	if kubeconfigPath != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	}

	config, err := rest.InClusterConfig()
	if err == nil {
		return config, nil
	}

	slog.Info("In-cluster config failed, trying default local kubeconfig", "error", err)
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	configOverrides := &clientcmd.ConfigOverrides{}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)
	return kubeConfig.ClientConfig()
}
