package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/controller"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/store"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}

func run() error {
	port := flag.Int("port", 50051, "The server port")
	kubeconfig := flag.String("kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	controllerWorkers := flag.Int("controller-workers", 1, "The number of workers for the controller")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var config *rest.Config
	var err error

	if *kubeconfig != "" {
		config, err = clientcmd.BuildConfigFromFlags("", *kubeconfig)
		if err != nil {
			return fmt.Errorf("failed to build config from kubeconfig flag: %w", err)
		}
	} else {
		config, err = rest.InClusterConfig()
		if err != nil {
			log.Printf("In-cluster config failed, trying default local kubeconfig: %v", err)
			loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
			configOverrides := &clientcmd.ConfigOverrides{}
			kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)
			config, err = kubeConfig.ClientConfig()
			if err != nil {
				return fmt.Errorf("failed to load kubernetes config: %w", err)
			}
		}
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
	snapshotAgentStore := store.NewGRPCSnapshotAgentStore(0)
	ctrl := controller.NewController(
		clientset,
		nodeInformerFactory.Core().V1().Nodes(),
		podInformerFactory.Core().V1().Pods(),
		groupStore,
		jobStore,
		snapshotAgentStore,
	)

	// Start informers
	nodeInformerFactory.Start(ctx.Done())
	podInformerFactory.Start(ctx.Done())

	log.Printf("Starting Accelerator Orchestrator server...")
	return server.StartServer(ctx, *port, ctrl, *controllerWorkers)
}
