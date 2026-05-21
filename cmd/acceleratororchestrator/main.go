package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/accelerator-orchestrator/server"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}

func run() error {
	port := flag.Int("port", 50051, "The server port")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	log.Printf("Starting Accelerator Orchestrator server...")
	return server.StartServer(ctx, *port)
}
