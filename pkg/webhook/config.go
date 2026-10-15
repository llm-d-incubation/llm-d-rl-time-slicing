// Package webhook is the timeslice admission webhook: it derives the timeslice fields of a pod
// from the two boolean labels timeslice.io/donor and timeslice.io/guest.
//
// Donors get their job-id and group derived from the KubeRay labels (or keep explicit ones), and
// optionally the orchestrator client wiring. Guests get the guest toleration and the steering to
// the virtual node. Pods without either label never reach the webhook (the objectSelector of each
// webhook entry) and would pass through unchanged if they did.
//
// Admit is a pure function: no Kubernetes client, no I/O. The HTTP handler in this package wraps
// it with the AdmissionReview envelope, one log line per request and Prometheus metrics.
package webhook

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"path"
	"strings"
)

// Guest steering modes (--guest-steering).
const (
	// SteeringRequired sets nodeSelector <virtual-node-label>=true on the guest: it runs only on
	// lent GPUs and stays Pending while none is lent.
	SteeringRequired = "required"
	// SteeringPreferred (the default) adds a weight-100 preferred node affinity term
	// <virtual-node-label> In [true]: the guest prefers lent GPUs and otherwise schedules on a
	// regular GPU node, where it stays (placement is decided once, at scheduling).
	SteeringPreferred = "preferred"
)

// Donor client wiring modes (--donor-client-wiring).
const (
	// WiringEnv injects TIMESLICE_JOB_ID, TIMESLICE_GROUP and TIMESLICE_ORCH_ADDR into every container.
	WiringEnv = "env"
	// WiringDownward mounts the pod labels through the downward API at --podinfo-path.
	WiringDownward = "downward"
	// WiringNone injects nothing: the donor only gets its identity labels.
	WiringNone = "none"
)

// GroupFormatNsJobGroup is the only supported --group-format: <namespace>.<job-id>.<ray group>.
const GroupFormatNsJobGroup = "ns.job.group"

// Config is the webhook configuration, built by ConfigFromFlags.
type Config struct {
	// GuestSteering is SteeringPreferred (default) or SteeringRequired.
	GuestSteering string
	// VirtualNodeLabel is the node label key the guest is steered to (value "true").
	VirtualNodeLabel string
	// DonorClientWiring is WiringEnv, WiringDownward or WiringNone.
	DonorClientWiring string
	// OrchestratorAddr is the TIMESLICE_ORCH_ADDR value for WiringEnv.
	OrchestratorAddr string
	// PodinfoPath is where WiringDownward mounts the podinfo volume.
	PodinfoPath string
	// GroupFormat is GroupFormatNsJobGroup.
	GroupFormat string
	// AgentPort is the snapshot agent's port, injected as TIMESLICE_AGENT_PORT into the Ray pods
	// of an application-aware donor (timeslice.io/backend: app_channel).
	AgentPort int
	// RLIntegrationImage, when set, is the image of the init container injected into donor pods and
	// pods labelled timeslice.io/rl-integration=true: it copies the timeslice Python packages into
	// an emptyDir that every container mounts at RLIntegrationPath, with PYTHONPATH pointing there.
	RLIntegrationImage string
	// RLIntegrationPath is where the containers mount the injected packages (absolute).
	RLIntegrationPath string
	// RLIntegrationVerl also injects the verl build the init image carries (<path>/verl, after
	// <path>/python on PYTHONPATH): a verl with the fully-async lifecycle hooks, trainer registry
	// and per-pool placement-group resources the timeslice trainer needs, so the RL image can
	// ship a stock verl. Off: the RL image must provide such a verl itself.
	RLIntegrationVerl bool
	// DonorGPUMemory is how much a donor container's memory limit grows per GPU it holds
	// (nvidia.com/gpu), because the trainer's GPU memory is checkpointed into the container's
	// memory while the GPU is lent. "auto" takes the GPU model from the pod's node selection
	// (GPUMemoryByModel) and falls back to DefaultDonorGPUMemory; "0" turns the raise off;
	// anything else is a quantity (e.g. 80Gi).
	DonorGPUMemory string
	// VKUsername is the virtual kubelet's user name (system:serviceaccount:<ns>:<name>). Its
	// requests pass through untouched (mirror pods). Empty means no identity is trusted.
	VKUsername string
	// Port is the HTTPS port the server listens on.
	Port int
	// CertDir holds tls.crt and tls.key, mounted from a Secret.
	CertDir string
}

// ConfigFromFlags parses the webhook flags (without the program name) and validates them.
func ConfigFromFlags(args []string) (Config, error) {
	var cfg Config
	var vkSA string
	fs := flag.NewFlagSet("timeslice-webhook", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&cfg.GuestSteering, "guest-steering", SteeringPreferred,
		"How guests are steered to the virtual node: preferred (node affinity; falls back to regular GPU nodes) "+
			"or required (nodeSelector; lent GPUs only).")
	fs.StringVar(&cfg.VirtualNodeLabel, "virtual-node-label", "timeslice.io/virtual-node",
		"Node label key (value \"true\") that guests are steered to.")
	fs.StringVar(&cfg.DonorClientWiring, "donor-client-wiring", WiringNone,
		"Client wiring injected into donors: env (TIMESLICE_* env), downward (podinfo volume) or none.")
	fs.StringVar(&cfg.OrchestratorAddr, "orchestrator-addr", "",
		"Orchestrator host:port injected as TIMESLICE_ORCH_ADDR. Required with --donor-client-wiring=env.")
	fs.StringVar(&cfg.PodinfoPath, "podinfo-path", "/etc/timeslice/podinfo",
		"Mount path of the downward-API podinfo volume (--donor-client-wiring=downward).")
	fs.StringVar(&cfg.GroupFormat, "group-format", GroupFormatNsJobGroup,
		"Format of the derived timeslice.io/group value. Only ns.job.group is supported.")
	fs.IntVar(&cfg.AgentPort, "agent-port", DefaultAgentPort,
		"Snapshot agent port injected as TIMESLICE_AGENT_PORT into the Ray pods of an application-aware donor.")
	fs.StringVar(&cfg.RLIntegrationImage, "rl-integration-image", "",
		"Image of the RL integration init container injected into donors and timeslice.io/rl-integration=true pods; "+
			"empty injects nothing.")
	fs.StringVar(&cfg.RLIntegrationPath, "rl-integration-path", DefaultRLIntegrationPath,
		"Mount path of the injected RL integration packages; PYTHONPATH gets <path>/python first.")
	fs.BoolVar(&cfg.RLIntegrationVerl, "rl-integration-verl", false,
		"Also inject the verl build carried by the RL integration image (<path>/verl on PYTHONPATH after <path>/python).")
	fs.StringVar(&cfg.DonorGPUMemory, "donor-gpu-memory", DonorGPUMemoryAuto,
		"Memory added to a donor container's memory limit per nvidia.com/gpu it holds (room for the GPU checkpoint): "+
			"auto (by GPU model, fallback "+DefaultDonorGPUMemory+"), 0 (off) or a quantity.")
	fs.StringVar(&vkSA, "vk-service-account", "",
		"<namespace>:<name> of the virtual kubelet ServiceAccount; its pods (mirrors) pass through untouched.")
	fs.IntVar(&cfg.Port, "port", 8443, "HTTPS port.")
	fs.StringVar(&cfg.CertDir, "cert-dir", "/etc/timeslice-webhook/certs", "Directory holding tls.crt and tls.key.")
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	if fs.NArg() > 0 {
		return Config{}, fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	if vkSA != "" {
		ns, name, ok := strings.Cut(vkSA, ":")
		if !ok || ns == "" || name == "" || strings.Contains(name, ":") {
			return Config{}, fmt.Errorf("--vk-service-account must be <namespace>:<name>, got %q", vkSA)
		}
		cfg.VKUsername = "system:serviceaccount:" + ns + ":" + name
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	var errs []error
	switch c.GuestSteering {
	case SteeringRequired, SteeringPreferred:
	default:
		errs = append(errs, fmt.Errorf("--guest-steering must be required or preferred, got %q", c.GuestSteering))
	}
	if c.VirtualNodeLabel == "" {
		errs = append(errs, errors.New("--virtual-node-label must not be empty"))
	}
	switch c.DonorClientWiring {
	case WiringNone:
	case WiringEnv:
		if c.OrchestratorAddr == "" {
			errs = append(errs, errors.New("--orchestrator-addr is required with --donor-client-wiring=env"))
		}
	case WiringDownward:
		if !path.IsAbs(c.PodinfoPath) {
			errs = append(errs, fmt.Errorf("--podinfo-path must be absolute, got %q", c.PodinfoPath))
		}
	default:
		errs = append(errs, fmt.Errorf("--donor-client-wiring must be env, downward or none, got %q", c.DonorClientWiring))
	}
	if c.RLIntegrationImage != "" && (!path.IsAbs(c.RLIntegrationPath) || path.Clean(c.RLIntegrationPath) == "/") {
		errs = append(errs, fmt.Errorf("--rl-integration-path must be an absolute path other than /, got %q", c.RLIntegrationPath))
	}
	if _, _, err := parseDonorGPUMemory(c.DonorGPUMemory); err != nil {
		errs = append(errs, fmt.Errorf("--donor-gpu-memory: %w", err))
	}
	if c.GroupFormat != GroupFormatNsJobGroup {
		errs = append(errs, fmt.Errorf("--group-format must be %s, got %q", GroupFormatNsJobGroup, c.GroupFormat))
	}
	if c.Port < 1 || c.Port > 65535 {
		errs = append(errs, fmt.Errorf("--port must be 1-65535, got %d", c.Port))
	}
	if c.AgentPort < 1 || c.AgentPort > 65535 {
		errs = append(errs, fmt.Errorf("--agent-port must be 1-65535, got %d", c.AgentPort))
	}
	if c.CertDir == "" {
		errs = append(errs, errors.New("--cert-dir must not be empty"))
	}
	return errors.Join(errs...)
}
