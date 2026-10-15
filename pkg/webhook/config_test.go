package webhook_test

import (
	"strings"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/webhook"
)

func TestConfigFromFlags_Defaults(t *testing.T) {
	cfg := mustConfig(t, nil)
	want := webhook.Config{
		GuestSteering: webhook.SteeringPreferred, VirtualNodeLabel: "timeslice.io/virtual-node",
		DonorClientWiring: webhook.WiringNone, PodinfoPath: "/etc/timeslice/podinfo",
		GroupFormat: webhook.GroupFormatNsJobGroup, Port: 8443, CertDir: "/etc/timeslice-webhook/certs",
		RLIntegrationPath: webhook.DefaultRLIntegrationPath, DonorGPUMemory: webhook.DonorGPUMemoryAuto,
		AgentPort: webhook.DefaultAgentPort,
	}
	if *cfg != want {
		t.Errorf("defaults = %+v, want %+v", *cfg, want)
	}
}

func TestConfigFromFlags_Profiles(t *testing.T) {
	today := mustConfig(t, todayFlags)
	if today.DonorClientWiring != webhook.WiringEnv || today.OrchestratorAddr != orchAddr || today.VKUsername != vkUsername {
		t.Errorf("today = %+v", *today)
	}
	ns := mustConfig(t, nsFlags)
	if ns.GuestSteering != webhook.SteeringPreferred || ns.VirtualNodeLabel != "timeslice.io/guest" ||
		ns.DonorClientWiring != webhook.WiringDownward {
		t.Errorf("ns = %+v", *ns)
	}
}

func TestConfigFromFlags_Invalid(t *testing.T) {
	cases := map[string][]string{
		"--guest-steering":      {"--guest-steering=soft"},
		"--virtual-node-label":  {"--virtual-node-label="},
		"--donor-client-wiring": {"--donor-client-wiring=file"},
		"--orchestrator-addr":   {"--donor-client-wiring=env"},
		"--podinfo-path":        {"--donor-client-wiring=downward", "--podinfo-path=podinfo"},
		"--group-format":        {"--group-format=ns-job-group"},
		"--vk-service-account":  {"--vk-service-account=guest-kubelet"},
		"--port":                {"--port=0"},
		"--cert-dir":            {"--cert-dir="},
		"--agent-port":          {"--agent-port=70000"},
		"--rl-integration-path": {"--rl-integration-image=img", "--rl-integration-path=rel"},
		"unexpected arguments":  {"extra"},
		"not defined":           {"--no-such-flag"},
	}
	for want, args := range cases {
		_, err := webhook.ConfigFromFlags(args)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("ConfigFromFlags(%v) = %v, want an error naming %s", args, err, want)
		}
	}
}
