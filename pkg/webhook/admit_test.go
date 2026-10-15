package webhook_test

import (
	"bytes"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/webhook"
)

func kuberayDonor() *corev1.Pod {
	return newPod(map[string]string{
		"app": "rl", webhook.LabelDonor: "true",
		webhook.LabelRayCluster: "rc-7f2x9", webhook.LabelRayGroup: "trainers", "ray.io/node-type": "worker",
	})
}

func explicitDonor() *corev1.Pod {
	return newPod(map[string]string{
		"app": "rl", webhook.LabelDonor: "true", webhook.LabelJobID: "rl-trainer", webhook.LabelGroup: "trainers",
	})
}

func guest(opts ...func(*corev1.Pod)) *corev1.Pod {
	return newPod(map[string]string{"app": "vllm", webhook.LabelGuest: "true"}, opts...)
}

func withGuestToleration(pod *corev1.Pod) {
	pod.Spec.Tolerations = append(pod.Spec.Tolerations, corev1.Toleration{
		Key: webhook.GuestTaintKey, Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule,
	})
}

func TestAdmit_UnlabelledPodsPassThrough(t *testing.T) {
	sampler := newPod(map[string]string{"app": "rl", webhook.LabelRayCluster: "rc-7f2x9", webhook.LabelRayGroup: "samplers"})
	gpuTol := newPod(map[string]string{"app": "x"}, func(p *corev1.Pod) {
		p.Spec.Tolerations = []corev1.Toleration{{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists}}
	})
	for _, flags := range [][]string{todayFlags, nsFlags} {
		cfg := mustConfig(t, flags)
		for _, pod := range []*corev1.Pod{newPod(map[string]string{"app": "x"}), newPod(nil), sampler, gpuTol} {
			res := create(t, cfg, pod)
			mustAllow(t, res)
			if len(res.resp.Patch) != 0 {
				t.Errorf("unlabelled pod %v patched: %s", pod.Labels, res.resp.Patch)
			}
		}
	}
}

func TestAdmit_DonorDerivedFromKubeRay_Env(t *testing.T) {
	pod := mustAllow(t, create(t, mustConfig(t, todayFlags), kuberayDonor()))
	if got := pod.Labels[webhook.LabelJobID]; got != "rc-7f2x9" {
		t.Errorf("job-id = %q, want rc-7f2x9", got)
	}
	wantGroup := testNS + ".rc-7f2x9.trainers"
	if got := pod.Labels[webhook.LabelGroup]; got != wantGroup {
		t.Errorf("group = %q, want %q", got, wantGroup)
	}
	want := map[string]string{webhook.EnvJobID: "rc-7f2x9", webhook.EnvGroup: wantGroup, webhook.EnvOrchAddr: orchAddr}
	for name, val := range want {
		if got, ok := env(&pod.Spec.Containers[0], name); !ok || got != val {
			t.Errorf("env %s = %q (set %v), want %q", name, got, ok, val)
		}
	}
	if len(pod.Spec.Volumes) != 0 {
		t.Errorf("env wiring added volumes: %v", pod.Spec.Volumes)
	}
}

func TestAdmit_DonorDerivedFromKubeRay_Downward(t *testing.T) {
	pod := mustAllow(t, create(t, mustConfig(t, nsFlags), kuberayDonor()))
	if pod.Labels[webhook.LabelJobID] != "rc-7f2x9" || pod.Labels[webhook.LabelGroup] != testNS+".rc-7f2x9.trainers" {
		t.Errorf("identity labels = %v", pod.Labels)
	}
	if len(pod.Spec.Volumes) != 1 || pod.Spec.Volumes[0].Name != webhook.PodinfoVolume ||
		pod.Spec.Volumes[0].DownwardAPI == nil ||
		pod.Spec.Volumes[0].DownwardAPI.Items[0].FieldRef.FieldPath != "metadata.labels" {
		t.Fatalf("podinfo volume = %+v", pod.Spec.Volumes)
	}
	mounts := pod.Spec.Containers[0].VolumeMounts
	if len(mounts) != 1 || mounts[0].MountPath != "/etc/timeslice/podinfo" || !mounts[0].ReadOnly {
		t.Errorf("podinfo mount = %+v", mounts)
	}
	if len(pod.Spec.Containers[0].Env) != 0 {
		t.Errorf("downward wiring added env: %v", pod.Spec.Containers[0].Env)
	}
}

func TestAdmit_DonorWiringNone(t *testing.T) {
	res := create(t, mustConfig(t, nil), kuberayDonor())
	pod := mustAllow(t, res)
	if res.patchOps(t) != 2 || len(pod.Spec.Volumes) != 0 || len(pod.Spec.Containers[0].Env) != 0 {
		t.Errorf("wiring none: %d ops, pod %+v", res.patchOps(t), pod.Spec)
	}
}

func TestAdmit_DonorExplicitIdentityKept(t *testing.T) {
	for _, flags := range [][]string{todayFlags, nsFlags} {
		pod := mustAllow(t, create(t, mustConfig(t, flags), explicitDonor()))
		if pod.Labels[webhook.LabelJobID] != "rl-trainer" || pod.Labels[webhook.LabelGroup] != "trainers" {
			t.Errorf("explicit identity changed: %v", pod.Labels)
		}
	}
	pod := mustAllow(t, create(t, mustConfig(t, todayFlags), explicitDonor()))
	if got, _ := env(&pod.Spec.Containers[0], webhook.EnvGroup); got != "trainers" {
		t.Errorf("TIMESLICE_GROUP = %q, want the explicit group", got)
	}
}

func TestAdmit_DonorMixedIdentity(t *testing.T) {
	pod := kuberayDonor()
	pod.Labels[webhook.LabelJobID] = "my-job"
	got := mustAllow(t, create(t, mustConfig(t, nsFlags), pod))
	if got.Labels[webhook.LabelJobID] != "my-job" || got.Labels[webhook.LabelGroup] != testNS+".rc-7f2x9.trainers" {
		t.Errorf("labels = %v", got.Labels)
	}
}

func TestAdmit_Denials(t *testing.T) {
	cases := []struct {
		name string
		pod  *corev1.Pod
		rule string
	}{
		{"donor without identity", newPod(map[string]string{webhook.LabelDonor: "true"}), "[R4]"},
		{"donor with cluster but no ray group", newPod(map[string]string{
			webhook.LabelDonor: "true", webhook.LabelRayCluster: "rc",
		}), "[R4]"},
		{"donor with group only", newPod(map[string]string{webhook.LabelDonor: "true", webhook.LabelGroup: "g"}), "[R4]"},
		{"group without job-id", newPod(map[string]string{webhook.LabelGroup: "trainers"}), "[W6]"},
		{"donor and guest", newPod(map[string]string{
			webhook.LabelDonor: "true", webhook.LabelGuest: "true",
			webhook.LabelRayCluster: "rc", webhook.LabelRayGroup: "g",
		}), "[R19]"},
		{"guest with group", guest(func(p *corev1.Pod) { p.Labels[webhook.LabelGroup] = "trainers" }), "[W9]"},
		{"guest with job-id", guest(func(p *corev1.Pod) { p.Labels[webhook.LabelJobID] = "x" }), "[W9]"},
		{"guest with role", guest(func(p *corev1.Pod) { p.Labels[webhook.LabelRole] = "background" }), "[W9]"},
		{"guest with pod claim", guest(func(p *corev1.Pod) {
			name := "shared-claim"
			p.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "accelerator", ResourceClaimName: &name}}
		}), "[W2]"},
		{"guest with container claim", guest(func(p *corev1.Pod) {
			p.Spec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "accelerator"}}
		}), "[W2]"},
		{"background role by a user", newPod(map[string]string{
			webhook.LabelRole: "background", webhook.LabelGroup: "trainers", webhook.LabelJobID: "guest-uid-1",
		}), "[W10]"},
		{"donor with role", donorWithRole(), "[W10]"},
		{
			"unlabelled pod asks for the pooled shadow",
			newPod(map[string]string{"app": "x"}, withShadow("timeslice.io/gpu-shadow", false)), "[W11]",
		},
		{"donor asks for a per-GPU shadow limit", explicitDonorWith(withShadow("timeslice.io/gpu-shadow-0", true)), "[W11]"},
		{"guest asks for the shadow in an init container", guest(func(p *corev1.Pod) {
			p.Spec.InitContainers = []corev1.Container{{Name: "i", Image: "x"}}
			p.Spec.InitContainers[0].Resources.Requests = corev1.ResourceList{"timeslice.io/gpu-shadow": resource.MustParse("1")}
		}), "[W11]"},
		{"privileged guest", guest(withPrivileged(false)), "[W12]"},
		{"privileged guest init container", guest(withPrivileged(true)), "[W12]"},
	}
	for _, flags := range [][]string{todayFlags, nsFlags} {
		cfg := mustConfig(t, flags)
		for _, tc := range cases {
			res := create(t, cfg, tc.pod)
			if res.resp.Allowed {
				t.Errorf("%s: allowed, want %s denial", tc.name, tc.rule)
				continue
			}
			if !strings.HasPrefix(res.message(), tc.rule+" ") {
				t.Errorf("%s: message %q, want prefix %q", tc.name, res.message(), tc.rule)
			}
			if res.resp.Result.Code != 403 {
				t.Errorf("%s: code %d, want 403", tc.name, res.resp.Result.Code)
			}
		}
	}
}

func withShadow(name corev1.ResourceName, limit bool) func(*corev1.Pod) {
	return func(p *corev1.Pod) {
		rl := corev1.ResourceList{name: resource.MustParse("1")}
		if limit {
			p.Spec.Containers[0].Resources.Limits = rl
		} else {
			p.Spec.Containers[0].Resources.Requests = rl
		}
	}
}

func withPrivileged(initContainer bool) func(*corev1.Pod) {
	return func(p *corev1.Pod) {
		yes := true
		c := corev1.Container{Name: "priv", Image: "x", SecurityContext: &corev1.SecurityContext{Privileged: &yes}}
		if initContainer {
			p.Spec.InitContainers = append(p.Spec.InitContainers, c)
		} else {
			p.Spec.Containers = append(p.Spec.Containers, c)
		}
	}
}

func explicitDonorWith(opt func(*corev1.Pod)) *corev1.Pod {
	pod := explicitDonor()
	opt(pod)
	return pod
}

// The virtual kubelet's mirror pods request the shadow resource; only they may (W11). A guest
// that is explicitly not privileged is admitted (W12 looks at privileged=true only).
func TestAdmit_ShadowOnlyFromVirtualKubelet(t *testing.T) {
	mirror := newPod(map[string]string{
		webhook.LabelRole: "background", webhook.LabelGroup: "trainers", webhook.LabelJobID: "g-uid-a1",
	}, withShadow("timeslice.io/gpu-shadow", false))
	notPriv := guest(func(p *corev1.Pod) {
		no := false
		p.Spec.Containers[0].SecurityContext = &corev1.SecurityContext{Privileged: &no}
	})
	for _, flags := range [][]string{todayFlags, nsFlags} {
		cfg := mustConfig(t, flags)
		mustAllow(t, admit(t, cfg, mirror, vkUsername, admissionv1.Create))
		if res := create(t, cfg, mirror); res.resp.Allowed || !strings.HasPrefix(res.message(), "[W11] ") {
			t.Errorf("the mirror from a user: allowed %t, message %q; want W11", res.resp.Allowed, res.message())
		}
		mustAllow(t, create(t, cfg, notPriv))
		// nvidia.com/gpu and look-alike names are not shadow resources.
		mustAllow(t, create(t, cfg, newPod(map[string]string{"app": "x"}, withShadow("nvidia.com/gpu", true))))
		mustAllow(t, create(t, cfg, newPod(map[string]string{"app": "x"}, withShadow("timeslice.io/gpu-shadowx", true))))
	}
	for _, name := range []corev1.ResourceName{"timeslice.io/gpu-shadow", "timeslice.io/gpu-shadow-3"} {
		if !webhook.IsShadowResource(name) {
			t.Errorf("%s is a shadow resource", name)
		}
	}
}

func donorWithRole() *corev1.Pod {
	pod := explicitDonor()
	pod.Labels[webhook.LabelRole] = "background"
	return pod
}

func TestAdmit_MirrorFromVirtualKubeletPassesThrough(t *testing.T) {
	mirror := newPod(map[string]string{
		"timeslice.io/mirror-of": "00000000-0000-0000-0000-000000000001",
		webhook.LabelRole:        "background", webhook.LabelGroup: "trainers",
		webhook.LabelJobID: "00000000-0000-0000-0000-000000000001-a1",
	})
	for _, flags := range [][]string{todayFlags, nsFlags} {
		cfg := mustConfig(t, flags)
		res := admit(t, cfg, mirror, vkUsername, admissionv1.Create)
		mustAllow(t, res)
		if len(res.resp.Patch) != 0 {
			t.Errorf("mirror patched: %s", res.resp.Patch)
		}
		if create(t, cfg, mirror).resp.Allowed {
			t.Error("the same pod from a user was allowed; want W10")
		}
	}
}

func TestAdmit_GuestRequiredSteering(t *testing.T) {
	pod := mustAllow(t, create(t, mustConfig(t, todayFlags), guest()))
	if countTolerations(pod, webhook.GuestTaintKey) != 1 {
		t.Errorf("tolerations = %v", pod.Spec.Tolerations)
	}
	tol := pod.Spec.Tolerations[0]
	if tol.Operator != corev1.TolerationOpExists || tol.Effect != corev1.TaintEffectNoSchedule {
		t.Errorf("toleration = %+v", tol)
	}
	if pod.Spec.NodeSelector["timeslice.io/virtual-node"] != "true" {
		t.Errorf("nodeSelector = %v", pod.Spec.NodeSelector)
	}
	if pod.Spec.Affinity != nil {
		t.Errorf("required mode added affinity: %+v", pod.Spec.Affinity)
	}
	if _, ok := pod.Labels[webhook.LabelGroup]; ok || len(pod.Spec.ResourceClaims) != 0 || len(pod.Spec.ReadinessGates) != 0 {
		t.Errorf("guest got group, claims or gates: %+v", pod)
	}
}

func TestAdmit_GuestPreferredSteering(t *testing.T) {
	pod := mustAllow(t, create(t, mustConfig(t, nsFlags), guest()))
	if countTolerations(pod, webhook.GuestTaintKey) != 1 {
		t.Errorf("tolerations = %v", pod.Spec.Tolerations)
	}
	terms := preferredTerms(pod)
	if len(terms) != 1 || terms[0].Weight != 100 {
		t.Fatalf("preferred terms = %+v", terms)
	}
	req := terms[0].Preference.MatchExpressions
	if len(req) != 1 || req[0].Key != "timeslice.io/guest" || req[0].Operator != corev1.NodeSelectorOpIn ||
		len(req[0].Values) != 1 || req[0].Values[0] != "true" {
		t.Errorf("match expressions = %+v", req)
	}
	if pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil || len(pod.Spec.NodeSelector) != 0 {
		t.Errorf("preferred mode added a hard constraint: %+v %v", pod.Spec.Affinity, pod.Spec.NodeSelector)
	}
}

// The default steering is preferred: a guest may fall back to a regular GPU node.
func TestAdmit_GuestDefaultSteering(t *testing.T) {
	pod := mustAllow(t, create(t, mustConfig(t, nil), guest()))
	terms := preferredTerms(pod)
	if len(terms) != 1 || len(terms[0].Preference.MatchExpressions) != 1 ||
		terms[0].Preference.MatchExpressions[0].Key != "timeslice.io/virtual-node" {
		t.Fatalf("preferred terms = %+v", terms)
	}
	if len(pod.Spec.NodeSelector) != 0 || pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		t.Errorf("default steering added a hard constraint: %+v %v", pod.Spec.Affinity, pod.Spec.NodeSelector)
	}
}

func TestAdmit_GuestKeepsUserFields(t *testing.T) {
	pod := guest(func(p *corev1.Pod) {
		p.Spec.Tolerations = []corev1.Toleration{{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists}}
		p.Spec.NodeSelector = map[string]string{"pool": "cpu"}
		p.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{{
				Weight: 10, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: "zone", Operator: corev1.NodeSelectorOpIn, Values: []string{"a"}},
				}},
			}},
		}}
	})
	today := mustAllow(t, create(t, mustConfig(t, todayFlags), pod))
	if len(today.Spec.Tolerations) != 2 || today.Spec.NodeSelector["pool"] != "cpu" ||
		today.Spec.NodeSelector["timeslice.io/virtual-node"] != "true" {
		t.Errorf("required: %+v", today.Spec)
	}
	ns := mustAllow(t, create(t, mustConfig(t, nsFlags), pod))
	if terms := preferredTerms(ns); len(terms) != 2 || terms[0].Weight != 10 || terms[1].Weight != 100 {
		t.Errorf("preferred terms = %+v", terms)
	}
	emptyAffinity := guest(func(p *corev1.Pod) { p.Spec.Affinity = &corev1.Affinity{} })
	if terms := preferredTerms(mustAllow(t, create(t, mustConfig(t, nsFlags), emptyAffinity))); len(terms) != 1 {
		t.Errorf("empty affinity: preferred terms = %+v", terms)
	}
}

func TestAdmit_GuestPrefilledIsIdempotent(t *testing.T) {
	prefilled := guest(withGuestToleration, func(p *corev1.Pod) {
		p.Spec.NodeSelector = map[string]string{"timeslice.io/virtual-node": "true"}
		p.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{{
				Weight: 100, Preference: corev1.NodeSelectorTerm{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: "timeslice.io/guest", Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
				}},
			}},
		}}
	})
	for _, flags := range [][]string{todayFlags, nsFlags} {
		res := create(t, mustConfig(t, flags), prefilled)
		pod := mustAllow(t, res)
		if len(res.resp.Patch) != 0 || countTolerations(pod, webhook.GuestTaintKey) != 1 || len(preferredTerms(pod)) != 1 {
			t.Errorf("prefilled guest patched: %s", res.resp.Patch)
		}
	}
}

// TestAdmit_Reinvocation feeds every admitted pod back through Admit: the second pass is a no-op
// (reinvocationPolicy IfNeeded).
func TestAdmit_Reinvocation(t *testing.T) {
	for _, flags := range [][]string{todayFlags, nsFlags, nil} {
		cfg := mustConfig(t, flags)
		for _, pod := range []*corev1.Pod{kuberayDonor(), explicitDonor(), guest(), guest(withGuestToleration)} {
			first := mustAllow(t, create(t, cfg, pod))
			second := create(t, cfg, first)
			mustAllow(t, second)
			if len(second.resp.Patch) != 0 {
				t.Errorf("flags %v, pod %v: second pass patched %s", flags, pod.Labels, second.resp.Patch)
			}
		}
	}
}

func TestAdmit_EnvWiringKeepsUserEnv(t *testing.T) {
	pod := explicitDonor()
	pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{Name: "side", Image: "x"})
	pod.Spec.Containers[0].Env = []corev1.EnvVar{{Name: webhook.EnvOrchAddr, Value: "custom:1"}, {Name: "A", Value: "b"}}
	got := mustAllow(t, create(t, mustConfig(t, todayFlags), pod))
	if v, _ := env(&got.Spec.Containers[0], webhook.EnvOrchAddr); v != "custom:1" {
		t.Errorf("user TIMESLICE_ORCH_ADDR overwritten: %q", v)
	}
	for i := range got.Spec.Containers {
		for _, name := range []string{webhook.EnvJobID, webhook.EnvGroup, webhook.EnvOrchAddr} {
			if _, ok := env(&got.Spec.Containers[i], name); !ok {
				t.Errorf("container %d lacks %s", i, name)
			}
		}
	}
	if len(got.Spec.Containers[0].Env) != 4 {
		t.Errorf("env = %v", got.Spec.Containers[0].Env)
	}
}

func TestAdmit_DownwardWiringKeepsUserVolumes(t *testing.T) {
	pod := kuberayDonor()
	pod.Spec.Volumes = []corev1.Volume{{Name: "data", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}}
	pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "data", MountPath: "/data"}}
	got := mustAllow(t, create(t, mustConfig(t, nsFlags), pod))
	if len(got.Spec.Volumes) != 2 || got.Spec.Volumes[1].Name != webhook.PodinfoVolume {
		t.Errorf("volumes = %+v", got.Spec.Volumes)
	}
	if len(got.Spec.Containers[0].VolumeMounts) != 2 {
		t.Errorf("mounts = %+v", got.Spec.Containers[0].VolumeMounts)
	}
}

// TestAdmit_SameAsHandForms checks that the webhook output of the minimal demo manifests carries the
// same timeslice fields as the hand-written forms of each stack.
func TestAdmit_SameAsHandForms(t *testing.T) {
	today := mustAllow(t, create(t, mustConfig(t, todayFlags), explicitDonor()))
	hand := []corev1.EnvVar{
		{Name: webhook.EnvJobID, Value: "rl-trainer"},
		{Name: webhook.EnvGroup, Value: "trainers"},
		{Name: webhook.EnvOrchAddr, Value: orchAddr},
	}
	if len(today.Spec.Containers[0].Env) != len(hand) {
		t.Fatalf("env = %v, want %v", today.Spec.Containers[0].Env, hand)
	}
	for _, ev := range hand {
		if v, ok := env(&today.Spec.Containers[0], ev.Name); !ok || v != ev.Value {
			t.Errorf("env %s = %q, want %q", ev.Name, v, ev.Value)
		}
	}
	todayGuest := mustAllow(t, create(t, mustConfig(t, todayFlags), guest()))
	if len(todayGuest.Spec.Tolerations) != 1 || len(todayGuest.Spec.NodeSelector) != 1 {
		t.Errorf("today guest = %+v", todayGuest.Spec)
	}
}

func TestAdmit_UpdateIsNotMutated(t *testing.T) {
	for _, pod := range []*corev1.Pod{kuberayDonor(), guest()} {
		res := admit(t, mustConfig(t, nsFlags), pod, testUser, admissionv1.Update)
		mustAllow(t, res)
		if len(res.resp.Patch) != 0 {
			t.Errorf("UPDATE patched: %s", res.resp.Patch)
		}
	}
}

func TestAdmit_LongGroupIsHashed(t *testing.T) {
	long := func(cluster, group string) *corev1.Pod {
		return newPod(map[string]string{webhook.LabelDonor: "true", webhook.LabelRayCluster: cluster, webhook.LabelRayGroup: group})
	}
	cluster := "rc-" + strings.Repeat("a", 50) + "-7f2x9"
	cfg := mustConfig(t, nsFlags)
	first := mustAllow(t, create(t, cfg, long(cluster, "trainers-"+strings.Repeat("b", 20))))
	second := mustAllow(t, create(t, cfg, long(cluster, "trainers-"+strings.Repeat("b", 20))))
	group := first.Labels[webhook.LabelGroup]
	if errs := validation.IsValidLabelValue(group); len(errs) != 0 || len(group) > 63 {
		t.Errorf("group %q (%d chars) is not a valid label value: %v", group, len(group), errs)
	}
	if second.Labels[webhook.LabelGroup] != group {
		t.Errorf("group not stable: %q vs %q", group, second.Labels[webhook.LabelGroup])
	}
	if first.Labels[webhook.LabelJobID] != cluster {
		t.Errorf("job-id = %q", first.Labels[webhook.LabelJobID])
	}
}

func TestAdmit_NonPodAndDecodeErrors(t *testing.T) {
	cfg := mustConfig(t, nsFlags)
	res := webhook.Admit(t.Context(), &admissionv1.AdmissionRequest{
		UID: "u", Operation: admissionv1.Create, Resource: podsGVR("configmaps"),
	}, *cfg)
	if !res.Allowed || len(res.Patch) != 0 {
		t.Errorf("non-pod: %+v", res)
	}
	res = webhook.Admit(t.Context(), &admissionv1.AdmissionRequest{
		UID: "u", Operation: admissionv1.Create, Resource: podsGVR("pods"),
	}, *cfg)
	if res.Allowed || !strings.HasPrefix(res.Result.Message, "[decode]") {
		t.Errorf("empty object: %+v", res)
	}
	if nilRes := webhook.Admit(t.Context(), nil, *cfg); nilRes == nil || nilRes.Allowed {
		t.Errorf("nil request: %+v", nilRes)
	}
}

func TestAdmit_ResourceQuantitiesSurvive(t *testing.T) {
	pod := guest(func(p *corev1.Pod) {
		p.Spec.Containers[0].Resources.Requests = corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("10m")}
	})
	res := create(t, mustConfig(t, todayFlags), pod)
	mustAllow(t, res)
	if !bytes.Contains(res.raw, []byte(`"cpu":"10m"`)) {
		t.Errorf("admitted pod lost its requests: %s", res.raw)
	}
}
