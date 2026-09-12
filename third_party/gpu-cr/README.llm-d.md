# GPU-CR (fork)

This directory contains a fork of [gpu-os/GPU-CR](https://github.com/gpu-os/GPU-CR),
imported pristine at commit `e9bbb52e1f52986587fc631217c0f2b50b46245a`
(upstream HEAD as of August 2026). GPU-CR provides transparent checkpoint/restore
of GPU state via an LD_PRELOAD-injected vGPU library; this project uses it as the
checkpoint engine for GPU time-slicing.

## Provenance and maintenance status

Upstream is a research project with limited maintenance activity; changes we
proposed upstream (https://github.com/gpu-os/GPU-CR/pull/5) have not been reviewed. We therefore carry
this copy, with local modifications landing via reviewed PRs in this repository
(see the git history of this path). This fork is maintained on a best-effort
basis for the needs of this project; we may re-sync with upstream, contribute
our changes back, or retire this copy in the future.

This follows established Kubernetes-ecosystem practice for carrying forks of
external projects: see `third_party/forked/` in kubernetes/kubernetes,
[kubernetes/klog](https://github.com/kubernetes/klog) (fork of golang/glog), and
[kube-openapi's in-tree fork of go-openapi](https://github.com/kubernetes/kube-openapi/tree/master/pkg/validation).

## Local changes

This copy carries upstream `e9bbb52` plus the changes GPU time-slicing
required, cleaned up and tested:

- memory-address (selective) checkpoint backend, unrounded allocation-size
  dumps, and granule-chunked copies
- destination-path selective checkpoints,
  control channel off hugetlbfs (readiness advertisements),
  runtime-sizable checkpoint buffers (`GPU_CR_SHM_GB`/`_MB`,
  `GPU_CR_STAGING_MB`) — destination-path-only deployments should set
  `GPU_CR_SHM_MB=0`: dest-path ops never touch the dump buffer, so
  deferred mode drops the reservation entirely (pool sizing rule in
  `src/gpu_cr_config.h`)
- Google C++ Style Guide cleanup of the added code, and cr_client
  hardening (`GPU_CR_CUDA_CHECKPOINT` override; restore fails on a failed
  cuda-checkpoint toggle)
- unit, integration, e2e and perf-regression suites under `tests/`
  (see `tests/README.md`); `Dockerfile.build` runs the GPU-free suites on
  every image build

Like the original import, the prebuilt `cuda-checkpoint` binary is not
vendored; deployments provide it (or set `GPU_CR_CUDA_CHECKPOINT`).

## Licensing

Upstream's Apache-2.0 [LICENSE](./LICENSE) applies to this directory and is
retained, as are upstream copyright headers in imported files. Files created
locally carry this project's headers. Modifications to imported files are
recorded in this repository's git history (Apache-2.0 §4(b)).

## Standards

Code in this directory is held to the same CI, review, and test standards as
the rest of this repository, except purely cosmetic linters (typos/formatting),
which are not applied to imported upstream files.
