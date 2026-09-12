# Integration test matrix — `tests/integration/`

GPU-free integration tier: each harness drives a **real production
binary** against a fake for the side it talks to, on any Linux machine
with the CUDA toolkit installed (no GPU or driver at run time — see
"What you need" in [`../README.md`](../README.md)). Every harness is
registered with ctest, so the whole tier runs as:

```sh
ctest -R integration --output-on-failure
```

Each harness gets its own subheader below; when a new integration test
lands, add its case table here under a new subheader.

## cr_client

`cr_client_integration_test.sh` drives the real `cr_client` binary
against `fake_workload`, which reuses the production control-channel
code (ShareMemComm, FINISH bookkeeping, `ValidateDumpFd`) and fakes only
the GPU work (pattern bytes for VRAM). The `cuda-checkpoint` binary is a
stub that records having been invoked, so freeze/thaw behavior is
observable without a driver. Exit-code contract asserted throughout:
0 OK / 1 usage / 2 op failed / 3 refused pre-signal / 4 timeout.

In this tree the suite runs the shared cases under the **ctl-mode env**
(`EXPORT_FILE_PATH` + `GPU_CR_CTL_PATH`, the "ctl:" prefix in test
names) and adds ctl-plane sections: advertisement gating, broken ctl
paths, zero-config `<data>/ctl` discovery, and legacy mode (env-only
control channel).

`fake_workload` knobs used by the cases below:

| Env | Fake behavior |
|---|---|
| `FAKE_OP_STATUS=<errno>` | Handler reports the op failed with that status |
| `FAKE_NO_FINISH=1` | Handler never signals FINISH (wedged workload) |
| `FAKE_SKIP_COMMIT=1` | Dest dump written without the trailing commit marker (torn write) |
| `FAKE_NOT_READY=1` | `.so` never publishes `selective_ready` |

Test cases (in script order):

| Group | Test case | Setup | `cr_client` invocation | Expected |
|---|---|---|---|---|
| Happy path | init | — | `-i` | 0 |
| Happy path | dest-path selective ckpt | — | `-c -s <regions> -o <path>` | 0; dump file created |
| Happy path | precreated dest handed to the target | — | `-c -s -o` | dump owned by the target's uid, mode 0600 (the .so writes it; root-owned would EACCES a non-root workload) |
| Happy path | dest-path selective restore | — | `-r -s <regions> -o <path>` | 0 |
| Happy path | buffer selective ckpt | — | `-c -s <regions>` | 0 |
| Happy path | full ckpt | — | `-c` | 0 |
| Happy path | full restore | — | `-r` | 0; toggle stub invoked |
| Buffer-only | full ckpt `-b` never toggles | — | `-c -b` | 0; toggle NOT invoked |
| Buffer-only | full restore `-b` never toggles | — | `-r -b` | 0; toggle NOT invoked |
| Torn dump | ckpt fails post-op validation | `FAKE_SKIP_COMMIT=1` | `-c -s -o` | 2 |
| Torn dump | restore refuses a marker-less file | 100-byte junk file at `-o` | `-r -s -o` | 2 |
| op_status | selective ckpt surfaces handler failure | `FAKE_OP_STATUS=28` | `-c -s -o` | 2 |
| op_status | full ckpt fails cleanly, never freezes | `FAKE_OP_STATUS=28` | `-c` | 2; toggle NOT invoked |
| op_status | full restore fails after thaw | `FAKE_OP_STATUS=28` | `-r` | 2; toggle invoked (thaw precedes outcome — upstream ordering) |
| Recycled dest | precreate truncates a reused path | 2-region dump, then 1-region dump to the same path | `-c -s -o` twice | both 0; file shrinks (`O_TRUNC` pins the torn-write false-validate fix) |
| Timeout | wedged workload fails the op | `FAKE_NO_FINISH=1`, 2s deadline | `-c -s` | 4 |
| Timeout | dest-path timeout leaves an uncommitted artifact | `FAKE_NO_FINISH=1 FAKE_SKIP_COMMIT=1`, 2s deadline | `-c -s -o` | 4; precreated dest survives, no marker |
| Timeout | restore refuses the timeout artifact | fresh fake | `-r -s -o <artifact>` | 2 |
| Not ready | dest-path ckpt refused pre-signal | `FAKE_NOT_READY=1` | `-c -s -o` | 3 |
| Not ready | buffer ckpt refused pre-signal | `FAKE_NOT_READY=1` | `-c -s` | 3 |
| Not ready | dest-path restore refused pre-signal (the gate's worst case: stale-buffer replay) | `FAKE_NOT_READY=1` | `-r -s -o` | 3 |
| Not ready | buffer restore refused pre-signal | `FAKE_NOT_READY=1` | `-r -s` | 3 |
| Not ready | full ckpt keeps historical tolerance | `FAKE_NOT_READY=1` | `-c` | 0 |
| Usage | `-o` without `-s` | no fake needed | `-c -o -p 1` | 1 |
| Usage | `-s` without `-c`/`-r` | no fake needed | `-i -s -p 1` | 1 |
| Usage | `-o` path over the 256-byte max | no fake needed | 300-char `-o` | 1 |
| Usage | malformed `-s` fails at parse | armed fake (parse sits after the ready-gate) | `-c -s bogus` | 1 |
| Usage | restore from a missing file | — | `-r -s -o <nonexistent>` | 2 |
| Dest hardening | relative `-o` refused | — | `-o rel/dump.bin` | 2 |
| Dest hardening | symlink dest refused | `ln -s /etc/hostname <path>` | `-c -s -o <path>` | 2 (ELOOP via `O_NOFOLLOW`) |

Ctl-plane, discovery, and legacy-mode cases (this tree only):

| Group | Test case | Setup | `cr_client` invocation | Expected |
|---|---|---|---|---|
| Advertisement gate | no advertisement → refused | `FAKE_NOT_READY=1` (nothing advertised) | `-c -s` | 3 |
| Advertisement gate | starttime mismatch (PID reuse) → refused | advert edited to `starttime=1` | `-c -s` | 3 |
| Advertisement gate | owner uid mismatch (forged advert) → refused | advert `chown`ed to 65534 (root-only; skipped elsewhere) | `-c -s` | 3 |
| Broken ctl path | non-tmpfs `GPU_CR_CTL_PATH` refused | `GPU_CR_CTL_PATH` on disk-backed fs | `-c -s` | 3 |
| Broken ctl path | missing `GPU_CR_CTL_PATH` dir refused | `GPU_CR_CTL_PATH=/nonexistent-ctl` | `-c -s` | 3 |
| Discovery | advert + control channel land on `<data>/ctl` | no ctl env at all (tmpfs data dir) | `-i` | 0; both files under `<data>/ctl` |
| Discovery | dest-path selective ckpt/restore | no ctl env | `-c -s -o`, `-r -s -o` | 0 each |
| Discovery | starttime mismatch → refused | advert edited to `starttime=1` | `-c -s` | 3 |
| Legacy mode | init / buffer ckpt / dest-path ckpt | `EXPORT_FILE_PATH` only | `-i`, `-c -s`, `-c -s -o` | 0 each |

The five not-ready refusal rows in the main table run in legacy mode in
this tree (ctl mode refuses earlier, at the advertisement gate).

File-existence assertions (`dump created`, toggle-marker checks, the
timeout artifact) count as separate pass/fail lines in the script's
summary, so the printed total exceeds the row count above.
