package utils

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// GPU-CR leaves per-process artifacts in the shared checkpoint dir:
//
//	ckpt-<id>.data, ckpt-<id>-host.data
//	                  dump + staging buffers, file backend — the naming
//	                  GPU-CR uses whenever EXPORT_FILE_PATH is set, i.e. in
//	                  every agent-managed deployment (MAP_SHARED
//	                  reservations stick to the FILE, so a leaked pair pins
//	                  its whole extent even after the process dies)
//	<id>, <id>-host   the same pair in hugepage mode (EXPORT_FILE_PATH
//	                  unset, hardcoded /mnt/huge-ckpt)
//	control-<pid>     shared-memory control channel
//	pid_map_<pid>     pid→id map (may be empty on hugetlbfs)
//
// Nothing deletes them on process exit, so the agent sweeps them.

var (
	fileDumpRe = regexp.MustCompile(`^ckpt-(\d+)(-host)?\.data$`)
	hugeDumpRe = regexp.MustCompile(`^(\d+)(-host)?$`)
	pidFileRe  = regexp.MustCompile(`^(?:control|pid_map)[-_](\d+)$`)
)

// isDumpFile reports whether name is a GPU-CR dump/staging buffer file, in
// either naming scheme.
func isDumpFile(name string) bool {
	return fileDumpRe.MatchString(name) || hugeDumpRe.MatchString(name)
}

// gcMinAge guards against racing a process that created its files but hasn't
// mmap'd them yet (files appear before the mapping does).
const gcMinAge = 5 * time.Minute

// StartGPUCRSweeper sweeps stale GPU-CR artifacts at startup and every
// interval.
func StartGPUCRSweeper(ctx context.Context, ctlDir string, interval time.Duration) {
	go func() {
		sweep(ctlDir)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				sweep(ctlDir)
			}
		}
	}()
}

func sweep(ctlDir string) {
	entries, err := os.ReadDir(ctlDir)
	if err != nil {
		slog.Warn("GC: cannot read checkpoint dir", "dir", ctlDir, "err", err)
		return
	}

	liveInodes, complete := liveMappedInodes()
	if !complete {
		slog.Warn("GC: procfs scan incomplete; keeping all dump files this sweep", "dir", ctlDir)
	}
	now := time.Now()
	var removed []string

	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil || now.Sub(info.ModTime()) < gcMinAge {
			continue
		}

		if isDumpFile(name) {
			key, ok := fileDevIno(filepath.Join(ctlDir, name))
			if complete && ok && !liveInodes[key] {
				if os.Remove(filepath.Join(ctlDir, name)) == nil {
					removed = append(removed, name)
				}
			}
			continue
		}
		if m := pidFileRe.FindStringSubmatch(name); m != nil {
			if _, err := os.Stat("/proc/" + m[1]); os.IsNotExist(err) {
				if os.Remove(filepath.Join(ctlDir, name)) == nil {
					removed = append(removed, name)
				}
			}
		}
		// Never touch the bare "control" id-counter file or anything else.
	}
	if len(removed) > 0 {
		slog.Info("GC: removed stale GPU-CR artifacts", "count", len(removed), "files", removed)
	}
}

// procfsRoot is the procfs mount scanned for live mappings; a var so tests
// can point the scan at a fixture tree.
var procfsRoot = "/proc"

// pidDirRe matches the numeric process dirs under /proc.
var pidDirRe = regexp.MustCompile(`^\d+$`)

// liveMappedInodes returns the device:inode key of every file currently
// mmap'd by any live process (agent runs with hostPID, so /proc covers the
// whole node), plus whether the scan was complete. Liveness matches by
// inode, never by pathname: a maps line renders the path in the OWNING
// process's mount namespace, so the same checkpoint file can appear under a
// path the agent has never mounted — dev:inode is namespace- and
// name-independent. PID dirs are enumerated with ReadDir rather than a
// glob: Glob silently drops paths it cannot traverse, which would
// under-report live mappings with no error to classify. The scan is
// incomplete when the enumeration or any maps read fails for a reason other
// than the process exiting mid-scan: a partial scan cannot prove a dump is
// unmapped, so the caller must skip dump deletion for that sweep.
func liveMappedInodes() (map[string]bool, bool) {
	live := make(map[string]bool)
	complete := true
	entries, err := os.ReadDir(procfsRoot)
	if err != nil {
		return live, false
	}
	for _, entry := range entries {
		if !entry.IsDir() || !pidDirRe.MatchString(entry.Name()) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(procfsRoot, entry.Name(), "maps"))
		if err != nil {
			// Gone between enumerate and read is normal process churn;
			// anything else (EACCES, hidepid, ...) hides mappings from
			// the scan.
			if !os.IsNotExist(err) && !errors.Is(err, syscall.ESRCH) {
				complete = false
			}
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			// address perms offset dev inode [pathname]
			fields := strings.Fields(line)
			if len(fields) < 5 || fields[4] == "0" {
				continue // anonymous or malformed: nothing to key on
			}
			if key, ok := mapsDevIno(fields[3], fields[4]); ok {
				live[key] = true
			}
		}
	}
	return live, complete
}

// mapsDevIno turns a maps-line device field (hex "major:minor") and decimal
// inode field into the key fileDevIno produces for the same file.
func mapsDevIno(dev, ino string) (string, bool) {
	majField, minField, found := strings.Cut(dev, ":")
	if !found {
		return "", false
	}
	major, err := strconv.ParseUint(majField, 16, 64)
	if err != nil {
		return "", false
	}
	minor, err := strconv.ParseUint(minField, 16, 64)
	if err != nil {
		return "", false
	}
	inode, err := strconv.ParseUint(ino, 10, 64)
	if err != nil {
		return "", false
	}
	return fmt.Sprintf("%d:%d:%d", major, minor, inode), true
}

// fileDevIno stats path into the same key space as mapsDevIno.
func fileDevIno(path string) (string, bool) {
	var st syscall.Stat_t
	if err := syscall.Stat(path, &st); err != nil {
		return "", false
	}
	nums := devMajMin(st.Dev)
	return fmt.Sprintf("%d:%d:%d", nums.major, nums.minor, st.Ino), true
}

// devNums is a stat dev_t split into its device-number halves.
type devNums struct {
	major, minor uint64
}

// devMajMin splits a stat dev_t with the Linux huge-dev encoding — the same
// split the kernel uses to print the maps device column.
func devMajMin(dev uint64) devNums {
	return devNums{
		major: ((dev >> 8) & 0xfff) | ((dev >> 32) &^ 0xfff),
		minor: (dev & 0xff) | ((dev >> 12) &^ 0xff),
	}
}
