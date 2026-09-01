package tpu

// SetProcRootForTest points TPU process discovery at a fixture tree and
// returns a restore function.
func SetProcRootForTest(root string) func() {
	old := procRoot
	procRoot = root
	return func() { procRoot = old }
}

// SetVfioRootForTest points the vfio gate at a fixture tree and returns a
// restore function.
func SetVfioRootForTest(root string) func() {
	old := vfioRoot
	vfioRoot = root
	return func() { vfioRoot = old }
}

// SetOpenVfioGroupForTest replaces the vfio group openability probe and
// returns a restore function.
func SetOpenVfioGroupForTest(f func(path string) error) func() {
	old := openVfioGroup
	openVfioGroup = f
	return func() { openVfioGroup = old }
}
