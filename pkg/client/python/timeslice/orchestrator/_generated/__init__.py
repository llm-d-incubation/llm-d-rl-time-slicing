import pathlib
import sys

# Temporarily add this directory to sys.path to resolve flat imports inside
# the generated files, then eagerly import them and clean up.
_current_dir = str(pathlib.Path(__file__).parent)
_added = False

if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
    _added = True

try:
    from . import accelerator_orchestrator_pb2 as pb2
    from . import accelerator_orchestrator_pb2_grpc as pb2_grpc
finally:
    if _added and _current_dir in sys.path:
        sys.path.remove(_current_dir)

__all__ = ["pb2", "pb2_grpc"]
