"""Tests for the BackendConfig builder helpers."""

import pytest

from timeslice.snapshot_agent import cuda_config, direct_memory_config


@pytest.mark.parametrize(
    "builder,oneof",
    [
        (cuda_config, "cuda"),
        (direct_memory_config, "direct_memory"),
    ],
)
def test_builds_config_with_pids(builder, oneof):
    cfg = builder([101, 102])
    assert cfg.WhichOneof("backend") == oneof
    target = getattr(cfg, oneof).explicit_target
    assert list(target.pids) == [101, 102]


@pytest.mark.parametrize("builder", [cuda_config, direct_memory_config])
@pytest.mark.parametrize("pids", [[], [0], [-5], [1.5], ["123"], [True]])
def test_rejects_invalid_pids(builder, pids):
    with pytest.raises((ValueError, TypeError)):
        builder(pids)
