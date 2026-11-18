# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cgroup memory readers and SHM allocation preflight checks."""

import sys
from io import StringIO
from unittest import mock

import pytest

from vllm.utils import cpu_resource_utils as cru
from vllm.utils.mem_constants import GiB_bytes

_V2_LIMIT_PATH = "/sys/fs/cgroup/memory.max"
_V2_USAGE_PATH = "/sys/fs/cgroup/memory.current"
_V1_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_V1_USAGE_PATH = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_V2_STAT_PATH = "/sys/fs/cgroup/memory.stat"
_V1_STAT_PATH = "/sys/fs/cgroup/memory/memory.stat"


@pytest.fixture(autouse=True)
def _clear_cgroup_cache():
    cru.get_cgroup_memory_limit.cache_clear()
    yield
    cru.get_cgroup_memory_limit.cache_clear()


def _stub_files(monkeypatch, files: dict):
    """Stub ``open()`` for a fixed set of paths; ``None`` -> OSError."""
    files.setdefault(_V2_STAT_PATH, None)
    files.setdefault(_V1_STAT_PATH, None)
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path in files:
            content = files[path]
            if content is None:
                raise OSError(f"no such file: {path}")
            return StringIO(content)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cgroup_v2_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: f"{20 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{5 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 20 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 5 * GiB_bytes


def test_cgroup_v2_unlimited_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: "max\n",
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 1 * GiB_bytes


def test_cgroup_v1_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: None,
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{2 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 2 * GiB_bytes


def test_cgroup_v1_unlimited_sentinel_is_ignored(monkeypatch):
    """An unlimited cgroup v1 sentinel must not be treated as a real limit."""
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: f"{(1 << 63) - 1}\n"},
    )

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_no_limit_files_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: None})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_skipped_on_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: f"{1 * GiB_bytes}\n"})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_limit_is_cached_until_explicitly_cleared(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes
    files[_V2_LIMIT_PATH] = f"{2 * GiB_bytes}\n"
    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes

    cru.get_cgroup_memory_limit.cache_clear()
    assert cru.get_cgroup_memory_limit() == 2 * GiB_bytes


def test_cgroup_usage_is_read_without_cache(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_usage() == 512 << 20
    files[_V2_USAGE_PATH] = f"{768 << 20}\n"
    assert cru.get_cgroup_memory_usage() == 768 << 20


def test_check_cgroup_memory_available_warns_on_low_headroom(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(600 << 20, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_called_once()
    assert "current headroom is below" in log_warning.call_args.args[-1]


def test_check_cgroup_memory_available_logs_success_at_debug(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(256 << 20, "mmap")

    log_debug.assert_called_once()
    log_warning.assert_not_called()


@pytest.mark.parametrize(
    ("limit", "usage"),
    [(None, 512 << 20), (1 << 30, None)],
)
def test_check_cgroup_memory_available_skips_without_snapshot(
    monkeypatch, limit, usage
):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: limit)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: usage)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(1 << 60, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_not_called()


class TestCgroupMemoryUsageWithoutReclaimable:
    """Usage excludes the file cache and slab memory the kernel can reclaim."""

    V2_STAT = (
        f"anon {40 * GiB_bytes}\n"
        f"file {18 * GiB_bytes}\n"
        f"shmem {2 * GiB_bytes}\n"
        f"active_file {4 * GiB_bytes}\n"
        f"inactive_file {10 * GiB_bytes}\n"
        f"slab_reclaimable {1 * GiB_bytes}\n"
        f"slab_unreclaimable {1 * GiB_bytes}\n"
    )
    V1_STAT = (
        f"cache {4 * GiB_bytes}\n"
        f"active_file {1 * GiB_bytes}\n"
        f"total_rss {40 * GiB_bytes}\n"
        f"total_active_file {4 * GiB_bytes}\n"
        f"total_inactive_file {10 * GiB_bytes}\n"
    )

    @staticmethod
    def write_v2(root, usage, stat=None):
        (root / "memory.max").write_text(f"{72 * GiB_bytes}\n")
        (root / "memory.current").write_text(f"{usage}\n")
        if stat is not None:
            (root / "memory.stat").write_text(stat)

    @staticmethod
    def write_v1(root, usage, stat=None):
        v1 = root / "memory"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text(f"{72 * GiB_bytes}\n")
        (v1 / "memory.usage_in_bytes").write_text(f"{usage}\n")
        if stat is not None:
            (v1 / "memory.stat").write_text(stat)

    def test_v2_subtracts_file_lru_and_reclaimable_slab(self, tmp_path):
        self.write_v2(tmp_path, 60 * GiB_bytes, self.V2_STAT)

        usage = cru.get_cgroup_memory_usage(cgroup_root=str(tmp_path))

        assert usage == 45 * GiB_bytes

    def test_v1_subtracts_hierarchical_file_lru(self, tmp_path):
        self.write_v1(tmp_path, 60 * GiB_bytes, self.V1_STAT)

        usage = cru.get_cgroup_memory_usage(cgroup_root=str(tmp_path))

        assert usage == 46 * GiB_bytes

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_missing_stat_keeps_raw_usage(self, tmp_path, version):
        getattr(self, f"write_{version}")(tmp_path, 60 * GiB_bytes)

        usage = cru.get_cgroup_memory_usage(cgroup_root=str(tmp_path))

        assert usage == 60 * GiB_bytes

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_reclaimable_above_usage_clamps_to_zero(self, tmp_path, version):
        stat = self.V1_STAT if version == "v1" else self.V2_STAT
        getattr(self, f"write_{version}")(tmp_path, 1 * GiB_bytes, stat)

        usage = cru.get_cgroup_memory_usage(cgroup_root=str(tmp_path))

        assert usage == 0

    def test_no_limit_files_returns_none(self, tmp_path):
        assert cru.get_cgroup_memory_usage(cgroup_root=str(tmp_path)) is None
