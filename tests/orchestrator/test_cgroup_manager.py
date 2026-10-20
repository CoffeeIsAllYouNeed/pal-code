from unittest.mock import MagicMock, patch

import pytest

import pal.orchestrator.components.cgroup_manager as mgr_module
from pal.orchestrator.components.cgroup_manager import (
    CgroupError,
    CgroupManager,
    NullResourceManager,
    ResourceLimits,
    create_resource_manager,
)


def _ok():
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_module, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(mgr_module, "_is_root", lambda: False)
    (tmp_path / "pal2.slice").mkdir()
    return tmp_path


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
def test_create_slice_writes_limits_via_tee_without_a_shell(mock_run, root):
    mock_run.return_value = _ok()
    manager = CgroupManager(parent_slice="pal2.slice")
    path = manager.create_slice("agent-1", ResourceLimits(memory_max="256M", pids_max="50"))
    assert path == root / "pal2.slice" / "agent-1"

    writes = [(c.args[0], c.kwargs.get("input")) for c in mock_run.call_args_list if c.args[0][:2] == ["sudo", "tee"]]
    assert any("memory.max" in cmd[2] and value == "256M\n" for cmd, value in writes)
    assert any("pids.max" in cmd[2] and value == "50\n" for cmd, value in writes)
    # nothing is ever handed to a shell
    assert not any("bash" in c.args[0] or "-c" in c.args[0] for c in mock_run.call_args_list)


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
@pytest.mark.parametrize("bad", ["../x", "a b", "x; rm -rf /", "$(id)", "", "a/b", ".hidden"])
def test_slice_names_are_validated(mock_run, root, bad):
    mock_run.return_value = _ok()
    manager = CgroupManager(parent_slice="pal2.slice")
    with pytest.raises(CgroupError, match="invalid slice name"):
        manager.create_slice(bad, ResourceLimits())


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
def test_limit_values_are_validated(mock_run, root):
    mock_run.return_value = _ok()
    manager = CgroupManager(parent_slice="pal2.slice")
    with pytest.raises(CgroupError, match="invalid limit value"):
        manager.create_slice("ok", ResourceLimits(memory_max="1G; reboot"))


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
def test_delete_slice_refuses_if_pids_present(mock_run, root):
    slice_dir = root / "pal2.slice" / "agent-1"
    slice_dir.mkdir()
    (slice_dir / "cgroup.procs").write_text("1234\n", encoding="utf-8")
    mock_run.return_value = _ok()
    with pytest.raises(CgroupError, match="1234"):
        CgroupManager(parent_slice="pal2.slice").delete_slice("agent-1")


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
def test_list_slices_returns_directory_names(mock_run, root):
    (root / "pal2.slice" / "agent-a").mkdir()
    (root / "pal2.slice" / "agent-b").mkdir()
    mock_run.return_value = _ok()
    assert set(CgroupManager(parent_slice="pal2.slice").list_slices()) == {"agent-a", "agent-b"}


@patch("pal.orchestrator.components.cgroup_manager.subprocess.run")
def test_root_does_not_use_sudo(mock_run, root, monkeypatch):
    monkeypatch.setattr(mgr_module, "_is_root", lambda: True)
    mock_run.return_value = _ok()
    CgroupManager(parent_slice="pal2.slice").create_slice("a", ResourceLimits(pids_max="5"))
    assert all(c.args[0][0] != "sudo" for c in mock_run.call_args_list)


# -- portability: no cgroups (macOS, Windows, Linux without cgroup v2) ---------------------------
def test_supported_is_false_off_linux(monkeypatch):
    monkeypatch.setattr(mgr_module.sys, "platform", "win32")
    assert CgroupManager.supported() is False
    monkeypatch.setattr(mgr_module.sys, "platform", "darwin")
    assert CgroupManager.supported() is False


def test_factory_falls_back_to_null_manager_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(CgroupManager, "supported", staticmethod(lambda: False))
    with caplog.at_level("WARNING"):
        m = create_resource_manager("x.slice")
    assert isinstance(m, NullResourceManager) and m.enforced is False
    assert "NOT be enforced" in caplog.text


def test_null_manager_has_the_same_interface_and_tracks_slices():
    m = NullResourceManager()
    m.create_slice("a", ResourceLimits(memory_max="1G"))
    with pytest.raises(CgroupError):
        m.create_slice("a", ResourceLimits())
    m.assign_pid("a", 123)
    assert m.list_slices() == ["a"] and m.stats("a") == {"enforced": False} and m.current_pids("a") == []
    m.delete_slice("a")
    assert m.list_slices() == []
    with pytest.raises(CgroupError):
        m.assign_pid("a", 1)
