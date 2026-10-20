import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("pal_run", Path(__file__).resolve().parents[1] / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_venv_python_path_per_platform(tmp_path):
    assert run.venv_python(tmp_path, "win32") == tmp_path / "Scripts" / "python.exe"
    assert run.venv_python(tmp_path, "linux") == tmp_path / "bin" / "python"
    assert run.venv_python(tmp_path, "darwin") == tmp_path / "bin" / "python"


def test_reinstall_only_when_pyproject_changes(tmp_path):
    (tmp_path / "pyproject.toml").write_text("a", encoding="utf-8")
    stamp = tmp_path / "stamp"
    assert run.needs_install(stamp, tmp_path)
    stamp.write_text(run.stamp_value(tmp_path), encoding="utf-8")
    assert not run.needs_install(stamp, tmp_path)
    (tmp_path / "pyproject.toml").write_text("b", encoding="utf-8")
    assert run.needs_install(stamp, tmp_path)


def test_install_falls_back_to_fewer_extras_and_forwards_args(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "VENV", tmp_path / ".venv")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("x", encoding="utf-8")
    calls = []

    def fake(cmd):
        calls.append(cmd)
        return 1 if any(str(c).endswith("[full]") for c in cmd) else 0

    monkeypatch.setattr(run, "_run", fake)
    run.ensure_installed(Path("py"))
    installs = [c[-1] for c in calls if "-e" in c]
    assert installs == [".[full]", ".[documents]"] and not run.needs_install(tmp_path / ".venv" / ".pal-installed", tmp_path)


def test_main_uses_current_venv_and_forwards_arguments(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(run, "ensure_installed", lambda py: None)
    monkeypatch.setattr(run, "_run", lambda cmd: seen.append(cmd) or 0)
    monkeypatch.setattr(run.sys, "prefix", "/some/venv")
    monkeypatch.setattr(run.sys, "base_prefix", "/usr")
    assert run.main(["--port", "8080"]) == 0
    assert seen[-1][1:] == ["-m", "pal", "serve", "--open", "--port", "8080"]


def test_one_command_really_is_documented():
    readme = (Path(run.ROOT) / "README.md").read_text(encoding="utf-8")
    assert "python run.py" in readme


def test_the_reinstall_stamp_also_covers_the_hash_locked_litellm_pin(tmp_path):
    (tmp_path / "pyproject.toml").write_text("a", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    lock = tmp_path / "requirements" / "litellm.lock"
    lock.write_text("litellm==1 --hash=sha256:aa", encoding="utf-8")
    before = run.stamp_value(tmp_path)
    lock.write_text("litellm==2 --hash=sha256:bb", encoding="utf-8")
    assert run.stamp_value(tmp_path) != before


def test_litellm_is_installed_first_from_the_hash_lock_and_a_failure_is_announced_not_hidden(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "VENV", tmp_path / ".venv")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("x", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "litellm.lock").write_text("litellm==1 --hash=sha256:aa", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run, "_run", lambda cmd: calls.append(cmd) or (1 if "--require-hashes" in cmd else 0))
    run.ensure_installed(Path("py"))
    hashed = [i for i, c in enumerate(calls) if "--require-hashes" in c]
    editable = [i for i, c in enumerate(calls) if "-e" in c]
    assert hashed and editable and hashed[0] < editable[0]                 # the verified install comes before everything else
    assert "-r" in calls[hashed[0]] and calls[hashed[0]][-1].endswith("litellm.lock")
    assert "without hash checking" in capsys.readouterr().out


def test_cli_opens_the_terminal_session(monkeypatch):
    seen = []
    monkeypatch.setattr(run, "ensure_installed", lambda py: None)
    monkeypatch.setattr(run, "_run", lambda cmd: seen.append(cmd) or 0)
    monkeypatch.setattr(run.sys, "prefix", "/some/venv")
    monkeypatch.setattr(run.sys, "base_prefix", "/usr")
    assert run.main(["cli", "--mode", "plan"]) == 0
    assert seen[-1][1:] == ["-m", "pal.terminal.cli", "cli", "--mode", "plan"]
