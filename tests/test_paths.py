import pytest

from pal import paths


@pytest.mark.parametrize("key", ["a.json", "docs/a-b_c.json", "memory/1234.json", "deep/er/path/x.md"])
def test_safe_keys(key):
    assert paths.is_safe_relative_key(key)


@pytest.mark.parametrize("key", [
    "", " a", "../x", "a/../b", "/abs", "a//b", "a/./b", "a\\b", "C:x", "c:/x", "a:b", "con", "CON.txt", "nul",
    "lpt1.log", "a/aux/b", "trail.", "trail ", "a<b", "a|b", "a?b", "a*b", 'a"b', "a\x00b", "a\nb"])
def test_unsafe_keys_are_rejected_on_every_os(key):
    assert not paths.is_safe_relative_key(key)


def test_safe_id_strips_everything_odd():
    assert paths.safe_id("../../etc") == "etc"
    assert paths.safe_id("") == "default" and paths.safe_id("!!!") == "default"
    assert paths.safe_id("CON") == "_CON"
    assert paths.safe_id("a" * 200) == "a" * 64
    assert paths.safe_id("ok-id_1") == "ok-id_1"


def test_pal_home_follows_the_env_var_and_defaults_to_the_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "h"))
    assert paths.data_dir("storage") == tmp_path / "h" / "storage"
    monkeypatch.delenv("PAL_HOME")
    assert paths.pal_home().name == ".pal"


def test_atomic_write_replaces_and_leaves_no_temp_files(tmp_path):
    f = tmp_path / "sub" / "x.txt"
    paths.atomic_write_text(f, "one")
    paths.atomic_write_text(f, "twö")
    assert f.read_text(encoding="utf-8") == "twö" and [p.name for p in f.parent.iterdir()] == ["x.txt"]
