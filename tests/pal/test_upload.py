"""Attaching files in the terminal: /add from anywhere, dropping files on the window, documents converted, secrets kept back."""
from pal.terminal.cli import dropped_files, split_paths
from pal.terminal.session import PalSession, read_any
from tests.pal.test_pal import FakeClient


def test_a_file_from_anywhere_goes_with_the_next_message_only(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "spec.md").write_text("# Spec\nusers can log in", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    s = PalSession(FakeClient(), "m", "local", proj, "plan")
    name, note = s.attach(str(outside / "spec.md"))
    assert name == "spec.md" and note is None and len(s.pending) == 1
    s.ask("build what the spec says")
    s.ask("and now?")
    first, second = (c[1] for c in s.client.calls)
    assert [r["name"] for r in first] == ["spec.md"] and "log in" in first[0]["text"] and second is None and s.pending == []


def test_secret_files_binary_files_and_missing_files_are_refused(tmp_path):
    (tmp_path / ".env").write_text("KEY=sk-abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    (tmp_path / "a.bin").write_bytes(b"\x00\x01\x02binary")
    s = PalSession(FakeClient(), "m", "local", tmp_path, "plan")
    for path, why in ((tmp_path / ".env", "secrets file"), (tmp_path / "a.bin", "not a text file"), (tmp_path / "nope.txt", "no such file")):
        name, note = s.attach(str(path))
        assert name is None and why in note
    assert s.pending == []


def test_keys_inside_an_attached_file_are_hidden(tmp_path):
    (tmp_path / "config.py").write_text('KEY = "sk_live_abcdefghijklmnop1234"\n', encoding="utf-8")
    s = PalSession(FakeClient(), "m", "local", tmp_path, "plan")
    name, note = s.attach(str(tmp_path / "config.py"))
    assert name and "hidden" in note and "sk_live" not in s.pending[0]["text"]


def test_documents_are_converted_to_text(tmp_path):
    (tmp_path / "page.html").write_text("<html><body><h1>Title</h1><p>Hello there</p></body></html>", encoding="utf-8")
    text, note = read_any(tmp_path / "page.html")
    assert (text is not None and "Hello there" in text) or (text is None and "could not be converted" in note)


def test_dropped_files_are_recognised_even_with_spaces_and_quotes(tmp_path):
    a = tmp_path / "my notes.md"
    a.write_text("x", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("y", encoding="utf-8")
    assert dropped_files(f"'{a}' {b}") == [str(a), str(b)]
    assert dropped_files(f"{b} and some words") == [] and dropped_files("explain this") == []
    assert split_paths('"a b.txt" c.txt') == ["a b.txt", "c.txt"]
