"""Builds land in the folder Pal was started in; pal bench shows its comparison as bars and can keep Pal's files."""
import io

from rich.console import Console

from pal.studio.runs import build_target
from pal.terminal import cli
from pal.terminal.bench import Arm, BenchResult, Step


def test_empty_folder_is_used_directly(tmp_path):
    (tmp_path / ".git").mkdir()
    assert build_target(tmp_path, "flappy-bird") == tmp_path


def test_busy_folder_gets_a_new_subfolder(tmp_path):
    (tmp_path / "notes.txt").write_text("mine")
    assert build_target(tmp_path, "flappy-bird") == tmp_path / "flappy-bird"
    (tmp_path / "flappy-bird").mkdir()
    (tmp_path / "flappy-bird" / "index.html").write_text("old")
    assert build_target(tmp_path, "flappy-bird") == tmp_path / "flappy-bird-2"


def _result():
    plain, pal = Arm("plain", [Step("t", "build", 900, 100, 4.0, ["game.py"])]), Arm("pal", [Step("t", "build", 300, 100, 2.0, ["game.py"])])
    return BenchResult("make a game", "m", ".", plain, pal, pal_files={"game.py": b"print('hi')\n"})


def test_bench_shows_input_and_output_apart(tmp_path):
    buf = io.StringIO()
    (tmp_path / "app.py").write_text("print(1)\n")
    cli.show_bench(Console(file=buf, width=140), _result(), "local", "m", tmp_path)
    out = buf.getvalue()
    assert "Input tokens" in out and "Output tokens" in out and "Time" in out and "66.7% less" in out
    assert "folder is empty" not in out


def test_bench_says_when_there_was_nothing_to_save(tmp_path):
    buf = io.StringIO()
    cli.show_bench(Console(file=buf, width=140), _result(), "local", "m", tmp_path)
    assert "folder is empty" in buf.getvalue()


def test_bench_keeps_pal_files_only_when_asked(tmp_path, monkeypatch):
    console = Console(file=io.StringIO())
    monkeypatch.setattr(cli.ui, "confirm", lambda c, q: False)
    assert cli.keep_bench_files(console, _result(), tmp_path) == [] and not (tmp_path / "game.py").exists()
    monkeypatch.setattr(cli.ui, "confirm", lambda c, q: True)
    assert cli.keep_bench_files(console, _result(), tmp_path) == ["game.py"] and (tmp_path / "game.py").read_text() == "print('hi')\n"
