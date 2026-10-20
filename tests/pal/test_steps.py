"""The workflow lines: each step of a request is announced and ticked off, and changed files are listed without their code."""
import io

from rich.console import Console

from pal import steps
from pal.terminal import ui
from pal.terminal.session import Change


def test_steps_are_announced_with_what_they_did():
    seen = []
    with steps.listen(lambda event, key, text: seen.append((event, text))):
        with steps.step("input_saver"):
            steps.detail("12 tokens less")
        steps.note("security", "Security check  ·  nothing found")
    assert seen[0] == ("begin", "Trimming your message")
    assert seen[1][0] == "end" and "12 tokens less" in seen[1][1]
    assert seen[2] == ("end", "Security check  ·  nothing found")


def test_nothing_happens_without_a_listener():
    with steps.step("context") as s:
        steps.detail("x")
    assert s.detail == ""


def test_a_build_is_a_bar_and_a_row_of_files_without_code():
    buf = io.StringIO()
    c = Change("app.py", "", "secret_code = 1\n", True, [["add", "secret_code = 1"]], [])
    d = Change("styles.css", "a{}", "b{}", False, [["del", "a{}"], ["add", "b{}"]], [])
    ui.show_changes(Console(file=buf, width=100), [c, d])
    out = buf.getvalue()
    assert "Built" in out and "app.py \u2705 | \U0001F4C4 styles.css \u2705" in out and "secret_code" not in out


def test_questions_are_choices_and_default_to_the_safe_one():
    out = io.StringIO()
    console = Console(file=out)
    assert ui.select(console, "What do you want to do?", ["Debug and retry once", "Roll back the changes"]) == "Debug and retry once"
    assert ui.confirm(console, "Apply 2 files?") is False                 # without a terminal: No
    assert ui.select(console, "x", []) is None


def test_which_model_answers_is_never_shown():
    seen = []
    from pal.latency_tracker.pipeline import LatencyTracker  # noqa: F401
    from pal.latency_tracker import pipeline

    class T:
        stage_seconds: dict = {}
    with steps.listen(lambda event, key, text: seen.append(text)):
        with pipeline.LatencyTracker.stage(T(), "routing"):
            pass
        with pipeline.LatencyTracker.stage(T(), "input_saver"):
            pass
    assert seen and not any("answer" in t.lower() for t in seen) and any("Trimming" in t for t in seen)
