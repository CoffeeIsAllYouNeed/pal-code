"""pal cli asks for the models at every start: an API model or a local model that suits the GPU (installed ones marked), then the helper."""
from rich.console import Console

from pal.terminal import backend
from pal.terminal.hardware import from_spec
from pal.terminal.logo import BANNER_ROWS, banner
from pal.terminal.setup import API_OPTION, NO_HELPER, choose_models, helper_choices, main_choices

INFO = {"ok": True, "models": [{"name": "mistral:7b", "size_gb": 4.1}, {"name": "qwen2.5:1.5b", "size_gb": 1.0}]}
GPU8 = from_spec("gpu:8:16")


def _run(monkeypatch, answers, info=INFO, pulled=None, confirm=True):
    saved, asked = {}, []
    monkeypatch.setattr(backend, "save_prefs", lambda **k: saved.update(k))

    def pick(question, options, recommended=None, other=False, header=""):
        asked.append((header, options, recommended))
        return answers.pop(0)
    ok = choose_models(Console(file=open("/dev/null", "w")), GPU8, info=info, pick=pick, puller=lambda t: (pulled.append(t) if pulled is not None else None) or True,
                       confirm=lambda q: confirm, api_chooser=lambda c, s: ("openrouter", "some/model"))
    return ok, saved, asked


def test_nothing_is_preselected_and_installed_models_come_first(monkeypatch):
    ok, saved, asked = _run(monkeypatch, ["mistral:7b", "qwen2.5:1.5b"])
    assert ok and saved == {"provider": "local", "model": "mistral:7b", "helper_model": "qwen2.5:1.5b", "setup_done": "1"}
    (h1, main, rec1), (h2, helpers, rec2) = asked
    assert rec1 is None and rec2 is None
    assert main[0][0] == API_OPTION
    assert main[1] == ("mistral:7b", "installed · general") or main[1][1].startswith("installed")
    assert any(label == "qwen2.5:1.5b" and about.startswith("installed") for label, about in helpers)
    assert helpers[-1][0] == NO_HELPER


def test_a_model_that_is_not_installed_is_downloaded(monkeypatch):
    pulled = []
    ok, saved, _ = _run(monkeypatch, ["qwen2.5-coder:7b", NO_HELPER], pulled=pulled)
    assert ok and pulled == ["qwen2.5-coder:7b"] and saved["model"] == "qwen2.5-coder:7b" and saved["helper_model"] == ""


def test_api_model_and_declined_download(monkeypatch):
    ok, saved, _ = _run(monkeypatch, [API_OPTION, "llama3.2:3b"], confirm=False)
    assert ok and saved["helper_model"] == "" and "model" not in saved           # the API chooser saves the main model itself


def test_without_ollama_only_api_models_are_offered(monkeypatch):
    ok, saved, asked = _run(monkeypatch, [API_OPTION], info={"ok": False})
    assert ok and [o[0] for o in asked[0][1]] == [API_OPTION] and len(asked) == 1


def test_escape_chooses_nothing(monkeypatch):
    ok, saved, _ = _run(monkeypatch, [None])
    assert not ok and saved == {}


def test_choices_only_fit_the_gpu():
    tags = [t for t, _ in main_choices(from_spec("gpu:4"), [])]
    assert tags and all("7b" not in t and "14b" not in t for t in tags)
    assert all(t != "mistral:7b" for t, _ in helper_choices(GPU8, ["mistral:7b"]))


def test_logo_sizes_fit_short_windows():
    assert BANNER_ROWS["full"] <= 11 and BANNER_ROWS["small"] < BANNER_ROWS["full"] and BANNER_ROWS["word"] == 5
    assert all("⣿" not in "".join(c for c, _ in line) for line in banner("full"))        # solid pixels, no braille dots


def test_helper_is_chosen_and_installed_automatically(monkeypatch):
    from pal.terminal.setup import auto_helper
    saved, pulled = {}, []
    monkeypatch.setattr(backend, "save_prefs", lambda **k: saved.update(k))
    c = Console(file=open("/dev/null", "w"))
    assert auto_helper(c, GPU8, info=INFO, puller=lambda t: pulled.append(t) or True) == "qwen2.5:1.5b" and pulled == []   # installed: left as it is
    big = from_spec("gpu:24:64")
    assert auto_helper(c, big, info=INFO, puller=lambda t: pulled.append(t) or True) == "llama3.2:3b" and pulled == ["llama3.2:3b"]
    assert saved["helper_model"] == "llama3.2:3b"
    assert auto_helper(c, from_spec("cpu:16"), info=INFO) == ""                       # no GPU: no helper
    assert auto_helper(c, GPU8, info={"ok": False}) == ""                              # Ollama not running
    assert auto_helper(c, big, info=INFO, puller=lambda t: False) == "" and saved["helper_model"] == ""


def test_installed_models_keep_their_real_name_once():
    tags = [t for t, _ in main_choices(GPU8, ["mistral:7b", "llama3.1:8b-instruct-q4_K_M", "qwen2.5:1.5b"])]
    assert "llama3.1:8b-instruct-q4_K_M" in tags and "llama3.1:8b" not in tags
    assert tags.index("llama3.1:8b-instruct-q4_K_M") < tags.index("qwen2.5-coder:7b")     # installed first


def test_recording_goes_to_whisper_without_a_temporary_file():
    import numpy as np

    from pal.terminal import voice
    audio = (0.2 * np.sin(np.linspace(0, 100, 16000))).astype("float32")
    back = voice._samples(voice.wav_bytes(audio), ".wav")
    assert back is not None and len(back) == 16000 and abs(float(back[100]) - float(audio[100])) < 1e-3
    assert voice._samples(b"webm", ".webm") is None


def test_the_input_box_has_no_two_helpers_with_the_same_name():
    """A second inner function with the same name silently replaces the first (Tab once broke this way)."""
    import ast
    from collections import Counter
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[2] / "src/pal/terminal/ui.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "read_input")
    names = Counter(n.name for n in fn.body if isinstance(n, ast.FunctionDef) and n.name != "_")
    assert [n for n, c in names.items() if c > 1] == []


def test_each_agent_has_its_own_hint():
    from pal.terminal.ui import PLACEHOLDERS
    assert PLACEHOLDERS["build"].startswith("Build anything") and PLACEHOLDERS["plan"].startswith("Plan") and PLACEHOLDERS["ask"].startswith("Ask")
