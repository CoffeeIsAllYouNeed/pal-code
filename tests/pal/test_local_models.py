"""Picking local models for the computer Pal runs on: hardware detection, the catalogue by GPU memory, setup, and the helper model."""
import pytest
from rich.console import Console

from pal.terminal import backend, hardware
from pal.terminal import local_models as lm
from pal.terminal.hardware import GPU, Hardware, from_spec
from pal.terminal.setup import run_setup, show_table


@pytest.mark.parametrize("spec,main,helper", [
    ("gpu:4", "qwen2.5-coder:3b", "qwen2.5:0.5b"),
    ("gpu:8", "qwen2.5-coder:7b", "qwen2.5:1.5b"),
    ("gpu:12", "qwen2.5-coder:14b", "qwen2.5:1.5b"),
    ("gpu:16", "qwen2.5-coder:14b", "llama3.2:3b"),
    ("gpu:20", "devstral-small-2:24b", "llama3.2:3b"),
    ("gpu:24", "qwen3-coder:30b", "llama3.2:3b"),
    ("gpu:48", "qwen3-coder:30b", "llama3.2:3b"),
    ("mac:16", "qwen2.5-coder:14b", "qwen2.5:1.5b"),
])
def test_the_recommendation_grows_with_the_gpu(spec, main, helper):
    r = lm.recommend(from_spec(spec))
    assert (r.main.tag, r.helper.tag) == (main, helper)


def test_the_main_model_always_fits_and_reasoning_models_are_never_picked_by_themselves():
    for spec in ("gpu:4", "gpu:6", "gpu:8", "gpu:10", "gpu:12", "gpu:16", "gpu:20", "gpu:24", "gpu:32", "gpu:48", "gpu:80", "mac:8", "mac:24", "mac:64"):
        hw = from_spec(spec)
        r = lm.recommend(hw)
        assert not r.main.thinks and lm.fit(r.main.need_gb, hw) == "gpu", spec
        assert all(lm.fit(m.need_gb, hw) == "gpu" for m in r.others), spec


def test_without_a_gpu_there_is_no_local_model_at_all():
    for spec in ("cpu:4", "cpu:16", "cpu:64"):
        hw = from_spec(spec)
        assert lm.api_only(hw) and lm.recommend(hw) is None and lm.fit(2.0, hw) == "no gpu"
        assert lm.best_installed([{"name": "llama3.2:3b", "size_gb": 2.0, "params_b": 3.2, "embedding": False}], hw) is None
    assert not lm.api_only(from_spec("mac:8")) and not lm.api_only(from_spec("gpu:4"))


def test_bigger_gpus_never_get_a_weaker_model():
    ranks = [lm.recommend(from_spec(f"gpu:{v}")).main.rank for v in (4, 6, 8, 12, 16, 24, 48, 80)]
    assert ranks == sorted(ranks)


def test_every_model_is_in_exactly_one_gpu_group_and_fits_that_gpu():
    groups = lm.by_tier()
    assert sum(len(ms) for _, ms in groups) == len(lm.CATALOGUE)
    for tier, models in groups:
        for m in models:
            assert m.need_gb <= tier * lm.USABLE + 1e-9


def test_fit_labels_for_gpu_partial_cpu_and_too_big():
    gpu8 = from_spec("gpu:8:16")
    assert lm.fit(5.0, gpu8) == "gpu" and lm.fit(10.0, gpu8) == "partial" and lm.fit(40.0, gpu8) == "too big"
    assert lm.fit(4.0, from_spec("cpu:16")) == "no gpu"


def test_installed_models_are_judged_by_size_when_unknown_and_the_best_fitting_one_is_chosen():
    hw = from_spec("gpu:8")
    models = [{"name": "qwen2.5-coder:14b", "size_gb": 9.0, "params_b": 14.7, "embedding": False}, {"name": "llama3.1:8b", "size_gb": 4.9, "params_b": 8.0, "embedding": False},
              {"name": "mystery:3b", "size_gb": 2.0, "params_b": 3.0, "embedding": False}, {"name": "nomic-embed-text", "size_gb": 0.3, "params_b": None, "embedding": True}]
    rows = {r["name"]: r["fit"] for r in lm.installed_fit(models, hw)}
    assert rows == {"qwen2.5-coder:14b": "partial", "llama3.1:8b": "gpu", "mystery:3b": "gpu"}
    assert lm.best_installed(models, hw) == "llama3.1:8b"
    assert lm.best_installed(models, hw, exclude="llama3.1:8b") == "mystery:3b"


def test_apple_silicon_uses_part_of_its_shared_memory_and_nvidia_cards_add_up():
    assert from_spec("mac:16").vram_gb == 11.2
    hw = Hardware("Linux", 64, [GPU("RTX 3090", 24, "nvidia"), GPU("RTX 3090", 24, "nvidia"), GPU("Intel UHD", 1, "intel")])
    assert hw.vram_gb == 48 and "RTX 3090 24 GB" in hw.summary()
    assert Hardware("Windows", 16, [GPU("Intel Iris", 0, "intel")]).kind == "cpu"


def test_nvidia_smi_output_is_read(monkeypatch):
    monkeypatch.setattr(hardware, "_run", lambda cmd, timeout=4.0: "NVIDIA GeForce RTX 3060, 12288\n" if cmd[0] == "nvidia-smi" else "")
    gpus = hardware._nvidia()
    assert gpus == [GPU("NVIDIA GeForce RTX 3060", 12.0, "nvidia")]


def test_amd_cards_are_read_from_sysfs(tmp_path):
    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_vram_total").write_text(str(16 * 1024**3))
    (dev / "product_name").write_text("Radeon RX 7800 XT")
    (tmp_path / "card0-HDMI-A-1").mkdir()
    assert hardware._amd_linux(tmp_path) == [GPU("Radeon RX 7800 XT", 16.0, "amd")]


def test_detect_never_raises():
    hw = hardware.detect()
    assert hw.ram_gb >= 0 and isinstance(hw.summary(), str)


def test_setup_downloads_what_is_missing_and_remembers_main_and_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": True, "models": [{"name": "qwen2.5:1.5b", "embedding": False}]})
    pulled = []
    out = run_setup(Console(record=True, width=120), from_spec("gpu:8"), puller=lambda tag: pulled.append(tag) or True, ollama=lambda: True, ask=lambda q: True)
    assert pulled == ["qwen2.5-coder:7b"] and out["downloaded"]
    prefs = backend.load_prefs()
    assert (prefs["model"], prefs["helper_model"], prefs["setup_done"]) == ("qwen2.5-coder:7b", "qwen2.5:1.5b", "1")


def test_setup_without_ollama_says_how_to_install_it_and_still_saves_the_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    console = Console(record=True, width=160)
    out = run_setup(console, Hardware("Windows", 16, [GPU("RTX 4060", 8, "nvidia")]), ollama=lambda: False)
    text = console.export_text()
    assert "ollama.com/download" in text and not out["downloaded"] and backend.load_prefs()["model"] == "qwen2.5-coder:7b"
    assert "setup_done" not in backend.load_prefs()            # it runs again next time, when Ollama may be there


def test_declining_the_download_is_remembered_so_setup_does_not_nag(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": True, "models": []})
    out = run_setup(Console(record=True), from_spec("gpu:12"), puller=lambda tag: pytest.fail("must not download"), ollama=lambda: True, ask=lambda q: False)
    assert not out["downloaded"] and backend.load_prefs()["setup_done"] == "1"


def test_the_full_table_lists_every_model_with_its_fit():
    console = Console(record=True, width=160)
    show_table(console, from_spec("gpu:12"))
    text = console.export_text()
    assert all(m.tag in text for m in lm.CATALOGUE) and "fits your GPU" in text and "too big for this computer" in text


def test_a_helper_is_only_built_when_setup_chose_one_and_it_is_installed():
    assert backend.make_helper({}, {"ok": True, "models": []}) is None
    assert backend.make_helper({"helper_model": "qwen2.5:1.5b"}, {"ok": True, "models": []}) is None


def test_the_last_model_still_wins_but_otherwise_the_best_fit_is_chosen():
    info = {"ok": True, "recommended": "qwen2.5-coder:32b", "models": [{"name": "qwen2.5-coder:32b", "size_gb": 20, "params_b": 32.8, "embedding": False},
                                                                       {"name": "qwen2.5-coder:7b", "size_gb": 4.7, "params_b": 7.6, "embedding": False}]}
    assert backend.pick_default(info, {}, from_spec("gpu:8")) == ("local", "qwen2.5-coder:7b")
    assert backend.pick_default(info, {"provider": "local", "model": "qwen2.5-coder:32b"}, from_spec("gpu:8")) == ("local", "qwen2.5-coder:32b")


def test_setup_on_a_computer_without_a_gpu_offers_api_models_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    console = Console(record=True, width=160)
    out = run_setup(console, from_spec("cpu:16"), puller=lambda tag: pytest.fail("no downloads without a GPU"), ollama=lambda: True)
    text = console.export_text()
    assert out["api_only"] and "No GPU on this computer, so Pal uses an API model" in text and "routing, trimming" in text
    assert backend.load_prefs()["setup_done"] == "1" and backend.load_prefs().get("helper_model", "") == ""


def test_asking_for_an_api_model_keeps_the_key_out_of_every_file(tmp_path, monkeypatch):
    from pal.terminal.setup import ask_api
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import os
    answers = iter(["openrouter", "some/model"])
    try:
        got = ask_api(Console(record=True), "Linux", ask=lambda q: next(answers), secret=lambda q: "sk-or-secret-123")
        assert got == ("openrouter", "some/model") and os.environ["OPENROUTER_API_KEY"] == "sk-or-secret-123"
        assert "sk-or-secret" not in (tmp_path / "pal.json").read_text()
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)              # ask_api sets it for the process, like a real session


def test_without_a_gpu_the_engine_sends_the_message_as_typed_and_never_asks_a_local_model(tmp_path):
    from pal.chat import ChatService
    from pal.config import load_settings
    settings = load_settings(env={}, overrides={"storage_dir": str(tmp_path), "local_llm": {"enabled": False}, "memory": {"enabled": False}})
    svc = ChatService(settings)
    assert svc.local_off and svc.router().core is None and not svc.router().enabled and settings.task_routing.enabled and svc.local_status()["ok"] is None and svc.warm_up_local() is None
    res = svc.reduce(len, "please could you kindly fix the bug", None, None, [])
    assert res.reduced == res.original and res.tokens_saved == 0
    svc.close()


def test_an_api_only_session_never_runs_local_helper_jobs(tmp_path):
    from pal.terminal.session import make_session
    from tests.pal.fakes import ScriptedCore
    core = ScriptedCore()
    s = make_session(core, "some/model", "openrouter", tmp_path, "plan", local=False)
    try:
        t = s.ask("Please could you kindly explain what this folder is for, thank you so much")
        helper_calls = [c for c in core.calls if c[0].startswith(("You route", "You audit")) or c[1].startswith("<text>")]
        assert helper_calls == [] and t.helper_tokens == 0 and not s.local
    finally:
        s.close()


def test_a_session_with_a_gpu_still_runs_them(tmp_path):
    from pal.terminal.session import make_session
    from tests.pal.fakes import ScriptedCore
    core = ScriptedCore()
    s = make_session(core, "m", "local", tmp_path, "plan")
    try:
        s.ask("Please could you kindly explain what this folder is for, thank you so much")
        assert any(c[1].startswith("<text>") or c[0].startswith(("You route", "You audit")) for c in core.calls)
    finally:
        s.close()


def test_with_an_api_main_model_only_the_helper_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": True, "models": []})
    pulled = []
    run_setup(Console(record=True), from_spec("gpu:8"), puller=lambda tag: pulled.append(tag) or True, ollama=lambda: True, ask=lambda q: True,
                    main_kind="api", api_chooser=lambda console, system: backend.save_prefs(provider="openai", model="gpt-5") or ("openai", "gpt-5"))
    prefs = backend.load_prefs()
    assert pulled == ["qwen2.5:1.5b"] and (prefs["provider"], prefs["model"], prefs["helper_model"]) == ("openai", "gpt-5", "qwen2.5:1.5b")


def test_a_local_main_model_is_chosen_from_the_ones_that_suit_the_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": True, "models": [{"name": "qwen2.5:1.5b"}]})
    asked, pulled = [], []

    def pick(question, options, recommended=0, other=False):
        asked.append((question, [o[0] for o in options], other))
        return "Local model on this computer" if len(asked) == 1 else "llama3.1:8b"
    run_setup(Console(record=True), from_spec("gpu:8"), puller=lambda tag: pulled.append(tag) or True, ollama=lambda: True, ask=lambda q: True, pick=pick)
    assert asked[1][1][0] == "qwen2.5-coder:7b" and "llama3.1:8b" in asked[1][1] and asked[1][2] is True
    assert pulled == ["llama3.1:8b"] and backend.load_prefs()["model"] == "llama3.1:8b"


def test_a_model_typed_by_name_is_downloaded_too(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": True, "models": [{"name": "qwen2.5:1.5b"}]})
    pulled = []
    run_setup(Console(record=True), from_spec("gpu:12"), puller=lambda tag: pulled.append(tag) or True, ollama=lambda: True, ask=lambda q: True,
              main_kind="local", pick=lambda q, o, r=0, other=False: "qwen3:8b")
    assert pulled == ["qwen3:8b"]


def test_a_6_gb_gpu_gets_a_coding_model():
    assert lm.recommend(from_spec("gpu:6")).main.tag == "qwen2.5-coder:3b"


def test_suitable_models_put_the_best_first_and_mark_installed_ones():
    rows = lm.suitable(from_spec("gpu:8"), ["mistral:7b", "llama3.1:8b-instruct-q4_K_M"])
    assert rows[0][0] == "qwen2.5-coder:7b" and not rows[0][2]
    installed = [r for r in rows if r[2]]
    assert {r[0] for r in installed} == {"mistral:7b", "llama3.1:8b"} and all("installed" in r[1] for r in installed)
    assert lm.suitable(from_spec("cpu:16")) == []


def test_an_api_main_model_does_not_make_setup_ask_for_a_local_one(tmp_path, monkeypatch):
    from pal.terminal.setup import ensure_local
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    backend.save_prefs(provider="openai", model="gpt-5", helper_model="qwen2.5:1.5b", setup_done="1")
    st = ensure_local(Console(record=True), from_spec("gpu:8"), ollama=lambda: True, info=lambda h: {"ok": True, "models": [{"name": "qwen2.5:1.5b"}]},
                      setup=lambda *a, **k: pytest.fail("no setup: only the helper is needed and it is installed"))
    assert st.state == "ready" and "helper qwen2.5:1.5b" in st.detail


def test_the_model_picker_downloads_a_model_that_is_not_installed(tmp_path, monkeypatch):
    from pal.terminal import cli, decide, setup, ui
    from pal.terminal.session import PalSession
    monkeypatch.setattr(backend, "_HW", from_spec("gpu:8"))
    s = PalSession(object(), "qwen2.5-coder:3b", "local", tmp_path, "build")
    s.helper_model = "qwen2.5:1.5b"
    seen, pulled = {}, []
    monkeypatch.setattr(decide, "ask", lambda console, q, reader=None: seen.update(labels=[o.label for o in q.options]) or "qwen2.5-coder:7b")
    monkeypatch.setattr(ui, "confirm", lambda console, q: True)
    monkeypatch.setattr(setup, "ollama_installed", lambda: True)
    monkeypatch.setattr(setup, "pull", lambda tag: pulled.append(tag) or True)
    chosen = cli.pick_model(ui.make_console(), s, {"ok": True, "models": [{"name": "qwen2.5-coder:3b"}, {"name": "qwen2.5:1.5b"}]})
    assert chosen == "qwen2.5-coder:7b" and pulled == ["qwen2.5-coder:7b"]
    assert seen["labels"][0] == "qwen2.5-coder:7b" and "qwen2.5:1.5b" not in seen["labels"] and seen["labels"][-1] == cli.API_CHOICE
