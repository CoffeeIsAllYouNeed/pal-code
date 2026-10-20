"""The web side of the Studio: a page, and the endpoints it talks to. Thin: the work is in the other modules."""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from flask import Response, jsonify, render_template, request

from pal.errors import UserInputError
from pal.llm_wrapper.components.base import WrapperConfig
from pal.llm_wrapper.modes import PROVIDER_KEYS, default_mode, describe_modes, providers_with_keys
from pal.paths import data_dir
from pal.studio import plan as planlib
from pal.studio.catalog import catalog
from pal.studio.editor import MAX_INSTRUCTION, lookup, save_file
from pal.studio.planner import Planner
from pal.studio.policy import MAX_FILE_BYTES, MAX_FILES, PolicyError, safe_path
from pal.studio.runs import RunStore
from pal.terminal.models import discover

MAX_REFERENCES, MAX_REFERENCE_BYTES, MAX_REFERENCES_TOTAL = 8, 150_000, 400_000


class _NoModel:
    """Stands in for the planner's model when none can be reached: every call fails, so the rules-based plan is used."""

    def chat(self, syscall):
        raise ConnectionError("no local model is available")


def clean_references(raw) -> list[dict]:
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or len(raw) > MAX_REFERENCES:
        raise UserInputError(f"attach at most {MAX_REFERENCES} reference files")
    out, total = [], 0
    for r in raw:
        if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not isinstance(r.get("name"), str):
            raise UserInputError("a reference file must have a name and text")
        name = re.sub(r"[^A-Za-z0-9_. \-]", "_", Path(r["name"].replace("\\", "/")).name)[:80] or "reference.txt"
        size = len(r["text"].encode("utf-8"))
        total += size
        if size > MAX_REFERENCE_BYTES or total > MAX_REFERENCES_TOTAL:
            raise UserInputError(f"reference files are too big (at most {MAX_REFERENCE_BYTES // 1000} KB each and {MAX_REFERENCES_TOTAL // 1000} KB together)")
        out.append({"name": name, "text": r["text"]})
    return out


def register_studio(app, service, *, planner_core: Callable | None = None, core_factory: Callable | None = None, discover_fn: Callable | None = None,
                    hot_turns: int | None = None, project_dir: Path | None = None) -> RunStore:
    """Add the Studio page and endpoints to the app. The three callables are for tests: the planner's model, the builder's model, local-model discovery."""
    cfg = service.settings
    home = Path(cfg.storage_dir).expanduser() if cfg.storage_dir else data_dir()
    host = lambda: (cfg.local_llm.get("host") or "http://localhost:11434")  # noqa: E731

    def num_ctx() -> int:
        return cfg.studio.num_ctx

    def local_core(model: str, ctx: int):
        from pal.llm_wrapper import build_wrapper
        return build_wrapper(WrapperConfig(backend="ollama", model=model, host=host(), extra={"timeout": 900, "litellm_params": {"num_ctx": ctx, "keep_alive": "30m"}}))

    def default_core_factory(plan: dict, api_key: str | None):
        from pal.chat.providers import build_api_core
        m = plan["model"]
        if m["mode"] == "api":
            return build_api_core(m["provider"], m["model"], api_key)
        if not m["model"]:
            raise UserInputError("no local model is chosen: start Ollama and pull a model, or switch to an API model")
        return local_core(m["model"], num_ctx())

    runs = RunStore(home / "studio", core_factory or default_core_factory, max_runs=cfg.studio.max_runs, hot_turns=hot_turns, project_dir=project_dir)
    app.extensions["studio_runs"] = runs

    def local() -> dict:
        found = (discover_fn or discover)(host())
        if discover_fn is not None or not found.get("ok"):
            return found
        return fit_to_hardware(found)

    def fit_to_hardware(found: dict) -> dict:
        """Mark each installed model with how it fits this computer's GPU, and recommend the best one that fits (not just the largest)."""
        from pal.terminal.backend import hardware
        from pal.terminal.local_models import FIT_LABEL, api_only, best_installed, installed_fit
        hw = hardware()
        if api_only(hw):
            return {"ok": False, "url": found.get("url"), "models": [], "recommended": None, "hardware": hw.summary(),
                    "error": "No GPU on this computer, so Pal uses API models only: choose an API model (and paste a key) in the plan's Model step."}
        fits = {r["name"]: r["fit"] for r in installed_fit(found["models"], hw)}
        models = [{**m, "fit": fits.get(m["name"], ""), "label": m["label"] + (f" | {FIT_LABEL[fits[m['name']]]}" if m["name"] in fits else "")} for m in found["models"]]
        best = best_installed(found["models"], hw) or found["recommended"]
        return {**found, "models": models, "recommended": best, "hardware": hw.summary()}

    def ready_tools() -> list[str]:
        return [t["name"] for t in catalog(cfg.tools, os.environ)["tools"] if t["ready"]]

    def info(model: str | None = None) -> tuple[dict, dict]:
        loc = local()
        installed = [m["name"] for m in loc["models"] if not m["embedding"]]
        chosen = model if model in installed else loc["recommended"]
        return {"installed": installed, "recommended": chosen, "ready_tools": ready_tools()}, loc

    def planner_for(i: dict, loc: dict) -> Planner:
        if not loc["ok"] or not i["recommended"]:
            return Planner(_NoModel(), i)
        if planner_core:
            return Planner(planner_core(i["recommended"]), i)
        return Planner(local_core(i["recommended"], cfg.studio.planner_num_ctx), i)

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise UserInputError("send a JSON object")
        return data

    def user_set(data: dict) -> list[str]:
        v = data.get("user_set") or []
        return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []

    def normalized(data: dict, i: dict) -> tuple[dict, list[str]]:
        if not isinstance(data.get("plan"), dict):
            raise UserInputError("send the plan")
        return planlib.normalize(data["plan"], installed=i["installed"], recommended=i["recommended"], user_set=user_set(data), ready_tools=i["ready_tools"])

    @app.get("/studio")
    def studio_page():
        resp = Response(render_template("studio.html"), mimetype="text/html")
        # The page shows text a model wrote, so it may not run inline scripts or load anything from another site.
        resp.headers["Content-Security-Policy"] = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                                                   "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        return resp

    @app.get("/api/studio/catalog")
    def studio_catalog():
        loc = local()
        return jsonify({**catalog(cfg.tools, os.environ), "local": loc, "llm_modes": describe_modes(), "llm_mode": default_mode(), "providers": providers_with_keys(),
                        "limits": {"num_ctx": num_ctx(), "max_files": MAX_FILES, "max_file_kb": MAX_FILE_BYTES // 1000, "max_references": MAX_REFERENCES, "max_reference_kb": MAX_REFERENCE_BYTES // 1000}})

    @app.post("/api/studio/plan")
    def studio_plan():
        data = body()
        command = (data.get("command") or "").strip() if isinstance(data.get("command"), str) else ""
        if not command:
            raise UserInputError("tell me what you want built")
        if len(command) > 4000:
            raise UserInputError("that is too long: describe it in a few sentences (under 4000 characters)")
        i, loc = info(data.get("model") if isinstance(data.get("model"), str) else None)
        r = planner_for(i, loc).plan(command, user_set(data))
        return jsonify({"plan": r.plan, "notes": r.notes, "source": r.source, "message": r.message, "planner_model": i["recommended"], "local": {"ok": loc["ok"], "error": loc["error"]}})

    @app.post("/api/studio/revise")
    def studio_revise():
        data = body()
        instruction = (data.get("instruction") or "").strip() if isinstance(data.get("instruction"), str) else ""
        if not instruction or len(instruction) > 2000:
            raise UserInputError("say what to change (under 2000 characters)")
        i, loc = info(data.get("model") if isinstance(data.get("model"), str) else None)
        current, _ = normalized(data, i)
        r = planner_for(i, loc).revise(current, instruction, user_set(data))
        return jsonify({"plan": r.plan, "notes": r.notes, "source": r.source, "message": r.message, "changes": r.changes})

    @app.post("/api/studio/validate")
    def studio_validate():
        data = body()
        i, _ = info()
        plan, notes = normalized(data, i)
        return jsonify({"plan": plan, "notes": notes})

    @app.post("/api/studio/run")
    def studio_run():
        data = body()
        i, _ = info()
        plan, notes = normalized(data, i)
        key = data.get("api_key") if isinstance(data.get("api_key"), str) else None
        m = plan["model"]
        if m["mode"] == "api" and not (key or os.environ.get(PROVIDER_KEYS[m["provider"]])):
            raise UserInputError(f"no API key for {m['provider']}: paste one, or switch to the local model")
        run = runs.start(plan, clean_references(data.get("references")) if plan["permissions"].get("read_references") else [], key)
        return jsonify({"run_id": run.id, "notes": notes})

    @app.get("/api/studio/run/<run_id>")
    def studio_run_status(run_id):
        run = runs.get(run_id)
        try:
            since = int(request.args.get("since", "0"))
        except ValueError:
            raise UserInputError("since must be a number") from None
        return jsonify({**run.summary(), "events": run.events_since(since)})

    @app.post("/api/studio/run/<run_id>/cancel")
    def studio_cancel(run_id):
        runs.get(run_id).cancel.set()
        return jsonify({"ok": True})

    @app.get("/api/studio/run/<run_id>/files")
    def studio_files(run_id):
        state = runs.get(run_id).arm(request.args.get("arm", ""))
        files = state.build.files if state.build else {}
        return jsonify({"files": [{"path": p, "bytes": len(c.encode("utf-8"))} for p, c in sorted(files.items())]})

    @app.get("/api/studio/run/<run_id>/file")
    def studio_file(run_id):
        state = runs.get(run_id).arm(request.args.get("arm", ""))
        try:
            path = safe_path(request.args.get("path", ""))
        except PolicyError as e:
            raise UserInputError(str(e)) from None
        files = {p.lower(): (p, c) for p, c in (state.build.files if state.build else {}).items()}
        if path.lower() not in files:
            raise UserInputError("no such file in this build")
        name, content = files[path.lower()]
        limit = MAX_FILE_BYTES if request.args.get("full") == "1" else 60_000       # the editor needs the whole file: saving a shortened one would lose the rest
        return jsonify({"path": name, "content": content[:limit], "truncated": len(content) > limit})

    @app.put("/api/studio/run/<run_id>/file")
    def studio_save_file(run_id):
        data = body()
        arm, path, content = (str(data.get(k)) if isinstance(data.get(k), str) else "" for k in ("arm", "path", "content"))
        state = runs.editable(runs.get(run_id), arm)
        return jsonify(save_file(state, path, content))

    @app.post("/api/studio/run/<run_id>/ask")
    def studio_ask(run_id):
        """Ask Pal to change the open file (or answer a question about the project). Proposes only: nothing is saved until the person accepts."""
        data = body()
        instruction = (data.get("instruction") or "").strip() if isinstance(data.get("instruction"), str) else ""
        if not instruction or len(instruction) > MAX_INSTRUCTION:
            raise UserInputError(f"say what to change (under {MAX_INSTRUCTION} characters)")
        run = runs.get(run_id)
        arm = str(data.get("arm")) if isinstance(data.get("arm"), str) else ""
        state = runs.editable(run, arm)
        path = str(data.get("path")) if isinstance(data.get("path"), str) else ""
        build = state.build
        assert build is not None                                    # editable() checked it  # noqa: S101
        if path and lookup(build.files, path) is None:
            raise UserInputError("that file is not in this build")
        key = data.get("api_key") if isinstance(data.get("api_key"), str) else None
        m = run.plan["model"]
        if m["mode"] == "api" and not (key or os.environ.get(PROVIDER_KEYS[m["provider"]])):
            raise UserInputError(f"no API key for {m['provider']}: paste one, or switch to the local model")
        return jsonify(runs.editor(run, arm, key).ask(build.files, path, instruction))

    @app.get("/api/studio/run/<run_id>/download")
    def studio_download(run_id):
        run = runs.get(run_id)
        arm = request.args.get("arm", "")
        data = runs.zip_bytes(run, arm)
        workspace = run.arm(arm).workspace
        folder = workspace.name if workspace else "project"
        return Response(data, mimetype="application/zip", headers={"Content-Disposition": f'attachment; filename="{folder}-{arm}.zip"'})

    return runs
