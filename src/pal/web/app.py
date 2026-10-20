"""HTTP layer: a thin Flask app around ChatService. No business logic lives here.

- create_app() is a factory, so tests and servers build their own app around their own service.
- Errors: PalError subclasses map to their HTTP status with a safe message; anything else is
  logged with a traceback and answered with a generic 500 (no traceback ever reaches the client).
- API keys arrive per request and are passed straight through; they are never logged or stored.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict
from typing import Any, Optional

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from pal.chat import API_PROVIDERS, ChatService
from pal.chat.demo import Demo
from pal.chat.sessions import validate_session_id
from pal.chat.uploads import (  # noqa: F401  (read_upload is part of this module's API)
    MEDIA_TYPES,
    _clean_filename,
    _is_text,
    read_upload,
)
from pal.config import Settings
from pal.errors import NotFoundError, PalError, UserInputError
from pal.input_savers.components.methods import METHODS
from pal.llm_wrapper.modes import default_mode, describe_modes
from pal.sdk.agent import list_agents

log = logging.getLogger(__name__)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _saver_json(res) -> Optional[dict]:
    if res is None:  # turn handled without an API call: no input saver ran
        return None
    return {
        "original": res.original, "reduced": res.reduced,
        "tokens_original": res.tokens_original, "tokens_reduced": res.tokens_reduced,
        "tokens_saved": res.tokens_saved,
        "steps": [asdict(s) for s in res.steps],
    }


def _flag(v: Any) -> bool:
    return v is True or str(v).lower() in ("1", "true", "on", "yes")


def _pasted(v: Any) -> list:
    """Pasted spans the client tracked; the LLM never sees or edits them."""
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    try:
        out = json.loads(v or "[]")
    except ValueError:
        return []
    return [x for x in out if isinstance(x, str)] if isinstance(out, list) else []


def create_app(service: Optional[ChatService] = None, settings: Optional[Settings] = None, studio: Optional[dict] = None) -> Flask:
    """`studio` passes options to the Studio (its planner model, builder model and model discovery); tests use it to plug in fakes."""
    service = service or ChatService(settings)
    cfg = service.settings
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.server.max_upload_mb * 1024 * 1024
    app.extensions["pal_service"] = service
    demo = Demo(service)

    @app.after_request
    def _headers(resp: Response) -> Response:
        for k, v in _SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        return resp

    @app.errorhandler(PalError)
    def _pal_error(e: PalError):
        if e.http_status >= 500:
            log.error("request failed: %s: %s", type(e).__name__, e)
        msg = str(e) if e.public else "internal error"
        return jsonify({"error": msg, "code": type(e).__name__}), e.http_status

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        msg = "upload too large" if e.code == 413 else (e.description or e.name)
        return jsonify({"error": msg, "code": e.name.replace(" ", "")}), e.code or 500

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        log.exception("unhandled error")
        return jsonify({"error": "internal error", "code": "InternalError"}), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.get("/api/config")
    def config():
        return jsonify({
            "providers": {p: bool(os.environ.get(env)) for p, env in API_PROVIDERS.items()},
            "llm_modes": describe_modes(), "llm_mode": default_mode(),
            "methods": [{"id": m, "name": METHODS[m].name} for m in cfg.input_methods],
            "local_llm": {k: v for k, v in cfg.local_llm.items() if k != "api_key"},
            "mode": cfg.input_mode,
            "routing": {"enabled": cfg.task_routing.enabled, "default": cfg.task_routing.default},
            "graph": {"enabled": service.graph.enabled, "available": service.graph.available()},
            "memory": {"enabled": cfg.memory.enabled},
            "agents": list_agents(service.agents) if cfg.agents.enabled else [],
            "metrics": {"enabled": cfg.metrics.enabled},
        })

    @app.get("/api/llm/modes")
    def llm_modes():
        return jsonify({"default": default_mode(), "modes": describe_modes()})

    @app.get("/api/llm/models")
    def llm_models():
        """OpenRouter's models for the Explore mode: ?q=words&free=1&limit=50."""
        try:
            limit = max(1, min(int(request.args.get("limit", "50")), 500))
        except ValueError:
            raise UserInputError("limit must be a number") from None
        free = request.args.get("free", "").lower() in ("1", "true", "yes")
        return jsonify({"models": service.model_catalogue(request.args.get("q", ""), free, limit)})

    @app.get("/api/voice")
    def voice_status():
        """Which speech-to-text engines this server has: the page records and sends the audio here when there is one."""
        from pal.terminal import voice
        return jsonify({"engines": voice.engines()})

    @app.post("/api/transcribe")
    def transcribe():
        """The recorded audio (the request body, as the browser's MediaRecorder made it) -> text. Nothing is kept."""
        from pal.terminal import voice
        audio = request.get_data(cache=False)
        if not audio:
            raise UserInputError("no audio received")
        kind = (request.content_type or "").split(";")[0]
        suffix = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"}.get(kind, ".webm")
        try:
            heard = voice.transcribe(audio, suffix)
        except voice.VoiceUnavailable as e:
            raise UserInputError(str(e)) from None
        return jsonify({"text": heard.text, "engine": heard.engine})

    @app.get("/api/context_window")
    def context_window_size():
        """The model's context window, for the page's "tokens left" counter."""
        from pal.context_window import context_window
        return jsonify({"window": context_window(request.args.get("provider", ""), request.args.get("model", ""))})

    @app.get("/api/local_status")
    def local_status():
        return jsonify(service.local_status())

    @app.post("/api/chat")
    def chat():
        d: dict[str, Any]
        if request.is_json:
            d, files = request.get_json(silent=True) or {}, []
        else:
            d = dict(request.form.to_dict())
            if "methods" in request.form:
                d["methods"] = request.form.getlist("methods")
            files = request.files.getlist("files")
        sid = d.get("session_id") or uuid.uuid4().hex
        attachments, errors = [], []
        for f in files:
            try:
                attachments.append(read_upload(service, f, sid))
            except UserInputError as e:
                errors.append(str(e))
        message = d.get("message") or ""
        if errors and not (message.strip() or attachments):
            return jsonify({"error": "; ".join(errors)}), 400
        res = service.send(
            sid, d.get("provider", ""), d.get("model", ""), d.get("api_key") or None, message, attachments,
            d.get("methods"), d.get("mode") or None, _pasted(d.get("pasted")), d.get("route") or None,
            agent=d.get("agent") or None)
        return jsonify({
            "reply": res.reply, "sent": res.sent,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "saver": _saver_json(res.saver), "file_errors": errors,
            "handled_by": res.handled_by, "route_reason": res.route_reason,
            "graph": res.graph, "documents": res.documents, "memory": res.memory,
            "context": res.context, "tools": res.tools, "latency": res.latency, "fallback": res.fallback,
            "cost_usd": res.cost_usd, "files": res.files,
        })

    @app.post("/api/experiment")
    def experiment():
        d = request.get_json(silent=True) or {}
        if not (d.get("text") or "").strip():
            raise UserInputError("enter some text")
        out = service.experiment(
            d["text"], d.get("methods"), d.get("provider", ""), d.get("model", ""), d.get("api_key") or None,
            _flag(d.get("call_api")), d.get("mode") or None, _pasted(d.get("pasted")))
        return jsonify({"saver": _saver_json(out["saver"]), "counter": out["counter"], "api": out["api"]})

    @app.post("/api/reset")
    def reset():
        service.reset((request.get_json(silent=True) or {}).get("session_id", ""))
        return jsonify({"ok": True})

    @app.get("/api/workspace")
    def workspace_list():
        """Files an agent saved for this session (its workspace in the OS kernel interface)."""
        sess = _session_or_404(request.args.get("session_id", ""))
        return jsonify({"files": service.os_kernel.workspace(sess.agent_id).list()})

    @app.get("/api/workspace/file")
    def workspace_file():
        sess = _session_or_404(request.args.get("session_id", ""))
        path = request.args.get("path", "")
        data = service.os_kernel.workspace(sess.agent_id).read_bytes(path)   # validated: inside this session's workspace only
        ext = "." + path.rsplit(".", 1)[1].lower() if "." in path else ""
        mime = MEDIA_TYPES.get(ext, "text/plain" if _is_text(data) else "application/octet-stream")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", path.rsplit("/", 1)[-1]) or "file.txt"
        return Response(data, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    def _session_or_404(session_id: str):
        validate_session_id(session_id)                                     # a malformed id is a 400
        sess = service.sessions.get(session_id)
        if sess is None:
            raise NotFoundError("no such session")
        return sess

    @app.get("/api/metrics")
    def metrics():
        sid = request.args.get("session") or None
        return jsonify({"summary": service.metrics.chat_summary(sid), "recent": service.metrics.turns()[-30:],
                        "scheduler": service.scheduler.snapshot(), "memory": service.metrics.memory.snapshot()})

    @app.post("/api/metrics/reset")
    def metrics_reset():
        service.metrics.reset()
        return jsonify({"ok": True})

    @app.post("/api/demo/start")
    def demo_start():
        d = request.get_json(silent=True) or {}
        return jsonify(demo.start(d.get("provider", ""), d.get("model", ""), d.get("api_key") or None))

    @app.post("/api/demo/step")
    def demo_step():
        d = request.get_json(silent=True) or {}
        return jsonify(demo.step(d.get("run_id", ""), d.get("api_key") or None))

    @app.post("/api/demo/cancel")
    def demo_cancel():
        demo.cancel((request.get_json(silent=True) or {}).get("run_id", ""))
        return jsonify({"ok": True})

    from pal.studio.routes import register_studio
    register_studio(app, service, **(studio or {}))         # the Studio: describe a product, review the plan, watch it being built

    @app.get("/api/memory")
    def memory_status():
        return jsonify({"enabled": cfg.memory.enabled, "notes": service.memory_count() if cfg.memory.enabled else 0,
                        "stats": service.notes.stats() if cfg.memory.enabled else None})

    @app.delete("/api/memory")
    def memory_clear():
        return jsonify({"cleared": service.clear_memory() if cfg.memory.enabled else 0})

    return app
