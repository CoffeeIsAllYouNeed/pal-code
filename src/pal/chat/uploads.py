"""Reading a file someone attached (from `pal chat --file` or an agent): what the model should see.

A source-code zip (or a big code file) becomes an indexed code graph; HTML, PDF, Word, ODT/RTF, PowerPoint, EPUB, LaTeX and Excel are converted to
markdown; images and audio go to the session's workspace for a media tool; other text files pass through.
"""
from __future__ import annotations

import logging
import os

from pal.chat import ChatService
from pal.chat.services.base import AUDIO_TYPES, IMAGE_TYPES
from pal.errors import PalError, UserInputError
from pal.input_converters import convert_to_markdown
from pal.input_converters.pipeline import CONVERTIBLE_EXTENSIONS
from pal.paths import safe_id

log = logging.getLogger(__name__)


def _clean_filename(name: str) -> str:
    return os.path.basename((name or "file").replace("\\", "/"))[:200] or "file"


MEDIA_TYPES = {**IMAGE_TYPES, **AUDIO_TYPES}
_MEDIA_TOOLS = ("image_caption", "visual_question_answering", "speech_to_text")


def _is_text(data: bytes) -> bool:
    try:
        data[:4096].decode("utf-8")
        return b"\x00" not in data[:4096]
    except UnicodeDecodeError:
        return False


def _keep_original(service: ChatService, name: str, raw: bytes, session_id: str) -> None:
    """A document or source zip is converted or indexed on upload as before. The original is also kept in the session's workspace, so
    file_to_markdown and build_code_graph can work on it. It is deleted when the chat ends. Too big or unwritable: simply not kept."""
    try:
        service.os_kernel.workspace(f"chat-{safe_id(session_id)}").write_bytes(f"uploads/{name}", raw)
    except (PalError, OSError):
        log.debug("original of %s not kept in the workspace", name)


def _save_media(service: ChatService, name: str, ext: str, raw: bytes, session_id: str) -> dict:
    """An image or audio file is not text: it goes into the session's workspace, and the model is told its path."""
    from pal.chat.services import selected
    kind = "image" if ext in IMAGE_TYPES else "audio"
    tools = [s.name for s in selected(service.settings.tools.services) if s.name in _MEDIA_TOOLS and s.needs == kind]
    if not tools:
        raise UserInputError(f"{name}: {kind} files can only be used with a media tool. Add "
                             f"{'image_caption or visual_question_answering' if kind == 'image' else 'speech_to_text'} to tools.services "
                             "and set HF_TOKEN")
    path = f"uploads/{name}"
    service.os_kernel.workspace(f"chat-{safe_id(session_id)}").write_bytes(path, raw)        # size-limited and path-checked there
    use = " or ".join(f"{t}" for t in tools)
    return {"name": name, "kind": "media", "text": f"[{kind} saved in your workspace as {path}. Use {use} with that path.]"}


def read_upload(service: ChatService, f, session_id: str) -> dict:
    """Attachment -> what the API model should see. A source-code zip (or a big code file) becomes an
    indexed code graph; HTML, PDF, Word, ODT/RTF, PowerPoint, EPUB, LaTeX and Excel are converted to
    markdown; other text files pass through."""
    name, raw = _clean_filename(f.filename), f.read()
    ext = "." + name.rsplit(".", 1)[1].lower() if "." in name else ""
    if ext in MEDIA_TYPES:
        return _save_media(service, name, ext, raw, session_id)
    if ext.lstrip(".") in CONVERTIBLE_EXTENSIONS or ext == ".zip":
        _keep_original(service, name, raw, session_id)
    code = service.scheduler.code        # conversion and graph building run in code, a few at a time, beside the LLM lines
    if service.graph.wants(name, raw):
        return code.run(lambda: service.graph.ingest(session_id, name, raw))
    res = code.run(lambda: convert_to_markdown(name, raw))
    limit = service.settings.documents.max_file_chars
    if len(res.text) > limit:
        raise UserInputError(f"{name}: over {limit:,} characters after conversion")
    out = {"name": name, "text": res.text, "kind": "document"}
    if res.converted:  # the API model is told the text is a conversion, not the original file
        out["note"] = f"converted from {res.source_format} to markdown"
    return out
