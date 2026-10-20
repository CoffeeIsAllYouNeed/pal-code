"""Hugging Face media tools: describe an image, answer a question about it, make an image, speak text, transcribe audio.

They call the hf-inference route of Hugging Face Inference Providers (router.huggingface.co) with an HF_TOKEN. Uploaded images and
audio live in the session workspace; the model sees only their path, never the bytes. Made images and audio are saved to the
workspace under generated/ and can be downloaded from the page. Which models a route serves changes over time: every tool's model
can be set in tools.hf_models, and Hugging Face's own error text is passed on when a model is not available. These tools were
tested against fake servers only, never against Hugging Face.
"""
from __future__ import annotations

import hashlib
import re

from pal.chat.services.base import ServiceCtx, ServiceSpec, clean_text, excerpt, explain, param
from pal.errors import PalError

HF = "HF_TOKEN"
_C = re.I
_BASE = "https://router.huggingface.co/hf-inference/models/"
DEFAULT_MODELS = {
    "image_caption": "Salesforce/blip-image-captioning-large",
    "visual_question_answering": "dandelin/vilt-b32-finetuned-vqa",
    "text_to_image": "black-forest-labs/FLUX.1-schnell",
    "text_to_audio": "facebook/mms-tts-eng",
    "speech_to_text": "openai/whisper-large-v3",
}


def _call(ctx: ServiceCtx, tool: str, **params) -> dict:
    model = ctx.hf_model(tool, DEFAULT_MODELS[tool])
    headers = {"Authorization": "Bearer " + ctx.key(HF), **params.pop("headers", {})}
    r = ctx.net(_BASE + model, headers=headers, **params)
    if r["status"] == 503:
        raise PalError(f"Hugging Face is loading {model}; try again in a minute" + _hf_detail(r))
    if r["status"] >= 400:
        raise PalError(explain(f"Hugging Face ({model})", r["status"], r.get("text", ""), HF))
    return r


def _hf_detail(r: dict) -> str:
    import json
    try:
        d = json.loads(r.get("text", ""))
        return f" (estimated {d['estimated_time']:.0f}s)" if isinstance(d, dict) and d.get("estimated_time") else ""
    except (ValueError, TypeError, KeyError):
        return ""


def _json(r: dict, what: str):
    import json
    try:
        return json.loads(r["text"])
    except ValueError as e:
        raise PalError(f"Hugging Face returned something that is not {what}") from e


def image_caption(ctx: ServiceCtx, path: str) -> str:
    p, ctype = ctx.media_path(path, "image")
    r = _call(ctx, "image_caption", method="POST", body_path=p, content_type=ctype)
    d = _json(r, "JSON")
    text = d[0].get("generated_text") if isinstance(d, list) and d and isinstance(d[0], dict) else (d.get("generated_text") if isinstance(d, dict) else None)
    return str(text).strip() if text else "no caption: " + excerpt(d)


def visual_question_answering(ctx: ServiceCtx, path: str, question: str) -> str:
    p, _ = ctx.media_path(path, "image")
    q = clean_text(question, "a question about the image", 300)
    b64 = ctx.os("file_read_b64", {"path": p})["b64"]
    r = _call(ctx, "visual_question_answering", method="POST", json={"inputs": {"image": b64, "question": q}})
    d = _json(r, "JSON")
    rows = [x for x in d if isinstance(x, dict) and x.get("answer")] if isinstance(d, list) else []
    return "; ".join(f"{x['answer']} ({x.get('score', 0):.0%})" for x in rows[:3]) or "no answer: " + excerpt(d)


def _saved(prefix: str, seed: str) -> str:
    return f"generated/{prefix}-{hashlib.sha256(seed.encode()).hexdigest()[:8]}.{{ext}}"


def text_to_image(ctx: ServiceCtx, prompt: str) -> str:
    text = clean_text(prompt, "a description of the image", 500)
    r = _call(ctx, "text_to_image", method="POST", json={"inputs": text}, save_as=_saved("image", text))
    if not r.get("saved"):
        raise PalError("Hugging Face did not return an image: " + (r.get("text") or r.get("content_type", ""))[:200])
    ctx.saved.append(r["saved"])
    return f"saved the image as {r['saved']} ({r['bytes']:,} bytes); the user can download it"


def text_to_audio(ctx: ServiceCtx, text: str) -> str:
    t = clean_text(text, "the text to speak", 800)
    r = _call(ctx, "text_to_audio", method="POST", json={"inputs": t}, save_as=_saved("speech", t))
    if not r.get("saved"):
        raise PalError("Hugging Face did not return audio: " + (r.get("text") or r.get("content_type", ""))[:200])
    ctx.saved.append(r["saved"])
    return f"saved the audio as {r['saved']} ({r['bytes']:,} bytes); the user can download it"


def speech_to_text(ctx: ServiceCtx, path: str) -> str:
    p, ctype = ctx.media_path(path, "audio")
    r = _call(ctx, "speech_to_text", method="POST", body_path=p, content_type=ctype)
    d = _json(r, "JSON")
    text = d.get("text") if isinstance(d, dict) else None
    return str(text).strip() if text else "no speech found: " + excerpt(d)


SERVICES = [
    ServiceSpec("image_caption", "image_caption(path)", (param("path", str),), image_caption, "describe an uploaded image in one sentence",
                re.compile(r"\bimage\b|\bpicture\b|\bphoto\b|\bpng\b|\bjpe?g\b|\bwebp\b|\bgif\b|\bcaption\b|\bdescribe\b|\bwhat(?:'s| is) in\b", _C),
                HF, needs="image", sends="the image goes to Hugging Face"),
    ServiceSpec("visual_question_answering", "visual_question_answering(path, question)", (param("path", str), param("question", str)),
                visual_question_answering, "answer a question about an uploaded image",
                re.compile(r"\bimage\b|\bpicture\b|\bphoto\b|\bpng\b|\bjpe?g\b|\bwebp\b|\bgif\b|\bin (?:the|this) (?:image|picture|photo)\b", _C),
                HF, needs="image", sends="the image and your question go to Hugging Face"),
    ServiceSpec("text_to_image", "text_to_image(prompt)", (param("prompt", str),), text_to_image, "make an image from a description",
                re.compile(r"\b(?:draw|generate|create|make|paint)\b.{0,30}\b(?:image|picture|photo|illustration|logo)\b|\bimage of\b|\bpicture of\b", _C),
                HF, sends="your description goes to Hugging Face"),
    ServiceSpec("text_to_audio", "text_to_audio(text)", (param("text", str),), text_to_audio, "turn text into spoken audio",
                re.compile(r"\bread (?:this )?aloud\b|\btext[- ]to[- ]speech\b|\bsay this\b|\bspeak\b|\bvoice\b|\baudio of\b|\bnarrat", _C),
                HF, sends="your text goes to Hugging Face"),
    ServiceSpec("speech_to_text", "speech_to_text(path)", (param("path", str),), speech_to_text,
                "transcribe an uploaded audio file (voice activity recognition in the AIOS paper)",
                re.compile(r"\baudio\b|\brecording\b|\bvoice\b|\btranscri\w+\b|\bspeech\b|\bwav\b|\bmp3\b|\bflac\b|\bogg\b|\bm4a\b", _C),
                HF, needs="audio", sends="the audio goes to Hugging Face"),
]
