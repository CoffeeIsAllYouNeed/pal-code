"""Talk instead of typing: record from the microphone, turn it into text, and put the text in the input box to edit and send.

In the terminal: tap the mic in the box (or /voice). In the web chat: the microphone button. The words are never sent on their own: they land in the box.

Turning speech into text, in this order:
  1. on this computer with faster-whisper (installed with Pal): nothing leaves the computer;
  2. else OpenAI's transcription (OPENAI_API_KEY) or Groq's Whisper (GROQ_API_KEY), with your key;
  3. else Pal says what to install.
Recording in the terminal needs sounddevice (in the same extra). Dictation apps such as Wispr Flow need none of this: they type into the
box like a keyboard, in the terminal and in the browser.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

RATE = 16_000
MAX_SECONDS = 120
LOCAL_MODEL = os.environ.get("PAL_WHISPER_MODEL", "base")


class VoiceUnavailable(RuntimeError):
    """Recording or transcription cannot work here; the message says what to install or set."""


@dataclass
class Transcript:
    text: str
    engine: str          # "local whisper (base)", "openai", "groq"


def can_record() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:  # noqa: BLE001  (not installed, or PortAudio missing)
        return False


def engines() -> list[str]:
    """The transcription engines available now, best first."""
    out = []
    try:
        import faster_whisper  # noqa: F401
        out.append("local")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    if os.environ.get("GROQ_API_KEY"):
        out.append("groq")
    return out


def wav_bytes(samples, rate: int = RATE) -> bytes:
    """16-bit mono WAV from float samples in [-1, 1] (a numpy array or a list)."""
    try:
        import numpy as np
        pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    except Exception:  # noqa: BLE001  (no numpy: the slow way)
        import array
        pcm = array.array("h", (max(-32768, min(32767, int(s * 32767))) for s in samples)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def record(stop: threading.Event, max_seconds: int = MAX_SECONDS, on_level: Callable[[float], None] | None = None) -> bytes:
    """Record until `stop` is set (or max_seconds), as WAV bytes. `on_level` gets how loud each slice is (0 to 1), for the voice graph."""
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001
        raise VoiceUnavailable("the microphone could not be opened (on Linux: sudo apt install libportaudio2)") from e
    chunks: list = []

    def take(data, frames, time_info, status) -> None:
        chunks.append(data.copy())
        if on_level is not None:
            try:
                on_level(min(1.0, float(np.sqrt(np.mean(np.square(data)))) * 8.0))     # speech is quiet in float samples: scaled up
            except Exception:  # noqa: BLE001
                pass
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=take):
            stop.wait(max_seconds)
    except Exception as e:  # noqa: BLE001  (no microphone, or permission denied)
        raise VoiceUnavailable(f"could not use the microphone: {e}") from e
    audio = np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype="float32")
    return wav_bytes(audio)


_LOCAL: dict = {}
_LOAD = threading.Lock()


def local_model():
    """The Whisper model on this computer, loaded once (and downloaded the first time). int8 on the CPU: quick to load, quick to run,
    and it works on every computer (a GPU build of the runtime is not needed)."""
    with _LOAD:
        if LOCAL_MODEL not in _LOCAL:
            from faster_whisper import WhisperModel
            _LOCAL[LOCAL_MODEL] = WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")
        return _LOCAL[LOCAL_MODEL]


def preload() -> None:
    """Load the speech model in the background when Pal starts, so the first time the mic is used it answers at once."""
    if "local" not in engines():
        return
    try:
        local_model()
    except Exception:  # noqa: BLE001  (offline the first time, no disk ...: it is loaded when the mic is used)
        pass


def _samples(audio: bytes, suffix: str):
    """WAV bytes as float samples, without a temporary file; None for other formats (the browser's webm and ogg)."""
    if suffix.lower() != ".wav":
        return None
    try:
        import numpy as np
        with wave.open(io.BytesIO(audio)) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != RATE:
                return None
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
    except Exception:  # noqa: BLE001
        return None


def _local(audio: bytes, suffix: str) -> str:
    model = local_model()
    options = {"beam_size": 1, "vad_filter": True, "condition_on_previous_text": False}       # one pass: much faster, as good for dictation
    samples = _samples(audio, suffix)
    if samples is not None:
        segments, _ = model.transcribe(samples, **options)
        return " ".join(s.text.strip() for s in segments).strip()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        name = f.name
    try:
        segments, _ = model.transcribe(name, **options)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        Path(name).unlink(missing_ok=True)


def _api(audio: bytes, suffix: str, engine: str) -> str:
    from pal.llm_wrapper.components.litellm_wrapper import _import_litellm
    ll = _import_litellm()
    model = {"openai": "whisper-1", "groq": "groq/whisper-large-v3"}[engine]
    key = os.environ.get("OPENAI_API_KEY" if engine == "openai" else "GROQ_API_KEY")
    f = io.BytesIO(audio)
    f.name = "speech" + suffix
    r = ll.transcription(model=model, file=f, api_key=key)
    return str(getattr(r, "text", "") or (r.get("text", "") if isinstance(r, dict) else "")).strip()


def transcribe(audio: bytes, suffix: str = ".wav", prefer: list[str] | None = None,
               runners: dict[str, Callable[[bytes, str], str]] | None = None) -> Transcript:
    """Speech to text with the best engine available. Raises VoiceUnavailable when there is none (or every one failed)."""
    order = prefer or engines()
    if not order:
        raise VoiceUnavailable('no speech-to-text here: reinstall Pal (pip install -U pal-code), or set OPENAI_API_KEY or GROQ_API_KEY')
    runners = runners or {"local": _local, "openai": lambda a, s: _api(a, s, "openai"), "groq": lambda a, s: _api(a, s, "groq")}
    problems = []
    for engine in order:
        try:
            text = runners[engine](audio, suffix)
        except Exception as e:  # noqa: BLE001  (try the next engine)
            problems.append(f"{engine}: {str(e)[:120]}")
            continue
        return Transcript(text, f"local whisper ({LOCAL_MODEL})" if engine == "local" else engine)
    raise VoiceUnavailable("speech-to-text failed: " + "; ".join(problems))
