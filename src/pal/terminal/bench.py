"""`pal bench`: the same task, on the same model, two ways, with the tokens the model really used.

    plain   a plain chat: the project's source files are pasted into the first message, and every later message resends the whole
            conversation (what a simple chat client does when you paste your code into it).
    pal     Pal: the folder's project map, code graph and documents, with only the passages that match each message sent, and the
            history paged and compressed.

Each way works in its own temporary copy of the folder, and applies its own changes there, so the follow-up steps build on them. Your
folder is never changed. Token counts are the model's own when it reports them (most APIs and Ollama do); the report says when some were
estimated. The quality of the two results is yours to judge: the report lists the files each way wrote.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pal.paths import pal_home
from pal.terminal.project import MAX_CODE_BYTES, Project, scan

DEFAULT_FOLLOW_UPS = ("Add a short automated test for what you just built.", "In three sentences: how does it work, and what would you improve next?")
PLAIN_PASTE_LIMIT = 400_000          # bytes of source the plain chat pastes (about 100k tokens): beyond that no chat would take it
STEP_MODES = ("build", "build", "ask")


@dataclass
class Step:
    message: str
    mode: str
    prompt: int = 0
    completion: int = 0
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@dataclass
class Arm:
    name: str
    steps: list[Step] = field(default_factory=list)
    estimated_calls: int = 0
    calls: int = 0
    error: str = ""

    @property
    def total(self) -> int:
        return sum(s.total for s in self.steps)

    @property
    def seconds(self) -> float:
        return round(sum(s.seconds for s in self.steps), 1)


@dataclass
class BenchResult:
    task: str
    model: str
    folder: str
    plain: Arm
    pal: Arm
    when: float = field(default_factory=time.time)
    pal_files: dict = field(default_factory=dict, repr=False)     # what Pal's way wrote: {path: bytes}, offered to keep in your folder

    @property
    def saved(self) -> int:
        return self.plain.total - self.pal.total

    @property
    def saved_pct(self) -> float:
        return round(100.0 * self.saved / self.plain.total, 1) if self.plain.total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("pal_files", None)
        d.update(saved=self.saved, saved_pct=self.saved_pct, plain_total=self.plain.total, pal_total=self.pal.total)
        return d


def copy_folder(src: Path, dst: Path) -> None:
    """The files Pal would see (no dependencies, build output, hidden folders or secrets), copied for one arm of the bench."""
    found = scan(src)
    for rel in list(found.code) + list(found.docs):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
    for name in ("package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts"):
        if (src / name).is_file() and not (dst / name).exists():
            shutil.copy2(src / name, dst / name)


def source_files(folder: Path, limit: int = PLAIN_PASTE_LIMIT) -> list[dict]:
    """The source files a plain chat pastes, up to `limit` bytes in all."""
    out, total = [], 0
    for rel in sorted(scan(folder).code):
        try:
            text = (folder / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        size = len(text.encode("utf-8"))
        if size > MAX_CODE_BYTES or total + size > limit:
            continue
        total += size
        out.append({"name": rel, "text": text})
    return out


def _run_arm(name: str, session, steps: list[tuple[str, str]], progress: Callable[[str], None]) -> Arm:
    arm = Arm(name)
    for i, (message, mode) in enumerate(steps, 1):
        progress(f"{'Building it as a plain chat' if name == 'plain' else 'Building it with Pal'}  \u00B7  step {i} of {len(steps)}")
        session.mode = mode
        try:
            turn = session.ask(message)
        except Exception as e:  # noqa: BLE001  (a model error ends this arm; the other still runs)
            arm.error = f"{type(e).__name__}: {str(e)[:200]}"
            break
        written = [p for p, status in session.apply(turn.changes) if status == "written"] if turn.changes else []
        arm.steps.append(Step(message, mode, turn.prompt_tokens, turn.completion_tokens, turn.seconds, written))
    counter = getattr(session.client, "main", None) or getattr(session.client, "core", None)
    arm.calls = getattr(counter, "calls", 0)
    arm.estimated_calls = getattr(counter, "estimated", 0)
    return arm


def run_bench(folder: Path, task: str, core_factory: Callable[[], Any], model: str, follow_ups: list[str] | None = None,
              progress: Callable[[str], None] | None = None, home: Path | None = None) -> BenchResult:
    """Run the task and its follow-ups both ways. `core_factory()` makes a fresh model wrapper (one per arm, so nothing is shared)."""
    from pal.terminal.clients import PalClient, PlainClient
    from pal.terminal.session import SYSTEM, PalSession
    progress = progress or (lambda _m: None)
    folder = Path(folder).resolve()
    follow = list(DEFAULT_FOLLOW_UPS if follow_ups is None else follow_ups)
    steps = [(task, "build")] + [(m, STEP_MODES[min(i + 1, len(STEP_MODES) - 1)]) for i, m in enumerate(follow)]
    with tempfile.TemporaryDirectory(prefix="pal-bench-") as tmp:
        root = Path(tmp)
        plain_dir, pal_dir = root / "plain" / folder.name, root / "pal" / folder.name
        for d in (plain_dir, pal_dir):
            d.mkdir(parents=True)
            copy_folder(folder, d)

        progress("Building it as a plain chat  \u00B7  pasting the source files")
        plain = PalSession(PlainClient(core_factory(), SYSTEM, max_tokens=4000), model, "bench", plain_dir, "build")
        plain.pending = source_files(plain_dir)                  # pasted into the first message; a plain chat then resends it every time
        plain_arm = _run_arm("plain", plain, steps, progress)

        progress("Building it with Pal  \u00B7  setting up the folder")
        project = Project(pal_dir, home=root / "pal-home")
        project.prepare(graph_in_background=False)
        client = PalClient(core_factory(), None, frozenset(), network=False, home=root / "pal-engine", model_name=model, max_tokens=4000,
                           local=False, project=project)
        pal = PalSession(client, model, "bench", pal_dir, "build")
        pal.project = project
        pal.brief = project.brief()
        try:
            pal_arm = _run_arm("pal", pal, steps, progress)
        finally:
            client.close()
        kept = {}
        for rel in sorted({f for st in pal_arm.steps for f in st.files}):
            f = pal_dir / rel
            if f.is_file():
                kept[rel] = f.read_bytes()
    result = BenchResult(task, model, str(folder), plain_arm, pal_arm, pal_files=kept)
    save(result, home)
    return result


def save(result: BenchResult, home: Path | None = None) -> Path:
    out = (home or pal_home()) / "bench" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    return out
