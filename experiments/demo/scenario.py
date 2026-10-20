"""The workload: a long coding-assistant session over a real project, built on purpose to show the largest savings Pal can make.

The project is LocalRAG (experiments/projects/localrag): a retrieval-augmented generation engine that is also an MCP server, about 3,400
lines of Python with its own documentation and test suite. The session attaches its documentation in turn 1 and its source part-way, and
asks the kind of questions a developer asks an assistant (which function does X, what does it take, what is this default, who calls it),
mixed with easy questions (arithmetic, the time, small facts) and follow-ups, every message written with greetings and thanks.

Every question and its expected answer are derived from the real files with Python's `ast` module and the documentation tables, so the
answers are checkable. A plain chat client resends the documentation, the source and the whole conversation every turn, so this is the
case where the gap is widest. That is the point of the experiment, and it is not a typical workload: the report says so.
"""
from __future__ import annotations

import ast
import io
import random
import re
import zipfile
from collections import Counter
from functools import lru_cache, cache
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / "localrag"
SRC = PROJECT_ROOT / "src" / "localrag"
DOC_FILES = ["README.md", "docs/ARCHITECTURE.md", "docs/RETRIEVAL.md", "docs/MCP.md", "docs/CONFIG.md"]

# which parts of the project a session of each size attaches, and how long it is
SIZES = {
    "small": dict(packages=("core", "embed", "tree"), top=False, turns=16),
    "big": dict(packages=("core", "embed", "tree", "index", "retrieve", "ingest"), top=True, turns=36),
    "max": dict(packages=("core", "embed", "tree", "index", "retrieve", "ingest", "mcp", "eval", "cli"), top=True, turns=48),
}

GREETINGS = ["Hey there! Hope you're having a great day. ", "Hi, thanks so much for your help earlier. I was just wondering, ",
             "Hello! Quick question for you, if you don't mind: ", "Hi again! Sorry to bother you, but ", "Good morning! I hope all is well. "]
SIGNOFFS = [" Thanks a lot in advance, I really appreciate your help!", " Please let me know if you need any more details from me. Thank you!",
            " Thanks so much, you are the best!", " I really appreciate it, thank you so much for your time!"]
EASY = [("what is the capital of France?", "Paris"), ("who wrote Romeo and Juliet?", "Shakespeare"),
        ("what colour do you get by mixing blue and yellow?", "green"), ("how many days are in a leap year?", "366"),
        ("what is the chemical symbol for gold?", "Au"), ("how many sides does a hexagon have?", "six"),
        ("what is the largest planet in the solar system?", "Jupiter"), ("what is the freezing point of water in Celsius?", "0"),
        ("which ocean is the largest?", "Pacific"), ("how many continents are there?", "seven")]
EXPLAINERS = ["That is how the project's own code and documentation describe it.", "It is the same in every part of the project that uses it.",
              "If you want to change it, the place to look is where it is defined.", "Nothing else in the project overrides it by default."]


@dataclass(frozen=True)
class Turn:
    i: int
    kind: str                      # math | time | chat | doc | code | followup
    text: str                      # exactly what the user types
    expect: tuple = ()             # strings a correct answer contains (all of them)
    expect_regex: str = ""         # or a pattern it matches
    answer: str = ""               # the scripted answer the SIMULATED models give (never used with a real model)
    short_answer: str = ""         # the scripted answer of the SIMULATED helper for an easy question
    route: str = "api"             # what the SIMULATED router says
    attach: str = ""               # "document" or "project": attached on this turn


@dataclass
class Scenario:
    size: str
    seed: int
    document_name: str
    document: str
    project_name: str
    project_files: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    include_code: bool = True

    def project_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, code in self.project_files.items():
                z.writestr(f"{self.project_name}/{name}", code)
        return buf.getvalue()

    def project_text(self) -> str:
        """The whole source as one block: what a plain chat client has to paste to talk about the project."""
        return "\n\n".join(f"# {name}\n{code}" for name, code in self.project_files.items())

    def project_lines(self) -> int:
        return sum(code.count("\n") + 1 for code in self.project_files.values())

    def kinds(self) -> dict:
        out: dict = {}
        for t in self.turns:
            out[t.kind] = out.get(t.kind, 0) + 1
        return out


# -- reading the real project -----------------------------------------------------------------------------------------------------------
def load_docs() -> str:
    """README and the guides, one text, each file marked with a comment line."""
    return "\n\n".join(f"<!-- file: {name} -->\n{(PROJECT_ROOT / name).read_text(encoding='utf-8').strip()}" for name in DOC_FILES)


def load_source(packages: tuple, top: bool) -> dict:
    """The source files of the chosen packages (and, when `top`, the files directly in the package), keyed by path inside localrag/."""
    files = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        parts = rel.parts
        in_slice = parts[0] in packages if len(parts) > 1 else top
        if in_slice and path.stat().st_size > 0:
            files[rel.as_posix()] = path.read_text(encoding="utf-8")
    return files


@dataclass(frozen=True)
class Func:
    name: str
    module: str
    doc: str
    params: tuple


def _first_sentence(doc: str) -> str:
    one = " ".join(doc.split())
    m = re.match(r"(.+?[.!?])(\s|$)", one)
    return (m.group(1) if m else one).rstrip(".")


@cache
def _tree(code: str) -> ast.Module:
    return ast.parse(code)


@cache
def _called_names(code: str) -> frozenset:
    """Every function or method name called anywhere in the code."""
    names = set()
    for node in ast.walk(_tree(code)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return frozenset(names)


def functions(files: dict) -> list:
    """Module-level functions with a real docstring and a name that is defined only once in the whole slice (so the question is unambiguous)."""
    counts = Counter(n.name for code in files.values() for n in ast.walk(_tree(code)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    docs, out = Counter(), []
    candidates = []
    for module, code in files.items():
        for node in _tree(code).body:
            if isinstance(node, ast.FunctionDef) and counts[node.name] == 1 and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                sentence = _first_sentence(doc)
                if len(sentence.split()) >= 5 and len(sentence) <= 120:
                    params = tuple(a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls"))
                    candidates.append(Func(node.name, module, sentence, params))
                    docs[sentence] += 1
    out = [f for f in candidates if docs[f.doc] == 1]
    return sorted(out, key=lambda f: (f.module, f.name))


def callers(files: dict, func: Func) -> list:
    """The file names (not the defining one) that call the function; empty unless there are one to three, each with a name unique in the slice."""
    basenames = Counter(PurePosixPath(m).name for m in files)
    found = []
    for module, code in files.items():
        if module == func.module:
            continue
        if func.name in _called_names(code):
            found.append(PurePosixPath(module).name)
    return sorted(found) if 1 <= len(found) <= 3 and all(basenames[b] == 1 for b in found) else []


def setting_defaults(files: dict) -> list:
    """(setting, default) for every simple number or string default in the settings classes, when the slice has them."""
    code = files.get("core/config.py")
    if not code:
        return []
    out = []
    for node in ast.walk(_tree(code)):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, (int, float, str)) and not isinstance(stmt.value.value, bool) and stmt.value.value != ""):
                    out.append((stmt.target.id, str(stmt.value.value)))
    return out


def doc_facts() -> dict:
    """Facts read from the documentation tables: setting defaults, error codes and tool arguments."""
    config = (PROJECT_ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")
    mcp = (PROJECT_ROOT / "docs" / "MCP.md").read_text(encoding="utf-8")
    defaults = [(m.group(1), m.group(2).strip("'\"")) for m in re.finditer(r"^\| `(\w+)` \| `([^`]+)` \| .+ \|$", config, re.M)
                if m.group(2).strip("'\"") not in ("", "()", "None")]
    errors = [(m.group(1), m.group(2).split(":")[0].strip()) for m in re.finditer(r"^\| (-\d+) \| (.+) \|$", mcp, re.M)]
    tools = []
    for m in re.finditer(r"^\| `(\w+)` \| .+? \| (.+) \|$", mcp, re.M):
        names = re.findall(r"`(\w+)`", m.group(2))
        if names:
            tools.append((m.group(1), names))
    return {"defaults": defaults, "errors": errors, "tools": tools}


# -- building the session ---------------------------------------------------------------------------------------------------------------
def _wrap(rnd: random.Random, question: str) -> str:
    return rnd.choice(GREETINGS) + question + rnd.choice(SIGNOFFS)


def _say(sentence: str, rnd: random.Random) -> str:
    return sentence + " " + " ".join(rnd.sample(EXPLAINERS, 3))


def _pool(items: list, rnd: random.Random) -> list:
    items = list(items)
    rnd.shuffle(items)
    return items


class _Cycle:
    """Hands out the items of a shuffled list in turn, starting again at the beginning when it runs out."""

    def __init__(self, items: list):
        self.items, self.n = items, 0

    def next(self):
        item = self.items[self.n % len(self.items)]
        self.n += 1
        return item

    def __bool__(self) -> bool:
        return bool(self.items)


def build(size: str = "big", seed: int = 7, include_code: bool = True) -> Scenario:
    if size not in SIZES:
        raise ValueError(f"size must be one of {', '.join(SIZES)}")
    cfg, rnd = SIZES[size], random.Random(seed)
    files = load_source(cfg["packages"], cfg["top"])
    facts = doc_facts()
    sc = Scenario(size, seed, "localrag_docs.md", load_docs(), "localrag", files if include_code else {}, [], include_code)

    funcs = functions(files)
    where = _Cycle(_pool([f for f in funcs], rnd))
    params = _Cycle(_pool([f for f in funcs if f.params], rnd))
    used = _Cycle(_pool([(f, callers(files, f)) for f in funcs if callers(files, f)], rnd))
    defaults = _Cycle(_pool(setting_defaults(files), rnd))
    doc_defaults, doc_errors, doc_tools = (_Cycle(_pool(facts[k], rnd)) for k in ("defaults", "errors", "tools"))

    def code_turn(i: int, attach: str, n: int) -> Turn:
        order = [where, defaults, used, params]
        for step in range(len(order)):
            pick = order[(n + step) % len(order)]
            if not pick:
                continue
            if pick is where:
                f = where.next()
                return Turn(i, "code", _wrap(rnd, f'In the project, which function is documented as "{f.doc}"?'), (f.name,), attach=attach,
                            answer=_say(f"That is the function {f.name}, defined in {f.module}.", rnd))
            if pick is defaults:
                name, value = defaults.next()
                return Turn(i, "code", _wrap(rnd, f"In the project's settings, what is the default value of `{name}`?"), (value,), attach=attach,
                            answer=_say(f"The default value of {name} is {value}.", rnd))
            if pick is used:
                f, who = used.next()
                return Turn(i, "code", _wrap(rnd, f"In the project, which files call the function `{f.name}`?"), tuple(who), attach=attach,
                            answer=_say(f"The function {f.name} is called from {' and '.join(who)}.", rnd))
            f = params.next()
            return Turn(i, "code", _wrap(rnd, f"In the project, what parameters does the function `{f.name}` take?"), f.params, attach=attach,
                        answer=_say(f"The function {f.name} takes {', '.join(f.params)}.", rnd))
        raise ValueError("the chosen part of the project has nothing to ask about")

    def doc_turn(i: int, attach: str, n: int) -> Turn:
        kind = n % 3
        if kind == 0 and doc_defaults:
            name, value = doc_defaults.next()
            return Turn(i, "doc", _wrap(rnd, f"According to the configuration reference, what is the default value of `{name}`?"), (value,), attach=attach,
                        answer=_say(f"The configuration reference gives {value} as the default of {name}.", rnd))
        if kind == 1 and doc_errors:
            code, meaning = doc_errors.next()
            return Turn(i, "doc", _wrap(rnd, f'According to the MCP guide, which error code means "{meaning}"?'), (code,), attach=attach,
                        answer=_say(f"The MCP guide lists {code} for {meaning}.", rnd))
        tool, names = doc_tools.next()
        return Turn(i, "doc", _wrap(rnd, f"According to the MCP guide, which arguments does the `{tool}` tool take?"), tuple(names), attach=attach,
                    answer=_say(f"The {tool} tool takes {', '.join(names)}.", rnd))

    # a repeating pattern of kinds; most turns are about the attached documentation and source, or follow them
    pattern = ["doc", "math", "chat", "doc", "time", "code", "followup", "doc", "math", "chat", "code", "doc", "chat", "followup", "time", "code"]
    code_from = max(3, cfg["turns"] // 4)
    d_n = c_n = e_n = 0
    for i in range(1, cfg["turns"] + 1):
        kind = pattern[(i - 1) % len(pattern)]
        if kind == "code" and (not include_code or i < code_from):
            kind = "doc"
        if kind == "followup" and i == 1:
            kind = "doc"
        attach = "document" if i == 1 else ("project" if (include_code and i == code_from) else "")
        if include_code and i == code_from:
            kind = "code"
        if kind == "doc":
            sc.turns.append(doc_turn(i, attach, d_n))
            d_n += 1
        elif kind == "code":
            sc.turns.append(code_turn(i, attach, c_n))
            c_n += 1
        elif kind == "followup":
            name, value = doc_defaults.next()
            sc.turns.append(Turn(i, "followup", _wrap(rnd, f"And what about `{name}`? What is the default value of that one?"), (value,),
                                 answer=_say(f"The default value of {name} is {value}.", rnd)))
        elif kind == "math":
            a, b = 100 + (i * 37) % 900, 100 + (i * 53) % 900
            sc.turns.append(Turn(i, "math", f"what is {a} * {b}?", (str(a * b),), answer=f"The result is {a * b}.", route="local"))
        elif kind == "time":
            sc.turns.append(Turn(i, "time", "what time is it?", expect_regex=r"\b\d{1,2}:\d{2}\b", answer="It is 14:32.", route="local"))
        else:
            q, a = EASY[e_n % len(EASY)]
            e_n += 1
            sc.turns.append(Turn(i, "chat", _wrap(rnd, q), (a,), answer=_say(f"The answer is {a}.", rnd), short_answer=f"{a}.", route="local"))
    return sc
