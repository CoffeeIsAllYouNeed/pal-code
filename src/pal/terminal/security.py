"""Pal's security layer. The checks follow the vibe-security checklist (Chris Raroque, MIT) where it applies to a coding CLI:

- keys never leave the machine: secret files are never sent to a model, key-shaped strings are hidden first (`guard_outgoing`);
- the model's output is untrusted: code it proposes is checked before anything is written (`review_code`), and text it prints
  cannot drive the terminal (`clean_terminal`);
- the project can be audited on demand (`audit_folder`): secrets in files, `.env` outside `.gitignore`, and the usual
  mistakes AI-written code makes (client-side keys, eval, shell and SQL built from strings, unsafe HTML, trusting a price
  or role sent by the browser, open CORS, debug left on, TLS checks off).

Everything is pattern matching: it finds the common cases quickly and cannot prove code is safe. Findings say so.
"""
from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SEVERITIES = ("critical", "high", "medium", "low")

# -- secrets ---------------------------------------------------------------------------------------------------------------
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")),
    ("Stripe secret key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("password in a connection string", re.compile(r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s:/@'\"]+:[^\s@/'\"]{3,}@[^\s'\"]+", re.I)),
    ("hardcoded secret", re.compile(r"(?i)\b[\w.\-]*(?:secret|password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|service[_-]?role)[\w\-]*"
                                    r"\s*[:=]\s*[\"'](?![\"'])(?!\s*(?:your|xxx|changeme|example|placeholder|<|\$\{|process\.env|os\.environ))[^\"'\s]{8,}[\"']")),
]
SECRET_FILE_GLOBS = (".env", ".env.*", "*.env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks",
                     "credentials.json", "service-account*.json", "*serviceaccount*.json", ".npmrc", ".pypirc", ".netrc", "secrets.*", ".git-credentials")
PLACEHOLDER_FILES = (".env.example", ".env.sample", ".env.template")


def is_secret_file(path: str) -> bool:
    """A file that holds keys by its very name: never sent to a model, never written by one."""
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    if name in PLACEHOLDER_FILES:
        return False
    return any(fnmatch.fnmatch(name, g) for g in SECRET_FILE_GLOBS)


def find_secrets(text: str) -> list[tuple[int, str]]:
    """[(line number, what it looks like)] for every key-shaped string."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for kind, rx in SECRET_PATTERNS:
            if rx.search(line):
                out.append((i, kind))
                break
    return out


def redact_secrets(text: str) -> tuple[str, int]:
    """The text with every key-shaped string replaced by [hidden secret], and how many were hidden."""
    count = 0

    def hide(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "[hidden secret]"
    for kind, rx in SECRET_PATTERNS:
        if kind == "private key":
            text, n = re.subn(r"-----BEGIN [A-Z ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END [A-Z ]*PRIVATE KEY[A-Z ]*-----|\Z)", "[hidden private key]", text, flags=re.S)
            count += n
        else:
            text = rx.sub(hide, text)
    return text, count


def guard_outgoing(name: str, text: str) -> tuple[str | None, str | None]:
    """(the text to send, or None to send nothing; a warning for the person, or None)."""
    if is_secret_file(name):
        return None, f"@{name}: not sent: it looks like a secrets file (keys stay on this computer)"
    clean, n = redact_secrets(text)
    return clean, (f"@{name}: {n} secret{'s' if n != 1 else ''} hidden before sending" if n else None)


# -- terminal safety -------------------------------------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[PX^_][^\x1b]*\x1b\\|\x1b.")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


def clean_terminal(text: str) -> str:
    """Text a model wrote, made safe to print: no escape sequences (colours, cursor moves, window titles, links, clipboard writes)
    and no control characters, so a reply or a file cannot rewrite what the screen shows or act on the terminal."""
    return _CONTROL.sub("", _ANSI.sub("", text.replace("\r\n", "\n")))


# -- reviewing code ----------------------------------------------------------------------------------------------------------
@dataclass
class Finding:
    severity: str
    path: str
    line: int
    title: str
    why: str


_CLIENT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".vue", ".svelte", ".html", ".htm")
_PY = (".py",)
RULES: list[tuple[str, str, re.Pattern, tuple[str, ...], str]] = [
    # severity, title, pattern, file kinds ('' = any), what an attacker could do
    ("critical", "Secret key in a client-side variable", re.compile(r"\b(?:NEXT_PUBLIC|VITE|EXPO_PUBLIC|REACT_APP)_\w*(?:SECRET|PRIVATE|SERVICE|PASSWORD|SK_|TOKEN)\w*"), (),
     "these prefixes are copied into the browser or app bundle, so anyone can read the key and use it at your expense"),
    ("critical", "Supabase service_role key in front-end code", re.compile(r"service_role", re.I), _CLIENT,
     "the service_role key bypasses all row-level security: anyone who opens the page can read or delete every row"),
    ("high", "AI or payment key used directly in front-end code", re.compile(r"(?:api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|api\.stripe\.com)"), _CLIENT,
     "calling the provider from the browser exposes the key; call it from your server instead"),
    ("high", "eval of dynamic code", re.compile(r"(?<![\w.])(?:eval|exec)\s*\((?!\s*[\"')])|new\s+Function\s*\("), _PY + _CLIENT,
     "if any part of the string comes from a user or a model, they can run their own code"),
    ("high", "Shell command built from a string", re.compile(r"shell\s*=\s*True|os\.system\s*\(|os\.popen\s*\(|child_process[\s\S]{0,40}\bexec(?:Sync)?\s*\(\s*[`'\"].*?(?:\$\{|\"\s*\+|'\s*\+)"), (),
     "a value with ; or && in it runs extra commands on your machine or server"),
    ("high", "SQL built from strings", re.compile(r"(?i)(?:execute|query|raw|exec)\s*\(\s*(?:f[\"']|[\"'][^\"']*(?:select|insert|update|delete)\b[^\"']*[\"']\s*(?:%|\+|\.format))"
                                                  r"|`\s*(?:select|insert|update|delete)\b[^`]*\$\{"), (),
     "SQL injection: a crafted value can read, change or delete any data; use parameters (?, %s, $1) instead"),
    ("high", "HTML inserted from a variable", re.compile(r"\.innerHTML\s*[+]?=\s*(?![\"'`]\s*[;<]?[\"'`]?\s*;)|dangerouslySetInnerHTML|document\.write\s*\(|v-html\s*="), _CLIENT,
     "cross-site scripting: if the text comes from a user or a model, it can run scripts in your users' browsers"),
    ("high", "Price, amount or role taken from the request", re.compile(r"(?i)(?:req\.body|request\.(?:json|form|get_json\(\)|data)|body)\s*(?:\.|\[\s*[\"'])(?:price|amount|total|unit_amount|role|is_?admin|plan|credits)\b"), (),
     "never trust the client: anyone can send price=0.01 or role=admin; look the value up on the server"),
    ("high", "TLS certificate checks turned off", re.compile(r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0|InsecureSkipVerify\s*:\s*true"), (),
     "anyone on the network can read or change the traffic"),
    ("high", "Unsafe deserialisation", re.compile(r"pickle\.loads?\s*\(|yaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.(?:Safe|C?Safe)Loader)|marshal\.loads\s*\("), _PY,
     "loading untrusted data this way runs code hidden in it"),
    ("medium", "JWT read without checking its signature", re.compile(r"verify_signature[\"']?\s*:\s*False|jwt\.decode\([^)]*verify\s*=\s*False|algorithms\s*=\s*\[\s*[\"']none[\"']"), (),
     "anyone can forge a token and log in as anyone"),
    ("medium", "CORS open to every site", re.compile(r"Access-Control-Allow-Origin[\"']?\s*[,:]\s*[\"']\*|origin\s*:\s*[\"']\*[\"']|allow_origins\s*=\s*\[\s*[\"']\*[\"']\s*\]|CORS\([^)]*origins\s*=\s*[\"']\*"), (),
     "any website can call your API from a visitor's browser; list the origins you trust"),
    ("medium", "Debug mode left on", re.compile(r"\.run\([^)]*debug\s*=\s*True|^\s*DEBUG\s*=\s*True\b", re.M), _PY,
     "debug pages show code and settings, and Flask's debugger can run code; switch it off outside development"),
    ("low", "Random numbers that are not secure", re.compile(r"Math\.random\(\)|random\.(?:random|randint|choice)\("), (),
     "fine for games; for tokens, codes or passwords use secrets (Python) or crypto.getRandomValues (JavaScript)"),
]
_COMMENT = re.compile(r"^\s*(?:#|//|\*|/\*|<!--)")
_LOW_ONLY_IF = re.compile(r"(?i)token|password|secret|otp|code|session|salt|nonce")


def review_code(path: str, text: str) -> list[Finding]:
    """Findings for one file's text: secrets first, then the code rules that apply to this kind of file."""
    out: list[Finding] = []
    suffix = PurePosixPath(path).suffix.lower()
    is_placeholder = PurePosixPath(path).name.lower() in PLACEHOLDER_FILES
    for at, kind in find_secrets(text):
        out.append(Finding("high" if is_placeholder else "critical", path, at, f"Hardcoded {kind}",
                           "keep real keys out of files: read them from the environment, and rotate any key that was ever committed"
                           + (" (an example file must hold placeholders only)" if is_placeholder else "")))
    lines = text.splitlines()
    for sev, title, rx, kinds, why in RULES:
        if kinds and suffix not in kinds:
            continue
        for i, line in enumerate(lines, 1):
            if _COMMENT.match(line):                    # a comment that mentions eval() is not a call
                continue
            if rx.search(line):
                if sev == "low" and not _LOW_ONLY_IF.search(line):
                    continue
                out.append(Finding(sev, path, i, title, why))
                break                                   # one finding per rule per file is enough to act on
    return out


def worst(findings: list[Finding]) -> str:
    return min((f.severity for f in findings), key=SEVERITIES.index, default="")


# -- auditing a folder -------------------------------------------------------------------------------------------------------
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".idea", ".vscode", "coverage"})
TEXT_SUFFIXES = frozenset({".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte", ".html", ".htm", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
                           ".env", ".sh", ".rb", ".php", ".go", ".rs", ".java", ".kt", ".cs", ".sql", ".md", ".txt", ".example", ".sample"})


def _ignored_by_gitignore(root: Path, name: str) -> bool:
    gi = root / ".gitignore"
    if not gi.is_file():
        return False
    for raw in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        pat = raw.strip().lstrip("/")
        if pat and not pat.startswith(("#", "!")) and fnmatch.fnmatch(name, pat.rstrip("/")):
            return True
    return False


def _tracked_by_git(root: Path, rel: str) -> bool:
    git = shutil.which("git")
    if not git or not (root / ".git").exists():
        return False
    try:
        return subprocess.run([git, "-C", str(root), "ls-files", "--error-unmatch", rel], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def audit_folder(root: Path, max_files: int = 400, max_bytes: int = 300_000) -> tuple[list[Finding], int]:
    """(findings, files checked) for the folder. Reads files only; runs nothing except an optional `git ls-files` to see if a secrets file is committed."""
    root = Path(root).resolve()
    findings: list[Finding] = []
    checked = 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or not p.is_file() or p.is_symlink():
            continue
        name, rel_s = p.name, rel.as_posix()
        if is_secret_file(name):
            if _tracked_by_git(root, rel_s):
                findings.append(Finding("critical", rel_s, 0, "Secrets file committed to git", "everyone with the repository has these keys; rotate them, then remove the file from git history"))
            elif not _ignored_by_gitignore(root, name):
                findings.append(Finding("high", rel_s, 0, "Secrets file not in .gitignore", f"add {name} to .gitignore before your next commit, or the keys go to the repository"))
            continue                                     # its content is not read: nothing in it is shown or sent
        if p.suffix.lower() not in TEXT_SUFFIXES and name not in PLACEHOLDER_FILES:
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(review_code(rel_s, text))
        checked += 1
        if checked >= max_files:
            break
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.path, f.line))
    return findings, checked


AUDIT_PROMPT = """Review this project for security problems, the way a careful reviewer of AI-written code would. Report only real, exploitable issues, worst first
(Critical, High, Medium, Low). For each: the file and line, the problem, what an attacker could actually do, and the fix in a few lines.
Check: secrets in code or in client-side variables; database access without per-user rules (Supabase RLS, Firebase rules); routes and actions
that do not check who is logged in; prices, roles or plans trusted from the client; payment webhooks without signature checks; missing rate
limits on login and AI endpoints; AI keys in front-end code and no usage caps; model output rendered as HTML or run as code; SQL or shell
built from strings; open CORS; debug or source maps left on in production.
The automatic scanner already found the items below; confirm or dismiss them, then add what it cannot see. Do not write files."""
