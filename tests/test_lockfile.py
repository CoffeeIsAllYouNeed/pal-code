"""The hash-locked install must work on Windows, where tqdm and click need colorama. The lock is generated on Linux, so it has to carry that by hand."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = (ROOT / "requirements" / "litellm.lock").read_text(encoding="utf-8")


def entries():
    """name -> (version, marker, hashes) for every pinned requirement in the lock."""
    out = {}
    for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)(?:\s*;\s*([^\\\n]+?))?\s*\\\n((?:    --hash=sha256:[0-9a-f]{64}(?: \\)?\n)+)", LOCK, re.M):
        out[m.group(1).lower()] = (m.group(2), (m.group(3) or "").strip(), re.findall(r"sha256:([0-9a-f]{64})", m.group(4)))
    return out


def test_every_pinned_requirement_has_at_least_one_hash():
    names = re.findall(r"^([A-Za-z0-9_.\-]+)==", LOCK, re.M)
    assert len(names) >= 55 and sorted(n.lower() for n in names) == sorted(entries()), "a pinned requirement has no hash (or an unparsable block)"


def test_colorama_is_pinned_with_hashes_for_windows_only():
    version, marker, hashes = entries()["colorama"]
    assert version == "0.4.6" and marker == 'sys_platform == "win32"' and len(hashes) == 2
    assert "4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6" in hashes        # the wheel, as PyPI publishes it


def test_the_platform_check_script_is_shipped_for_when_the_lock_is_regenerated():
    text = (ROOT / "scripts" / "check_lock_platforms.py").read_text(encoding="utf-8")
    assert "Windows" in text and "required but not pinned" in text
