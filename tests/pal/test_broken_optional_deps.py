"""A broken optional library (e.g. torch failing with a Windows DLL error) must not stop Pal from starting."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pal_imports_when_optional_libraries_raise_oserror(tmp_path):
    for name in ("spacy", "torch", "sentence_transformers", "nltk", "chromadb", "proselint"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text('raise OSError("[WinError 126] fbgemm.dll")\n')
    code = textwrap.dedent("""
        import pal.chat, pal.compression
        from pal.compression.pipeline import CompressionPipeline
        from pal.context_manager.utils import relevance
        CompressionPipeline()
        assert relevance._EMBED_AVAILABLE is False
        print("ok")
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT / "src")]))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
