"""The converter as a tool: a file already in a workspace becomes markdown, saved under converted/, never overwriting."""
import io
import sys

import pytest

from pal.input_converters.components.base import ConversionError
from pal.input_converters.workspace import OUT_DIR, convert_workspace_file, convertible, describe

sys.path.insert(0, ".")


def xlsx_bytes():
    openpyxl = pytest.importorskip("openpyxl")                                           # an optional library: skip without it
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["item", "cost"])
    ws.append(["hosting", 500])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def store():
    files: dict = {}
    return files, dict(read_bytes=lambda p: files[p], write_text=lambda p, t: files.__setitem__(p, t.encode()), exists=lambda p: p in files)


def test_a_workspace_xlsx_becomes_markdown_saved_next_to_a_converted_folder():
    files, io_ = store()
    files["uploads/budget.xlsx"] = xlsx_bytes()
    r = convert_workspace_file("uploads/budget.xlsx", **io_)
    assert r.saved == f"{OUT_DIR}/budget.xlsx.md" and r.converted is True and "hosting" in r.text and "500" in r.text
    assert files["converted/budget.xlsx.md"].decode() == r.text and r.source_format == "xlsx"


def test_a_converted_file_is_never_overwritten_and_two_formats_with_one_stem_do_not_collide():
    files, io_ = store()
    files["a/report.xlsx"] = xlsx_bytes()
    first = convert_workspace_file("a/report.xlsx", **io_)
    files["converted/report.xlsx.md"] = b"EDITED BY A PERSON"
    again = convert_workspace_file("a/report.xlsx", **io_)
    assert first.saved and again.saved == "" and "already exists and was not overwritten" in again.note
    assert files["converted/report.xlsx.md"] == b"EDITED BY A PERSON"
    files["a/report.html"] = b"<h1>Report</h1><p>text</p>"
    assert convert_workspace_file("a/report.html", **io_).saved == "converted/report.html.md"


@pytest.mark.parametrize("path", ["notes.txt", "main.py", "data.csv", "image.png", "archive.zip", "README", "a/b/c.json"])
def test_only_convertible_formats_are_accepted_and_the_error_lists_them(path):
    files, io_ = store()
    files[path] = b"x"
    with pytest.raises(ConversionError, match="only .*docx.*pdf.*can be converted"):
        convert_workspace_file(path, **io_)
    assert not convertible(path)


def test_too_large_a_result_is_refused_and_nothing_is_written():
    files, io_ = store()
    files["big.html"] = ("<p>" + "word " * 5000 + "</p>").encode()
    with pytest.raises(ConversionError, match="over the 1,000 limit"):
        convert_workspace_file("big.html", max_chars=1000, **io_)
    assert "converted/big.html.md" not in files


def test_a_broken_file_is_a_clear_conversion_error():
    files, io_ = store()
    files["broken.pdf"] = b"not really a pdf"
    with pytest.raises(ConversionError):
        convert_workspace_file("broken.pdf", **io_)


def test_the_description_says_what_happened_and_shows_only_a_preview():
    files, io_ = store()
    files["big.html"] = ("<h1>Title</h1>" + "<p>" + "word " * 2000 + "</p>").encode()
    r = convert_workspace_file("big.html", **io_)
    out = describe(r, preview_chars=100, indexed="Added to your documents.")
    assert out.startswith("converted big.html (html to markdown):") and "Saved the full text as converted/big.html.md." in out
    assert "Added to your documents." in out and out.endswith("[preview ends; read the rest from the saved file]")
    assert len(out) < 400
    again = convert_workspace_file("big.html", **io_)
    assert "already exists" in describe(again)
