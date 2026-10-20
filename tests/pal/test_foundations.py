"""Pal's building blocks: finding the strongest local model, the path rule, and reading files out of a model's reply."""
import pytest

from pal.terminal.files import parse_files
from pal.terminal.models import LocalModel, discover, parse_tags, strongest
from pal.terminal.policy import PolicyError, safe_path

TAGS = {"models": [
    {"name": "llama3.2:1b", "size": 1_300_000_000, "details": {"parameter_size": "1.2B", "family": "llama", "quantization_level": "Q8_0"}},
    {"name": "llama3.1:8b-instruct-q4_K_M", "size": 4_900_000_000, "details": {"parameter_size": "8.0B", "family": "llama", "quantization_level": "Q4_K_M"}},
    {"name": "llama3:latest", "size": 4_700_000_000, "details": {"parameter_size": "8.0B", "family": "llama"}},
    {"name": "qwen2.5-coder:32b", "size": 19_000_000_000, "details": {"parameter_size": "32.8B"}},
    {"name": "nomic-embed-text:latest", "size": 274_000_000, "details": {"parameter_size": "137M", "family": "nomic-bert"}}, "junk", {"size": 5}]}


def test_ollamas_model_list_is_read_with_sizes_embedding_models_marked_and_junk_skipped():
    models = {m.name: m for m in parse_tags(TAGS)}
    assert set(models) == {"llama3.2:1b", "llama3.1:8b-instruct-q4_K_M", "llama3:latest", "qwen2.5-coder:32b", "nomic-embed-text:latest"}
    assert models["llama3.1:8b-instruct-q4_K_M"].params_b == 8.0 and models["llama3.1:8b-instruct-q4_K_M"].size_gb == 4.9 and models["nomic-embed-text:latest"].embedding
    assert models["nomic-embed-text:latest"].params_b == 0.137 and not models["qwen2.5-coder:32b"].embedding


def test_the_strongest_model_is_the_largest_chat_model_never_an_embedding_model():
    assert strongest(parse_tags(TAGS)).name == "qwen2.5-coder:32b"
    without_big = {"models": [m for m in TAGS["models"] if isinstance(m, dict) and "32b" not in m.get("name", "")]}
    assert strongest(parse_tags(without_big)).name in ("llama3.1:8b-instruct-q4_K_M", "llama3:latest")              # a tie on size goes to the bigger file
    assert strongest(parse_tags({"models": [TAGS["models"][4]]})) is None and strongest([]) is None


@pytest.mark.parametrize("name,expect", [("llama3.1:8b-instruct-q4_K_M", 8.0), ("qwen2.5-coder:32b", 32.0), ("llama3.2:1b", 1.0), ("phi3:3.8b", 3.8), ("llama3:latest", None), ("mistral:7b-q4", 7.0)])
def test_the_size_is_read_from_the_name_when_ollama_gives_no_details(name, expect):
    assert parse_tags({"models": [{"name": name}]})[0].params_b == expect


def test_a_model_with_no_size_at_all_is_ranked_by_its_file_size_and_labelled_for_the_page():
    m = parse_tags({"models": [{"name": "mystery:latest", "size": 5_500_000_000}]})[0]
    assert m.params_b is None and m.power == 10.0 and "5.5 GB" in m.label() and LocalModel("x", None, None, "", "", False).power == 0.0


def test_discovery_reports_what_is_wrong_in_plain_words():
    ok = discover("http://x", get=lambda url, timeout: type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: TAGS})())
    assert ok["ok"] and ok["recommended"] == "qwen2.5-coder:32b" and ok["error"] is None and ok["models"][0]["label"]

    def down(url, timeout):
        raise ConnectionError("refused")
    bad = discover("http://x", get=down)
    assert not bad["ok"] and "ollama serve" in bad["error"] and bad["models"] == []
    empty = discover("http://x", get=lambda url, timeout: type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: {"models": []}})())
    assert not empty["ok"] and "no models" in empty["error"] and "ollama pull" in empty["error"]
    embed_only = discover("http://x", get=lambda url, timeout: type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: {"models": [TAGS["models"][4]]}})())
    assert not embed_only["ok"] and "only embedding models" in embed_only["error"]


@pytest.mark.parametrize("bad,why", [("../x.html", ".."), ("a/../../x.html", ".."), ("/etc/passwd", "absolute"), ("C:\\Windows\\a.html", "absolute"), ("C:a.html", "absolute"), ("a/.env", "hidden"),
                                     ("run.bat", "not allowed"), ("x.exe", "not allowed"), ("setup.sh", "not allowed"), ("con.txt", "reserved"), ("NUL.html", "reserved"),
                                     ("a./b.html", "dot"), ("x.html.", "dot"), ("a\x00b.html", "control"), ("", "no name"), ("   ", "no name"), ("a/b/", "not allowed"),
                                     ("Makefile", "not allowed"), ("é.html", "not allowed"), ("a/" + "x" * 70 + ".html", "too long")])
def test_dangerous_or_odd_paths_are_refused_with_a_reason(bad, why):
    with pytest.raises(PolicyError, match=why):
        safe_path(bad)


@pytest.mark.parametrize("good,clean", [("index.html", "index.html"), ("src\\app.js", "src/app.js"), ("./a//b.css", "a/b.css"), ("docs/My File.md", "docs/My File.md"), (".gitignore", ".gitignore"), ("A.PY", "A.PY")])
def test_ordinary_paths_are_cleaned_and_accepted(good, clean):
    assert safe_path(good) == clean


def names(reply, default=""):
    files, unnamed = parse_files(reply, default)
    return [f.path for f in files], unnamed


@pytest.mark.parametrize("reply,expect", [
    ("### FILE: index.html\n```html\n<p>x</p>\n```", ["index.html"]),
    ("**style.css**\n```css\na{}\n```", ["style.css"]),
    ("`app.js`:\n```js\nlet a;\n```", ["app.js"]),
    ("Here is the file `src/util.py`, the main one:\n```python\nx = 1\n```", ["src/util.py"]),
    ("```js title=\"app.js\"\nlet a;\n```", ["app.js"]),
    ("```html filename=index.html\n<p></p>\n```", ["index.html"]),
    ("```html\n<!-- file: index.html -->\n<p></p>\n```", ["index.html"]),
    ("```python\n# file: main.py\nprint(1)\n```", ["main.py"]),
    ("File: a.html\n```html\n1\n```\nFile: b.css\n```css\n2\n```", ["a.html", "b.css"]),
    ("### FILE: ../evil.html\n```html\n1\n```", ["../evil.html"]), ("### FILE: /etc/x.txt\n```\n1\n```", ["/etc/x.txt"]),
    ("### FILE: C:\\Windows\\a.html\n```html\n1\n```", ["C:\\Windows\\a.html"]), ("### FILE: a/../../x.html\n```html\n1\n```", ["a/../../x.html"])])
def test_a_file_name_is_found_however_the_model_labels_the_block(reply, expect):
    assert names(reply)[0] == expect


def test_an_unnamed_block_is_skipped_unless_it_is_the_only_one_and_a_default_name_is_given():
    assert names("```html\n<p>x</p>\n```") == ([], 1)
    assert names("```html\n<p>x</p>\n```", "index.html") == (["index.html"], 0)
    assert names("```html\n1\n```\n```css\n2\n```", "index.html") == ([], 2)                  # with two unnamed blocks a default would be a guess
    assert names("no code here") == ([], 0)


def test_the_first_line_comment_is_not_part_of_the_content_and_files_end_with_one_newline():
    f, _ = parse_files("```html\n<!-- file: index.html -->\n<p>x</p>\n```")
    assert f[0].content == "<p>x</p>\n" and f[0].how == "first-line comment"
    assert parse_files("### FILE: a.txt\n```\nhi\n\n\n```")[0][0].content == "hi\n"
