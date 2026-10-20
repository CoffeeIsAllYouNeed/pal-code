"""The Studio's foundations: ranking local models, the catalogue, the plan rules, the access policy, reading replies, and the file checks."""
import pytest

from pal.studio import plan as planlib
from pal.studio.catalog import DEFAULT_PERMISSIONS, OFFERED_TOOLS, PERMISSIONS, ROLES, catalog, tool_needs
from pal.studio.checks import check_file, check_project
from pal.studio.files import parse_files
from pal.studio.models import LocalModel, discover, parse_tags, strongest
from pal.studio.policy import MAX_FILE_BYTES, MAX_FILES, Policy, PolicyError, safe_path

TAGS = {"models": [
    {"name": "llama3.2:1b", "size": 1_300_000_000, "details": {"parameter_size": "1.2B", "family": "llama", "quantization_level": "Q8_0"}},
    {"name": "llama3.1:8b-instruct-q4_K_M", "size": 4_900_000_000, "details": {"parameter_size": "8.0B", "family": "llama", "quantization_level": "Q4_K_M"}},
    {"name": "llama3:latest", "size": 4_700_000_000, "details": {"parameter_size": "8.0B", "family": "llama"}},
    {"name": "qwen2.5-coder:32b", "size": 19_000_000_000, "details": {"parameter_size": "32.8B"}},
    {"name": "nomic-embed-text:latest", "size": 274_000_000, "details": {"parameter_size": "137M", "family": "nomic-bert"}}, "junk", {"size": 5}]}


# -- ranking local models -------------------------------------------------------------------------------------------------------------------------
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


# -- the catalogue ------------------------------------------------------------------------------------------------------------------------------------
def test_the_catalogue_offers_five_roles_the_permissions_and_only_tools_a_build_can_use():
    from pal.config import load_settings
    cat = catalog(load_settings(env={}).tools, {})
    assert [r["id"] for r in cat["roles"]] == ["designer", "coder", "tester", "docs", "reviewer"] and len(ROLES) == 5
    names = {t["name"] for t in cat["tools"]}
    assert {"calculator", "current_time", "wikipedia", "search_documents"} <= names and not names & {"write_file", "save_file", "read_file", "build_code_graph", "system_info"}
    assert {p["id"]: (p["default"], p["risky"], p["locked"]) for p in cat["permissions"]}["network_access"] == (False, True, False)
    assert {p.id for p in PERMISSIONS if p.locked} == {"run_code", "outside_project", "delete_files"} and not any(DEFAULT_PERMISSIONS[p.id] for p in PERMISSIONS if p.locked or p.risky)


def test_a_tool_says_which_permission_it_needs_and_whether_it_is_ready_in_the_settings():
    from pal.config import load_settings
    assert tool_needs("wikipedia") == "network_access" and tool_needs("search_documents") == "read_references" and tool_needs("calculator") is None
    tools = {t["name"]: t for t in catalog(load_settings(env={}).tools, {})["tools"]}
    assert tools["calculator"]["ready"] and not tools["wikipedia"]["ready"] and "tools.network" in tools["wikipedia"]["why_not"]
    assert not tools["google_search"]["ready"] and set(OFFERED_TOOLS) == set(tools)


# -- the plan rules ------------------------------------------------------------------------------------------------------------------------------------
def norm(raw, **kw):
    kw.setdefault("installed", ["llama3.1:8b"])
    kw.setdefault("recommended", "llama3.1:8b")
    return planlib.normalize(raw, **kw)


def test_a_hostile_plan_is_made_safe_and_every_change_is_explained():
    plan, notes = norm({"product": {"name": "X" * 200, "kind": "spaceship"}, "roles": ["wizard", "designer"], "tools": ["calculator", "wikipedia", "rm_rf"],
                        "permissions": {"network_access": True, "run_code": True, "delete_files": True, "outside_project": True, "root": True},
                        "model": {"mode": "local", "model": "gpt-9"}})
    assert plan["roles"] == ["designer", "coder"] and plan["tools"] == ["calculator"] and plan["product"]["kind"] == "web_app" and len(plan["product"]["name"]) == 60
    assert not any(plan["permissions"][p] for p in ("network_access", "run_code", "delete_files", "outside_project")) and "root" not in plan["permissions"]
    assert plan["model"]["model"] == "llama3.1:8b"
    text = " | ".join(notes)
    for part in ("wizard", "added the Coder", "only you can switch it on", "needs a sandbox", "never allowed", "rm_rf", "gpt-9 is not installed"):
        assert part in text, part


def test_only_the_person_can_switch_a_risky_permission_on_and_then_its_tools_are_allowed():
    raw = {"roles": ["coder"], "tools": ["wikipedia"], "permissions": {"network_access": True}}
    off, _ = norm(raw, ready_tools=["wikipedia", "calculator"])
    assert not off["permissions"]["network_access"] and off["tools"] == []
    on, notes = norm(raw, user_set=["network_access"], ready_tools=["wikipedia", "calculator"])
    assert on["permissions"]["network_access"] and on["tools"] == ["wikipedia"] and not notes
    unready, notes = norm(raw, user_set=["network_access"], ready_tools=["calculator"])
    assert unready["tools"] == [] and "not switched on in the settings" in " ".join(notes)


def test_the_model_choice_keeps_a_valid_api_model_and_falls_back_to_the_recommended_local_one():
    api, _ = norm({"roles": ["coder"], "model": {"mode": "api", "provider": "anthropic", "model": "claude-x"}})
    assert api["model"] == {"mode": "api", "provider": "anthropic", "model": "claude-x", "api_mode": "own_key"}
    router, _ = norm({"roles": ["coder"], "model": {"mode": "api", "provider": "openrouter", "model": "meta/x:free"}})
    assert router["model"]["api_mode"] == "explore"
    for bad in ({"mode": "api", "provider": "evilcorp", "model": "x"}, {"mode": "api", "provider": "openai", "model": ""}, "nonsense", None):
        assert norm({"roles": ["coder"], "model": bad})[0]["model"] == {"mode": "local", "model": "llama3.1:8b", "provider": "ollama", "api_mode": ""}
    assert norm({"roles": ["coder"]}, installed=[], recommended=None)[0]["model"]["model"] == ""


def test_steps_follow_the_chosen_agents_are_capped_and_default_in_build_order():
    plan, _ = norm({"roles": ["reviewer", "docs", "coder", "designer"]})
    assert [s["role"] for s in plan["steps"]] == ["designer", "coder", "docs", "reviewer"]
    custom, _ = norm({"roles": ["coder"], "steps": [{"role": "coder", "task": "A"}, {"role": "tester", "task": "B"}, {"role": "coder", "task": ""}, "x"] + [{"role": "coder", "task": f"t{i}"} for i in range(20)]})
    assert custom["steps"][0] == {"role": "coder", "task": "A"} and all(s["role"] == "coder" for s in custom["steps"]) and len(custom["steps"]) == 8
    assert norm({"roles": ["a", "b", "c", "d", "e", "f"]})[0]["roles"] == ["coder"]


def test_reasons_are_kept_only_for_things_that_are_in_the_plan_and_the_product_gets_a_safe_folder_name():
    plan, _ = norm({"product": {"name": "../My Cool App!!"}, "roles": ["coder", "docs"], "tools": ["calculator"],
                    "reasons": {"role:coder": "writes code", "role:tester": "x", "tool:calculator": "sums", "tool:wikipedia": "x", "model": "m", "permission:network_access": "off", "junk": "x"}})
    assert set(plan["reasons"]) == {"role:coder", "tool:calculator", "model", "permission:network_access"} and plan["product"]["folder"] == "my-cool-app"
    assert planlib.slug("   ") == "project" and planlib.slug("A/B\\C") == "a-b-c"
    assert [planlib.slug(x) for x in ("CON", "nul", "Com1", "console")] == ["p-con", "p-nul", "p-com1", "console"]


def test_a_plan_that_is_already_valid_is_unchanged_when_normalised_again():
    plan, _ = norm({"product": {"name": "A", "summary": "s", "kind": "cli_tool"}, "roles": ["coder", "tester"], "tools": ["calculator"], "metrics": True})
    again, notes = norm(plan)
    assert again == plan and notes == []


def test_diff_describes_changes_in_words():
    a, _ = norm({"roles": ["coder"]})
    b, _ = norm({"roles": ["designer", "coder"], "tools": ["calculator"], "metrics": True}, user_set=[])
    c = dict(b, permissions={**b["permissions"], "modify_own_files": False})
    d = planlib.diff(a, b)
    assert "added agent: Designer" in d and "added tool: calculator" in d and "build twice to compare: on" in d and any(x.startswith("steps changed") for x in d)
    assert "turned off: change files it already wrote" in planlib.diff(b, c) and planlib.diff(a, a) == []


# -- the access policy -------------------------------------------------------------------------------------------------------------------------------
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


def test_the_policy_enforces_permissions_limits_and_ignores_case_like_windows_does():
    p = Policy({"write_project": True, "modify_own_files": False})
    assert p.check_write("index.html", "x").allowed
    d = p.check_write("INDEX.HTML", "y")
    assert not d.allowed and "already written" in d.reason                                       # one file on Windows and macOS
    assert not p.check_write("big.html", "x" * (MAX_FILE_BYTES + 1)).allowed
    q = Policy({"write_project": True, "modify_own_files": True})
    assert q.check_write("a.html", "1").allowed and q.check_write("A.html", "2").allowed and len(q.written) == 1
    for i in range(MAX_FILES - 1):
        assert q.check_write(f"f{i}.txt", "x").allowed
    assert not q.check_write("one-more.txt", "x").allowed and "too many files" in q.blocked[-1][1]
    dry = Policy({"write_project": False})
    assert not dry.check_write("a.html", "x").allowed and "dry run" in dry.check_write("a.html", "x").reason and dry.blocked == []
    assert p.blocked and all(path and reason for path, reason in p.blocked)


def test_the_total_size_of_a_project_is_capped():
    p = Policy({"write_project": True, "modify_own_files": True})
    for i in range(10):
        assert p.check_write(f"f{i}.txt", "x" * 199_000).allowed
    assert not p.check_write("more.txt", "x" * 199_000).allowed and "over 2 MB" in p.blocked[-1][1]
    assert p.check_write("f0.txt", "tiny").allowed                                               # a smaller version of an existing file fits


def test_tools_are_checked_against_the_permissions():
    p = Policy({"network_access": False, "read_references": True})
    assert not p.check_tool("wikipedia").allowed and p.check_tool("search_documents").allowed and p.check_tool("calculator").allowed
    assert p.allowed_tools(["wikipedia", "calculator", "search_documents"]) == {"calculator", "search_documents"}
    assert Policy({"network_access": True}).check_tool("wikipedia").allowed and not Policy({}).check_tool("search_documents").allowed


# -- reading a reply ---------------------------------------------------------------------------------------------------------------------------------
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


# -- the static checks -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path,content,expect", [
    ("a.py", "def f(:\n  pass", "syntax error on line 1"), ("a.py", "x = 1\n", None), ("a.json", "{'a': 1}", "not valid JSON"), ("a.json", '{"a": 1}', None),
    ("a.html", "<html><body><div>hi</body></html>", "<div> is never closed"), ("a.html", "<html><body></body></html></div>", "</div> closes something"),
    ("a.html", "<html><body><br><img src='x.png'><p>ok<input></body></html>", None), ("a.html", "<script>if (a) {</script>", None),
    ("a.js", "function f() { return 1;", "never closed"), ("a.js", "function f() { return (1; }", "no matching opener"), ("a.js", "let s = '}'; // {\nlet t = `${1}`;", None),
    ("a.css", "body { color: red;", "never closed"), ("a.css", "/* { */ body { color: red; }", None), ("a.md", "{{{ [[", None), ("a.txt", "(", None)])
def test_the_static_checks_find_the_mistakes_small_models_make_and_leave_correct_files_alone(path, content, expect):
    found = check_file(path, content)
    assert (expect is None and not found) or (expect is not None and any(expect in f for f in found)), found


def test_a_page_that_links_to_a_file_nobody_wrote_is_reported_and_links_that_are_fine_are_not():
    html = '<html><head><link rel="stylesheet" href="style.css"><script src="js/app.js"></script><script src="https://cdn.x/y.js"></script></head><body><img src="data:image/png;base64,AA"><a href="#top">t</a></body></html>'
    out = check_project({"index.html": html, "style.css": "a{}"})
    assert out == {"index.html": ["it refers to js/app.js, but no file with that name was written"]}
    assert check_project({"index.html": html, "style.css": "a{}", "JS/App.js": "1"}) == {}                  # case-insensitive, like Windows
    sub = check_project({"pages/a.html": '<script src="../app.js"></script>', "app.js": "1"})
    assert sub == {}


def test_single_file_checks_run_only_on_the_files_that_just_changed():
    files = {"old.py": "def f(:", "new.py": "x = 1"}
    assert check_project(files, only={"new.py"}) == {} and "old.py" in check_project(files)
