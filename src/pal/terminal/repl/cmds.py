"""Every \\command: plans, context, branches, history, changes, config and models. Each prints what it did in the same words every time."""
from __future__ import annotations

import fnmatch
import re
import time
from pathlib import Path

from pal.terminal.repl import changes, term
from pal.terminal.repl.term import BOLD, RED_HI, WHITE, Exit, accent, c, fail, out, print_cmds, spinner
from pal.terminal.workflow import AUTO_MODES, CONFIG, Plan, save_default_config
from pal.tokens import estimate_tokens

SHRUG = "\U0001F937‍♂️"
COIN = "\U0001FA99"


def green(text: str) -> str:
    return c(f"{BOLD};{term.HI_GREEN}", text)


# -- tables shared with the welcome ---------------------------------------------------------------------------------------------------
def auto_mode_table(cfg: dict) -> str:
    ctx = "auto" if cfg["auto-load-context"] else "manual"
    apply = "auto" if cfg["auto-apply"] else "approve"
    exe = "auto" if cfg["auto-exec"] else ("approve" if cfg["can-exec"] else "disabled")
    commits = "auto" if cfg["auto-commit"] else ("skip" if cfg["skip-commit"] else "manual")
    return term.table([[c(f"{BOLD};{term.HI_MAGENTA}", cfg["auto-mode"]), ctx, apply, exe, commits]],
                      header=["\U0001F680 Auto Mode", "Context", "Apply", "Execution", "Commits"])


def model_table(name: str) -> str:
    return term.table([[c(f"{BOLD};{term.HI_MAGENTA}", name)]], header=["\U0001F9E0 Model"])


def current_plan_table(wf) -> str:
    ctx, convo = wf.tokens()
    name = c(f"{BOLD};{term.HI_GREEN}", wf.plan.name)
    return term.table([[name, term.rel_time(wf.plan.updated), term.rel_time(wf.plan.created), wf.plan.branch, f"{ctx} {COIN}", f"{convo} {COIN}"]],
                      header=["Current Plan", "Updated", "Created", "Branch", "Context", "Convo"])


# -- plans ------------------------------------------------------------------------------------------------------------------------------
def cmd_new(ctx, args: list[str]) -> None:
    name = ""
    if "-n" in args or "--name" in args:
        i = args.index("-n") if "-n" in args else args.index("--name")
        name = args[i + 1] if i + 1 < len(args) else ""
    flag = next((a[2:] for a in args if a in ("--no-auto", "--none", "--basic", "--plus", "--semi", "--full")), "")
    flag = {"no-auto": "none"}.get(flag, flag)
    spinner.start("")
    try:
        made = ctx.new_plan(name)
    except ValueError as e:
        fail(f"Error creating plan: {e}")
    spinner.stop()
    wf = ctx.wf
    out(f"✅ Started new plan {c(f'{BOLD};{term.HI_GREEN}', made)} and set it to current plan")
    out("⚙️  Using default config")
    if flag and flag != wf.plan.autonomy:
        old = wf.plan.config()
        wf.plan.set_auto(flag)
        wf.save()
        out(auto_mode_table(old), end="")
    spinner.start("")
    spinner.stop()
    out(model_table(ctx.model_name()), end="")
    if wf.plan.setting("auto-load-context"):
        out("\U0001F4E5 Automatic context loading is enabled")
    else:
        out()
    out()
    cmds = ["config", "plans", "cd", "models"]
    if not wf.plan.setting("auto-load-context"):
        cmds.insert(0, "load")
    print_cmds(*cmds)


def _plans(ctx, archived: bool = False) -> list[Plan]:
    store = ctx.wf.store
    plans = []
    for n in store.names():
        try:
            p = store.load(n) if n != ctx.wf.plan.name else ctx.wf.plan
        except Exception:  # noqa: BLE001
            continue
        if p.archived == archived:
            plans.append(p)
    plans.sort(key=lambda p: -p.updated)
    return plans


def _plan_tokens(p: Plan, cwd: Path) -> tuple[int, int]:
    from pal.terminal.session import read_any
    b = p.b
    ctx = 0
    for key in b.context:
        f = Path(key) if Path(key).is_absolute() else cwd / key
        text, _ = read_any(f) if f.is_file() else (None, None)
        ctx += estimate_tokens(text or "")
    ctx += sum(estimate_tokens(n) for n in b.notes)
    convo = sum(int(m.get("tokens") or estimate_tokens(m.get("text", ""))) for m in b.convo)
    return ctx, convo


def cmd_plans(ctx, args: list[str]) -> None:
    if "--archived" in args or "-a" in args:
        spinner.start("")
        plans = _plans(ctx, archived=True)
        spinner.stop()
        if not plans:
            out(f"{SHRUG} No archived plans")
            out()
            print_cmds("archive")
            return
        rows = []
        for i, p in enumerate(plans, 1):
            num = green(str(i)) if p.name == ctx.wf.plan.name else str(i)
            rows.append([c(BOLD, num), p.name, term.rel_time(p.updated)])
        changes.page(term.table(rows, header=["#", "Name", "Updated"]))
        out()
        print_cmds("unarchive")
        return
    spinner.start("")
    plans = _plans(ctx)
    spinner.stop()
    if not plans:
        out(f"{SHRUG} No plans")
        out()
        print_cmds("new")
        return
    rows = []
    for i, p in enumerate(plans, 1):
        cur = p.name == ctx.wf.plan.name
        t_ctx, t_convo = _plan_tokens(p, ctx.wf.session.cwd)
        rows.append([c(BOLD, green(str(i)) if cur else str(i)), (green(p.name) + " \U0001F448") if cur else p.name, term.rel_time(p.updated),
                     p.branch, f"{t_ctx} {COIN}", f"{t_convo} {COIN}"])
    changes.page("\n" + term.table(rows, header=["#", "Name", "Updated", "Branch", "Context", "Convo"]) + "\n")
    out()
    print_cmds("new", "cd", "delete-plan", "plans --archived", "archive")


def _pick_plan(ctx, arg: str, plans: list[Plan], question: str) -> str:
    names = [p.name for p in plans]
    if not arg:
        try:
            return term.select(question, names)
        except term.Interrupted:
            raise Exit(0) from None
    if arg.isdigit():
        i = int(arg)
        if not 0 < i <= len(names):
            fail("Plan index out of range")
        return names[i - 1]
    if arg not in names:
        fail("Plan not found")
    return arg


def cmd_cd(ctx, args: list[str]) -> None:
    spinner.start("")
    plans = _plans(ctx)
    spinner.stop()
    if not plans:
        out(f"{SHRUG} No plans")
        out()
        print_cmds("new")
        return
    name = _pick_plan(ctx, args[0] if args else "", plans, "Select a plan")
    try:
        ctx.switch_plan(name)
    except ValueError as e:
        fail(f"Error setting current plan: {e}")
    out(f"✅ Changed current plan to {c(f'{term.HI_GREEN};{BOLD}', name)}")
    out()
    print_cmds("current")


def cmd_current(ctx, args: list[str]) -> None:
    spinner.start("")
    table = current_plan_table(ctx.wf)
    spinner.stop()
    out(table)
    print_cmds("ls", "plans")


def cmd_rename(ctx, args: list[str]) -> None:
    if args:
        new = args[0]
    else:
        try:
            new = term.ask_text("New name:")
        except term.Interrupted:
            raise Exit(0) from None
    if not new:
        out(f"{SHRUG} No new name provided")
        return
    spinner.start("")
    try:
        ctx.wf.rename(new)
    except ValueError as e:
        fail(f"Error renaming plan: {e}")
    spinner.stop()
    out(f"✅ Plan renamed to {green(new)}")


def cmd_delete_plan(ctx, args: list[str]) -> None:
    wf = ctx.wf
    if "--all" in args:
        spinner.start("")
        for p in _plans(ctx) + _plans(ctx, archived=True):
            if p.name != wf.plan.name:
                try:
                    wf.delete(p.name)
                except ValueError:
                    pass
        wf.delete(wf.plan.name)
        ctx.after_plan_change()
        spinner.stop()
        out("✅ Deleted all plans")
        out(f"✅ Started new plan {green(wf.plan.name)} and set it to current plan")
        return
    spinner.start("")
    plans = _plans(ctx)
    spinner.stop()
    if not plans:
        out(f"{SHRUG} No plans")
        out()
        print_cmds("new")
        return
    names = [p.name for p in plans]
    arg = args[0] if args else ""
    if not arg:
        try:
            chosen = [term.select("Select a plan:", names)]
        except term.Interrupted:
            raise Exit(0) from None
    elif re.fullmatch(r"\d+-\d+", arg):
        a, b = (int(x) for x in arg.split("-"))
        chosen = [names[i - 1] for i in range(a, b + 1) if 0 < i <= len(names)]
    elif "*" in arg:
        chosen = [n for n in names if fnmatch.fnmatchcase(n, arg)]
    elif arg.isdigit():
        i = int(arg)
        if not 0 < i <= len(names):
            fail("Plan index out of range")
        chosen = [names[i - 1]]
    else:
        chosen = [arg] if arg in names else []
    if not chosen:
        fail("No matching plans found")
    out()
    out(f"The following {len(chosen)} plan(s) will be deleted:")
    for n in chosen:
        out(f"  - {accent(n)}")
    out()
    if not term.confirm("Are you sure you want to delete these plans?"):
        out("Operation cancelled")
        return
    was = wf.plan.name
    for n in chosen:
        spinner.start("")
        try:
            wf.delete(n)
        except ValueError as e:
            fail(f"Error deleting plan {n}: {e}")
        spinner.stop()
    ctx.after_plan_change()
    out(f"✅ Deleted plan '{chosen[0]}'" if len(chosen) == 1 else f"✅ Deleted {len(chosen)} plans")
    if wf.plan.name != was:
        out(f"✅ Current plan is now {green(wf.plan.name)}")


def cmd_archive(ctx, args: list[str]) -> None:
    spinner.start("")
    plans = _plans(ctx)
    spinner.stop()
    if not plans:
        out(f"{SHRUG} No plans available to archive")
        return
    names = [p.name for p in plans]
    arg = args[0] if args else ""
    if not arg:
        try:
            name = term.select("Select a plan to archive", names)
        except term.Interrupted:
            raise Exit(0) from None
    elif arg.isdigit() and 0 < int(arg) <= len(names):
        name = names[int(arg) - 1]
    else:
        name = arg
    plan = next((p for p in plans if p.name == name), None)
    if plan is None:
        fail("Plan not found")
    plan.archived = True
    if plan is ctx.wf.plan:
        ctx.wf.save()
    else:
        ctx.wf.store.save(plan, current=False)
    out(f"✅ Plan {c(f'{BOLD};{term.HI_YELLOW}', name)} archived")
    out()
    print_cmds("plans --archived", "unarchive")


def cmd_unarchive(ctx, args: list[str]) -> None:
    spinner.start("")
    plans = _plans(ctx, archived=True)
    spinner.stop()
    if not plans:
        out(f"{SHRUG} No archived plans")
        return
    names = [p.name for p in plans]
    arg = args[0] if args else ""
    if not arg:
        try:
            name = term.select("Select a plan:", names)
        except term.Interrupted:
            raise Exit(0) from None
    elif arg.isdigit() and 0 < int(arg) <= len(names):
        name = names[int(arg) - 1]
    else:
        name = arg
    plan = next((p for p in plans if p.name == name), None)
    if plan is None:
        fail("Plan not found")
    plan.archived = False
    ctx.wf.store.save(plan, current=False)
    out(f"✅ Plan {green(name)} unarchived")
    out()
    print_cmds("plans", "current")


# -- context ---------------------------------------------------------------------------------------------------------------------------
TYPES = {"file": ("file", "\U0001F4C4"), "note": ("note", "✏️ "), "tree": ("tree", "\U0001F5C2 "), "piped": ("piped", "↔️ ")}


def _short(name: str) -> str:
    return name[:20] + "⋯" + name[-20:] if len(name) > 40 else name


def _note_kind(note: str) -> tuple[str, str]:
    if note.startswith("\x00tree "):
        head = note.split("\n", 1)[0][6:]
        return "tree", head
    if note.startswith("\x00piped\n"):
        return "piped", "piped data"
    return "note", note.split("\n", 1)[0][:40]


def context_items(wf) -> list[dict]:
    """Everything in context, in order: {"name", "type", "tokens", "added", "updated", "key"}."""
    from pal.terminal.session import read_any
    items = []
    b = wf.plan.b
    for key in b.context:
        p = Path(key) if Path(key).is_absolute() else wf.session.cwd / key
        text, _ = read_any(p) if p.is_file() else (None, None)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = b.added.get(key, b.created)
        items.append({"name": key, "type": "file", "tokens": estimate_tokens(text or ""), "added": b.added.get(key, b.created), "updated": max(mtime, b.added.get(key, 0)),
                      "key": key})
    for i, note in enumerate(b.notes):
        kind, name = _note_kind(note)
        body = note.split("\n", 1)[1] if kind in ("tree", "piped") and "\n" in note else note
        t = b.added.get(f"note:{i}", b.created)
        items.append({"name": name, "type": kind, "tokens": estimate_tokens(body), "added": t, "updated": t, "key": f"note:{i}"})
    return items


def _load_table(items: list[dict], sign: str, colour: str) -> str:
    rows = []
    for it in items:
        label, icon = TYPES[it["type"]]
        rows.append([c(f"{colour};{BOLD}", f" {icon} {_short(it['name'])}"), c(colour, label), c(colour, f"{sign}{it['tokens']}")])
    return term.table(rows, header=["Name", "Type", COIN])


def _parts(items: list[dict], tree_word: str = "directory tree") -> list[str]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it["type"]] = counts.get(it["type"], 0) + 1
    parts = []
    if counts.get("note"):
        parts.append("a note" if counts["note"] == 1 else f"{counts['note']} notes")
    if counts.get("piped"):
        parts.append("piped data")
    if counts.get("file"):
        parts.append(f"{counts['file']} file{'s' if counts['file'] > 1 else ''}")
    if counts.get("tree"):
        parts.append(f"{counts['tree']} {tree_word}{'s' if counts['tree'] > 1 else ''}")
    return parts


def _join(parts: list[str]) -> str:
    if len(parts) <= 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _tree_text(root: Path, rel: str) -> str:
    from pal.terminal.session import SKIP_DIRS
    base = root / rel if rel not in ("", ".") else root
    lines = []
    for p in sorted(base.rglob("*")):
        parts = p.relative_to(base).parts
        if any(x in SKIP_DIRS or x.startswith(".") for x in parts):
            continue
        lines.append(("  " * (len(parts) - 1)) + p.name + ("/" if p.is_dir() else ""))
        if len(lines) >= 3000:
            break
    return "\n".join(lines)


def cmd_load(ctx, args: list[str], piped: str = "") -> None:
    wf = ctx.wf
    note, recursive, tree, force = "", False, False, False
    paths: list[str] = []
    it = iter(range(len(args)))
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-n", "--note"):
            note = args[i + 1] if i + 1 < len(args) else ""
            i += 2
            continue
        if a in ("-r", "--recursive"):
            recursive = True
        elif a == "--tree":
            tree = True
        elif a in ("-f", "--force"):
            force = True
        elif a in ("--map", "-d", "--detail"):
            pass
        else:
            paths.append(a[2:] if a.startswith("./") else a)
        i += 1
    del it, force
    spinner.start("\U0001F333 Loading directory tree..." if tree else "\U0001F4E5 Loading context...")
    before = context_items(wf)
    already: list[tuple[str, str]] = []
    added_keys: list[str] = []
    t0 = time.time()
    if note:
        wf.plan.b.notes.append(note[:20000])
        wf.plan.b.added[f"note:{len(wf.plan.b.notes) - 1}"] = t0
    if piped:
        wf.plan.b.notes.append("\x00piped\n" + piped[:200000])
        wf.plan.b.added[f"note:{len(wf.plan.b.notes) - 1}"] = t0
    for raw in paths:
        p = (wf.session.cwd / raw) if not Path(raw).expanduser().is_absolute() else Path(raw).expanduser()
        if tree:
            name = {".": "cwd", "..": "parent"}.get(raw, raw)
            if any(n.startswith(f"\x00tree {name}\n") for n in wf.plan.b.notes):
                already.append(("tree", name))
                continue
            wf.plan.b.notes.append(f"\x00tree {name}\n" + _tree_text(wf.session.cwd, raw))
            wf.plan.b.added[f"note:{len(wf.plan.b.notes) - 1}"] = t0
            continue
        if p.is_dir() and not recursive and not any(ch in raw for ch in "*?["):
            spinner.stop()
            fail(f"Failed to load context: failed to parse input paths: cannot process directory {raw}: requires --recursive/-r, or --tree flag")
        keys_before = set(wf.plan.b.context)
        added, problems = wf.load([raw])
        for prob in problems:
            spinner.stop()
            if "nothing found" in prob:
                fail(f"Failed to load context: failed to get file info for {raw}: no such file or directory")
            fail(f"Failed to load context: {prob}")
        for k in added:
            if k != "note":
                wf.plan.b.added[k] = t0
                added_keys.append(k)
        if not added and not problems:
            found = wf._resolve(raw)
            for f in found:
                k = wf._key(f)
                if k in keys_before:
                    already.append(("file", k))
    wf.save()
    after = context_items(wf)
    new_items = [x for x in after if x["key"] not in {b["key"] for b in before}]
    spinner.stop()
    if not new_items:
        out(f"{SHRUG} No context loaded")
        if already:
            _already(already)
        else:
            out()
            out("Use " + c(f"{term.BG_CYAN};{WHITE}", " \\load [file-path] ") + " to load a file:")
            out("\\load file.c file.h")
            out()
            out(accent("Load a whole directory") + " with the --recursive/-r flag:")
            out("\\load app/src -r")
            out()
            out(accent("Load a directory layout (file names only)") + " with the --tree flag:")
            out("\\load app/src --tree")
            out()
            out(c(f"{BOLD};{term.HI_YELLOW}", "Note:") + " file paths are relative to the current directory")
            out()
            out(accent("Load a note") + " with the -n flag:")
            out("\\load -n 'Some note here'")
        return
    added_tokens = sum(x["tokens"] for x in new_items)
    total = sum(x["tokens"] for x in after)
    summary = "Loaded " + _join(_parts(new_items)) + f" into context | added → {added_tokens} {COIN} |  total → {total} {COIN}"
    out("✅ " + summary + "\n\n" + _load_table(new_items, "+", term.HI_GREEN).strip())
    wf.log_event(summary + "\n\n" + term.strip(_load_table(new_items, "+", term.HI_GREEN)).strip())
    if already:
        _already(already)
    out()
    print_cmds("ls", "debug")


def _already(items: list[tuple[str, str]]) -> None:
    out(f"\n\U0001F645‍♂️ Skipped because {'it' if len(items) == 1 else 'they'}'{'s' if len(items) == 1 else 're'} already in context:")
    for kind, name in items:
        out(f"  • {TYPES[kind][1]} {name}")


def cmd_ls(ctx, args: list[str]) -> None:
    spinner.start("")
    items = context_items(ctx.wf)
    spinner.stop()
    if not items:
        out(f"{SHRUG} No context")
        out()
        print_cmds("load")
        return
    rows = []
    for i, it in enumerate(items, 1):
        label, icon = TYPES[it["type"]]
        rows.append([c(BOLD, str(i)), c(f"{term.HI_GREEN};{BOLD}", f" {icon} {_short(it['name'])}"), label, str(it["tokens"]), term.rel_time(it["added"]),
                     term.rel_time(it["updated"])])
    out(term.table(rows, header=["#", "Name", "Type", COIN, "Added", "Updated"]), end="")
    total = sum(it["tokens"] for it in items)
    out(term.table([[accent("Total tokens →") + c(BOLD, f" {total} {COIN}")]]), end="")
    out()
    print_cmds("load", "rm", "clear")


def _remove(ctx, keys: list[str]) -> None:
    wf = ctx.wf
    items = context_items(wf)
    gone = [it for it in items if it["key"] in keys]
    files = [k for k in keys if not k.startswith("note:")]
    notes = sorted((int(k[5:]) for k in keys if k.startswith("note:")), reverse=True)
    if files:
        wf.remove(files)
        for f in files:
            wf.plan.b.added.pop(f, None)
    for i in notes:
        del wf.plan.b.notes[i]
    if notes:
        fresh = {k: v for k, v in wf.plan.b.added.items() if not k.startswith("note:")}
        wf.plan.b.added = fresh
    wf.save()
    total = sum(x["tokens"] for x in context_items(wf))
    removed = sum(x["tokens"] for x in gone)
    n = len(gone)
    summary = f"Removed {n} piece{'s' if n > 1 else ''} of context | removed → {removed} {COIN} | total → {total} {COIN}"
    out("✅ " + summary + "\n\n" + _load_table(gone, "-", term.HI_RED))
    wf.log_event(summary + "\n\n" + term.strip(_load_table(gone, "-", term.HI_RED)))


def cmd_rm(ctx, args: list[str]) -> None:
    if not args:
        term.usage_error("requires at least 1 arg(s), only received 0")
    spinner.start("")
    items = context_items(ctx.wf)
    keys: list[str] = []
    for a in args:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", a)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            keys += [items[i - 1]["key"] for i in range(lo, hi + 1) if 0 < i <= len(items)]
            continue
        for it in items:
            if it["name"] == a or it["key"] == a or fnmatch.fnmatch(it["name"], a) or it["name"].startswith(a.rstrip("/") + "/"):
                keys.append(it["key"])
    spinner.stop()
    if not keys:
        out(f"{SHRUG} No context removed")
        return
    _remove(ctx, list(dict.fromkeys(keys)))


def cmd_clear(ctx, args: list[str]) -> None:
    spinner.start("")
    items = context_items(ctx.wf)
    spinner.stop()
    if not items:
        out(f"{SHRUG} No context removed")
        return
    _remove(ctx, [it["key"] for it in items])


def cmd_show(ctx, args: list[str]) -> None:
    if len(args) != 1:
        term.usage_error(f"accepts 1 arg(s), received {len(args)}")
    items = context_items(ctx.wf)
    a = args[0]
    if a.isdigit():
        i = int(a)
        if not 0 < i <= len(items):
            term.usage_error(f"invalid context index: {a}")
        it = items[i - 1]
    else:
        it = next((x for x in items if x["name"] == a or x["key"] == a), None)
        if it is None:
            term.usage_error(f"no context found with name: {a}")
    if it["type"] == "file":
        from pal.terminal.session import read_any
        p = Path(it["key"]) if Path(it["key"]).is_absolute() else ctx.wf.session.cwd / it["key"]
        body, _ = read_any(p)
        out(body or "")
    else:
        note = ctx.wf.plan.b.notes[int(it["key"][5:])]
        out(note.split("\n", 1)[1] if note.startswith("\x00") else note)


def cmd_update(ctx, args: list[str]) -> None:
    spinner.start("")
    spinner.stop()
    out("✅ Context is up to date")


# -- branches --------------------------------------------------------------------------------------------------------------------------
def cmd_branches(ctx, args: list[str]) -> None:
    wf = ctx.wf
    spinner.start("")
    rows = []
    cur_branch = wf.plan.branch
    for i, (name, br) in enumerate(wf.plan.branches.items(), 1):
        cur = name == cur_branch
        convo = sum(int(m.get("tokens") or estimate_tokens(m.get("text", ""))) for m in br.convo)
        keep = wf.plan.branch
        wf.plan.branch = name
        t_ctx = sum(x["tokens"] for x in context_items(wf))
        wf.plan.branch = keep
        rows.append([c(BOLD, green(str(i)) if cur else str(i)), (green(name) + " \U0001F448") if cur else name, term.rel_time(br.updated),
                     f"{t_ctx} {COIN}", f"{convo} {COIN}"])
    spinner.stop()
    out(term.table(rows, header=["#", "Name", "Updated", "Context", "Convo"]), end="")
    out()
    print_cmds("checkout", "delete-branch")


def cmd_checkout(ctx, args: list[str]) -> None:
    wf = ctx.wf
    yes = "-y" in args or "--yes" in args
    args = [a for a in args if a not in ("-y", "--yes")]
    names = list(wf.plan.branches)
    name = ""
    if args:
        a = args[0]
        if a.isdigit():
            i = int(a)
            if not 0 < i <= len(names):
                fail(f"Branch {i} not found")
            name = names[i - 1]
        elif a in names:
            name = a
        else:
            out(f"\U0001F331 Branch {accent(a)} not found")
            if yes:
                out("✅ --yes flag set, will create branch")
            elif not term.confirm("Create it now?"):
                return
            name = a
    else:
        try:
            pick = term.select("Select a branch", names + ["Create a new branch"])
        except term.Interrupted:
            raise Exit(0) from None
        if pick == "Create a new branch":
            try:
                name = term.ask_text("Branch name")
            except term.Interrupted:
                raise Exit(0) from None
        else:
            name = pick
    if not name:
        fail("Branch not found")
    spinner.start("")
    try:
        wf.checkout(name)
    except ValueError as e:
        fail(f"Error creating branch: {e}")
    spinner.stop()
    out(f"✅ Checked out branch {green(name)}")
    out()
    print_cmds("load", "branches", "delete-branch")


def cmd_delete_branch(ctx, args: list[str]) -> None:
    wf = ctx.wf
    if args and args[0] == "main":
        out("\U0001F6A8 Cannot delete main branch")
        return
    names = list(wf.plan.branches)
    if not args:
        opts = [n for n in names if n != "main"]
        if not opts:
            out(f"{SHRUG} No branches to delete")
            return
        try:
            name = term.select("Select a branch to delete", opts)
        except term.Interrupted:
            raise Exit(0) from None
    else:
        a = args[0]
        if a.isdigit():
            i = int(a)
            if not 0 < i <= len(names):
                fail("Branch index out of range")
            name = names[i - 1]
        elif a in names:
            name = a
        else:
            fail("Branch not found")
    if name not in names or name == "main":
        out(f"{SHRUG} Branch {accent(name)} does not exist")
        return
    spinner.start("")
    try:
        wf.delete_branch(name)
    except ValueError as e:
        fail(f"Error deleting branch: {e}")
    spinner.stop()
    out(f"✅ Deleted branch {accent(name)}")
    out()
    print_cmds("branches")


# -- history ---------------------------------------------------------------------------------------------------------------------------
def log_body(wf) -> str:
    entries = []
    for e in reversed(wf.plan.b.log):
        head = c(f"{term.RED_HI};1", f"\U0001F4DD Update {e['id']}") + " | " + c(term.RED_HI, term.log_time(e["t"]))
        entries.append(head + ("\n" + e["msg"] if e["msg"] else ""))
    return "\n\n".join(entries)


def cmd_log(ctx, args: list[str]) -> None:
    spinner.start("")
    body = log_body(ctx.wf)
    spinner.stop()
    changes.page(body)
    out()
    print_cmds("rewind", "continue", "convo")


def _format_log_message(msg: str) -> str:
    if "User prompt" in msg:
        return "\U0001F4AC " + msg
    if "Pal reply" in msg:
        return "\U0001F916 " + msg
    if "Build pending" in msg:
        return "\U0001F3D7️  Building changes"
    if "Loaded" in msg:
        return "\U0001F4DA " + msg
    return msg


def cmd_rewind(ctx, args: list[str]) -> None:
    wf = ctx.wf
    revert_flag = "--revert" in args
    skip_flag = "--skip-revert" in args
    if revert_flag and skip_flag:
        fail("Cannot pass both --revert and --skip-revert")
    commit_flag = "--commit" in args
    skip_commit_flag = "--skip-commit" in args
    args = [a for a in args if not a.startswith("--")]
    spinner.start("")
    log = list(reversed(wf.plan.b.log))
    if not log:
        spinner.stop()
        out(f"{SHRUG} Nothing to rewind")
        return
    steps_back = 0
    if not args:
        opts = [f"{e['id']} | {_format_log_message(e['msg'].split(chr(10))[0].strip())}" for e in log]
        spinner.stop()
        try:
            pick = term.select("Select step to rewind to:", opts)
        except term.Interrupted:
            raise Exit(0) from None
        target = pick.split(" | ")[0]
    elif args[0].isdigit() and 0 < int(args[0]) < 999:
        steps_back = int(args[0])
        if steps_back >= len(log):
            fail(f"Error getting logs: there are only {len(log)} updates")
        target = log[steps_back]["id"]
    else:
        target = args[0]
    entry = next((e for e in log if e["id"] == target or (len(target) >= 4 and e["id"].startswith(target))), None)
    if entry is None:
        fail(f"Error getting timestamp for target sha: {target}")
    target = entry["id"]
    idx = wf.plan.b.log.index(entry)
    st = entry["state"]
    undone = wf.plan.b.applied[st["applied"]:]
    cfg = wf.plan.config()
    should_revert = (revert_flag and not skip_flag) if (revert_flag or skip_flag) else cfg["auto-revert"]
    required: dict[str, str | None] = {}
    for ap in reversed(undone):
        for path, old in (ap.get("before") or {}).items():
            required[path] = old
    conflicts = [p for ap in undone for p, new in (ap.get("after") or {}).items() if wf.disk(p) != new and _last_after(undone, p) == new]
    spinner.stop()

    def do_rewind() -> None:
        spinner.start("")
        try:
            wf.rewind_to(target)
        except ValueError as e:
            fail(f"Error rewinding plan: {e}")
        spinner.stop()
        if steps_back:
            out(f"✅ Rewound {steps_back} step{'s' if steps_back > 1 else ''} to {target}")
        else:
            out(f"✅ Rewound to {target}")
        out()

    def no_changes() -> None:
        out("\U0001F645‍♂️ No project files were modified")
        out()

    if not undone or not required or not should_revert:
        do_rewind()
        no_changes()
        print_cmds("log", "continue")
        return
    del idx
    files_word = "file" if len(required) == 1 else "files"
    if conflicts:
        out(f"⏪ {len(required)} local project {files_word} differ from target plan state")
        out()
        out(f"Reverting the {files_word} will make these changes locally \U0001F447")
        out()
        to_remove = [p for p, v in required.items() if v is None]
        to_add = [p for p, v in required.items() if v is not None and wf.disk(p) is None]
        to_update = [p for p, v in required.items() if v is not None and p not in to_add]
        if to_add:
            out("To add:")
            for p in sorted(to_add):
                out(f" \u2022 {p}")
            out()
        if to_remove:
            out("To remove:")
            for p in sorted(to_remove):
                out(f" • {p}")
            out()
        if to_update:
            out("To update:")
            for p in sorted(to_update):
                out(f" • {p}")
            out()
        s = " A project file has" if len(conflicts) == 1 else " These project files have"
        out(f"⚠️  {s} been updated outside of Pal since the latest apply:")
        for p in sorted(set(conflicts)):
            out(f" • {p}")
        out()
        out("If you revert, you will lose those changes.")
        out()
        try:
            pick = term.select("What do you want to do?", [f"Revert project {files_word} to match rewound plan state (overwrite changes)",
                                                           f"Rewind plan, but skip reverting project {files_word}", "Cancel rewind"])
        except term.Interrupted:
            raise Exit(0) from None
        if pick == "Cancel rewind":
            raise Exit(0)
        should_revert = pick.startswith("Revert")
    do_rewind()
    if not should_revert:
        no_changes()
        print_cmds("log", "continue")
        return
    spinner.start("")
    root = wf.session.cwd
    for path, old in required.items():
        p = root / path
        try:
            if old is None:
                p.unlink(missing_ok=True)
                d = p.parent
                while d != root and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                    d = d.parent
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(old, encoding="utf-8", newline="")
        except OSError as e:
            spinner.stop()
            fail(f"Error restoring file state: {e}")
    spinner.stop()
    n = len(required)
    out(f"⏪ {n} project {'file was' if n == 1 else 'files were'} reverted")
    for p in sorted(required):
        out(f" • {p}")
    out()
    from pal.terminal.workflow import git_commit, is_git_repo
    commit = (commit_flag or cfg["auto-commit"]) and not skip_commit_flag
    if commit and is_git_repo(root):
        msg = "\U0001F916 Pal → rewound plan state and reverted these changes:" + "".join(f"\n   • {ap.get('msg') or ', '.join(ap.get('files', []))}" for ap in undone)
        if not git_commit(root, sorted(required), msg):
            term.simple_error("Error committing changes: git refused the commit")
    print_cmds("log", "continue")


def _last_after(undone: list[dict], path: str):
    for ap in reversed(undone):
        if path in (ap.get("after") or {}):
            return ap["after"][path]
    return None


def cmd_convo(ctx, args: list[str]) -> None:
    wf = ctx.wf
    plain = "--plain" in args or "-p" in args
    args = [a for a in args if not a.startswith("-")]
    spinner.start("")
    convo = wf.plan.b.convo
    spinner.stop()
    if not convo:
        out(f"{SHRUG} No conversation history")
        return
    start, end = 1, len(convo)
    if args:
        a = args[0]
        if "-" in a:
            m = re.fullmatch(r"(\d+)-(\d*)", a)
            if not m:
                fail(f"Invalid message range: {a}")
            start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else len(convo)
        elif a.isdigit():
            start = end = int(a)
        else:
            fail(f"Invalid message number: {a}")
    cut = start > 1 or end < len(convo)
    parts = []
    today = time.localtime()
    for i, m in enumerate(convo, 1):
        if i < start:
            continue
        if i > end:
            break
        who = "\U0001F916 Pal" if m["role"] == "pal" else "\U0001F4AC You"
        lt = time.localtime(m.get("t", 0))
        hhmm = time.strftime("%I:%M%p", lt).lstrip("0").lower() + " " + _zone(lt)
        if lt.tm_mday == today.tm_mday:
            ts = "Today | " + hhmm
        elif lt.tm_mday == time.localtime(time.time() - 86400).tm_mday:
            ts = "Yesterday | " + hhmm
        else:
            ts = time.strftime("%a %b ", lt) + f"{lt.tm_mday}, {lt.tm_year} | " + hhmm
        tokens = int(m.get("tokens") or estimate_tokens(m.get("text", "")))
        tags = " | ".join(m.get("tags") or [])
        head = f"#### {i} | {who} | {tags} | {ts} | {tokens} {COIN} " if tags else f"#### {i} | {who} | {ts} | {tokens} {COIN} "
        parts.append(head + "\n" + m.get("text", "") + "\n\n")
    if plain:
        out("\n" + "".join(parts))
        return
    from pal.terminal.repl.stream import markdown
    w = term.term_width()
    body = "\n" + "\n".join(markdown(p, w) + "\n" for p in parts)
    if not cut:
        total = sum(int(m.get("tokens") or estimate_tokens(m.get("text", ""))) for m in convo)
        body += c(term.MUTED, "─" * w) + "\n" + accent("  Conversation size →") + f" {total} {COIN}" + "\n\n"
    changes.page(body)
    out()
    print_cmds("log")


def _zone(lt) -> str:
    z = time.strftime("%Z", lt)
    if " " in z:                                   # Windows: "India Standard Time" -> IST
        z = "".join(w[0] for w in z.split() if w)
    return z.split("/")[0]


def cmd_summary(ctx, args: list[str]) -> None:
    wf = ctx.wf
    spinner.start("")
    b = wf.plan.b
    spinner.stop()
    if not b.steps:
        out(f"{SHRUG} No summary available")
        return
    text = "### Plan\n\n" + "\n".join(f"{i}. {'[x]' if i <= b.done else '[ ]'} {s}" for i, s in enumerate(b.steps, 1))
    if "--plain" in args or "-p" in args:
        out(text)
        return
    from pal.terminal.repl.stream import markdown
    out(markdown(text, term.term_width()))


# -- config and auto-mode ---------------------------------------------------------------------------------------------------------------
def _visible(cfg: dict, name: str) -> bool:
    if name == "auto-debug":
        return bool(cfg["auto-exec"])
    if name == "auto-debug-tries":
        return bool(cfg["auto-debug"])
    if name == "auto-exec":
        return bool(cfg["can-exec"])
    return True


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def config_table(cfg: dict, key: str = "") -> str:
    rows: list[list[str]] = []
    for name, _typ, desc in CONFIG:
        if key and name != key:
            continue
        if not _visible(cfg, name):
            continue
        if rows:
            rows.append(["", "", ""])
        rows.append([name, _value(cfg[name]), desc])
    return term.table(rows, header=["Name", "Value", "Description"], autowrap=True)


def cmd_config(ctx, args: list[str]) -> None:
    if args[:1] == ["default"]:
        from pal.terminal.workflow import default_plan
        p = default_plan("default")
        out(accent("⚙️  Default Config"))
        out(config_table(p.config()))
        print_cmds("set-config default", "config", "set-config")
        return
    spinner.start("")
    cfg = ctx.wf.plan.config()
    spinner.stop()
    out(accent("⚙️  Plan Config"))
    out(config_table(cfg))
    print_cmds("set-config", "config default", "set-config default")


_TRUE = ("enabled", "true", "t", "yes", "y", "1")
_FALSE = ("disabled", "false", "f", "no", "n", "0")


def _norm(key: str) -> str:
    k = key.lower().replace("-", "")
    for name, _, _ in CONFIG:
        if name.replace("-", "") == k:
            return name
    return ""


def _update(plan: Plan, args: list[str]) -> str:
    if args:
        key = _norm(args[0])
        if not key:
            fail(f"Unknown setting: {args[0].lower().replace('-', '')}\n")
    else:
        opts = [f"{name} → {desc}" for name, _, desc in CONFIG]
        try:
            pick = term.select("Choose a setting to update:", opts)
        except term.Interrupted:
            raise Exit(0) from None
        key = pick.split(" →")[0]
    typ = next(t for n, t, _ in CONFIG if n == key)
    value = args[1] if len(args) > 1 else None
    if typ == "bool":
        if value is None:
            try:
                value = term.select(f"Set {key}:", ["Enabled", "Disabled"])
            except term.Interrupted:
                raise Exit(0) from None
            v = value == "Enabled"
        elif value in _TRUE:
            v = True
        elif value in _FALSE:
            v = False
        else:
            fail(f"Invalid value for {key} ({value})")
        plan.set(key, v)
    elif typ == "int":
        if value is None:
            try:
                value = term.ask_text(f"Set {key} (number)")
            except term.Interrupted:
                raise Exit(0) from None
        if not value.lstrip("-").isdigit():
            fail(f"Invalid number value for {key} ({value})")
        plan.set(key, int(value))
    elif key == "auto-mode":
        if value is None:
            try:
                pick = term.select("Set auto-mode:", [f"{label} → {desc}" for _, label, desc in AUTO_MODES])
            except term.Interrupted:
                raise Exit(0) from None
            value = next(k for k, label, _ in AUTO_MODES if pick.startswith(label))
        plan.set_auto(value)
    else:
        if value is None:
            try:
                value = term.ask_text("Enter the command to open the editor")
            except term.Interrupted:
                raise Exit(0) from None
        plan.set("editor", " ".join(args[1:]) if len(args) > 1 else value)
    return key


def cmd_set_config(ctx, args: list[str]) -> None:
    if args[:1] == ["default"]:
        from pal.terminal.workflow import default_plan
        p = default_plan("default")
        key = _update(p, args[1:])
        save_default_config(p)
        out("✅ Default config updated")
        out(config_table(p.config(), key))
        print_cmds("config default", "config", "set-config")
        return
    wf = ctx.wf
    spinner.start("")
    before = wf.plan.config()
    spinner.stop()
    key = _update(wf.plan, args)
    wf.save()
    after = wf.plan.config()
    out("✅ Config updated")
    out(config_table(after, key))
    warn = f"{term.HI_YELLOW};{BOLD}"
    if not (before["auto-apply"] and before["auto-exec"]) and after["auto-apply"] and after["auto-exec"]:
        out(c(warn, "⚠️  You enabled automatic apply and execution."))
        out()
    elif after["auto-apply"] and not before["auto-apply"]:
        out(c(warn, "⚠️  You enabled automatic apply."))
        out()
    elif after["auto-exec"] and not before["auto-exec"]:
        out(c(warn, "⚠️  You enabled automatic execution."))
        out()
    print_cmds("config", "config default", "set-config default")


def cmd_set_auto(ctx, args: list[str]) -> None:
    if args[:1] == ["default"]:
        cmd_set_config(ctx, ["default", "auto-mode"] + args[1:2])
        return
    cmd_set_config(ctx, ["auto-mode"] + args[:1])


# -- models ------------------------------------------------------------------------------------------------------------------------------
def cmd_models(ctx, args: list[str]) -> None:
    if args[:1] == ["available"]:
        return cmd_models_available(ctx, args[1:])
    s = ctx.session_or_boot()
    title = (green("Default") if args[:1] == ["default"] else green(ctx.wf.plan.name)) + " Model Settings"
    out(term.table([[title]]), end="")
    out()
    rows = []
    for role, mode in (("coder", "build"), ("planner", "plan"), ("chat", "ask")):
        prov, name = s.model_for(mode)
        rows.append([role, f"{prov}:{name}" if prov and prov != "local" else name])
    helper = getattr(s, "helper_model", "") or "none"
    rows.append(["helper", helper])
    out(c(f"{BOLD};{term.HI_CYAN}", "\U0001F916 Models"))
    out(term.table(rows, header=["Role", "Model"], align=["L", "L"]))
    print_cmds("set-model", "models available")
    return None


def cmd_models_available(ctx, args: list[str]) -> None:
    spinner.start("")
    choices = ctx.model_choices()
    spinner.stop()
    out(accent("\U0001F3E0 Local and API Models"))
    rows = [[label, about or ""] for label, about, spec in choices if not spec.startswith("\x00")]
    out(term.table(rows, header=["Model", "About"], align=["L", "L"]))
    print_cmds("models", "set-model")


def cmd_providers(ctx, args: list[str]) -> None:
    import os

    from pal.terminal import backend
    rows = [["local", "Ollama on this computer", "-", "yes"]]
    for prov, env in backend.PROVIDER_KEYS.items():
        rows.append([prov, prov.capitalize(), env, "yes" if os.environ.get(env) else "no"])
    out(accent("\U0001F3E0 Providers"))
    out(term.table(rows, header=["Provider", "Name", "Key", "Key set"], align=["L", "L", "L", "L"]))
    print_cmds("models", "set-model", "key")


def cmd_set_model(ctx, args: list[str]) -> None:
    default = args[:1] == ["default"]
    if default:
        args = args[1:]
    s = ctx.session_or_boot()
    old_p, old_m = s.model_for("build")
    old = f"{old_p}:{old_m}" if old_p and old_p != "local" else old_m
    if args:
        spec = args[0]
    else:
        spinner.start("")
        choices = ctx.model_choices()
        spinner.stop()
        labels = [f"{label}" + (f"  ({about})" if about else "") for label, about, _ in choices]
        try:
            pick = term.select("Select a model:", labels)
        except term.Interrupted:
            raise Exit(0) from None
        spec = choices[labels.index(pick)][2]
        if spec.startswith("\x00"):
            try:
                spec = term.ask_text("Model (provider:model, for example openai:gpt-4o)")
            except term.Interrupted:
                raise Exit(0) from None
    problem = ctx.switch_model(spec)
    if problem:
        out(c(f"{RED_HI};{BOLD}", f"\U0001F6A8 {problem}"))
        out()
        print_cmds("models available")
        raise Exit(1)
    p, m = ctx.session_or_boot().model_for("build")
    new = f"{p}:{m}" if p and p != "local" else m
    if new == old:
        out(f"{SHRUG} No model settings were updated")
        return
    out(f"⚙️  Updated {'default ' if default else ''}model settings:")
    out(f"  • model | {old} → {new}")
    out()
    print_cmds("models", "log")


# -- streams ----------------------------------------------------------------------------------------------------------------------------
def cmd_ps(ctx, args: list[str]) -> None:
    from pal.terminal.repl import stream
    spinner.start("")
    rows = []
    for st in reversed(stream.ACTIVE):
        if st.done.is_set() and time.time() - st.ended > 3600:
            continue
        status = {"active": "Active", "finished": "Finished", "stopped": "Stopped", "error": "Error"}[st.status]
        if st.status != "active":
            status += " " + term.rel_time(st.ended)
        rows.append([c(BOLD, st.id[:4]), st.plan, st.branch, term.rel_time(st.started), status])
    spinner.stop()
    if not rows:
        out(f"{SHRUG} No active or recently finished streams")
        return
    out(term.table(rows, header=["Pid", "Plan", "Branch", "Started", "Status"]), end="")
    out()
    print_cmds("connect", "stop")


def cmd_stop(ctx, args: list[str]) -> None:
    from pal import steps
    from pal.terminal.repl import stream
    st = stream.active()
    if st is None:
        out(f"{SHRUG} No active plan stream")
        out()
        print_cmds("ps")
        return
    spinner.start("")
    steps.request_stop()
    st.stopped = True
    st.done.wait(3)
    spinner.stop()
    out("✅ Plan stream stopped")
    out()
    print_cmds("convo", "log")


def cmd_connect(ctx, args: list[str]) -> None:
    from pal.terminal.repl import stream
    st = stream.active()
    if st is None:
        out(f"{SHRUG} No active plan stream")
        out()
        print_cmds("ps")
        return
    ctx.follow(st, mode="build")
