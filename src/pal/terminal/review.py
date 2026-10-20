"""/review: the pending changes as a page in your browser, in Pal's colours. A plain file on this computer: no server, nothing online.

Every text from the model is escaped before it goes in the page, and the page carries a strict content security policy (no scripts at all),
so a change that contains HTML is shown as text and cannot run.
"""
from __future__ import annotations

import html
import time
import webbrowser
from pathlib import Path

from pal.paths import pal_home
from pal.terminal.workflow import Workflow

CSS = """
:root{color-scheme:dark;--bg:#000;--panel:#121212;--line:#262626;--ink:#e8e8e8;--muted:#8a8a8a;--red:#E50914;--red-hi:#FF4D57;--add:#1d2b1f;--del:#2b1214}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}
header{position:sticky;top:0;background:#0a0a0a;border-bottom:1px solid var(--line);padding:14px 24px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:18px}header h1 b{color:var(--red)}header span{color:var(--muted);font-size:13px}
.wrap{display:grid;grid-template-columns:260px minmax(0,1fr);gap:0}
nav{border-right:1px solid var(--line);padding:16px 12px;position:sticky;top:56px;align-self:start;max-height:calc(100vh - 56px);overflow:auto}
nav a{display:block;color:var(--ink);text-decoration:none;font:13px ui-monospace,Menlo,Consolas,monospace;padding:5px 8px;border-radius:6px;overflow-wrap:anywhere}
nav a:hover{background:var(--panel)}nav a i{font-style:normal;color:var(--red-hi);margin-right:6px}
main{padding:16px 24px 60px;min-width:0}
section{border:1px solid var(--line);border-radius:10px;margin:0 0 20px;overflow:hidden}
section h2{margin:0;padding:10px 14px;background:var(--panel);font:600 13px ui-monospace,Menlo,Consolas,monospace;border-bottom:1px solid var(--line)}
section h2 small{color:var(--muted);font-weight:400;margin-left:8px}
.find{padding:8px 14px;border-bottom:1px solid var(--line);font-size:13px}.find b{color:#fff;background:var(--red);padding:0 6px;border-radius:4px;margin-right:6px;font-size:11px}
pre{margin:0;padding:8px 0;overflow-x:auto;font:12.5px/1.55 ui-monospace,Menlo,Consolas,monospace}
pre span{display:block;padding:0 14px;white-space:pre}.a{background:var(--add);color:#cfe9d3}.d{background:var(--del);color:#ffc2c6}.h{color:var(--muted)}.c{color:#a8a8a8}
footer{color:var(--muted);font-size:13px;padding:0 24px 24px}code{color:var(--ink)}
@media (max-width:800px){.wrap{grid-template-columns:1fr}nav{position:static;max-height:none;border-right:0;border-bottom:1px solid var(--line)}}
"""


def render(wf: Workflow) -> str:
    diffs = wf.diffs()
    esc = html.escape
    nav = "".join(f'<a href="#f{i}"><i>{"+" if new else "~"}</i>{esc(path)}</a>' for i, (path, new, _, _) in enumerate(diffs))
    parts = []
    for i, (path, new, lines, findings) in enumerate(diffs):
        body = "".join(f'<span class="{ {"add": "a", "del": "d", "hunk": "h"}.get(tag, "c") }">{esc(("+ " if tag == "add" else "- " if tag == "del" else "" if tag == "hunk" else "  ") + text)}</span>' for tag, text in lines)
        found = "".join(f'<div class="find"><b>{esc(f.severity.upper())}</b>{esc(f.title)} (line {f.line}): {esc(f.why)}</div>' for f in findings)
        parts.append(f'<section id="f{i}"><h2>{esc(path)}<small>{"new file" if new else "changed"}</small></h2>{found}<pre>{body}</pre></section>')
    cmds = wf.plan.b.commands
    empty = "<p>Nothing is pending.</p>" if not diffs else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>Pal review: {esc(wf.plan.name)}</title><style>{CSS}</style></head><body>
<header><h1><b>Pal</b> review</h1><span>plan {esc(wf.plan.name)} · branch {esc(wf.plan.branch)} · {len(diffs)} file{'s' if len(diffs) != 1 else ''} pending · {time.strftime('%H:%M')}</span></header>
<div class="wrap"><nav>{nav or '<span>No files</span>'}</nav><main>{empty}{''.join(parts)}</main></div>
<footer>Back in the terminal: <code>/apply</code> writes these changes{', then runs ' + esc(', '.join(cmds)) + ' if you approve' if cmds else ''}; <code>/reject &lt;file&gt;</code> drops one; <code>/rewind</code> goes back a step.</footer>
</body></html>"""


def open_review(wf: Workflow, open_browser=webbrowser.open) -> Path:
    out = pal_home() / "review" / f"{wf.plan.name}-{wf.plan.branch}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(wf), encoding="utf-8")
    try:
        open_browser(out.resolve().as_uri())
    except Exception:  # noqa: BLE001  (no browser: the path is shown instead)
        pass
    return out
