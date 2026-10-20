"""The AIOS architecture diagram with the renamed and split components (LLM Wrapper, Local LLM, LLM Scheduler, Context Manager with
Knowledge Graph and Compression, Notes Manager, and the Token / Latency / Memory trackers). Same dark, colour-coded design as before."""
# ruff: noqa: I001
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, BAND, BAND_EDGE = "#0B1020", "#111936", "#26315F"
TEXT, MUTED = "#E8ECFA", "#9AA7D6"
VIOLET, TEAL, AMBER, BLUE = "#9D8CFF", "#2DD4BF", "#F5B14C", "#7AA2F7"
ARROW = "#B4BEE6"

W, H = 160, 122
fig, ax = plt.subplots(figsize=(16, 12.2), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal"), ax.axis("off")
ax.set_position([0, 0, 1, 1])


def box(x, y, w, h, fc, ec, lw=1.4, r=1.2, alpha=1.0, shadow=False, z=2, ls="-"):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.35, y - 0.35), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="#04070F", ec="none", alpha=0.55, zorder=z - 0.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, linestyle=ls))


def label(x, y, s, size=10, color=TEXT, weight="normal", ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, family="DejaVu Sans", **kw)


def arrow(x1, y1, x2, y2, color=ARROW, lw=1.7, both=True, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=6,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=lw, mutation_scale=13, shrinkA=0, shrinkB=0,
                                linestyle=(0, (2.2, 2.2)) if dashed else "-"))


def chip(x, y, w, h, text, accent, size=9.5, caption=None, fc="#0D1430", dot=False, cap_size=5.6):
    box(x, y, w, h, fc, accent, lw=1.5, r=1.0, shadow=True, z=3)
    if dot:
        ax.add_patch(plt.Circle((x + 2.2, y + h / 2), 0.75, fc=accent, ec="none", zorder=5))
    cx = x + (w + (2.0 if dot else 0)) / 2
    if caption:
        label(cx, y + h * 0.66, text, size, TEXT, "bold")
        label(cx, y + h * 0.27, caption, cap_size, MUTED)
    else:
        label(cx, y + h / 2, text, size, TEXT)


# -- the three layers ----------------------------------------------------------------------------------------------------------------
for (y0, y1, accent, name) in [(1.5, 14.5, BLUE, "HARDWARE LAYER"), (18, 92, TEAL, "KERNEL LAYER"), (96, 120, VIOLET, "APPLICATION LAYER")]:
    box(2, y0, 156, y1 - y0, BAND, BAND_EDGE, lw=1.2, r=2.0, z=1)
    ax.add_patch(FancyBboxPatch((2, y0 + 1.2), 0.7, y1 - y0 - 2.4, boxstyle="round,pad=0,rounding_size=0.35", fc=accent, ec="none", zorder=2))
    label(156.2, (y0 + y1) / 2, name, 8, accent, "bold", rotation=270)

# -- application layer (unchanged, moved up) ---------------------------------------------------------------------------------------------
D = 32
label(8, 84.3 + D, "Agent Application", 13, VIOLET, "bold", ha="left")
agents = ["Travel Agent", "Rec Agent", "Coding Agent", "Math Agent", "Narrative Agent"]
dots = ["#F472B6", "#FB923C", "#60A5FA", "#34D399", "#C084FC"]
for i, (name, c) in enumerate(zip(agents, dots)):
    chip(56 + i * 20, 76.5 + D, 18, 8, name, c, 9.5, dot=True)
box(56, 67.5 + D, 66, 6, "#1B2350", VIOLET, lw=1.8, r=1.2, shadow=True, z=3)
label(89, 70.5 + D, "AIOS-Agent SDK", 12, TEXT, "bold")
arrow(89, 76.5 + D, 89, 73.5 + D, VIOLET)
box(8, 65.8 + D, 40, 9.4, "#0D1430", AMBER, lw=1.3, r=1.0, shadow=True, z=3)
label(28, 72.4 + D, "Non-LLM queries", 10, AMBER, "bold")
label(28, 69.3 + D, "File operation  ·  Network request", 8, MUTED)
label(28, 67.2 + D, "Keyboard / Mouse", 8, MUTED)
box(128, 65.8 + D, 26, 9.4, "#0D1430", TEAL, lw=1.3, r=1.0, shadow=True, z=3)
label(141, 72.4 + D, "LLM queries", 10, TEAL, "bold")
label(141, 69.4 + D, "Reasoning & planning", 8, MUTED)
label(141, 67.3 + D, "Tool parsing · Code generation", 7.2, MUTED)
arrow(56, 70.5 + D, 48, 70.5 + D, AMBER, both=False)
arrow(122, 70.5 + D, 128, 70.5 + D, TEAL, both=False)
arrow(28, 65.8 + D, 28, 90, AMBER, both=False)
arrow(141, 65.8 + D, 141, 90, TEAL, both=False)

# -- kernel layer: OS kernel ---------------------------------------------------------------------------------------------------------
box(6, 20, 44, 70, "#15152B", AMBER, lw=1.8, r=1.6, z=2)
label(28, 86.3, "OS Kernel", 12, AMBER, "bold")
for i, name in enumerate(["Process Scheduler", "Memory Manager", "Filesystem", "Hardware Driver"]):
    chip(9, 74 - i * 13, 38, 7, name, AMBER, 9)

# -- kernel layer: AIOS kernel -------------------------------------------------------------------------------------------------------
box(56, 29, 98, 61, "#0F2230", TEAL, lw=1.8, r=1.6, z=2)
label(105, 86.7, "AIOS Kernel", 12, TEAL, "bold")
box(60, 78, 90, 5.4, "#17475A", TEAL, lw=1.6, r=1.0, shadow=True, z=3)
label(105, 80.7, "AIOS System Call", 10.5, TEXT, "bold")

CARD_Y, CARD_H = 44, 31
cards = [  # (title, accent, x, width)
    ("MODEL ACCESS", "#F472B6", 60.0, 29.0), ("CONTEXT", "#FACC15", 90.2, 22.0), ("NOTES & MEMORY", "#FB923C", 113.4, 22.0), ("TOOLS & ACCESS", "#60A5FA", 136.6, 13.4)]
centers = []
for title, accent, x, w in cards:
    box(x, CARD_Y, w, CARD_H, "#0C1A28", accent, lw=1.3, r=1.2, z=2.5)
    label(x + w / 2, CARD_Y + CARD_H - 2.1, title, 7.4, accent, "bold")
    centers.append((x + w / 2, accent))
    arrow(x + w / 2, 78, x + w / 2, CARD_Y + CARD_H, accent, lw=1.5)

# model access
x, w = 60.0, 29.0
cw = w - 2.4
cx0 = x + 1.2
box(cx0, 60.8, cw, 10.6, "#0D1430", "#F472B6", lw=1.5, r=1.0, shadow=True, z=3)
label(cx0 + cw / 2, 69.6, "LLM Wrapper", 8.8, TEXT, "bold")
label(cx0 + cw / 2, 67.0, "own key  ·  explore models", 5.6, MUTED)
for j, (nm, c) in enumerate([("LiteLLM", "#F9A8D4"), ("OpenRouter", "#FDA4AF")]):
    mw = (cw - 3.0) / 2
    mx = cx0 + 1.0 + j * (mw + 1.0)
    box(mx, 61.9, mw, 3.7, "#150E22", c, lw=1.1, r=0.7, z=4)
    label(mx + mw / 2, 63.75, nm, 6.6, TEXT, "bold")
chip(cx0, 53.2, cw, 6.8, "Local LLM", "#FB7185", 8.8, caption="Ollama · vLLM · LM Studio · llama.cpp", cap_size=4.9)
chip(cx0, 46.0, cw, 6.4, "LLM Scheduler", "#34D399", 8.8, caption="API line  ·  local priority  ·  code slots", cap_size=4.9)

# context
x, w = 90.2, 22.0
cw = w - 2.4
chip(x + 1.2, 62.0, cw, 9.2, "Context Manager", "#FACC15", 8.6, caption="turns · one token budget")
chip(x + 1.2, 53.2, cw, 7.4, "Knowledge Graph", "#A3E635", 8.6, caption="code excerpts")
chip(x + 1.2, 45.4, cw, 7.0, "Compression", "#38BDF8", 8.6, caption="old replies · facts")

# notes and memory
x, w = 113.4, 22.0
cw = w - 2.4
chip(x + 1.2, 62.0, cw, 9.2, "Notes Manager", "#FBBF24", 8.6, caption="keep · recall · tidy")
chip(x + 1.2, 53.2, cw, 7.4, "Memory Manager", "#FB923C", 8.6, caption="RAM · LRU-K")
chip(x + 1.2, 45.4, cw, 7.0, "Storage Manager", "#C084FC", 8.6, caption="files · vectors")

# tools and access
x, w = 136.6, 13.4
cw = w - 2.4
chip(x + 1.2, 58.6, cw, 10.0, "Tool Manager", "#60A5FA", 6.6, caption="34 tools", cap_size=5.4)
chip(x + 1.2, 46.6, cw, 10.0, "Access Manager", "#2DD4BF", 6.6, caption="groups · confirm", cap_size=5.0)

# trackers
box(60, 31, 90, 10.5, "#0F1A33", "#7C8DD6", lw=1.3, r=1.2, z=2.5, ls=(0, (4, 2)))
label(61.6, 38.6, "TRACKERS", 7.4, "#AAB6EE", "bold", ha="left")
label(61.6, 35.2, "measure every request", 5.8, MUTED, ha="left")
for i, (nm, cap, c) in enumerate([("Token Tracker", "tokens · savings · cost", "#A78BFA"), ("Latency Tracker", "where the time goes", "#22D3EE"),
                                 ("Memory Tracker", "RAM · notes · process", "#FDBA74")]):
    chip(82 + i * 22.5, 32.5, 21.0, 7.5, nm, c, 8.0, caption=cap, cap_size=5.4)
for cx, _c in centers:
    arrow(cx, 41.5, cx, CARD_Y, "#7C8DD6", lw=1.1, both=False, dashed=True)

# -- OS system call bar and the links ---------------------------------------------------------------------------------------------------
box(56, 20, 98, 5.4, "#1F1B33", AMBER, lw=1.6, r=1.0, shadow=True, z=3)
label(105, 22.7, "OS System Call", 10.5, TEXT, "bold")
arrow(105, 29, 105, 25.4, ARROW)
arrow(50, 22.7, 56, 22.7, AMBER)

# -- hardware layer ------------------------------------------------------------------------------------------------------------------------
box(8, 4, 40, 8, "#14213D", BLUE, lw=1.8, r=1.2, shadow=True, z=3)
label(28, 8, "Hardware", 12, BLUE, "bold")
arrow(28, 20, 28, 12, BLUE)
spec = [("CPU", 14), ("GPU", 14), ("Memory", 16), ("Disk", 14), ("Peripheral Devices", 28)]
for i, (name, w) in enumerate(spec):
    chip(56 + sum(ww + 2.4 for _, ww in spec[:i]), 5, w, 6, name, BLUE, 9.5)

fig.savefig("aios_components.png", facecolor=BG, dpi=200)
fig.savefig("aios_components.svg", facecolor=BG)
print("ok")
