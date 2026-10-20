"""How much does compressing old assistant replies save, and does it keep exact details?

A SCRIPTED session (no real model): an agent iterating a README through several drafts and editing code, with a budget figure
mentioned in turn 1 and asked about at the end in other words. The replies are written to look like real agent output (long drafts, long code), so the numbers show the effect
of the method on this kind of session, not what a real model would produce. Three ways of handling the history are compared on
the same turns:
  full         every earlier turn is sent (what a plain chat client does)
  paging       the last turns, plus older ones only when relevant (what Pal did before compression)
  facts only   paging, plus a facts block with the exact details of older turns
  compression  paging, plus compressed older replies and the facts block (the context manager now)

Run:  python experiments/compression_bench.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

from pal.chat import ChatService  # noqa: E402
from pal.config import load_settings  # noqa: E402
from pal.llm_wrapper.components.base import LLMWrapper, LLMResponse  # noqa: E402
from pal.tokens import estimate_tokens  # noqa: E402


def readme(rev: int) -> str:
    return (f"# Papertrail\n\n## Install\n\nRun pip install papertrail, then create config.yaml in your project folder. Revision {rev} "
            "adds a short note about virtual environments and why they keep the install clean, which most readers appreciate.\n\n"
            f"## Usage\n\nCall search with a query string and read the ranked results. Revision {rev} explains the ranking in two "
            "extra sentences and links the example notebook for people who want to see the whole flow end to end.\n\n"
            f"## Configuration\n\nSet index_path and the number of results in config.yaml. Revision {rev} documents every option "
            "with its default value and an example, so nobody has to read the source to find out what a setting does.\n\n"
            f"## License\n\nMIT. See the LICENSE file. Revision {rev} adds the contributing guide and the code of conduct.")


def code(rev: int) -> str:
    body = "".join(f"    part_{k} = normalise_{k}(item) + weigh_{k}(item, rev={rev})\n" for k in range(16))
    return (f"Here is revision {rev} of the ranking function.\n```python\ndef rank_results(item, config):\n{body}    return part_0\n```\n"
            f"It uses config.yaml and returns scores in 0.{rev}0 steps. Use this version because the old one failed with KeyError "
            "on empty queries. Hope it helps with the next change.")


def session(n: int) -> tuple[list[str], list[str]]:
    """n turns: a README written and rewritten, and every third turn a code change. The budget is mentioned only in turn 1."""
    users = ["my budget for this project is 500 dollars, please write a readme for papertrail"]
    replies = [readme(1)]
    for i in range(2, n + 1):
        users.append(f"change the ranking function, version {i}" if i % 3 == 0 else f"improve the readme again, attempt {i}")
        replies.append(code(i) if i % 3 == 0 else readme(i))
    return users, replies + ["You can spend 500 dollars."]


# a question that shares no words with the sentence that holds the figure, so keyword recall cannot find that turn
QUESTION = "how much money can we spend on this?"


class Local(LLMWrapper):
    """Stands in for the local model: routes everything to the API and leaves the message as it is."""

    name = "local"

    def __init__(self):
        pass

    def chat(self, syscall):
        system = syscall.messages[0]["content"]
        if system.startswith("You route"):
            return LLMResponse(text='{"route": "api", "reason": "needs the big model"}')
        user = syscall.messages[-1]["content"]
        return LLMResponse(text=user if "<text>" in user else '{"ok": true, "lost": []}')

    def token_count(self, text):
        return estimate_tokens(text)

    def capabilities(self):
        return None


class Api(LLMWrapper):
    name = "api"

    def __init__(self):
        self.seen, self.i = [], 0

    def chat(self, syscall):
        self.seen.append(syscall.messages)
        text = REPLIES[min(self.i, len(REPLIES) - 1)]
        self.i += 1
        n = sum(estimate_tokens(m["content"]) for m in syscall.messages)
        return LLMResponse(text=text, prompt_tokens=n, completion_tokens=estimate_tokens(text))

    def token_count(self, text):
        return estimate_tokens(text)

    def capabilities(self):
        return None


def run(mode: str, n: int) -> tuple[list[int], str]:
    global REPLIES
    ctx = {"full": {"hot_turns": 100, "max_history_tokens": 10**6, "total_budget_tokens": 10**6, "compress": False},
           "paging": {"compress": False}, "facts only": {"compress_turns": 0}, "compression": {}}[mode]
    home = tempfile.mkdtemp()
    settings = load_settings(env={}, overrides={"storage_dir": home, "knowledge_graph": {"dir": home + "/g"}, "context": ctx,
                                               "scheduler": {"enabled": False}, "memory": {"enabled": False}})
    users, REPLIES = session(n)
    api = Api()
    svc = ChatService(settings, local_core=Local(), verifier_core=Local())
    svc.api_core = lambda provider, model, key=None: api
    for msg in users + [QUESTION]:
        svc.send("bench", "anthropic", "m", "k", msg, [], route="api")
    svc.close()
    tokens = [sum(estimate_tokens(m["content"]) for m in call) for call in api.seen]
    last = " ".join(m["content"] for m in api.seen[-1])
    return tokens, last


def main() -> None:
    for n in (14, 34):
        rows = {m: run(m, n) for m in ("full", "paging", "facts only", "compression")}
        print(f"=== a session of {n} turns, then one question (input tokens sent to the API model, estimate = chars / 4) ===")
        if n <= 14:
            print(f"{'turn':>4} | {'full':>6} | {'paging':>6} | {'facts only':>10} | {'compression':>11}")
            for i in range(n + 1):
                print(f"{i + 1:>4} | {rows['full'][0][i]:>6} | {rows['paging'][0][i]:>6} | {rows['facts only'][0][i]:>10} | {rows['compression'][0][i]:>11}")
        tot = {m: sum(t) for m, (t, _) in rows.items()}
        last = {m: t[-1] for m, (t, _) in rows.items()}
        print("total over the session: " + ", ".join(f"{m} {v:,}" for m, v in tot.items()))
        print("last request:           " + ", ".join(f"{m} {v:,}" for m, v in last.items()))
        for m in ("facts only", "compression"):
            print(f"{m} vs full: {100 * (1 - tot[m] / tot['full']):.0f}% fewer tokens in total, "
                  f"{100 * (1 - last[m] / last['full']):.0f}% fewer on the last request;  vs paging only: {100 * (tot[m] / tot['paging'] - 1):+.0f}% in total")
        print('does the last request still hold the figure from turn 1 ("500 dollars")?  ' + ",  ".join(
            f"{m}: {'yes' if '500 dollars' in text else 'NO'}" for m, (_, text) in rows.items()))
        print()


if __name__ == "__main__":
    main()
