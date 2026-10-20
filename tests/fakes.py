"""Shared test doubles. No test talks to a real model, the network, or a real graphify unless it says so."""
from __future__ import annotations

import re

from pal.llm_wrapper.components.base import DeploymentType, LLMResponse, LLMWrapper, WrapperCapabilities


class FakeLocal(LLMWrapper):
    """Plays both applier and verifier. `edits` maps method rule keyword -> fn(text)."""
    name = "fake"

    def __init__(self, verdict='{"ok": true, "lost": []}', edits=None, boom=False):
        super().__init__(None)
        self.verdict, self.boom, self.calls, self.seen = verdict, boom, [], []
        self.edits = {
            "articles": lambda t: re.sub(r"\b(a|an|the)\s+", "", t, flags=re.I),
            "punctuation": lambda t: re.sub(r"[.,;:!?\"'\-]", "", t),
            "filler": lambda t: re.sub(r"\b(please|just|really|basically)\s+", "", t),
            "shorter": lambda t: t.replace("information", "info"),
            "rules": lambda t: re.sub(r"\b(the|a|an|please|just)\s+", "", t, flags=re.I),
            "whitespace": lambda t: re.sub(r"[ \t]+", " ", t).strip(),
        }
        self.edits.update(edits or {})

    def chat(self, syscall):
        sys_, user = syscall.messages[0]["content"], syscall.messages[1]["content"]
        self.seen.append(user)
        self.calls.append("verify" if sys_.startswith("You audit") else "apply")
        if sys_.startswith("You audit"):
            return LLMResponse(text=self.verdict)
        if self.boom:
            raise ConnectionError("ollama down")
        text = re.search(r"<text>\n(.*)\n</text>", user, re.S).group(1)
        rule = re.search(r"RULE: (.*)\n<text>", user)
        if rule is None:
            return LLMResponse(text=f"<text>\n{self.edits['rules'](text)}\n</text>")
        rule = rule.group(1)
        key = ("articles" if "articles" in rule else "punctuation" if "punctuation marks" in rule
               else "filler" if "filler" in rule else "shorter" if "shorter" in rule else "whitespace")
        return LLMResponse(text=f"<text>\n{self.edits[key](text)}\n</text>")

    def token_count(self, text):
        return len(text.split())

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


class RouterLocal(FakeLocal):
    """Fake local LLM: the classifier says `route`, the answerer says `answer`."""

    def __init__(self, route='{"route": "local", "reason": "simple math"}', answer="4", boom=False):
        super().__init__()
        self.route_reply, self.answer_reply, self.route_boom = route, answer, boom
        self.classified, self.answered = [], []

    def chat(self, syscall):
        sys_ = syscall.messages[0]["content"]
        if sys_.startswith("You route"):
            self.classified.append(syscall.messages[-1]["content"])
            if self.route_boom:
                raise ConnectionError("ollama down")
            return LLMResponse(text=self.route_reply)
        if sys_.startswith("You answer"):
            self.answered.append(syscall.messages)
            return LLMResponse(text=self.answer_reply, prompt_tokens=7, completion_tokens=2)
        return super().chat(syscall)


class _Cfg:
    api_key = "sk-test-secret-key-000000"


class FakeAPI:
    """Fake API model. `replies` are returned in order (the last one repeats); each call is recorded."""

    def __init__(self, replies=None, boom=None):
        self.seen = []
        self.replies = list(replies or ["api reply"])
        self.boom = boom
        self.config = _Cfg()

    def chat(self, syscall):
        self.seen.append(syscall.messages)
        if self.boom:
            raise self.boom
        i = min(len(self.seen) - 1, len(self.replies) - 1)
        return LLMResponse(text=self.replies[i], prompt_tokens=10, completion_tokens=3)

    def token_count(self, text):
        return len(text.split())


class FakeRunner:
    """Stands in for graphify: same command line, no real parsing. Writes a tiny graph.json."""

    def __init__(self, nodes=None, fail=False):
        self.calls, self.fail = [], fail
        self.nodes = nodes if nodes is not None else [("checkout()", "src/shop/pay.py", 4)]

    def __call__(self, args, cwd):
        from pathlib import Path
        self.calls.append(args)
        if self.fail:
            return 1, "", "boom"
        if args[0] == "update":
            out = Path(cwd) / "src" / "graphify-out"
            out.mkdir(parents=True, exist_ok=True)
            (out / "graph.json").write_text(
                '{"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b"}]}', encoding="utf-8")
            return 0, "", ""
        if args[0] == "god-nodes":
            return 0, '[{"label": "Cart", "degree": 3}]', ""
        if args[0] == "query":
            return 0, "Graph: x\n\n" + "\n".join(
                f"NODE {label} [src={src} loc=L{n} community=c]" for label, src, n in self.nodes), ""
        return 1, "", "unknown"


class FakeLiteLLM:
    """A stand-in for the `litellm` module: records every call, returns scripted replies, no network, no install needed."""

    def __init__(self, reply="ok", tool_calls=None, usage=(10, 3), error=None, cost=0.0001):
        self.calls, self.reply, self.tool_calls, self.usage, self.error, self.cost = [], reply, tool_calls, usage, error, cost
        self.model_cost = {"anthropic/claude-x": {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6}}
        self.closed = 0

    def completion(self, **kw):
        from types import SimpleNamespace as NS
        self.calls.append(kw)
        if self.error:
            raise self.error
        if kw.get("stream"):
            return self._stream()
        calls = [NS(function=NS(name=n, arguments=a)) for n, a in (self.tool_calls or [])]
        msg = NS(content=self.reply, tool_calls=calls or None)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=self.usage[0], completion_tokens=self.usage[1]))

    def _stream(self):
        from types import SimpleNamespace as NS
        outer = self

        class It:
            def __iter__(self):
                for piece in ("str", None, "eam"):
                    yield NS(choices=[NS(delta=NS(content=piece))])

            def close(self):
                outer.closed += 1
        return It()

    def get_model_info(self, model):
        """Like litellm.get_model_info: exact, provider-prefixed names only (it raises for a model it does not know)."""
        if model not in self.model_cost:
            raise ValueError(f"this model isn't mapped yet: {model}")
        return self.model_cost[model]

    def completion_cost(self, completion_response=None):
        if self.cost is None:
            raise ValueError("no price for this model")
        return self.cost

    def token_counter(self, model, text):
        return len(text.split())

    def supports_function_calling(self, model):
        return True

    def get_supported_openai_params(self, model):
        return ["temperature", "response_format", "tools"]
