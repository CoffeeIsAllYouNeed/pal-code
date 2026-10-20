"""Builders used across the chat, router, graph and web tests."""
from __future__ import annotations

import io
import zipfile

from pal.chat import ChatService
from pal.config import load_settings
from pal.knowledge_graph import KnowledgeGraph
from pal.web import create_app
from tests.fakes import FakeAPI, FakeRunner, RouterLocal

SHOP = {
    "shop/cart.py": "class Cart:\n    def __init__(self):\n        self.items = []\n\n"
                    "    def add(self, item, qty):\n        self.items.append((item, qty))\n\n"
                    "    def total(self, prices):\n        return sum(prices[i] * q for i, q in self.items)\n",
    "shop/pay.py": "from shop.cart import Cart\n\n\ndef checkout(cart: Cart, prices):\n"
                   "    amount = cart.total(prices)\n    return charge_card(amount)\n\n\n"
                   "def charge_card(amount):\n    return {'charged': amount}\n",
    "shop/unrelated.py": "def weather():\n    return 'sunny'\n",
}


def make_zip(files=None, wrapper="", extra=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, t in {**(files or SHOP), **(extra or {})}.items():
            z.writestr(wrapper + n, t)
    return buf.getvalue()


def make_settings(tmp_path, **overrides):
    base = {"storage_dir": str(tmp_path / "home"), "knowledge_graph": {"dir": str(tmp_path / "graphs"), "budget_tokens": 800}}
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return load_settings(env={}, overrides=base)


def make_service(tmp_path, local=None, api=None, runner=None, **overrides):
    """A ChatService wired to fakes: fake local LLM, fake API model, fake graphify."""
    settings = make_settings(tmp_path, **overrides)
    graph = KnowledgeGraph(settings.knowledge_graph, runner or FakeRunner())
    svc = ChatService(settings, local_core=local or RouterLocal(), graph=graph)
    api = api or FakeAPI()
    svc.api_core = lambda *a, **k: api
    return svc, api


def send(svc, msg, atts=None, route=None, sid="session-1", **kw):
    return svc.send(sid, "anthropic", "m", "k", msg, atts or [], None, mode="rules", route=route, **kw)


def doc(name="notes.md", text="text", **kw):
    return {"name": name, "text": text, "kind": "document", **kw}


def client(svc):
    return create_app(service=svc).test_client()
