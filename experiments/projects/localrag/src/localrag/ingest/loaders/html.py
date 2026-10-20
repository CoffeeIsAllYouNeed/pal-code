"""HTML pages, reduced to text with headings kept as markdown so that heading-aware chunking and the document tree can use them."""
from __future__ import annotations

from html.parser import HTMLParser

from localrag.core.textutil import normalize_text
from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes

_SKIP = {"script", "style", "noscript", "template", "svg"}
_BLOCK = {"p", "div", "section", "article", "ul", "ol", "table", "tr", "br", "header", "footer", "blockquote", "pre"}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
        else:
            self.parts.append(data)


class HtmlLoader(Loader):
    extensions = (".html", ".htm", ".xhtml")

    def load(self, raw: bytes, name: str) -> LoadedText:
        parser = _Extractor()
        parser.feed(decode_bytes(raw, name))
        parser.close()
        text = normalize_text("".join(parser.parts))
        return LoadedText(text, {"format": "html", "title": parser.title.strip() or name})
