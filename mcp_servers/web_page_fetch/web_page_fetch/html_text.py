"""Bounded HTML decoding and text extraction, without a browser or XML parser."""

import codecs
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
import re
import time

from .errors import ErrorCode as E, FetchError

_SKIP = {"script", "style", "noscript", "template"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_BLOCK = {"article", "main", "body", "div", "p", "section", "header", "footer", "nav", "aside", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "th", "blockquote", "pre", "br", "hr"}


def _charset(content_type: str) -> str | None:
    message = Message()
    message["content-type"] = content_type
    return message.get_content_charset()


class _MetaParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.labels: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        values = dict(attrs)
        if values.get("charset"):
            self.labels.append(values["charset"])
        elif (values.get("http-equiv") or "").lower() == "content-type":
            label = _charset(values.get("content") or "")
            if label:
                self.labels.append(label)


def decode_html(data: bytes, content_type: str) -> str:
    # UTF-32 LE starts with the UTF-16 LE prefix: order is significant.
    for bom, encoding in ((codecs.BOM_UTF32_LE, "utf-32"),
                          (codecs.BOM_UTF32_BE, "utf-32"),
                          (codecs.BOM_UTF8, "utf-8-sig"),
                          (codecs.BOM_UTF16_LE, "utf-16"),
                          (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    meta = _MetaParser()
    meta.feed(data[:1024].decode("latin-1"))
    labels = [_charset(content_type), *meta.labels, "utf-8"]
    for label in labels:
        if not label:
            continue
        try:
            codecs.lookup(label)
            return data.decode(label, errors="replace")
        except (LookupError, ValueError, TypeError):
            continue
    return data.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class ExtractedText:
    title: str
    content: str
    truncated: bool


class ReadableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.regions = {name: [] for name in ("article", "main", "body", "fallback")}
        self.title: list[str] = []
        self.title_seen = False
        self.in_first_title = False

    def _append(self, text: str):
        if any(hidden for _, hidden in self.stack):
            return
        tags = {tag for tag, _ in self.stack}
        if self.in_first_title:
            self.title.append(text)
            return
        if "head" in tags or "title" in tags:
            return
        self.regions["fallback"].append(text)
        for name in ("article", "main", "body"):
            if name in tags:
                self.regions[name].append(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        hidden = tag in _SKIP or "hidden" in values or (values.get("aria-hidden") or "").lower() == "true"
        if tag not in _VOID:
            self.stack.append((tag, hidden))
        if tag == "title" and not self.title_seen:
            self.title_seen = True
            self.in_first_title = True
        if tag in _BLOCK:
            self._append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in _BLOCK:
            self._append("\n")
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                removed = self.stack[i:]
                del self.stack[i:]
                if any(t == "title" for t, _ in removed):
                    self.in_first_title = False
                break

    def handle_data(self, data):
        self._append(data)

    def result(self) -> ExtractedText:
        content = ""
        for name in ("article", "main", "body", "fallback"):
            content = _clean("".join(self.regions[name]))
            if content:
                break
        title = " ".join(_clean("".join(self.title)).split())[:500]
        return ExtractedText(title, content[:12_000], len(content) > 12_000)


def extract_html_text(html: str, deadline: float | None = None) -> ExtractedText:
    parser = ReadableHTMLParser()
    for offset in range(0, len(html), 16_384):
        if deadline is not None and time.monotonic() >= deadline:
            raise FetchError(E.TIMEOUT)
        parser.feed(html[offset:offset + 16_384])
    parser.close()
    result = parser.result()
    if deadline is not None and time.monotonic() >= deadline:
        raise FetchError(E.TIMEOUT)
    return result
