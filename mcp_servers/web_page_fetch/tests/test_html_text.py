import codecs
import time

import pytest

from web_page_fetch.errors import FetchError
from web_page_fetch.html_text import decode_html, extract_html_text


def test_title_and_readable_text():
    result = extract_html_text('''<html><head><title> Hello &amp; world </title></head><body>
    nav<main>main<article><h1>Heading</h1><p>A&nbsp; B</p><article>nested</article>
    <script>script_canary</script><style>style_canary</style><noscript>noscript</noscript>
    <template>template</template><div hidden>hidden</div><span aria-hidden="true">secret</span>
    <!-- comment --></article>main end</main>footer</body></html>''')
    assert result.title == "Hello & world"
    assert result.content == "Heading\nA B\nnested"
    assert not result.truncated


@pytest.mark.parametrize("html,expected", [
    ("<body>body<main>main<article>article</article></main></body>", "article"),
    ("<body>body<main>main<article><script>x</script></article></main></body>", "main"),
    ("<body><main> </main>body</body>", "body"),
    ("<head><title>T</title></head><p>fragment</p>", "fragment"),
    ("<article>one</article><article>two</article>", "one\ntwo"),
    ("<body><img hidden>visible<br>line</body>", "visible\nline"),
    ("<title>first</title><title>second</title>", ""),
])
def test_region_priority(html, expected):
    assert extract_html_text(html).content == expected


@pytest.mark.parametrize("size,truncated", [(11999, False), (12000, False), (12001, True)])
def test_truncation(size, truncated):
    result = extract_html_text(f"<title>{'t' * 501}</title><body>{'字' * size}</body>")
    assert len(result.title) == 500
    assert len(result.content) == min(size, 12000)
    assert result.truncated is truncated


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_bom_beats_http_and_meta(encoding):
    text = '<meta charset="ascii"><body>中文</body>'
    assert decode_html(text.encode(encoding), "text/html; charset=ascii") == text


@pytest.mark.parametrize("bom,encoding", [(codecs.BOM_UTF16_BE, "utf-16-be"), (codecs.BOM_UTF32_BE, "utf-32-be")])
def test_big_endian_bom(bom, encoding):
    assert decode_html(bom + "中文".encode(encoding), "text/html") == "中文"


def test_http_beats_meta():
    text = '<meta charset="utf-8"><body>中文</body>'
    assert decode_html(text.encode("gb18030"), "text/html; charset=gb18030") == text


@pytest.mark.parametrize("meta", ['<meta charset="gb18030">',
    '<meta http-equiv="Content-Type" content="text/html; charset=gb18030">'])
def test_meta_after_invalid_http(meta):
    text = meta + "中文"
    assert decode_html(text.encode("gb18030"), "text/html; charset=unknown-encoding") == text


def test_utf8_fallback_and_no_guessing():
    assert decode_html(b"bad\xff", "text/html") == "bad\ufffd"
    assert decode_html(b"bad\xff", "text/html; charset=base64_codec") == "bad\ufffd"
    assert decode_html("中文".encode(), "text/html; charset=ascii") == "\ufffd" * 6


def test_expired_parse_budget():
    with pytest.raises(FetchError, match="TIMEOUT"):
        extract_html_text("<p>text</p>", time.monotonic() - 1)
