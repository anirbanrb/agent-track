from supportrag.corpus import clean_mdx, page_url, parse_front_matter, split_fences

PAGE = '''---
title: Troubleshooting
description: "Fix your install."
---

<head>
  <script type="application/ld+json">{`{"@type": "FAQPage"}`}</script>
</head>

import useBaseUrl from '@docusaurus/useBaseUrl';

Add the snippet to your site's `<head>` and [clear the cache](cache.md).

<div class="browser">
    <img alt="Banner" src={useBaseUrl('img/banner.webp')} />
</div>

:::tip Check twice
Only one snippet per page.
:::

---

## Install

```html
<script async src="https://plausible.io/js/script.js"></script>
```
'''


def test_front_matter_is_split_from_body():
    meta, body = parse_front_matter(PAGE)
    assert meta == {"title": "Troubleshooting", "description": "Fix your install."}
    assert body.lstrip().startswith("<head>")


def test_page_without_front_matter_is_returned_unchanged():
    assert parse_front_matter("# Title\n") == ({}, "# Title\n")


def test_cleaning_removes_markup_that_is_not_content():
    cleaned = clean_mdx(parse_front_matter(PAGE)[1])
    assert "FAQPage" not in cleaned  # JSON-LD block
    assert "useBaseUrl" not in cleaned  # import line and JSX expression
    assert "<div" not in cleaned and "<img" not in cleaned
    assert ":::" not in cleaned
    assert "\n---" not in cleaned  # horizontal rule


def test_cleaning_keeps_content():
    cleaned = clean_mdx(parse_front_matter(PAGE)[1])
    assert "clear the cache" in cleaned and "cache.md" not in cleaned  # link text kept, target dropped
    assert "Tip: Check twice" in cleaned and "Only one snippet per page." in cleaned
    assert "## Install" in cleaned


def test_inline_code_survives_tag_stripping():
    # `<head>` in backticks is content. It must not be stripped as a tag, and
    # must not be mistaken for the start of a <head> block.
    cleaned = clean_mdx(parse_front_matter(PAGE)[1])
    assert "your site's `<head>` and" in cleaned


def test_fenced_code_is_left_exactly_as_written():
    cleaned = clean_mdx(parse_front_matter(PAGE)[1])
    assert '<script async src="https://plausible.io/js/script.js"></script>' in cleaned


def test_split_fences_alternates_prose_and_code():
    segments = split_fences("a\n```\ncode\n```\nb\n")
    assert [is_code for is_code, _ in segments] == [False, True, False]
    assert "".join(text for _, text in segments) == "a\n```\ncode\n```\nb\n"


def test_page_url_uses_slug_when_present():
    assert page_url("introduction.md", {"slug": "/"}) == "https://plausible.io/docs"
    assert page_url("proxy/introduction.md", {}) == "https://plausible.io/docs/proxy/introduction"
