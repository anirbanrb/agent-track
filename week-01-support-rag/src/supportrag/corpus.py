"""Corpus: fetch the documentation and turn each page into clean text.

The corpus is the public Plausible Analytics documentation, fetched from
GitHub at a pinned commit. Pinning matters: the eval set names specific
pages, so the corpus must not change underneath it.

The pages are Docusaurus MDX, which is Markdown with JSX mixed in. Most of
this module is about removing what is not content (imports, components,
JSON-LD blocks, images) without damaging what is (code samples, inline code).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

CORPUS_REPO = "https://github.com/plausible/docs"
CORPUS_COMMIT = "8c3b2383026e3826d0921d32957ce9140e613c2e"
DOCS_SUBDIR = "docs"
BASE_URL = "https://plausible.io/docs"


@dataclass(frozen=True)
class Document:
    path: str  # relative to the docs directory, e.g. "proxy/introduction.md"
    title: str
    url: str
    description: str
    body: str  # cleaned Markdown


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def fetch_corpus(dest: Path, repo: str = CORPUS_REPO, commit: str = CORPUS_COMMIT) -> Path:
    """Fetch exactly one commit of the docs repo into `dest`.

    Three flags keep this small and reproducible:
    - `--depth 1 <sha>` downloads a single pinned commit, no history;
    - `--filter=blob:none` defers downloading file contents;
    - the sparse checkout then only materialises `docs/`, so the repo's
      screenshots (most of its size) are never downloaded.
    """
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True)

    if not (dest / ".git").exists():
        git("init", "-q")
        git("remote", "add", "origin", repo)
        git("sparse-checkout", "set", "--no-cone", f"/{DOCS_SUBDIR}/")
    if _head(dest) != commit:
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        git("checkout", "-q", "FETCH_HEAD")
    return dest / DOCS_SUBDIR


def _head(repo_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a page into its `key: value` front matter and its body.

    The docs only use flat string keys, so a real YAML parser would be a
    dependency with nothing to do.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[match.end() :]


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^(```|~~~)")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_HEAD_BLOCK = re.compile(r"^<head>.*?^</head>\s*$", re.DOTALL | re.MULTILINE)
_IMPORT_LINE = re.compile(r"^(import|export)\s.*?['\"];?\s*$", re.MULTILINE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_TAG = re.compile(r"</?[A-Za-z][^<>]*?>", re.DOTALL)
_ADMONITION_OPEN = re.compile(r"^:::(\w+)[ \t]*(.*)$", re.MULTILINE)
_ADMONITION_CLOSE = re.compile(r"^:::\s*$", re.MULTILINE)
_RULE = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Split Markdown into (is_code, text) segments at fenced code blocks.

    Everything else in the pipeline uses this, because the same characters
    mean different things inside and outside a code block: `<script>` in
    prose is markup to remove, `<script>` in a code sample is the answer to
    "how do I install the snippet".
    """
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line.strip()):
            if in_fence:
                buffer.append(line)
                segments.append((True, "".join(buffer)))
                buffer = []
            else:
                if buffer:
                    segments.append((False, "".join(buffer)))
                buffer = [line]
            in_fence = not in_fence
        else:
            buffer.append(line)
    if buffer:
        segments.append((in_fence, "".join(buffer)))
    return segments


def _clean_prose(text: str) -> str:
    # Protect inline code first. "Add the snippet to your site's `<head>`"
    # must survive tag stripping, and must not be mistaken for the start of
    # a <head> block.
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)

    text = _HEAD_BLOCK.sub("", text)  # JSON-LD duplicates of the page content
    text = _IMPORT_LINE.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)  # keep the link text, drop the target
    text = _TAG.sub("", text)  # JSX/HTML tags go, their inner text stays
    text = _ADMONITION_OPEN.sub(
        lambda m: f"{m.group(1).capitalize()}: {m.group(2)}".rstrip(), text
    )
    text = _ADMONITION_CLOSE.sub("", text)
    text = _RULE.sub("", text)

    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def clean_mdx(body: str) -> str:
    parts = [segment if is_code else _clean_prose(segment) for is_code, segment in split_fences(body)]
    text = "".join(parts)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANK_RUNS.sub("\n\n", text).strip() + "\n"


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def page_url(path: str, meta: dict[str, str]) -> str:
    slug = meta.get("slug")
    if slug is not None:
        return BASE_URL + ("" if slug == "/" else "/" + slug.strip("/"))
    return f"{BASE_URL}/{path.removesuffix('.md')}"


def load_document(docs_dir: Path, file: Path) -> Document:
    path = file.relative_to(docs_dir).as_posix()
    meta, body = parse_front_matter(file.read_text(encoding="utf-8"))
    return Document(
        path=path,
        title=meta.get("title") or path.removesuffix(".md"),
        url=page_url(path, meta),
        description=meta.get("description", ""),
        body=clean_mdx(body),
    )


def load_documents(docs_dir: Path) -> list[Document]:
    return [load_document(docs_dir, file) for file in sorted(docs_dir.rglob("*.md"))]
