"""Chunking: turn a cleaned page into retrievable units.

The strategy is structural rather than fixed-size. A documentation page is
already divided by its author into sections that each answer one question,
so the heading is the natural boundary. Only sections that are too long get
split further, on paragraph boundaries, and never inside a code block.

Each chunk also remembers where it sits ("Billing > Refunds"). That path is
prepended to the text that gets embedded and indexed, because a section body
on its own often does not say what it is about: "Yes, at any time from your
account settings" is unfindable without its heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from supportrag.corpus import Document, split_fences

# Sections are split at h1-h3. An h4 stays inside its parent section.
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_CUSTOM_ID = re.compile(r"\s*\{#([\w-]+)\}\s*$")


def estimate_tokens(text: str) -> int:
    """Rough token count: about four characters per token for English.

    Good enough to decide where to split. Use a real tokenizer only when you
    need to bill or enforce a hard limit.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc path>#<ordinal>", stable for a given corpus + chunker
    doc_path: str
    title: str
    heading: str  # "Section > Subsection", empty for the page preamble
    url: str
    text: str

    @property
    def context_header(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title

    @property
    def embed_text(self) -> str:
        """What the embedding model sees: location first, then content."""
        return f"{self.context_header}\n\n{self.text}"


@dataclass
class _Section:
    path: tuple[str, ...]
    anchor: str
    lines: list[str]


def slugify(heading: str) -> str:
    """Approximate Docusaurus heading anchors (lowercase, hyphenated)."""
    slug = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"[\s]+", "-", slug).strip("-")


def split_sections(body: str) -> list[_Section]:
    sections = [_Section(path=(), anchor="", lines=[])]
    stack: list[tuple[int, str]] = []
    for is_code, segment in split_fences(body):
        for line in segment.splitlines():
            match = None if is_code else _HEADING.match(line)
            if not match:
                sections[-1].lines.append(line)
                continue
            level, text = len(match.group(1)), match.group(2)
            custom = _CUSTOM_ID.search(text)
            if custom:
                text = _CUSTOM_ID.sub("", text)
            anchor = custom.group(1) if custom else slugify(text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            # The page's own h1 repeats the title, so it is not part of the path.
            path = tuple(name for lvl, name in stack if lvl > 1)
            sections.append(_Section(path=path, anchor=anchor, lines=[]))
    return sections


def split_blocks(text: str) -> list[str]:
    """Paragraph-level blocks. A fenced code block is always one block."""
    blocks: list[str] = []
    for is_code, segment in split_fences(text):
        if is_code:
            blocks.append(segment.strip("\n"))
        else:
            blocks.extend(part.strip("\n") for part in re.split(r"\n\s*\n", segment) if part.strip())
    return blocks


def pack_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    """Greedily pack blocks into pieces of at most `max_tokens`.

    When a section has to be split, the last block of one piece is repeated
    at the start of the next. That overlap keeps a sentence and the list or
    code sample it introduces together in at least one chunk.
    A single block larger than the limit is kept whole: a long code sample
    cut in half is worse than an oversized chunk.
    """
    pieces: list[list[str]] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        block_size = estimate_tokens(block)
        if current and size + block_size > max_tokens:
            pieces.append(current)
            overlap = current[-1]
            if estimate_tokens(overlap) <= max_tokens // 4 and not overlap.startswith(("```", "~~~")):
                current, size = [overlap], estimate_tokens(overlap)
            else:
                current, size = [], 0
        current.append(block)
        size += block_size
    if current:
        pieces.append(current)
    return ["\n\n".join(piece) for piece in pieces]


def chunk_document(doc: Document, max_tokens: int = 400) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(doc.body):
        text = "\n".join(section.lines).strip()
        if not text:
            continue  # a heading followed directly by a sub-heading
        url = f"{doc.url}#{section.anchor}" if section.anchor else doc.url
        for piece in pack_blocks(split_blocks(text), max_tokens):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.path}#{len(chunks)}",
                    doc_path=doc.path,
                    title=doc.title,
                    heading=" > ".join(section.path),
                    url=url,
                    text=piece,
                )
            )
    return chunks


def chunk_documents(docs: list[Document], max_tokens: int = 400) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, max_tokens)]
