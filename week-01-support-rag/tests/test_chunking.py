from supportrag.chunking import chunk_document, estimate_tokens, pack_blocks, slugify, split_sections
from supportrag.corpus import Document


def doc(body: str) -> Document:
    return Document(path="page.md", title="Page", url="https://example.test/page", description="", body=body)


def test_each_section_becomes_a_chunk_with_its_heading_path():
    chunks = chunk_document(doc("Intro.\n\n## Billing\n\nPay monthly.\n\n### Refunds\n\nAsk us.\n\n## Other\n\nMore.\n"))
    assert [(c.heading, c.text) for c in chunks] == [
        ("", "Intro."),
        ("Billing", "Pay monthly."),
        ("Billing > Refunds", "Ask us."),
        ("Other", "More."),
    ]


def test_chunk_ids_and_anchors():
    chunks = chunk_document(doc("Intro.\n\n## What happens after you cancel?\n\nAccess continues.\n"))
    assert [c.chunk_id for c in chunks] == ["page.md#0", "page.md#1"]
    assert chunks[0].url == "https://example.test/page"
    assert chunks[1].url == "https://example.test/page#what-happens-after-you-cancel"


def test_custom_heading_id_is_used_and_removed_from_the_text():
    chunks = chunk_document(doc("## Setup {#install}\n\nDo it.\n"))
    assert chunks[0].heading == "Setup"
    assert chunks[0].url.endswith("#install")


def test_embed_text_carries_title_and_heading():
    chunk = chunk_document(doc("## Billing\n\nYes, at any time.\n"))[0]
    assert chunk.embed_text == "Page > Billing\n\nYes, at any time."


def test_heading_with_no_body_produces_no_chunk():
    chunks = chunk_document(doc("## Parent\n\n### Child\n\nText.\n"))
    assert [c.heading for c in chunks] == ["Parent > Child"]


def test_hash_inside_code_block_is_not_a_heading():
    sections = split_sections("## Real\n\n```bash\n# just a comment\n```\n")
    assert [s.path for s in sections if s.lines or s.path] == [("Real",)]
    chunk = chunk_document(doc("## Real\n\n```bash\n# just a comment\n```\n"))[0]
    assert "# just a comment" in chunk.text


def test_h4_stays_inside_its_parent_section():
    chunks = chunk_document(doc("## Section\n\nBody.\n\n#### Detail\n\nMore.\n"))
    assert len(chunks) == 1 and "#### Detail" in chunks[0].text


def test_long_section_is_split_on_paragraphs_with_overlap():
    paragraphs = [f"Paragraph {n} " + "word " * 18 for n in range(10)]  # ~25 tokens each
    pieces = pack_blocks(paragraphs, max_tokens=120)
    assert len(pieces) > 1
    assert all(estimate_tokens(piece) <= 120 for piece in pieces)
    # The last paragraph of one piece opens the next.
    assert pieces[1].startswith(pieces[0].split("\n\n")[-1])


def test_large_paragraphs_are_not_repeated_as_overlap():
    paragraphs = [f"Paragraph {n} " + "word " * 60 for n in range(4)]  # ~78 tokens each
    pieces = pack_blocks(paragraphs, max_tokens=120)
    assert "\n\n".join(pieces).count("Paragraph 1 ") == 1


def test_oversized_code_block_is_never_cut():
    code = "```\n" + "line\n" * 400 + "```"
    pieces = pack_blocks(["Before.", code, "After."], max_tokens=100)
    assert sum(code in piece for piece in pieces) == 1


def test_slugify():
    assert slugify("How do I get an invoice?") == "how-do-i-get-an-invoice"
