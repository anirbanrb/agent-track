"""Command line: one sub-command per pipeline stage.

    supportrag fetch            download the pinned documentation snapshot
    supportrag ingest           clean, chunk and build the keyword index   (no API key)
    supportrag embed            embed every chunk                           (API key)
    supportrag search "..."     show what retrieval returns
    supportrag ask "..."        run the agent
    supportrag eval-retrieval   measure retrieval against evals/retrieval.jsonl
    supportrag eval-answers     measure agent behaviour against evals/answers.jsonl

The stages are separate commands on purpose. Each one can be run, inspected
and timed alone, and the first three questions when something is wrong are
always "was it fetched, was it chunked sensibly, was it retrieved".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supportrag.config import Settings, load_settings

EVAL_DIR = Path("evals")


def _openai_client():
    import os

    from openai import OpenAI  # imported late so offline commands need no key

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Put it in .env or the environment (see .env.example).")
    return OpenAI()


def _retriever(settings: Settings, mode: str):
    from supportrag.embeddings import OpenAIEmbedder
    from supportrag.retrieval import Retriever
    from supportrag.store import Store

    if not settings.db_path.exists():
        sys.exit("No index found. Run `supportrag fetch` and `supportrag ingest` first.")
    embedder = None
    if mode != "bm25":
        embedder = OpenAIEmbedder(_openai_client(), settings.embedding_model)
    return Retriever(Store(settings.db_path), embedder, mode=mode, candidates=settings.candidates_per_retriever)


# -- commands -----------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.corpus import CORPUS_COMMIT, fetch_corpus, load_documents

    docs_dir = fetch_corpus(settings.corpus_dir)
    print(f"Fetched {len(load_documents(docs_dir))} pages at commit {CORPUS_COMMIT[:10]} into {docs_dir}")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.chunking import chunk_documents, estimate_tokens
    from supportrag.corpus import CORPUS_COMMIT, DOCS_SUBDIR, load_documents
    from supportrag.store import Store

    docs_dir = settings.corpus_dir / DOCS_SUBDIR
    if not docs_dir.exists():
        sys.exit("No corpus found. Run `supportrag fetch` first.")
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs, max_tokens=args.max_tokens)
    store = Store(settings.db_path)
    store.replace_chunks(chunks)
    store.set_meta("corpus_commit", CORPUS_COMMIT)
    sizes = sorted(estimate_tokens(c.text) for c in chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages into {settings.db_path}")
    print(f"Chunk size (est. tokens): median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print("Keyword search is ready. Run `supportrag embed` to add vector search.")
    return 0


def cmd_embed(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.chunking import estimate_tokens
    from supportrag.embeddings import OpenAIEmbedder
    from supportrag.store import Store

    store = Store(settings.db_path)
    rows = store.all_chunks()
    if not rows:
        sys.exit("Index is empty. Run `supportrag ingest` first.")
    texts = [chunk.embed_text for _, chunk in rows]
    print(f"Embedding {len(texts)} chunks (~{sum(map(estimate_tokens, texts))} tokens) with {settings.embedding_model} ...")
    matrix = OpenAIEmbedder(_openai_client(), settings.embedding_model).embed(texts)
    store.set_embeddings(settings.embedding_model, [rowid for rowid, _ in rows], matrix)
    print(f"Stored {matrix.shape[0]} vectors of dimension {matrix.shape[1]}.")
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    hits = _retriever(settings, args.mode).search(args.query, k=args.k)
    for position, hit in enumerate(hits, start=1):
        ranks = ", ".join(f"{name} #{rank}" for name, rank in hit.ranks.items())
        print(f"{position}. {hit.chunk.context_header}   [{hit.chunk.chunk_id}; {ranks}]")
        preview = " ".join(hit.chunk.text.split())
        print(f"   {preview[:200]}{'...' if len(preview) > 200 else ''}")
    if not hits:
        print("No results.")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.agent import SupportAgent

    agent = SupportAgent(
        _openai_client(), _retriever(settings, args.mode), settings.chat_model, settings.max_steps, settings.top_k
    )
    result = agent.answer(args.question)

    if args.trace:
        for event in result.trace:
            print(f"  -> {json.dumps(event)}", file=sys.stderr)
    print(result.answer)
    if result.cited:
        print("\nSources:")
        for source in result.cited:
            print(f"  [{source.number}] {source.chunk.context_header}\n      {source.chunk.url}")
    if result.escalated:
        print(f"\n(escalated to a human: {result.escalation['reason']})")
    # The grounding check is enforced here, in code. A prompt can ask for
    # citations; only code can tell you whether you got them.
    if not result.grounded:
        problem = (
            f"cites sources that were never retrieved: {result.invalid_citations}"
            if result.invalid_citations
            else "contains no citations"
        )
        print(f"\nWARNING: this answer {problem}. Treat it as unverified.", file=sys.stderr)
    if result.stopped_early:
        print("WARNING: step budget exhausted; the answer was forced.", file=sys.stderr)
    print(
        f"\n[{result.steps} model calls, {result.usage['input_tokens']} tokens in, "
        f"{result.usage['output_tokens']} out]",
        file=sys.stderr,
    )
    return 0


def cmd_eval_retrieval(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.evals import format_retrieval_report, load_cases, run_retrieval_eval

    retriever = _retriever(settings, args.mode)
    cases = load_cases(EVAL_DIR / "retrieval.jsonl")
    report = run_retrieval_eval(cases, lambda question, k: retriever.search(question, k=k), k=args.k)
    print(format_retrieval_report(report, label=args.mode))
    # A non-zero exit code is what lets CI fail a change that hurts retrieval.
    return 0 if report.hit_rate(args.k) >= args.min_hit_rate else 1


def cmd_eval_answers(args: argparse.Namespace, settings: Settings) -> int:
    from supportrag.agent import SupportAgent
    from supportrag.evals import format_answer_report, load_cases, run_answer_eval

    agent = SupportAgent(
        _openai_client(), _retriever(settings, args.mode), settings.chat_model, settings.max_steps, settings.top_k
    )
    cases = load_cases(EVAL_DIR / "answers.jsonl")
    if args.limit:
        cases = cases[: args.limit]
    rows = run_answer_eval(cases, agent.answer)
    print(format_answer_report(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Full results written to {args.out}")
    pass_rate = sum(1 for r in rows if r["passed"]) / len(rows)
    return 0 if pass_rate >= args.min_pass_rate else 1


# -- wiring -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="supportrag", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    modes = ["bm25", "vector", "hybrid"]

    sub.add_parser("fetch", help="download the pinned documentation snapshot").set_defaults(func=cmd_fetch)

    ingest = sub.add_parser("ingest", help="clean, chunk and index the corpus (no API key needed)")
    ingest.add_argument("--max-tokens", type=int, default=400, help="split sections longer than this (default 400)")
    ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("embed", help="embed every chunk (needs OPENAI_API_KEY)").set_defaults(func=cmd_embed)

    search = sub.add_parser("search", help="show what retrieval returns for a query")
    search.add_argument("query")
    search.add_argument("--mode", choices=modes, default="hybrid")
    search.add_argument("-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="ask the support agent a question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=modes, default="hybrid")
    ask.add_argument("--trace", action="store_true", help="print each tool call to stderr")
    ask.set_defaults(func=cmd_ask)

    eval_r = sub.add_parser("eval-retrieval", help="measure retrieval quality")
    eval_r.add_argument("--mode", choices=modes, default="hybrid")
    eval_r.add_argument("-k", type=int, default=5)
    eval_r.add_argument("--min-hit-rate", type=float, default=0.0, help="exit 1 if hit@k is below this")
    eval_r.set_defaults(func=cmd_eval_retrieval)

    eval_a = sub.add_parser("eval-answers", help="measure agent behaviour (calls the model)")
    eval_a.add_argument("--mode", choices=modes, default="hybrid")
    eval_a.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    eval_a.add_argument("--out", help="write full per-case results to this JSON file")
    eval_a.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 if the pass rate is below this")
    eval_a.set_defaults(func=cmd_eval_answers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())
