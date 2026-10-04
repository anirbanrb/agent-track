# Week 1: Support agent (RAG)

A support agent that answers questions from product documentation, cites the passages it used, and hands the conversation to a human when the documentation does not cover the question.

It proves one thing: **retrieval-augmented generation done properly**, meaning retrieval you can measure, answers you can check against their sources, and a defined behaviour for "I don't know".

The corpus is the public [Plausible Analytics documentation](https://github.com/plausible/docs): 133 pages of real, messy Docusaurus MDX. This project is a learning exercise and is not affiliated with Plausible.

## What is in it

| Stage | Module | What it does |
|---|---|---|
| Fetch | `corpus.py` | Downloads one pinned commit of the docs (1.6 MB, sparse checkout) |
| Clean | `corpus.py` | Strips JSX, imports, JSON-LD and images; leaves code samples untouched |
| Chunk | `chunking.py` | Splits on headings; each chunk carries its heading path |
| Embed | `embeddings.py` | Batched OpenAI embeddings, normalised to unit length |
| Store | `store.py` | One SQLite file: chunks, FTS5 keyword index, vectors |
| Retrieve | `retrieval.py` | BM25, vector search, and reciprocal rank fusion of the two |
| Agent | `agent.py` | Tool loop on the Responses API: `search_docs`, `escalate_to_human` |
| Evaluate | `evals.py` | Retrieval metrics (hit rate, MRR) and code-graded answer checks |

No agent framework and no vector database. About 1,400 lines of Python, a good third of it comments explaining why, and two dependencies (`openai`, `numpy`).

## Quick start

Requires Python 3.11+ and git.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 53 tests, no network, no API key

supportrag fetch                         # download the docs snapshot
supportrag ingest                        # clean, chunk, build the keyword index
supportrag search "x-plausible-dropped" --mode bm25
supportrag eval-retrieval --mode bm25    # works without an API key
```

Then add your key and turn on the rest:

```bash
cp .env.example .env                     # put OPENAI_API_KEY in it

supportrag embed                         # about 157k tokens, well under one US cent
supportrag eval-retrieval --mode vector
supportrag eval-retrieval --mode hybrid
supportrag ask "I want to stop paying for Plausible" --trace
supportrag eval-answers --out data/answers.json
```

Run every command from this directory: the eval files are read from `./evals`.

## Results

Retrieval eval: 40 questions in three kinds. `hit@k` is the share of questions where a correct page appears in the top k results.

| Mode | Subset | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|
| bm25 | all | 40 | 0.62 | 0.72 | 0.72 | 0.68 |
| bm25 | direct | 14 | 0.71 | 1.00 | 1.00 | 0.86 |
| bm25 | identifier | 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| bm25 | paraphrase | 19 | 0.42 | 0.42 | 0.42 | 0.42 |
| vector | all | 40 | _run it_ | | | |
| hybrid | all | 40 | _run it_ | | | |

The BM25 rows are measured. The vector and hybrid rows are yours to fill in: they need an API key, and comparing the three is the first exercise of the week. The BM25 numbers already tell the story the other two exist to fix: keyword search is perfect on exact identifiers and finds fewer than half of the questions phrased in a customer's own words.

## Status: what has and has not been run

| Part | State |
|---|---|
| Fetch, clean, chunk, keyword index, BM25 search | Run against the real corpus |
| Retrieval eval, BM25 mode | Run; numbers above |
| Agent loop, tool handling, citation checks, graders | Covered by tests using fakes built from the SDK's own types |
| Request and response format | Covered by tests running the real OpenAI SDK against a fake HTTP server |
| Live calls to OpenAI (embeddings, `ask`, `eval-answers`) | **Not run.** Built without an API key. Expect to debug the first live run |
| Model names (`gpt-6-luna`, `text-embedding-3-small`) | Taken from OpenAI's documentation in October 2026; override in `.env` if they have changed |

## Layout

```
src/supportrag/   the package (read in the order of the table above)
tests/            53 tests; fakes.py holds the test doubles
evals/            retrieval.jsonl (40 cases), answers.jsonl (16 cases)
docs/
  WALKTHROUGH.md  every decision, why it was made, what failed first
  ARCHITECTURE.md data flow and the interfaces later weeks build on
  EXERCISES.md    what to change yourself, with pass criteria
```

Start with `docs/WALKTHROUGH.md`.

## Known limits

- Single-turn only. There is no conversation memory.
- No reranker and no query rewriting; both are exercises.
- Answer checks are behavioural (cited the right page, escalated when it should). They do not judge whether the prose is correct. That needs a model as judge and arrives in Week 4.
- Retrieved text is treated as data by instruction only. Proper prompt-injection defences are Week 5.
- Token counts are estimated at four characters per token.

## Corpus licence

The Plausible documentation is licensed [CC BY-SA 4.0](https://github.com/plausible/docs/blob/master/LICENSE.md). This repository does not contain or redistribute it; `supportrag fetch` downloads it to `data/`, which is git-ignored.
