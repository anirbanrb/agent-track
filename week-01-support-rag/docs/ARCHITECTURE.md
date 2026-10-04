# Architecture

## Two pipelines

There are two separate flows, run at different times.

**Indexing** runs once per corpus version, offline:

```
GitHub (pinned commit)
   │  fetch_corpus            corpus.py
   ▼
133 MDX files
   │  parse_front_matter, clean_mdx
   ▼
133 Documents (title, url, clean Markdown)
   │  chunk_documents         chunking.py
   ▼
925 Chunks (text + heading path + url)
   │                          store.py
   ├──► chunks table + FTS5 keyword index      (no API key)
   │
   │  OpenAIEmbedder.embed    embeddings.py
   └──► embeddings table (float32 vectors)     (API key)
```

**Answering** runs once per question:

```
question
   │
   ▼
SupportAgent.answer                            agent.py
   │   ┌─────────────────────────────────────────────┐
   │   │ model call (instructions + items + tools)   │
   │   │   ├─ function_call search_docs(query)       │
   │   │   │     └─ Retriever.search ──► numbered    │
   │   │   │        sources appended to items        │
   │   │   ├─ function_call escalate_to_human(...)   │
   │   │   │     └─ recorded in the trace            │
   │   │   └─ text ──► leave the loop                │
   │   └──────────── repeat, at most max_steps ──────┘
   ▼
AgentResult: answer, cited sources, invalid citations,
             escalated, trace, token usage
```

The model never sees the index and never receives documents it did not ask for. Its only route to facts is the `search_docs` tool, so everything it learned during a run is recorded in the source registry. That is what makes the citation check possible.

## Retrieval

```
query ─┬─► keyword_query ─► FTS5 MATCH, ranked by BM25 ──► top 20 row ids ─┐
       │                                                                   ├─► RRF ─► top k
       └─► embed ─► matrix @ query (cosine) ─────────────► top 20 row ids ─┘
```

`mode="bm25"` and `mode="vector"` skip one branch. With a single ranking, fusion returns it unchanged, so all three modes share one code path.

## Module responsibilities

| Module | Knows about | Does not know about |
|---|---|---|
| `corpus.py` | Git, MDX syntax, the docs site URL scheme | Chunks, storage, models |
| `chunking.py` | Markdown headings and code fences | Where the text came from, how it is stored |
| `embeddings.py` | The embeddings endpoint | What the texts are |
| `store.py` | SQLite, FTS5, vector bytes | Ranking, fusion, models |
| `retrieval.py` | Ranking and fusion | OpenAI, the agent |
| `agent.py` | The Responses API, tools, citations | SQLite, chunking, how search works |
| `evals.py` | Cases, graders, reports | How answers or rankings are produced (takes callables) |
| `cli.py` | Wiring everything together | Any logic of its own |

The dependencies point one way: `cli → agent → retrieval → store → chunking → corpus`. Nothing lower in that chain imports anything higher.

## Interfaces later weeks build on

These are the seams to keep stable. Change what is behind them freely.

**`Retriever.search(query, k, mode) -> list[Hit]`**
Week 2 wraps this as an MCP tool. Week 3's research agent calls it as one tool among several.

**`TOOLS` and `SupportAgent._run_tool`**
A tool is a JSON schema plus a Python function returning a string. Week 2 replaces the hard-coded list with tools discovered from an MCP server. Week 5 puts an approval check in front of `_run_tool`.

**`AgentResult`**
`trace` (what the tools were asked and returned) and `usage` (tokens) are the raw material for Week 6's tracing and cost dashboard. `grounded`, `escalated` and `invalid_citations` are what the graders read.

**Eval case files (`evals/*.jsonl`)**
One JSON object per line. Week 4 turns `evals.py` into a general harness that reads these same files, adds model-graded checks and runs in CI.

```json
{"id": "r15", "kind": "paraphrase", "question": "I want to stop paying for Plausible", "expected_docs": ["cancel-subscription.md"]}
{"id": "a04", "question": "How many Google Analytics properties can I import into one dashboard?", "should_escalate": false, "expected_docs": ["google-analytics-import.md"], "must_mention": [["5", "five"]]}
{"id": "e02", "question": "What is your refund policy for annual plans?", "should_escalate": true}
```

## Storage schema

```sql
chunks(id, chunk_id, doc_path, title, heading, url, text)
chunks_fts   -- FTS5 over (title, heading, text), external content = chunks
embeddings(chunk_rowid, vector)   -- vector is float32 bytes
meta(key, value)                  -- corpus_commit, embedding_model
```

`meta.embedding_model` exists to stop one specific mistake: querying an index with a different embedding model from the one that built it. The vectors would have the same shape and the results would be silently meaningless, so `Retriever` checks and raises.

## Failure behaviour

| Situation | Behaviour |
|---|---|
| Model sends malformed tool arguments | Error text is returned to the model as the tool output; the loop continues |
| Model calls a tool that does not exist | Same |
| Search returns nothing | Tool output tells the model to rephrase or escalate |
| Model keeps searching | After `max_steps` turns, one final call with `tool_choice="none"` forces an answer; `stopped_early` is set |
| Answer cites a number no search returned | `invalid_citations` is non-empty, `grounded` is false, the CLI prints a warning |
| Answer has no citations and did not escalate | `grounded` is false, the CLI prints a warning |
| Model says it escalated but did not call the tool | `escalated` is false; state comes from the trace, not from the text |
| API errors (rate limit, timeout) | Not handled here. The SDK retries twice by default; anything after that raises. Week 6 covers retries and failure handling |
