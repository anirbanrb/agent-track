# Walkthrough

Read this with the code open. Each section names the file it explains, the decision that was made, what was rejected, and how to check the result yourself. The goal is that after this week you could rebuild the project from an empty directory without looking.

Contents

1. The mental model
2. Run it first
3. Corpus: fetch and pin
4. Cleaning: the unglamorous half of RAG
5. Chunking
6. Embeddings
7. Storage
8. Retrieval: keyword, vector, fusion
9. The agent loop
10. Grounding is enforced in code
11. Evals
12. What went wrong while building this
13. What has not been verified
14. Questions you should now be able to answer

---

## 1. The mental model

A language model knows what was in its training data and nothing about your product's current documentation. Retrieval-augmented generation (RAG) is an open-book exam: before the model answers, you look up the relevant pages and put them in front of it.

That gives a system with two halves that fail independently:

- **Retrieval**: did the right passage get found? This is a search problem. No model is involved in judging it.
- **Generation**: given the passage, did the model answer correctly and stay within it? This is a model-behaviour problem.

Most bad RAG answers are retrieval failures. The model cannot use a passage it was never shown, and when it is shown the wrong one it will often answer confidently from it. So this project measures retrieval separately, before any model call.

An **agent** is a model that can call functions, running in a loop until it decides it is done. Here the model is not handed search results; it is handed a `search_docs` tool and decides what to search for, whether the results are good enough, and whether to search again. That is the difference between a RAG pipeline (retrieve once, then generate) and a RAG agent.

If you know web development, the mapping is close:

| Here | Roughly like |
|---|---|
| Keyword index (FTS5, BM25) | MySQL `FULLTEXT` search |
| Embedding | A hash that puts similar meanings near each other instead of scattering them |
| Tool definition | An API endpoint's schema, written for a model to read |
| Agent loop | A request/response cycle where the server (your code) executes what the client (the model) asks for |
| Eval set | A test suite whose assertions are statistical |

## 2. Run it first

Do this before reading further. Seeing the output makes the rest concrete.

```bash
pip install -e ".[dev]"
pytest
supportrag fetch
supportrag ingest
supportrag search "x-plausible-dropped" --mode bm25
supportrag search "I want to stop paying for Plausible" --mode bm25
supportrag eval-retrieval --mode bm25
```

The first search finds the right page immediately. The second returns the page about *upgrading to* a paid plan, because it shares the word "paying" and the cancellation page does not contain it. Hold on to that failure: sections 6 to 8 exist to fix it.

## 3. Corpus: fetch and pin

*File: `corpus.py`, `fetch_corpus`*

The corpus is fetched from GitHub at one specific commit hash, written into the code as `CORPUS_COMMIT`.

**Why pin.** The eval set says things like "the answer to this question is on `cancel-subscription.md`". If the docs change tomorrow, a drop in your score could be your change or theirs, and you cannot tell which. A pinned corpus makes every eval run comparable with every other. In production you would re-index on a schedule; the principle that carries over is that an eval set is tied to a corpus version.

**Why real documentation.** I could have written twenty tidy Markdown files. Retrieval would then score well for a reason that says nothing about retrieval: text written to be found is easy to find. Real docs have the problems you will meet at work, and section 4 is entirely about them.

**Why not bundled.** The docs are CC BY-SA licensed. Downloading them on your machine is use; committing them to your repository is redistribution, with attribution and share-alike obligations. A fetch script avoids the question.

## 4. Cleaning: the unglamorous half of RAG

*File: `corpus.py`, `clean_mdx` and `split_fences`*

Open `data/corpus/docs/troubleshoot-integration.md`. Before the first sentence of real content there are 74 lines of JSON-LD inside a `<head>` tag, then a JavaScript import. Further down there are `<div class="browser">` wrappers, `<img>` tags with JSX expressions, and `:::tip` blocks.

None of that is content. Left in, it costs embedding tokens, matches keyword searches it should not match, and dilutes the embedding of the passage it sits in. The JSON-LD block is worse than noise: it restates the page's FAQ, so every answer would exist twice with different surrounding text.

The cleaner removes it, and the only hard part is knowing where *not* to clean.

**Trap 1: code blocks.** The docs contain this, inside a fenced code block:

```html
<script async src="https://plausible.io/js/script.js"></script>
```

To a tag-stripping regex that is markup. To a customer it is the answer to "how do I install the snippet". So `split_fences` first cuts the page into alternating prose and code segments, and cleaning only ever touches prose. The chunker reuses the same function for the same reason: `# a comment` inside a bash block is not a heading.

**Trap 2: inline code.** "Add the snippet to your site's `` `<head>` ``" contains a tag in backticks. A tag stripper deletes it and leaves "your site's  and"; a `<head>...</head>` block remover might treat it as the start of a block and delete everything up to the next `</head>`. So `_clean_prose` swaps every inline code span for a placeholder, cleans, and swaps them back.

**Decisions inside the cleaner**

- Markdown links keep their text and lose their target. `[clear the cache](cache.md)` becomes `clear the cache`. The target is noise to both search methods.
- Tags are removed, their inner text is kept. Content inside `<details>` survives.
- Images are dropped entirely, alt text included. That discards some information; keeping alt text is one of the exercises.
- `:::tip Title` becomes `Tip: Title` and the body stays. Admonitions in these docs often hold the most important sentence on the page.

**Check it yourself.** `supportrag ingest`, then open `data/index.sqlite` in any SQLite browser and read twenty random rows of the `chunks` table. Reading your own chunks is the highest-value debugging habit in RAG, and almost nobody does it.

## 5. Chunking

*File: `chunking.py`*

You cannot embed a whole page as one vector: a page about billing covers fifteen different questions, and one vector would be the average of all of them and close to none. You split pages into chunks and index each one.

**Decision: split on headings, not on size.** The common default is "500 tokens with 50 overlap", which cuts wherever the count runs out, often mid-list or between a sentence and the code sample it introduces. Documentation already has boundaries chosen by a person: each section answers one question. `split_sections` cuts at `#`, `##` and `###`, and leaves `####` inside its parent because at that depth sections are usually fragments.

Result on this corpus: 925 chunks from 133 pages, median about 115 estimated tokens.

**Decision: carry the heading path.** This chunk exists in the corpus:

> Paddle calculates a pro-rated charge based on what you've already paid and the time remaining on your current billing cycle. You only pay the difference.

Nothing in it says "upgrade". Its heading does: *Billing FAQ > How does pro-rating work when I upgrade?* So `Chunk.embed_text` puts `title > heading path` in front of the body before embedding, and the keyword index has separate, more heavily weighted `title` and `heading` columns. This is the cheapest retrieval improvement there is.

**Decision: split long sections on paragraphs, with overlap.** 47 of the 865 sections exceed 400 estimated tokens. `pack_blocks` splits those between paragraphs and repeats the last paragraph of one piece at the start of the next, but only when that paragraph is short. A code block is always one block and is never cut, and a table larger than the limit is kept whole. That is why 20 chunks are still over the limit after splitting, the largest at about 1,200 tokens. An oversized chunk is a smaller problem than half a code sample.

**Decision: no merging of small sections.** Eight chunks are under 15 tokens, for example "We have an official Plausible connector for Data Studio." With the heading path attached they are still findable and precise. Merging them into neighbours would be more code for a difference the eval cannot currently detect. If the eval later shows small chunks hurting, that is the time.

**Decision: chunk ids are positional.** `billing.md#3` is the fourth chunk of that page. It is stable for a fixed corpus and chunker, and it changes when either changes. Remember that when you reach section 11.

## 6. Embeddings

*File: `embeddings.py`*

An embedding model turns a text into a list of numbers (1,536 of them for `text-embedding-3-small`) such that texts with similar meaning get similar lists. "Stop paying" and "cancel your subscription" share no words and end up close together. That is the property keyword search lacks.

Three details in a short file:

- **Batching.** 64 texts per request. The whole corpus takes 15 requests, not 925.
- **Order.** The API tags each returned vector with the index of its input. The code sorts by that index instead of assuming the order.
- **Normalising.** Every vector is scaled to length 1 before it is stored. Cosine similarity between unit vectors is just a dot product, so search becomes one matrix multiplication (section 8).

`Embedder` is a `Protocol`: anything with a `model` name and an `embed` method fits. The tests use a `HashEmbedder` that needs no network, which is how the vector path is tested without a key.

## 7. Storage

*File: `store.py`*

Everything lives in one SQLite file: the chunks, a full-text index over them, and the vectors as raw `float32` bytes.

**Why no vector database.** 925 vectors of 1,536 floats is under 6 MB. Comparing a query against all of them is one matrix-vector product and takes well under a millisecond. A vector database solves problems this project does not have: collections too large for memory, approximate search over millions of vectors, concurrent writers, metadata filtering at scale. Know roughly where the line is (hundreds of thousands of vectors, or the need for filtering and live updates) and be able to say why you are on this side of it. "I didn't need one, here is the arithmetic" is a stronger answer in an interview than a logo.

**FTS5.** SQLite's built-in full-text index, ranked by BM25, which scores a chunk higher when it contains rare query words often. The `porter` tokenizer stems, so "cancelling" matches "cancel". The index is declared with `content='chunks'`, meaning it indexes the `chunks` table without storing a second copy of the text.

**`replace_chunks` rebuilds everything.** No incremental update, no change detection. It takes a fraction of a second and cannot leave stale rows or vectors pointing at chunks that no longer exist. Write the simple version first and replace it when a measurement tells you to.

**The model lock.** `meta.embedding_model` records which model produced the vectors, and `Retriever` refuses to search with a different one. Vectors from two models have the same shape and no relationship; without the check you get plausible-looking nonsense and no error.

## 8. Retrieval: keyword, vector, fusion

*File: `retrieval.py`*

**Keyword search** needs one piece of care. FTS5 has a query language, so raw user text is unsafe: `what's` contains a quote and is a syntax error, and words like `NOT` are operators. `keyword_query` extracts plain terms, drops function words, quotes each term and joins them with `OR`. `OR` rather than `AND` because BM25 already ranks chunks matching more terms higher, while `AND` returns nothing as soon as one word is missing.

**Vector search** is three lines: embed the question, multiply the matrix of chunk vectors by it, take the highest scores.

**They fail in opposite directions.** The measured BM25 results on the 40-question eval:

| Kind | Example | hit@5 |
|---|---|---|
| identifier | `x-plausible-dropped` | 1.00 |
| direct | "How do I cancel my subscription?" | 1.00 |
| paraphrase | "I want to stop paying for Plausible" | 0.42 |

Keyword search is exact, and exactness is what you want for header names, error codes and config keys. It is helpless when the customer's words differ from the author's. Vector search is the reverse: good at paraphrase, and prone to treating `hashBasedRouting` as roughly the same thing as any other camel-cased option.

**Fusion.** The two methods produce scores on unrelated scales (BM25 scores are unbounded, cosine similarity is between -1 and 1), so they cannot be added. Reciprocal rank fusion ignores scores and uses positions: each list contributes `1 / (60 + rank)` for every chunk in it.

| Chunk | BM25 rank | Vector rank | Fused score |
|---|---|---|---|
| A | 1 | 8 | 1/61 + 1/68 = 0.0311 |
| B | not found | 1 | 1/61 = 0.0164 |
| C | 3 | 2 | 1/63 + 1/62 = 0.0320 |

C wins: both methods rate it well. A is second. B, top of one list and absent from the other, is last. That is the behaviour you want, with no tuning and no calibration. The constant 60 flattens the difference between ranks 1 and 2 so that one method cannot dominate on its own.

Each `Hit` records its rank in each list. `supportrag search "..." --mode hybrid` prints them, so you can see which method found each result.

## 9. The agent loop

*File: `agent.py`, `SupportAgent.answer`*

This is the part to know by heart, because every agent in the following weeks is a variation on it.

The conversation is a list of typed items. One run looks like this:

```
request 1 input:   [ {role: user, content: "How do I cancel?"} ]
response 1 output: [ {type: function_call, name: search_docs,
                      arguments: '{"query":"cancel subscription"}', call_id: "call_1"} ]

        your code runs the search and appends the result

request 2 input:   [ {role: user, ...},
                     {type: function_call, ..., call_id: "call_1"},
                     {type: function_call_output, call_id: "call_1",
                      output: '<source number="1" ...>...</source>'} ]
response 2 output: [ {type: message, content: "Open Account Settings ... [1]."} ]
```

The model does not execute anything. It emits a request to call a function; your code decides whether and how to run it, and sends back a string. The `call_id` ties each result to the call that asked for it. The API is stateless here: every request carries the whole conversation.

The loop:

1. Call the model with the instructions, the items so far, and the tool definitions.
2. Append everything it returned to the items (including reasoning items, which some models require you to send back).
3. If there are no function calls, the text is the answer. Stop.
4. Otherwise run each call, append a `function_call_output` for each, and go to 1.

**Decisions in and around the loop**

*The model searches; it is not handed results.* It can rewrite "stop paying" as "cancel subscription" before searching, and it can search twice. This partly compensates for weak retrieval, at the cost of an extra model call per search.

*Tool descriptions say when, not only what.* The description of `search_docs` tells the model to use the docs' vocabulary and to retry with different wording. The description of `escalate_to_human` lists the situations that call for it. A tool definition is the only documentation the model has for your function; write it for a capable colleague who cannot ask you questions.

*Strict schemas.* `"strict": True` with `additionalProperties: false` and every property required makes the API guarantee that arguments match the schema. `reason` is an enum so the hand-off is machine-readable.

*Tool errors go back to the model.* Malformed arguments, an empty query or an unknown tool name produce a string starting with `Error:` as the tool output. The model can read it and correct itself. An exception would end the conversation over something recoverable.

*A step budget.* After `max_steps` turns the loop makes one last call with `tool_choice="none"`, which forbids tool calls and forces an answer from what has been gathered. The tool list is still sent unchanged, so the request prefix stays identical and remains eligible for prompt caching. Every agent needs a budget: without one, a model that keeps searching runs until it hits a limit you did not choose.

*Token usage is summed* across calls into `AgentResult.usage`. Week 6 turns that into cost.

## 10. Grounding is enforced in code

*File: `agent.py`, the source registry and `AgentResult.grounded`*

The instructions tell the model to answer only from search results and to cite them. Instructions are requests. What makes the behaviour dependable is that the code checks it.

**The source registry.** Every chunk a search returns gets a number that stays fixed for the run: the same passage is `[2]` whichever search returned it. The registry is the complete list of what the model was shown.

**The checks, after the final answer:**

- Parse every `[n]` from the text.
- Any number not in the registry is an invalid citation: the model cited a source that does not exist.
- An answer with no citations, that did not escalate, is ungrounded.

`AgentResult.grounded` is false in both cases and the CLI prints a warning. A production system would retry or escalate instead of showing the answer.

**Escalation is a tool, not a phrase.** "I don't know" is implemented as a call to `escalate_to_human` with a reason and a summary. That makes it countable (how often do we escalate, and why?) and testable (did it escalate when it should?). `escalated` is derived from the tool trace. If the model writes "I've passed this to our team" without calling the tool, `escalated` is false, and there is a test for exactly that.

**What this does not prove.** A citation check confirms the cited passage was retrieved. It does not confirm the passage supports the sentence. An answer can cite `[1]` and misstate what `[1]` says. Catching that needs a second model reading both, which is the model-graded eval in Week 4.

## 11. Evals

*Files: `evals.py`, `evals/retrieval.jsonl`, `evals/answers.jsonl`*

An eval is a test whose result is a rate, not a pass. You change one thing, rerun, and compare.

**Retrieval eval (40 cases, no model call, instant).** Each case is a question and the pages that would answer it. Metrics:

- `hit@k`: the share of questions with a correct page in the top k.
- `MRR`: the average of 1/rank of the first correct result. It distinguishes "right answer first" from "right answer fifth", which hit@5 does not.

Three design choices matter more than the metrics:

1. *Cases are labelled by page, not by chunk.* Chunk ids change whenever the chunker changes. Page paths do not. An eval set labelled by chunk would be invalidated by the experiments it is meant to judge.
2. *Cases are split by kind.* An overall 0.72 hides that identifiers score 1.00 and paraphrases 0.42. The split tells you what to fix.
3. *The cases were written before any tuning.* If you tune until the eval passes and then report the eval, you have measured your tuning. Add new cases when you find new failures; do not edit old ones to pass.

**Answer eval (16 cases, calls the model).** Twelve answerable questions and four that should be escalated: two needing account access, two the docs do not cover. The graders are code:

| Check | Catches |
|---|---|
| `escalation_correct` | Answering when it should hand off, and handing off when the docs had the answer |
| `citations_valid` | Citing sources that were never retrieved |
| `has_citation` | Unsupported answers |
| `cites_expected_page` | Right-sounding answer from the wrong page |
| `mentions_*` | Citing the right page and still omitting the fact asked for |
| `within_budget` | Runs that exhausted the step budget |

Case `a12`, "Please delete my account for me", is there to catch over-escalation. It sounds like an account action, but the docs describe a self-service path, so the correct behaviour is to answer. An agent that escalates everything is safe and useless; the eval has to penalise that direction too.

`mentions_*` matches whole phrases and strips citation markers first, so "5" does not match "15" or the citation `[5]`. It is still brittle: it will fail a correct answer that says "fourteen days" where the case lists "14". That brittleness is the argument for model-graded checks in Week 4.

Both commands take a threshold (`--min-hit-rate`, `--min-pass-rate`) and exit non-zero below it. That is all CI needs.

## 12. What went wrong while building this

None of these are in the final code. They are here because the mistakes are more instructive than the result.

1. **The first fetch downloaded 232 MB.** A shallow clone of the docs repository is almost entirely screenshots. Adding `--filter=blob:none` and a sparse checkout of `docs/` brought it to 1.6 MB and about one second.
2. **A check for leftover markup flagged 30 chunks.** Every one was inline code such as `` `<head>` `` or `` `<a>` ``, which is content. The check was wrong, the cleaner was right, and it confirmed why the inline-code protection exists.
3. **"Can I get a refund" returns the VAT page.** The only occurrence of "refund" in the whole corpus is about VAT. The documentation has no refund policy, which is why that question is an escalation case in the answer eval. The right response to a gap in the corpus is a hand-off, not better retrieval.
4. **"What's the best way to cancel" ranks a reviews page first.** That page contains "the best way". Filler words that are not in the stopword list outweigh the one word that matters. It is left unfixed: adding stopwords until your examples pass is tuning to the examples.
5. **The test fakes broke on the installed SDK.** The first fake built a full `ResponseUsage` object, and the current SDK requires a nested field older versions did not have. The fake now builds only the two totals the agent reads. Also, the SDK's HTTP library changed name between major versions, which the wire test has to allow for. SDKs move; test against the real types, and keep the fakes minimal.
6. **A test asserted overlap that the rule does not produce.** The chunker only repeats a paragraph if it is short. The test used long paragraphs and failed. The rule was right and the test encoded a wrong mental model, which is the more useful kind of test failure.

## 13. What has not been verified

This project was built without an OpenAI API key. Be clear about what that means.

**Verified:** fetching, cleaning, chunking and keyword search on the real corpus; the BM25 eval numbers; the agent loop, tool handling, citation checks and graders through tests; the request and response format through tests that run the real OpenAI SDK against a fake HTTP server.

**Not verified:** any live call. Specifically:

- that `gpt-6-luna` and `text-embedding-3-small` are the right model ids for your account (taken from OpenAI's documentation in October 2026);
- how well vector and hybrid retrieval score;
- how the model actually behaves: whether it cites reliably, whether it escalates the four cases it should, whether it over-escalates `a12`;
- cost. The estimate is under one US cent to embed the corpus and a cent or two for the full answer eval on the default model.

Expect the first live run to need a fix or two. Prompts in particular are never right on the first run; the answer eval exists to tell you how wrong, and where.

## 14. Questions you should now be able to answer

Say each answer out loud. If you cannot, reread the section.

1. Why measure retrieval separately from answer quality? (1, 11)
2. Why does a fenced code block need different handling from prose, in both the cleaner and the chunker? (4, 5)
3. What does prepending the heading path to a chunk buy you, and when would it hurt? (5)
4. Why can you not add a BM25 score to a cosine similarity, and what does RRF do instead? (8)
5. At what scale would you move from brute-force search to a vector database, and what would you gain? (7)
6. Walk through the items list of a two-turn tool call. What is `call_id` for? (9)
7. Why return a tool error to the model instead of raising? (9)
8. Why must an agent loop have a step budget, and what is the cleanest way to enforce one? (9)
9. What does a citation check prove, and what does it not? (10)
10. Why is "I don't know" a tool call here and not a sentence? (10)
11. Why label eval cases by page and not by chunk? (11)
12. Why does the answer eval include a case where escalating is the wrong behaviour? (11)
