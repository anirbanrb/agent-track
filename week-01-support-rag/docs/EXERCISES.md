# Exercises

Reading this project will not make you able to build it. Changing it will. Each exercise has a pass criterion so you know when you are done, and each is small enough for one sitting.

Do Part A this week. Part B is optional; pick what interests you.

Rule for all of them: **measure before and after**. Run the relevant eval, write the number down, make the change, run it again. A change you cannot measure is a guess.

---

## Part A: this week

### A1. Make it run live (30 minutes)

Add your API key, then:

```bash
supportrag embed
supportrag eval-retrieval --mode vector
supportrag eval-retrieval --mode hybrid
```

Fill in the results table in `README.md`.

*Pass:* the table is complete, and you can explain in two sentences why hybrid beats each single method on this eval, or, if it does not, why not.

*Expect:* vector to do much better than BM25 on paraphrases and possibly worse on identifiers. If a model id is rejected, set `SUPPORTRAG_CHAT_MODEL` or `SUPPORTRAG_EMBEDDING_MODEL` in `.env`.

### A2. Run the answer eval and read every failure (45 minutes)

```bash
supportrag eval-answers --out data/answers.json
```

For each failing case, open `data/answers.json` and classify the cause as one of:

- **retrieval**: the right page was never returned (check `searches` and `cited`);
- **model**: the right page was returned and the answer is still wrong or uncited;
- **grader**: the answer is fine and the check is too strict.

*Pass:* a short table of case id, cause, one-line explanation. This classification habit is the single most transferable skill of the week.

### A3. Fix one failure of each kind you found (1 to 2 hours)

- A retrieval failure: change chunking, ranking or the tool description.
- A model failure: change `INSTRUCTIONS` in `agent.py`.
- A grader failure: fix the case in `evals/answers.jsonl`.

*Pass:* the case passes, and no previously passing case now fails. If your prompt change fixes one case and breaks two, you have learned why evals exist.

### A4. Add five eval cases of your own (30 minutes)

Read a few pages of the docs and write three retrieval cases and two answer cases, at least one of which should escalate. Write them before trying the questions.

*Pass:* the cases are in the JSONL files and you recorded which ones failed on the first run.

### A5. Rebuild the loop from memory (1 hour)

Close `agent.py`. In a new file, write a function that takes a question and a list of tools and runs the model/tool loop until the model answers. No citations, no registry, just the loop, a step budget, and tool errors returned as strings. Then compare with `SupportAgent.answer`.

*Pass:* your loop answers a question using `search_docs`. This is the piece you will write again in every later week, so it is worth being able to write it cold.

---

## Part B: go further

### B1. Reranking

Retrieve 20 candidates, then ask a model to score each one against the question from 0 to 3 and keep the top 5. Add it as an optional stage after fusion.

*Pass:* hit@1 and MRR improve on the paraphrase subset. Report the extra latency and tokens per query; a reranker is always a cost trade.

### B2. Query rewriting

Before searching, have the model produce two alternative phrasings of the question, search with all three, and fuse the rankings with the existing `reciprocal_rank_fusion`.

*Pass:* BM25-only hit@5 on paraphrases improves from 0.42. Then answer: now that the agent can rewrite its own queries through the tool, is a separate rewriting stage still worth its cost?

### B3. Small-to-big retrieval

Search over the small chunks as now, but return the whole parent section (or the whole page, if short) to the model.

*Pass:* the `mentions_*` pass rate in the answer eval goes up or stays level, and you report what happened to input tokens per answer.

### B4. Keep image alt text

The cleaner drops images. Change it to keep the alt text as `(Screenshot: ...)`.

*Pass:* tests updated, retrieval eval rerun, and a one-line conclusion on whether it helped. A null result is a real result.

### B5. Index the page description

Each page's front matter has a `description`, currently unused. Add it to the first chunk of each page.

*Pass:* retrieval eval before and after.

### B6. Conversation memory

Make `answer` accept prior turns so a follow-up like "and what happens to my data?" works after a question about cancelling.

*Pass:* a two-turn test using `FakeClient`, and an answer to this: what goes wrong when the follow-up is searched literally, and where is the right place to fix it?

### B7. Swap the vector store

Replace the brute-force matrix with `sqlite-vec` or `pgvector`, behind the existing `Store` interface.

*Pass:* identical eval numbers, and a measured query time for both at 925 vectors. Then estimate, with arithmetic, the corpus size at which the swap would start to matter.

### B8. Off-topic questions

Ask "what's the weather in Berlin?". Decide what the correct behaviour is (answering is wrong; is escalating right?), add eval cases for it, and make them pass.

*Pass:* three off-topic cases in the eval and a sentence justifying the behaviour you chose.

---

## What to put in your portfolio from this week

Not "built a RAG chatbot". Three things a reviewer can verify:

1. The filled-in results table with BM25, vector and hybrid, and one paragraph on what it shows.
2. Your failure classification from A2.
3. One before-and-after from A3 or Part B, with the numbers.

A measured improvement, however small, says more than a feature list.
