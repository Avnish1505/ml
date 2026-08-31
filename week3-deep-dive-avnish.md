# WEEK 3 — Deep Dive
### Retrieval & RAG that survives contact with production

**Dates:** Day 15 = 11 Sep · Day 21 = 17 Sep
**Daily:** 120 min learn · 120 min build · 30 min recall · 30 min outreach
**Deliverable:** one RAG system with a golden evaluation set and a before/after ablation table.

---

## The one fact that should govern your whole week

When production RAG fails, **the failure is in retrieval, not generation** — industry analysis through 2026 consistently puts it at roughly three-quarters of failures, and naive vector-only pipelines miss badly. The LLM then writes a confident, well-structured, correctly-formatted answer **grounded in the wrong documents**, which is worse than no answer because it's undetectable without evaluation.

So: **your entire week goes into retrieval quality and measurement, not prompt tuning.** Anyone can build a RAG demo in an evening. Almost nobody can show a table proving theirs got better. That table is the deliverable.

---

## Revision calendar

| Topic (learnt on) | +1 | +3 | +7 | +21 |
|---|---|---|---|---|
| D15 Chunking (11 Sep) | 12 Sep | 14 Sep | 18 Sep | 2 Oct |
| D16 BM25 + hybrid + RRF (12 Sep) | 13 Sep | 15 Sep | 19 Sep | 3 Oct |
| D17 Reranking (13 Sep) | 14 Sep | 16 Sep | 20 Sep | 4 Oct |
| D18 Evaluation metrics (14 Sep) | 15 Sep | 17 Sep | 21 Sep | 5 Oct |
| D19 ANN indexes / vector DBs (15 Sep) | 16 Sep | 18 Sep | 22 Sep | 6 Oct |
| D20 Advanced patterns (16 Sep) | 17 Sep | 19 Sep | 23 Sep | 7 Oct |

---

## DAY 15 — Chunking (where pipelines silently die)

### Why this day exists
Chunking is the least glamorous and highest-leverage decision in RAG. A bad chunk boundary means the answer physically cannot be retrieved, no matter how good your embedding model is. It's an invisible ceiling on everything downstream.

### Core concepts

**1. The central trade-off.** One chunk gets one vector.
- **Too small** → precise match, but the chunk lacks the context needed to answer. Also more chunks = more index, more noise.
- **Too large** → the single vector is an *average* of many topics, so it sits in the middle of embedding space and matches nothing strongly. This is called semantic dilution and it's why "just use bigger chunks" fails.

Practical starting point: **512–1024 tokens with 10–20% overlap.** Then measure and adjust. Never accept a default without measuring.

**2. The strategies, in increasing order of sophistication:**

| Strategy | How | When |
|---|---|---|
| Fixed-size | split every N characters/tokens | never, except as a baseline |
| **Recursive character** | try `\n\n`, then `\n`, then `. `, then ` ` — split on the largest natural boundary that fits | the sane default |
| **Structure-aware** | split on markdown headers, HTML sections, code functions (AST), PDF layout blocks | whenever the document has real structure — usually the winner |
| **Semantic** | embed sentences, walk through, cut where cosine similarity between consecutive sentences drops below a threshold | prose with topic shifts; costs an embedding pass over the corpus |
| **Agentic** | an LLM decides the boundaries | expensive; rarely worth it |

**3. Overlap exists for one reason:** so that an answer sitting on a boundary appears intact in at least one chunk. 10–20% is the usual range. More overlap = more storage and more duplicate retrievals.

**4. Two techniques that beat better chunking:**
- **Contextual retrieval.** Before embedding, prepend a one-line, LLM-generated description of where this chunk sits in its parent document ("This section of the 2024 annual report discusses Q3 revenue in the APAC region."). Costs a one-time LLM pass over the corpus; delivers a large recall improvement, because an isolated chunk saying "revenue fell 12%" is otherwise unretrievable by any query mentioning the region or the year.
- **Small-to-big / parent-document retrieval.** Embed and search over *small* chunks (precision), but hand the *parent* section to the LLM (context). Best of both. Easy to implement, disproportionate gain.

**5. Metadata is not optional.** Every chunk stores: source document, section/heading path, page, date, doc type, and a stable chunk id. You need these for filtering (Day 19), for citations, and — critically — for your golden set on Day 18, where "expected chunk id" is the label.

**6. Tokens, not characters.** Chunk boundaries should respect token boundaries (Week 2, Day 12), because your context budget is measured in tokens and a chunk that splits mid-token wastes the boundary.

### Resources
- **45 min** — LangChain text-splitters documentation (concepts are framework-independent; read for the taxonomy)
- **30 min** — Read about contextual retrieval (Anthropic's engineering write-up on the technique)
- **30 min** — Greg Kamradt's "5 Levels of Text Splitting" (the standard reference on this topic)
- **20 min** — Unstructured.io docs on document partitioning, for the PDF/HTML reality

### Practice
```
practice/d15_chunking.py
```
1. Take one real corpus (your own docs, a set of PDFs, a wiki dump). Chunk it **three ways**: recursive-character, structure-aware, semantic. **Keep all three indexes** — you will compare them on Day 18.
2. Implement semantic chunking yourself: embed sentences, compute consecutive cosine similarity, plot the similarity curve, place the cut points at the troughs. Look at the plot — you'll see the topic boundaries.
3. Implement contextual retrieval: generate a one-line context per chunk with an LLM, prepend it, re-embed. Keep as a fourth index.
4. Implement parent-document retrieval: small chunks in the index, parent section returned.
5. Write the metadata schema you'll use for the rest of the week. Freeze it today — changing it on Day 19 will cost you a full re-index.
6. Manually inspect 20 random chunks. **Count how many are answerable on their own.** That percentage is your ceiling. Write it down.

### Practice resources
- Your own project docs (AegisOps blog, OmitBench README) — a corpus you already understand is the best possible test set
- Any public PDF set: annual reports, arXiv papers, government policy documents

### Self-test
1. What is semantic dilution and why do large chunks cause it?
2. Why does overlap exist, and what does too much overlap cost?
3. Explain contextual retrieval in three sentences and why it improves recall.
4. What is small-to-big retrieval solving?
5. What metadata does every chunk need, and why is chunk id essential?

---

## DAY 16 — BM25, hybrid retrieval, and RRF

### Why this day exists
Pure vector search has a specific, predictable failure mode, and hybrid retrieval is the standard production answer in 2026. Being able to explain *why* — not just "we used hybrid" — separates you immediately.

### Core concepts

**1. What dense retrieval cannot do.** Embeddings capture meaning, so they systematically miss things where the *exact token* is the point:
- product codes, SKUs, error codes (`ERR_5041`)
- rare proper nouns and acronyms not seen in training
- version numbers, dates, legal citations
- exact-phrase requirements

**2. What sparse retrieval cannot do.** BM25 misses paraphrase entirely: query "how do I reset my password" against a document saying "credential recovery procedure" scores near zero.

They fail on **disjoint** sets of queries. That's the whole argument for hybrid — not "it's better," but "their failure modes don't overlap."

**3. BM25, understood not memorised.**
```
score(D,Q) = Σᵢ IDF(qᵢ) · [ f(qᵢ,D) · (k₁+1) ] / [ f(qᵢ,D) + k₁·(1 − b + b·|D|/avgdl) ]
```
Three ideas, and you should be able to state each in one line:
- **IDF** — rare terms are more informative than common ones
- **k₁ (≈1.2–2.0)** — term-frequency **saturation**: the 10th occurrence of a word adds far less than the 2nd. This is what BM25 fixes about raw TF-IDF.
- **b (≈0.75)** — length normalisation: long documents shouldn't win just by containing more words. `b=0` disables it, `b=1` is full normalisation.

**4. Reciprocal Rank Fusion — the merge step.**
```
RRF_score(d) = Σ over rankers r of  1 / (k + rank_r(d)),   k = 60
```
**Why RRF and not score normalisation?** BM25 scores are unbounded and corpus-dependent; cosine similarities live in [-1,1] with a totally different distribution. Normalising them against each other requires tuning that breaks whenever the corpus changes. **Ranks are scale-free.** RRF needs no tuning, no calibration, and no per-corpus constants. `k=60` is the standard default and it comes from the original paper; it damps the influence of the very top ranks so one confident ranker can't dominate.

**Hybrid retrieval with RRF at k=60 is the correct default for a production system in 2026.** Start there. The only remaining question is whether to add a reranker, and the answer is yes (tomorrow).

**5. Weighted hybrid** (α·dense + (1−α)·sparse after normalisation) is the alternative. It can beat RRF *if* you tune α on your data. It also breaks silently when the corpus shifts. Default to RRF; tune weights only when you have the eval harness to prove the gain.

**6. Retrieve wide.** Fetch top-100 from each retriever, fuse, then hand a wide candidate list to the reranker. Recall is cheap at this stage; precision is tomorrow's job. Optimising for precision here is the classic mistake — you can't rerank a document you never retrieved.

### Resources
- **30 min** — The Wikipedia/Elastic explanation of BM25, plus Robertson & Zaragoza's "The Probabilistic Relevance Framework" §3 if you want the derivation
- **20 min** — The original RRF paper (Cormack et al., 2009) — it's two pages and shockingly simple
- **45 min** — Pinecone or Weaviate's hybrid search docs (vendor docs, but the concepts are clean)
- **30 min** — `rank_bm25` source code — it's short, read the whole thing

### Practice
```
practice/d16_hybrid.py
```
1. Implement BM25 **from scratch** in numpy — IDF, term frequencies, the full scoring function. Verify against `rank_bm25`.
2. Sweep `k₁` and `b`. Watch how ranking changes. Find a query where `b=0` and `b=1` give different top results and understand why.
3. Build your dense retriever over the Day 15 index.
4. **Implement RRF yourself.** It's about 15 lines. Do not import it.
5. **Construct the diagnostic set:** find 5 queries where dense wins decisively and 5 where BM25 wins decisively. Then confirm hybrid gets both. **These 10 queries are your demo and your interview story** — concrete evidence beats "we used hybrid because it's best practice."
6. Compare RRF against weighted fusion at α = 0.3, 0.5, 0.7. Note which wins and by how much.

### Self-test
1. Give three query types where dense retrieval fails and BM25 succeeds. Now reverse it.
2. What do `k₁` and `b` control in BM25?
3. Write the RRF formula. Why `k=60`?
4. Why is rank fusion more robust than score normalisation?
5. Why retrieve top-100 rather than top-10 before reranking?

---

## DAY 17 — Reranking (the largest single precision gain)

### Why this day exists
**Re-ranking is the biggest precision improvement you can add to any RAG pipeline.** It's one component, a few hours of work, and it routinely moves systems from "sometimes useful" to "production-grade." If you learn one thing this week that you'll use at work, it's this.

### Core concepts

**1. The architecture (you already know this from Day 13).**
```
query → hybrid retrieval → top-100 candidates → cross-encoder reranker → top 5–10 → LLM
```
The bi-encoder was fast and approximate because doc vectors were precomputed and it never saw query and document together. The cross-encoder now reads query and document **jointly**, with full attention between their tokens, and scores relevance directly. Vastly more accurate — and affordable, because it only runs on 100 candidates, not the whole corpus.

**2. Why "fewer, better chunks" beats "more chunks."**
**"Lost in the middle"**: LLMs attend most reliably to the beginning and end of a long context, and information buried in the middle of a long list of retrieved results gets ignored. So stuffing 50 chunks into the prompt actively *hurts* compared to 5 good ones — you pay more tokens, add more distractors, and the model reads it worse. The reranker's job is to make the top 5 actually be the top 5.

**3. Options.**
- **Open-weight cross-encoders:** BGE-reranker family; the Qwen3 reranker (which pairs with the Qwen3 embedding family, so retrieval and reranking come from one model family)
- **API:** Cohere Rerank — the low-friction default
- **LLM-as-reranker:** prompt a model to score relevance. Flexible, slow, expensive; pointwise/pairwise/listwise variants exist.
- **Late interaction (ColBERT):** a middle option — more accurate than bi-encoder, cheaper than cross-encoder, more storage

**4. Measure the latency, always.** A cross-encoder over 100 candidates is 100 forward passes. Typical additions of 100–500 ms depending on model size and batching. **Write your actual number down.** In interviews, "we added a reranker" is a claim; "we added a reranker, nDCG@10 went from 0.61 to 0.78 at a cost of 180 ms p95" is engineering.

**5. Tuning knobs that matter:** how many candidates you rerank (100 is standard; 50 is cheaper, 200 rarely helps), how many you keep (5–10), and batching for throughput.

**6. Where reranking does NOT help:** if the answer-bearing chunk was never in the top-100, reranking cannot save you. That's a **recall** problem — fix chunking or retrieval. Diagnosing recall-vs-ranking correctly is exactly what Day 18 is for.

### Resources
- **30 min** — Cohere's rerank documentation and their explanation of two-stage retrieval
- **30 min** — "Lost in the Middle" (Liu et al.) — read the abstract and Figure 1, that figure is worth the whole paper
- **30 min** — `sentence-transformers` CrossEncoder documentation
- **20 min** — Skim the ColBERT paper's late-interaction diagram

### Practice
```
practice/d17_rerank.py
```
1. Add a cross-encoder reranker over your Day 16 hybrid top-100.
2. **Manually inspect a re-ranking:** print the top-10 before and after for 5 queries, side by side. Look at what moved and why. This builds intuition no metric can.
3. Measure p50 and p95 latency added, at candidate counts of 20, 50, 100, 200.
4. Sweep the number of retained chunks: 3, 5, 10, 20. Measure answer quality. **You will find a peak, not a monotonic improvement** — that peak is "lost in the middle" showing up in your own data.
5. Build a small **diagnostic split**: for each golden query, check whether the correct chunk was in the top-100 at all. That single number separates recall failures from ranking failures, permanently.

### Self-test
1. Why can't you just use the cross-encoder for the whole corpus?
2. Explain "lost in the middle" and its implication for top-k choice.
3. Your reranker doesn't help. Two possible causes, and how do you tell them apart?
4. What's the latency cost of your reranker, in your own measurements?
5. When is ColBERT the right middle ground?

---

## DAY 18 — Evaluation (the day that makes you employable)

### Why this day exists
Everyone in your applicant pool has built a RAG demo. Almost none of them can produce a measured ablation table. **You already have the rarest form of this skill** — you pre-registered a precision bar on OmitBench, missed it, and published the negative result. Apply that same discipline here and this becomes the strongest thing on your CV.

### Core concepts

**1. The golden set — build it first, not last.**
50–100 question/answer pairs, each labelled with the **chunk id(s) that should be retrieved**. Sources: real user questions if you have them; otherwise generate candidates with an LLM and **hand-verify every single one**. An unverified LLM-generated golden set measures your generator's agreement with itself, not your system's quality. Budget 3 hours. It is the highest-value 3 hours of the week.

Cover: easy single-chunk lookups, multi-chunk synthesis, exact-token queries (the BM25 cases), paraphrase queries (the dense cases), and **questions your corpus genuinely cannot answer** — you must test that the system abstains rather than confabulates.

**2. Retrieval metrics — implement all three yourself.**
- **Recall@k** — was the right chunk retrieved at all? *The ceiling on everything downstream.*
- **Precision@k** — what fraction of retrieved chunks were relevant?
- **MRR** — `mean(1 / rank_of_first_relevant)`. Cares only about the first hit. Right metric when one good chunk suffices.
- **nDCG@k** — `DCG = Σ relᵢ / log₂(i+1)`, divided by the ideal DCG. Position-discounted and handles graded relevance. **The standard for retrieval comparison** — it's the MTEB retrieval metric too.

**Diagnose in this order:** if Recall@100 is low, it's a chunking/retrieval problem. If Recall@100 is high but nDCG@10 is low, it's a ranking problem. Never tune the reranker to fix a recall failure.

**3. Generation metrics — RAGAS.**
- **Faithfulness** — are the claims in the answer supported by the retrieved context? (the anti-hallucination metric)
- **Answer relevancy** — does the answer address the question?
- **Context precision** — are the relevant chunks ranked highly?
- **Context recall** — did retrieval get everything needed for the ground-truth answer?

Commonly cited production bars: **faithfulness > 0.9, answer relevancy > 0.85, context precision > 0.8, context recall > 0.8.** Treat these as starting targets, not laws — the right bar depends on the cost of being wrong in your domain.

**4. LLM-as-judge — and you, specifically, should be sceptical.**
Your OmitBench result is that a judge model outperformed a deterministic detector. Fine. But judges have documented, systematic biases:
- **position bias** (favours whichever option came first)
- **verbosity bias** (favours longer answers)
- **self-preference bias** (favours outputs from the same model family)
- poor calibration at the boundary between "mostly right" and "subtly wrong"

**The discipline:** label 50 examples by hand, run the judge on the same 50, and report **agreement** (Cohen's κ, or MCC — you already know why MCC). Only then are the judge's numbers meaningful. Randomise option order to kill position bias. Report the agreement figure alongside the judge scores, always.

This paragraph is your differentiator. Most candidates say "we used an LLM judge." You can say "we used an LLM judge and validated it against 50 human labels at κ = 0.7, and here's the bias we controlled for."

**5. The ablation table is the deliverable.** Not a demo. This:

| Configuration | Recall@100 | nDCG@10 | Faithfulness | p95 latency | $/1k queries |
|---|---|---|---|---|---|
| Naive (fixed chunks, dense only, top-5) | | | | | |
| + recursive chunking | | | | | |
| + hybrid (BM25 + RRF) | | | | | |
| + cross-encoder rerank | | | | | |
| + contextual retrieval | | | | | |

**Change one variable at a time.** If you change chunking and retrieval together and the number moves, you've learnt nothing. This is the same experimental discipline you already applied in OmitBench — carry it across.

**6. Regression gating.** Wire the eval into CI: every pipeline change re-runs the golden set, and the build fails if nDCG@10 drops more than a threshold. **You have already built exactly this** — the OmitBench GitHub Actions gate with the precision ≥ 0.80 bar. Reuse the pattern. Being the person who gates RAG quality in CI is a rare and immediately hireable trait.

### Resources
- **60 min** — RAGAS documentation, all core metric pages. Read how each metric is actually computed — several are LLM-based and that matters.
- **45 min** — The BEIR paper (zero-shot retrieval evaluation) — abstract, methodology, and the metric definitions
- **30 min** — Read up on LLM-judge bias (the "Judging LLM-as-a-Judge / MT-Bench" work is the standard reference)
- **20 min** — `trec_eval` metric definitions for precise nDCG/MRR semantics

### Practice
```
practice/d18_eval.py
```
1. **Build the 50–100 pair golden set by hand.** Store as JSONL: `{question, expected_chunk_ids, ground_truth_answer, difficulty, type}`.
2. Implement Recall@k, Precision@k, MRR, and nDCG@k **from scratch**. Verify against a library.
3. Run the full ablation table across your four Day 15 chunking indexes × your Day 16/17 retrieval configs.
4. Add RAGAS. Note where its numbers disagree with your retrieval metrics — that disagreement is the interesting part.
5. **Judge validation:** hand-label 50 answers, run the LLM judge on the same 50, compute agreement (κ and MCC). Report it.
6. Compute bootstrap confidence intervals on the nDCG difference between your best and second-best config. **If the CI includes zero, you did not actually improve anything** — say so in your write-up. That honesty is your brand now; don't break it.

### Self-test
1. Write nDCG's definition from memory.
2. Recall@100 = 0.95, nDCG@10 = 0.41. What's broken, and what do you fix?
3. Name three LLM-judge biases and one control for each.
4. Why must the golden set be hand-verified?
5. Why change one variable at a time?
6. Your new config beats the old by 0.02 nDCG with a CI that includes zero. What do you report?

---

## DAY 19 — Vector stores, ANN indexes, and cost

### Why this day exists
This is the infrastructure question in every RAG interview: "how does it scale, and what does it cost?" Most candidates have only ever used Chroma's defaults and cannot answer either half.

### Core concepts

**1. Exact vs approximate.** Brute-force (`IndexFlatIP`) is exact and completely fine up to ~100k vectors on a laptop. ANN is a deliberate trade of some recall for large speed gains. **Know your corpus size before reaching for an ANN index** — people add HNSW to 5,000 documents and lose recall for nothing.

**2. IVF (inverted file).** k-means the vectors into `nlist` cells; at query time search only the `nprobe` nearest cells. `nprobe` is the recall/latency dial: higher = better recall, slower. Fails when the query sits near a cell boundary.

**3. HNSW (the current default).** A multi-layer navigable small-world graph — upper layers are sparse for long jumps, lower layers dense for fine search. Parameters:
- **M** — connections per node. Higher = better recall, more memory.
- **efConstruction** — build-time search breadth. Higher = better graph, slower build.
- **efSearch** — query-time breadth. **The runtime recall/latency dial.**

Be able to say: "efSearch is the knob I'd tune to hit a recall target; M and efConstruction are build-time decisions I'd fix from a recall/memory budget."

**4. Quantization.** Product Quantization or scalar quantization compresses vectors (4–32×) at some recall cost. Combined with Matryoshka truncation (Week 2, Day 13) you can cut storage dramatically. Standard pattern: quantized index for the first pass, exact vectors for rescoring the top candidates.

**5. Filtering — the subtle killer.**
- **Post-filter:** retrieve top-k, then drop the ones failing the filter → you can end up with **fewer than k results**, or zero, and it fails silently.
- **Pre-filter:** restrict the search space before the ANN search. Requires index support, and a very selective filter can degrade graph traversal badly.
Know the difference. It is a favourite interview question because it catches people who only used defaults.

**6. Choosing a store — by deployment model, not by benchmark:**
- **pgvector** — you already run Postgres. One system, transactional, joins with your relational data. Usually the right answer for a startup.
- **Qdrant** — self-hosted, strong filtering, good performance
- **Chroma** — prototyping and local dev
- **Pinecone / managed** — you don't want to operate infrastructure
- **FAISS** — a library, not a database: no persistence layer, no filtering, no API. Fine inside a research pipeline (as in your CineMind work), wrong as a production service.

**7. The cost model — compute this for your own system today.**
```
$/1k queries = embedding(query) + vector search + rerank + LLM input tokens + LLM output tokens
one-time     = corpus embedding + (contextual retrieval LLM pass, if used) + storage/month
```
Then find the dominant term. It is almost always **LLM input tokens**, which is precisely why the reranker pays for itself: sending 5 good chunks instead of 20 mediocre ones cuts your largest cost line *and* improves quality. **Being able to say that sentence with your own numbers behind it is a senior-engineer signal.**

### Resources
- **45 min** — The FAISS wiki, "Guidelines to choose an index" — the best practical document on ANN indexes anywhere
- **30 min** — The HNSW paper: read the abstract and the figures, skip the proofs
- **30 min** — pgvector README and Qdrant's filtering documentation
- **20 min** — Your LLM provider's pricing page. Actually compute your cost per 1k queries.

### Practice
```
practice/d19_index.py
```
1. Index your corpus with FAISS `IndexFlatIP` (exact). Record Recall@10 — **this is your ground truth**.
2. Build HNSW. Sweep `efSearch` = 16, 32, 64, 128, 256. **Plot recall vs latency.** That single curve is the whole topic.
3. Build IVF. Sweep `nprobe`. Compare the curve to HNSW's.
4. Add scalar quantization. Measure the recall loss and the memory saved.
5. Implement pre-filter and post-filter on a metadata field. **Construct a query where post-filtering returns fewer than k results.** Now you'll never confuse them.
6. Write your cost model as an actual spreadsheet or script with your real numbers. Identify the dominant term.

### Self-test
1. HNSW's M, efConstruction, efSearch — what does each control and which is a runtime dial?
2. When is exact search the right choice?
3. Pre-filter vs post-filter — what breaks, and how?
4. When is FAISS the wrong tool?
5. What dominates your cost per 1k queries, and what's the cheapest lever on it?

---

## DAY 20 — Advanced patterns (know the map, build only one)

### Why this day exists
You need to be able to say "we considered GraphRAG and rejected it because…" That sentence, backed by understanding, is worth more than having built it. **Today is mostly reading and one implementation.**

### Core concepts

**1. Query-side techniques.**
- **Query rewriting** — turn a conversational follow-up ("and what about last year?") into a standalone query. **Mandatory for multi-turn RAG.** Most chatbot RAG failures are actually this missing.
- **Query decomposition** — split a multi-part question into sub-questions, retrieve for each. Essential for comparisons and multi-hop.
- **HyDE** (Hypothetical Document Embeddings) — have the LLM write a *hypothetical answer*, then embed **that** and search with it. Why it works: queries and documents are written in different registers ("how do I fix X?" vs "To resolve X, configure…"), and a hypothetical answer lives in document-space. It closes the query/document vocabulary asymmetry.
- **Multi-query fan-out** — generate 3–5 paraphrases, retrieve for each, fuse with RRF. Cheap, reliable recall gain.

**2. Agentic RAG.** Replace the linear pipeline with a loop: plan → retrieve → **judge sufficiency** → re-retrieve if needed → synthesize. Highest quality ceiling, handles multi-hop, can self-correct bad retrieval. Costs: **2–10 seconds per query, sometimes more**, and several times the tokens. Worth it when accuracy is non-negotiable (legal, medical, financial) — which is precisely the AegisOps use case, so you have a real argument to make here.

**3. GraphRAG.** Extract entities and relationships into a graph, build community summaries, query over the structure. Genuinely better for:
- "what connects X and Y?"
- global questions requiring a whole-corpus view ("what are the main themes?")
Costs: expensive to construct, hard to keep fresh, and it fails on ordinary lookup questions that plain retrieval handles for a fraction of the price. **Know when to say no to it** — that's the more valuable skill.

**4. Adaptive routing.** Classify the incoming query, then route: simple lookup → cheap naive path; multi-hop → agentic path; relationship question → graph. Keeps average cost low while preserving the quality ceiling for hard queries. This is the shape of a mature 2026 system.

**5. The decision tree — write your own version of this today:**
```
Is the answer inside one chunk?
  ├─ Yes → naive retrieval + reranker
  └─ No → Does it need 2–3 documents?
       ├─ Yes → hybrid + rerank + query decomposition
       └─ No → Does it need reasoning across many documents?
            ├─ relationships → GraphRAG
            ├─ multi-step → agentic RAG
            └─ mixed traffic → adaptive routing
```

**6. Guardrails that belong in every RAG system:**
- cite sources with chunk ids in the answer (non-negotiable — it's how users verify you)
- if retrieval scores fall below a threshold, **say you don't know** rather than answering from parametric memory
- strip/neutralise prompt injection in retrieved content (a retrieved document is untrusted input)
- cap context tokens and total cost per query

### Resources
- **30 min** — The HyDE paper (short, clear)
- **45 min** — Microsoft's GraphRAG documentation/blog, plus one critical write-up on its costs
- **30 min** — Self-RAG and Corrective RAG (CRAG) papers — abstracts and architecture figures only
- **30 min** — LlamaIndex documentation on query transformations and routing

### Practice
```
practice/d20_advanced.py
```
1. Implement **query rewriting** for multi-turn. Test with a 3-turn conversation containing a pronoun reference. Measure retrieval quality with and without.
2. Implement **multi-query fan-out + RRF**. Add it as a row in your ablation table.
3. Implement **HyDE**. Add it as a row. **In many corpora it will not help** — report that honestly rather than including it because it's fashionable.
4. Implement a **sufficiency check**: after retrieval, ask the model whether the context suffices; if not, re-retrieve with a rewritten query. Cap at 2 loops. Measure the latency cost.
5. Implement the **abstention threshold**: below a retrieval score, refuse. Test it against the unanswerable questions you deliberately put in your golden set.
6. Write your decision tree as a one-pager for your blog post.

### Self-test
1. Explain HyDE and the exact asymmetry it fixes.
2. When is GraphRAG worth its construction cost, and when is it a mistake?
3. What does agentic RAG buy and what does it cost, in seconds and dollars?
4. Why is query rewriting mandatory for multi-turn?
5. How does your system behave on a question the corpus cannot answer?
6. Retrieved documents are untrusted input. What's the risk and the mitigation?

---

## DAY 21 — Ship it and write it up

### Why this day exists
Unshipped work is invisible work. And your write-up is a hiring artifact, not a diary.

### Build
- [ ] FastAPI service: `/query`, `/health`, and **`/eval`** (runs the golden set on demand)
- [ ] Dockerfile, and a `make` target that runs the full eval
- [ ] GitHub Actions: run the golden-set eval on every push, **fail the build if nDCG@10 drops beyond a threshold** — the same gating pattern you built in OmitBench
- [ ] Structured logging: query, retrieved chunk ids, scores, latency per stage, tokens, cost
- [ ] README with the ablation table **at the top**, plus limitations
- [ ] Deploy somewhere that stays up. **A dead demo link is worse than no demo link** — you already got burnt by this once in your resume audit.

### Write
The blog post. Structure it like this and it will outperform 95% of RAG posts on the internet:

1. **The problem and the corpus** — concrete, specific, with real numbers
2. **The naive baseline and its measured failure rate** — start with what didn't work
3. **The ablation table** — one variable at a time, with confidence intervals
4. **What surprised you** — including the techniques that *didn't* help. HyDE not working on your corpus is a more interesting finding than hybrid working.
5. **Cost and latency** — real numbers per 1k queries
6. **Limitations** — what you know is still wrong
7. **What you'd do with more time**

**Lead with the failure.** It's what made OmitBench credible and it's your established voice now. A post titled "Techniques that didn't improve my RAG pipeline" gets read; "How I built a RAG chatbot" does not.

### Publish
- [ ] Blog on your site + LinkedIn post pointing to it
- [ ] Repo public, README polished, `git clone && make test` verified on a clean machine
- [ ] Send it to 3 of your warm contacts with a specific ask, not a generic one

### Week 3 pass/fail
- [ ] Golden set of 50–100 hand-verified pairs exists
- [ ] I implemented BM25, RRF, Recall@k, MRR and nDCG **from scratch**
- [ ] I have an ablation table with one variable changed per row
- [ ] I validated my LLM judge against human labels and reported the agreement
- [ ] I have measured latency and cost per 1k queries
- [ ] I can plot recall vs efSearch for my own index
- [ ] The service is deployed and the link works
- [ ] Blog post published
- [ ] 7 more outreach messages sent

**7/9 or better = Week 3 worked.**

---

## PART C — Fixed resource list for Week 3

**Docs (primary source, better than any tutorial):** RAGAS · sentence-transformers · FAISS wiki · pgvector · Qdrant · LangChain text-splitters
**Papers:** RRF (Cormack 2009) · HyDE · Lost in the Middle · ColBERT · BEIR · Self-RAG/CRAG · HNSW
**Datasets:** BEIR · MS MARCO subset · **your own corpus** (best option — you can judge relevance yourself)
**Libraries:** `sentence-transformers`, `rank_bm25`, `faiss-cpu`, `ragas`, `pgvector`

Everything here is free. None of it requires a subscription to anything.
