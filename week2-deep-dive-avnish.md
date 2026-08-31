# WEEK 2 — Deep Dive
### Transformers, tokenization, embeddings — built from a blank file

**Dates:** Day 8 = 4 Sep · Day 14 = 10 Sep
**Daily:** 120 min learn · 120 min build · 30 min recall · 30 min outreach
**End-of-week test:** draw a transformer block from memory with every tensor shape labelled.

> **Prerequisite check.** If you cannot rebuild micrograd from a blank file in under 40 minutes, do not start this week. Repeat Day 5. Everything below is backprop wearing a costume, and skipping the foundation is exactly how people end up with LLM projects they can't explain.

---

## Revision calendar for this week

| Topic (learnt on) | +1 | +3 | +7 | +21 |
|---|---|---|---|---|
| D8 Embedding tables / MLP LM (4 Sep) | 5 Sep | 7 Sep | 11 Sep | 25 Sep |
| D9 Activations / init / manual backprop (5 Sep) | 6 Sep | 8 Sep | 12 Sep | 26 Sep |
| D10 Attention (6 Sep) | 7 Sep | 9 Sep | 13 Sep | 27 Sep |
| D11 GPT block / KV cache (7 Sep) | 8 Sep | 10 Sep | 14 Sep | 28 Sep |
| D12 Tokenization / BPE (8 Sep) | 9 Sep | 11 Sep | 15 Sep | 29 Sep |
| D13 Embeddings / bi vs cross encoder (9 Sep) | 10 Sep | 12 Sep | 16 Sep | 30 Sep |

Same four reps as Week 1: **blank page → derive → blank file → explain out loud on camera.** Nothing here changes that.

---

## DAY 8 — What an embedding actually is (makemore 1–2)

### Why this day exists
Half the industry says "embeddings" without knowing that an embedding table is just a matrix you index into, trained by gradient descent like any other weight. Get this straight now and Day 13 becomes easy.

### Core concepts

**1. The bigram model.** Count how often character `b` follows character `a` in a 27×27 matrix, normalise rows to probabilities, sample. No neural network at all. Then the key reveal: **a neural network with a single linear layer and cross-entropy loss converges to exactly these counts.** Counting and learning are the same thing here. That equivalence is the cleanest intuition in all of ML.

**2. Negative log likelihood.** `L = −(1/n) Σ log P(actual_next_char)`. Minimising NLL = maximising the probability the model assigns to the real data. Perplexity is just `exp(NLL)` — "how many options is the model effectively choosing between." A perplexity of 1 means certainty; a perplexity of 27 on characters means it learnt nothing.

**3. Smoothing.** Add 1 to all counts (or add a small L2 penalty on the neural version) so unseen bigrams don't get probability 0 — because `log(0) = −inf` and your loss explodes. Regularization and smoothing are the same idea in two vocabularies.

**4. The embedding table (Bengio 2003, the MLP model).** `C` is a `(vocab_size, n_embd)` matrix. `C[idx]` is a lookup — and **a lookup is mathematically identical to a one-hot vector times the matrix.** That's why it's differentiable and trainable. There is no magic: an embedding is a row of learned weights, and gradient descent pushes rows of characters that behave similarly closer together.

**5. Context window, seen for the first time.** The MLP takes a fixed block of previous characters (say 3), concatenates their embeddings, feeds an MLP. The limitation is obvious and it's exactly the limitation attention will solve on Day 10: fixed context, no weighting, positions treated by concatenation order.

### Resources
- **75 min** — Karpathy makemore Part 1 ("The spelled-out intro to language modeling")
- **75 min** — Karpathy makemore Part 2 ("MLP")
- **15 min** — Skim Bengio et al. 2003, "A Neural Probabilistic Language Model" — §2 only, so you see where the MLP came from

### Practice (blank file)
```
practice/d8_bigram_mlp.py
```
1. Bigram by counting. Sample 20 names. Compute the NLL.
2. Bigram as a one-layer network. Train it. **Confirm the loss converges to the counting model's NLL.** If it doesn't, your loss is wrong.
3. MLP with an embedding table, block_size=3. Train to loss < 2.2.
4. Plot the learned 2D embeddings (set n_embd=2 for this experiment). Look at which characters cluster — vowels group together. **You just watched representation learning happen.**
5. Implement a train/val/test split properly and find where val loss stops improving.

### Practice resources
- `names.txt` from the makemore repo (32k names) — or any word list

### Self-test
1. Why is `C[idx]` the same as a one-hot matmul?
2. What is perplexity, and what does perplexity = 5 mean concretely?
3. Why does an unsmoothed count model produce infinite loss?
4. What are the two hard limitations of the fixed-window MLP language model?

---

## DAY 9 — Activations, initialization, and manual backprop

### Why this day exists
This is where you learn to *debug* training instead of guessing. Every "my loss isn't going down" question in an interview gets answered from this day. It also closes the loop on Week 1 Day 6.

### Core concepts

**1. The "hockey stick" loss curve is a bug, not a feature.** At init, if the output logits aren't roughly uniform, the model spends the first hundreds of steps just squashing the confidently-wrong logits. Fix: initialise the final layer's weights small (× 0.01) and its bias to zero, so the initial loss equals `−log(1/vocab_size)`. **Compute that expected initial loss before every training run** — it's the cheapest bug detector in deep learning.

**2. Saturated tanh kills learning.** Plot the histogram of `tanh` pre-activations. If most values sit at ±1, the local gradient `1 − tanh²` is ≈ 0 and no gradient flows back. Cause: weights initialised too large. This is the concrete, visible version of "vanishing gradients."

**3. Kaiming/He init, properly understood.** You want `Var(output) ≈ Var(input)` layer over layer. For a linear layer, `Var(out) = fan_in · Var(w) · Var(x)`, so `Var(w) = 1/fan_in` preserves it. ReLU zeroes half the units and halves the variance, hence the factor of 2: `Var(w) = 2/fan_in`. That's the entire derivation — three lines.

**4. BatchNorm and why it's a mess.** It normalises each feature across the batch, which:
- couples examples in a batch to each other (a bug surface, and a subtle train/test mismatch)
- needs running statistics for inference
- breaks with tiny batches
It works because it makes the loss landscape better-conditioned, not because of "internal covariate shift" (that original explanation has been largely debunked). Know that. It's a good interview answer.

**5. Manual backprop through the whole thing.** Karpathy's Part 4 makes you derive gradients through cross-entropy, BatchNorm, matmul and tanh by hand. It is painful. It is also the exercise that converts "I watched backprop" into "I know backprop." Non-negotiable.

**6. The diagnostic toolkit** (write this list on your cheat sheet):
- expected loss at init
- activation histograms per layer
- gradient histograms per layer
- **update-to-data ratio**: `(lr · grad).std() / param.std()`, should sit around 1e-3. Too high = learning rate too big; too low = stuck.

### Resources
- **60 min** — Karpathy makemore Part 3 ("Activations & Gradients, BatchNorm")
- **75 min** — Karpathy makemore Part 4 ("Becoming a Backprop Ninja") — do the exercises, don't just watch
- **20 min** — Optional: skim "How Does Batch Normalization Help Optimization?" (Santurkar et al.) abstract + conclusion

### Practice
```
practice/d9_diagnostics.py
```
1. Take your Day 8 MLP. Print the expected loss at init and the actual loss at init. Fix the gap.
2. Plot activation histograms for every layer, before and after fixing the init scale.
3. Plot gradient histograms. Find a dead layer.
4. Implement the update-to-data ratio plot across training steps. Tune the LR using it, not by guessing.
5. Do at least the first half of the "backprop ninja" exercises by hand.
6. Add BatchNorm, then LayerNorm. Compare the histograms and the loss curves.

### Self-test
1. What loss should a well-initialised char-level model show at step 0, for vocab size 27? *(−log(1/27) ≈ 3.30)*
2. Your tanh activations are all at ±1. What's wrong and what do you change?
3. Derive He init in three lines.
4. Give the real reason BatchNorm helps, not the textbook one.
5. What is the update-to-data ratio and what value are you targeting?

---

## DAY 10 — Attention (theory day, no video)

### Why this day exists
This is the single most-asked concept in any LLM role. It is also the one people fake best and get caught on fastest, because the follow-up is always "what are the tensor shapes?"

### Core concepts

**1. The mechanism.** Every token emits three vectors:
- **Query** — what I'm looking for
- **Key** — what I contain
- **Value** — what I'll give you if you attend to me

```
Q = X·W_Q    K = X·W_K    V = X·W_V        X: (B, T, C)
scores  = Q @ K.transpose(-2,-1) / √d_k     → (B, T, T)
scores  = scores.masked_fill(tril == 0, -inf)   # causal
weights = softmax(scores, dim=-1)           → (B, T, T)
out     = weights @ V                       → (B, T, head_size)
```
**Memorise these shapes.** Write them out five times today. `(B, T, T)` is the attention matrix — it is a T×T table saying "how much does token i care about token j."

**2. Why divide by √d_k — derive it.** If the components of q and k are roughly independent with unit variance, their dot product over `d_k` dimensions has variance `d_k`, so a standard deviation of `√d_k`. Large-magnitude scores push softmax toward a one-hot distribution, where its gradient is nearly zero. Dividing by `√d_k` restores unit variance and keeps softmax in its useful range. **This is a variance-control argument, exactly like He init.** Say it that way and you sound like someone who understands the field rather than the tutorial.

**3. The causal mask.** Set the upper triangle to `−inf` **before** softmax (not after — after softmax you'd have to renormalise). A decoder token may only see the past. This is the one line that separates a language model from BERT.

**4. Self vs cross attention.** Self: Q, K, V all come from the same sequence. Cross: Q comes from the decoder, K and V from the encoder. Cross-attention is how translation models and most multimodal models wire two streams together.

**5. Multi-head — why not one big head?** Split `C` into `h` heads of size `C/h`, run attention in each independently, concatenate, then project with `W_O`. A single head produces **one** attention distribution, which averages all relationship types into one pattern. Multiple heads let different heads specialise — one tracks syntax, another tracks coreference, another position. Same parameter count, strictly more expressive.

**6. The full block.** Pre-norm is the modern default:
```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```
MLP is `Linear(C → 4C) → GELU → Linear(4C → C)`. Two things to understand:
- **Residual connections are a gradient highway.** The `x +` gives backprop an unobstructed path to early layers. Without them, deep transformers don't train.
- **Pre-norm vs post-norm:** the original paper was post-norm and needed a learning-rate warmup to be stable. Pre-norm keeps the residual stream clean and trains stably. Everything modern is pre-norm.

**7. Complexity — the fact that shapes the entire industry.** Attention is `O(T²·d)` in time and `O(T²)` in memory for the attention matrix. Doubling context quadruples attention cost. This is why:
- long context is expensive
- **FlashAttention** exists (tiles the computation and uses an online softmax so the T×T matrix is never materialised in HBM — it's an IO-optimisation, not an approximation, and the output is exact)
- everyone is hunting for sub-quadratic alternatives

**8. Positional information.** Attention is permutation-invariant — without position info, "dog bites man" and "man bites dog" are identical to it. Options:
- sinusoidal (original paper), learned absolute (GPT-2)
- **RoPE** (rotary): rotate Q and K by an angle proportional to position, so their dot product ends up depending on **relative** position. This is what nearly every modern LLM uses. Know the name and the one-line reason.
- ALiBi: add a distance-proportional penalty to attention scores

**9. KV heads.** MQA (one shared K/V head) and **GQA** (grouped: several query heads share one K/V head) exist purely to shrink the KV cache. Tomorrow you'll see why that matters.

### Resources
- **60 min** — "Attention Is All You Need", §3 only. Read it **twice**. Second read, annotate shapes in the margin.
- **45 min** — Jay Alammar, "The Illustrated Transformer" — the best diagrams available
- **30 min** — 3Blue1Brown's transformer/attention videos (visual intuition for QKV)
- **15 min** — Skim the RoPE paper abstract + Figure 1

### Practice (no video open)
```
practice/d10_attention.py
```
1. Implement single-head self-attention in raw numpy. Print the shape after every line and check against your written list.
2. Add the causal mask. Print the attention matrix for a 5-token sequence and verify the upper triangle is zero after softmax.
3. Implement multi-head by reshaping to `(B, h, T, head_size)`. Confirm output shape returns to `(B, T, C)`.
4. **Ablation:** remove the `/√d_k`. Print the softmax outputs with `d_k = 64`. Watch them collapse to near-one-hot. That's your permanent memory of why the scaling exists.
5. Hand-draw a full transformer block on paper with every shape labelled. Photograph it for your cheat sheet.

### Self-test
1. Write the attention formula from memory, with shapes.
2. Derive why `√d_k` and not `d_k` or `d_k²`.
3. Why mask before softmax and not after?
4. Multi-head with h=8 vs single head with the same total params — what's the actual gain?
5. Why does attention need positional encoding at all?
6. Context 2k → 8k. What happens to attention cost, and why?
7. What does FlashAttention change, and what does it *not* change? *(memory IO; not the maths — output is exact)*

---

## DAY 11 — Build GPT from a blank file

### Why this day exists
Day 10 was theory. Today it becomes real. After today, "have you implemented a transformer?" has a true answer.

### Core concepts

**1. The assembly.** Token embedding + position embedding → N × transformer blocks → final LayerNorm → linear head to vocab → cross-entropy. That's the whole model. GPT-2 is this with bigger numbers.

**2. Weight tying.** Sharing the token-embedding matrix with the output projection saves parameters and usually improves quality — input and output token spaces are the same space.

**3. Sampling matters as much as the model.**
- **Temperature:** divide logits by `T` before softmax. `T<1` sharpens (more deterministic), `T>1` flattens (more random). `T→0` is greedy.
- **Top-k:** keep only k highest-probability tokens, renormalise.
- **Top-p / nucleus:** keep the smallest set whose cumulative probability exceeds p. Adapts to how confident the distribution is, which is why it usually beats top-k.
- Greedy decoding produces repetitive text — the model's most likely token is often "the same thing again."

**4. KV cache — the inference optimisation you must be able to explain.**
Naively, generating token `t` re-runs attention over all `t` previous tokens, recomputing every K and V from scratch. Total cost for a sequence: `O(T²)` *per generated token*, `O(T³)` overall. But K and V for past tokens **never change**. Cache them; each new token then only computes its own Q, K, V and attends against the cache — `O(T)` per token.

The cost is memory:
```
KV cache bytes ≈ 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_dtype
```
This is why long-context serving is memory-bound, why GQA/MQA exist (fewer K/V heads → smaller cache), and why vLLM's paged attention was a big deal (it manages this memory like OS page tables instead of one contiguous over-allocated block). **This is a very common senior-level interview question.** You will have implemented it, so answer from experience.

### Resources
- **120 min** — Karpathy, "Let's build GPT: from scratch, in code, spelled out." Watch once at 1x.
- Reference after building: `karpathy/nanoGPT` — read `model.py` line by line, it's ~300 lines

### Practice (the day's real work)
```
practice/d11_gpt.py
```
1. **Close the video.** Build from an empty file: `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`. Train on any text (Tiny Shakespeare, or Hindi text for a twist).
2. Get val loss below 1.6 on Shakespeare. Generate 500 characters. It should look like broken English with correct structure.
3. Implement temperature, top-k, and top-p sampling yourself. Generate the same prompt at `T = 0.2, 0.8, 1.5`. Read the three outputs. **Now you understand temperature for life.**
4. **Implement the KV cache in your generation loop.** Time generation of 200 tokens with and without it. Write the two numbers down — that measured speedup is an interview anecdote.
5. Compute your model's parameter count by hand from the config, then verify with `sum(p.numel() for p in model.parameters())`. Match them exactly.
6. Scaling mini-experiment: train at 2, 4, and 6 layers. Plot val loss vs parameter count. You've just reproduced the shape of a scaling law on a laptop budget.

### Self-test
1. Draw the full GPT forward pass with shapes, from memory.
2. What does the KV cache store, why is it valid, and what does it cost?
3. Why do temperature and top-p exist? When would you use `T=0`?
4. Compute the parameter count of a 6-layer, 384-dim, 6-head model.
5. Why is weight tying reasonable?
6. Your generated text loops the same phrase forever. Three causes?

---

## DAY 12 — Tokenization

### Why this day exists
Tokenization causes more real production bugs than any other LLM component and almost nobody can explain it. It is a cheap way to be visibly better than other candidates. It also directly affects your costs — and your Hindi-language use cases.

### Core concepts

**1. BPE, the algorithm.** Start with a vocabulary of the 256 byte values. Then repeat: find the most frequent adjacent pair in the corpus, merge it into a new token, add it to the vocabulary. Store the **ordered list of merges**. To encode new text, apply those merges in the same order. That's it — the entire algorithm is a loop with a `Counter`.

**2. Why byte-level.** Starting from bytes means the tokenizer can never fail on unseen input — every possible string is representable. No `<UNK>` token, ever.

**3. The consequences (this is the valuable part):**
- **Arithmetic is bad** because numbers tokenize inconsistently: "1234" might be one token while "1235" is two. The model isn't seeing digits.
- **Non-English costs more.** Hindi/Devanagari text uses substantially more tokens per word than English in most tokenizers, because the merge list was learnt on English-heavy data. You pay more per sentence *and* burn context faster. Directly relevant to any Indian-language product you build.
- **Trailing whitespace bugs.** `"hello"` and `"hello "` tokenize differently, and a trailing space in your prompt can measurably degrade output.
- **Code tokenizes badly** in older tokenizers — indentation eating whole tokens is why GPT-2 was poor at Python.
- **Glitch tokens** (the `SolidGoldMagikarp` family): tokens present in the vocabulary but nearly absent from training data, so their embeddings are essentially untrained garbage. Feeding them produces bizarre behaviour.
- **Chunk boundaries in RAG** are token boundaries, not character boundaries. This matters next week.

**4. Vocabulary size is a trade-off.** Bigger vocab → shorter sequences (cheaper attention, since attention is quadratic in T) but a larger embedding table and softmax. Typical modern range: 32k–200k.

**5. SentencePiece vs tiktoken.** SentencePiece (used by Llama, T5) operates on raw text with a unigram or BPE model and handles whitespace as a real character (`▁`). tiktoken is OpenAI's fast BPE. HuggingFace `tokenizers` is the general library. Know the difference between the *algorithm* (BPE, WordPiece, Unigram) and the *library*.

### Resources
- **120 min** — Karpathy, "Let's build the GPT Tokenizer" — the most useful two hours available on this topic anywhere
- **20 min** — Play with the OpenAI tokenizer visualiser. Paste English, Hindi, Python code, and a long number. **Record the token counts.** Those four numbers are a great thing to cite in an interview.

### Practice
```
practice/d12_tokenizer.py
```
1. Implement BPE training from scratch: `get_stats()`, `merge()`, the training loop, and the merge list.
2. Implement `encode()` and `decode()`. Assert `decode(encode(s)) == s` for 100 random strings **including emoji and Devanagari**. This round-trip test will find your bugs.
3. Train two tokenizers with vocab sizes 300 and 1000 on the same corpus. Compare compression ratios.
4. **Measurement exercise:** tokenize the same paragraph in English and in Hindi with a standard tokenizer. Compute the ratio. Write it in your notes — it's a real number you can quote about multilingual cost.
5. Tokenize `"1234567"` and `"1,234,567"`. Look at the split. Now you can explain LLM arithmetic failure from first principles.

### Self-test
1. Describe BPE training in five sentences.
2. Why byte-level rather than character-level?
3. Why are LLMs bad at arithmetic and at counting letters in a word?
4. Why does Hindi cost more per sentence than English?
5. What is a glitch token and how does one come to exist?
6. Vocab 32k vs 128k — what do you gain and lose?

---

## DAY 13 — Embeddings, properly (the bridge into Week 3)

### Why this day exists
This is the load-bearing day for all of RAG. You already built a two-tower retrieval model with FAISS (CineMind, Recall@10 = 0.038, ~7.6× random). Today you learn what you actually built — and that reframing turns a paused side project into a strong interview story.

### Core concepts

**1. Bi-encoder vs cross-encoder — the most useful distinction in retrieval.**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | query and doc **separately** | query + doc **concatenated** |
| Output | one vector each → cosine | a single relevance score |
| Doc vectors precomputable? | **Yes** | No |
| Cost per query over N docs | 1 encode + N cheap dot products | **N full forward passes** |
| Accuracy | lower | **much higher** |

Bi-encoders can search millions of documents because the docs were embedded offline. Cross-encoders see query and document tokens attending to each other, which is far more accurate but means you cannot scan a corpus with one. **Hence the entire two-stage architecture of modern retrieval: bi-encoder fetches top-100, cross-encoder reranks to top-10.** That sentence is the whole of next Wednesday.

Middle ground: **ColBERT / late interaction** — store per-token vectors and score with MaxSim. More accurate than a single vector, more storage, more compute.

**2. What "similar" means.**
- **Cosine** = dot product of L2-normalised vectors. Angle only, magnitude ignored.
- **Dot product** keeps magnitude, which some models use to encode importance or frequency.
- **Euclidean** on normalised vectors is monotonically related to cosine — same ranking.
- **Rule:** use whatever the model was trained with. Using cosine on a model trained with unnormalised dot product silently degrades your retrieval, and it fails quietly, which is the worst kind.

**3. How embeddings are trained: contrastive learning.** InfoNCE loss:
```
L = −log[ exp(sim(q, d⁺)/τ) / Σⱼ exp(sim(q, dⱼ)/τ) ]
```
Pull the query toward its positive document, push it away from negatives. Two things dominate quality:
- **Hard negatives.** Random negatives are too easy — the model learns nothing from distinguishing "how to reset my password" from "recipe for dal". Mine negatives that are *plausible but wrong*. This is the single biggest lever in embedding quality, and it's what most people skip.
- **Temperature τ.** Low τ sharpens the contrast and emphasises hard negatives; too low destabilises training.
- **In-batch negatives:** treat every other document in the batch as a negative. This is why bigger batches help contrastive training so much.

**This is exactly what your two-tower CineMind model did.** You can now say: "I trained a bi-encoder with contrastive loss and in-batch negatives, indexed with FAISS, and got Recall@10 at 7.6× the random baseline; the obvious next lever was hard-negative mining." That's a senior-sounding sentence and it's true.

**4. Choosing a model without embarrassing yourself.**
The MTEB leaderboard is the standard reference and it is <cite index="38-1">the most misread benchmark in the space: the headline number is an unweighted average across seven task categories, but if you're building RAG you care about the Retrieval sub-score (nDCG@10) — a model with a headline of 65 but retrieval of 55 will lose to one with a headline of 62 and retrieval of 60</cite>. **Always filter to the retrieval sub-leaderboard.**

Current landscape (verify before you cite it — this moves every few months):
- <cite index="39-1">Qwen3-Embedding (0.6B / 4B / 8B) is the open-weight model to beat, and the same team ships a matching Qwen3 reranker, so you can build a two-stage pipeline from one family</cite>
- <cite index="39-1">EmbeddingGemma (308M, trained with Matryoshka, 768 dims truncatable to 512/256/128) is built for on-device</cite>
- <cite index="39-1">OpenAI's text-embedding-3-large is the low-friction API default; the small variant plus a reranker works if budget matters</cite>
- BGE-M3 for serious multilingual work
- <cite index="38-1">For specialised domains — biomedical, financial filings, source code — domain-specific models almost always beat general-purpose ones</cite>

**5. Matryoshka representation learning (MRL).** The model is trained so that *prefixes* of the vector are themselves valid embeddings. So you can truncate 1024 → 256 dims and keep most of the quality, cutting storage and search cost by 4×. Very practical; know the name.

**6. The decisive rule.** <cite index="35-1">No benchmark captures your corpus. Document style, query phrasing and domain vocabulary interact in ways that shape retrieval quality, so the leaderboard is a starting point and the decision should come from small-scale evaluation on your own data with MRR and nDCG.</cite> When someone asks "which embedding model should we use?", the correct senior answer is **"I'd run three on a 50-query golden set from our corpus and measure nDCG@10"** — not a model name.

### Resources
- **45 min** — Sentence-Transformers documentation, "Training Overview" and "Loss Functions" pages — genuinely excellent
- **30 min** — Skim the MTEB retrieval leaderboard on HuggingFace. Note the top 5 and their sizes.
- **30 min** — Read about hard negative mining (Sentence-Transformers docs cover it well)
- **20 min** — Skim the ColBERT paper abstract + Figure 1

### Practice
```
practice/d13_embeddings.py
```
1. Embed 500 sentences from a real corpus with a small open model. Reduce with UMAP/t-SNE, colour by category, plot.
2. **Find the failure cases:** pairs with high cosine similarity but different meaning (negation is a classic — "the flight was cancelled" vs "the flight was not cancelled" often embed very close). Collect five. These are gold in interviews.
3. Implement cosine, dot and Euclidean yourself. Show that on normalised vectors, cosine and Euclidean give identical rankings.
4. Implement InfoNCE loss from scratch and train a tiny two-tower model on any query→document pairs.
5. Add hard negatives (mine them by retrieving top-k with the current model and taking the wrong ones). **Measure Recall@10 before and after.** This one experiment is a whole blog post.
6. Take a Matryoshka-capable model, truncate 768 → 256 dims, and measure the retrieval quality drop. Report the number.

### Practice resources
- MS MARCO (small subset), BEIR benchmark datasets, or your own project corpus
- `sentence-transformers`, `faiss-cpu`

### Self-test
1. Bi-encoder vs cross-encoder — architecture, cost, accuracy, and when to use each.
2. Write the InfoNCE loss from memory.
3. Why are hard negatives more valuable than more random negatives?
4. Why can cosine vs dot silently break your retrieval?
5. Why is MTEB's headline score misleading for RAG?
6. **"Which embedding model should we use?"** — give the senior answer.
7. What is Matryoshka training and what does it buy you?

---

## DAY 14 — The LLM mental model + consolidation

### Why this day exists
You need one coherent story from raw internet text to ChatGPT. Without it, every LLM question gets answered in fragments.

### Core concepts

**1. The three (really four) stages:**

| Stage | Data | Objective | What it produces |
|---|---|---|---|
| **Pretraining** | trillions of web tokens | next-token prediction | a **document simulator** — not an assistant. It will happily continue your question with more questions. |
| **Midtraining / SFT** | curated conversations | next-token prediction on assistant turns | something that behaves like an assistant; learns the chat format and refusal behaviour |
| **RLHF / RLAIF** | human or AI preference pairs | maximise a learned reward | tone, helpfulness, harmlessness; optimises what humans *prefer*, not what is *true* |
| **RL with verifiable rewards** | maths/code with checkable answers | reward = correct or not | reasoning ability; no reward model needed because correctness is checkable |

**2. PPO vs GRPO.** PPO needs a separate learned value network as a baseline — expensive and fiddly. **GRPO** (used in nanochat and DeepSeek-R1) samples a *group* of G responses to the same prompt and uses the group's mean reward as the baseline. No value network. Simpler, cheaper, and it works. Know this contrast; it's a current, frequently-asked distinction.

**3. Where hallucination comes from.** The pretraining objective rewards *plausible continuation*, never "I don't know." Nothing in next-token prediction teaches abstention. Models must be explicitly trained to express uncertainty, and even then calibration is imperfect. **This is the honest framing of why RAG exists** — you're grounding the generation in retrieved evidence because the objective itself does not reward truth.

**4. Base vs instruct models.** A base model given "What is the capital of France?" may reply with a list of similar quiz questions. That's not a failure — it's a correct document continuation. The instruct model is the one that's been taught the conversation contract.

**5. Context engineering > prompt engineering.** The 2026 framing: the job isn't clever phrasing, it's deciding **what information occupies the context window** — retrieved chunks, tool outputs, conversation history, system instructions — under a fixed token budget. That's the discipline that makes Weeks 3 and 4 work.

### Resources
- **210 min** — Karpathy, "Deep Dive into LLMs like ChatGPT" (3.5 hr). Watch at 1.5x with notes. This is the single best overview in existence of the whole pipeline.
- **30 min** — Skim the `karpathy/nanochat` repo *structure* (do not train it — it costs roughly $100 on an 8×H100 node). Trace the directory layout: tokenizer → pretraining → midtraining → SFT → RL → inference → web UI. <cite index="2-1">It's about 8,000 lines covering tokenizer training in Rust, transformer pretraining on FineWeb, supervised finetuning, optional RL with GRPO, KV-cached inference, and both CLI and browser interfaces.</cite> Reading a complete, minimal, real pipeline end-to-end is worth more than three tutorials.

### Practice / consolidation
1. **Write the pipeline diagram from memory**, one page: data → tokenizer → pretrain → SFT → RL → inference. Annotate what each stage costs and what it buys.
2. Write 300 words: "Why do LLMs hallucinate, and why is RAG the standard mitigation?" — no jargon, as if for a product manager.
3. **Rebuild your GPT from a blank file for the second time.** Should take under 60 min now.
4. Handwrite the Week 2 one-page A4 cheat sheet: attention formula + shapes, KV cache formula, BPE loop, bi vs cross encoder, InfoNCE, the four training stages.
5. Record yourself for 3 minutes: "Explain the transformer architecture." Watch it back. Redo once.

### Week 2 pass/fail
- [ ] I built a working GPT from a blank file, **twice**
- [ ] I can write the attention formula and every tensor shape from memory
- [ ] I derived why `√d_k`
- [ ] I implemented a KV cache and measured the speedup
- [ ] I implemented BPE and my round-trip test passes on Devanagari and emoji
- [ ] I can explain bi-encoder vs cross-encoder and why the two-stage pipeline exists
- [ ] I know the four training stages and what each one buys
- [ ] 7 more outreach messages sent

**6/8 or better = proceed to Week 3.** Below that, repeat Days 10–11. Week 3 assumes you understand embeddings; the whole thing collapses without Day 13.
