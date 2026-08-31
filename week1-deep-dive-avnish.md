# WEEK 1 — Deep Dive
### Classical ML + backprop, taught properly, with a revision system that actually sticks

**Dates:** Day 1 = 28 Aug · Day 7 = 3 Sep
**Daily:** 120 min learn · 120 min build · 30 min recall · 30 min outreach
**End-of-week test:** 12 questions cold, on camera, 3 min each.

---

# PART A — THE REVISION SYSTEM (read this first, it is the actual answer to your question)

You asked "how can I feed these topics in my mind". The honest answer: **not by better explanations.** You already understand more than you think. Things don't stick because you are *consuming* instead of *retrieving*. Reading a good explanation feels like learning; it is recognition, not recall. In the interview room nobody shows you the notes.

## A1. The four reps (every topic gets all four)

| Rep | What you do | Time | Why it works |
|---|---|---|---|
| **R1 — Blank page** | Close everything. Write the concept from memory on paper. Then open notes and mark in red what you missed. | 10 min | Forces retrieval; the red marks are your real syllabus |
| **R2 — Derive** | Write the formula/derivation by hand, no reference | 10 min | Maths you can't derive is maths you'll blank on |
| **R3 — Blank file** | Implement it from an empty `.py`, no copy-paste, no tutorial open | 45–90 min | Code memory is motor memory. Typing along ≠ knowing |
| **R4 — Explain out loud** | 3-minute spoken answer, recorded on phone, as if to an interviewer | 5 min | Exposes the gap between "I get it" and "I can say it" |

If you skip R3 and R4 this week will be worth 20% of what it should be.

## A2. Spaced repetition schedule

Each day's topic gets revisited on **+1, +3, +7, +21** days. Revisit = 5-minute blank-page test, not re-watching.

| Topic (learnt on) | +1 | +3 | +7 | +21 |
|---|---|---|---|---|
| D1 Bias-variance / regularization (28 Aug) | 29 Aug | 31 Aug | 4 Sep | 18 Sep |
| D2 Metrics & imbalance (29 Aug) | 30 Aug | 1 Sep | 5 Sep | 19 Sep |
| D3 Linear/logistic derivations (30 Aug) | 31 Aug | 2 Sep | 6 Sep | 20 Sep |
| D4 Trees / bagging / boosting (31 Aug) | 1 Sep | 3 Sep | 7 Sep | 21 Sep |
| D5 Backprop / micrograd (1 Sep) | 2 Sep | 4 Sep | 8 Sep | 22 Sep |
| D6 Optimizers / init / norm (2 Sep) | 3 Sep | 5 Sep | 9 Sep | 23 Sep |

Put these as recurring 10-min blocks in your calendar right now. Your 30-min recall slot each evening covers that day's new topic **plus** whatever is due from this table. That's it — never more than ~20 min.

## A3. Anki, done right (30 cards max for Week 1)

Most people make Anki useless by dumping paragraphs into cards. Rules:

- **Atomic.** One card = one fact or one step. Not "explain boosting."
- **Question form, not topic form.** ❌ "Bias-variance tradeoff" ✅ "Why does bagging reduce variance but not bias?"
- **Include the trap.** ✅ "Why can ROC-AUC look excellent on a 1:1000 imbalanced set while the model is useless?"
- **Formula cards go both ways.** Card A: "MCC formula?" Card B: "This formula → what metric, and what does it correlate?"
- **Your-project cards.** ✅ "Why MCC and not F1 in OmitBench?" These are the ones that get you hired.
- Cap it at **30 cards for the whole week.** More cards = you stop doing them by day 10.

## A4. The one-page cheat sheet you write yourself

Sunday (Day 7): compress the entire week onto **one side of one A4 page**, handwritten. The compression is the learning. Photograph it. Revise from that page only, forever. Don't download someone else's cheat sheet — the value is entirely in you making it.

## A5. Interleaving (counter-intuitive but real)

Don't block-practise. On Day 4, your recall slot should mix Day 1, 2, 3 and 4 questions in random order. Blocked practice feels smoother and produces worse retention. Shuffle deliberately.

---

# PART B — DAY BY DAY

> **Note on CampusX:** it's good and it's in Hindi, which is a genuine advantage for you. But I'm not going to quote you fake episode numbers. Open the *100 Days of ML* playlist and search the playlist by topic name (e.g. "bias variance", "ridge lasso", "bagging", "gradient boosting"). Use it as the **first pass for intuition only**, at 1.5x, then close it and do R1–R4. It is not the source of truth; your blank page is.

---

## DAY 1 — Bias, variance, and why regularization works

### Why this day exists
It is the most-asked ML question on earth, and it is also the frame you'll use to answer half of everything else ("why did your model do X?"). You already live this: your OmitBench detector was low-variance/high-bias by design (deterministic rules), the LLM judge was the opposite. Say that in an interview and you've won the question.

### Core concepts

**1. The decomposition — derive it, don't memorise it.**

Assume `y = f(x) + ε`, with `E[ε]=0`, `Var(ε)=σ²`. For a fitted `f̂`:

```
E[(y − f̂)²] = (f − E[f̂])²  +  E[(f̂ − E[f̂])²]  +  σ²
                  Bias²              Variance         irreducible
```

Steps: substitute `y = f + ε`, expand, the cross-term vanishes because ε is independent of f̂ with mean 0, then add-and-subtract `E[f̂]` inside the remaining square. **Do this on paper today.** Three lines.

- **Bias** = error from your model class being too simple to represent f
- **Variance** = how much f̂ jumps around if you resample the training data
- **Irreducible** = noise. No model fixes it. If someone promises 99% on noisy labels, they're leaking.

**2. What actually moves each term**

| Move | Bias | Variance |
|---|---|---|
| More model capacity (depth, degree, params) | ↓ | ↑ |
| More training data | — | ↓ |
| Regularization (L1/L2, dropout, early stop) | ↑ | ↓ |
| Bagging / averaging | — | ↓ |
| Boosting | ↓ | ↑ |

**3. L1 vs L2, geometrically**

Constrained view: minimise loss subject to `‖w‖₁ ≤ t` (diamond) or `‖w‖₂ ≤ t` (circle). The loss contours are ellipses expanding outward; the solution is where they first touch the constraint region. A **diamond has corners on the axes**, so the first touch is often exactly at a corner → some `w_i = 0` → sparsity. A **circle has no corners** → coefficients shrink smoothly, never exactly zero.

Bonus fact interviewers like: ridge has a closed form `w = (XᵀX + λI)⁻¹Xᵀy`, and the `λI` term also makes `XᵀX` invertible when features are collinear or `p > n`. Lasso has no closed form (non-differentiable at 0) — hence coordinate descent / LARS.

**4. The modern nuance (say this and you sound current)**

The classic U-shaped test-error curve is not the whole story. In heavily overparameterized models you get **double descent**: test error rises, then falls again past the interpolation threshold. Which is why "more parameters = overfitting" is a 2015 statement, not a 2026 one. Know the phenomenon exists and that implicit regularization from SGD is part of the explanation.

### Resources (timeboxed — do not exceed)
- **60 min** — CampusX: bias-variance + ridge/lasso videos (search playlist by name), 1.5x
- **20 min** — StatQuest: "Bias and Variance", "Ridge Regression", "Lasso Regression" — best intuition-per-minute on YouTube
- **30 min** — ISLR (free PDF, `statlearning.com`) §2.2 and §6.2. Read, don't skim.
- Optional depth: ESL §7.1–7.4

### Practice (R3 — blank file)
```
practice/d1_bias_variance.py
```
1. Generate `y = sin(2πx) + N(0, 0.2)`, n=25 points
2. Fit polynomials of degree 1…15
3. Plot train MSE vs val MSE against degree — reproduce the U-curve yourself
4. Now bootstrap: refit degree-15 on 50 resamples, plot all 50 curves on one chart. **That spaghetti is variance. Look at it.**
5. Add ridge with λ ∈ {0, 0.01, 0.1, 1, 10}, watch the spaghetti tighten
6. Add lasso, print the coefficient vector, count the exact zeros

Success criteria: you can point at a chart and say which term dominates.

### Practice resources
- `sklearn.datasets.make_regression`, `load_diabetes`
- ISLR labs (Python version exists: `ISLP`)

### Self-test (R1/R4 — answer cold, then check)
1. Derive the decomposition in three lines
2. You have high bias. Name three fixes. Now high variance — name three.
3. Why does adding data help variance but not bias?
4. Why does L1 produce exact zeros and L2 doesn't?
5. Your train error is 2%, val 3%, prod 18%. It's not overfitting. What is it? *(distribution shift / leakage / label drift)*
6. Explain overfitting to a PM in 4 sentences with no jargon.

---

## DAY 2 — Metrics, imbalance, and defending a negative result

### Why this day exists
This is **your** day. You have a headline result of `paired MCC −0.202, CI excluding zero`. Right now that's a sentence on a repo. After today it should be a 3-minute answer you can give under pressure, including the statistics. This single answer is worth more to you than any other topic in Week 1.

### Core concepts

**1. The confusion matrix is the parent of every metric.** Derive each one from it, never memorise:
- Precision = TP/(TP+FP) — "of what I flagged, how much was real"
- Recall = TP/(TP+FN) — "of what was real, how much did I catch"
- F1 = harmonic mean of the two. **Note what's missing: TN.** F1 never looks at true negatives.
- Specificity = TN/(TN+FP), FPR = 1 − specificity

**2. MCC — and why you chose it**
```
MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
It is literally the Pearson correlation coefficient between the binary prediction vector and the binary label vector. Range −1 to +1, 0 = random. **It uses all four cells.** In your corpus most patches are *not* omissions, so true negatives carry real information — F1 would have thrown that away. That's your one-line justification. A negative MCC means the classifier is anti-correlated with truth: worse than a coin.

**3. Why ROC-AUC lies under imbalance**
FPR = FP/(FP+TN). When TN is enormous (1:1000), you can add hundreds of false positives and FPR barely moves → the ROC curve still looks beautiful. Precision, by contrast, is FP-sensitive and collapses. So:
- ROC-AUC baseline is always 0.5, regardless of imbalance
- **PR-AUC baseline is the prevalence** (0.001 for 1:1000) — so PR-AUC of 0.3 there is excellent, not bad
- Rule: rare positive class + you care about the positives → report PR-AUC

**4. The statistics you're already using — now understand them**
- **Paired** comparison: both systems scored on the *same* instances, so instance difficulty cancels out. Unpaired comparison of the same numbers would have much wider error bars. This is why paired designs are stronger.
- **McNemar's test**: the standard test for comparing two classifiers on the same data. Uses only the discordant pairs (b, c) — cases where one was right and the other wrong. Statistic `(|b−c|−1)²/(b+c)`.
- **Bootstrap CI**: resample instances with replacement 10,000×, recompute ΔMCC each time, take the 2.5th and 97.5th percentiles. "CI excluding zero" = in fewer than 5% of resamples did the sign flip.
- **Pre-registration** (you did this with the precision ≥ 0.80 bar): declaring the threshold before seeing results is what makes a negative result credible instead of a post-hoc excuse. Say this out loud in interviews.

**5. Threshold selection is a product decision, not a metric decision.** For a safety-critical gate (AegisOps), high recall matters more than precision — the cost of a missed hazard ≫ cost of a false alarm reviewed by a human. Be able to state the cost asymmetry in words, then pick the threshold from it.

### Resources
- **30 min** — StatQuest: ROC/AUC, Confusion Matrix, Precision/Recall
- **30 min** — Read the MCC-vs-F1 argument (Chicco & Jurman, "The advantages of the Matthews correlation coefficient…", BMC Genomics 2020 — free, short, and directly defends your choice)
- **30 min** — Chip Huyen, *Machine Learning Interviews* book (free online) — metrics + evaluation sections
- **30 min** — sklearn docs: `metrics` module. Read the actual docs, they're excellent.

### Practice (R3)
```
practice/d2_metrics.py
```
1. Implement from scratch in numpy: confusion matrix, precision, recall, F1, MCC. Assert equality with sklearn to 1e-9.
2. Build a synthetic 1:1000 imbalanced set. Train anything. Print accuracy (it'll be ~99.9% and meaningless), ROC-AUC, PR-AUC, MCC. **Write down the gap.**
3. Implement a bootstrap CI for ΔMCC between two dummy classifiers, 10k resamples.
4. Implement McNemar's test by hand on a 2×2 discordance table.
5. Sweep the threshold 0→1, plot precision, recall, F1, MCC on one chart. Find where each peaks. They peak at different thresholds — understand why.

### Practice resources
- Kaggle **Credit Card Fraud Detection** dataset (492 frauds / 284,807 — real 1:578 imbalance)
- `sklearn.datasets.make_classification(weights=[0.999, 0.001])`

### Self-test
1. Write the MCC formula from memory. What is it the correlation of?
2. Why does F1 ignoring TN matter in *your* corpus specifically?
3. 1:1000 imbalance, ROC-AUC 0.94 — is the model good? What do you ask for next?
4. What does "CI excluding zero" mean in plain English to a non-statistician?
5. **The big one:** "Your detector lost to an LLM judge. Why should we hire someone whose method failed?" — 3 minutes, out loud, recorded. Frame: pre-registered bar → method missed it → reported honestly → the failure localised the real bottleneck (requirement extraction, MCC −0.935) → that's a more valuable finding than a win would have been.

---

## DAY 3 — Linear & logistic regression from first principles

### Why this day exists
Every "explain X from scratch" interview starts here, and the log-loss gradient is the same shape you'll see again in backprop on Day 5. Two hours here saves you on Day 5.

### Core concepts

**1. Gradients — derive both**
- MSE: `L = (1/n)‖Xw − y‖²` → `∇L = (2/n) Xᵀ(Xw − y)`
- Log-loss: `L = −(1/n) Σ [y log p + (1−y) log(1−p)]`, `p = σ(Xw)` → `∇L = (1/n) Xᵀ(σ(Xw) − y)`

Notice they're the same shape: `Xᵀ(prediction − target)`. **That is not a coincidence** — both are generalized linear models with their canonical link, and the canonical link is exactly what makes the ugly derivative terms cancel. Saying this sentence marks you out.

**2. Why MSE is wrong for classification (two reasons — know both)**
- *Gradient saturation:* sigmoid + MSE gives a gradient containing `σ'(z) = σ(1−σ)`, which → 0 when the model is confidently **wrong**. The more wrong it is, the slower it learns. With log-loss, that `σ'` term cancels exactly, so the gradient is proportional to the error.
- *Convexity:* log-loss is convex in `w`; sigmoid + MSE is not, so you get local minima.

**3. Normal equation vs gradient descent.** `w = (XᵀX)⁻¹Xᵀy` is O(p³) — fine at p=100, unusable at p=10⁶, and breaks on collinearity. GD is O(np) per step. Know when each applies.

**4. Multiclass:** softmax generalises sigmoid; cross-entropy generalises log-loss. `softmax(z)_i = e^{z_i}/Σe^{z_j}`. Numerical stability trick: subtract `max(z)` before exponentiating — you will hit this bug for real.

**5. Assumptions of linear regression** (linearity, independence, homoscedasticity, normal residuals, no perfect multicollinearity) — asked as a filter question in service-company interviews (Quantiphi, Fractal, Tiger). Know all five and which ones actually matter for prediction vs inference.

### Resources
- **45 min** — CampusX: linear regression from scratch + logistic regression (playlist search)
- **30 min** — StatQuest: "Logistic Regression" series, "Maximum Likelihood"
- **30 min** — ISLR §3.1–3.3, §4.3
- **15 min** — Karpathy's "Yes you should understand backprop" blog post (short; sets up Day 5)

### Practice (R3)
```
practice/d3_from_scratch.py
```
1. Linear regression via GD in pure numpy. Compare `w` against `sklearn.LinearRegression` — match to 3 decimals.
2. Same via the normal equation. Confirm identical.
3. Logistic regression via GD in pure numpy on the breast-cancer dataset. Match sklearn's accuracy.
4. Implement softmax + cross-entropy **with** the max-subtraction trick. Then remove the trick and feed it `z = [1000, 1001]` — watch it produce `nan`. Now you'll never forget it.
5. Deliberately train sigmoid+MSE vs sigmoid+log-loss on the same data. Plot both loss curves. **See the saturation with your own eyes.**

### Practice resources
- `sklearn.datasets.load_breast_cancer`, `load_diabetes`
- Kaggle Titanic (for logistic + feature handling)

### Self-test
1. Derive the log-loss gradient on paper. No reference.
2. Why doesn't the sigmoid derivative appear in the final log-loss gradient?
3. When would you use the normal equation over GD?
4. Why subtract max before softmax?
5. Your logistic regression's coefficients are enormous and unstable. Two likely causes? *(multicollinearity; perfectly separable data → weights diverge → fix with regularization)*

---

## DAY 4 — Trees, bagging, boosting

### Why this day exists
Tabular ML is still where most Indian AI-services interviews live, and your own Startup Success Predictor used Random Forest. If you can't explain *why* RF works you have a hole exactly where they'll poke.

### Core concepts

**1. Splitting.** Gini `= 1 − Σpᵢ²`, Entropy `= −Σ pᵢ log₂ pᵢ`. Information gain = parent impurity − weighted average of child impurities. Gini is cheaper (no log) and picks almost identical splits — that's the whole practical difference. **Compute one split by hand on 8 rows today.** It takes 10 minutes and it permanently kills the fuzziness.

**2. Why Random Forest works — the formula that wins the question.**
The variance of the average of B identically-distributed variables with pairwise correlation ρ:
```
Var = ρσ² + (1−ρ)σ²/B
```
As B → ∞ the second term vanishes, but **`ρσ²` is a floor**. So bagging alone hits a wall. RF's extra move — sampling a random subset of features at each split (`max_features`) — is what *reduces ρ* and lowers the floor. That's the answer: bootstrap gives you averaging, feature subsampling gives you decorrelation, and decorrelation is where the real gain is.

**3. Boosting — write the loop from memory.**
```
F₀ = argmin_γ Σ L(yᵢ, γ)
for m = 1..M:
    rᵢ = −[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]   # pseudo-residuals
    fit tree hₘ to rᵢ
    Fₘ = Fₘ₋₁ + ν · hₘ              # ν = learning rate
```
Bagging = parallel, independent, reduces variance. Boosting = sequential, each learner fixes the last one's errors, reduces **bias**. Hence boosting overfits and bagging generally doesn't.

**4. XGBoost's actual innovations** (don't say "it's just faster"): second-order Taylor expansion of the loss (uses gradient *and* Hessian), regularization inside the objective (`γT + ½λ‖w‖²`), sparsity-aware split finding with a default direction for missing values, and the histogram/approximate split algorithm.

**5. Feature importance is a trap.** Default `feature_importances_` (impurity-based) is biased toward high-cardinality and continuous features. Use **permutation importance** or SHAP. Interviewers love catching people on this.

### Resources
- **45 min** — CampusX: decision trees, bagging, random forest, gradient boosting (playlist search)
- **45 min** — StatQuest: Decision Trees, Random Forests Part 1&2, **Gradient Boost Parts 1–4** (the Gradient Boost series is the single best explanation available anywhere — do all four)
- **30 min** — ISLR ch. 8
- Optional: XGBoost paper §2 (Chen & Guestrin 2016) — 3 pages, very readable

### Practice (R3)
```
practice/d4_trees.py
```
1. By hand on paper: 8 rows, 2 features, compute Gini for both candidate splits, pick the winner. Then verify with sklearn's tree.
2. Implement a depth-2 decision tree from scratch in numpy (recursive best-split search).
3. Implement **bagging** yourself: 50 bootstrap samples, 50 trees, majority vote. Compare to a single tree. Then hand-implement feature subsampling and watch the gap widen — you just built a Random Forest.
4. XGBoost/LightGBM on a tabular set. Then compute permutation importance **and** default importance. **Compare the two rankings and note where they disagree** — that disagreement is your interview anecdote.
5. Plot boosting train vs val error against `n_estimators`. Find where val turns up. That's overfitting in boosting, seen live.

### Practice resources
- Kaggle: Titanic, House Prices, Credit Card Fraud
- UCI Adult Income (good for categorical handling)
- `shap` library — 20 minutes to learn, pays off in every project demo

### Self-test
1. Gini vs entropy — real difference?
2. Write the correlated-average variance formula. What does RF's feature subsampling change in it?
3. Bagging vs boosting: which reduces bias, which variance, and why?
4. Write the gradient-boosting loop from memory.
5. Why is default feature importance misleading?
6. RF with 500 trees vs 5000 — when does it stop helping, and does it ever overfit?

---

## DAY 5 — Backprop, by hand and from a blank file

### Why this day exists
This is the load-bearing day of the whole month. Everything in Weeks 2–4 sits on it. If you only truly do one day properly, do this one.

### Core concepts

**1. Backprop is the chain rule plus bookkeeping.** Nothing more. Forward pass builds a DAG of operations; backward pass walks that DAG in reverse topological order, and each node multiplies the incoming gradient by its local derivative.

**2. Local gradients you must know cold:**
- `+` : passes the gradient through unchanged to both inputs (a distributor)
- `*` : each input gets `grad × the other input` (a swapper)
- `max`: routes the whole gradient to the winner, zero to the rest (a router)
- `σ` : `σ(1−σ)`
- `tanh`: `1 − tanh²`
- `ReLU`: 1 if x>0 else 0

**3. The bug you will write: `+=` not `=`.** If a node feeds two downstream consumers, its gradient is the **sum** of both paths. Assigning instead of accumulating silently halves your gradients and the model still "kind of trains". This exact bug class is what your IIA project detects in generated code — so you have a personal reason to internalise it.

**4. Topological sort.** You can't call backward on a node until every consumer of it has already been processed. That ordering *is* the algorithm.

**5. Karpathy's warning:** backprop is a leaky abstraction. Vanishing gradients, dead ReLUs, saturated tanh — these are all backprop failures that a framework hides from you until your loss silently plateaus.

### Resources
- **150 min** — Karpathy, *"The spelled-out intro to neural networks and backpropagation: building micrograd."* Watch at 1x. Pause and predict before he reveals each step.
- **15 min** — Karpathy's blog post "Yes you should understand backprop" (if not read on Day 3)
- Reference only: `karpathy/micrograd` repo (~150 lines). Read it **after** you've written yours.

### Practice (R2 + R3 — the most important reps of the week)
1. **By hand on paper:** a 3→2→1 network with tanh. Forward pass with actual numbers, then backward pass, computing every `∂L/∂w` manually. Verify against PyTorch autograd. It will take 45 minutes and it will be the most valuable 45 minutes of your week.
2. **Blank file:** rebuild the `Value` class from scratch — `__add__`, `__mul__`, `__pow__`, `tanh`, `backward()`, topological sort, gradient accumulation. **Video closed.**
3. Train a tiny MLP on it (4 points, binary classification) until loss < 0.01.
4. Implement gradient checking: numerically estimate `(L(w+ε) − L(w−ε))/2ε` and compare against your analytic gradient. Any mismatch > 1e-6 means a bug. Find it.
5. **Tomorrow, do step 2 again from a blank file.** The second rep is where it actually lands.

### Self-test
1. Backprop a 2-layer net by hand, cold.
2. Why `+=` and not `=` for gradients?
3. What's the gradient through a `max` node?
4. Why does topological order matter?
5. Loss isn't decreasing. Give five diagnostic steps in order.
6. What does a gradient check catch that unit tests don't?

---

## DAY 6 — Optimizers, initialization, normalization

### Why this day exists
"Why AdamW and not Adam" and "why LayerNorm in transformers" are standard LLM-role questions, and they gate your Week 2.

### Core concepts

**1. The optimizer family, each solving one specific problem:**

| Optimizer | Problem it solved |
|---|---|
| SGD | baseline; noisy, slow in ravines |
| + Momentum | `v = βv + g; w −= ηv` — EMA of gradients damps oscillation across a ravine, accelerates along it |
| RMSProp | `s = βs + (1−β)g²; w −= ηg/√(s+ε)` — per-parameter step size; rescues rare features |
| Adam | momentum + RMSProp + bias correction `m̂ = m/(1−β₁ᵗ)` (needed because m starts at 0 and is biased low early) |
| AdamW | fixes Adam's broken weight decay |

**2. Why AdamW exists — the crisp answer.** In Adam, an L2 penalty added to the gradient then gets divided by `√v`. So parameters with a large gradient history get *less* effective regularization — the decay strength becomes accidentally coupled to the gradient magnitude. AdamW **decouples** it: apply `w −= ηλw` directly to the weights, outside the adaptive scaling. That's the entire paper in two sentences, and it's why every modern LLM trains with AdamW.

**3. Adam vs SGD generalization.** Adam converges faster but sometimes generalises worse — SGD's noise acts as implicit regularization and tends to find flatter minima. Common recipe: Adam/AdamW for transformers, SGD+momentum for CNNs on vision.

**4. Initialization.** Variance must be preserved layer to layer, or activations explode/vanish through depth.
- **Xavier/Glorot:** `Var(w) = 2/(fan_in + fan_out)` — for tanh/sigmoid
- **He:** `Var(w) = 2/fan_in` — for ReLU. The extra factor of 2 exists because ReLU zeroes roughly half the units, halving the variance.
- Zero init = every neuron computes the same thing forever (symmetry never breaks).

**5. Normalization.**
- **BatchNorm** normalizes each feature across the batch. Depends on batch statistics → breaks with small batches, needs different train/inference behaviour (running averages), and is awkward for variable-length sequences.
- **LayerNorm** normalizes across features within a single sample. Batch-independent, works with any sequence length → this is why transformers use it.
- **RMSNorm** (modern LLMs): drops the mean-subtraction, keeps only the scaling. Cheaper, works about as well.
- **Pre-norm vs post-norm:** the original Transformer was post-norm; modern models are pre-norm because it gives a clean residual path and trains stably without a warmup crutch.

**6. Vanishing/exploding gradients** and their fixes: ReLU over sigmoid, residual connections (gradient highway), gradient clipping, careful init, normalization.

### Resources
- **45 min** — Karpathy makemore part 3 ("Activations & Gradients, BatchNorm") — the practical view of all of this
- **30 min** — distill.pub "Why Momentum Really Works" (interactive; you'll finally *feel* momentum)
- **30 min** — Skim the AdamW paper (Loshchilov & Hutter), §2 and §3 only
- **20 min** — Sebastian Raschka's blog on LayerNorm vs BatchNorm

### Practice (R3)
```
practice/d6_optim.py
```
1. Implement SGD, Momentum, RMSProp, and Adam **from scratch** on a 2D loss surface (Rosenbrock or a stretched quadratic). Plot all four trajectories on one contour plot. The picture explains momentum better than any sentence.
2. Train the same small net with zero init, then random-normal(σ=1), then He init. Plot activation histograms per layer for each. **Watch them die or explode.**
3. Add LayerNorm to a small net, replot the activation histograms.
4. Implement gradient clipping, then deliberately blow up the learning rate and watch clipping save the run.

### Self-test
1. What does momentum fix that plain SGD doesn't?
2. Why does Adam need bias correction?
3. Explain AdamW vs Adam in two sentences.
4. Why does He init use `2/fan_in` instead of `1/fan_in`?
5. Why LayerNorm and not BatchNorm in transformers? Give two reasons.
6. Why can't you initialise all weights to zero?

---

## DAY 7 — Consolidation (no new input)

**Rule for today: zero new videos, zero new articles.** If you watch anything today you have misunderstood the week.

### Schedule
- **90 min** — Write `ML_interview_answers.md`: the 12 questions from the roadmap, cold, timed 3 min each, no notes. Then open your notes and mark the gaps in a different colour.
- **45 min** — Handwrite the **one-page A4 cheat sheet** (§A4). Photograph it.
- **45 min** — Rebuild micrograd from a blank file for the **second** time. Time yourself. Should be under 40 min now.
- **30 min** — Record yourself on video answering Q2 (MCC choice) and Q3 (the negative result). Watch it back. Note filler words, hedging, and any moment you apologise for the result. Re-record once.
- **30 min** — Build your 30 Anki cards from the red gaps you marked, not from the whole syllabus.
- **30 min** — Outreach, as every day.

### Week 1 pass/fail (be honest)
- [ ] I derived the bias-variance decomposition on paper without reference
- [ ] I wrote MCC from memory and explained why it beat F1 *for my corpus*
- [ ] I derived the log-loss gradient cold
- [ ] I wrote the gradient-boosting loop from memory
- [ ] I rebuilt micrograd from a blank file **twice**
- [ ] I explained AdamW vs Adam in two sentences
- [ ] I have a one-page handwritten cheat sheet
- [ ] I recorded 2 answers and don't cringe at the second take
- [ ] 7 outreach messages sent

**6/9 or better = the week worked.** Below that, do not start Week 2 — repeat the two weakest days. Moving forward on a cracked foundation is how you end up with impressive projects you can't defend, which is the exact problem we're fixing.

---

# PART C — Fixed resource list (do not add a 9th)

**Video:** Karpathy Zero-to-Hero · StatQuest · CampusX (Hindi, intuition pass only)
**Text:** ISLR (free PDF) · Chip Huyen's *ML Interviews* (free) · Sebastian Raschka's blog
**Docs:** scikit-learn user guide — genuinely one of the best ML textbooks in existence, and nobody reads it
**Code:** `karpathy/micrograd`
**Practice:** Kaggle (Titanic, Fraud, House Prices) · UCI Adult · sklearn built-ins
**Tool:** Anki

Anything you add beyond this is procrastination wearing a lab coat.
