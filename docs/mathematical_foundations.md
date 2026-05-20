# Mathematical Foundations

This document collects the mathematical content distributed across the
notebooks and module docstrings into a single reference.

---

## 1. Vector similarity geometry

For unit-norm vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$,

$$
\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v},
\qquad
\|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2\mathbf{u}^\top \mathbf{v}.
$$

Therefore cosine similarity, inner product, and squared Euclidean
distance are *monotonically equivalent* on the unit sphere, and the
ranking induced by any of the three is identical. Our encoder
normalizes outputs to the unit sphere precisely to eliminate
metric-mismatch bugs at backend boundaries.

---

## 2. HNSW level assignment

A point $x$ is inserted at maximum layer

$$
\ell(x) = \left\lfloor -\ln(u) \cdot m_L \right\rfloor,
\qquad u \sim U(0,1),
\qquad m_L = \frac{1}{\ln M}.
$$

The induced layer-occupancy distribution is geometric with parameter
$1 - 1/M$, giving expected layer size

$$
\mathbb{E}\bigl[\,|\text{layer } \ell|\,\bigr] = \frac{N}{M^\ell}.
$$

Greedy descent from the top layer is therefore expected to traverse
$O(\log_M N)$ points before reaching the bottom layer, yielding
$O(\log N)$ search cost for uniform data.

---

## 3. Bi-Encoder vs. Cross-Encoder

**Bi-Encoder.** Query and document are embedded independently:

$$
s_{\text{bi}}(q, d) = \phi(q)^\top \phi(d), \qquad \phi:\Sigma^\* \to \mathbb{R}^d.
$$

Document embeddings $\phi(d)$ can be pre-computed once and cached,
making retrieval over $N$ documents $O(Nd)$ per query.

**Cross-Encoder.** The pair is concatenated and processed jointly:

$$
s_{\text{cross}}(q, d) = f_\theta\bigl([\text{CLS}]\,q\,[\text{SEP}]\,d\bigr) \in \mathbb{R}.
$$

Joint attention captures fine-grained token-level interactions but
forbids caching: re-ranking $N$ candidates costs $N$ Transformer
forward passes.

**Training objectives** implemented in `src/reranking/cross_encoder.py`:

- *Pointwise BCE:*

  $$
  \mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \bigl(y_i\log\sigma(s_i) + (1-y_i)\log(1-\sigma(s_i))\bigr).
  $$

- *Pairwise margin* (hinge):

  $$
  \mathcal{L}_{\text{margin}} = \frac{1}{N}\sum_i \max\bigl(0, m - (s_i^+ - s_i^-)\bigr).
  $$

---

## 4. HyDE

For query $q$, let $\mathrm{LLM}(q)$ denote a hallucinated answer
paragraph. HyDE replaces the retrieval score

$$
s(q, d) = \phi(q)^\top \phi(d)
\qquad\text{with}\qquad
s_{\text{HyDE}}(q, d) = \phi\bigl(\mathrm{LLM}(q)\bigr)^\top \phi(d).
$$

With $K$ hallucinations sampled at temperature $T > 0$, we use the
Monte-Carlo estimate

$$
\hat{\phi}_q = \frac{1}{K} \sum_{k=1}^K \phi\bigl(\mathrm{LLM}_k(q)\bigr).
$$

Averaging reduces the variance contributed by individual
hallucinations. We additionally include $\phi(q)$ itself in the
average so that an off-topic hallucination cannot completely destroy
recall.

---

## 5. nDCG@k

For a ranking with relevance grades $r_i \in \{0,1,2,\dots\}$,

$$
\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)},
\qquad
\text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k},
$$

where IDCG is computed from the ideally-sorted ranking. This is the
metric reported by `scripts/benchmark_retrieval.py`.

---

## 6. Termination of the multi-agent loop

Let the state at iteration $t$ be $s_t = (c_t, e_t)$, where $c_t$ is
the code and $e_t$ the most recent error. The orchestrator
maintains a set $\mathcal{H}$ of SHA-256 hashes of seen states. If
$\text{hash}(s_t) \in \mathcal{H}$, the loop halts. Because
$\mathcal{H}$ is finite and grows monotonically — and because the
state space, while large, is bounded in expectation by the LLM's
output distribution — the loop terminates almost surely within
$|\mathcal{H}|$ iterations. The `max_iterations` budget provides a
deterministic worst-case bound.
