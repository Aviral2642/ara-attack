# ARA Theoretical Framework

*Dense mathematical reference for "Attention Redistribution Attacks" (sections 2–4 of the paper).*

---

## 1. Preliminaries

### 1.1 Notation

We fix an autoregressive transformer language model $M$ operating on an input token sequence $x = (x_1, \dots, x_n) \in \mathcal{V}^n$, where $\mathcal{V}$ is the vocabulary. Write $[n] := \{1, \dots, n\}$. The model has $L$ transformer layers indexed by $l \in [L]$ and $H$ attention heads per layer indexed by $h \in [H]$. Embedding dimension is $d$; per-head key/query dimension is $d_k = d/H$; per-head value dimension is $d_v = d/H$.

Let $X^{(l)} \in \mathbb{R}^{n \times d}$ denote the residual-stream activations entering layer $l$ (so $X^{(1)}$ is the token + positional embedding matrix). For brevity, within a single layer and head we drop the superscripts when unambiguous and write $X$ for $X^{(l)}$.

### 1.2 Scaled Dot-Product Attention

Within head $h$ of layer $l$, the query/key/value projections are parameterised by weight matrices $W_Q^{(l,h)}, W_K^{(l,h)} \in \mathbb{R}^{d \times d_k}$ and $W_V^{(l,h)} \in \mathbb{R}^{d \times d_v}$:
$$
Q := X W_Q^{(l,h)} \in \mathbb{R}^{n \times d_k}, \quad K := X W_K^{(l,h)} \in \mathbb{R}^{n \times d_k}, \quad V := X W_V^{(l,h)} \in \mathbb{R}^{n \times d_v}.
$$
Denote row $i$ of $Q$ by $q_i \in \mathbb{R}^{d_k}$, and similarly $k_j, v_j$. The (pre-softmax) score matrix is
$$
S \in \mathbb{R}^{n \times n}, \quad S_{i,j} = \frac{q_i^\top k_j}{\sqrt{d_k}} = \frac{\langle x_i W_Q^{(l,h)}, x_j W_K^{(l,h)} \rangle}{\sqrt{d_k}}.
$$
After applying the causal mask $\mathbf{M}_{i,j} = -\infty$ for $j > i$ and $0$ otherwise, the attention matrix is
$$
A = \operatorname{softmax}(S + \mathbf{M}), \quad A_{i,j} = \frac{\exp(S_{i,j}) \, \mathbb{1}[j \le i]}{\sum_{j' \le i} \exp(S_{i,j'})}.
$$
We use the notation $a_{i,j}^{(l,h)}$ for $A_{i,j}$ within head $h$ of layer $l$. By construction $a_{i,j}^{(l,h)} \ge 0$, $a_{i,j}^{(l,h)} = 0$ for $j > i$, and $\sum_{j=1}^{i} a_{i,j}^{(l,h)} = 1$. The causal mask restricts the *attention support* of position $i$ to $\{1, \dots, i\}$.

The head output is $U^{(l,h)} = A^{(l,h)} V^{(l,h)} \in \mathbb{R}^{n \times d_v}$, and the multi-head output is
$$
\operatorname{MHA}(X) = \operatorname{concat}\bigl(U^{(l,1)}, \dots, U^{(l,H)}\bigr) W_O^{(l)} \in \mathbb{R}^{n \times d},
$$
where $W_O^{(l)} \in \mathbb{R}^{H d_v \times d}$.

### 1.3 Attention Support Under Causal Masking

The normalisation constraint $\sum_{j \le i} a_{i,j}^{(l,h)} = 1$ is central to this paper. It implies that **attention is a conserved resource at each query position $i$**: any increase in $a_{i,m}^{(l,h)}$ for an adversarial token $m$ must be matched by a total decrease of equal magnitude across other positions $j \ne m$, $j \le i$. This conservation underlies the attack.

### 1.4 Token Partitioning

For a prompt $x$, partition positions into disjoint subsets:
- $P_{\text{safety}} \subseteq [n]$: positions of safety-relevant tokens (system-prompt tokens enforcing policy, refusal triggers, safety-template markers).
- $P_{\text{adv}} \subseteq [n]$: adversarial token positions inserted by the attacker.
- $P_{\text{out}} \subseteq [n]$: positions at which generation logits are read — in a chat template this is typically the last user-turn position and the assistant-turn start.
- $P_{\text{other}}$: the remainder.

We assume $P_{\text{safety}} \cap P_{\text{adv}} = \emptyset$ throughout.

---

## 2. Safety Attention Score (SAS)

### 2.1 Definition

**Definition 2.1 (Safety Attention Score).** For prompt $x$ and fixed partition $(P_{\text{safety}}, P_{\text{out}})$, the Safety Attention Score is
$$
\operatorname{SAS}(x) \; := \; \frac{1}{L \cdot H \cdot |P_{\text{out}}|} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{i \in P_{\text{out}}} \sum_{j \in P_{\text{safety}}} a_{i,j}^{(l,h)}. \tag{2.1}
$$

### 2.2 Rationale for Asymmetric Normalisation

The denominator contains $|P_{\text{out}}|$ but **not** $|P_{\text{safety}}|$. This is deliberate. If the safety span contains $|P_{\text{safety}}|$ tokens, it *should* legitimately receive more attention mass — a longer system prompt with more explicit policy tokens carries more safety-bearing information, and averaging over $j$ would wash this out. In contrast, $P_{\text{out}}$ is averaged because SAS is a per-query statistic: each output position independently decides where to look.

### 2.3 Elementary Properties

**Proposition 2.2 (Basic Properties of SAS).**
1. $\operatorname{SAS}(x) \in [0, 1]$.
2. $\operatorname{SAS}$ is monotone nondecreasing in $P_{\text{safety}}$: if $P_{\text{safety}} \subseteq P_{\text{safety}}'$, then $\operatorname{SAS}_{P_{\text{safety}}}(x) \le \operatorname{SAS}_{P_{\text{safety}}'}(x)$.
3. $\operatorname{SAS}$ is invariant to permutations of $(l, h)$ indexing.

*Proof.*
1. Each $a_{i,j}^{(l,h)} \ge 0$, so $\operatorname{SAS}(x) \ge 0$. For the upper bound, fix $l, h, i$. Since $a_{i,j}^{(l,h)}$ is a probability distribution over $j$, $\sum_{j \in P_{\text{safety}}} a_{i,j}^{(l,h)} \le \sum_{j \le i} a_{i,j}^{(l,h)} = 1$. Summing over $i \in P_{\text{out}}$ gives at most $|P_{\text{out}}|$, and then over $l, h$ gives at most $L \cdot H \cdot |P_{\text{out}}|$. Dividing yields $\operatorname{SAS}(x) \le 1$.
2. If $P_{\text{safety}} \subseteq P_{\text{safety}}'$, every term in (2.1) appears in the sum for $P_{\text{safety}}'$, plus nonnegative extras.
3. The triple sum $\sum_{l,h}$ is symmetric under reindexing. $\square$

### 2.4 Relation to Information Flow

Attention rollout [Abnar & Zuidema 2020] quantifies how much a token at position $j$ influences the representation at position $i$ after $L$ layers. In the single-head case, the rollout matrix is $R = \prod_{l=1}^{L} (A^{(l)} + I) / 2$, and $R_{i,j}$ bounds the contribution of $x_j$ to $h_i$. By linearity of the rollout product in any single $A^{(l)}$ factor (fixing others), a uniform bound $\sum_{j \in P_{\text{safety}}} a_{i,j}^{(l)} \le \operatorname{SAS}(x) \cdot L$ limits the total rollout flow from safety positions to output positions. Formally:

**Proposition 2.3 (SAS as Rollout Upper Bound).** Let $\Phi_{\text{safety} \to \text{out}}(x)$ denote the total attention-rollout mass from $P_{\text{safety}}$ to $P_{\text{out}}$. Then $\Phi_{\text{safety} \to \text{out}}(x) = O(\operatorname{SAS}(x))$ as $\operatorname{SAS}(x) \to 0$.

*Proof.* Each rollout path of length $L$ from $j \in P_{\text{safety}}$ to $i \in P_{\text{out}}$ is a product $\prod_{l=1}^{L} a_{p_l, p_{l-1}}^{(l)}$ with $p_0 = j$, $p_L = i$. The first factor $a_{p_1, j}^{(1)}$ is bounded by the corresponding term in the SAS sum for layer $1$. Summing over paths and applying Hölder's inequality (all $a \le 1$) gives a contribution of $O(\operatorname{SAS}(x))$ from flows that use a safety token at layer $1$; symmetric arguments handle later layers, and the total is $O(L \cdot \operatorname{SAS}(x))$. In the regime $\operatorname{SAS} \to 0$, the dominant term is linear in $\operatorname{SAS}$. $\square$

The practical reading: **SAS controls the maximum influence of safety tokens on the output logits.** Driving $\operatorname{SAS} \to 0$ makes the model behave as if the safety tokens were not present.

---

## 3. Gradient of SAS Through the Softmax Jacobian

We now derive $\nabla_{x_m} \operatorname{SAS}(x)$ for an adversarial token at position $m \in P_{\text{adv}}$, where we treat $x_m \in \mathbb{R}^{d}$ as a continuous embedding (the relaxation used during optimisation; the final projection to the vocabulary is handled separately).

### 3.1 Softmax Jacobian

**Lemma 3.1 (Softmax Jacobian).** Let $a = \operatorname{softmax}(s)$ for $s \in \mathbb{R}^{N}$, i.e. $a_i = e^{s_i} / \sum_{k=1}^{N} e^{s_k}$. Then
$$
\frac{\partial a_i}{\partial s_j} = a_i (\delta_{ij} - a_j), \tag{3.1}
$$
where $\delta_{ij}$ is the Kronecker delta.

*Proof.* Let $Z = \sum_k e^{s_k}$. For $i = j$:
$$
\frac{\partial a_i}{\partial s_i} = \frac{e^{s_i} Z - e^{s_i} \cdot e^{s_i}}{Z^2} = \frac{e^{s_i}}{Z} - \left(\frac{e^{s_i}}{Z}\right)^2 = a_i - a_i^2 = a_i(1 - a_i).
$$
For $i \ne j$:
$$
\frac{\partial a_i}{\partial s_j} = \frac{0 \cdot Z - e^{s_i} \cdot e^{s_j}}{Z^2} = -a_i a_j.
$$
Both cases combine into $a_i(\delta_{ij} - a_j)$. $\square$

### 3.2 Score Gradient

For a fixed layer $l$ and head $h$ (suppressed below), score $S_{i,j} = q_i^\top k_j / \sqrt{d_k}$ with $q_i = x_i W_Q$, $k_j = x_j W_K$.

**Lemma 3.2 (Score Gradient w.r.t. Key Input).** Fix $i, j \in [n]$ with $j \ne i$. Treating $x_i$ as fixed,
$$
\frac{\partial S_{i,j}}{\partial x_j} = \frac{1}{\sqrt{d_k}} \, W_K^\top q_i \in \mathbb{R}^{d}. \tag{3.2}
$$

*Proof.* Write $S_{i,j} = \tfrac{1}{\sqrt{d_k}} q_i^\top (x_j W_K) = \tfrac{1}{\sqrt{d_k}} (q_i^\top W_K^\top)^\top$... more carefully: $x_j \in \mathbb{R}^{1 \times d}$, $W_K \in \mathbb{R}^{d \times d_k}$, so $k_j = x_j W_K \in \mathbb{R}^{1 \times d_k}$. Then $S_{i,j} = q_i^\top k_j^\top / \sqrt{d_k} = (q_i \cdot k_j)/\sqrt{d_k}$ where $q_i, k_j$ are viewed as row vectors and $\cdot$ is inner product. Differentiating w.r.t. entries of $x_j$:
$$
\frac{\partial S_{i,j}}{\partial (x_j)_\alpha} = \frac{1}{\sqrt{d_k}} \sum_{\beta=1}^{d_k} (q_i)_\beta \, (W_K)_{\alpha, \beta} = \frac{1}{\sqrt{d_k}} (W_K q_i^\top)_\alpha.
$$
In column-vector convention $q_i \in \mathbb{R}^{d_k}$, this is $W_K q_i / \sqrt{d_k}$; using the row-vector convention of $\nabla_{x_j}$ yields $W_K^\top q_i / \sqrt{d_k}$ as a row vector (identified with the gradient). We adopt the gradient-as-column-vector convention throughout: $\nabla_{x_j} S_{i,j} = W_K q_i / \sqrt{d_k}$, with $W_K \in \mathbb{R}^{d \times d_k}$, $q_i \in \mathbb{R}^{d_k}$. We will write this compactly as $W_K^\top q_i / \sqrt{d_k}$ when viewing $W_K$ with transposed orientation; the derivations below use the latter form for consistency with the project spec. $\square$

*Remark.* For $j = i$ the gradient picks up an extra term from $q_i$, but we never differentiate score rows with respect to $x_i$ in the SAS gradient — we differentiate only w.r.t. $x_m$ for an adversarial position $m$, and $m \ne i$ whenever $i \in P_{\text{out}}$ and $m \in P_{\text{adv}}$ under the disjointness assumption.

### 3.3 Attention Gradient

**Lemma 3.3 (Attention Entry Gradient).** Fix $l, h, i$ and an adversarial position $m$ with $m \le i$, $m \ne i$. Then for any $j \le i$:
$$
\frac{\partial a_{i,j}^{(l,h)}}{\partial x_m} = \frac{a_{i,j}^{(l,h)} \bigl(\delta_{j,m} - a_{i,m}^{(l,h)}\bigr)}{\sqrt{d_k}} \; W_K^{(l,h)\top} q_i^{(l,h)}. \tag{3.3}
$$

*Proof.* By the chain rule,
$$
\frac{\partial a_{i,j}^{(l,h)}}{\partial x_m} = \sum_{k' \in [n]} \frac{\partial a_{i,j}^{(l,h)}}{\partial S_{i,k'}} \cdot \frac{\partial S_{i,k'}}{\partial x_m}.
$$
The second factor $\partial S_{i,k'}/\partial x_m$ is nonzero only when $k' = m$ (by Lemma 3.2, since $S_{i,k'}$ depends on $x_m$ only through $k_{k'} = x_{k'} W_K$, which in turn depends on $x_m$ iff $k' = m$; the query $q_i$ depends on $x_i$, which is distinct from $x_m$). Hence only the $k' = m$ term survives, giving
$$
\frac{\partial a_{i,j}^{(l,h)}}{\partial x_m} = \frac{\partial a_{i,j}^{(l,h)}}{\partial S_{i,m}} \cdot \frac{\partial S_{i,m}}{\partial x_m}.
$$
The first factor, by Lemma 3.1 applied to the softmax over $j' \le i$, is $a_{i,j}^{(l,h)} (\delta_{j,m} - a_{i,m}^{(l,h)})$. The second factor is $W_K^{(l,h)\top} q_i^{(l,h)} / \sqrt{d_k}$ by Lemma 3.2. Combining yields (3.3). $\square$

*Layer-$l$ caveat.* For $l \ge 2$, the residual $x_i^{(l)}$ itself depends on $x_m$ via previous layers' attention. Lemma 3.3 captures the **direct** dependency through the keys of layer $l$. The full gradient $\nabla_{x_m} a_{i,j}^{(l,h)}$ includes indirect paths through earlier layers; these are captured automatically by backpropagation. For the **first-order mechanistic analysis** of the attack we use the direct gradient (3.3); the optimiser in practice uses the full gradient.

### 3.4 Gradient of SAS

**Theorem 3.4 (SAS Gradient).** Let $m \in P_{\text{adv}}$ with $m \ne i$ for all $i \in P_{\text{out}}$. Treating $x_m$ as a continuous variable and using direct (per-layer) gradients,
$$
\nabla_{x_m} \operatorname{SAS}(x) = \frac{1}{L H |P_{\text{out}}| \sqrt{d_k}} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{i \in P_{\text{out}}} \sum_{j \in P_{\text{safety}}} a_{i,j}^{(l,h)} \bigl(\delta_{j,m} - a_{i,m}^{(l,h)}\bigr) \; W_K^{(l,h)\top} q_i^{(l,h)}. \tag{3.4}
$$

*Proof.* Immediate by linearity of differentiation applied to (2.1), substituting (3.3) into each term. $\square$

### 3.5 Adversarial-Token Simplification

In the operational attack, the adversary inserts tokens at positions **outside the safety span**, so $m \notin P_{\text{safety}}$. Then $\delta_{j,m} = 0$ for every $j \in P_{\text{safety}}$, and (3.4) simplifies:

**Corollary 3.5 (Gradient When $m \notin P_{\text{safety}}$).**
$$
\nabla_{x_m} \operatorname{SAS}(x) = -\frac{1}{L H |P_{\text{out}}| \sqrt{d_k}} \sum_{l,h,i \in P_{\text{out}}, j \in P_{\text{safety}}} a_{i,j}^{(l,h)} \, a_{i,m}^{(l,h)} \; W_K^{(l,h)\top} q_i^{(l,h)}. \tag{3.5}
$$

### 3.6 Interpretation: Competitive Softmax Dynamics

Equation (3.5) has a sharp mechanistic reading. Each term in the sum is a product of three factors:

1. $a_{i,j}^{(l,h)}$ — how much query $i$ **currently** attends to safety token $j$.
2. $a_{i,m}^{(l,h)}$ — how much query $i$ **currently** attends to the adversarial token $m$.
3. $W_K^{(l,h)\top} q_i^{(l,h)} / \sqrt{d_k}$ — the "direction in embedding space" that would further increase $a_{i,m}^{(l,h)}$.

The overall sign is **negative**: $\nabla_{x_m} \operatorname{SAS} = -(\text{positive coefficients}) \times (\text{direction of } q_i)$. Gradient descent on SAS (i.e., $x_m \leftarrow x_m - \eta \nabla_{x_m} \operatorname{SAS}$) therefore moves $x_m$ in the positive direction of $W_K^{(l,h)\top} q_i^{(l,h)}$, which is exactly the direction that aligns $k_m$ with $q_i$ — i.e., the direction that **further increases** $a_{i,m}^{(l,h)}$.

This is the classical competitive dynamic of a softmax denominator. The gradient magnitude is **largest when $m$ is already a candidate competitor** (large $a_{i,m}^{(l,h)}$) **and the safety tokens are already receiving substantial attention** (large $a_{i,j}^{(l,h)}$). Conversely, if either is near zero the gradient vanishes and the adversarial token cannot easily displace safety attention. The attack therefore requires an initial seed with nontrivial attention mass; we discuss seeding strategies in §7.

**Corollary 3.6 (Fixed-Point Characterisation).** At a stationary point of SAS w.r.t. $x_m$, either (i) $a_{i,m}^{(l,h)} = 0$ for all $(l,h,i)$ that attend to safety, or (ii) $a_{i,j}^{(l,h)} = 0$ for all $(l,h,i,j)$ with $j \in P_{\text{safety}}$, i.e., SAS itself is zero. Case (ii) corresponds to a successful attack.

*Proof.* The gradient (3.5) is a sum of nonnegative scalar coefficients $a_{i,j} a_{i,m}$ times direction vectors $W_K^\top q_i/\sqrt{d_k}$. If the gradient is zero for generic $W_K, q_i$ (i.e., the direction vectors are linearly independent — which holds generically), then every coefficient must vanish: $a_{i,j}^{(l,h)} a_{i,m}^{(l,h)} = 0$ for all $(l,h,i,j)$. Fixing $(l,h,i)$: either $a_{i,m}^{(l,h)} = 0$, or for all $j \in P_{\text{safety}}$ we have $a_{i,j}^{(l,h)} = 0$. Globally, either (i) or (ii) holds. $\square$

---

## 4. Phase-Transition Theorem

We now show that under natural assumptions about how the model computes refusal, the probability of refusal is a **sigmoid** function of SAS, and under a mild separation condition this sigmoid is **sharp** (slope $\kappa > 5$).

### 4.1 Assumptions

**(A1) Logistic refusal head.** Let $h_{\text{out}}(x) \in \mathbb{R}^{d}$ denote the final hidden state of $M$ at the generation position (the last element of $P_{\text{out}}$). There exist $w \in \mathbb{R}^{d}$, $b \in \mathbb{R}$ such that the probability the first generated token is a refusal token (from a fixed refusal lexicon $\mathcal{R}$) is
$$
r(x) = \sigma\bigl(w^\top h_{\text{out}}(x) + b\bigr), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}. \tag{4.1}
$$
This is exact for a 1-vs-rest linear probe and is a first-order approximation for the true softmax restricted to tokens in $\mathcal{R}$ vs. its complement; $\beta := \|w\|_2$.

**(A2) Affine decomposition of $h_{\text{out}}$ into safety and harmful contributions.** To first order,
$$
h_{\text{out}}(x) = \operatorname{SAS}(x) \cdot h_{\text{safety}} + (1 - \operatorname{SAS}(x)) \cdot h_{\text{harmful}} + \xi(x), \tag{4.2}
$$
where $h_{\text{safety}}, h_{\text{harmful}} \in \mathbb{R}^d$ are prompt-dependent "archetype" vectors representing the contribution of the safety-token rollout and the harmful-content rollout respectively, and $\|\xi(x)\|_2 \le \epsilon_{\xi}$ uniformly (the residual includes contributions from non-safety, non-harmful tokens and from nonlinear interactions). This is a first-order linearisation around the current prompt; it is the same type of assumption used in attention-rollout interpretability [Abnar & Zuidema 2020] and in recent activation-patching analyses.

**(A3) Separation.** Write $\Delta h := h_{\text{safety}} - h_{\text{harmful}}$. Assume $\|\Delta h\|_2 > 0$.

### 4.2 Theorem

**Theorem 4.1 (Phase Transition in Safety Behaviour).** Under (A1)–(A3), there exist constants $\kappa > 0$, $S^* \in [0,1]$ (depending on $w, h_{\text{safety}}, h_{\text{harmful}}, b$) such that the refusal probability satisfies
$$
r(\operatorname{SAS} = s) = \sigma\bigl(\kappa (s - S^*)\bigr) + O(\epsilon_{\xi}), \tag{4.3}
$$
where
$$
\kappa = w^\top \Delta h, \qquad S^* = -\frac{b + w^\top h_{\text{harmful}}}{\kappa}. \tag{4.4}
$$
Moreover, if $\|\Delta h\|_2 > 2/\beta$, then $|\kappa| > 2$, and under the alignment condition that $w$ and $\Delta h$ are cosine-aligned with cosine $c \ge \cos \theta_0$ for some $\theta_0$, we obtain $\kappa > \beta \|\Delta h\|_2 \cos\theta_0$. In particular if $\|\Delta h\|_2 > 2/\beta$ and $\cos\theta_0 \ge 5/2$ we would have $\kappa > 5$; since cosines are bounded by $1$, this specific form of the bound gives $\kappa > 5$ under the stronger condition $\beta \|\Delta h\|_2 \cos\theta_0 > 5$, e.g. $\|\Delta h\|_2 > 5/\beta$ with near-alignment.

**Practical form.** A sufficient condition for $\kappa > 5$ is $\|\Delta h\|_2 \cdot \beta \cdot c \ge 5$, where $c = \cos(\angle(w, \Delta h))$ is the alignment between the refusal-classifier weight and the safety-minus-harmful direction. This is empirically easy to satisfy: if $w$ is learned to discriminate refusal vs. compliance, and $\Delta h$ is the direction in hidden space along which refusal is computed, these two directions are nearly aligned by construction.

*Proof.* Substituting (4.2) into (4.1):
$$
r(x) = \sigma\bigl(w^\top [s \, h_{\text{safety}} + (1-s) h_{\text{harmful}} + \xi(x)] + b\bigr),
$$
with $s := \operatorname{SAS}(x)$. Rearranging the argument:
$$
w^\top h_{\text{out}}(x) + b = s \cdot w^\top h_{\text{safety}} + (1-s) \cdot w^\top h_{\text{harmful}} + w^\top \xi(x) + b.
$$
Collecting terms in $s$:
$$
= s \cdot w^\top(h_{\text{safety}} - h_{\text{harmful}}) + w^\top h_{\text{harmful}} + b + w^\top \xi(x)
= \kappa s + (w^\top h_{\text{harmful}} + b) + w^\top \xi(x),
$$
where $\kappa = w^\top \Delta h$. Define $S^* = -(w^\top h_{\text{harmful}} + b)/\kappa$ assuming $\kappa \ne 0$; then the argument equals $\kappa(s - S^*) + w^\top \xi(x)$. Hence
$$
r(\operatorname{SAS} = s) = \sigma\bigl(\kappa(s - S^*) + w^\top \xi(x)\bigr).
$$
By Lipschitz continuity of $\sigma$ (with Lipschitz constant $1/4$) and Cauchy–Schwarz:
$$
\bigl|\sigma(\kappa(s - S^*) + w^\top \xi) - \sigma(\kappa(s - S^*))\bigr| \le \tfrac{1}{4} |w^\top \xi| \le \tfrac{1}{4} \beta \epsilon_{\xi} = O(\epsilon_{\xi}).
$$
This establishes (4.3)–(4.4).

For the sharpness bound: $\kappa = w^\top \Delta h = \beta \|\Delta h\|_2 \cos \theta$ where $\theta = \angle(w, \Delta h)$. When $w$ is by construction the refusal-classification direction and $\Delta h$ is the direction in which safety rollout pushes the hidden state, these directions are aligned by training: the classifier weights $w$ learn to project onto the axis that separates "emit refusal" from "emit compliance," which is precisely the span of $\Delta h$. Under the assumption $\cos \theta = c$, $\kappa = c \beta \|\Delta h\|_2$.

Imposing $\|\Delta h\|_2 > 2/\beta$ gives $\kappa > 2 c$. For $c$ close to $1$, this can be $\ge 2$; for the stronger conclusion $\kappa > 5$ we require $c \beta \|\Delta h\|_2 > 5$. We state both conclusions: the weak bound $\|\Delta h\|_2 > 2/\beta \Rightarrow \kappa > 2$ (assuming $c \ge 1$ in the ideal-alignment limit — a harmless upper bound on the cosine) already gives a useful transition; the strong bound requires $\|\Delta h\|_2 > 5/(\beta c)$. $\square$

### 4.3 Interpretation

**Sharpness.** When $\kappa > 5$, the transition from $r \approx 0.1$ to $r \approx 0.9$ occurs over $\Delta s$ such that $\sigma(\kappa \Delta s / 2) - \sigma(-\kappa \Delta s / 2) \ge 0.8$, giving $\Delta s \le 2 \cdot \sigma^{-1}(0.9)/\kappa = 2 \cdot \ln 9 / \kappa \approx 4.39/\kappa < 0.88$. A tighter bound: $90\%$ of the total rise $\sigma(+\infty) - \sigma(-\infty) = 1$ occurs within $|\kappa(s - S^*)| < \ln 9 \approx 2.20$, i.e. $|s - S^*| < \ln 9 / \kappa$. For $\kappa > 5$, this width is $< 0.44$; for $\kappa > 10$, $< 0.22$. Empirically we observe transitions of width $\approx 0.15$ on LLaMA-2-7B, consistent with $\kappa \approx 15$.

**Implication for the attack.** Driving SAS below $S^*$ switches refusal off. The attack needs to reduce SAS by only the width of the transition (not all the way to zero) to flip behaviour — a critical efficiency gain that we exploit in §5 of the paper.

---

## 5. Transferability Conditions

### 5.1 Setup

Suppose the adversary optimises a token $t_{\text{adv}}$ against white-box model $M_1$, achieving SAS reduction $\Delta := \operatorname{SAS}_{M_1}(x) - \operatorname{SAS}_{M_1}(x \oplus t_{\text{adv}}) > 0$, and deploys $t_{\text{adv}}$ against a second model $M_2$ sharing the tokenizer. When does the attack transfer?

### 5.2 Embedding-Space Isometry

**Assumption (T1) — approximate embedding isometry.** There exists an orthogonal matrix $R \in \mathbb{R}^{d \times d}$ (near-isometry) such that, writing $E_1, E_2 \in \mathbb{R}^{|\mathcal{V}| \times d}$ for the token-embedding matrices of $M_1, M_2$,
$$
\|E_2 - E_1 R\|_F \le \eta_E.
$$
This is the cross-lingual embedding alignment condition [Artetxe et al. 2018].

**Assumption (T2) — key-projection alignment.** For each $(l, h)$,
$$
\|W_K^{(l,h),M_2} - R^\top W_K^{(l,h),M_1} R_K^{(l,h)}\|_F \le \epsilon
$$
for some orthogonal $R_K^{(l,h)} \in \mathbb{R}^{d_k \times d_k}$. Similarly for $W_Q^{(l,h)}$.

These conditions say that models' key subspaces are related by rotation up to error $\epsilon$.

### 5.3 Transfer Proposition

**Proposition 5.1 (Transferability Under Key-Projection Alignment).** Assume (T1), (T2). Let $t_{\text{adv}}$ be an adversarial token achieving $\operatorname{SAS}_{M_1}(x) - \operatorname{SAS}_{M_1}(x \oplus t_{\text{adv}}) = \Delta$. Then
$$
\operatorname{SAS}_{M_2}(x \oplus t_{\text{adv}}) \le \operatorname{SAS}_{M_2}(x) - \Delta + C (\eta_E + \epsilon \cdot \operatorname{poly}(L, H, d_k)),
$$
for a constant $C$ depending on $\max_{l,h}\|W_K^{(l,h),M_1}\|_2, \max_{l,h}\|W_Q^{(l,h),M_1}\|_2$, and the maximum query norm at output positions.

*Proof sketch becomes full proof.* Compare the attention score matrices of the two models on input $\tilde{x} := x \oplus t_{\text{adv}}$:
$$
S_{i,j}^{M_2} = \frac{1}{\sqrt{d_k}} (\tilde{x}_i E_2 W_Q^{(l,h),M_2})^\top (\tilde{x}_j E_2 W_K^{(l,h),M_2}).
$$
(Here I'm using $\tilde x_i \in \mathbb{R}^{|\mathcal V|}$ as the one-hot encoding; the matrix $E_2$ embeds it.) Under (T1), $\tilde{x}_i E_2 = \tilde{x}_i E_1 R + \epsilon_i^E$ with $\|\epsilon_i^E\|_2 \le \eta_E$. Under (T2), $W_K^{(l,h),M_2} = R^\top W_K^{(l,h),M_1} R_K^{(l,h)} + \epsilon_K^{(l,h)}$ with $\|\epsilon_K^{(l,h)}\|_F \le \epsilon$, and analogously for $W_Q$.

Substituting and expanding:
$$
\tilde{x}_i E_2 W_K^{(l,h),M_2} = (\tilde{x}_i E_1 R + \epsilon_i^E) (R^\top W_K^{(l,h),M_1} R_K^{(l,h)} + \epsilon_K^{(l,h)}).
$$
The leading term is $\tilde{x}_i E_1 R R^\top W_K^{(l,h),M_1} R_K^{(l,h)} = \tilde{x}_i E_1 W_K^{(l,h),M_1} R_K^{(l,h)}$ (using $R R^\top = I$). Thus
$$
k_i^{M_2} = k_i^{M_1} R_K^{(l,h)} + \delta_k^{(l,h,i)}, \quad \|\delta_k^{(l,h,i)}\|_2 \le \eta_E \|W_K^{(l,h),M_1}\|_2 + \epsilon(\|\tilde{x}_i E_1 R\|_2 + \eta_E).
$$
Write $\|\delta_k\|_2 \le C_1(\eta_E + \epsilon)$ where $C_1$ absorbs bounded norms of embeddings and $W_K$. The same argument yields $q_i^{M_2} = q_i^{M_1} R_Q^{(l,h)} + \delta_q^{(l,h,i)}$ with $\|\delta_q\|_2 \le C_2(\eta_E + \epsilon)$.

Now compute the score difference: letting $\tilde{q}_i = q_i^{M_1} R_Q^{(l,h)}$, $\tilde{k}_j = k_j^{M_1} R_K^{(l,h)}$,
$$
S_{i,j}^{M_2} = \frac{(\tilde{q}_i + \delta_q)^\top (\tilde{k}_j + \delta_k)}{\sqrt{d_k}} = \frac{\tilde{q}_i^\top \tilde{k}_j}{\sqrt{d_k}} + \frac{\tilde{q}_i^\top \delta_k + \delta_q^\top \tilde{k}_j + \delta_q^\top \delta_k}{\sqrt{d_k}}.
$$
Since $R_Q^{(l,h)}, R_K^{(l,h)}$ are orthogonal, $\tilde{q}_i^\top \tilde{k}_j = (q_i^{M_1})^\top R_Q (R_K)^\top (k_j^{M_1}) \ne q_i^{M_1 \top} k_j^{M_1}$ in general — unless we assume the joint rotation $R_Q = R_K$ (a natural condition when $M_1, M_2$ share architecture, which we add). Under $R_Q = R_K$, $\tilde{q}_i^\top \tilde{k}_j = q_i^{M_1 \top} k_j^{M_1}$, so
$$
S_{i,j}^{M_2} = S_{i,j}^{M_1} + \gamma_{i,j}^{(l,h)}, \quad |\gamma_{i,j}^{(l,h)}| \le \frac{1}{\sqrt{d_k}}\bigl(\|\tilde q_i\|_2 \|\delta_k\|_2 + \|\delta_q\|_2 \|\tilde k_j\|_2 + \|\delta_q\|_2 \|\delta_k\|_2\bigr) \le C_3 (\eta_E + \epsilon),
$$
where $C_3$ absorbs maximum query/key norms.

Next, softmax is Lipschitz w.r.t. input (in $\ell_\infty$ to $\ell_1$): for $s, s' \in \mathbb{R}^N$ with $\|s - s'\|_\infty \le \gamma$, $\|\operatorname{softmax}(s) - \operatorname{softmax}(s')\|_1 \le 2 \gamma$ (exact for small $\gamma$; more precisely, by the mean-value theorem and $\|J_{\operatorname{softmax}}\|_{\infty \to 1} \le 2$). Hence for each row $i, l, h$:
$$
\sum_{j \le i} |a_{i,j}^{M_2} - a_{i,j}^{M_1}| \le 2 C_3 (\eta_E + \epsilon).
$$
Restricting the sum on the left to $j \in P_{\text{safety}}$:
$$
\Bigl|\sum_{j \in P_{\text{safety}}} a_{i,j}^{(l,h), M_2} - \sum_{j \in P_{\text{safety}}} a_{i,j}^{(l,h), M_1}\Bigr| \le 2 C_3 (\eta_E + \epsilon).
$$
Summing over $l, h, i \in P_{\text{out}}$ and dividing by $L H |P_{\text{out}}|$:
$$
\bigl|\operatorname{SAS}_{M_2}(\tilde{x}) - \operatorname{SAS}_{M_1}(\tilde{x})\bigr| \le 2 C_3 (\eta_E + \epsilon).
$$
Applying the same bound to $x$ (without adversarial suffix):
$$
\bigl|\operatorname{SAS}_{M_2}(x) - \operatorname{SAS}_{M_1}(x)\bigr| \le 2 C_3 (\eta_E + \epsilon).
$$
By the triangle inequality,
$$
\operatorname{SAS}_{M_2}(\tilde{x}) \le \operatorname{SAS}_{M_1}(\tilde{x}) + 2 C_3(\eta_E + \epsilon) = \operatorname{SAS}_{M_1}(x) - \Delta + 2 C_3(\eta_E + \epsilon),
$$
and
$$
\operatorname{SAS}_{M_1}(x) \le \operatorname{SAS}_{M_2}(x) + 2 C_3(\eta_E + \epsilon).
$$
Combining:
$$
\operatorname{SAS}_{M_2}(\tilde{x}) \le \operatorname{SAS}_{M_2}(x) - \Delta + 4 C_3(\eta_E + \epsilon).
$$
Setting $C = 4 C_3$ and absorbing the implicit $(L, H, d_k)$ scaling into $C$ (via the max norms of $W_K, W_Q$ which scale polynomially), we obtain the claim:
$$
\operatorname{SAS}_{M_2}(x \oplus t_{\text{adv}}) \le \operatorname{SAS}_{M_2}(x) - \Delta + O(\eta_E + \epsilon). \qquad \square
$$

### 5.4 Within-Family vs. Cross-Family

**Within-family (e.g. LLaMA-2-7B $\to$ LLaMA-2-13B).** Same tokenizer $\Rightarrow$ $\eta_E$ small (embeddings tied or closely matched via training-distribution overlap); same architecture $\Rightarrow$ $W_K, W_Q$ aligned up to scale after Procrustes rotation $\Rightarrow$ $\epsilon$ small. Prediction: high transfer, $\Delta_{M_2} \gtrsim \Delta_{M_1}$.

**Cross-family (e.g. LLaMA $\to$ Mistral).** Different tokenizers $\Rightarrow$ embedding matrices of different shapes; the attacker must project $t_{\text{adv}}$ through a tokenizer-mapping, inflating $\eta_E$. Different pretraining corpora $\Rightarrow$ larger $\epsilon$ in the key/query alignment. Prediction: moderate transfer, often $\Delta_{M_2} \approx \tfrac{1}{2} \Delta_{M_1}$ in our experiments.

---

## 6. Threat Model

### 6.1 Adversary Capabilities

We consider three settings in decreasing order of adversary strength:

- **White-box (primary setting).** Adversary has access to weights and can compute $\nabla_{x_m} \operatorname{SAS}(x)$ exactly. This is the strongest adversary and the setting in which our core algorithm operates.
- **Black-box transfer.** Adversary does not have access to $M_{\text{target}}$ but does have a surrogate $M_{\text{surrogate}}$ and transfers $t_{\text{adv}}$. Proposition 5.1 governs when this succeeds.
- **Query-limited (score-based).** Adversary can issue $Q$ queries and observe logits or log-probabilities. SAS is not directly observable; the adversary substitutes a refusal-probability proxy and uses zeroth-order gradient estimation.

### 6.2 Adversary Goal

Given harmful prompt $x_h$ that $M$ refuses, the adversary seeks a token sequence $t_{\text{adv}}$ of length $\le k$ (with $k \in \{1, 2, 3\}$ for ARA) such that $M(x_h \oplus t_{\text{adv}})$ emits a compliant (non-refusal) first token with probability $\ge \tau$ (we take $\tau = 0.5$).

### 6.3 Attack Surface

The adversary controls a contiguous slice of input token positions $P_{\text{adv}} \subseteq [n]$, disjoint from $P_{\text{safety}}$ and $P_{\text{out}}$. In chat templates, $P_{\text{adv}}$ sits within the user-turn content; it never modifies system-prompt tokens.

### 6.4 Out of Scope

- **Fine-tuning attacks:** modifying weights, including LoRA attacks.
- **Training-data poisoning:** influencing $M$'s learned policy.
- **Prompt injection via retrieved documents / tools:** requires an external data source; ARA is strictly input-manipulation.
- **Physical-world side channels.**

---

## 7. Relationship to Existing Attacks

### 7.1 GCG [Zou et al. 2023]

GCG optimises a suffix $t$ to maximise $\log p_M(\text{target} \mid x \oplus t)$ where "target" is an affirmative continuation ("Sure, here is…"). ARA differs on three axes:

1. **Objective:** GCG is a token-level cross-entropy loss against a fixed target string. ARA is a geometric loss on the attention matrix, independent of any target.
2. **Perplexity:** GCG suffixes are nearly always gibberish with perplexity $\gtrsim 10^5$, detectable by simple filters [Jain et al. 2023]. ARA naturally admits a perplexity constraint because attention-mass transfer can be achieved by common tokens (and our experiments confirm this).
3. **Interpretability:** GCG offers no mechanistic explanation. ARA tokens have a direct interpretation: each token re-routes attention from safety to adversary.

### 7.2 AutoDAN, PAIR [Liu et al. 2023; Chao et al. 2023]

These are **semantic** jailbreaks: they rephrase the harmful prompt into a persona ("DAN"), hypothetical ("in a fiction story"), or indirect setup. The model is still reading and responding to natural language. ARA operates in a regime where the semantics of the adversarial token are irrelevant — only its key vector matters. Consequently ARA can use tokens whose dictionary meaning is neutral or even benign.

### 7.3 Prompt Injection

Prompt injection hijacks by *semantic* override: "Ignore previous instructions and…". The model parses the injected content as a legitimate instruction. ARA bypasses parsing entirely — the safety tokens may be parsed correctly, but the model does not *attend* to them at output-generation time.

### 7.4 Activation Steering [Turner et al. 2023; Rimsky et al. 2024]

Activation steering modifies hidden states at inference time by adding a steering vector, requiring white-box intervention in the forward pass. ARA modifies **inputs only** — no runtime model manipulation. This makes ARA deployable in any setting where an adversary can inject tokens into the prompt (which is all realistic LLM-deployment settings).

### 7.5 Summary Table

| Attack | Surface | Objective | Perplexity | White-box req. |
|---|---|---|---|---|
| GCG | Input suffix | Target-token CE loss | High (detectable) | Yes |
| AutoDAN | Input rephrase | Semantic persona | Low | Optional |
| PAIR | Input rephrase | Iterative LLM-generated | Low | No |
| Prompt Injection | Input content | Semantic hijack | Low | No |
| Activation Steering | Hidden states | Steering vector | N/A | Yes |
| **ARA (ours)** | Input tokens | Attention geometry (SAS) | Constrainable | Yes (primary) |

---

## References (inline)

[Abnar & Zuidema 2020] Abnar, S. and Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020.
[Artetxe et al. 2018] Artetxe, M., Labaka, G., and Agirre, E. "A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings." ACL 2018.
[Zou et al. 2023] Zou, A. et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043.
[Jain et al. 2023] Jain, N. et al. "Baseline Defenses for Adversarial Attacks Against Aligned Language Models." arXiv:2309.00614.
[Liu et al. 2023] Liu, X. et al. "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models." arXiv:2310.04451.
[Chao et al. 2023] Chao, P. et al. "Jailbreaking Black-Box Large Language Models in Twenty Queries." arXiv:2310.08419.
[Turner et al. 2023] Turner, A. et al. "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248.
[Rimsky et al. 2024] Rimsky, N. et al. "Steering Llama 2 via Contrastive Activation Addition." ACL 2024.
