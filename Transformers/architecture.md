# Sequence‑to‑Sequence, Attention, and Transformers — Detailed Notes

A complete, interview‑ready guide from classic encoder‑decoder RNNs to modern Transformers. Includes math, code, and practical tips.

---

## 1. Motivation: Sequence‑to‑Sequence Tasks

- Input length ≠ output length (e.g., translation, summarization).
- Order matters; outputs depend on the entire input and prior outputs.

Classic solution: Encoder‑Decoder architecture.

---

## 2. Encoder‑Decoder (General Idea)

- **Encoder**: reads the input sequence and produces a representation (context vector and/or sequence of hidden states).
- **Decoder**: generates the output sequence token‑by‑token, conditioned on encoder outputs and previous decoded tokens.

RNN/LSTM encoder‑decoder pipeline:

- Encoder: $h_i^{enc} = \text{RNN}(x_i, h_{i-1}^{enc})$ → final state $h_T^{enc}$
- Decoder: $s_t^{dec} = \text{RNN}(y_{t-1}, s_{t-1}^{dec}, \text{context})$, then $p(y_t \mid \cdot) = \text{softmax}(W_o s_t^{dec})$

---

## 3. Pre‑Transformer Era

- **RBMT (Rule‑Based MT)**: hand‑crafted rules; brittle and hard to scale.
- **NMT (Neural MT)**: RNN/LSTM encoder‑decoder; learns from data.
  - Better than RBMT but limited by:
    - Bottleneck: compressing all info into a single vector.
    - Vanishing gradients for long dependencies.
    - Poor parallelization due to sequential nature.

Is RNN equal to NMT? No. RNNs are one architecture used inside NMT; NMT is a system for translation.

---

## 4. Attention Mechanism (RNN/LSTM + Attention)

Attention alleviates the bottleneck by letting the decoder focus on relevant encoder states at each step.

Bahdanau (Additive) Attention:

$$
\begin{aligned}
 e_{t,i} &= v_a^\top \tanh(W_h h_i^{enc} + W_s s_{t-1}^{dec}) \\
 \alpha_{t,i} &= \text{softmax}_i(e_{t,i}) \\
 c_t &= \sum_i \alpha_{t,i} h_i^{enc}
\end{aligned}
$$

Decoder update uses $c_t$:

$$
 s_t^{dec} = \text{RNN}([y_{t-1}; c_t], s_{t-1}^{dec})
$$

Dot/Scaled‑Dot Product Attention (used in Transformers):

$$
 \text{score}(q, k) = \frac{q k^\top}{\sqrt{d_k}}, \quad \alpha = \text{softmax}(\text{score}), \quad \text{Attn}(Q,K,V) = \alpha V
$$

Difference vs plain RNN/LSTM:

- With attention: decoder looks at all encoder states, not just a fixed context → reduces bottleneck.
- Still sequential decoding → limited parallelism; many steps remain.

---

## 5. Transformers: Removing Recurrence

Key ideas:

- Self‑attention allows modeling all positions jointly.
- Full parallelization within layers (no RNN recurrence).
- Positional encodings inject order information.

### 5.1 Scaled Dot‑Product Attention

Given $Q \in \mathbb{R}^{n_q \times d_k}$, $K \in \mathbb{R}^{n_k \times d_k}$, $V \in \mathbb{R}^{n_k \times d_v}$:

$$
 \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 5.2 Multi‑Head Attention

- Multiple heads with separate learned projections $(W_Q^h, W_K^h, W_V^h)$.
- Heads capture diverse relations; concatenated and projected.

### 5.3 Positional Encoding (Sinusoidal)

Inject position $pos$ and dimension $i$:

$$
\begin{aligned}
 PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
 PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
$$

- Continuous, scale‑aware; enables relative position reasoning.
- Indexing alone is discrete and lacks smooth distance properties.

### 5.4 Residual Connections + LayerNorm

For sublayer output $\text{sublayer}(x)$:

$$
 \text{LayerNorm}(x + \text{sublayer}(x))
$$

- Stabilizes gradients and training of deep stacks.

### 5.5 Masked Self‑Attention (Causal)

- During autoregressive training, prevent attending to future tokens.
- Implemented by applying $-\infty$ mask to upper‑triangle before softmax.

### 5.6 Cross‑Attention (Encoder‑Decoder)

- Decoder queries attend to encoder keys/values: $Q=H^{dec}, K=H^{enc}, V=H^{enc}$.

---

### 5.7 Self‑ vs Cross‑ vs Masked Attention — Intuition & Analogies

- Self‑attention (encoder): each word asks, “Who in my sentence helps define me?”
  - Analogy: a round‑table where every participant listens to everyone else at the same time, then updates their own summary.
  - Math: $Q=K=V=X$ (same sequence); result is a weighted mix of token representations in the same layer.
- Cross‑attention (decoder): each generated token asks the encoder, “Which input words should I consult right now?”
  - Analogy: a translator (decoder) continually glances back at the source text (encoder outputs) while writing the translation.
  - Math: $Q=H^{dec}$, $K=V=H^{enc}$.
- Masked self‑attention (decoder): like writing left‑to‑right with a cover over future words so you can’t peek ahead.
  - Enforced by a causal mask (strictly lower‑triangular visibility).

Why masking? During training for autoregressive decoding, the model must learn to predict the next token using only past and present context, mirroring inference.

---

### 5.8 Multi‑Head Attention — From One Spotlight to a Team of Spotlights

- Single head = one similarity space (one “spotlight” view of relationships).
- Multi‑head = several independent projections and attentions in parallel; heads specialize (syntax, coreference, positional, etc.).
- Analogy: a committee of experts each reads the same paragraph and highlights different relevant parts; their notes are combined.

Math detail (one layer):

$$
\begin{aligned}
Q_h &= X W_Q^{(h)},\; K_h = X W_K^{(h)},\; V_h = X W_V^{(h)}, \quad h=1..H \\
	ext{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h \\
	ext{MHA}(X) &= \Big[\text{head}_1;\dots;\text{head}_H\Big] W_O
\end{aligned}
$$

with $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d_{model} \times d_k}$, concatenation over heads, and $W_O \in \mathbb{R}^{(H\cdot d_v)\times d_{model}}$.

Key design choices:

- $d_k = d_v = d_{model}/H$ keeps compute constant with varying $H$.
- $\tfrac{1}{\sqrt{d_k}}$ stabilizes softmax (prevents extremely peaky/flat distributions as dimensionality grows).

---

### 5.9 Masking — Causal and Padding Masks (Shapes and Use)

- Padding mask: hides padded tokens in attention so they contribute neither keys nor values.
  - Shape broadcast: typically `(B, 1, 1, T)` for encoder self‑attention; `(B, 1, T_q, T_k)` generally.
- Causal mask (decoder self‑attention): prevents attending to future positions.
  - Base mask: lower triangular of shape `(T, T)`; broadcast to `(B, heads, T, T)`.

Combining masks: logical AND in boolean space or addition in logit space (add `-inf` where masked).

---

### 5.10 Tiny Worked Example (Self‑Attention)

Suppose 3 tokens with 2‑dim embeddings after linear projections (toy values):

$Q = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$, $K = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$, $V = \begin{bmatrix}1&2\\2&1\\0&3\end{bmatrix}$, $d_k=2$.

Scores $= QK^\top/\sqrt{2}$:

$$
\frac{1}{\sqrt{2}}\begin{bmatrix}
1&0&1\\0&1&1\\1&1&2
\end{bmatrix}
$$

Row‑wise softmax → attention weights $\alpha$; output $= \alpha V$ yields each token as a weighted mix of $V$ rows. The third token (last row with highest self‑score) will put more weight on itself and its close neighbors.

Takeaway: attention is a content‑based, learned weighted average — not a fixed window.

---

### 5.11 Self vs Cross in Code

```python
# Self-attention (encoder): Q=K=V=enc
enc = enc + mha(enc, enc, enc, mask=enc_pad_mask)

# Decoder masked self-attention: Q=K=V=dec (causal)
dec = dec + mha(dec, dec, dec, mask=causal_mask & dec_pad_mask)

# Cross-attention: Q=dec, K=V=enc
dec = dec + mha(dec, enc, enc, mask=enc_pad_mask)
```

Note: many frameworks (e.g., PyTorch `nn.MultiheadAttention`) expect masks in particular shapes/types (bool vs additive). Always check API docs.

---

### 5.12 Practical Pitfalls & Tips

- Mask dtype/semantics: some APIs expect additive masks (float with `-inf` where masked), others expect boolean. Mismatch causes silent bugs.
- Precision: use fp16/bf16 with care; softmax stability benefits from scaled dot products and fused kernels (FlashAttention).
- Long sequences: attention is $\mathcal{O}(T^2)$. Consider windowed/sparse/linear attention or chunking for very long inputs.
- Head redundancy: more heads ≠ always better; pruning can remove redundant heads with minimal loss.
- Attention dropout: regularizes reliance on single strong alignments.

---

### 5.13 Advanced Variants (Pointer Map)

- Relative/rotary position (T5, RoPE): better extrapolation to longer lengths.
- ALiBi: biasing scores by position differences; removes explicit PE.
- Multi‑query attention: share K/V across heads to reduce memory bandwidth.
- Cross‑attention patterns beyond MT: retrieval‑augmented generation (K,V from an external memory/index).

---

## 6. Training Details

- **Teacher forcing**: feed ground‑truth previous token to the decoder during training to speed convergence.
- **Softmax**: converts logits to a probability distribution; used with cross‑entropy.
- **Masks**: padding masks (ignore pads) and causal masks (prevent future‑lookahead).
- **Label smoothing**: regularization that improves calibration in MT.

Note: "BPTL" is not standard; for RNNs we use BPTT (Backprop Through Time). Transformers use standard backprop through layers.

---

## 7. PyTorch: Scaled Dot‑Product and Multi‑Head Attention

```python
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (B, heads, T_q, d_k)
    # K: (B, heads, T_k, d_k)
    # V: (B, heads, T_k, d_v)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, heads, T, d_k)
        return x

    def forward(self, x_q, x_kv, mask=None):
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_kv))
        V = self.split_heads(self.W_v(x_kv))
        attn, _ = scaled_dot_product_attention(Q, K, V, mask)
        B, heads, T, d_k = attn.shape
        attn = attn.transpose(1, 2).contiguous().view(B, T, heads * d_k)
        return self.W_o(attn)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, mem=None):
        # Self-attention (use mem=x for encoder; mem for cross-att in decoder)
        attn_out = self.mha(x, mem if mem is not None else x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
```

Causal mask construction:

```python
def causal_mask(T):
    mask = torch.tril(torch.ones(T, T)).bool()  # (T, T)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
```

---

## 8. Positional Encodings (Sinusoidal) in Code

```python
import torch
import math

def sinusoidal_positional_encoding(T, d_model):
    pe = torch.zeros(T, d_model)
    position = torch.arange(0, T).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (T, d_model)
```

---

## 9. Practical Tips

- Use padding masks in encoder self‑attention to ignore padded tokens.
- Use causal masks in decoder self‑attention to prevent future peeking.
- Beam search improves sequence decoding quality vs greedy.
- Label smoothing can yield better BLEU and calibration.
- Residual connections and LayerNorm are critical for deep Transformers.

---

## 10. Interview Questions

- Conceptual:
  - Explain the encoder‑decoder architecture. What is the bottleneck and how does attention fix it?
  - Compare RNN/LSTM with attention vs Transformer. Why is Transformer faster to train?
  - Why do Transformers need positional encodings? Why not simple indexing?
  - Distinguish self‑attention, cross‑attention, and masked self‑attention.
  - Why multi‑head attention? What happens if we use only one head?
- Mathematical:
  - Derive scaled dot‑product attention and explain the $\sqrt{d_k}$ term.
  - Write Bahdanau attention equations and interpret $\alpha_{t,i}$.
  - Show how residual + LayerNorm stabilizes optimization in deep stacks.
- Practical/Coding:
  - Implement a causal mask and explain where it’s applied.
  - Add padding masks in attention to handle variable‑length batches.
  - Outline teacher forcing and its trade‑offs in seq2seq training.

---

## 11. Quick Formula Sheet

- Scaled Dot‑Product: $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$
- Sinusoidal PE: $\sin(\frac{pos}{10000^{2i/d}}),\ \cos(\cdot)$ alternating dims
- Bahdanau: $e_{t,i} = v_a^\top \tanh(W_h h_i + W_s s_{t-1}),\ \alpha=\text{softmax}(e)$
- Residual+Norm: $\text{LayerNorm}(x + \text{sublayer}(x))$
- Softmax: $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

---

## 12. Glossary

- **Encoder**: maps input tokens to representations.
- **Decoder**: generates output tokens autoregressively.
- **Self‑attention**: query/key/value from the same sequence.
- **Cross‑attention**: queries from decoder attend to encoder outputs.
- **Masked attention**: prevents access to future positions.
- **Teacher forcing**: training technique using ground‑truth previous outputs.
