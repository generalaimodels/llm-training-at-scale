# Chapter 8 Transformer-Based Language Models  

---

## 8.0 General Definition  
A transformer-based language model is a parameterized function  
$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$  
implemented by stacking $L$ self-attention and position-wise feed-forward layers that map a discrete token sequence $x_{1:T}$ to a distribution over the next token. Core operations:  

1. Token/position embedding  
   $$h_t^{(0)}=E\,x_t+P_t$$  
2. Repeated layers $l=1\dots L$  
   $$\tilde h^{(l)}=\mathrm{LN}\!\bigl(h^{(l-1)}+ \mathrm{SA}^{(l)}(h^{(l-1)})\bigr)$$  
   $$h^{(l)}=\mathrm{LN}\!\bigl(\tilde h^{(l)}+\mathrm{FFN}^{(l)}(\tilde h^{(l)})\bigr)$$  
3. Output softmax  
   $$p_\theta(x_{t+1}\!=\!v\mid x_{\le t})=\mathrm{softmax}_v\bigl(W_o h_t^{(L)}\bigr)$$  

---

## 8.1 GPT-n Series (Decoder-Only, Dense Attention)

### 8.1.1 Concise Definition  
GPT-n models (GPT-1, GPT-2, GPT-3, GPT-4, …) are unidirectional, decoder-only transformers trained with the causal language-modeling objective across massive text corpora, scaling parameters from $1.2\times10^8$ to $>\!10^{12}$.

### 8.1.2 Mathematical Formulation  

Self-attention (masked):  
$$\mathrm{SA}(H)=\mathrm{Mask}\bigl(\alpha\bigr)V,\qquad  
\alpha=\mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)$$  
where $Q=HW_Q$, $K=HW_K$, $V=HW_V$, $H\in\mathbb R^{T\times d_\text{model}}$. The autoregressive mask sets $\alpha_{ij}=0$ for $j>i$.

Training loss (cross-entropy):  
$$\mathcal L = -\frac1{N}\sum_{n=1}^N\sum_{t=1}^{T_n}\log p_\theta(x_{n,t}\!\mid\!x_{n,<t}).$$  

### 8.1.3 End-to-End Explanation  

• Data Pipeline  
 – Diverse internet corpora → cleaning → de-duplication → tokenization via byte-pair encoding or SentencePiece → shuffled contiguous blocks length $T$ ($T\!\in\!\{512,2048,4096\}$).  

• Model Architecture  
 – Embedding size $d_\text{model}$, heads $h$, layers $L$, feed-forward dimension $d_\text{ff}=4d_\text{model}$.  
 – Rotary or learned positional encodings for long-context variants.  

• Parallelization  
 – Tensor-parallel (TP): split $W_Q,W_K,W_V,W_O$ across devices.  
 – Pipeline-parallel (PP): contiguous layer stages.  
 – ZeRO-DPS / FSDP: optimizer state + gradient sharding.  
 – NVLink/InfiniBand for high-bandwidth all-reduce.  

• Training Procedure  
 – Mixed-precision (fp16/bf16) with loss-scale.  
 – AdamW with linear warm-up then cosine decay; learning-rate $\eta\propto \text{batch}^{-0.5}$.  
 – Gradient-clip by global norm.  
 – Checkpoint-reuse and activation recomputation to fit memory.  

• Scaling Laws  
 – Empirical: $$\mathcal L(N, D, C)\approx A N^{-\alpha}+B D^{-\beta}+C_\text{data}C^{-\gamma}$$  
  where $N$ params, $D$ data tokens, $C$ compute. Observed $\alpha\!\approx\!0.076$. Guides optimal allocation.  

• Inference  
 – Next-token logits $z$ → sampling strategy:  
  Greedy: $\arg\max_v z_v$  
  Temperature: $p_v\propto\exp(z_v/T)$  
  Top-$k$, nucleus (top-$p$), beam search with length-penalty.  
 – KV-cache: store $(K,V)$ per layer to reduce $O(T^2)$ to $O(T)$ for incremental decoding.  

• Evaluation  
 – Perplexity on held-out corpora.  
 – Downstream zero-shot tasks: LAMBADA, SuperGLUE, MMLU.  
 – Human preference via RLHF.  

• Alignment & Safety  
 – Supervised fine-tuning on human-written instructions.  
 – RLHF: reward model $r_\phi(y\mid x)$, policy gradient (PPO, DPO) to maximize $\mathbb E_{y\sim\pi_\theta}[r_\phi(y)]$.  

---

## 8.2 Sparse Transformer  

### 8.2.1 Concise Definition  
Sparse Transformer replaces full $O(T^2)$ attention with structured sparse patterns (e.g., block-local + strided) achieving $O(T\sqrt T)$ complexity and enabling sequence lengths $T\!>\!16\!$K.

### 8.2.2 Mathematical Formulation  

Let $\mathcal A\subseteq\{0,\dots,T-1\}^2$ be the allowed query-key pairs. Attention weights:  
$$\alpha_{ij}=  
\begin{cases} 
\displaystyle\frac{\exp(q_i\cdot k_j/\sqrt{d_k})}{\sum_{(i,j')\in\mathcal A_i}\exp(q_i\cdot k_{j'}/\sqrt{d_k})} & (i,j)\in\mathcal A\\
0 & \text{otherwise}
\end{cases}$$  
where $\mathcal A_i=\{j\mid(i,j)\in\mathcal A\}$.

### 8.2.3 End-to-End Explanation  

• Attention Pattern  
 – Block-local: each position attends to tokens within its block of size $b$ ($O(Tb)$).  
 – Strided: each query attends to keys at stride $s$ ($O(T\frac{T}{s})$). Combined complexity $O(T(b+\frac{T}{s}))$.  

• Implementation  
 – Pre-compute sparse index tensors; use cuda kernel with gather/scatter.  
 – Memory savings proportional to sparsity density $|\mathcal A|/T^2$.  

• Training & Convergence  
 – Same objective as GPT, but may require larger $b$ early epochs to stabilize gradients.  
 – Empirically comparable perplexity with $\sim1.5$× speed-up.  

• Long-Context Use-Cases  
 – Genomic sequences ($T\!\sim\!10^5$), autoregressive image generation on 1D rasterized pixels, music modeling.  

---

## 8.3 Reformer  

### 8.3.1 Concise Definition  
Reformer attains $O(T\log T)$ time and $O(T)$ memory by (i) LSH-based attention that clusters similar queries/keys and (ii) reversible residual layers eliminating activation storage.

### 8.3.2 Mathematical Formulation  

LSH Rounds: for hash round $r=1\dots R$  
$$h_i^{(r)} = \mathrm{sign}(\mathbf R^{(r)} q_i),\qquad
\mathbf R^{(r)}\sim\mathcal N(0,1)^{d_k\times d_k}$$  
Indices with identical $h_i^{(r)}$ share buckets. Attention computed only within buckets ⇒ expected neighbors $O(\log T)$.

Reversible layer: given functions $F, G$  
$$y_1 = x_1 + F(x_2),\quad y_2 = x_2 + G(y_1)$$  
Backward reconstructs $x_1,x_2$ from $y_1,y_2$ ⇒ no forward activations stored.

### 8.3.3 End-to-End Explanation  

• Architecture  
 – Split $d_\text{model}$ into two halves $(x_1,x_2)$ for reversible blocks.  
 – Replace standard SA with LSH-attention; maintain causal chunking to preserve autoregression.  

• Complexity Analysis  
 – Hash sorting: $O(T\log T)$ via radix sort.  
 – Memory: only $O(1)$ extra for states; major matmul memory dominates at $O(Td_k)$.  

• Training Pipeline  
 – Sequence length up to 64 K on a single GPU.  
 – Adam optimizer; position encodings via sinusoid older than $T$.  

• Limitations & Mitigations  
 – LSH approximation may miss far dependencies; use multiple hash rounds $R=8$ and trainable rotation matrices.  
 – Hash collisions add noise; use causal chunk length to upper-bound loss.  

• Applications  
 – Document level translation, long-form summarization, time-series forecasting.  

---

## 8.4 Comparative Summary  

| Property | GPT-n (Dense) | Sparse Transformer | Reformer |
|----------|---------------|--------------------|----------|
| Attention Cost | $O(T^2)$ | $O(Tb+T^2/s)$ | $O(T\log T)$ |
| Memory | $O(T^2)$ | $\propto$ sparsity | $O(T)$ (reversible) |
| Maximum practical $T$ | 4 K–8 K | 16 K–32 K | 64 K–256 K |
| Approximate? | No | Deterministic sparse | Hash-based approximate |
| Typical Use | General LM | Very long context | Ultra-long context |

---

## 8.5 Research Frontiers  

• Mixture-of-Experts GPT (Switch, GLaM) for conditional compute.  
• Flash-Attention & Triton kernels bridging dense and sparse regimes.  
• Continual pre-training with data/pruning curricula optimizing scaling exponents.