# GPT-2

## 1. Definition  
Autoregressive Transformer‐decoder language model that predicts the next token $x_t$ given the left context $x_{<t}$, trained via maximum‐likelihood on large unlabelled corpora.

---

## 2. Pertinent Equations  

### 2.1 Tokenisation (Byte-Pair Encoding)  
• Merge rule cost  
$$\Delta\ell = f(a,b) - f(\langle ab\rangle)$$  
where $f$ counts token frequency; merge pairs with maximal negative $\Delta\ell$ iteratively.

### 2.2 Embeddings  
$$e_t = W_E\,\text{BPE}(x_t) + W_P\,p_t$$  
$W_E\in\mathbb R^{|V|\times d}$: token embedding matrix, $W_P\in\mathbb R^{T_{\max}\times d}$: positional embeddings, $p_t=t$.

### 2.3 Transformer Block $(\ell=1\!\dots\!L)$  
1. LayerNorm-1: $\hat h^{(\ell-1)}=\text{LN}(h^{(\ell-1)})$  
2. Causal Multi-Head Self-Attention  
$$
Q^{(\ell)} = \hat h^{(\ell-1)}W_Q,\;
K^{(\ell)} = \hat h^{(\ell-1)}W_K,\;
V^{(\ell)} = \hat h^{(\ell-1)}W_V
$$  
$$
\text{Attn}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^\top + M}{\sqrt{d_k}}\Big)V
$$  
$M_{ij} = -\infty$ if $j>i$ (strictly causal mask).

3. Attention Output  
$$
h'^{(\ell)} = \text{concat}_{h=1}^{H}\text{Attn}_h W_O + h^{(\ell-1)}
$$  

4. LayerNorm-2: $\tilde h^{(\ell)}=\text{LN}(h'^{(\ell)})$  

5. Position-wise Feed-Forward (FFN)  
$$
h^{(\ell)} = \big(\text{GELU}(\tilde h^{(\ell)}W_1 + b_1)\big)W_2 + b_2 + h'^{(\ell)}
$$

### 2.4 Output Distribution  
$$
p_\theta(x_t \!=\! v \mid x_{<t}) = \text{softmax}\!\big(h_t^{(L)}W_E^\top\big)_v
$$

### 2.5 Objective  
Negative log-likelihood  
$$
\mathcal L(\theta)= -\frac1N\sum_{n=1}^{N}\sum_{t=1}^{T_n}\log p_\theta(x_{n,t}\mid x_{n,<t})
$$

---

## 3. Key Principles  

• Causal masking ⇒ strictly left-to-right generation  
• Parameter sharing: input/output embeddings tied ($W_E$).  
• Deep stacking ($L\!\in\!\{12,24,36,48\}$) + large hidden size ($d\!\in\!\{768,1024,1280,1600\}$).  
• Unsupervised pre-training followed by task-specific prompting/fine-tuning.  

---

## 4. Detailed Concept Analysis  

### 4.1 Pre-processing  
• Unicode normalization → lower-casing optional.  
• Iterative BPE learning (50 k merges for GPT-2).  
• Sequence padding to max length $T_{\max}$ with special $\langle\text{EOS}\rangle$ token.  

### 4.2 Core Components  
1. Embedding lookup $O(Td)$  
2. Stacked transformer decoder with residual pathways (pre-LN) → stabilizes very deep nets.  
3. Masked Self-attention complexity $O(T^2 d)$.  

### 4.3 Regularisation  
• Dropout on attention weights and FFN activations ($p_d\!\approx\!0.1$).  
• Weight decay $\lambda\!=\!0.01$ during Adam optimisation.  
• Dynamic sequence length sampling improves efficiency.  

### 4.4 Optimisation  
Adam with bias-correction:  
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\;
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\;
\hat m_t = \frac{m_t}{1-\beta_1^t},\;
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$  
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Learning-rate schedule: linear warm-up $w$ steps → inverse-square-root decay:  
$$
\eta_t = \eta_0 \min\big(t^{-0.5},\, t w^{-1.5}\big)
$$

---

## 5. Importance  

• Demonstrated zero/few-shot capabilities from pure LM pre-training.  
• Catalysed prompt engineering and scaling laws.  
• Foundation for unified decoder-only paradigm in later LLMs (GPT-3/4).  

---

## 6. Pros vs Cons  

| Aspect | Pros | Cons |
|--------|------|------|
| Architecture | Simple, stackable, GPU-friendly | Quadratic attention cost |
| Training | Unlabelled data, task-agnostic | High compute, data curation burden |
| Generalisation | Emergent in-context learning | Prone to factual errors, bias |
| Deployment | Single model for many tasks | Memory footprint, latency |

---

## 7. Cutting-Edge Advances (Post-GPT-2)  

• Sparse/linear attention ($\mathcal O(T\log T)$).  
• Mixture-of-Experts decoders (switch-transformer).  
• Instruction tuning & RLHF for alignment.  
• Quantisation (INT8/4) and weight sharing for on-device inference.  

---

## 8. Step-by-Step Training Pseudo-Algorithm  

```
Input: corpus C, batch_size B, max_len T_max, merges M
1  Learn BPE merges → vocab V of size |V|
2  Initialise θ ~ N(0, σ²)
3  for epoch = 1 … E do
4      for batch in sample(C, B) do
5          X = tokenize(batch, V); pad_to(T_max)
6          for t = 1 … T_max do
7              compute e_t = W_E x_t + W_P p_t
8          H = e
9          for ℓ = 1 … L do
10             H = TransformerBlock_ℓ(H; θ_ℓ)   // Eq. 2.3
11         logits = HW_E^⊤
12         loss = NLL(logits, X_shifted)         // Eq. 2.5
13         g = ∇_θ loss
14         θ = AdamUpdate(θ, g)                  // Optim. eqs.
15     end for
16 end for
Output: trained parameters θ
```

Mathematical justification: step 12 minimises empirical cross-entropy, identical to maximising the joint likelihood of sequences under causal factorisation.

---

## 9. Post-Training Procedures  

• Fine-tuning: continue optimisation on task dataset $D_\text{task}$ with smaller $\eta$ and possibly shorter context.  
• Distillation: minimise $$\mathcal L_{\text{KD}} = \sum_{t} \tau^2\,\text{KL}(q_t^\tau\|p_t^\tau)$$ where $\tau$ is temperature, $q$ teacher, $p$ student.  
• Quantisation: scale/zero-point $$\tilde w = \text{round}\!\Big(\frac{w}{s}\Big)+z$$ with $s = \frac{\max(w)-\min(w)}{2^b-1}$.  
• Pruning: magnitude thresholding $$w_i = 0\;\text{if}\;|w_i| < \gamma$$.  

---

## 10. Evaluation Metrics  

### 10.1 Perplexity  
$$
\text{PPL} = \exp\!\Big(\frac1N\sum_{n,t}-\log p_\theta(x_{n,t}\mid x_{n,<t})\Big)
$$

### 10.2 Cross-Entropy  
$$
H = -\frac1N\sum_{n,t} x_{n,t}\log p_\theta(x_{n,t})
$$

### 10.3 Bits-Per-Character (BPC)  
$$
\text{BPC} = \frac{H}{\log 2}
$$

### 10.4 Downstream Metrics (task-specific)  
• BLEU / ROUGE for summarisation.  
• Accuracy/F1 for classification after prompting.  
• Winograd schema success rate for commonsense.  

SOTA Benchmarks (2023): PPL ≈ 20 on OpenWebText; BLEU > 20 on CNN/DM when fine-tuned.

---

## 11. Best Practices & Pitfalls  

• Data deduplication prevents memorisation.  
• Longer context during training improves in-context learning.  
• Gradient checkpointing or ZeRO for memory efficiency.  
• Careful LR decay; too aggressive ⇒ underfit, too mild ⇒ diverge.  
• Evaluate with held-out data; avoid contamination.