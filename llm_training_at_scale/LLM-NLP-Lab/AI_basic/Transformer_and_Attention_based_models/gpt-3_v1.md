# GPT-3

## 1. Definition  
Autoregressive decoder-only transformer with up to $L{=}96$ layers, hidden size $d{=}12288$, $H{=}96$ attention heads, and $\approx 175$ B parameters, trained via maximum-likelihood to model the distribution $$p_\theta(x)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).$$  

---

## 2. Pertinent Equations  

### 2.1 Byte-Pair Encoding (BPE)  
Merge selection: $$\langle a,b\rangle^\*=\arg\min_{(a,b)}\bigl[-\Delta f(a,b)\bigr],$$  
update corpus until $|V|=50{,}257$.

### 2.2 Token & Positional Embeddings  
$$e_t = W_E\,\text{onehot}(x_t)+W_P\,p_t,$$  
$W_E\in\mathbb R^{|V|\times d},\;W_P\in\mathbb R^{T_{\max}\times d}.$  

### 2.3 Pre-LayerNorm Transformer Block $(\ell)$  
1. LN-1: $$\hat h^{(\ell-1)}=\mathrm{LN}(h^{(\ell-1)}).$$  
2. Multi-Head Causal Attention  
   • Projections $$Q,K,V=\hat h^{(\ell-1)}W_Q,\,\hat h^{(\ell-1)}W_K,\,\hat h^{(\ell-1)}W_V.$$  
   • Scores $$A=\frac{QK^\top+M}{\sqrt{d_k}},\quad M_{ij}=\begin{cases}0&i\ge j\\-\infty&i<j\end{cases}.$$  
   • Weights $$P=\mathrm{softmax}(A).$$  
   • Output $$Z=\mathrm{concat}_{h=1}^{H}(P_hV_h)W_O.$$  
3. Residual-1: $$h'^{(\ell)}=h^{(\ell-1)}+Z.$$  
4. LN-2: $$\tilde h^{(\ell)}=\mathrm{LN}(h'^{(\ell)}).$$  
5. FFN (GELU)  
   $$\phi(\tilde h^{(\ell)})=(\mathrm{GELU}(\tilde h^{(\ell)}W_1+b_1))W_2+b_2.$$  
6. Residual-2: $$h^{(\ell)}=h'^{(\ell)}+\phi(\tilde h^{(\ell)}).$$  

### 2.4 Output Softmax  
$$p_\theta(x_t=v\mid x_{<t})=\mathrm{softmax}(h_t^{(L)}W_E^\top)_v.$$

### 2.5 Loss  
Negative log-likelihood  
$$\mathcal L=-\frac1N\sum_{n=1}^{N}\sum_{t=1}^{T_n}\log p_\theta(x_{n,t}\mid x_{n,<t}).$$  

---

## 3. Key Principles  

• Strict left-to-right causal masking  
• Large-scale pre-training (“compute‐optimal” scaling laws)  
• Weight tying ($W_E$ for input/output)  
• Pre-LayerNorm for stability at $>100$ B parameters  
• Context window $T_{\max}=2048$  

---

## 4. Detailed Concept Analysis  

### 4.1 Data Pipeline  
• Common Crawl + Books + Wikipedia → 570 GB deduplicated text  
• Quality filtering via $$s(x)=\sigma(w^\top\mathrm{tfidf}(x))$$ retain $s(x)>\tau$.  
• Random window sampling to length $T\!\sim\!\mathrm{Uniform}(1,\,T_{\max})$ each step.

### 4.2 Regularisation  
• Dropout $p_d=0.1$ on attention/FFN outputs  
• Weight decay $\lambda=0.01$  
• Gradient clipping $\Vert g\Vert_2\le1$.

### 4.3 Optimisation  
AdamW:  
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\;v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$  
$$\theta_{t+1}=\theta_t-\eta_t\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}-\eta_t\lambda\theta_t.$$  

LR schedule: $$\eta_t=\eta_{\max}\cdot\min\!\bigl(t/w,\,(t/E)^{-0.5}\bigr).$$  

### 4.4 Memory & Parallelism  
• Model parallel tensor-splits across $G$ GPUs: $$W_Q=[W_Q^{(1)}\;\dots\;W_Q^{(G)}].$$  
• ZeRO-3 sharded optimizer state.

---

## 5. Importance  

• Enabled zero-shot and few-shot prompting at unprecedented scale  
• Empirical validation of power-law scaling $$\mathcal L\propto N^{-0.095}$$ with tokens $N$  
• Foundation for instruction-following models (InstructGPT, ChatGPT).

---

## 6. Pros versus Cons  

| Pros | Cons |
|------|------|
| Strong emergent abilities | Quadratic attention $\mathcal O(T^2)$ |
| Single model, many tasks | Massive compute/energy footprint |
| Few-shot adaptability | Bias, toxicity, factual errors |
| Public API monetisation | Difficult on-device deployment |

---

## 7. Cutting-Edge Advances  

• Sparse & linear attention (FlashAttention, Hyena)  
• Mixture-of-Experts: $$y=\sum_{k}p_k(x)f_k(x)$$ reduces flops/param ratio  
• Alignment via RLHF: $$\max_\theta\mathbb E_{x\sim\pi_\theta}[r(x)]-\beta \mathrm{KL}(\pi_\theta\|\pi_{\text{ref}}).$$  
• Quantisation to 4-bit (GPTQ, AWQ) with negligible perplexity loss.

---

## 8. Training Pseudo-Algorithm  

```
Input: corpus C, vocab V, max_len 2048, epochs E, batch B
1  θ ← N(0,σ²); opt ← AdamW(β1=.9,β2=.95,ε=1e−8)
2  for epoch = 1…E:
3      for batch = 1…⌈|C|/B⌉:
4          X ← sample_window(C, B, 1…2048)   # BPE-tokenised
5          for t = 1…len(X):
6              e_t ← W_E x_t + W_P t
7          H ← e
8          for ℓ = 1…L:
9              H ← TransformerBlock_ℓ(H, θ_ℓ)    # Sec 2.3
10         logits ← H W_E^⊤
11         loss ← NLL(logits, X_shifted)
12         g ← ∇_θ loss ; g ← clip(g,1)
13         θ ← AdamW(θ,g)                        # Sec 4.3
14     end
15 end
Output: θ
```

Justification: Step 11 minimises cross-entropy, equivalent to maximising the empirical log-likelihood of sequences under causal factorisation.

---

## 9. Post-Training Procedures  

• Instruction fine-tuning: continue SGD on $(\text{prompt},\text{response})$ pairs  
• RLHF: PPO with reward model $r_\phi$  
  $$\mathcal L_{PPO}=\mathbb E\big[\min(r_t A_t,\;\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\big]$$  
• Knowledge distillation: $$\mathcal L_{KD}=\tau^2\mathrm{KL}(q^\tau\|p^\tau).$$  
• Quantisation: symmetric scale $$\tilde w=\mathrm{round}(w/s),\;s=\tfrac{\max|w|}{2^{b-1}-1}.$$  

---

## 10. Evaluation Metrics  

### 10.1 Perplexity  
$$\mathrm{PPL}=\exp\!\bigl(\tfrac1N\sum_{n,t}-\log p_\theta(x_{n,t}\mid x_{n,<t})\bigr).$$  

### 10.2 Cross-Entropy  
$$H=-\tfrac1N\sum_{n,t}\log_2 p_\theta(x_{n,t}\mid x_{n,<t}).$$  

### 10.3 Zero/Few-Shot Benchmarks  
• LAMBADA accuracy, TriviaQA EM, Winogrande, ARC-Challenge.  

### 10.4 Calibration  
Expected calibration error  
$$\mathrm{ECE}=\sum_{k=1}^{K}\frac{|B_k|}{N}\bigl|\mathrm{acc}(B_k)-\mathrm{conf}(B_k)\bigr|.$$  

SOTA (public, 2023): GPT-3 175 B perplexity ≈ 20 (OpenWebText), LAMBADA 86 % (0-shot).  

---

## 11. Best Practices & Pitfalls  

• Rigorous deduplication avoids memorisation leakage  
• Mixed-precision + gradient checkpointing for memory efficiency  
• Monitor divergence signs (spikes in $\Vert g\Vert_2$) early; apply LR back-off  
• Evaluate on unseen domains to detect overfitting  
• Mitigate bias with curated data & alignment objectives