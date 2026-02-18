# GPT-3.5

## 1. Definition  
Autoregressive decoder-only transformer obtained by continued pre-training of a GPT-3-class model on ∼450 B extra tokens, followed by supervised instruction tuning and RLHF alignment, yielding improved reasoning, safety, and chat capabilities while retaining ≈175 B parameters and context window $T_{\max}=4096$.

---

## 2. Pertinent Equations  

### 2.1 Tokenisation (BPE-R)  
Iterative rank-based merges:  
$$\langle a,b\rangle^\*=\arg\max_{(a,b)}\,f(a,b),\qquad f=\text{freq}(\langle a,b\rangle)$$  
Stop when $|V|=50\,257$; mapping $\text{BPE}:\mathcal X\!\to\!\mathbb N^{\le T_{\max}}$.

### 2.2 Embedding Layer  
$$e_t=W_E\,\text{onehot}(x_t)+W_P\,\varphi(t),\;\;W_E\in\mathbb R^{|V|\times d},\;\;W_P\in\mathbb R^{T_{\max}\times d}$$  
$\varphi(t)=t$ (absolute) or rotary: $\varphi_{\text{RoPE}}(t)=\bigl[\cos(t\omega_k),\sin(t\omega_k)\bigr]_{k=1}^{d/2}$.

### 2.3 Transformer Block (Pre-LN, $\ell=1\dots L$)  
LayerNorm-1: $$\hat h=\text{LN}(h)$$  
Causal Multi-Head Attention:  
$$\begin{aligned}
Q&=\hat h W_Q,\;K=\hat h W_K,\;V=\hat h W_V\\
A&=\frac{QK^\top+M}{\sqrt{d_k}},\;M_{ij}=-\infty\,[j>i]\\
P&=\text{softmax}(A),\quad Z=P V\\
h'&=h+Z W_O
\end{aligned}$$  
LayerNorm-2: $$\tilde h=\text{LN}(h')$$  
Gated FFN (SwiGLU):  
$$g=\sigma(\tilde h W_{1g})\odot(\tilde h W_{1})$$  
$$h^{\text{new}}=h'+g W_2$$  

### 2.4 Output Distribution  
$$p_\theta(x_t\!=\!v\mid x_{<t})=\text{softmax}(h_t^{(L)}W_E^\top)_v$$

### 2.5 Training Losses  
• Language modelling: $$\mathcal L_{\text{LM}}=-\tfrac1N\sum_{n,t}\log p_\theta(x_{n,t}\mid x_{n,<t})$$  
• Supervised instruction tuning: $$\mathcal L_{\text{sup}}=-\tfrac1N\sum_{n,t}\log p_\theta(y_{n,t}\mid x_n,y_{n,<t})$$  
• RLHF (PPO):  
$$\mathcal L_{\text{PPO}}=\mathbb E\bigl[\min(r_tA_t,\;\text{clip}(r_t,1\!-\!\epsilon,1\!+\!\epsilon)A_t)\bigr]$$  
$r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$.

---

## 3. Key Principles  

• Causal masking + rotary/absolute positions  
• Pre-LN for deep-scale stability ($L{=}96$, $d{=}12\,288$)  
• SwiGLU FFN ⇒ higher capacity at same FLOPs  
• Instruction supervision → alignment to natural language commands  
• RLHF maximises reward model $r_\phi$ while constraining KL to reference policy  

---

## 4. Detailed Concept Analysis  

### 4.1 Data Pre-processing  
- Deduplication via MinHash Jaccard $J(A,B)=\tfrac{|A\cap B|}{|A\cup B|}$ with threshold $<0.7$  
- Toxicity filter: $$\text{tox}(x)=\sigma(w^\top\text{CLS}(x))<\tau$$  
- Balanced mixture: 55 % Common-Crawl, 20 % code, 12 % books, 8 % Wikipedia, 5 % dialogues.

### 4.2 Core Architectural Parameters  
| Variant | $L$ | $d$ | $H$ | Params | $T_{\max}$ |
|---------|-----|-----|-----|--------|------------|
| Davinci-002 | 96 | 12 288 | 96 | 175 B | 4096 |
| Curie-002  | 48 | 6 144 | 48 | 13 B | 4096 |

### 4.3 Optimisation  
AdamW: $$\theta_{t+1}=\theta_t-\eta_t\bigl(\hat m_t/\sqrt{\hat v_t+\epsilon}+\lambda\theta_t\bigr)$$  
LR schedule: cosine decay after $w$ warm-up steps:  
$$\eta_t=\eta_{\max}\frac12\Bigl(1+\cos\!\frac{t-w}{T-w}\pi\Bigr)$$  
Gradient clipping: $\lVert g\rVert_2\le1$;  
Mixed precision (bfloat16); sequence parallelism + ZeRO-3 state sharding.

### 4.4 Regularisation  
- Dropout $0.1$ on attention & FFN  
- Attention dropout $p_a=0.1$  
- Weight decay $\lambda=0.1$

### 4.5 Alignment Pipeline  
1. Supervised fine-tune on $\mathcal D_{\text{sup}}$  
2. Reward model $r_\phi$ trained via pairwise ranking loss  
   $$\mathcal L_{\text{RM}}=-\log\sigma\bigl(r_\phi(y^+)-r_\phi(y^-)\bigr)$$  
3. PPO optimisation (Eq. 2.5) with KL-penalty $\beta$ automatically tuned.

---

## 5. Importance  

• Bridges gap between GPT-3 and GPT-4 on reasoning/chain-of-thought  
• Underpins ChatGPT; turned LLMs into mainstream interactive tools  
• Demonstrates effectiveness of iterative alignment (SFT → RLHF).

---

## 6. Pros vs Cons  

| Pros | Cons |
|------|------|
| Improved factuality & coherence | Still quadratic attention cost |
| Better safety vs GPT-3 | Susceptible to jailbreak prompts |
| Handles 4 k context | Context window still limiting for long docs |
| Strong code generation | Significant inference compute |

---

## 7. Cutting-Edge Advances  

• FlashAttention-2 reduces memory, exact $$O(T^2)$$ → nearly bandwidth-bound.  
• Multi-Query-Attention: single $K,V$ per head $$\Rightarrow O(THd_k)$$ memory.  
• Parameter-efficient tuning (LoRA, $W\mapsto W+\Delta W\!:=\!BA$ with low-rank $B,A$).  
• 4-bit GPTQ / AWQ quantisation with <1 PPL degradation.

---

## 8. Step-by-Step Training Pseudo-Algorithm  

```python
# Pre-training
for step in range(T_tokens // (B*T_max)):
    X = sample_tokens(corpus, B, T_max)   # BPE already applied
    H = embed(X)                          # Eq 2.2
    for ℓ in range(L):
        H = block[ℓ](H)                   # Eq 2.3
    logits = H @ W_E.T
    loss = cross_entropy(logits, X_shift) # Eq 2.5
    g = ∇θ loss ; g = clip(g, 1)
    θ = AdamW(θ, g)                       # Sec 4.3
# Supervised instruction tuning
θ ← SGD(θ, 𝔻_sup, loss=ℒ_sup)
# RLHF
for epoch in range(E):
    trajectories = sample_policy(π_θ, prompts)
    advantage = compute_GAE(trajectories, r_φ)
    θ = PPO_update(θ, trajectories, advantage)
return θ
```

Justification: Each optimisation stage minimises a surrogate bound on KL-regularised reward maximisation.

---

## 9. Post-Training Procedures  

• **LoRA**: $$\Delta W=BA,\;B\in\mathbb R^{d\times r},\;A\in\mathbb R^{r\times d},\;r\ll d$$  
• **Quantisation-Aware Calibration**: scale $$s=\frac{\max|w|}{2^{b-1}-1},\;\tilde w=\text{round}(w/s)$$  
• **RAG**: prepend retrieved context $c$; modifies probability $$p_\theta(x\mid c).$$  
• **Continual RLHF**: online collection of $$\{(x,y^+,y^-)\}$$, periodic PPO refresh.

---

## 10. Evaluation Metrics  

| Metric | Equation |
|--------|----------|
| Perplexity | $$\text{PPL}=\exp\!\bigl(\tfrac1N\sum_{n,t}-\log p_\theta(x_{n,t}\mid x_{n,<t})\bigr)$$ |
| Exact Match (QA) | $$\text{EM}=\tfrac1N\sum_{n}\mathbf1[y_n=\hat y_n]$$ |
| BLEU (gen.) | $$\text{BLEU}=BP\exp\!\bigl(\sum_{n=1}^{4}w_n\log p_n\bigr)$$ |
| CodeEval pass@k | $$\text{pass@k}=1-\prod_{i=1}^{m}\frac{\binom{n_i-k}{m_i}}{\binom{n_i}{m_i}}$$ |
| Toxicity (Perspective) | $$\text{TOX}=\tfrac1N\sum_{n}\text{persp}(\hat y_n)$$ |
| ECE | $$\text{ECE}=\sum_{b=1}^{B}\frac{|B_b|}{N}\bigl|\text{acc}(B_b)-\text{conf}(B_b)\bigr|$$ |

SOTA (public, 2024):  
• OpenAI eval harness GPT-3.5: LAMBADA 88 % (0-shot), MMLU 70 % (5-shot), HumanEval 48 % pass@1.

---

## 11. Best Practices & Pitfalls  

• Ensure training–eval split deduplication ($J<0.3$).  
• Monitor loss curvature; apply damped Adam if $\lambda_{\max}(\nabla^2\mathcal L)$ spikes.  
• Align temperature sampling: $$x_{t}\sim\text{softmax}(z_t/\tau)$$, small $\tau$ for deterministic output.  
• Guard against over-compression in quantisation ($b<4$) → sharp PPL increase.  
• Prompt sanitisation + system messages reduce jailbreak risk.