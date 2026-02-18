### 0. Notation  
• $X=\{x_1,\dots,x_T\}$ : token sequence (pre-BPE IDs)  
• $\tilde X=\{\tilde x_1,\dots,\tilde x_T\}$ : BPE-encoded IDs  
• $V$ : vocabulary size, $d_\text{model}$: hidden width, $L$: # decoder layers, $H$: # attention heads, $d_h=d_\text{model}/H$  
• $W_\*$: trainable matrices, $b_\*$: trainable biases, $\theta$: all parameters  
• $f_{\text{LN}}(\cdot)$: layer-norm, $σ(\cdot)$: GELU, $∘$: concatenation, $⊕$: element-wise add  

---

## 1. Data Pre-Processing  

#### 1.1 Byte-Pair Encoding (BPE)  
Iteratively merge highest-frequency symbol pairs:  
$$\text{freq}(a,b)=\max_{(a,b)}\ \text{count}(ab)$$  
until $|V|=V_{\max}$.  
$\tilde X = \operatorname{BPE}(X)$  

#### 1.2 Sequence Construction  
Add BOS/ EOS tokens: $\tilde X'=[\texttt{<bos>},\tilde x_1,\dots,\tilde x_T,\texttt{<eos>}]$  
Pad/trim to fixed $T_{\max}$.

---

## 2. Core Model (Autoregressive Transformer Decoder)

### 2.1 Token & Positional Embeddings  
$$E_t = W_E[\tilde x_t] \in\mathbb R^{d_\text{model}}$$  
$$P_t = W_P[t] \in\mathbb R^{d_\text{model}}$$  
$$h^{(0)}_t = E_t ⊕ P_t$$  

### 2.2 Single Decoder Layer $l\;(1\le l\le L)$  

#### 2.2.1 Causal Multi-Head Self-Attention  
For head $h$  
$$Q^{(l,h)}_t=W_Q^{(l,h)}h^{(l-1)}_t,\;
  K^{(l,h)}_t=W_K^{(l,h)}h^{(l-1)}_t,\;
  V^{(l,h)}_t=W_V^{(l,h)}h^{(l-1)}_t$$  

Attention scores (masked):  
$$\alpha^{(l,h)}_{t,j}=
  \begin{cases}
    \dfrac{Q^{(l,h)}_t\!\cdot\!K^{(l,h)}_j}{\sqrt{d_h}} &
    j\le t\\[4pt]
    -\infty & j>t
  \end{cases}$$  

$$A^{(l,h)}_{t,j} = \operatorname{softmax}_j(\alpha^{(l,h)}_{t,j})$$  
$$Z^{(l,h)}_t = \sum_{j\le t} A^{(l,h)}_{t,j} V^{(l,h)}_j$$  

Head concatenation & projection  
$$Z^{(l)}_t = W_O^{(l)}\bigl(Z^{(l,1)}_t∘\dots∘Z^{(l,H)}_t\bigr)$$  

Residual & norm  
$$\hat h^{(l)}_t = f_{\text{LN}}\bigl(h^{(l-1)}_t + Z^{(l)}_t\bigr)$$  

#### 2.2.2 Position-wise Feed-Forward  
$$u^{(l)}_t = σ\bigl(W_1^{(l)}\hat h^{(l)}_t + b_1^{(l)}\bigr)$$  
$$h^{(l)}_t = f_{\text{LN}}\bigl(\hat h^{(l)}_t + W_2^{(l)}u^{(l)}_t + b_2^{(l)}\bigr)$$  

### 2.3 Output Projection  
$$\hat y_t = W_\text{LM}\,h^{(L)}_t + b_\text{LM}$$  
$$p_\theta(x_t\mid x_{<t}) = \operatorname{softmax}(\hat y_t)$$  

---

## 3. Training

### 3.1 Objective (Token-level NLL)  
$$\mathcal L_{\text{NLL}}(\theta)
  = -\sum_{t=1}^{T}\log p_\theta(\tilde x_t\mid\tilde x_{<t})$$  

### 3.2 Regularisers  
• Weight decay: $λ\lVert\theta\rVert_2^2$  
• Dropout $p_d$ in attention/FFN.  
• Label-smoothing $ε$: replace one-hot with $(1-ε),\,ε/(V-1)$.

### 3.3 Optimizer (AdamW)  
$$m_k=β_1 m_{k-1}+(1-β_1)\nabla_\theta \mathcal L_k$$  
$$v_k=β_2 v_{k-1}+(1-β_2)(\nabla_\theta \mathcal L_k)^2$$  
$$\hat m_k=m_k/(1-β_1^k),\;\hat v_k=v_k/(1-β_2^k)$$  
$$\theta_{k+1}=\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+ϵ}-\eta λ \theta_k$$  

### 3.4 Curriculum & Mixed Precision (best practice)  
• Sequence length warm-up.  
• FP16/bfloat16 with loss-scaling: $\tilde{\mathcal L}=s·\mathcal L;$ back-prop, then divide gradients by $s$.

### 3.5 Distributed Data-Parallel (ZeRO-style)  
Partition parameters, grads, optimizer states across $N$ GPUs:  
$$\theta = \bigcup_{i=1}^{N} \theta^{(i)},\;
\nabla_\theta = \bigcup_{i=1}^{N} \nabla_{\theta^{(i)}}$$  
All-reduce for gradient-synchronisation each step.

### 3.6 Pseudo-Algorithm  
```
for epoch = 1 … E:
    for batch (X) in dataloader:
        X_ids = BPE(X)                       # pre-processing
        loss = NLL(X_ids; θ)                 # forward & loss
        loss.backward()                      # back-prop
        AdamW_step(θ,∇θ)                     # optimizer
        zero_grad()
```

---

## 4. Post-Training Procedures  

### 4.1 Instruction / RLHF Fine-Tuning  
1. Supervised Fine-Tune (SFT) on $(\text{prompt},\text{response})$.  
2. Reward Model $R_\phi$:  
   $$R_\phi(\text{prompt},\text{response}) \approx \text{human score}$$  
3. Proximal Policy Optimisation:  
   $$\max_\theta \; \mathbb E_{\pi_\theta}\bigl[R_\phi - β\log\frac{\pi_\theta}{\pi_{\text{SFT}}}\bigr]$$  
   PPO gradient:  
   $$\nabla_\theta J = \mathbb E\!\left[\nabla_\theta\log\pi_\theta
      \cdot\operatorname{clip}(A,1-ϵ,1+ϵ)\right]$$  

### 4.2 Parameter-Efficient Adapters (LoRA)  
Inject low-rank $\Delta W=BA^\top$ into $W$:  
$$W' = W + \alpha\frac{BA^\top}{r}$$  
Back-prop only through $A,B$.

---

## 5. Evaluation  

### 5.1 Intrinsic Metrics  
• Cross-Entropy (token level): $\mathcal L_{\text{NLL}}/T$  
• Perplexity: $PP= \exp\bigl(\mathcal L_{\text{NLL}}/T\bigr)$  

### 5.2 Benchmark Scores (SOTA references)  
• MMLU, TruthfulQA, GSM8K, DROP (exact-match/F1).  
Formal EM:  
$$\text{EM}= \frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\hat y_i = y_i]$$  

### 5.3 Hallucination / Factuality  
• FactScore: cosine similarity between generated triple embeddings & KG.  

### 5.4 Toxicity / Bias  
• Perspective API toxicity probability $\in[0,1]$.  
• Winogender bias gap:  
  $$\Delta = \frac{1}{|S|}\sum_{s\in S}\bigl(P_\theta(\text{male}\mid s)-P_\theta(\text{female}\mid s)\bigr)$$  

### 5.5 Human Eval  
Pairwise preference rate:  
$$\text{Pref}=\frac{1}{M}\sum_{j=1}^{M}\mathbf 1[
  R_\text{human}(y_j^{(A)})>R_\text{human}(y_j^{(B)})]$$  

---

## 6. Pros vs Cons  

Pros  
• Scales $$\mathcal O(LH d_\text{model}^2)$$ yet parallelisable.  
• Causal masking enables open-ended generation.  
• Pre-training on web scale yields strong zero-shot.  

Cons  
• Quadratic attention memory $$\mathcal O(T^2)$$.  
• Brittle to prompt attacks, data leakage.  
• High compute & carbon cost.  

---

## 7. Recent Advances / Best Practices  

• Attention sparsification ($\mathcal O(T\sqrt T)$).  
• RoPE positional encoding: replace $P_t$ with rotary mixing  
  $$R(\theta_t) = 
    \begin{bmatrix}\cos\theta_t&-\sin\theta_t\\\sin\theta_t&\cos\theta_t\end{bmatrix}$$  
• FlashAttention algorithm (I/O-aware tiling).  
• Q-LoRA & 4-bit NF4 quantisation for memory-efficient tuning.  

---

Potential Pitfalls  
• Tokenisation mismatch at evaluation → inflated perplexity.  
• Gradient underflow in FP16 without dynamic loss scaling.  
• Length extrapolation failures with absolute positions.