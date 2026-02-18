## 1. Definition  
Recurrent Neural Networks (RNNs) are parameter-shared neural architectures that model variable-length sequences by recursively updating hidden states $h_t$ using current input $x_t$ and previous state $h_{t-1}$.

---

## 2. Pertinent Equations  

### 2.1 Core Recurrence  
$$h_t = \phi\big(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\big)$$  
$$\hat{y}_t = g\big(W_{hy}\,h_t + b_y\big)$$  

### 2.2 Parameter Count (Elman RNN)  
$$\#\text{params}=d_{\text{in}}\,d_h+ d_h^2 + d_h + d_h\,d_{\text{out}} + d_{\text{out}}$$  
where  
$d_{\text{in}}$ = input size, $d_h$ = hidden size, $d_{\text{out}}$ = output size.

### 2.3 Loss (token-level classification)  
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \sum_{c=1}^{C} y_{t,c}\,\log \hat{y}_{t,c}$$  

### 2.4 Back-Propagation Through Time (BPTT)  
$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\,h_{t-1}^\top,\qquad
\delta_t = \Bigg(\frac{\partial \mathcal{L}}{\partial h_t} + W_{hh}^\top\,\delta_{t+1}\Bigg)\odot\phi'(a_t)$$  
$a_t = W_{xh}x_t + W_{hh}h_{t-1}+b_h$  

---

## 3. Key Principles  

* Temporal weight sharing enables processing of arbitrary-length sequences.  
* Hidden state serves as dynamic memory.  
* BPTT unfolds the graph for $T$ steps; gradients accumulate across time.  
* Vanishing/exploding gradients stem from repeated Jacobian multiplication $$\prod_{k=1}^{T} \frac{\partial h_k}{\partial h_{k-1}}.$$

---

## 4. Detailed Concept Analysis  

### 4.1 Data Pre-Processing  
* Tokenisation → integer indices.  
* Embedding lookup: $$x_t = E\,i_t,\;E\in\mathbb{R}^{V\times d_{\text{in}}}.$$  
* Optional normalisation: $$x_t \leftarrow \frac{x_t-\mu}{\sigma}.$$

### 4.2 Model Components  

| Component | Equation | Shape |
|-----------|----------|-------|
| Input–hidden | $W_{xh}$ | $d_{\text{in}}\times d_h$ |
| Hidden–hidden | $W_{hh}$ | $d_h\times d_h$ |
| Hidden bias | $b_h$ | $d_h$ |
| Hidden–output | $W_{hy}$ | $d_h\times d_{\text{out}}$ |
| Output bias | $b_y$ | $d_{\text{out}}$ |

Activation choices: $\phi$ ∈ {$\tanh$, $\text{ReLU}$, $\text{GELU}$}; $g$ ∈ {$\text{softmax}$, $\text{sigmoid}$}.

### 4.3 Training Pipeline (PyTorch-style Pseudo-Algorithm)  

```python
# Pseudo-code (BPTT, teacher forcing)
for epoch in range(E):
    for X, Y in dataloader:                      # X: [B, T], Y: [B, T]
        h = torch.zeros(B, d_h)                  # init hidden
        loss = 0
        for t in range(T):
            x_t = embedding(X[:, t])             # [B, d_in]
            h = phi(x_t @ W_xh + h @ W_hh + b_h) # recurrence
            y_hat = softmax(h @ W_hy + b_y)      # [B, C]
            loss += criterion(y_hat, Y[:, t])
        loss /= T
        loss.backward()                          # BPTT
        clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
        optimizer.step(); optimizer.zero_grad()
```

Mathematical justification: gradient clipping caps $$\Vert \nabla\mathcal{L} \Vert_2$$ to mitigate exploding gradients.

### 4.4 Post-Training Procedures  
* Weight quantisation: $$W' = \operatorname{round}\big(\frac{W}{\Delta}\big)\Delta$$.  
* Pruning: magnitude-based mask $M = \mathbb{1}_{|W|>\tau}$ then $W\leftarrow W\odot M$.  
* Knowledge distillation: teacher logits $z_t$ soften targets $$\mathcal{L}_{\text{KD}} = \text{KL}\big(\sigma(z_t/T)\,\Vert\,\sigma(\hat{z}_t/T)\big).$$  

---

## 5. Importance  

* Captures temporal dependencies in speech, text, and time-series.  
* Parameter efficiency via weight sharing.  
* Foundation for advanced gated models (LSTM, GRU) and attention mechanisms.

---

## 6. Pros vs Cons  

Pros  
• Handles variable-length input/outputs.  
• Expressive with modest parameter counts.  
• Online/streaming inference feasible.  

Cons  
• Vanishing/exploding gradients hamper long-term dependency learning.  
• Sequential processing limits parallelism.  
• Training instability without careful initialisation and clipping.

---

## 7. Cutting-Edge Advances  

* Gated architectures (LSTM/GRU) mitigate gradient issues by additive memory paths.  
* Nanoscale RNNs via low-rank factorisation $$W_{hh}=U\,V^\top$$ reducing parameters to $d_h(r+ d_h)$ with rank $r\!\ll\!d_h$.  
* Recurrent depthwise separable RNNs improve FLOPs.  
* Recurrent Highway/IndRNN: element-wise recurrent weight $$h_t = \phi\big(W_{xh}x_t + u\odot h_{t-1}\big)$$ easing gradient flow.  
* Combining RNNs with self-attention for hybrid temporal-context capture.  

---

## 8. Evaluation Metrics  

| Task | Metric | Formal Definition |
|------|--------|-------------------|
| Language Modeling | Perplexity | $$\text{PPL}= \exp\Big(\frac{1}{N}\sum_{i=1}^{N}-\log p(w_i)\Big)$$ |
| Sequence Classification | Accuracy | $$\text{Acc}= \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}_{\hat{y}_i=y_i}$$ |
| Seq2Seq | BLEU-4 | $$\text{BLEU}= \text{BP}\,\exp\Big(\sum_{n=1}^{4} w_n \log p_n\Big)$$ |
| Time-Series Forecast | MAE | $$\text{MAE}= \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|$$ |

Best practices: pad/pack sequences for batch efficiency; detach hidden states between mini-batches; schedule sampling to bridge training–inference gap; monitor gradient norms for debugging.


### TRAINING & EVALUATION PSEUDO-ALGORITHMS  

#### 1. Notation  
* $B$ – batch size | $T$ – sequence length | $d_{\text{in}},d_h,d_{\text{out}}$ – dims  
* $E$ – embedding matrix | $W_{xh},W_{hh},b_h,W_{hy},b_y$ – learnable params  
* $\phi$ – hidden activation ($\tanh$/ReLU) | $g$ – output activation (softmax/sigmoid)  
* $\eta$ – learning rate | $\lambda$ – gradient-clip threshold  

---

#### 2. End-to-End Training (BPTT, teacher forcing)  

```python
# ──────────────────────────────────────────────────────────────
# PRE-PROCESS
tokens  = tokenizer(raw_text)                     # str → ints
dataset = pad_pack(tokens, pad_id=0)              # pad/pack to (B,T)

# ──────────────────────────────────────────────────────────────
# INITIALISE
E        = nn.Embedding(V, d_in)
W_xh     = nn.Parameter(torch.randn(d_in, d_h) * k)
W_hh     = nn.Parameter(torch.randn(d_h,  d_h) * k)
b_h      = nn.Parameter(torch.zeros(d_h))
W_hy     = nn.Parameter(torch.randn(d_h, d_out) * k)
b_y      = nn.Parameter(torch.zeros(d_out))
optim    = torch.optim.Adam(params, lr=η)
criterion= nn.CrossEntropyLoss(reduction="mean")

# ──────────────────────────────────────────────────────────────
# TRAIN LOOP
for epoch in range(num_epochs):
    for X, Y in dataloader:                       # X,Y ⇒ [B,T]
        h = torch.zeros(B, d_h)                   # h₀
        loss = 0
        for t in range(T):                        # ── forward pass
            x_t = E(X[:, t])                      # x_t = E i_t
            a_t = x_t @ W_xh + h @ W_hh + b_h     # affine
            h    = φ(a_t)                         # hidden update
            ŷ_t = g(h @ W_hy + b_y)              # logits→probs
            loss += criterion(ŷ_t, Y[:, t])      # accumulate
        loss /= T                                 # mean over time

        optim.zero_grad()
        loss.backward()                           # ── BPTT
        torch.nn.utils.clip_grad_norm_(           # optional
            params, max_norm=λ, norm_type=2)
        optim.step()                              # params ← params − η∇
```

Mathematical justification  
• Forward: $h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$; $ŷ_t=g(W_{hy}h_t+b_y)$  
• Loss: $$\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T}\ell(ŷ_t, y_t)$$  
• BPTT: $$\nabla_{W_{hh}}\mathcal{L}= \sum_{t}\delta_t h_{t-1}^\top,$$ $$\delta_t=(\nabla_{h_t}\mathcal{L}+W_{hh}^\top\delta_{t+1})\odot\phi'(a_t)$$  
• Gradient clipping: $$\nabla\mathcal{L} \gets \nabla\mathcal{L}\,\min\!\Bigl(1,\frac{\lambda}{\|\nabla\mathcal{L}\|_2}\Bigr)$$  

---

#### 3. Inference / Evaluation  

```python
def evaluate(model, X):
    h = torch.zeros(1, d_h)
    logprob = 0
    for t in range(T):
        x_t = E(X[:, t])
        h   = φ(x_t @ W_xh + h @ W_hh + b_h)
        ŷ_t= g(h @ W_hy + b_y)
        logprob += torch.log(ŷ_t)[0, Y[:, t]]
    ppl = torch.exp(-logprob / T)                 # if LM task
    return ppl
```

Metrics examples  
• Perplexity: $$\text{PPL}= \exp\Big(-\frac{1}{T}\sum_{t=1}^T\log p(w_t)\Big)$$  
• Accuracy: $$\text{Acc}= \frac{1}{N}\sum_{i}\mathbb{1}_{\hat{y}_i=y_i}$$  

---

#### 4. Parameter Count (Elman RNN)  
$$\bigl|W_{xh}\bigr|=d_{\text{in}}d_h,\;
  \bigl|W_{hh}\bigr|=d_h^2,\;
  \bigl|W_{hy}\bigr|=d_hd_{\text{out}},\;
  \bigl|b_h\bigr|=d_h,\;
  \bigl|b_y\bigr|=d_{\text{out}}$$  
Total $$\#\text{params}=d_{\text{in}}d_h+d_h^2+d_h+d_hd_{\text{out}}+d_{\text{out}}$$  

---

Best practices  
* Detach hidden state between batches: `h = h.detach()` to stop gradient leakage.  
* Use `pack_padded_sequence` for variable-length batching.  
* Seed and determinism flags for reproducibility (`torch.manual_seed`).