# Long Short-Term Memory (LSTM)

## 1 Definition  
Recurrent neural‐network (RNN) architecture with gating mechanisms that mitigates vanishing/exploding gradients, enabling long-range temporal credit assignment.

## 2 Pertinent Equations  

### 2.1 Single-Time-Step LSTM Cell  
For time index $t$, input $x_t\in\mathbb R^{d_x}$, previous hidden state $h_{t-1}\in\mathbb R^{d_h}$, previous cell state $c_{t-1}\in\mathbb R^{d_h}$:

$$
\begin{aligned}
\;i_t &= \sigma\!\big(W_i x_t + U_i h_{t-1} + b_i\big) \quad &\text{(input gate)}\\
f_t &= \sigma\!\big(W_f x_t + U_f h_{t-1} + b_f\big) \quad &\text{(forget gate)}\\
o_t &= \sigma\!\big(W_o x_t + U_o h_{t-1} + b_o\big) \quad &\text{(output gate)}\\
\tilde{c}_t &= \tanh\!\big(W_c x_t + U_c h_{t-1} + b_c\big) &\text{(cell proposal)}\\[4pt]
c_t &= f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t &\text{(new cell state)}\\
h_t &= o_t \odot \tanh(c_t) &\text{(new hidden state)}
\end{aligned}
$$  
where $\sigma$ = logistic sigmoid, $\odot$ = element-wise product.

### 2.2 Parameter Count  
For hidden size $d_h$, input dim $d_x$:  
$$
\text{Params} = 4\big[(d_x + d_h)d_h + d_h\big]
$$

## 3 Key Principles  
• Gating controls information flow ➜ alleviates gradient decay.  
• Additive memory path ($c$) preserves linear derivative.  
• Shared parameters across time ➜ temporal generalization.

## 4 Detailed Concept Analysis  

### 4.1 Multi-Layer / Bidirectional  
Stack $L$ layers; output of layer $\ell$ feeds layer $\ell{+}1$. Bidirectional variant concatenates forward $\overrightarrow{h_t}$ and backward $\overleftarrow{h_t}$.

### 4.2 Peephole Variant  
Adds cell-to-gate connections: e.g.\ $i_t = \sigma(W_i x_t + U_i h_{t-1} + P_i c_{t-1}+b_i)$.

### 4.3 Projection Head (LSTMP)  
Compute $h_t$ then linearly project to lower dim $p$:  
$$h^{(p)}_t=V h_t,\; V\in\mathbb R^{p\times d_h}$$.

## 5 Importance  
Dominant in speech, language, time-series before Transformers; still valuable for low-latency/mobile settings.

## 6 Pros vs Cons  
Pros:  
• Handles long contexts; • Stable gradients; • Parameter-efficient vs attention.  
Cons:  
• Sequential dependency hampers parallelism; • Memory heavier than GRU; • Inferior to attention on very long sequences.

## 7 Cutting-Edge Advances  
• Inductive bias injection (e.g.\ Conv-LSTM, Chem-LSTM).  
• Low-rank & block-circulant weight factorization for edge deploy.  
• Hybrid Transformer-LSTM encoders.  
• Differentiable neuro-symbolic controllers add external memories.

---

# End-to-End Workflow

## 8 Data Pre-Processing

| Step | Math | Note |
|------|------|------|
| Padding/Masking | create mask $m_{t}=1_{t\leq T_i}$ | keep BPTT stable |
| Normalization | $x_t'=\dfrac{x_t-\mu}{\sigma}$ | per-feature z-score |
| Tokenisation (NLP) | map token $\tau$→index $k$ | Vocab size $V$ |
| Embedding | $e_t = E[k],\;E\in\mathbb R^{V\times d_x}$ | trainable |

Best practice: batch sequences of similar length to minimize padding.

## 9 Training

### 9.1 Loss Functions  
• Cross-Entropy: $$\mathcal L_{\text{CE}}= -\sum_{t} y_t^\top\log \hat y_t$$  
• CTC (speech): $$\mathcal L_{\text{CTC}}=-\log p(y|x)$$  
• MSE (regression): $$\mathcal L_{\text{MSE}}=\frac1T\sum_{t}\lVert y_t-\hat y_t\rVert_2^2$$

### 9.2 BPTT Derivatives  
Gradients flow via chain rule through $c_t$:  
$$
\frac{\partial \mathcal L}{\partial c_t} = \frac{\partial \mathcal L}{\partial h_t}\odot o_t\odot(1-\tanh^2(c_t)) \;+\; \frac{\partial \mathcal L}{\partial c_{t+1}}\odot f_{t+1}
$$  
maintains additive term, circumventing vanishing.

### 9.3 PyTorch-Style Training Loop (pseudo)

```python
for epoch in range(E):
    for X, Y in loader:                # minibatch
        h = c = zeros(num_layers, B, H, device)
        optimizer.zero_grad()
        for t in range(seq_len):
            i = sigmoid(Wi@X[t] + Ui@h + bi)
            f = sigmoid(Wf@X[t] + Uf@h + bf)
            o = sigmoid(Wo@X[t] + Uo@h + bo)
            ct~ = tanh(Wc@X[t] + Uc@h + bc)
            c = f*c + i*ct~
            h = o*tanh(c)
            logits = W_out@h
            loss_t = criterion(logits, Y[t])
            loss_t.backward(retain_graph=True)  # truncated BPTT
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
```
Mathematically justified by applying stochastic gradient descent on $$\hat\theta = \arg\min_\theta \mathbb E_{(x,y)}[\mathcal L(y,f_\theta(x))]$$.

### 9.4 Hyperparameters  
• Hidden size $d_h$: 256-1024; • Layers: 2-4; • Dropout $p=0.1$-0.3; • Optimizer: AdamW, $$\eta=10^{-3}$$, weight decay $10^{-2}$.

### 9.5 Regularisation  
• Dropout on $h_t$ ($p$).  
• Recurrent dropout (no time correlation).  
• LayerNorm on gates: $g_t = \sigma(\text{LN}( \cdot ))$.

## 10 Post-Training

### 10.1 Quantisation  
Uniform-affine: $$\hat w = \text{clip}\Big(\big\lfloor\frac{w}{\Delta}\big\rceil + z\Big, 0, 2^b-1\Big)\Delta$$  
with scale $\Delta=\frac{\max w - \min w}{2^b-1}$.

### 10.2 Pruning  
Magnitude mask $M=\mathbb I(|w|>\tau)$, new weight $w'=w\odot M$.

### 10.3 Knowledge Distillation  
Teacher logits $z_t^{(T)}$, student $z_t^{(S)}$:  
$$
\mathcal L_{\text{KD}} = \alpha\,\mathcal L_{\text{CE}} + (1-\alpha)T^2\,\text{KL}\big(\text{softmax}(z^{(T)}/T)\,\|\,\text{softmax}(z^{(S)}/T)\big)
$$

## 11 Evaluation

### 11.1 Sequence Classification  
Accuracy $$\text{Acc}= \frac{1}{N}\sum_{i=1}^{N}\mathbb I(\hat y_i=y_i)$$  
F1 $$\text{F1}=2\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$$

### 11.2 Language Modelling  
Perplexity $$\text{PPL}= \exp\!\Big(\frac1T \sum_{t}\mathcal L_{\text{CE},t}\Big)$$

### 11.3 Speech / OCR (CTC)  
Word Error Rate (WER): $$\text{WER}= \frac{S+D+I}{N}$$

### 11.4 Regression  
RMSE $$\text{RMSE} = \sqrt{\frac1T\sum_{t}(y_t-\hat y_t)^2}$$

State-of-the-art benchmarks: GLUE (NLP), Librispeech (speech), M4 (time-series).

## 12 Best Practices & Pitfalls  
• Clip gradients $<5$ to avoid explosion.  
• Use LayerNorm for >2 layers to stabilize.  
• Truncated BPTT length 100–200 steps balances compute vs memory.  
• Watch out for overfitting on small datasets—apply dropout & early stopping.  
• Batch-first tensors improve cuDNN kernel utilisation (`batch_first=True` in PyTorch).