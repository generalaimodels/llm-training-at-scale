## Perceptron  

### 1. Definition  
$$\hat y=\operatorname{sign}(w^{\top}x+b)$$  

### 2. Pertinent Equations  
- Decision rule  
  $$\hat y=\begin{cases}+1& w^{\top}x+b\ge0\\-1& w^{\top}x+b<0\end{cases}$$  
- Update (online, learning-rate $\eta$)  
  $$w\leftarrow w+\eta\,(y_i-\hat y_i)\,x_i,\qquad b\leftarrow b+\eta\,(y_i-\hat y_i)$$  
- Perceptron loss  
  $$\mathcal L_{\text{perc}}=-\sum_{i:y_i(w^{\top}x_i+b)<0}y_i\,(w^{\top}x_i+b)$$  

### 3. Key Principles  
- Linear separability & margin  
- Mistake-driven online learning  
- Convergence bound: $$M\le\left(\tfrac{R}{\gamma}\right)^{2}$$  

### 4. Detailed Concept Analysis  
- Margin $m_i=y_i\frac{w^{\top}x_i+b}{\|w\|}$ governs robustness.  
- Bias trick: append $x_0=1$, absorb $b$ into $w_0$.  
- Kernelized variant replaces inner product with kernel $k(x,x')$.  

### 5. Importance  
- Historical root of neural networks, illustrates gradient-free online learning.  

### 6. Pros vs Cons  
- Pros: simplicity, fast updates, guaranteed convergence if separable.  
- Cons: fails on non-linearly separable data, sensitive to feature scaling.  

### 7. Cutting-Edge Advances  
- Averaged/K-best kernel perceptrons, adaptive margin variants, perceptron-SGD hybrids.  

### 8. Workflow  

#### 8.1 Pre-Processing  
$$\tilde x_j=\frac{x_j-\mu_j}{\sigma_j}$$  

#### 8.2 Training Algorithm  
```text
Input: {(x_i,y_i)}_{i=1}^N , η , epochs T
Initialize w←0
for t=1…T:
  for i=1…N:
     ŷ ← sign(wᵀx_i)
     if ŷ ≠ y_i:
        w ← w + η y_i x_i
```

#### 8.3 Post-Training  
- Weight normalization: $$w\leftarrow\frac{w}{\|w\|}$$  
- Platt scaling for calibrated probabilities: $$\sigma(a)=\frac1{1+\exp(Aa+B)}$$  

#### 8.4 Evaluation  

| Metric | Equation |
|--------|----------|
| Accuracy | $$\text{Acc}=\frac1N\sum_{i=1}^N\mathbf1(\hat y_i=y_i)$$ |
| Precision | $$\frac{\text{TP}}{\text{TP+FP}}$$ |
| Recall | $$\frac{\text{TP}}{\text{TP+FN}}$$ |
| F1 | $$2\frac{\text{Prec}\cdot\text{Rec}}{\text{Prec+Rec}}$$ |

---

## Multi-Layer Perceptron (MLP)  

### 1. Definition  
Layer $l$:  
$$a^{(l)}=W^{(l)}z^{(l-1)}+b^{(l)},\qquad z^{(l)}=\phi\!\bigl(a^{(l)}\bigr),\qquad z^{(0)}=x$$  
Output layer $L$: classification $\hat y=\operatorname{softmax}(a^{(L)})$, regression $\hat y=a^{(L)}$.  

### 2. Pertinent Equations  
- Cross-entropy loss  
  $$\mathcal L=-\sum_{i=1}^{N}\sum_{k}y_{i,k}\log\hat y_{i,k}$$  
- Backpropagation  
  $$\delta^{(L)}=\hat y-y,\qquad\delta^{(l)}=\bigl(W^{(l+1)}\bigr)^{\top}\delta^{(l+1)}\odot\phi'\!\bigl(a^{(l)}\bigr)$$  
  $$\nabla_{W^{(l)}}\mathcal L=\delta^{(l)}z^{(l-1)\top},\qquad\nabla_{b^{(l)}}\mathcal L=\delta^{(l)}$$  

### 3. Key Principles  
- Universal approximation via nonlinear activations.  
- End-to-end differentiability enabling gradient descent.  
- Depth-width trade-off, regularization, initialization.  

### 4. Detailed Concept Analysis  
- Activations: ReLU ($\max(0,a)$), GELU, tanh.  
- Initialization: He ($\sqrt{2/n_\text{in}}$) for ReLU; Xavier for tanh.  
- Regularizers: Dropout ($p$), $L_2$ weight decay ($\lambda\|W\|_2^2$).  
- Optimization: AdamW update  
  $$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\;v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$  
  $$\hat m_t=\frac{m_t}{1-\beta_1^t},\;\hat v_t=\frac{v_t}{1-\beta_2^t}$$  
  $$\theta_t=\theta_{t-1}-\eta\;\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_{t-1}$$  

### 5. Importance  
- Fundamental building block for vision, speech, language models pre-transformer.  
- Baseline for tabular and small-scale tasks.  

### 6. Pros vs Cons  
- Pros: high expressivity, arbitrary differentiable modules.  
- Cons: vanishing/exploding gradients, over-parameterization risk, data hunger.  

### 7. Cutting-Edge Advances  
- Residual/Highway connections, LayerNorm, Swish/GELU activations, sparsity-inducing pruning, lottery-ticket rewinding.  

### 8. Workflow  

#### 8.1 Pre-Processing  
- Standardization: $$\tilde x=\frac{x-\mu}{\sigma}$$  
- One-hot categorical encoding.  
- Data augmentation (domain-specific).  

#### 8.2 Training Algorithm (Mini-Batch AdamW)  
```text
Input: {(x_i,y_i)}, batch size B, epochs T, lr schedule η_t
Initialize {W^{(l)},b^{(l)}} with He/Xavier
for epoch=1…T:
  for each mini-batch:
     Forward: compute {a^{(l)}, z^{(l)}}
     Compute loss ℒ
     Backward: compute {δ^{(l)}, ∇W^{(l)}, ∇b^{(l)}}
     AdamW parameter update
```

#### 8.3 Post-Training  
- Pruning: mask weights where |W_{ij}|<τ, retrain.  
- Quantization: $$\hat W=\operatorname{round}(W\cdot2^{b-1})/2^{b-1}$$  
- Knowledge distillation: minimize $$\mathcal L=\alpha\mathcal L_\text{CE}(y,\hat y)+ (1-\alpha)T^2\mathcal L_\text{KL}(s_T,\hat y_T)$$  

#### 8.4 Evaluation  

| Task | Primary Loss | Core Metric | SOTA Benchmark Examples |
|------|--------------|-------------|-------------------------|
| Classification | Cross-entropy | $$\text{Top-1}=\frac1N\sum\mathbf1(\arg\max\hat y_i=y_i)$$ | ImageNet Top-1 88%+ |
| Regression | MSE $$\frac1N\sum\|y-\hat y\|^2$$ | $$\text{RMSE}=\sqrt{\text{MSE}}$$ | UCI datasets |
| NLP | CE | $$\text{BLEU}=\text{BP}\exp(\sum_n w_n\log p_n)$$ | WMT |
| Speech | CTC | $$\text{WER}=\frac{S+I+D}{N}$$ | LibriSpeech |