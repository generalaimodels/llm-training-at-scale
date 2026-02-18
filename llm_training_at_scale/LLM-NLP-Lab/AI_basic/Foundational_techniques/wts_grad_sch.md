# I. Data Pre-processing  
## 1. Input Standardization  
- Definition: zero-mean unit-variance scaling  
- Equation: $$\hat x_i = \frac{x_i - \mu}{\sigma},\quad \mu=\frac1N\sum_{i=1}^N x_i,\;\sigma^2=\frac1N\sum_{i=1}^N(x_i-\mu)^2$$  

# II. Model Construction  
## 1. Feedforward Layer  
- Equation: $$h^{(l)} = f\bigl(W^{(l)}h^{(l-1)} + b^{(l)}\bigr),\quad h^{(0)}=\hat x$$  
- Activation $f(\cdot)$: ReLU, tanh, etc.  

# III. Training Stage  
## A. Weight Initialization  
### Definition  
Randomly set $W^{(l)},b^{(l)}$ to break symmetry and preserve signal variance.  
### Equations  
- Xavier-Uniform: $$W^{(l)}_{ij}\sim U\!\bigl(-\sqrt{\tfrac{6}{n_{in}+n_{out}}},\sqrt{\tfrac{6}{n_{in}+n_{out}}}\bigr)$$  
- Kaiming-Normal: $$W^{(l)}_{ij}\sim\mathcal N\bigl(0,\tfrac{2}{n_{in}}\bigr)$$  
### Key Principles  
- Symmetry breaking  
- Variance preservation: $$\mathrm{Var}(h^{(l)})\approx\mathrm{Var}(h^{(l-1)})$$  
### Detailed Analysis  
- Too small $\Rightarrow$ vanishing gradients  
- Too large $\Rightarrow$ exploding activations  
### Importance  
- Faster convergence  
- Stable gradient flow  
### Pros vs Cons  
• Xavier: balanced for tanh.   – May underperform with ReLU  
• Kaiming: suited for ReLU.   – Assumes positive‐half activations  
### Cutting-edge Advances  
- LSUV: data‐dependent scaling  
- MetaInit: learns initialization  

## B. Gradient Clipping  
### Definition  
Limit gradient norm/value to avoid explosion.  
### Equations  
- Norm‐clipping: $$g \leftarrow g\cdot\min\!\Bigl(1,\frac{\tau}{\|g\|_2}\Bigr)$$  
- Value‐clipping: $$g_i\leftarrow\mathrm{clip}(g_i,-\tau,\tau)$$  
### Key Principles  
- Bound $\|\nabla_\theta L\|$  
- Preserve direction if below threshold  
### Detailed Analysis  
- Heavy‐tailed gradients in RNNs/deep nets  
- Threshold $\tau$ trade‐off  
### Importance  
- Prevent numerical overflow  
- Ensure stable updates  
### Pros vs Cons  
• Pros: simple, effective.   – Cons: may bias update magnitudes  
### Cutting-edge Advances  
- Adaptive clipping (per‐layer, per‐parameter)  
- NormAN: clipping based on noise‐scale estimates  

## C. Learning Rate Scheduling  
### Definition  
Vary $\eta_t$ over training to balance exploration/exploitation.  
### Equations  
- Step decay: $$\eta_t=\eta_0\cdot\gamma^{\lfloor t/s\rfloor}$$  
- Exponential: $$\eta_t=\eta_0\exp(-\kappa t)$$  
- Cosine annealing: $$\eta_t=\eta_{\min}+\tfrac{\eta_0-\eta_{\min}}2\Bigl(1+\cos\frac{t}{T}\pi\Bigr)$$  
- One-cycle: cyclical between $\eta_{\min},\eta_{\max}$  
### Key Principles  
- Large $\eta$→fast escape  
- Small $\eta$→fine convergence  
### Detailed Analysis  
- Warm-up avoids unstable start  
- Restarts improve local minima escape  
### Importance  
- Improves generalization  
- Speeds up convergence  
### Pros vs Cons  
• Pros: dynamic adaptation. – Cons: hyperparameter tuning  
### Cutting-edge Advances  
- RAdam/Lookahead: adaptive schedules  
- Hypergradient descent: learns $\eta_t$  

## D. Training Pseudo-Algorithm  
```
Input: data {(x_i,y_i)}_{i=1}^N, batch size B, epochs E
Initialize: W^{(l)},b^{(l)} via Kaiming‐Normal
for epoch=1…E:
  for each batch B_k:
    # 1. Forward
    compute h^{(l)} via h^{(l)}=f(W^{(l)}h^{(l-1)}+b^{(l)})
    # 2. Compute loss
    L = −(1/B)∑_{i∈B_k} y_i·log ŷ_i
    # 3. Backward
    g = ∇_{W,b}L
    # 4. Gradient clipping
    g ← g·min(1,τ/∥g∥_2)
    # 5. LR scheduling
    η ← schedule(epoch,iter)
    # 6. Parameter update
    W ← W − η·g_W ; b ← b − η·g_b
```

# IV. Post-Training Procedures  
- Temperature scaling: $$\hat p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)},\;T>0$$  
- Quantization/pruning with error‐bounds  

# V. Evaluation  
## 1. Loss Function  
- Cross-entropy: $$L=-\frac1N\sum_{i=1}^N\sum_{c}y_{i,c}\log\hat y_{i,c}$$  
## 2. Metrics  
- Accuracy: $$\mathrm{Acc}=\frac1N\sum_{i=1}^N\mathbf1(\hat y_i=y_i)$$  
- Precision: $$\mathrm{P}=\frac{TP}{TP+FP},\;\mathrm{R}=\frac{TP}{TP+FN}$$  
- F1: $$F1=2\frac{P\cdot R}{P+R}$$  
## 3. SOTA & Domain-Specific  
- mAP (detection): mean over IoU thresholds  
- BLEU (MT): $$\mathrm{BLEU}=BP\exp\bigl(\sum_n w_n\log p_n\bigr)$$  

# Best Practices & Pitfalls  
- Tune $\tau$ for clipping; avoid over‐clipping  
- Warm-up LR before decay  
- Match init to activation  
- Validate with multiple metrics for robustness