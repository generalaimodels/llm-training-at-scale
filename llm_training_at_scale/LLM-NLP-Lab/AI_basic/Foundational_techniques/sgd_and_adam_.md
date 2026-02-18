# 1. Stochastic Gradient Descent (SGD)

## 1.1 Definition  
Iterative first-order optimization method that updates parameters using noisy gradients computed on single examples or small mini-batches.

---

## 1.2 Core Equations  

- Loss (empirical risk):  
  $$\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell\bigl(f_\theta(x_i),y_i\bigr)$$  
- SGD update (single sample $i_t$ at step $t$):  
  $$\theta_{t+1}=\theta_t-\eta_t\,g_t,\quad g_t=\nabla_\theta\ell\bigl(f_\theta(x_{i_t}),y_{i_t}\bigr)$$  
- Mini-batch size $B$:  
  $$g_t=\frac{1}{B}\sum_{j=1}^{B}\nabla_\theta\ell\bigl(f_\theta(x_{j}),y_{j}\bigr)$$  

Variables:  
$N$ – dataset size; $x_i,y_i$ – data/label; $\ell$ – per-sample loss; $f_\theta$ – model; $\theta$ – parameters; $\eta_t$ – learning-rate (step size); $g_t$ – stochastic gradient.

---

## 1.3 Key Principles  

- Unbiased gradient estimate: $\mathbb{E}[g_t]=\nabla_\theta \mathcal{L}(\theta_t)$  
- Variance proportional to mini-batch size ($\mathcal{O}(1/B)$).  
- Convergence under Robbins-Monro conditions: $\sum_t\eta_t=\infty$, $\sum_t\eta_t^2<\infty$.  
- Noise acts as implicit regularizer, aiding generalization.

---

## 1.4 Detailed Concept Analysis  

- Learning-rate scheduling: constant, step decay, exponential, cosine, warm restarts, polynomial.  
- Momentum augmentation: $$v_{t+1}=\mu v_t+\eta_t g_t;\;\theta_{t+1}=\theta_t-v_{t+1}$$ ($\mu$ friction).  
- Nesterov accelerated gradient (NAG): look-ahead gradient evaluation.  
- Implicit bias: SGD tends toward flat minima with lower generalization error.  
- Scaling laws: optimal $\eta_t\propto\sqrt{B}$ up to critical batch size.

---

## 1.5 Significance & Use Cases  

- De-facto standard for deep learning training (CNNs, RNNs, Transformers).  
- Suited for large-scale, streaming, or online data.  
- Foundation for advanced optimizers (Adam, LAMB, Adagrad).

---

## 1.6 Advantages vs. Disadvantages  

Pros  
- Memory-efficient (store single/mini-batch).  
- Fast initial progress, anytime-usable.  
- Robust to very large datasets.

Cons  
- Sensitive to $\eta_t$; hand-tuning required.  
- Slow convergence near minima (high variance).  
- Poor handling of ill-conditioned curvature without momentum/adaptive methods.

---

## 1.7 Cutting-Edge Advances  

- SGD with adaptive batch sizes (ABS, GradMatch).  
- Variance-reduced SGD (SVRG, SAGA, SARAH).  
- Lookahead SGD; Sharpness-Aware Minimization (SAM) layering.  
- SGD in federated and decentralized settings.

---

## 1.8 Pseudo-Algorithm  

```
Input: θ0, learning-rate schedule {ηt}, batch size B
Initialize t ← 0
repeat
    Sample mini-batch 𝔅t ⊂ {1…N}, |𝔅t|=B
    Compute gradient: gt ← (1/B) Σ_{i∈𝔅t} ∇θ ℓ(fθt(xi), yi)
    θt+1 ← θt − ηt · gt
    t ← t + 1
until convergence criterion satisfied
return θt
```

Justification: unbiased gradient estimate ensures convergence; step size shrinks noise; batching amortizes compute.

---

## 1.9 Best Practices / Pitfalls  

Best Practices  
- Start with $\eta_0$ from linear scaling rule: $\eta_0=η_{\text{ref}}\cdot B/B_{\text{ref}}$.  
- Combine with momentum (0.9-0.99).  
- Employ warm-up for first 5-10 epochs.  
- Use gradient clipping for RNNs.

Pitfalls  
- Too large $\eta_t$ ⇒ divergence.  
- Small batch ⇒ noisy estimates, unstable with BN.  
- Inconsistent data shuffling reduces stochasticity.  

---

# 2. Adam (Adaptive Moment Estimation)

## 2.1 Definition  
First-order optimizer that adaptively rescales learning rates per parameter using exponential moving averages of both first and second gradient moments.

---

## 2.2 Core Equations  

At step $t$ with gradient $g_t$:

- First-moment estimate: $$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$$  
- Second-moment estimate: $$v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$  
- Bias correction:  
  $$\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad\hat{v}_t=\frac{v_t}{1-\beta_2^t}$$  
- Parameter update:  
  $$\theta_{t+1}=\theta_t-\eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$  

Variables:  
$\beta_1,\beta_2$ – decay rates ($≈0.9,0.999$); $\epsilon$ – numerical stability ($≈10^{-8}$).

---

## 2.3 Key Principles  

- Adaptive scaling: element-wise step size $\eta/\sqrt{\hat{v}_t}$.  
- Incorporates momentum via first moment.  
- Bias correction removes initialization bias for small $t$.  
- Has implicit pre-conditioning akin to RMSProp + momentum.

---

## 2.4 Detailed Concept Analysis  

- Second moment tracks uncentered variance $\mathbb{E}[g^2]$.  
- Effective learning rate diminishes for dimensions with large historical variance, amplifies for rarely updated weights.  
- Stationary point condition: under certain assumptions, Adam behaves like AdaGrad long-term.  
- Convergence issues: original Adam may fail in convex settings (non-monotone $\eta_t$). Fixed by AMSGrad.

---

## 2.5 Significance & Use Cases  

- Default optimizer for NLP (Transformers, BERT/GPT), GANs, VAE.  
- Superior on sparse or highly non-stationary gradients.  
- Facilitates rapid prototyping with minimal tuning.

---

## 2.6 Advantages vs. Disadvantages  

Pros  
- Minimal hyper-parameter tuning; robust default settings.  
- Fast convergence, fewer epochs to good minima.  
- Handles sparse features (natural language, recommender systems).

Cons  
- Larger memory footprint (2× parameters).  
- Can converge to sharper minima, weaker generalization.  
- Learning-rate decay still needed for final convergence.  

---

## 2.7 Cutting-Edge Advances / Variants  

- AMSGrad: $$v_t'=\max(v_{t-1}',v_t); \theta_{t+1}=\theta_t-\eta\,\hat{m}_t/(\sqrt{v_t'}+\epsilon)$$ (ensures non-increasing step size).  
- AdamW: decoupled weight decay, update $\theta←\theta-\eta(\hat{m}/\sqrt{\hat{v}})-\eta\lambda\theta$.  
- AdaBelief: replaces $v_t$ with squared deviation $(g_t-m_t)^2$.  
- Lion: sign-based variant with two momentum buffers.  
- LAMB: layer-wise adaptive large-batch Adam for billion-scale models.  
- D-Adaptation: scale-free adaptive learning.

---

## 2.8 Pseudo-Algorithm  

```
Input: θ0, η, β1, β2, ε
Initialize m0 ← 0, v0 ← 0, t ← 0
repeat
    t ← t + 1
    Sample mini-batch 𝔅t
    gt ← (1/|𝔅t|) Σ_{i∈𝔅t} ∇θ ℓ(fθ(xi), yi)
    mt ← β1·mt−1 + (1−β1)·gt
    vt ← β2·vt−1 + (1−β2)·gt⊙gt
    m̂t ← mt / (1−β1^t)
    v̂t ← vt / (1−β2^t)
    θt ← θt − η · m̂t / (√v̂t + ε)
until convergence
return θt
```

Justification: Exponential averaging provides smooth estimates; bias correction restores unbiasedness; element-wise division adapts learning rate.

---

## 2.9 Best Practices / Common Pitfalls  

Best Practices  
- Default $(\beta_1,\beta_2)=(0.9,0.999)$; tune β1 down (0.8) for very noisy gradients.  
- Use warm-up then cosine/linear decay of η.  
- Pair with AdamW (decoupled weight decay) for better generalization.  
- Gradient clipping (global norm) for stability in seq2seq/GPT training.

Pitfalls  
- High β2→ sluggish adaptation to sudden gradient scale changes.  
- Large ε hides poor β2 choice (over-smooth).  
- Forgetting weight decay coupling in vanilla Adam deteriorates performance.  
- Relying solely on training loss; validate generalization to detect over-fitting due to flat-minima deficiency.

---
