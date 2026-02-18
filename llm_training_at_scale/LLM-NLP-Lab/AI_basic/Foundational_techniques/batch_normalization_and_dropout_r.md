# 1 Batch Normalization (BN)

## 1.1 Definition  
Technique that normalizes intermediate activations per mini-batch, then rescales/offsets them with learned parameters to stabilize and accelerate deep-network training.

---

## 1.2 Pertinent Equations  

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i,\qquad
\sigma_B^{2}= \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^{2}+\epsilon}},\qquad
y_i=\gamma\hat{x}_i+\beta
$$

Variables  
$m$: mini-batch size.  
$x_i$: pre-activation value of sample $i$.  
$\mu_B,\sigma_B^{2}$: batch mean/variance.  
$\epsilon$: small constant for numerical stability.  
$\gamma,\beta$: learned scaling and shifting parameters.  
$y_i$: normalized output forwarded to next layer.

---

## 1.3 Key Principles  

* Internal covariate shift reduction via on-the-fly normalization.  
* Maintains representational power through learnable $\gamma,\beta$.  
* Adds gradient smoothing and implicit regularization.  
* Employs running estimates $(\mu_{\text{EMA}},\sigma_{\text{EMA}}^{2})$ for inference.

---

## 1.4 Detailed Concept Analysis  

* Gradient Flow: normalizing variance ≈ 1 mitigates exploding/vanishing gradients, enabling higher learning rates.  
* Back-prop Derivatives: gradients propagate through normalization; $\gamma$ scales gradient magnitude.  
* Placement: applied before non-linearity (original) or after (common in residual blocks).  
* Inference Phase: replace $\mu_B,\sigma_B^{2}$ with exponentially averaged statistics accumulated during training.  
* Interaction with Dropout: BN’s estimation noise conflicts with high Dropout; recommended Dropout after BN.

---

## 1.5 Importance & Typical Use Cases  

* Ubiquitous in CNNs, ResNets, GANs, and vision transformers.  
* Reduces training epochs ≈ 2–5× versus unnormalized networks.  
* Enables very deep (>1000-layer) architectures.

---

## 1.6 Pros vs Cons  

Pros  
- Faster convergence, higher peak accuracy.  
- Allows larger learning rates.  
- Acts as regularizer, sometimes obviating Dropout.

Cons  
- Depends on mini-batch statistics ⇒ degraded with small $m$ or online inference.  
- Adds computation and memory overhead.  
- Sequence/graph models with variable lengths experience mismatch.

---

## 1.7 Cutting-Edge Advances & Variants  

* Layer Norm, Instance Norm, Group Norm (batch-size independent).  
* Batch Renorm: corrects disparity between batch and running stats.  
* Ghost BN / Virtual BN: splits large batch into virtual sub-batches for multi-GPU sync savings.  
* RevNorm, EvoNorm: fuse BN with activation for efficiency.  
* Switchable Norm: learns convex mixture of BN/LN/IN statistics.

---

## 1.8 Pseudo-Algorithm  

```
Input: mini-batch {x1,…,xm}, parameters γ,β, running stats (μ_E,σ_E²), momentum α
Training:
  μ_B ← mean(x1…xm)
  σ_B² ← var(x1…xm)
  for each i:
      x̂_i ← (x_i − μ_B) / sqrt(σ_B² + ε)
      y_i ← γ · x̂_i + β
  μ_E ← (1−α)·μ_E + α·μ_B
  σ_E² ← (1−α)·σ_E² + α·σ_B²
  return {y_i}
Inference:
  for each i:
      x̂_i ← (x_i − μ_E) / sqrt(σ_E² + ε)
      y_i ← γ · x̂_i + β
```

---

## 1.9 Best Practices & Common Pitfalls  

Best Practices  
- Momentum α≈0.1; ε≈1e−5.  
- Keep mini-batch ≥32 per device or use Group/Layer Norm.  
- Freeze BN (eval mode) during fine-tuning with small datasets.

Pitfalls  
- Desynchronized stats across devices without all-reduce.  
- Using Dropout before BN increases variance.  
- Updating running stats with mixed precision without proper casting yields NaNs.

---

# 2 Dropout

## 2.1 Definition  
Regularization method that randomly sets a subset of neural activations to zero during training, preventing co-adaptation and reducing overfitting.

---

## 2.2 Pertinent Equations  

Training mask sampling:  
$$
m_i \sim \text{Bernoulli}(p),\qquad \tilde{y}_i = \frac{m_i}{p}\,x_i
$$

Inference scaling equivalence (“inverted Dropout”) ensures expectation preservation:  
$$
\mathbb{E}[\tilde{y}_i] = x_i
$$

Variables  
$p$: keep probability (e.g., 0.5).  
$m_i$: binary mask for unit $i$.  
$x_i$: activation before Dropout.  
$\tilde{y}_i$: scaled, masked activation passed forward.

---

## 2.3 Key Principles  

* Model averaging: samples an exponential ensemble of subnetworks.  
* Implicit $L_2$ regularization effect via noise injection in activations.  
* Independence assumption: masked units chosen independently each iteration.

---

## 2.4 Detailed Concept Analysis  

* Scaling: dividing by $p$ at train time equals multiplying by $p$ at test time (standard vs inverted).  
* Layer-wise Rates: lower $p$ (higher drop) in fully connected layers; often disabled ($p=1$) in BN-heavy CNNs.  
* Bayesian Interpretation: Monte Carlo Dropout approximates variational inference, enabling uncertainty estimation.  
* Gradient Flow: mask applied after activation; zeroed gradients for dropped units.  
* Structured Dropout Variants: drop entire channels (SpatialDropout), attention heads, tokens.

---

## 2.5 Importance & Typical Use Cases  

* Prevents overfitting in fully connected nets (MLPs, early CNNs).  
* Acts as cheap Bayesian approximation for uncertainty in vision/NLP.  
* Essential in low-data regimes and reinforcement learning exploration.

---

## 2.6 Pros vs Cons  

Pros  
- Simple, framework-agnostic.  
- Reduces test error without extra inference cost.  
- Adds robustness to input noise.

Cons  
- Slows convergence; requires more epochs.  
- Interferes with BN statistics.  
- Less effective in modern residual/attention architectures compared to data augmentation and weight decay.

---

## 2.7 Cutting-Edge Advances & Variants  

* DropConnect: randomly mask weights instead of activations.  
* R-Drop: symmetric KL regularization between two Dropout passes.  
* Concrete / Adaptive Dropout: learn $p$ via continuous relaxation.  
* Stochastic Depth: randomly drop whole residual blocks.  
* Attention Dropout, Token Dropout, PatchDrop for Transformers/ViT.  
* ERM inference tricks: test-time Dropout ensembling, TTA-MC Dropout.

---

## 2.8 Pseudo-Algorithm  

```
Input: activations {x1,…,xn}, keep probability p
Training:
  for each i:
      m_i ← Bernoulli(p)
      ŷ_i ← (m_i / p) · x_i
  return {ŷ_i}
Inference:
  return {x_i}   # masks disabled, no scaling needed (inverted formulation)
```

---

## 2.9 Best Practices & Common Pitfalls  

Best Practices  
- Use $p∈[0.8,0.95]$ in convolutional layers; $0.5$ in dense layers.  
- Combine with weight decay and data augmentation for synergistic regularization.  
- For BN-heavy nets, apply Dropout after BN or adopt Stochastic Depth.

Pitfalls  
- Applying Dropout during evaluation unintentionally inflates variance.  
- High drop rates (>0.7) can underfit, especially with small models.  
- Forgetting to scale activations leads to biased output distribution.

---