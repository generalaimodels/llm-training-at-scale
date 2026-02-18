# Generative Adversarial Networks
### 1. Definition  
Generative Adversarial Networks ($\text{GANs}$) are two-player minimax frameworks where a generator $G$ and a discriminator $D$ engage in a zero-sum game to model a target data distribution $p_{\text{data}}(x)$.

---

### 2. Core Architecture & Mathematical Formulation  

#### 2.1 Generator ($G$)  
* Mapping: $$G:\; \mathbb{R}^{d_z}\!\to\! \mathbb{R}^{d_x}, \quad x\!=\!G(z;\,\theta_G)$$  
* Input: latent noise $z\!\sim\!p_z(z)$ (e.g., $\mathcal{N}(0,I)$ or $\text{U}[-1,1]$).  
* Output: synthetic sample $x$.  
* Common layer stack (image domain):  
  1. Dense $d_z\!\to\!4\!\times\!4\!\times\!f$  
  2. Reshape $\rightarrow$ 4×4 feature map  
  3. Sequence of $\text{Conv2DTranspose} \!\rightarrow\!$ BatchNorm $\rightarrow$ ReLU  
  4. Final $\text{Conv2DTranspose}$ with $\tanh$.

#### 2.2 Discriminator ($D$)  
* Mapping: $$D:\; \mathbb{R}^{d_x}\!\to\![0,1], \quad D(x;\,\theta_D)=\sigma(f(x))$$  
* Objective: estimate probability that $x$ came from $p_{\text{data}}$.  
* Common layer stack (image domain):  
  1. $\text{Conv2D}$ $\rightarrow$ LeakyReLU ($\alpha\!=\!0.2$)  
  2. SpectralNorm (stabilization)  
  3. Strided Conv blocks (down-sampling)  
  4. Dense $\rightarrow$ $\sigma$.

#### 2.3 Adversarial Objective  
Standard (Jensen–Shannon) loss:  
$$\min_{\theta_G}\max_{\theta_D} \; V(D,G)=\mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big]+\mathbb{E}_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]$$

Alternative formulations:  
* Non-saturating generator loss: $$\max_{\theta_G}\mathbb{E}_{z\sim p_z}[\log D(G(z))]$$  
* WGAN (Earth-Mover):  
  $$\min_{\theta_G}\max_{\theta_D\in\mathcal{D}_1}\; \mathbb{E}_{x\sim p_{\text{data}}}[D(x)]-\mathbb{E}_{z\sim p_z}[D(G(z))]$$  
  where $\mathcal{D}_1$ = 1-Lipschitz functions (enforced via weight-clipping, gradient penalty, or spectral norm).

---

### 3. Data Pre-processing  

| Step | Mathematical Expression | Purpose |
|------|-------------------------|---------|
| Scaling | $$x \leftarrow 2\frac{x-\min}{\max-\min}-1$$ | Center to $[-1,1]$ for $\tanh$ output. |
| Whitening (optional) | $$\tilde{x}= \Sigma^{-\frac{1}{2}}(x-\mu)$$ | Remove global correlations. |
| Augmentation | $$x' = T_\phi(x)$$ (random $T_\phi$) | Improves diversity and regularization.|

---

### 4. Training Pseudo-Algorithm (1-GPU, mini-batch)  

```text
Input: dataset X, epochs E, batch size m, learning rates η_D, η_G
Init θ_D, θ_G
for epoch = 1 … E:
    for each mini-batch {x_i}⁽ᵐ⁾ ⊂ X:
        # --- Discriminator step k times ---
        for t = 1 … k:
            z ← p_z(z)  ⟹  {z_j}⁽ᵐ⁾
            g ← G(z; θ_G)
            L_D ← -[ (1/m) Σ log D(x_i) + (1/m) Σ log(1-D(g_j)) ]   # JS
            θ_D ← θ_D - η_D ∇_{θ_D} L_D
            if WGAN-GP:
                 gp ← λ (||∇_{\hat{x}} D(\hat{x})||_2-1)²,  \hat{x} ← εx+(1-ε)g
                 L_D ← -[ ... ] + gp
        # --- Generator step ---
        z ← p_z(z)  ⟹  {z_j}⁽ᵐ⁾
        g ← G(z; θ_G)
        L_G ← -(1/m) Σ log D(g_j)               # non-saturating
        θ_G ← θ_G - η_G ∇_{θ_G} L_G
```
All gradients obtained via automatic differentiation; optimizers commonly Adam with $$\beta_1=0.5,\;\beta_2=0.999$$.

---

### 5. Post-Training Procedures  

1. Model Selection  
   * Pick $\theta_G^\*$ maximizing evaluation metric (e.g., lowest $\text{FID}$).  
2. Exponential Moving Average (EMA) of weights:  
   $$\theta_G^{\text{EMA}}\leftarrow \alpha\theta_G^{\text{EMA}}+(1-\alpha)\theta_G$$  
   stabilizes sample quality.  
3. Latent Space Traversal & Interpolation  
   * Linear: $$x(\lambda)=G\big((1-\lambda)z_1+\lambda z_2\big)$$  
   * Spherical: $$z(\lambda)=\frac{\sin((1-\lambda)\omega)}{\sin\omega}z_1+\frac{\sin(\lambda\omega)}{\sin\omega}z_2$$ with $$\omega=\arccos\frac{z_1^\top z_2}{\|z_1\|\|z_2\|}$$  
4. Model Compression (optional): pruning, knowledge distillation using feature matching loss  
   $$L_{\text{FM}}=\|f_D(x)-f_D(G(z))\|_2^2$$

---

### 6. Evaluation Metrics  

| Metric | Formal Definition | Notes |
|--------|-------------------|-------|
| Inception Score (IS) | $\exp\Big(\mathbb{E}_{x\sim p_G}\big[\text{KL}(p(y|x)\,\|\,p(y))\big]\Big)$ | Uses pretrained Inception; higher better. |
| Fréchet Inception Distance (FID) | $$\text{FID}= \|\mu_r-\mu_g\|_2^2 + \text{Tr}\!\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{\frac12}\big)$$ | Compares Gaussian fits of features; lower better. |
| Kernel Inception Distance (KID) | $$\text{MMD}^2_u=\frac{1}{m(m-1)}\sum_{i\ne j} k(x_i,x_j) + \frac{1}{n(n-1)}\sum_{i\ne j} k(y_i,y_j) - \frac{2}{mn}\sum_{i,j} k(x_i,y_j)$$ | Unbiased; polynomial kernel, lower better. |
| Precision & Recall (P/R) | Based on $k$-NN in feature space; measures coverage vs. quality. | Balances diversity and fidelity. |
| Perceptual Path Length (PPL) | $$\text{PPL}=\mathbb{E}_{z,\hat{z},\epsilon}\frac{\|f(G(z)) - f(G(\hat{z}))\|_2}{\epsilon^2}$$ | Smoothness of latent manifold. |
| Reconstruction Loss (cGAN/AE) | $$L_{\text{rec}}=\|x-G(E(x))\|_1$$ | Domain-specific tasks. |

Loss functions during training correspond to chosen GAN variant (JS, least-squares, Wasserstein, hinge). Evaluation separates loss from final sample quality metrics.

---

### 7. Pros vs. Cons  

Pros  
* Capable of high-fidelity, sharp samples.  
* Implicit density modeling – avoids explicit likelihood.  
* Flexible conditioning (cGAN, StyleGAN, BigGAN).

Cons  
* Training instability (mode collapse, vanishing gradients).  
* Evaluation metrics imperfect.  
* Sensitive to hyper-parameters & architecture.

---

### 8. Cutting-Edge Advances  

* StyleGAN-3: alias-free convolutions, $$G = f_{\text{style}}(z,w)$$ with modulation/demodulation.  
* Diffusion-probability distillation hybrids (e.g., Palette) integrating GAN discriminators for perceptual loss.  
* Consistency Regularization (CR, bCR, zCR): $$L_{\text{CR}}=\|D(T(x))-D(x)\|_2^2+\|D(T(G(z)))-D(G(z))\|_2^2$$  
* Transformers as Generators (TransGAN, MaskGit).  
* Uni-GAN frameworks unifying adversarial and likelihood training via dual objectives.

---

### 9. Best Practices & Pitfalls  

Best Practices  
* Use spectral normalization on $D$.  
* Two-time-scale update rule (TTUR): $$\eta_D ≫ \eta_G$$ for stable convergence.  
* Progressive growing for high-res images.  
* EMA of generator weights.  
* Data augmentation (ADA) with adaptive probability tuned via discriminator overfitting signal.

Pitfalls  
* Imbalanced capacity $G$ vs. $D$ → mode collapse/overpowering.  
* Inadequate batch size → gradient noise → training collapse.  
* Ignoring evaluation bias due to small sample size in FID.

---

Variable Glossary  
$z$: latent vector; $d_z$: latent dimension; $x$: data sample; $d_x$: sample dimension; $\theta_G,\theta_D$: parameters; $\mu_r,\Sigma_r$: real feature mean/cov; $\mu_g,\Sigma_g$: generated; $k(\cdot,\cdot)$: kernel; $f$: feature extractor; $p_G$: model distribution.