# StyleGAN – Comprehensive Technical Breakdown  

---

## 1. Definition  
StyleGAN is a generative adversarial network that separates **latent-code mapping** from **image synthesis**, introduces per-layer **style modulation**, stochastic **noise injection**, and specialized **regularizers** to generate high-fidelity, high-resolution images.

---

## 2. Pertinent Equations  

### 2.1 Latent Mapping  
- Input latent $$z \sim \mathcal{N}(0, I_{d_z})$$  
- Mapping network (8-layer MLP)  
  $$w = f_{\theta_m}(z)$$  
  where $$f_{\theta_m}: \mathbb{R}^{d_z} \rightarrow \mathbb{R}^{d_w}$$  

### 2.2 Learned Constant & Synthesis Input  
- Initial feature map $$x_{0} = c \in \mathbb{R}^{C_0\times 4 \times 4}$$  

### 2.3 Weight Modulation & Demodulation (StyleGAN2)  
Given convolutional weight tensor $$W_l \in \mathbb{R}^{C_{out}\times C_{in}\times k\times k}$$ and style $$s_l = A_l w$$,  
1. **Modulation**:  $$\hat{W}_{l}^{(i)} = s_{l,i} \cdot W_l^{(i)}, \quad i\!=\!1\ldots C_{in}$$  
2. **Demodulation**:  
   $$\tilde{W}_{l}^{(i)} = \dfrac{\hat{W}_{l}^{(i)}}{\sqrt{\sum_{j,k,k'} \hat{W}_{l,jk'k}^{2} + \epsilon}}$$  

### 2.4 Convolution + Noise Injection  
$$y_{l} = \tilde{W}_{l} * x_{l-1} + \beta_l\,n_l,\quad n_l \sim \mathcal{N}(0, I)$$  

### 2.5 Activation & AdaIN (StyleGAN1)  
$$\operatorname{AdaIN}(y_l, s_l)=s_{l,\text{scale}}\frac{y_l-\mu(y_l)}{\sigma(y_l)}+s_{l,\text{bias}}$$  

### 2.6 Style Mixing Regularization  
For two latents $$w^{(a)}, w^{(b)}$$ pick crossover layer $k$:  
$$\bar{w}_l=\begin{cases} w^{(a)} & l<k\\ w^{(b)} & l\ge k\end{cases}$$  

### 2.7 Generator & Discriminator Losses  
Non-saturating GAN with regularizers:  

Generator  
$$\mathcal{L}_G = \mathbb{E}_{z}[-\log D(G(z))] + \lambda_{\text{PL}}\; \mathbb{E}_{w}\bigl[(\|\nabla_{w}G(w)\|_2 - a )^{2}\bigr]$$  

Discriminator  
$$\mathcal{L}_D = \mathbb{E}_{x\sim p_{\text{data}}}[\max(0,1-D(x))] + \mathbb{E}_{z}[D(G(z))] + \lambda_{R1}\;\mathbb{E}_{x}[\|\nabla_{x} D(x)\|_2^{2}]$$  

where  
• $\lambda_{\text{PL}}$ = path-length penalty weight  
• $\lambda_{R1}$ = R1 gradient-penalty weight  
• $a$ ≈ expected path length (moving average)

---

## 3. Key Principles  
- **W-space disentanglement**: mapping network reduces latent entanglement.  
- **Per-layer styles**: control coarse-to-fine attributes.  
- **Stochastic noise**: injects micro-scale stochasticity.  
- **Weight demodulation**: removes “droplet” artifacts, equalizes signal magnitude.  
- **Progressive growing removed (StyleGAN2/3)**: replaced by constant resolution training + weight resampling.  
- **Path-length & R1 regularization**: stabilize training, enforce smooth latent mapping.

---

## 4. Detailed Concept Analysis  

### 4.1 Latent Spaces  
- $\mathcal{Z}$-space: original Gaussian latent.  
- $\mathcal{W}$-space: affine-transformed latent, more disentangled.  
- $\mathcal{W}^+$-space: per-layer, enables style mixing.  

### 4.2 Noise Channels  
Layer-specific, uncorrelated noise $n_l$ adds high-frequency variation; $\beta_l$ is learned scalar per channel.  

### 4.3 Equalized Learning Rate  
Weights initialized as $$W \sim \mathcal{N}(0, \frac{1}{\sqrt{C_{in}k^2}})$$ and scaled at runtime by $$\gamma = \sqrt{2/(C_{in}k^2)}$$.  

### 4.4 Minibatch Std-Dev Feature  
Append channel $$\sigma_{\text{mb}} = \text{std}(x)$$ to discourage mode collapse.  

---

## 5. Importance  
- Sets SOTA in unconditional image synthesis (FFHQ 1024×1024, FID < 3).  
- W-space enables intuitive semantic editing.  
- Foundation for domain adaptation (StyleGAN-ADA, StyleGAN-XL) and video (StyleGAN-V).  

---

## 6. Pros ⟂ Cons  

Pros  
• High-resolution, realistic outputs  
• Latent disentanglement → controllable editability  
• Modular—easy adaptation to other modalities  

Cons  
• Large memory/compute (~70 M params)  
• Training instability without heavy regularization  
• Limited temporal consistency (fixed-image prior)  

---

## 7. Cutting-Edge Advances  

- **StyleGAN-ADA**: adaptive augment $$\mathcal{L}_{\text{aug}}=D(T(x))$$; probability tuned via PID.  
- **StyleGAN-XL**: bigger $$C_{0}$$, class-conditional $$y$$ embedding added to modulation.  
- **StyleGAN-3**: alias-free synthesis; replace discrete upsampling with FIR filters $$k(f)$$, continuous generator.  
- **StyleGAN-V**: video-rate generator using 3D convolutions + temporal latent trajectory.  

---

## 8. Training Pseudo-Algorithm (Mathematically Annotated)  

```
Given dataset 𝔻 = {x_i}, batch size B
Initialize θ_G, θ_D, θ_m, running mean a ← 0

for iteration t = 1 … T:
    # 1. Sample latents and map
    z ← 𝒩(0, I)^{B×d_z}
    w ← f_{θ_m}(z)

    # 2. Generator forward
    x̂ ← G_{θ_G}(w)     # uses Eqs. (2.3)–(2.5)

    # 3. Discriminator forward
    d_real ← D_{θ_D}(x)
    d_fake ← D_{θ_D}(x̂)

    # 4. Losses
    L_G ← 𝔼[-log d_fake] + λ_PL (‖∇_w x̂‖₂ - a)²
    L_D ← 𝔼[ReLU(1 - d_real)] + 𝔼[d_fake] + λ_R1 ‖∇_x d_real‖₂²

    # 5. Update running path length
    a ← 0.99·a + 0.01·‖∇_w x̂‖₂

    # 6. Optimizer steps (Adam)
    θ_G ← θ_G - η·∇_{θ_G} L_G
    θ_D ← θ_D - η·∇_{θ_D} L_D
```

Hyper-parameters (FFHQ-1024):  
$B$ = 32, η = 0.002, β₁ = 0, β₂ = 0.99, λ_{PL}=2, λ_{R1}=10.

---

## 9. Post-Training Procedures  

### 9.1 Truncation Trick  
$$\tilde{w} = \bar{w} + \psi\,(w - \bar{w}),\quad \psi<1$$  
controls trade-off: fidelity ↑ vs diversity ↓.  

### 9.2 Latent Optimization (PTI/e4e)  
Minimize $$\min_{w}\|E(w) - x_{\text{target}}\|_p + \lambda_{\text{percep}}\,\text{LPIPS}(G(w), x_{\text{target}})$$ for editing & inversion.  

---

## 10. Evaluation Metrics  

| Metric | Formal Definition | Notes |
|--------|-------------------|-------|
| FID | $$\text{FID} = \|\mu_r - \mu_g\|_2^{2} + \operatorname{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$ | Lower is better |
| KID | $$\text{KID}= \frac{1}{m}\sum_{i=1}^{m} \widehat{\operatorname{MMD}}^2_i$$ | Unbiased vs FID |
| Inception Score | $$\exp\bigl( \mathbb{E}_{x} KL(p(y|x)\|p(y))\bigr)$$ | Sensitive to mode collapse |
| LPIPS | $$\frac{1}{HWC}\sum_{l} w_l \|\phi_l(x) - \phi_l(x')\|_2^{2}$$ | Perceptual similarity |
| Precision/Recall | $$\text{Prec}=\frac{|M_{gen}\cap M_{real}|}{|M_{gen}|},\; \text{Rec}=\frac{|M_{gen}\cap M_{real}|}{|M_{real}|}$$ in feature space | Diversity analysis |

Domain-specific: **ID similarity** (faces): $$\text{ID} = 1 - \cos(\phi(x), \phi(x'))$$; **e-LPIPS** for editing fidelity.

---

### Best Practices / Pitfalls  
- Use **Adaptive Augmentation** when $|𝔻|<100\,$k to prevent over-fitting.  
- Balance $\lambda_{\text{PL}}$ and batch size; too high ⇒ droplet artifacts.  
- R1 penalty on **real** samples only; full-gradient penalty slows convergence.  
- Monitor **precision/recall** jointly; optimizing only FID can reduce diversity.  

---