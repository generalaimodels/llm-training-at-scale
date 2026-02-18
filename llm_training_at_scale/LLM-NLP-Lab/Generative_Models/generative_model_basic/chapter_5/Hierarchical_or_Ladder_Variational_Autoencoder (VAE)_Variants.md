# Hierarchical / Ladder Variational Autoencoders (HVAE, LVAE, VLAE, NVAE, etc.)

---

## 1 Concise Definition  
Hierarchical or Ladder VAEs are generative models that employ a stack of latent variables $\mathbf{z}^{(1)},\dots,\mathbf{z}^{(L)}$ organized from coarse (top) to fine (bottom) resolutions. The decoder samples top–down, progressively refining the representation of the data, while the encoder infers posteriors bottom–up (ladder connections). This multi‐level structure improves expressiveness, enables disentanglement at multiple abstraction scales, and mitigates posterior collapse.

---

## 2 Mathematical Formulation  

### 2.1 Generative Process (Top–Down)  
Let $L$ denote the number of latent layers; $\mathbf{x}$ is the observed variable.

$$
\begin{aligned}
p_\theta(\mathbf{x},\mathbf{z}^{(1:L)}) &= p_\theta(\mathbf{z}^{(L)})\;
                                          \prod_{\ell=L-1}^{1} 
                                          p_\theta\!\bigl(\mathbf{z}^{(\ell)} \mid \mathbf{z}^{(\ell+1)}\bigr)\;
                                          p_\theta\!\bigl(\mathbf{x}\mid\mathbf{z}^{(1)}\bigr) \\
p_\theta(\mathbf{z}^{(L)}) &= \mathcal{N}\!\bigl(\mathbf{0},\mathbf{I}\bigr) \\
p_\theta\!\bigl(\mathbf{z}^{(\ell)} \mid \mathbf{z}^{(\ell+1)}\bigr) 
                           &= \mathcal{N}\!\bigl(\boldsymbol{\mu}_\theta^{(\ell)}(\mathbf{z}^{(\ell+1)}),
                                                 \operatorname{diag}\!\bigl(\boldsymbol{\sigma}_\theta^{2(\ell)}(\mathbf{z}^{(\ell+1)})\bigr)\bigr)
\end{aligned}
$$

The conditional means/variances are parameterized by neural networks.

### 2.2 Inference Model (Bottom–Up Ladder)  

Different variants adopt different factorizations; the most common (Ladder VAE) is

$$
q_\phi(\mathbf{z}^{(1:L)} \mid \mathbf{x}) 
= q_\phi\bigl(\mathbf{z}^{(1)}\mid\mathbf{x}\bigr)
  \prod_{\ell=2}^{L} q_\phi\!\bigl(\mathbf{z}^{(\ell)} \mid \mathbf{z}^{(\ell-1)}\bigr),
$$
where each posterior is Gaussian:

$$
q_\phi\!\bigl(\mathbf{z}^{(\ell)} \mid \mathbf{*}\bigr)
= \mathcal{N}\!\bigl(\boldsymbol{\mu}_\phi^{(\ell)}(\cdot),\,
                     \operatorname{diag}\!\bigl(\boldsymbol{\sigma}_\phi^{2(\ell)}(\cdot)\bigr)\bigr).
$$

Alternative: NVAE uses *skip flows*: $q_\phi(\mathbf{z}^{(\ell)}\mid\mathbf{x})$ obtains information from all previous encoder layers.

### 2.3 Evidence Lower Bound  

$$
\mathcal{L}(\theta,\phi;\mathbf{x}) =
\mathbb{E}_{q_\phi}\!\Bigl[\log p_\theta(\mathbf{x}\mid\mathbf{z}^{(1)})\Bigr] 
- \sum_{\ell=1}^{L}
    D_{\mathrm{KL}}\!\Bigl(
       q_\phi\bigl(\mathbf{z}^{(\ell)}\mid \mathbf{*}\bigr)
       \big\|\, p_\theta\bigl(\mathbf{z}^{(\ell)}\mid\mathbf{z}^{(\ell+1)}\bigr)
    \Bigr).
$$

The KL terms can be annealed or re‐weighted per layer: $\beta_\ell D_{\mathrm{KL}}^{(\ell)}$.

### 2.4 Reparameterization per Layer  

$$
\mathbf{z}^{(\ell)} =
\boldsymbol{\mu}_\phi^{(\ell)} + 
\boldsymbol{\sigma}_\phi^{(\ell)} \odot \boldsymbol{\epsilon}^{(\ell)},
\qquad
\boldsymbol{\epsilon}^{(\ell)}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

---

## 3 Architectural Design Patterns  

| Component | Typical Choice | Notes |
|-----------|----------------|-------|
| Encoder | $L$ conv blocks (images) or Transformer layers (text) with down‐sampling | produce feature maps $\mathbf{h}^{(1)},\dots,\mathbf{h}^{(L)}$ |
| Posterior Heads | $1\times1$ conv / MLP to $(\boldsymbol{\mu}_\phi^{(\ell)},\log\boldsymbol{\sigma}_\phi^{(\ell)})$ | ladder: each head sees $\mathbf{h}^{(\ell)}$ and top‐down prior |
| Decoder (top–down) | transpose conv / autoregressive PixelCNN | inputs $\mathbf{z}^{(\ell+1)}$ and sampled $\mathbf{z}^{(\ell)}$ |
| Skip Connections | feature concatenation or gating | critical to reduce information loss |
| Latent sizes | decreasing: $d_z^{(L)} < \dots < d_z^{(1)}$ | coarse→fine hierarchy |

---

## 4 Training Procedure  

1. **Forward (Encoder)**  
   • $\mathbf{x}\rightarrow$ produce $\mathbf{h}^{(\ell)}$ bottom–up.  
   • Sample $\mathbf{z}^{(1)}$ from $q_\phi(\mathbf{z}^{(1)}\mid\mathbf{x})$.  
   • For $\ell=2\ldots L$: sample $\mathbf{z}^{(\ell)}$ from $q_\phi(\mathbf{z}^{(\ell)}\mid\mathbf{z}^{(\ell-1)})$.

2. **Forward (Decoder)**  
   • Start with $\mathbf{z}^{(L)}$; for $\ell=L-1\ldots1$: compute prior $p_\theta(\mathbf{z}^{(\ell)}\mid\mathbf{z}^{(\ell+1)})$ and sample during generation (use mean during training).  
   • Produce reconstruction $\hat{\mathbf{x}}$.

3. **Compute ELBO** with layer‐wise KLs.

4. **Backpropagation** using reparameterization gradients; optionally apply  
   • KL annealing schedule per layer.  
   • Free‐bits: enforce $\max(D_{\mathrm{KL}}^{(\ell)},\tau)$.

5. **Optimization**: AdamW/LAMB with gradient clipping.

---

## 5 Algorithmic Pseudocode  

```python
def forward(x):
    h = encoder_backbone(x)          # list length L
    mu, logvar = posterior_heads(h)  # per level
    z = [None]*L
    for l in range(L):
        eps = torch.randn_like(mu[l])
        z[l] = mu[l] + torch.exp(0.5*logvar[l]) * eps

    # top-down decoder
    prior_kl = 0
    for l in reversed(range(L)):
        if l == L-1:
            prior_mu = torch.zeros_like(mu[l])
            prior_logvar = torch.zeros_like(logvar[l])
        else:
            prior_mu, prior_logvar = top_down_prior(z[l+1])
        prior_kl += kl_gaussian(mu[l], logvar[l], prior_mu, prior_logvar)
        z[l] = top_down_merge(z[l], z[l+1])

    x_hat = decoder(z[0])
    recon_loss = likelihood(x_hat, x)
    loss = recon_loss + sum(beta[l]*prior_kl_l[l] for l in range(L))
    return loss
```

---

## 6 Advantages over Single‐Layer VAE  

* **Expressiveness**: Non‐linear hierarchical prior approximates arbitrarily complex posteriors.  
* **Multi‐scale Representation**: Higher layers capture global semantics; lower layers encode local details.  
* **Reduced Posterior Collapse**: Information can bypass powerful decoders via upper latent layers.  
* **Disentanglement**: Layer‐wise KL regularization yields factors at different abstraction levels.

---

## 7 Representative Variants  

| Model | Key Idea | Inference Factorization |
|-------|----------|-------------------------|
| LVAE (Sønderby 2016) | deterministic upward pass + stochastic top‐down refinement | ladder structure |
| VLAE (Zhao 2017) | shallow encoder, deep decoder; emphasis on disentanglement | $q(\mathbf{z}^{(\ell)}\mid\mathbf{x})$ independent given $\mathbf{x}$ |
| BIVA (Maaløe 2019) | double hierarchy: stochastic bottom‐up and top‐down latents | hybrid | 
| NVAE (Vahdat 2020) | deep ConvNet with residual blocks, skip flows, per‐channel latents | skip‐flow encoder |
| Hierarchical VQ‐VAE | discrete codebooks at multiple scales | quantized latents |

---

## 8 Regularization & Stabilization Techniques  

* **Layer‐wise $\beta_\ell$**: smaller $\beta$ for lower levels to avoid under‐utilization.  
* **Free Bits ($\tau$)**: guarantee minimal information per latent.  
* **Warm‐up**: gradually introduce KL terms; schedule can differ across layers.  
* **Orthogonal Weight Reg**: prevents decoder overpowering; beneficial in NVAE.  
* **Variance Clamping**: keep $\log\sigma \in [-7,7]$ to avoid numerical issues.

---

## 9 Evaluation Metrics  

| Aspect | Metric |
|--------|--------|
| Negative ELBO / NLL | $\mathrm{NLL}=-\mathcal{L}$ |
| Bits‐Per‐Dim | $\frac{-\log_2 p(\mathbf{x})}{D}$ |
| Mutual Information per Layer | $I(\mathbf{x};\mathbf{z}^{(\ell)})$ |
| Hierarchical Disentanglement | DCI, MIG computed separately for each $\mathbf{z}^{(\ell)}$ |
| Reconstruction / FID (images) | FID, IS, PSNR |

---

## 10 Generation & Manipulation  

1. **Unconditional Generation**  
   • Sample $\mathbf{z}^{(L)}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$.  
   • For $\ell=L-1\ldots1$: sample $\mathbf{z}^{(\ell)}\sim p_\theta(\,\cdot\mid\mathbf{z}^{(\ell+1)})$.  
   • Decode $\mathbf{z}^{(1)}\to\mathbf{x}$.

2. **Latent Traversals**  
   • Modify higher‐level latent dims → global semantic changes.  
   • Modify lower‐level dims → fine texture variations.

3. **Conditional Generation**  
   • Inject conditioning signal at specific layers (e.g., class, text).  
   • Keep $\mathbf{z}^{(>k)}$ fixed to preserve global content while varying fine details.

---

## 11 Implementation Tips  

* **Initialise top prior mean = 0, logvar = 0** to match standard Gaussian.  
* **Group Normalization + Swish** improves deep decoder stability (NVAE).  
* **Parameter Sharing** between upward and downward paths saves memory.  
* **Gradient Checkpointing** scales to $L>30$ layers.  
* **Mixed Precision (FP16/BF16)** viable; maintain master copy of variance to avoid underflow.

---

## 12 Summary  
Hierarchical / Ladder VAEs extend the VAE framework with stacked latent variables and a top–down generative pathway. This architecture yields more powerful density estimators, multi‐scale controllable representations, and better utilization of latent capacity. Proper factorization of the posterior, careful KL scheduling, and architectural choices like skip connections are essential for stable training and high‐fidelity generation.