# Generative Adversarial Networks (GANs)
### 1. Definition  
Generative Adversarial Networks (GANs) are two-player zero-sum games between a generator $G$ and a discriminator $D$ where $G:\mathcal{Z}\!\to\!\mathcal{X}$ learns a data distribution $p_{\text{data}}$ by mapping noise $z\!\sim\!p_z$ into realistic samples while $D:\mathcal{X}\!\to\![0,1]$ distinguishes real from synthetic data.

---

### 2. Pertinent Equations  

Minimax objective (original):  
$$
\min_{G}\max_{D}V(D,G)=
\mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big] +
\mathbb{E}_{z\sim p_{z}}\big[\log\big(1-D\big(G(z)\big)\big)\big]
$$

Non-saturating generator loss:  
$$
L_G^{\text{NS}}=-\mathbb{E}_{z\sim p_z}\!\big[\log D(G(z))\big]
$$

Wasserstein GAN (WGAN):  
$$
\min_G\max_{D\in\mathcal{D}_1} \;
W(p_{\text{data}},p_G)=
\mathbb{E}_{x\sim p_{\text{data}}}[D(x)]-
\mathbb{E}_{z\sim p_z}[D(G(z))]
$$  
with 1-Lipschitz constraint $D\!\in\!\mathcal{D}_1$.

Gradient penalty (WGAN-GP):  
$$
L_{\text{GP}}=\lambda_{\text{GP}}\;
\mathbb{E}_{\hat{x}\sim p_{\hat{x}}}\!
\big[(\lVert\nabla_{\hat{x}}D(\hat{x})\rVert_2-1)^2\big]
$$

Least-Squares GAN (LSGAN):  
$$
L_D=\tfrac12\,
\mathbb{E}_{x\sim p_{\text{data}}}\!\big[(D(x)-b)^2\big] +
\tfrac12\,
\mathbb{E}_{z\sim p_z}\!\big[(D(G(z))-a)^2\big]
$$

Spectral normalization:  
$$
\hat{W}=\frac{W}{\sigma_{\max}(W)}
,\quad
\sigma_{\max}(W)=\max_{v\neq0}\frac{\lVert Wv\rVert_2}{\lVert v\rVert_2}
$$

---

### 3. Key Principles  
• Adversarial Training  
• Minimax Optimization & Nash Equilibrium  
• Implicit Density Modeling (no explicit likelihood)  
• Lipschitz Regularization for stability  
• Divergence/Distance selection (JS, Wasserstein, Pearson χ², etc.)  

---

### 4. Detailed Concept Analysis  

#### 4.1 Core Architecture  
• Generator: series of transposed convolutions (or style-modulated blocks) producing $x\_g=G(z)$.  
• Discriminator: CNN returning probability $D(x)$ (or critic score).  
• Conditioning: $G(z,c)$, $D(x,c)$ with embedding of class/attribute $c$.  
• StyleGAN: mapping network $f:\mathcal{Z}\!\to\!\mathcal{W}$, adaptive instance normalization (AdaIN):  
$$
\text{AdaIN}(h,w,y)=\gamma(y)\frac{h-\mu(h)}{\sigma(h)}+\beta(y)
$$

#### 4.2 Data Pre-Processing  
1. Normalize input images: $x\leftarrow (x-\mu)/\sigma$.  
2. Optional: resize to power-of-two resolutions with anti-aliasing.  
3. Augmentation (DiffAug, ADA): stochastic transforms $T$ with probability $p$, enforce $D(T(x))$ consistency.

Mathematical form: $$\tilde{x}=T(x;\theta_T),\quad\theta_T\sim p_T$$  

#### 4.3 Training Pseudo-Algorithm (Mini-batch size $m$)  
```
for each iteration do
    #— Discriminator update —#
    for t = 1 … n_d do
        {x_i} ← sample_real(m)            # x_i ∼ p_data
        {z_i} ← sample_noise(m)           # z_i ∼ p_z
        g_i   ← G(z_i; θ_G)               # fake
        θ_D ← θ_D + η_D · ∇_{θ_D} [
                  - (1/m) Σ_i log D(x_i)
                  - (1/m) Σ_i log(1 - D(g_i))
                  + L_GP(θ_D) ]           # if GP used
    #— Generator update —#
    {z_i} ← sample_noise(m)
    θ_G ← θ_G - η_G · ∇_{θ_G} [
              - (1/m) Σ_i log D(G(z_i)) ] # non-saturating
end for
```
All gradients computed using automatic differentiation; $\eta_D,\eta_G$ = learning rates. Use Adam with $(\beta_1,\beta_2)=(0.5,0.999)$.

#### 4.4 Post-Training Procedures  
• Moving-average (EMA) generator weights:  
$$
\theta_G^{\text{EMA}}\!\leftarrow\!\alpha\theta_G^{\text{EMA}}+(1-\alpha)\theta_G
$$  
• Truncation trick: sample $z$ conditioned on $$\lVert z\rVert_2 < \tau$$ to trade diversity for fidelity.  
• Latent space editing: linear/semi-linear directions $d$; manipulate output via $G(z+αd)$.  

---

### 5. Importance  
• State-of-the-art photorealistic synthesis, data augmentation, domain adaptation, super-resolution, 3-D shape generation, and creative content.  
• Foundation for diffusion models’ adversarial variants, adversarial representation learning (BiGAN, ALI).  

---

### 6. Pros vs Cons  

Pros  
• Sharp outputs (no regression blur).  
• Flexible, likelihood-free.  
• Competitive sample efficiency.  

Cons  
• Unstable training (mode collapse, gradient vanishing/explosion).  
• Evaluation is non-trivial (no log-likelihood).  
• Sensitive to hyper-parameters and architecture.  

---

### 7. Cutting-Edge Advances  

• StyleGAN3: alias-free convolutions ensuring translation equivariance.  
• Diffusion-GAN hybrids (e.g., Denoising Diffusion GAN).  
• Consistency regularization (CR, ADA, LeCam).  
• Self-supervised GANs using CL (e.g., Projected GAN).  
• Text-conditioned GANs with transformer backbones (e.g., MuseGAN).  

---

### 8. Evaluation Metrics  

Metric definitions ($x_r$ real, $x_g$ generated, $f$ pre-trained Inception):  

• Inception Score (IS):  
$$
\text{IS}= \exp\!\Big( \mathbb{E}_{x_g}\big[ KL(p(y|x_g)\,\|\,\mathbb{E}_{x_g}[p(y|x_g)]) \big] \Big)
$$

• Fréchet Inception Distance (FID):  
Compute $\mu_r,\Sigma_r$ from $f(x_r)$ and $\mu_g,\Sigma_g$ from $f(x_g)$:  
$$
\text{FID}= \lVert\mu_r-\mu_g\rVert_2^{2}
+ \text{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)
$$

• Precision/Recall ($k$-nearest in feature space):  
$$
P = \frac{|\,\mathcal{G}\cap\text{Manifold}(\mathcal{R})\,|}{|\mathcal{G}|},
\quad
R = \frac{|\,\mathcal{R}\cap\text{Manifold}(\mathcal{G})\,|}{|\mathcal{R}|}
$$

• Kernel Inception Distance (KID):  
$$
\text{KID} = \frac{1}{m(m-1)}\!\sum_{i\neq j} k(f(x_i),f(x_j))
$$  
with polynomial kernel $k$.

• LPIPS Diversity:  
$$
\text{Div} = \mathbb{E}_{z_1,z_2}\big[\lVert\phi(G(z_1))-\phi(G(z_2))\rVert_1\big]
$$

Loss monitoring: adversarial loss curves, gradient norms, singular value spectrum of $D$.

---

### 9. Best Practices & Pitfalls  

Best Practices  
• Use spectral norm or gradient penalty to enforce Lipschitzness.  
• Keep $n_d\!>\!n_g$ (e.g., 5:1 in WGAN).  
• Apply EMA on $G$ for evaluation.  
• Employ adaptive augmentations to prevent discriminator overfitting.  
• Prefer non-saturating or Wasserstein losses for stability.

Pitfalls  
• Mode collapse → detect via precision/recall; mitigate with minibatch std-dev layer.  
• Discriminator overpowering → adjust $n_d$ or learning rates.  
• Exploding gradients → clip or use GP.  

---