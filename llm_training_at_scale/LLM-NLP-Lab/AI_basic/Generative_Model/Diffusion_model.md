## 1. Definition  
Diffusion Models are generative latent-variable models that learn a **parameterized time-reverse stochastic process** to transform white noise into data samples by inverting a fixed **forward (diffusion) Markov chain** that gradually destroys structure.

---

## 2. Pertinent Equations  

| Component | Equation | Variables |
|-----------|----------|-----------|
| Forward diffusion | $$q(x_t\mid x_{t-1})=\mathcal N\!\bigl(x_t;\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t\mathbf I\bigr)$$ | $x_0$: data, $x_t$: latent at step $t$, $\beta_t\!\in\!(0,1)$: noise schedule |
| Closed form | $$q(x_t\mid x_0)=\mathcal N\!\bigl(x_t;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)\mathbf I\bigr),\quad\bar\alpha_t=\prod_{s=1}^t\alpha_s,\;\alpha_s=1-\beta_s$$ |  |
| Reverse model | $$p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\bigl(x_{t-1};\mu_\theta(x_t,t),\;\Sigma_\theta(x_t,t)\bigr)$$ | $\theta$: neural parameters |
| Optimal mean | $$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Bigl(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon_\theta(x_t,t)\Bigr)$$ | $\hat\epsilon_\theta$: predicted noise |
| Training loss (DDPM) | $$\mathcal L_{\text{MSE}}=\mathbb E_{t,x_0,\epsilon}\bigl[\lVert\epsilon-\hat\epsilon_\theta(x_t,t)\rVert_2^2\bigr]$$ | $\epsilon\sim\mathcal N(0,\mathbf I)$ |
| Variational lower-bound | $$\mathcal L_{\text{VB}}=\mathbb E_{q}\Bigl[\mathrm{KL}\bigl(q(x_T\mid x_0)\parallel p(x_T)\bigr)+\sum_{t=1}^T\mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\parallel p_\theta(x_{t-1}\mid x_t)\bigr)-\log p_\theta(x_0\mid x_1)\Bigr]$$ |  |
| Deterministic DDIM step | $$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\Bigl(\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_t}}\Bigr)+\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_\theta$$ |  |
| Guidance (classifier-free) | $$\hat\epsilon_{\text{guided}}=\hat\epsilon_\theta(x_t,t,y_\emptyset)+s\bigl(\hat\epsilon_\theta(x_t,t,y)-\hat\epsilon_\theta(x_t,t,y_\emptyset)\bigr)$$ | $y$: condition, $s$: guidance scale |
| Fréchet Inception Distance | $$\text{FID}=||\mu_r-\mu_g||_2^2+\operatorname{Tr}\bigl(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\bigr)$$ | $(\mu_r,\Sigma_r)$ real; $(\mu_g,\Sigma_g)$ generated features |

---

## 3. Key Principles  
- Noise schedules $(\beta_t)$ control information destruction; common: linear, cosine ($\beta_t\!\propto\!1-\cos(\frac{t}{T}\pi/2)$).  
- The reverse network learns **score function** $s_\theta(x_t,t)=\nabla_{x_t}\log q(x_t)$ (equivalent to $\hat\epsilon_\theta$).  
- Training reduces to denoising score matching with simple MSE.  
- **Markov chain length** $T$ trades compute vs. fidelity.  
- **Reparameterization** allows closed-form $\epsilon$ prediction.  
- In conditional setups, supply $y$ via cross-attention or feature concat.  
- Samplers: ancestral (DDPM), non-Markovian (DDIM), ODE solvers (score-based SDE).  

---

## 4. Detailed Concept Analysis  

### 4.1 Data Pre-processing  
1. Normalize images to $[-1,1]$.  
2. Optionally apply **VAE** encoder $E_\phi$ to obtain latent $z_0=E_\phi(x_0)$ (for latent diffusion).  
3. Choose schedule $\{\beta_t\}_{t=1}^T$; precompute $\{\alpha_t,\bar\alpha_t,\sqrt{1-\bar\alpha_t}\}$.

### 4.2 Core Architecture (U-Net variant)  
Sub-blocks:
- Convolutional residual block: $$\mathbf h_{l+1}=f_{\text{Res}}\bigl(\mathbf h_l,t,y\bigr)$$  
- Time embedding: $$\gamma(t)=\operatorname{MLP}\bigl([\sin(\omega t),\cos(\omega t)]\bigr)$$  
- Cross-attention (for text):  
  $$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Bigl(\frac{QK^\top}{\sqrt d}\Bigr)V$$  
  with $Q=W_Q \mathbf h,\,K=W_K e_y,\,V=W_V e_y$.

Full U-Net forward:  

```
x = input noisy image
h = StemConv(x)
for depth d:
    h = ResBlock(h, γ(t), y)
    skip[d] = h
    h = Downsample(h)
h = MiddleBlock(h, γ(t), y)
for depth d reversed:
    h = Upsample(h)
    h = concat(h, skip[d])
    h = ResBlock(h, γ(t), y)
ε̂ = OutConv(h)
```

### 4.3 Training Pseudo-Algorithm  

```
Given: Dataset D, steps T, schedule {βt}
for each iteration do
    x0 ← sample_batch(D)
    t  ← Uniform{1,…,T}
    ε  ← N(0,I)
    xt ← √ᾱt · x0 + √(1-ᾱt) · ε          # forward diffusion
    if rand()<p_uncond then y←∅ else keep label
    ε̂ ← Modelθ(xt, t, y)                  # noise prediction
    L  ← ||ε - ε̂||²                      # MSE loss
    θ  ← θ - η ∇θ L                       # optimizer step
end for
```

Mathematical justification: minimizing $L$ is equivalent to maximizing evidence lower bound because $$\mathcal L_{\text{MSE}}\propto\mathcal L_{\text{VB}}$$ up to constants when variance schedule is fixed.

### 4.4 Post-Training Procedures  
- **Guidance tuning**: adjust scale $s$ in $$\hat\epsilon_{\text{guided}}$$ to trade diversity vs. fidelity.  
- **Sampler acceleration**: use DDIM with $K\!\ll\!T$ steps (e.g., 50).  
- **EMA weights**: maintain $\theta_{\text{EMA}}=\lambda\theta_{\text{EMA}}+(1-\lambda)\theta$ to stabilize generation.  

### 4.5 Inference (Sampling)  

```
xT ~ N(0,I)
for t = T,…,1 do
    ε̂ = Modelθ(x_t, t, y)
    μ = (1/√αt) * (x_t - βt/√(1-ᾱt) * ε̂)
    if t>1 then σ = √βt else σ=0
    x_{t-1} = μ + σ · N(0,I)
return x0
```

For ODE sampling, solve $$\frac{dx}{dt}=f_\theta(x,t)=\frac{1}{2}\beta_t(\hat\epsilon_\theta(x,t)-x)/\sqrt{1-\bar\alpha_t}$$ via adaptive RK45.

---

## 5. Importance  
- Achieves SOTA image/audio/text generation with high fidelity and diversity.  
- Enables controllable synthesis via conditioning (text, layout, depth).  
- Provides likelihood estimates unlike GANs.  

---

## 6. Pros versus Cons  

| Pros | Cons |
|------|------|
| Stable training (simple MSE) | Slow sampling (hundreds of steps) |
| Likelihood computation | High memory footprint |
| Flexible conditioning | Performance sensitive to schedule |
| Superior mode coverage | Requires careful sampler tuning |

---

## 7. Cutting-Edge Advances  
- **Latent Diffusion (LDM)**: run diffusion in VAE latent space to accelerate $$\times\!(4\!-\!10)$$.  
- **ADM-CM3**: multi-modal cross-modal generation (image↔text↔audio).  
- **Consistency Models**: train to allow single-step sampling using consistency loss $$\lVert x_{t-1}-f_\theta(x_t,t)\rVert_2^2$$.  
- **Diffusion Transformers (DiT)**: replace U-Net with Vision Transformer blocks, improve scale-up.  
- **Rectified Flow / Flow Matching**: view diffusion as continuous normalizing flow, train with one-step trajectories.  
- **Progressive Distillation**: distill $T$-step diffuser into $T/2,\dots,1$-step models iteratively.  

---

## 8. Evaluation Metrics  

| Metric | Formal Definition | Goal |
|--------|-------------------|------|
| FID | see $$\text{FID}$$ above | Fidelity & diversity |
| Inception Score | $$\exp\bigl(\mathbb E_x KL(p(y\mid x)\,\|\,p(y))\bigr)$$ | Class entropies |
| sFID | FID in latent VAE space | Latent quality |
| Precision/Recall | $$P=\frac{|M\cap G|}{|G|},\;R=\frac{|M\cap G|}{|M|}$$ in feature space sets | Mode coverage |
| Negative Log-Likelihood | $$-\log p_\theta(x_0)\approx\mathcal L_{\text{VB}}$$ | Density modeling |
| CLIP Score (text-image) | $$\text{CS}=\frac{1}{N}\sum_i \frac{e_i^\top c_i}{\lVert e_i\rVert\lVert c_i\rVert}$$ | Alignment |
| RMSE on denoising | $$\sqrt{\mathcal L_{\text{MSE}}}$$ | Training diagnostic |

Best practices:  
- Use **50k** generated samples and **stats from train set** for robust FID.  
- Report confidence intervals via 3 independent seeds.  

Potential pitfalls:  
- Small batch FID variance.  
- Over-guidance $\to$ artifacts, CLIP score drop.  

---