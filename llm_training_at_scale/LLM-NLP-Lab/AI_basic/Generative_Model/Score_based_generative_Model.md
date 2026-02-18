## 1. Definition  
Score-Based Generative Models (SBGMs) synthesize data by learning the gradient of the log-density—called the score—of progressively noised data and then integrating a reverse-time diffusion (stochastic differential equation, SDE) or its deterministic probability-flow ordinary differential equation (ODE) to generate new samples.

---

## 2. Pertinent Equations  

### 2.1 Forward (Noising) SDEs  
1. Variance-Preserving (VP)  
   $$ d\mathbf{x}_t = -\tfrac{1}{2}\beta(t)\mathbf{x}_t\,dt + \sqrt{\beta(t)}\,d\mathbf{w}_t $$
2. Variance-Exploding (VE)  
   $$ d\mathbf{x}_t = \sqrt{\dfrac{d\sigma^2(t)}{dt}}\;d\mathbf{w}_t $$  
   • $ \mathbf{x}_t $: data at time $ t\in[0,1] $ ( $ t{=}0 $ → clean data, $ t{=}1 $ → nearly Gaussian )  
   • $ \mathbf{w}_t $: standard Wiener process  
   • $ \beta(t) $ or $ \sigma(t) $: variance schedule (monotonically increasing)

### 2.2 Reverse-Time SDE  
$$ d\mathbf{x}_t = \bigl[f(\mathbf{x}_t,t) - g^2(t)\,\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\bigr]\,dt + g(t)\,d\bar{\mathbf{w}}_t $$  
• $ f(\cdot,t) $: drift of forward SDE  
• $ g(t) $: diffusion coefficient  
• $ \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) \equiv s_\theta(\mathbf{x}_t,t) $ (score network)  
• $ \bar{\mathbf{w}}_t $: reverse-time Wiener process

### 2.3 Probability-Flow ODE (deterministic)  
$$ \dfrac{d\mathbf{x}_t}{dt} = f(\mathbf{x}_t,t) - \tfrac{1}{2}g^{2}(t)\,s_\theta(\mathbf{x}_t,t) $$

### 2.4 Denoising Score-Matching Loss  
Given data $ \mathbf{x}_0\sim p_\text{data} $ and noise level $ t\sim\mathcal{U}(0,1) $  
$$ L(\theta) = \mathbb{E}_{\mathbf{x}_0,t,\mathbf{w}}\Bigl[\lambda(t)\,\bigl\|s_\theta(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|\mathbf{x}_0)\bigr\|_2^2\Bigr] $$  
• $ \mathbf{x}_t $ generated via SDE at time $ t $  
• $ \lambda(t) $: weighting (e.g.\ $ \sigma^{2}(t) $ for VE)

---

## 3. Key Principles  

• Stochastic diffusion corrupts data → tractable Gaussian conditionals  
• Train neural net to estimate scores at multiple noise levels  
• Reverse SDE/ODE integration transforms Gaussian noise back to data distribution  
• Score-matching sidesteps likelihood normalization  
• Predictor–Corrector (PC) sampling: stochastic “predictor” step + Langevin “corrector” step for higher sample quality

---

## 4. Detailed Concept Analysis  

### 4.1 Data Pre-Processing  
- Normalize input to zero mean, unit variance  
- Optionally augment (flips, crops) to enhance diversity  
- Convert to continuous domain (e.g.\ dequantize images by uniform $ \mathcal{U}[0,1/256] $ )

### 4.2 Core Model Architecture  
- $ s_\theta(\mathbf{x},t) $ implemented via U-Net/ResNet with  
  • Time embedding $ \gamma(t)=\text{Fourier}(t) $  
  • Conditional layers: FiLM/Adaptive-GroupNorm  
  • Attention blocks for high-resolution  
  • For 3-D/graph data: PointNet++/GNN layers  

### 4.3 Training Objective Derivation  
Using denoising score matching:  
$$ \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|\mathbf{x}_0) = -\dfrac{\mathbf{x}_t-\mu(t)\mathbf{x}_0}{\sigma^2(t)} $$  
Insert into $ L(\theta) $ for closed-form target; minimizes Fisher divergence across all $ t $.

### 4.4 Post-Training Sampling  

#### 4.4.1 Predictor–Corrector (VE example)  
Predictor: Euler–Maruyama  
$$ \mathbf{x}_{t-\Delta t} = \mathbf{x}_t + g^2(t)\,s_\theta(\mathbf{x}_t,t)\,\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z},\qquad \mathbf{z}\sim\mathcal{N}(0,\mathbf{I}) $$  
Corrector: $ K $ Langevin steps  
$$ \mathbf{x} \leftarrow \mathbf{x} + \eta\,s_\theta(\mathbf{x},t) + \sqrt{2\eta}\,\mathbf{z} $$  
$ \eta $ adjusted to reach target signal-to-noise ratio (SNR).

#### 4.4.2 ODE Sampler (deterministic, fast)  
Integrate probability-flow ODE via RK45; enables exact likelihood computation by Hutchinson trace estimator.

---

## 5. Importance  
- State-of-the-art log-likelihoods on CIFAR-10 ($\,\leq$ 2.90 bits/dim)  
- High-fidelity image/audio/video synthesis surpassing GAN FID scores  
- Unified view of diffusion & denoising auto-encoders; bridges stochastic and deterministic flows

---

## 6. Pros vs Cons  

Pros  
• Stable training (no mode collapse)  
• Exact likelihood via ODE path integral  
• Flexible conditioning (text, depth, pose)  

Cons  
• Long sampling chains (hundreds–thousands steps)  
• Memory intensive multi-scale training  
• Sensitive to variance schedule hyper-parameters

---

## 7. Cutting-Edge Advances  

- DDIM: Non-stochastic fast sampler (≈50 steps)  
- EDM (Karras et al. 2022): optimal VE/VP blends, $ \sigma $-data parametrization  
- Consistency Models: single-step generator distilled from SBGM  
- Rectified Flow: improves linearity of trajectories, accelerates training  
- Latent Diffusion: projects to autoencoder latent to reduce compute  

---

## 8. Step-by-Step Training Pseudo-Algorithm  

```
Input: dataset D, SDE params {β(t) or σ(t)}, weighting λ(t), epochs E
Initialize θ randomly
for epoch = 1 … E:
    for minibatch {x0} ⊂ D:
        Sample t ~ Uniform(0,1)
        Generate noise ε ~ N(0,I)
        if VP: xt = exp(-0.5∫0^t β(s)ds)·x0 + sqrt(1-exp(-∫0^t β(s)ds))·ε
        if VE: xt = x0 + σ(t)·ε
        Compute target score  ŝ = -(xt - μ(t)x0) / σ²(t)
        Predict sθ = s_θ(xt, t)
        Compute loss L = λ(t)·‖sθ - ŝ‖²
        θ ← θ - η·∇θ L      # AdamW/SGD
return θ
```
Mathematical justification: minimizing $ L(\theta) $ estimates the Stein score, yielding consistency with reverse SDE per denoising score-matching theorem.

---

## 9. Evaluation Phase  

### 9.1 Generative Metrics  

• Negative Log-Likelihood  
  $$ \text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(\mathbf{x}_0^{(i)}) \quad [\text{bits/dim}] $$  
  computed via probability-flow ODE, Hutchinson trace, and change-of-variables.

• Frechet Inception Distance (FID)  
  $$ \text{FID} = \| \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \|_2^2 + \operatorname{Tr}\bigl(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\bigr) $$  
  $ (\boldsymbol{\mu}_r,\Sigma_r) $ reference; $ (\boldsymbol{\mu}_g,\Sigma_g) $ generated features.

• Inception Score (IS)  
  $$ \exp\Bigl(\mathbb{E}_{\mathbf{x}}\,D_\text{KL}(p(y|\mathbf{x})\;\|\;p(y))\Bigr) $$

• Perceptual Path Length, CLIP-Score for text-conditioned models.

### 9.2 Loss Functions During Training  

- Denoising Score-Matching (primary)  
- Augmented $ L_2 $ or $ L_H $ penalties for gradient norm regularization  
- EMAs of $ \theta $ to stabilize $ s_\theta $

### 9.3 Domain-Specific Metrics (examples)  

• Audio: Signal-to-Noise Ratio (SNR), Log-Spectral Distance (LSD)  
• Molecular graphs: Validity/Uniqueness/Novelty percentages  
• Point clouds: Chamfer Distance  

---

## 10. Best Practices & Pitfalls  

Best Practices  
• Use exponential moving average (EMA) of parameters for sampling  
• Employ noise-level conditioning layers for robust generalization  
• Antithetic $ t $ sampling to reduce variance  
• Gradient checkpointing + mixed precision to fit large U-Nets  

Pitfalls  
• Mis-tuned noise schedule → vanishing/exploding score magnitudes  
• Insufficient corrector steps → blurry outputs  
• Overlarge batch sizes without appropriate $ \lambda(t) $ scaling → instability  

---