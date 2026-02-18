## 5 Advanced & Sequential VAEs  

### 5.1 Variational Recurrent Autoencoder (VRAE)

#### 5.1.1 Definition  
VRAE extends VAEs to sequential data by integrating latent variables with recurrent neural architectures (RNN, GRU, LSTM, Transformer-XL).

#### 5.1.2 Generative Process  

Global-latent variant (simplest):  
1. $$z \sim p(z)=\mathcal N(0,I)$$  
2. For $t=1\!:\!T$:  
 a. $$h_t = f_\theta(h_{t-1},x_{t-1},z)$$ (RNN cell)  
 b. $$x_t \sim p_\theta(x_t\mid h_t)$$  

Sequential-latent (VRNN, SRNN) :  
1. $$z_1 \sim p_\theta(z_1)$$  
2. For $t=1\!:\!T$:  
 a. $$h_{t-1}$$ summarizes $x_{<t},z_{<t}$  
 b. $$z_t \sim p_\theta(z_t \mid h_{t-1})$$  
 c. $$x_t \sim p_\theta(x_t \mid z_t,h_{t-1})$$  
 d. $$h_t = f_\theta(h_{t-1},x_t,z_t)$$  

Assume $$p_\theta(z_t\mid h_{t-1})=\mathcal N(\mu_t^p,\operatorname{diag}(\sigma_t^{p\,2}))$$ and similarly for emissions.

#### 5.1.3 Inference Network  

Global-latent: $$q_\phi(z\mid x_{1:T})=\mathcal N(\mu_\phi(x_{1:T}),\operatorname{diag}(\sigma_\phi^2(x_{1:T})))$$ via bidirectional RNN/Transformer.

Sequential-latent:  
$$q_\phi(z_t\mid x_{\le t},h_{t-1})=\mathcal N(\mu_t^q,\operatorname{diag}(\sigma_t^{q\,2}))$$  
where $$h_{t-1}$$ is shared with the generative model (amortized inference).

#### 5.1.4 ELBO  

Global:  
$$
\mathcal L = \mathbb E_{q_\phi(z\mid x_{1:T})}\!\Big[\sum_{t=1}^{T}\log p_\theta(x_t\mid z,x_{<t})\Big] 
- D_{KL}\!\big(q_\phi(z\mid x_{1:T})\Vert p(z)\big)
$$  

Sequential:  
$$
\mathcal L = \sum_{t=1}^{T}\!\Big(
\mathbb E_{q_\phi}\![\log p_\theta(x_t\mid z_t,h_{t-1})]
- D_{KL}\!\big(q_\phi(z_t\mid x_{\le t},h_{t-1})\Vert p_\theta(z_t\mid h_{t-1})\big)
\Big)
$$  

#### 5.1.5 Training Details  

• Reparameterize each Gaussian $$z_t=\mu_t^q+\sigma_t^q\odot\epsilon_t,\;\epsilon_t\sim\mathcal N(0,I)$$.  
• Use teacher forcing: feed ground-truth $$x_t$$ to compute $$h_t$$ (stabilizes gradients).  
• KL‐annealing or free-bits to avoid posterior collapse.  
• Clip RNN gradients, adopt Adam with sequence-length-scaled LR.  

#### 5.1.6 Applications  

• Speech and music generation.  
• Language modeling (latent topic or sentence representations).  
• Human motion synthesis.  

---

### 5.2 Vector-Quantized VAE (VQ-VAE)

#### 5.2.1 Definition  
VQ-VAE replaces continuous latents with discrete codewords using vector quantization; back-propagation uses a straight-through estimator.

#### 5.2.2 Components  

• Encoder: $$z_e(x) \in \mathbb R^{h\times w\times d}$$ (conv or transformer).  
• Codebook: $$E=\{e_k\}_{k=1}^{K},\; e_k\in\mathbb R^{d}$$.  
• Quantization: $$k^\ast=\arg\min_k\lVert z_e(x)-e_k\rVert_2$$, $$z_q=e_{k^\ast}$$ (nearest-neighbor).  
• Decoder: $$p_\theta(x\mid z_q)$$ reconstructs input.  

#### 5.2.3 Loss Function  

$$
\mathcal L = 
\underbrace{\log p_\theta(x\mid z_q)}_{\text{Reconstruction}} +
\underbrace{\lVert \operatorname{sg}[z_e(x)]-e_{k^\ast}\rVert_2^2}_{\text{Codebook}} +
\underbrace{\beta\lVert z_e(x)-\operatorname{sg}[e_{k^\ast}]\rVert_2^2}_{\text{Commitment}}
$$

• $$\operatorname{sg}[\cdot]$$ denotes stop-gradient.  
• $$\beta$$ (typically 0.25) forces encoder outputs to commit to the chosen code.

#### 5.2.4 Gradient Flow (Straight-Through)  

Forward uses $$z_q$$; backward uses $$\partial z_q/\partial z_e \approx 1$$, enabling end-to-end training.

#### 5.2.5 Codebook Update (EMA variant)  

$$
N_k^{(t)} = \gamma N_k^{(t-1)} + (1-\gamma)\sum_i\mathbf 1[k^\ast_i = k]
$$  
$$
m_k^{(t)} = \gamma m_k^{(t-1)} + (1-\gamma)\sum_i\mathbf 1[k^\ast_i = k]\,z_e^{(i)}
$$  
$$
e_k = \frac{m_k^{(t)}}{N_k^{(t)}}
$$  
with decay $$\gamma\approx0.99$$.

#### 5.2.6 Prior Learning  

Train an autoregressive prior $$p_\psi(k_{1:h,w})$$ on code indices (PixelCNN, Transformer, GPT). Generation pipeline:  
sample $$k\sim p_\psi$$, reconstruct via decoder.

#### 5.2.7 Applications  

• High-fidelity image/audio compression.  
• Neural speech codec (EnCodec, SoundStream).  
• Non-autoregressive machine translation via discrete tokens.  

---

### 5.3 Hierarchical / Ladder VAEs

#### 5.3.1 Definition  
Hierarchy introduces multiple stochastic layers $$z_L, z_{L-1},\dots,z_1$$ to capture multi-scale variability; the Ladder VAE couples bottom-up inference with top-down generative priors.

#### 5.3.2 Generative Model  

$$
p_\theta(x,z_{1:L}) = p_\theta(z_L)\prod_{\ell=L-1}^{1} p_\theta(z_\ell \mid z_{\ell+1})\,p_\theta(x\mid z_1)
$$  

Common choice: $$p_\theta(z_\ell\mid z_{\ell+1})=\mathcal N(\mu_\ell(z_{\ell+1}),\operatorname{diag}(\sigma_\ell^2(z_{\ell+1})))$$.

#### 5.3.3 Inference Network (Ladder)  

Bottom-up pass provides data-dependent statistics $$\hat\mu_\ell,\hat\sigma_\ell$$.  
Top-down prior provides $$\mu_\ell,\sigma_\ell$$.  
Combine via precision-weighted fusion:  

$$
\sigma_{\ell}^{q\, -2} = \hat\sigma_\ell^{-2} + \sigma_\ell^{-2},\quad
\mu_\ell^{q} = \sigma_\ell^{q\,2}(\hat\sigma_\ell^{-2}\hat\mu_\ell + \sigma_\ell^{-2}\mu_\ell)
$$  

Sample $$z_\ell\sim q_\phi(z_\ell\mid \cdot)=\mathcal N(\mu_\ell^{q},\operatorname{diag}(\sigma_\ell^{q\,2}))$$ (reparameterized).

#### 5.3.4 ELBO  

$$
\mathcal L =
\mathbb E_{q_\phi(z_{1:L}\mid x)}[\log p_\theta(x\mid z_1)]
-
\sum_{\ell=1}^{L} D_{KL}\big(q_\phi(z_\ell\mid z_{<\ell},x)\Vert p_\theta(z_\ell\mid z_{>\ell})\big)
$$  

Hierarchical importance-weighted bound (HIWAE) tightens the ELBO with $K$ particles.

#### 5.3.5 Training Guidelines  

• KL warm-up per layer (top layers converge slower).  
• Skip connections in decoder to mitigate information bottleneck.  
• Use residual blocks / attention to model $$\mu_\ell(\cdot)$$, $$\sigma_\ell(\cdot)$$.  
• Latent sizes: coarse top $$d_{z_L}\ll d_{z_1}$$ (e.g., 32→256).  

#### 5.3.6 Variants  

• NVAE: depth-wise separable convs, batch-wise group-latent sampling, NLL ≈ SOTA.  
• VLAE: factorized likelihood, ladder inference only.  
• Hierarchical Flow-VAE: normalizing flows between layers for expressive posteriors.  

#### 5.3.7 Applications  

• High-resolution image generation.  
• Scalable speech synthesis (top-level phoneme rhythm, lower-level waveform).  
• Continual learning: allocate new latent layers for novel concepts.

---

### 5.4 Implementation Checklist (All Advanced VAEs)

• Data pipelines: shuffled sequences, chunking for long context.  
• Latent configuration: continuous vs discrete, depth $$L$$, codebook size $$K$$.  
• Objective scheduling: KL/commitment coefficients, KL-free-bits, cosine annealing LR.  
• Optimizer: AdamW with gradient clipping (1–5).  
• Evaluation:  
 – NLL, bpd, IWAE-$K$.  
 – Sample quality: FID, MS-SSIM, MUSHRA (audio).  
 – Latent diagnostics: mutual information, traversals, clustering.