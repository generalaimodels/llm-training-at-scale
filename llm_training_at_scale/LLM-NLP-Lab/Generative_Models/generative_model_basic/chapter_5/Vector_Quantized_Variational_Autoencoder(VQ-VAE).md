# Vector-Quantized Variational Autoencoder (VQ-VAE)

---

## 1  Concise Definition  
A VQ-VAE is a discrete latent variable generative model that replaces the continuous Gaussian latent of a conventional VAE with a learnable finite codebook. An encoder maps the input to a latent vector, which is quantized to its nearest codebook entry; a decoder reconstructs the input from this discrete embedding. Training maximizes a modified evidence lower bound that includes a vector-quantization loss and a commitment loss while gradients are propagated via a straight-through estimator.

---

## 2  Mathematical Formulation  

### 2.1  Components  
* Codebook (embedding dictionary)  
  $$\mathbf{E}=\{\mathbf{e}_k\}_{k=1}^{K},\qquad \mathbf{e}_k\in\mathbb{R}^{d_z}$$

* Encoder  
  $$\mathbf{z}_e = f_\text{enc}(\mathbf{x};\;\theta_e)\in\mathbb{R}^{d_z}$$

* Quantizer (nearest-neighbor)  
  $$k^\star = \arg\min_{k}\|\mathbf{z}_e-\mathbf{e}_k\|_2^2,\qquad  
    \mathbf{z}_q = \text{sg}(\mathbf{e}_{k^\star})$$  
  where $\text{sg}(\cdot)$ denotes stop-gradient.

* Decoder  
  $$\hat{\mathbf{x}} = f_\text{dec}(\mathbf{z}_q;\;\theta_d)$$

### 2.2  Generative Model  
Prior over indices $k$ is categorical with uniform probability or learned via an autoregressive prior $\pi_\psi(k\mid\text{context})$. The likelihood is  
$$p_\theta(\mathbf{x}\mid k) = p_\theta\bigl(\mathbf{x}\mid \mathbf{e}_k\bigr).$$

### 2.3  Objective (ELBO Modification)  
For a minibatch $\{\mathbf{x}^i\}_{i=1}^B$ the loss is  
$$
\mathcal{L} =
\underbrace{\frac{1}{B}\sum_{i=1}^{B} \| \mathbf{x}^i - \hat{\mathbf{x}}^i \|_2^2}_{\text{Reconstruction}}
\;+\;
\underbrace{\frac{1}{B}\sum_{i=1}^{B} \| \text{sg}(\mathbf{z}_e^i) - \mathbf{e}_{k^\star(i)}\|_2^2}_{\text{Codebook}}
\;+\;
\underbrace{\beta \frac{1}{B}\sum_{i=1}^{B} \| \mathbf{z}_e^i - \text{sg}(\mathbf{e}_{k^\star(i)})\|_2^2}_{\text{Commitment}}
$$
where $\beta$ (typically $0.25$) balances encoder commitment.

* **Reconstruction term** corresponds to negative log-likelihood for Gaussian decoder.
* **Codebook term** moves embeddings toward encoder outputs.
* **Commitment term** prevents encoder outputs from growing unbounded.

---

## 3  Gradient Propagation  

### 3.1  Straight-Through Estimator  
During back-propagation the quantizer is treated as identity:  
$$\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{z}_e} \approx
\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{z}_q},
\qquad
\frac{\partial \mathbf{z}_q}{\partial \mathbf{E}} = \mathbf{0}.
$$

### 3.2  Codebook Update Strategies  

#### 3.2.1  Loss-Gradient Method  
Embeddings receive gradients from the codebook term:  
$$\nabla_{\mathbf{e}_k}\; \tfrac{1}{2}\|\text{sg}(\mathbf{z}_e)-\mathbf{e}_k\|_2^2 
= (\mathbf{e}_k-\text{sg}(\mathbf{z}_e))\mathbb{I}[k=k^\star].$$

#### 3.2.2  Exponential Moving Average (EMA)  
Maintain assignment counts $N_k$ and sums $m_k$:  
$$
N_k^{(t)} = \gamma N_k^{(t-1)} + (1-\gamma) n_k^{\text{batch}},\qquad
m_k^{(t)} = \gamma m_k^{(t-1)} + (1-\gamma) \sum_{i\in\mathcal{B}_k}\mathbf{z}_e^i
$$
then  
$$\mathbf{e}_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)}}.$$

---

## 4  Architecture Design  

| Block | Typical Layout (Image) | Parameters |
|-------|------------------------|------------|
| Encoder | $C$ conv layers → residual stacks → $1\times1$ conv | stride for down-sampling, channels $C_e$ |
| Quantizer | reshape feature map to $(H'W',d_z)$ → nearest-neighbor | $K,d_z$ |
| Decoder | transposed conv / PixelCNN / Transformer | channels $C_d$ |
| Prior (optional) | autoregressive over index grid | $\psi$ |

For audio, use causal 1-D conv; for text, transformer encoder & decoder with discrete tokens from codebook.

---

## 5  End-to-End Data Flow  

1. Input $\mathbf{x}$ $\rightarrow$ encoder $\rightarrow$ latent map $\mathbf{z}_e$ (shape $H'\!\times\!W'\!\times\!d_z$).  
2. Flatten to $L=H'W'$ vectors, perform nearest-neighbor search against $\mathbf{E}$ (vectorized $\mathrm{argmin}$).  
3. Obtain quantized map $\mathbf{z}_q$ (indices + embeddings).  
4. Decoder reconstructs $\hat{\mathbf{x}}$.  
5. Compute losses, apply straight-through gradients.  
6. Update $\theta_e,\theta_d$ with Adam; update $\mathbf{E}$ via gradient or EMA.

---

## 6  Prior Modeling & Generation  

### 6.1  Training Prior  
Fit autoregressive model $\pi_\psi$ on index sequences extracted from training data:  
$$\log p_\psi(\mathbf{k}_{1:L}) = \sum_{l=1}^L \log \pi_\psi(k_l \mid \mathbf{k}_{<l}).$$

### 6.2  Sampling  
1. Sample $\mathbf{k}_{1:L}\sim \pi_\psi$.  
2. Look up embeddings to form $\mathbf{z}_q$.  
3. Pass through decoder → synthesized sample.

---

## 7  Evaluation Metrics  

| Metric | Expression | Interpretation |
|--------|------------|----------------|
| Reconstruction MSE | $\frac{1}{N}\|\mathbf{x}-\hat{\mathbf{x}}\|_2^2$ | fidelity |
| Bits-Per-Dim (BPD) | $\frac{-\log_2 p_\theta(\mathbf{x})}{D}$ | generative quality |
| Codebook Perplexity | $\exp\bigl( -\sum_{k} p_k\log p_k \bigr)$ | usage of entries |
| KL to Prior | $\text{KL}(\hat{p}(k) \,\|\, \text{Uniform})$ | collapse detector |

---

## 8  Implementation Details  

* Normalize encoder outputs before quantization to stabilize distances.  
* Vectorize nearest-neighbor with $\mathrm{einsum}$ or Faiss for $K{>}4096$.  
* Use $\beta=0.25$–$0.4$; larger $\beta$ promotes diverse code usage.  
* Maintain codebook on GPU to avoid PCIe bottleneck.  
* Mixed-precision requires $fp32$ master copy of $\mathbf{E}$ to prevent NaN during EMA.  
* For hierarchical VQ-VAE-2 stack two (or more) quantizers at different resolutions; top-level prior conditions lower level.

---

## 9  Extensions  

* **Gumbel-Softmax VQ**: differentiable soft quantization during early epochs.  
* **Residual VQ**: sequentially quantize residual errors for exponential codebook capacity.  
* **Masked Vector Quantization**: apply spatial masks to encourage locality.  
* **Discrete Diffusion Prior**: replace autoregressive prior with discrete diffusion transformer.

---

## 10  Summary Algorithm (Pseudo-Code)

```python
# Forward
z_e = Enc(x)                         # encoder
dist = (z_e**2).sum(-1, keepdim=True) \
     - 2 * z_e @ E.T + (E**2).sum(1) # squared L2
k = dist.argmin(-1)                  # nearest index
z_q = E[k]                           # quantized embed (stop-grad)

# Losses
recon = mse_loss(Dec(z_q), x)
codebook = ((z_q - z_e.detach())**2).mean()
commit = ((z_e - z_q.detach())**2).mean()
loss = recon + codebook + beta * commit

# Backward
loss.backward()          # straight-through for z_q -> z_e
optimizer.step()         # updates θ_e, θ_d
update_codebook(E, z_e, k)  # grad or EMA
```

---