# Hierarchical / Ladder Variational Autoencoder (VAE) Variants

## Definition

**Hierarchical VAEs** (also known as Ladder VAEs) are generative models that extend the standard VAE by introducing multiple layers of latent variables, organized in a hierarchy. This structure enables the model to capture complex, multi-scale data dependencies and disentangle factors of variation at different abstraction levels.

---

## Mathematical Formulation

### 1. Latent Variable Hierarchy

Let $\mathbf{x}$ denote the observed data and $\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_L$ denote $L$ layers of latent variables, with $\mathbf{z}_1$ being the lowest (closest to data) and $\mathbf{z}_L$ the highest (most abstract).

#### Generative Model

The joint distribution is factorized as:

$$
p(\mathbf{x}, \mathbf{z}_1, ..., \mathbf{z}_L) = p(\mathbf{x} | \mathbf{z}_1) \prod_{l=1}^{L} p(\mathbf{z}_l | \mathbf{z}_{l+1})
$$

where by convention, $\mathbf{z}_{L+1}$ is omitted or set to a fixed prior (e.g., standard normal).

- $p(\mathbf{z}_L)$ is the top-level prior, typically $\mathcal{N}(0, I)$.
- $p(\mathbf{z}_l | \mathbf{z}_{l+1})$ are conditional priors, parameterized by neural networks.

#### Inference Model (Encoder)

The variational posterior is factorized as:

$$
q(\mathbf{z}_1, ..., \mathbf{z}_L | \mathbf{x}) = q(\mathbf{z}_L | \mathbf{x}) \prod_{l=1}^{L-1} q(\mathbf{z}_l | \mathbf{z}_{l+1}, \mathbf{x})
$$

Each $q(\mathbf{z}_l | \cdot)$ is parameterized by neural networks.

### 2. Evidence Lower Bound (ELBO)

The ELBO for hierarchical VAEs is:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})} \left[ \log p(\mathbf{x}|\mathbf{z}_1) + \sum_{l=1}^{L} \log p(\mathbf{z}_l|\mathbf{z}_{l+1}) - \sum_{l=1}^{L} \log q(\mathbf{z}_l|\mathbf{z}_{l+1}, \mathbf{x}) \right]
$$

Or, equivalently:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})} \left[ \log p(\mathbf{x}|\mathbf{z}_1) \right] - \sum_{l=1}^{L} \mathbb{E}_{q(\mathbf{z}_{l+1:L}|\mathbf{x})} \left[ D_{KL}\left( q(\mathbf{z}_l|\mathbf{z}_{l+1}, \mathbf{x}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}) \right) \right]
$$

---

## Step-by-Step Explanation

### 1. Data Preprocessing

- Normalize and preprocess input data $\mathbf{x}$.
- For images: scale pixel values.
- For text/audio: tokenize and encode as needed.

### 2. Model Architecture

#### a. Encoder (Inference Network)

- **Bottom-up pass:** 
  - Input $\mathbf{x}$ is processed through a series of neural network layers (e.g., CNNs, MLPs).
  - At each layer $l$, compute parameters $(\boldsymbol{\mu}_l, \boldsymbol{\sigma}_l)$ for $q(\mathbf{z}_l | \mathbf{z}_{l+1}, \mathbf{x})$.
  - For the top layer, $q(\mathbf{z}_L | \mathbf{x})$.

- **Sampling:**
  - Sample $\mathbf{z}_L \sim q(\mathbf{z}_L | \mathbf{x})$.
  - For $l = L-1$ down to $1$, sample $\mathbf{z}_l \sim q(\mathbf{z}_l | \mathbf{z}_{l+1}, \mathbf{x})$.

#### b. Decoder (Generative Network)

- **Top-down pass:**
  - Sample $\mathbf{z}_L \sim p(\mathbf{z}_L)$.
  - For $l = L-1$ down to $1$, sample $\mathbf{z}_l \sim p(\mathbf{z}_l | \mathbf{z}_{l+1})$.
  - Finally, generate $\mathbf{x} \sim p(\mathbf{x} | \mathbf{z}_1)$.

- **Parameterization:**
  - Each $p(\mathbf{z}_l | \mathbf{z}_{l+1})$ and $p(\mathbf{x} | \mathbf{z}_1)$ is parameterized by neural networks.

### 3. Training Objective

- **Reconstruction Term:**
  $$
  \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})} [\log p(\mathbf{x}|\mathbf{z}_1)]
  $$
- **KL Divergence Terms:**
  $$
  \sum_{l=1}^{L} \mathbb{E}_{q(\mathbf{z}_{l+1:L}|\mathbf{x})} \left[ D_{KL}\left( q(\mathbf{z}_l|\mathbf{z}_{l+1}, \mathbf{x}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}) \right) \right]
  $$
- **Total Loss:**
  $$
  \mathcal{L} = -\mathcal{L}_{\text{ELBO}}
  $$

### 4. Optimization

- Use stochastic gradient descent (SGD) or Adam to minimize $\mathcal{L}$.
- Use the reparameterization trick for all Gaussian latent variables to enable backpropagation.

### 5. Inference and Generation

- **Encoding:** Given $\mathbf{x}$, sample $\mathbf{z}_{1:L}$ from the variational posterior.
- **Decoding:** Given sampled $\mathbf{z}_{1:L}$, generate $\hat{\mathbf{x}}$ via the generative model.
- **Sampling:** For unconditional generation, sample $\mathbf{z}_L \sim p(\mathbf{z}_L)$, then recursively sample lower layers and finally $\mathbf{x}$.

---

## Ladder VAE: Special Case

**Ladder VAE** (Sønderby et al., 2016) is a specific hierarchical VAE variant with a *ladder-like* structure in the inference network, combining bottom-up and top-down information at each layer.

### Ladder Inference Network

At each layer $l$:

$$
q(\mathbf{z}_l | \mathbf{z}_{l+1}, \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_l, \boldsymbol{\sigma}_l^2)
$$

where $\boldsymbol{\mu}_l$ and $\boldsymbol{\sigma}_l$ are computed as a function of both bottom-up (from $\mathbf{x}$) and top-down (from $\mathbf{z}_{l+1}$) signals, typically via neural network fusion (e.g., concatenation, addition).

---

## Key Aspects and Considerations

- **Expressivity:** Hierarchical structure enables modeling of complex, multi-scale data distributions.
- **Disentanglement:** Different layers can capture factors of variation at different abstraction levels.
- **Posterior Collapse:** Hierarchical VAEs can mitigate posterior collapse by distributing information across multiple latent layers.
- **Parameterization:** Both conditional priors and posteriors are parameterized by neural networks, often with skip connections or fusion mechanisms.
- **Extensions:** Hierarchical VAEs can be combined with discrete latents, attention, or autoregressive decoders for further expressivity.

---

## Summary Table

| Component         | Function                                      | Mathematical Formulation                          |
|-------------------|-----------------------------------------------|---------------------------------------------------|
| Generative Model  | Top-down sampling of latents and data         | $p(\mathbf{x}, \mathbf{z}_{1:L})$ as above        |
| Inference Model   | Bottom-up encoding of latents                 | $q(\mathbf{z}_{1:L}|\mathbf{x})$ as above         |
| ELBO              | Training objective                            | $\mathcal{L}_{\text{ELBO}}$ as above              |
| Optimization      | End-to-end training via SGD/Adam              | $\min_\theta \mathcal{L}$                         |

---

## References

- Sønderby, C. K., Raiko, T., Maaløe, L., Sřnderby, S. K., & Winther, O. (2016). Ladder Variational Autoencoders. arXiv:1602.02282
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114
- Vahdat, A., & Kautz, J. (2020). NVAE: A Deep Hierarchical Variational Autoencoder. arXiv:2007.03898

---