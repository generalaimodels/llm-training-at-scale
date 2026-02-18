
# Chapter 5 Advanced & Sequential VAEs

---

## 1. Variational Recurrent Autoencoder (VRAE)

### Definition

A **Variational Recurrent Autoencoder (VRAE)** is a sequential extension of the VAE framework, designed to model temporal dependencies in sequential data (e.g., time series, text, audio) by integrating recurrent neural networks (RNNs) into the encoder and/or decoder.

---

### Mathematical Formulation

#### Sequential Data

- Given a sequence $x_{1:T} = (x_1, x_2, ..., x_T)$.
- Latent variables $z_{1:T} = (z_1, z_2, ..., z_T)$.

#### Generative Model

- Prior: $p(z_{1:T}) = \prod_{t=1}^T p(z_t|z_{<t})$
- Likelihood: $p_\theta(x_{1:T}|z_{1:T}) = \prod_{t=1}^T p_\theta(x_t|z_{\leq t}, x_{<t})$

#### Inference Model

- Approximate posterior: $q_\phi(z_{1:T}|x_{1:T}) = \prod_{t=1}^T q_\phi(z_t|z_{<t}, x_{\leq t})$

#### ELBO for Sequences

$$
\mathcal{L}_{\text{VRAE}} = \mathbb{E}_{q_\phi(z_{1:T}|x_{1:T})} \left[ \sum_{t=1}^T \log p_\theta(x_t|z_{\leq t}, x_{<t}) \right] - \sum_{t=1}^T D_{KL}(q_\phi(z_t|z_{<t}, x_{\leq t}) \Vert p(z_t|z_{<t}))
$$

---

### Step-by-Step Explanation

#### 1. **Encoder (Inference Network)**

- At each time step $t$, the encoder RNN processes $x_{\leq t}$ and $z_{<t}$ to output parameters $(\mu_t, \sigma_t^2)$ for $q_\phi(z_t|z_{<t}, x_{\leq t})$.

#### 2. **Latent Sampling**

- Sample $z_t$ using the reparameterization trick:
  $$
  z_t = \mu_t + \sigma_t \odot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
  $$

#### 3. **Decoder (Generative Network)**

- The decoder RNN receives $z_{\leq t}$ and $x_{<t}$ to generate $x_t$ via $p_\theta(x_t|z_{\leq t}, x_{<t})$.

#### 4. **Loss Computation**

- For each time step, compute the reconstruction loss and KL divergence.
- Sum over all time steps for the sequence.

#### 5. **Optimization**

- Maximize the sequence ELBO over the dataset using stochastic gradient descent.

#### 6. **Applications**

- Sequence generation, time series modeling, sequential data imputation, speech synthesis.

---

## 2. Vector-Quantized VAE (VQ-VAE)

### Definition

A **Vector-Quantized VAE (VQ-VAE)** is a discrete latent variable VAE that replaces the continuous latent space with a finite set of learnable embedding vectors (codebook), enabling discrete representation learning and improved generative modeling for domains such as images, audio, and text.

---

### Mathematical Formulation

#### Latent Space

- Codebook: $e = \{e_k\}_{k=1}^K$, $e_k \in \mathbb{R}^D$
- Encoder output: $z_e(x) \in \mathbb{R}^D$

#### Quantization

- For each input $x$, encoder produces $z_e(x)$.
- Quantize $z_e(x)$ to nearest codebook entry:
  $$
  k^* = \arg\min_k \|z_e(x) - e_k\|_2
  $$
  $$
  z_q(x) = e_{k^*}
  $$

#### Decoder

- $z_q(x)$ is input to the decoder to reconstruct $x$.

#### Loss Function

$$
\mathcal{L}_{\text{VQ-VAE}} = \log p_\theta(x|z_q(x)) + \| \text{sg}[z_e(x)] - e_{k^*} \|_2^2 + \beta \| z_e(x) - \text{sg}[e_{k^*}] \|_2^2
$$

- $\text{sg}[\cdot]$ denotes the stop-gradient operator.
- First term: reconstruction loss.
- Second term: codebook loss (updates codebook vectors).
- Third term: commitment loss (encourages encoder outputs to commit to codebook entries).

---

### Step-by-Step Explanation

#### 1. **Encoder**

- Maps input $x$ to continuous latent $z_e(x)$.

#### 2. **Vector Quantization**

- Replace $z_e(x)$ with nearest codebook vector $e_{k^*}$.

#### 3. **Decoder**

- Receives $z_q(x) = e_{k^*}$ and reconstructs $x$.

#### 4. **Loss Computation**

- **Reconstruction Loss**: Negative log-likelihood of $x$ given $z_q(x)$.
- **Codebook Loss**: Moves $e_{k^*}$ towards $z_e(x)$.
- **Commitment Loss**: Penalizes $z_e(x)$ for straying from $e_{k^*}$.

#### 5. **Codebook Update**

- Codebook vectors are updated via the codebook loss, often using exponential moving averages for stability.

#### 6. **Optimization**

- Jointly optimize encoder, decoder, and codebook using stochastic gradient descent.

#### 7. **Applications**

- Discrete representation learning, speech synthesis (e.g., WaveNet), image compression, neural discrete generative modeling.

---

## 3. Hierarchical / Ladder VAE Variants

### Definition

**Hierarchical VAEs** (including Ladder VAEs) extend the VAE framework by introducing multiple layers of latent variables, enabling the model to capture complex, multi-scale data structure and dependencies.

---

### Mathematical Formulation

#### Latent Variables

- $L$ layers of latent variables: $z = \{z_1, z_2, ..., z_L\}$

#### Generative Model

- Top-down generative process:
  $$
  p_\theta(x, z_1, ..., z_L) = p_\theta(x|z_1) \prod_{l=1}^L p_\theta(z_l|z_{l+1})
  $$
  (with $z_{L+1}$ undefined, so $p_\theta(z_L)$ is the prior for the top layer)

#### Inference Model

- Bottom-up inference process:
  $$
  q_\phi(z_1, ..., z_L|x) = q_\phi(z_1|x) \prod_{l=2}^L q_\phi(z_l|z_{l-1}, ..., z_1, x)
  $$

#### ELBO

$$
\mathcal{L}_{\text{Hierarchical}} = \mathbb{E}_{q_\phi(z_{1:L}|x)}[\log p_\theta(x|z_1)] - \sum_{l=1}^L \mathbb{E}_{q_\phi(z_{l+1:L}|x)} \left[ D_{KL}(q_\phi(z_l|z_{<l}, x) \Vert p_\theta(z_l|z_{>l})) \right]
$$

- $z_{<l}$: all lower layers, $z_{>l}$: all higher layers.

---

### Step-by-Step Explanation

#### 1. **Encoder (Inference Network)**

- Processes $x$ through a series of neural network layers to infer parameters for each $q_\phi(z_l|z_{<l}, x)$.
- Each latent variable $z_l$ is sampled using the reparameterization trick.

#### 2. **Decoder (Generative Network)**

- Top-down: $z_L \rightarrow z_{L-1} \rightarrow ... \rightarrow z_1 \rightarrow x$.
- Each $p_\theta(z_l|z_{l+1})$ and $p_\theta(x|z_1)$ is parameterized by neural networks.

#### 3. **Loss Computation**

- For each layer, compute the KL divergence between the approximate posterior and the generative prior.
- Compute the reconstruction loss at the bottom layer.

#### 4. **Optimization**

- Maximize the hierarchical ELBO over the dataset.

#### 5. **Ladder VAE Specifics**

- **Ladder VAEs** introduce skip connections and combine bottom-up and top-down information at each layer, improving inference and generative performance.

#### 6. **Applications**

- Modeling complex data distributions, multi-scale generative modeling, improved representation learning, semi-supervised learning.

---

## Summary Table

| Model         | Latent Structure         | Encoder/Decoder Type         | Loss Function |
|---------------|-------------------------|-----------------------------|---------------|
| VRAE          | Sequence of $z_t$       | RNN-based                   | Sequence ELBO |
| VQ-VAE        | Discrete codebook       | CNN/MLP + quantization      | VQ-VAE loss   |
| Hierarchical  | Multi-layer $z_{1:L}$   | Stacked neural networks     | Hierarchical ELBO |

---

## End-to-End Considerations

- **Data Preprocessing**: Sequence alignment for VRAE, normalization for VQ-VAE, hierarchical feature extraction for Ladder VAE.
- **Model Architecture**: RNNs for VRAE, CNNs/MLPs for VQ-VAE, deep/stacked networks for hierarchical VAEs.
- **Latent Dimensionality**: Chosen per layer or per codebook size.
- **Training**: Use stochastic gradient descent, reparameterization, and (for VQ-VAE) codebook updates.
- **Evaluation**: ELBO, reconstruction quality, sample diversity, codebook utilization (VQ-VAE), hierarchical disentanglement.

---