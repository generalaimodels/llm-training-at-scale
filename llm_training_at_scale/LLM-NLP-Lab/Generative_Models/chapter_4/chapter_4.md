---

# Chapter 4 Core Variational Autoencoders

---

## 1. Basic Variational Autoencoder (VAE)

### Definition

A **Variational Autoencoder (VAE)** is a generative model that learns a probabilistic mapping from data space $x$ to a latent space $z$, enabling both efficient data encoding and generation. VAEs optimize a variational lower bound on the data likelihood, using neural networks to parameterize both the encoder (inference model) and decoder (generative model).

---

### Mathematical Formulation

#### Generative Process

- Latent variable prior: $p(z)$ (commonly $p(z) = \mathcal{N}(0, I)$)
- Likelihood: $p_\theta(x|z)$ (decoder, parameterized by $\theta$)
- Marginal likelihood: $p_\theta(x) = \int p_\theta(x|z) p(z) dz$

#### Inference Model

- Approximate posterior: $q_\phi(z|x)$ (encoder, parameterized by $\phi$)

#### Evidence Lower Bound (ELBO)

The log-likelihood $\log p_\theta(x)$ is intractable. The ELBO is:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \Vert p(z)) = \mathcal{L}(\theta, \phi; x)
$$

where:
- $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$: **Reconstruction term** (expected log-likelihood)
- $D_{KL}(q_\phi(z|x) \Vert p(z))$: **Regularization term** (Kullback-Leibler divergence)

---

### Step-by-Step Explanation

#### 1. **Data Encoding (Inference/Recognition Model)**

- Input $x$ is mapped to parameters of $q_\phi(z|x)$, typically a diagonal Gaussian:
  $$
  q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))
  $$
- $\mu_\phi(x)$ and $\sigma^2_\phi(x)$ are outputs of the encoder neural network.

#### 2. **Latent Sampling (Reparameterization Trick)**

- To enable backpropagation through stochastic nodes, use:
  $$
  z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
  $$

#### 3. **Data Decoding (Generative Model)**

- Sampled $z$ is passed through the decoder to parameterize $p_\theta(x|z)$, e.g., for images:
  $$
  p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)
  $$
- $\mu_\theta(z)$ is the output of the decoder neural network.

#### 4. **Loss Computation**

- **Reconstruction Loss**: Measures how well the decoder reconstructs $x$ from $z$.
- **KL Divergence**: Regularizes $q_\phi(z|x)$ to be close to the prior $p(z)$.

#### 5. **Optimization**

- Jointly optimize $\theta$ and $\phi$ to maximize the ELBO over the dataset:
  $$
  \max_{\theta, \phi} \sum_{i=1}^N \mathcal{L}(\theta, \phi; x^{(i)})
  $$

---

## 2. $\beta$-VAE

### Definition

A **$\beta$-VAE** is a variant of the VAE that introduces a hyperparameter $\beta$ to control the trade-off between the reconstruction and regularization terms in the ELBO, promoting disentangled latent representations.

---

### Mathematical Formulation

The $\beta$-VAE objective is:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) \Vert p(z))
$$

where $\beta \geq 1$.

---

### Step-by-Step Explanation

#### 1. **Motivation**

- Standard VAE ($\beta=1$) often learns entangled latent factors.
- Increasing $\beta$ ($\beta > 1$) enforces stronger regularization, encouraging the model to learn statistically independent latent factors (disentanglement).

#### 2. **Effect of $\beta$**

- **$\beta = 1$**: Standard VAE.
- **$\beta > 1$**: Increased pressure for $q_\phi(z|x)$ to match $p(z)$, often at the cost of reconstruction fidelity.
- **$\beta < 1$**: Weaker regularization, better reconstruction, less disentanglement.

#### 3. **Training**

- Identical to VAE, except the KL term is scaled by $\beta$.
- The choice of $\beta$ is critical and typically determined via validation.

#### 4. **Disentanglement**

- Empirically, higher $\beta$ values lead to latent variables that correspond to independent generative factors in the data.

---

## 3. Conditional VAE (CVAE)

### Definition

A **Conditional Variational Autoencoder (CVAE)** extends the VAE to model conditional distributions $p(x|y)$, where $y$ is an observed variable (e.g., class label, attribute), enabling conditional data generation.

---

### Mathematical Formulation

#### Generative Process

- Prior: $p(z|y)$ (often $p(z|y) = \mathcal{N}(0, I)$)
- Likelihood: $p_\theta(x|z, y)$

#### Inference Model

- Approximate posterior: $q_\phi(z|x, y)$

#### Conditional ELBO

$$
\log p_\theta(x|y) \geq \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - D_{KL}(q_\phi(z|x, y) \Vert p(z|y))
$$

---

### Step-by-Step Explanation

#### 1. **Data Encoding (Conditional Inference Model)**

- Input $(x, y)$ is mapped to parameters of $q_\phi(z|x, y)$:
  $$
  q_\phi(z|x, y) = \mathcal{N}(z; \mu_\phi(x, y), \mathrm{diag}(\sigma^2_\phi(x, y)))
  $$

#### 2. **Latent Sampling (Reparameterization Trick)**

- Sample $z$ as:
  $$
  z = \mu_\phi(x, y) + \sigma_\phi(x, y) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
  $$

#### 3. **Data Decoding (Conditional Generative Model)**

- $z$ and $y$ are passed to the decoder to parameterize $p_\theta(x|z, y)$.

#### 4. **Loss Computation**

- **Conditional Reconstruction Loss**: Measures how well the decoder reconstructs $x$ from $(z, y)$.
- **Conditional KL Divergence**: Regularizes $q_\phi(z|x, y)$ to be close to $p(z|y)$.

#### 5. **Optimization**

- Jointly optimize $\theta$ and $\phi$ to maximize the conditional ELBO over the dataset:
  $$
  \max_{\theta, \phi} \sum_{i=1}^N \mathcal{L}_{\text{CVAE}}(\theta, \phi; x^{(i)}, y^{(i)})
  $$

#### 6. **Conditional Generation**

- At inference, sample $z \sim p(z|y^*)$, then generate $x^* \sim p_\theta(x|z, y^*)$ for a desired condition $y^*$.

---

## Summary Table

| Model      | Encoder $q_\phi$ | Decoder $p_\theta$ | Loss Function |
|------------|------------------|--------------------|---------------|
| VAE        | $q_\phi(z|x)$    | $p_\theta(x|z)$    | $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \Vert p(z))$ |
| $\beta$-VAE| $q_\phi(z|x)$    | $p_\theta(x|z)$    | $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) \Vert p(z))$ |
| CVAE       | $q_\phi(z|x, y)$ | $p_\theta(x|z, y)$ | $\mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - D_{KL}(q_\phi(z|x, y) \Vert p(z|y))$ |

---

## End-to-End Considerations

- **Data Preprocessing**: Normalize $x$, encode $y$ (if categorical, use one-hot or embedding).
- **Model Architecture**: Encoder/decoder are typically deep neural networks (MLPs, CNNs, RNNs depending on data modality).
- **Latent Dimensionality**: Chosen based on data complexity and desired generative capacity.
- **Training**: Use stochastic gradient descent, with mini-batch sampling and reparameterization for efficient backpropagation.
- **Evaluation**: Assess via ELBO, reconstruction quality, sample diversity, and (for $\beta$-VAE) disentanglement metrics.

---