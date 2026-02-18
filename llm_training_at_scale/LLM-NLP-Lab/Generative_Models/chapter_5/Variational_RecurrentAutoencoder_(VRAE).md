# Variational Recurrent Autoencoder (VRAE)

## Definition

A **Variational Recurrent Autoencoder (VRAE)** is a generative model that combines the principles of Variational Autoencoders (VAE) with Recurrent Neural Networks (RNNs) to model sequential data. VRAEs learn a probabilistic latent representation of sequences, enabling the generation and reconstruction of variable-length sequences.

---

## Mathematical Formulation

### 1. Problem Setup

Given a dataset of sequences $\mathcal{D} = \{ \mathbf{x}^{(i)} \}_{i=1}^N$, where each sequence $\mathbf{x} = (x_1, x_2, ..., x_T)$, the goal is to model the data distribution $p(\mathbf{x})$ via a latent variable model.

### 2. Latent Variable Model

Introduce a latent variable $\mathbf{z}$ for each sequence:

$$
p(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p(\mathbf{x} | \mathbf{z})
$$

- $p(\mathbf{z})$: Prior over latent variables, typically $\mathcal{N}(0, I)$.
- $p(\mathbf{x} | \mathbf{z})$: Likelihood of sequence given latent code.

### 3. Variational Inference

Since $p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ is intractable, introduce an approximate posterior $q(\mathbf{z} | \mathbf{x})$.

#### Evidence Lower Bound (ELBO):

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})} [\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

- $\mathbb{E}_{q(\mathbf{z}|\mathbf{x})} [\log p(\mathbf{x}|\mathbf{z})]$: Reconstruction term.
- $D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$: Regularization term.

---

## Model Architecture

### 1. Encoder (Recognition Model)

- **Input:** Sequence $\mathbf{x} = (x_1, ..., x_T)$.
- **Architecture:** RNN (e.g., LSTM, GRU) processes the sequence.
- **Output:** Parameters of approximate posterior $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$.

#### Encoder RNN:

At each time step $t$:

$$
\mathbf{h}_t = \mathrm{RNN}_{enc}(x_t, \mathbf{h}_{t-1})
$$

After the final step $T$:

$$
\boldsymbol{\mu} = W_\mu \mathbf{h}_T + b_\mu \\
\log \boldsymbol{\sigma} = W_\sigma \mathbf{h}_T + b_\sigma
$$

### 2. Latent Variable Sampling

Sample $\mathbf{z}$ using the reparameterization trick:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

### 3. Decoder (Generative Model)

- **Input:** Latent variable $\mathbf{z}$.
- **Architecture:** RNN initialized with $\mathbf{z}$ as the initial hidden state or concatenated to inputs.
- **Output:** Reconstructed sequence $\hat{\mathbf{x}} = (\hat{x}_1, ..., \hat{x}_T)$.

#### Decoder RNN:

At each time step $t$:

$$
\mathbf{h}_t^{dec} = \mathrm{RNN}_{dec}(\hat{x}_{t-1}, \mathbf{h}_{t-1}^{dec}, \mathbf{z}) \\
\hat{x}_t = f_{out}(\mathbf{h}_t^{dec})
$$

---

## Training Procedure

### 1. Forward Pass

- Encode $\mathbf{x}$ to obtain $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$.
- Sample $\mathbf{z}$.
- Decode $\mathbf{z}$ to reconstruct $\hat{\mathbf{x}}$.

### 2. Loss Function

The loss for a single sequence $\mathbf{x}$:

$$
\mathcal{L}(\mathbf{x}) = -\mathbb{E}_{q(\mathbf{z}|\mathbf{x})} [\log p(\mathbf{x}|\mathbf{z})] + D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

- **Reconstruction Loss:** For continuous data, use MSE; for discrete, use cross-entropy.
- **KL Divergence:** Closed-form for Gaussian posteriors.

### 3. Optimization

- Minimize $\mathcal{L}(\mathbf{x})$ over the dataset using stochastic gradient descent (SGD) or variants (e.g., Adam).
- Gradients are backpropagated through both encoder and decoder RNNs.

---

## Step-by-Step End-to-End Workflow

1. **Data Preprocessing**
   - Tokenize and pad sequences to uniform length (if batching).
   - Normalize/standardize input features.

2. **Encoder Forward Pass**
   - Pass input sequence through encoder RNN.
   - Obtain final hidden state $\mathbf{h}_T$.
   - Compute $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

3. **Latent Sampling**
   - Sample $\mathbf{z}$ using reparameterization.

4. **Decoder Forward Pass**
   - Initialize decoder RNN with $\mathbf{z}$.
   - Generate output sequence step-by-step.

5. **Loss Computation**
   - Compute reconstruction loss between $\mathbf{x}$ and $\hat{\mathbf{x}}$.
   - Compute KL divergence between $q(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z})$.

6. **Backpropagation**
   - Compute gradients of total loss w.r.t. all parameters.
   - Update parameters using optimizer.

7. **Inference/Generation**
   - Sample $\mathbf{z} \sim p(\mathbf{z})$.
   - Decode to generate new sequences.

---

## Key Aspects and Considerations

- **Sequence Lengths:** VRAE can handle variable-length sequences via RNNs.
- **Posterior Collapse:** KL annealing or free bits can be used to prevent the model from ignoring $\mathbf{z}$.
- **Conditional VRAE:** Conditioning on additional information (e.g., class labels) is possible by concatenating to inputs or latent code.
- **Extensions:** Hierarchical VRAEs, bidirectional RNNs in encoder, attention mechanisms.

---

## Summary Table

| Component         | Function                                      | Mathematical Formulation                          |
|-------------------|-----------------------------------------------|---------------------------------------------------|
| Encoder RNN       | Maps $\mathbf{x}$ to $\boldsymbol{\mu}, \boldsymbol{\sigma}$ | $\mathbf{h}_T \rightarrow (\boldsymbol{\mu}, \boldsymbol{\sigma})$ |
| Latent Sampling   | Samples $\mathbf{z}$                          | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ |
| Decoder RNN       | Reconstructs/generates sequence               | $\mathbf{z} \rightarrow \hat{\mathbf{x}}$         |
| Loss              | Trains model via ELBO                         | $\mathcal{L}(\mathbf{x})$ as above                |

---

## References

- Fabius, O., & van Amersfoort, J. R. (2014). Variational Recurrent Auto-Encoders. arXiv:1412.6581
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114

---