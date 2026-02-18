# Vector-Quantized Variational Autoencoder (VQ-VAE)

## Definition

A **Vector-Quantized Variational Autoencoder (VQ-VAE)** is a generative model that combines the autoencoding framework with discrete latent representations via vector quantization. Unlike standard VAEs that use continuous latent variables, VQ-VAE encodes data into a finite set of learnable embedding vectors, enabling efficient and expressive discrete representations for downstream generative modeling.

---

## Mathematical Formulation

### 1. Model Structure

Given input data $\mathbf{x} \in \mathbb{R}^D$, the VQ-VAE consists of:

- **Encoder:** $E_\theta(\mathbf{x}) \rightarrow \mathbf{z}_e \in \mathbb{R}^{d}$
- **Codebook (Embedding Space):** $\mathcal{E} = \{\mathbf{e}_k\}_{k=1}^K$, $\mathbf{e}_k \in \mathbb{R}^{d}$
- **Quantizer:** Maps $\mathbf{z}_e$ to nearest codebook vector $\mathbf{e}_k$
- **Decoder:** $D_\phi(\mathbf{z}_q) \rightarrow \hat{\mathbf{x}}$

### 2. Quantization

For each encoded vector $\mathbf{z}_e$, quantization is performed as:

$$
k^* = \arg\min_{k} \|\mathbf{z}_e - \mathbf{e}_k\|_2 \\
\mathbf{z}_q = \mathbf{e}_{k^*}
$$

### 3. Loss Function

The total loss for VQ-VAE is:

$$
\mathcal{L} = \underbrace{\| \mathbf{x} - \hat{\mathbf{x}} \|_2^2}_{\text{Reconstruction Loss}} + 
\underbrace{\| \text{sg}[\mathbf{z}_e] - \mathbf{e}_{k^*} \|_2^2}_{\text{Codebook Loss}} + 
\underbrace{\beta \| \mathbf{z}_e - \text{sg}[\mathbf{e}_{k^*}] \|_2^2}_{\text{Commitment Loss}}
$$

- $\text{sg}[\cdot]$ denotes the stop-gradient operator (no gradient flows through this term).
- $\beta$ is a hyperparameter controlling the commitment loss.

---

## Step-by-Step Explanation

### 1. Data Preprocessing

- Normalize input data $\mathbf{x}$.
- For images: scale pixel values to $[0, 1]$ or $[-1, 1]$.
- For audio: normalize waveform or spectrogram.

### 2. Encoder

- The encoder $E_\theta$ (e.g., CNN for images, WaveNet for audio) maps $\mathbf{x}$ to a continuous latent vector $\mathbf{z}_e$.
- For sequence or spatial data, $\mathbf{z}_e$ may be a tensor of shape $(H, W, d)$.

### 3. Vector Quantization

- For each position in $\mathbf{z}_e$, find the nearest codebook vector in $\mathcal{E}$:
  $$
  k^* = \arg\min_{k} \|\mathbf{z}_e - \mathbf{e}_k\|_2
  $$
- Replace $\mathbf{z}_e$ with the selected codebook vector:
  $$
  \mathbf{z}_q = \mathbf{e}_{k^*}
  $$
- This operation discretizes the latent space.

### 4. Decoder

- The decoder $D_\phi$ reconstructs $\hat{\mathbf{x}}$ from the quantized latent $\mathbf{z}_q$.
- For images: upsampling CNN.
- For audio: autoregressive decoder (e.g., WaveNet).

### 5. Loss Computation

#### a. Reconstruction Loss

$$
\mathcal{L}_{\text{rec}} = \| \mathbf{x} - \hat{\mathbf{x}} \|_2^2
$$

#### b. Codebook Loss

$$
\mathcal{L}_{\text{codebook}} = \| \text{sg}[\mathbf{z}_e] - \mathbf{e}_{k^*} \|_2^2
$$

- Updates codebook vectors to match encoder outputs.

#### c. Commitment Loss

$$
\mathcal{L}_{\text{commit}} = \| \mathbf{z}_e - \text{sg}[\mathbf{e}_{k^*}] \|_2^2
$$

- Encourages encoder outputs to commit to codebook vectors.

#### d. Total Loss

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{codebook}} + \beta \mathcal{L}_{\text{commit}}
$$

### 6. Backpropagation and Optimization

- Gradients for $\mathcal{L}_{\text{rec}}$ and $\mathcal{L}_{\text{commit}}$ update encoder and decoder.
- Gradients for $\mathcal{L}_{\text{codebook}}$ update codebook vectors only.
- Use the **straight-through estimator** to pass gradients from decoder to encoder through the quantization operation.

### 7. Training

- Minimize $\mathcal{L}$ over the dataset using Adam or SGD.
- Update encoder, decoder, and codebook vectors.

### 8. Inference and Generation

- **Encoding:** Map input $\mathbf{x}$ to discrete code indices via encoder and quantizer.
- **Decoding:** Generate $\hat{\mathbf{x}}$ from code indices using decoder.
- **Sampling:** Sample code indices from a prior (e.g., autoregressive model over codebook indices), then decode to generate new data.

---

## Key Aspects and Considerations

- **Discrete Latent Space:** Enables use of powerful discrete priors (e.g., PixelCNN, Transformer) over code indices for high-quality generation.
- **Codebook Collapse:** Prevented by commitment loss and careful codebook initialization.
- **Scalability:** VQ-VAE scales to high-dimensional data (images, audio, video).
- **Hierarchical Extensions:** Stacking multiple VQ-VAE layers enables multi-scale modeling (VQ-VAE-2).
- **Applications:** Image, audio, and video generation; representation learning; compression.

---

## Summary Table

| Component         | Function                                      | Mathematical Formulation                          |
|-------------------|-----------------------------------------------|---------------------------------------------------|
| Encoder           | Maps $\mathbf{x}$ to $\mathbf{z}_e$           | $E_\theta(\mathbf{x}) \rightarrow \mathbf{z}_e$   |
| Quantizer         | Discretizes $\mathbf{z}_e$                    | $\mathbf{z}_q = \mathbf{e}_{k^*}$                 |
| Decoder           | Reconstructs $\hat{\mathbf{x}}$ from $\mathbf{z}_q$ | $D_\phi(\mathbf{z}_q) \rightarrow \hat{\mathbf{x}}$ |
| Loss              | Trains model via reconstruction, codebook, and commitment losses | $\mathcal{L}$ as above                |

---

## References

- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. arXiv:1711.00937
- Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. arXiv:1906.00446

---