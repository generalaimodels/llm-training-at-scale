---

# Chapter 7 Autoregressive Pixel & Audio Models

---

## 7.1 PixelRNN / PixelCNN

### Definition

PixelRNN and PixelCNN are autoregressive generative models for images. They model the joint distribution of image pixels as a product of conditional distributions, predicting each pixel sequentially conditioned on previously generated pixels.

### Mathematical Formulation

- **Autoregressive Factorization**:  
  For an image $x$ with $N$ pixels, the joint distribution is factorized as:
  $$
  P(x) = \prod_{i=1}^{N} P(x_i \mid x_{<i})
  $$
  where $x_i$ is the value of the $i$-th pixel in a raster-scan order.

#### PixelRNN

- **Recurrent Layer**:  
  At each pixel position $i$, a hidden state $h_i$ is updated as:
  $$
  h_i = f(h_{i-1}, x_{i-1}; \theta)
  $$
  where $f$ is a recurrent function (e.g., LSTM, GRU).

#### PixelCNN

- **Masked Convolution**:  
  The conditional distribution is modeled using masked convolutions:
  $$
  h = \text{MaskedConv}(x)
  $$
  The mask ensures that the prediction for $x_i$ depends only on $x_{<i}$.

### Step-by-Step Explanation

#### 1. Data Collection and Preprocessing

- **Dataset**: High-quality image datasets (e.g., CIFAR-10, ImageNet).
- **Preprocessing**: Normalization, quantization (e.g., 8-bit per channel), optional data augmentation.

#### 2. Model Architecture

##### PixelRNN

- **Input**: Image $x$ as a sequence of pixels.
- **Recurrent Layers**:  
  - Row LSTM: Processes each row sequentially.
  - Diagonal BiLSTM: Processes along diagonals for better context.
- **Output Layer**:  
  $$
  P(x_i \mid x_{<i}) = \text{softmax}(W_o h_i + b)
  $$
  where $W_o$ and $b$ are learnable parameters.

##### PixelCNN

- **Input**: Image $x$ as a 2D array.
- **Masked Convolutions**:  
  - Mask A: For the first layer, prevents access to current and future pixels.
  - Mask B: For subsequent layers, allows access to current pixel’s previous channels.
- **Stacked Layers**: Multiple masked convolutional layers for deep context.
- **Output Layer**:  
  $$
  P(x_i \mid x_{<i}) = \text{softmax}(W_o h_i + b)
  $$

#### 3. Training

- **Objective**: Minimize negative log-likelihood:
  $$
  \mathcal{L}(\theta) = -\sum_{i=1}^{N} \log P(x_i \mid x_{<i}; \theta)
  $$
- **Optimization**: Adam or RMSProp optimizers, batch normalization, regularization.

#### 4. Inference

- **Generation**:  
  - Sequentially sample each pixel $x_i$ from $P(x_i \mid x_{<i})$.
  - For PixelCNN, generation is parallelizable along independent pixels (with some constraints).

#### 5. Evaluation

- **Metrics**: Negative log-likelihood (bits per dimension), sample quality (visual inspection, FID).

---

## 7.2 MADE (Masked Autoencoder for Distribution Estimation)

### Definition

MADE is an autoregressive model that uses a feedforward neural network with carefully designed masks to enforce autoregressive dependencies, enabling efficient parallel training and inference for distribution estimation.

### Mathematical Formulation

- **Autoregressive Factorization**:  
  $$
  P(x) = \prod_{i=1}^{D} P(x_i \mid x_{<i})
  $$
  where $D$ is the dimensionality of $x$.

- **Masked Neural Network**:  
  The output for $x_i$ is computed as:
  $$
  \hat{x}_i = f_i(x; M)
  $$
  where $M$ is a binary mask matrix ensuring $\hat{x}_i$ depends only on $x_{<i}$.

### Step-by-Step Explanation

#### 1. Data Collection and Preprocessing

- **Dataset**: Tabular, binary, or continuous data.
- **Preprocessing**: Normalization, binarization (if required).

#### 2. Model Architecture

- **Input Layer**: $x \in \mathbb{R}^D$.
- **Masked Connections**:  
  - Assign an ordering to variables.
  - For each layer, construct mask matrices $M^{(l)}$ such that:
    $$
    M^{(l)}_{ij} = 1 \implies \text{unit } j \text{ in layer } l \text{ can depend on input } x_k \text{ only if } k < i
    $$
- **Output Layer**:  
  - Each output neuron predicts $P(x_i \mid x_{<i})$.

#### 3. Training

- **Objective**: Minimize negative log-likelihood:
  $$
  \mathcal{L}(\theta) = -\sum_{i=1}^{D} \log P(x_i \mid x_{<i}; \theta)
  $$
- **Optimization**: Adam or SGD.

#### 4. Inference

- **Generation**:  
  - Sequentially sample $x_i$ using the masked network.
  - For parallel evaluation, use multiple orderings (ensembling).

#### 5. Evaluation

- **Metrics**: Negative log-likelihood, sample quality, calibration.

---

## 7.3 WaveNet

### Definition

WaveNet is an autoregressive generative model for raw audio waveforms, using dilated causal convolutions to model long-range temporal dependencies in audio signals.

### Mathematical Formulation

- **Autoregressive Factorization**:  
  For an audio sequence $x = (x_1, x_2, ..., x_T)$:
  $$
  P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
  $$

- **Dilated Causal Convolution**:  
  The output at time $t$ is:
  $$
  h_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - r \cdot k}
  $$
  where $K$ is the filter size, $r$ is the dilation rate, $w_k$ are the convolution weights.

- **Gated Activation Unit**:  
  $$
  z = \tanh(W_{f} * x) \odot \sigma(W_{g} * x)
  $$
  where $*$ denotes convolution, $\odot$ is element-wise multiplication, $W_f$ and $W_g$ are learnable filters.

### Step-by-Step Explanation

#### 1. Data Collection and Preprocessing

- **Dataset**: Large-scale raw audio datasets (e.g., speech, music).
- **Preprocessing**: Quantization (e.g., $\mu$-law encoding), normalization.

#### 2. Model Architecture

- **Input**: Raw audio waveform $x$.
- **Stacked Dilated Causal Convolutions**:  
  - Multiple layers with exponentially increasing dilation rates.
  - Ensures large receptive field for each output.
- **Residual and Skip Connections**:  
  - Residual blocks for stable training.
  - Skip connections for improved gradient flow.
- **Output Layer**:  
  - Softmax over quantized audio levels:
    $$
    P(x_t \mid x_{<t}) = \text{softmax}(W_o h_t + b)
    $$

#### 3. Training

- **Objective**: Minimize negative log-likelihood:
  $$
  \mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
  $$
- **Optimization**: Adam optimizer, gradient clipping.

#### 4. Inference

- **Generation**:  
  - Sequentially sample each audio sample $x_t$ from $P(x_t \mid x_{<t})$.
  - Highly parallelizable during training, but sequential at inference.

#### 5. Evaluation

- **Metrics**: Negative log-likelihood, mean opinion score (MOS), audio quality metrics.

---

# Summary Table

| Model      | Domain   | Key Mechanism                | Complexity         | Parallelism (Train/Infer) | Sequence Length |
|------------|----------|-----------------------------|--------------------|--------------------------|----------------|
| PixelRNN   | Image    | RNN, sequential pixels      | $O(N)$             | Low / Low                | $<10^4$        |
| PixelCNN   | Image    | Masked convolutions         | $O(N)$             | High / Medium            | $<10^4$        |
| MADE       | General  | Masked feedforward nets     | $O(D)$             | High / Medium            | $<10^3$        |
| WaveNet    | Audio    | Dilated causal convolutions | $O(T)$             | High / Low               | $>10^4$        |

---