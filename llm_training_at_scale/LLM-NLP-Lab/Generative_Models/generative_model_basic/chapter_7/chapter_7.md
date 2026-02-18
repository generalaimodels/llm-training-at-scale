# Chapter 7 Autoregressive Pixel & Audio Models  

---

## 7.0 General Definition  

Given a $D$-dimensional signal $\mathbf x=(x_1,\dots,x_D)$ (e.g., raster-scanned pixels or audio samples), an autoregressive generative model factorises its joint density as  

$$p_\theta(\mathbf x)=\prod_{d=1}^{D}p_\theta(x_d\mid x_{<d})$$  

and trains the parameters $\theta$ by maximum likelihood, i.e. minimising  

$$\mathcal L(\theta)=-\mathbb E_{\mathbf x\sim\mathcal D}\Bigl[\sum_{d=1}^{D}\log p_\theta(x_d\mid x_{<d})\Bigr].$$  

The essential design problem is how to parameterise $p_\theta(\cdot\mid\cdot)$ such that  
• the full dependency structure is respected,  
• the model is parallelisable during training,  
• the receptive field grows rapidly with network depth to capture long-range correlations.  

---

## 7.1 PixelRNN / PixelCNN  

### 7.1.1 Concise Definition  
PixelRNN and PixelCNN are autoregressive models for images that respect a raster ordering over 2-D pixels. PixelRNN uses recurrent layers; PixelCNN replaces recurrence with masked convolutions to enable fully parallel training.

### 7.1.2 Mathematical Formulation  

Let an image have $H\times W$ pixels and $C$ colour channels (typically $C=3$). The lexicographic index is $d=i\!+\!H\,(j-1)$ for pixel $(j,i)$; the factorisation becomes  

$$p_\theta(\mathbf x)=\prod_{j=1}^{H}\prod_{i=1}^{W}p_\theta\bigl(x_{j,i}\mid\mathbf x_{<j,i}\bigr)$$  

where $\mathbf x_{<j,i}$ denotes all pixels above or left of $(j,i)$. Each conditional is often further factorised channel-wise, e.g.  

$$p_\theta(x_{j,i}^{R},x_{j,i}^{G},x_{j,i}^{B}\mid\mathbf x_{<j,i})
   =p_\theta(x_{j,i}^{R}\mid\mathbf x_{<j,i})
    p_\theta(x_{j,i}^{G}\mid\mathbf x_{<j,i},x_{j,i}^{R})
    p_\theta(x_{j,i}^{B}\mid\mathbf x_{<j,i},x_{j,i}^{R},x_{j,i}^{G}).$$  

### 7.1.3 PixelRNN Architectures  

1. Row LSTM  
   • Hidden state $h_{j,i}$ updated horizontally:  
     $$h_{j,i}= \mathrm{LSTM}\bigl([x_{j,i},\,h_{j-1,i}],\,h_{j,i-1}\bigr).$$  
   • Two nested loops $O(HW)$ at training time.

2. Diagonal BiLSTM  
   • Processes pixels along diagonals to enhance parallelism; still $O(HW)$ but with larger receptive field per step.

### 7.1.4 PixelCNN Architectures  

1. Masked Convolution  

   A convolution kernel $K\in\mathbb R^{k\times k}$ is masked to satisfy autoregressive constraints.  
   Define a binary mask $M$ with  

   $$M_{u,v}= \begin{cases}
   1,&\! (u<v_{\text{center}})\ \text{or }(u=v_{\text{center}},\,v<v_{\text{center}})\\
   0,&\text{otherwise}
   \end{cases}$$  

   and set $\tilde K = M\odot K$. Convolution with $\tilde K$ ensures no dependency on future pixels.

2. Gated PixelCNN  

   For feature map $\mathbf F$, the gated update is  

   $$\mathbf H = \tanh(W_f\ast\mathbf F)\odot\sigma(W_g\ast\mathbf F)$$  

   with $W_f,W_g$ both masked. Skip connections and residual blocks stack $L$ such layers; receptive field grows linearly with $L$.

3. PixelCNN++ Improvements  

   • Discretised logistic mixture likelihood:  

     $$p_\theta(x\mid\mu,\sigma,\pi)=\sum_{m=1}^{M}\pi_m\,\Bigl(\sigma_m^{-1}\bigl(\sigma\bigl({x+0.5-\mu_m\over\sigma_m}\bigr)-
     \sigma\bigl({x-0.5-\mu_m\over\sigma_m}\bigr)\bigr)\Bigr).$$  

   • Downsampling/upsampling (U-Net style) enlarges receptive field superlinearly.  
   • Weight normalisation and EMA of parameters accelerate convergence.

### 7.1.5 Training Pipeline  

• Dataset: CIFAR-10, ImageNet $32\times32$, CelebA-HQ, etc.  
• Quantise pixels to 8-bit integers or keep continuous via logistic mixture.  
• Optimiser: Adam with schedule $lr(t)=\eta_0(1+t/T)^{-0.5}$.  
• Batch-parallel convolutions ensure $\mathcal O(1)$ forward passes for $B$ samples.

### 7.1.6 Sampling  

Pseudocode for RGB PixelCNN:  
```
for j in 1..H:
  for i in 1..W:
    for c in [R,G,B]:
        p = model(x_{<j,i}, x_{j,i}^{<c})
        x_{j,i}^c ~ p
```
Caching intermediate feature maps allows $O(1)$ amortised per pixel instead of recomputing the entire network.

---

## 7.2 MADE (Masked Autoencoder for Distribution Estimation)  

### 7.2.1 Concise Definition  
MADE converts a standard fully connected autoencoder into an autoregressive density estimator by applying binary masks to connections such that each output dimension $\hat x_d$ only depends on input dimensions $<\!d$.

### 7.2.2 Mathematical Formulation  

Let $\mathbf h^{(l)}$ be layer $l$ with weight matrix $W^{(l)}$. Define a mask $M^{(l)}$ satisfying  

$$M^{(l)}_{ij}= \mathbb I\bigl[m_i^{(l-1)}<m_j^{(l)}\bigr]$$  

where $m_j^{(l)}\in\{1,\dots, D\}$ is an assigned “connectivity index”. The forward pass becomes  

$$\mathbf h^{(l)} = f\bigl((W^{(l)}\odot M^{(l)})\mathbf h^{(l-1)} + \mathbf b^{(l)}\bigr).$$  

For the output layer  

$$\hat x_d = g_d\bigl(\sum_{k}W^{(L)}_{dk}h^{(L-1)}_k + b^{(L)}_d\bigr),$$  

the mask enforces $M^{(L)}_{dk}=\mathbb I[m_k^{(L-1)}<d]$. Thus  

$$p_\theta(\mathbf x)=\prod_{d=1}^{D}p_\theta(x_d\mid x_{<d})$$  

with all conditionals computed in a single network evaluation.

### 7.2.3 Step-by-Step Construction  

1. Ordering choice  
   • Permute input dimensions with permutation $\pi$; set target ordering $d=\pi^{-1}(i)$.  
   • Randomly sample $\pi$ per mini-batch to ensemble multiple factorizations (“order-agnostic training”).

2. Mask generation  
   • Sample $m^{(0)}_i=d_i=\pi(i)$.  
   • For hidden layer $l$, sample $m^{(l)}_j\sim\{1,\dots,D-1\}$.  
   • Enforce monotonicity $m^{(l)}_j \ge \min_i m^{(l-1)}_i$ to guarantee each neuron sees earlier inputs only.

3. Output parameterisation  
   • Binary data: Bernoulli $\sigma(\hat x_d)$.  
   • 8-bit pixels: categorical softmax over 256 bins.  
   • Real data: Gaussian mixture heads.

4. Training  
   • Optimise negative log-likelihood with Adam.  
   • Dropout between hidden layers preserves mask validity.  
   • Model averaging across orderings reduces variance of log-probability estimates.

5. Sampling  
   • Sequentially loop over $d=1\dots D$; reuse the same feed-forward pass with masked inputs already sampled to amortise computation.

### 7.2.4 Complexity  

• Forward train cost: $O(B\sum_l n_{l-1}n_l)$ identical to vanilla MLP.  
• Sampling cost: $O(D)$ network evaluations, but can be parallelised by caching hidden activations per step.

---

## 7.3 WaveNet  

### 7.3.1 Concise Definition  
WaveNet is a 1-D autoregressive model for raw audio that employs stacks of dilated causal convolutions with gated activations, enabling a receptive field of thousands of timesteps while retaining training parallelism.

### 7.3.2 Mathematical Formulation  

Joint factorisation over $T$ audio samples:  

$$p_\theta(\mathbf x)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t},\mathbf c)$$  

where $\mathbf c$ is an optional conditioning signal (text phonemes, speaker embedding, mel-spectrogram).  

Dilated convolution: for layer $l$ with dilation $d_l=2^{l}$, filter $w^{(l)}$ of width $k$:  

$$z^{(l)}_t=\sum_{r=0}^{k-1}w^{(l)}_r\,h^{(l-1)}_{t-d_l\,r}.$$  

Gated activation with residual connection:  

$$h^{(l)}_t = \sigma\bigl(W^{(l)}_f\ast h^{(l-1)}_t + V^{(l)}_f\ast \mathbf c_t\bigr)
             \odot \tanh\bigl(W^{(l)}_g\ast h^{(l-1)}_t + V^{(l)}_g\ast \mathbf c_t\bigr) + h^{(l-1)}_t.$$

Skip connections accumulate:  

$$s_t = \sum_{l}U^{(l)}\ast h^{(l)}_t.$$

Output distribution (original): 8-bit $\mu$-law categorical with softmax. Later variants employ mixture of logistics:  

$$p_\theta(x_t)=\sum_{m=1}^{M}\pi_{t,m}\,\mathrm{Logistic}\bigl(x_t\mid\mu_{t,m},s_{t,m}\bigr).$$

### 7.3.3 Network Topology  

• Stack of $L$ layers organised into $B$ residual blocks, each block cycles dilations $\{1,2,4,\dots,512\}$.  
• Total receptive field  

$$R = (k-1)\sum_{l=0}^{L-1}2^{l}+1 \approx k\,2^{L}.$$

For $k=2$, $L=30$ ⇒ $R\approx 32\,768$ samples ($\approx2\,\mathrm{s}$ @ 16 kHz).

### 7.3.4 Conditioning Mechanisms  

1. Global conditioning: augment each layer with speaker embedding $g$: add $V^{(l)}_g\ast g$ term.  
2. Local conditioning: upsample frame-level features (e.g. mel-spectrogram) via transposed conv before addition.

### 7.3.5 Training Protocol  

• Input preprocessing:  
  – $\mu$-law companding: $$x\gets\operatorname{sign}(x)\,\frac{\ln(1+\mu|x|)}{\ln(1+\mu)},\quad\mu=255.$$  
  – Quantise to 256 categories.  

• Objective: categorical cross-entropy or logistic NLL.  
• Optimiser: Adam; LR schedule with warm-up and exponential decay.  
• Weight normalisation stabilises deep dilated stacks.  
• Gradient clipping @ $0.5$ prevents exploding gradients for long sequences.

### 7.3.6 Inference & Acceleration  

• Naïve sampling cost $O(TR)$ due to autoregression.  
• Caching dilated conv states reduces cost to $O(T)$ with constant per-step memory $O(\sum_l k)$.  
• Distillation approaches:  
  – Parallel WaveNet: trains a flow-based student via KL divergence from teacher.  
  – WaveRNN: single recurrent layer with dual softmax; drastically fewer multiplies per sample.  
  – WaveGrad, DiffWave: diffusion-based, non-autoregressive.

### 7.3.7 Applications  

• Neural text-to-speech (Tacotron 2 → WaveNet).  
• Music synthesis and transformation.  
• Neural vocoders for speech coding, audio bandwidth extension, packet loss concealment.

---

## 7.4 Comparative Overview  

| Attribute | PixelRNN | PixelCNN / PixelCNN++ | MADE | WaveNet |
|-----------|----------|-----------------------|------|---------|
| Domain | Images | Images | Tabular / any fixed-dim vector | Raw audio |
| Core op. | LSTM recurrence | Masked conv | Masked MLP | Dilated causal conv |
| Train parallelism | Low (sequential rows) | High | High | High |
| Sampling speed | Very slow | Slow | Moderate | Very slow (teacher) |
| Receptive field | Linearly grows per time-step | Linear in depth | Full by construction | Exponential in depth |
| Likelihood exact? | Yes | Yes | Yes | Yes |

---

## 7.5 Current Research Directions  

• Hybrid masked + latent models to amortise sampling (e.g., SPN, VDVAE).  
• Sparse / factorised priors for $4$-K resolution images with PixelCNN kernels.  
• Flow-matching and diffusion distillation to obtain parallel audio decoders matching WaveNet quality.  
• Autoregressive audio–visual joint models unifying WaveNet with PixelCNN for talking-head synthesis.