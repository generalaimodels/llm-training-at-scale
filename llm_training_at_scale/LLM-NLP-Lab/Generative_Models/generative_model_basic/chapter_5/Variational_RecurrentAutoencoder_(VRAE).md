# Variational Recurrent Autoencoder (VRAE)

---

## 1. Concise Definition  
A Variational Recurrent Autoencoder (VRAE) is a generative sequence model that combines  
(1) the latent‐variable formulation of a Variational Autoencoder (VAE) with  
(2) the temporal modeling capacity of a Recurrent Neural Network (RNN).  
It learns a probabilistic mapping from an observed variable‐length sequence $\mathbf{x}_{1:T}$ to a fixed‐dimensional latent vector $\mathbf{z}$, then reconstructs (or generates) sequences via a recurrent decoder while maximizing a variational Evidence Lower Bound (ELBO).

---

## 2. Mathematical Formulation  

### 2.1 Generative Process  
Given latent prior $p(\mathbf{z})$, the joint distribution factorizes as  
$$p_\theta(\mathbf{x}_{1:T},\mathbf{z}) = p(\mathbf{z}) \; p_\theta(\mathbf{x}_{1:T}\mid\mathbf{z}).$$

* **Prior**: $$p(\mathbf{z})=\mathcal{N}(\mathbf{z};\mathbf{0},\mathbf{I})$$  
* **Decoder (RNN)**:  
  At each timestep $t$  
  $$\mathbf{h}_t = f_\text{dec}(\mathbf{h}_{t-1}, \mathbf{x}_{t-1}, \mathbf{z};\;\theta_h),$$  
  $$p_\theta(\mathbf{x}_t\mid \mathbf{h}_t) = g_\text{dec}(\mathbf{h}_t;\;\theta_x).$$  

### 2.2 Inference Model  
The intractable posterior $p_\theta(\mathbf{z}\mid\mathbf{x}_{1:T})$ is approximated with a parameterized encoder RNN:  
$$q_\phi(\mathbf{z}\mid\mathbf{x}_{1:T})=\mathcal{N}\!\big(\mathbf{z};\;\boldsymbol{\mu}_\phi(\mathbf{x}_{1:T}),\operatorname{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}_{1:T}))\big).$$  
Encoder states:  
$$\mathbf{s}_t = f_\text{enc}(\mathbf{s}_{t-1}, \mathbf{x}_t;\;\phi_s), \qquad  
\mathbf{s}_T \xrightarrow{\;\text{MLP}\;} (\boldsymbol{\mu}_\phi, \log\boldsymbol{\sigma}_\phi).$$  

### 2.3 Objective (ELBO)  
For a training sequence $\mathbf{x}_{1:T}$,  
$$
\mathcal{L}(\theta,\phi;\mathbf{x}_{1:T}) = 
\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x}_{1:T})} 
\Big[\sum_{t=1}^{T}\log p_\theta(\mathbf{x}_t\mid \mathbf{z},\mathbf{x}_{<t})\Big] 
- D_{\text{KL}}\!\big(q_\phi(\mathbf{z}\mid\mathbf{x}_{1:T}) \,\|\, p(\mathbf{z})\big).
$$  
Maximizing $\mathcal{L}$ w.r.t. $(\theta,\phi)$ minimizes the variational gap to $\log p_\theta(\mathbf{x}_{1:T})$.

### 2.4 Reparameterization Trick  
Sample latent vector via  
$$\mathbf{z}=\boldsymbol{\mu}_\phi+\boldsymbol{\sigma}_\phi\odot\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),$$  
enabling low‐variance gradient estimates.

---

## 3. Architecture Design  

| Component | Structure | Key Parameters |
|-----------|-----------|----------------|
| Encoder   | Unidirectional/Bidirectional RNN (LSTM/GRU) | hidden size $d_h^\text{enc}$, layers $L_e$ |
| Latent Projection | MLP or linear | $\boldsymbol{\mu}_\phi,\log\boldsymbol{\sigma}_\phi\in\mathbb{R}^{d_z}$ |
| Decoder   | Autoregressive RNN (LSTM/GRU) conditioned on $\mathbf{z}$ | hidden size $d_h^\text{dec}$, layers $L_d$ |
| Output Head | Softmax (discrete) or parameterized distribution (continuous) | vocab size $V$ or dims $d_x$ |

Design choices:  
• **Global latent**: one $\mathbf{z}$ per sequence (standard VRAE).  
• **Local latents**: extend to VRNN to have $\mathbf{z}_t$ per timestep.  
• **Conditioning**: $\mathbf{z}$ concatenated to decoder input or used to initialize hidden state.

---

## 4. End-to-End Workflow  

### 4.1 Data Pipeline  
1. Collect raw sequential data (text, MIDI, sensor, speech frames).  
2. Preprocess  
   • Tokenize/quantize → integer IDs or normalized vectors.  
   • Pad or bucket to max length $T_{\max}$.  
   • Create input–target pairs $(\mathbf{x}_{1:T},\mathbf{x}_{0:T-1})$.  
3. (Optional) Alignment & augmentation: tempo stretch, pitch shift, noise.

### 4.2 Training Loop  
```
for epoch:
    for batch X in loader:
        μ, logσ = EncoderRNN(X)
        z = μ + σ * ε                     # reparameterize
        loglik = DecoderRNN(z, X[:,:-1])
        kl = 0.5 * sum(1 + 2logσ - μ² - σ²)
        loss = - (loglik - kl_weight * kl)
        backprop + Adam / LAMB
```
• **KL Annealing**: ramp `kl_weight` from 0→1.  
• **Gradient clipping**: $\|\nabla\|\leq \tau$ to stabilize RNNs.

### 4.3 Parallelization  
• **Data parallel** across GPUs (batch dimension).  
• **Sequence bucketing** minimizes padding overhead.  
• **Mixed precision (FP16/BF16)** speeds training; maintain FP32 master weights for stability.

### 4.4 Regularization  
• Dropout on RNN outputs.  
• $\beta$‐VAE style $\,\beta D_{\text{KL}}$ to control compression.  
• Word dropout / scheduled sampling for decoder robustness.

---

## 5. Inference & Generation  

### 5.1 Reconstruction  
Input $\mathbf{x}_{1:T}$ → sample $\mathbf{z}$ → decoder produces $\hat{\mathbf{x}}_{1:T}$. Evaluate reconstruction error.

### 5.2 Unconditional Generation  
Sample $\mathbf{z}\sim p(\mathbf{z})$ → decode autoregressively:  
$$\mathbf{h}_0 = \tanh(W_z\mathbf{z}),\quad  
  \mathbf{x}_0 = \texttt{<SOS>}$$  
For $t=1\dots T_\text{max}$: sample $\mathbf{x}_t\sim p_\theta(\cdot\mid\mathbf{h}_t)$ until \texttt{<EOS>}.

### 5.3 Latent Manipulation  
• **Interpolation**: $\mathbf{z}_\lambda = (1-\lambda)\mathbf{z}_1+\lambda\mathbf{z}_2$.  
• **Attribute vector arithmetic**: learn $\mathbf{v}_\text{attr}$, set $\mathbf{z}^\prime=\mathbf{z}+\alpha\mathbf{v}_\text{attr}$.

---

## 6. Evaluation Metrics  

| Aspect | Metric | Formula |
|--------|--------|---------|
| Probabilistic | Negative ELBO | $-\mathcal{L}$ (lower is better) |
| Likelihood | Bits-per-char / token | $\frac{ -\log_2 p(\mathbf{x}) }{T}$ |
| Sequence Quality | BLEU, ROUGE, CIDEr (text) / Fréchet Audio Distance (audio) | task-dependent |
| Latent Space | Mutual Information, MIG, DCI | disentanglement |

---

## 7. Extensions  

1. **Conditional VRAE (CVRAE)**: extra condition $\mathbf{c}$ (class, style)  
   $$q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{c}),\;
     p_\theta(\mathbf{x}\mid\mathbf{z},\mathbf{c}).$$  
2. **Hierarchical VRAE**: multiple latents $\{\mathbf{z}^{(l)}\}$ for multiscale structure.  
3. **Bidirectional Decoder**: Transformer decoder with $\mathbf{z}$ as memory token (hybrid VRAE–VAE).

---

## 8. Comparative Discussion  

| Model | Latent Granularity | Temporal Modeling | Pros | Cons |
|-------|-------------------|-------------------|------|------|
| VAE   | global | none | simple | ignores time |
| VRAE  | global | RNN ($\mathbf{x}_{<t}$) | captures sequence, compact latent | can underfit long-term structure |
| VRNN  | per-timestep | RNN | rich, expressive | expensive, harder training |
| GPT   | none | Transformer | strong generation | no explicit latent; poor controllability |

---

## 9. Implementation Tips  

• Initialize RNN forget gate bias $b_f=1$ (LSTM) to aid learning long dependencies.  
• Use LayerNorm in RNN cells for stable KL gradients.  
• Monitor posterior collapse: track $\operatorname{KL}$ term; if $\rightarrow 0$, reduce decoder capacity or increase $\beta$.  
• For speech/audio, use μ-law quantization and causal convolutions as decoder front-end before RNN.

---

## 10. Summary of End-to-End Flow  
1. Dataset $\rightarrow$ tokenization, batching.  
2. Encoder RNN $\rightarrow$ $(\boldsymbol{\mu}_\phi,\boldsymbol{\sigma}_\phi)$ $\rightarrow$ sample $\mathbf{z}$.  
3. Decoder RNN conditioned on $\mathbf{z}$ reconstructs sequence.  
4. Optimize ELBO with reparameterization + Adam; apply KL annealing, gradient clipping.  
5. Evaluate ELBO and downstream sequence metrics; iterate hyperparameters.  
6. In deployment, sample $\mathbf{z}$ from prior and decode to generate novel sequences.

---