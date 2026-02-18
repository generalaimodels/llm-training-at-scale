## 4 Core Variational Autoencoders

### 4.1 Basic VAE

#### 4.1.1 Definition  
A Variational Autoencoder (VAE) is a latent-variable generative model that jointly learns  
i) a probabilistic encoder (approximate posterior) $$q_\phi(z\mid x)$$ and  
ii) a probabilistic decoder (likelihood) $$p_\theta(x\mid z)$$  
by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood $$\log p_\theta(x)$$.

#### 4.1.2 Generative Process  
1. Draw latent $$z\sim p(z)=\mathcal N(0,I)$$.  
2. Draw observable $$x\sim p_\theta(x\mid z)$$.

#### 4.1.3 Objective  
$$
\mathcal L_{\text{VAE}}(x;\theta,\phi)=
\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]
-
D_{\text{KL}}\big(q_\phi(z\mid x)\Vert p(z)\big)
$$
Maximize $$\mathcal L_{\text{VAE}}$$ (equivalently minimize $$-\mathcal L_{\text{VAE}}$$).

#### 4.1.4 Derivation of ELBO  
$$
\log p_\theta(x)=
\mathcal L_{\text{VAE}}(x;\theta,\phi)+
D_{\text{KL}}\big(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\big)\ge\mathcal L_{\text{VAE}}
$$
Equality holds when $$q_\phi(z\mid x)=p_\theta(z\mid x)$$.

#### 4.1.5 Parameterization  

Encoder: $$q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma^2_\phi(x)))$$  
Reparameterization: $$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$$  
Decoder: Choice depends on data:
• Bernoulli for binary $$x$$ (sigmoid–parameterized logits).  
• Gaussian for real-valued $$x$$ (predict mean & log-var).  
• Categorical (softmax) for discrete tokens.

#### 4.1.6 Gradient Estimation  
$$
\nabla_\theta\mathcal L=\mathbb E_{q_\phi}\big[\nabla_\theta\log p_\theta(x\mid z)\big]
$$
$$
\nabla_\phi\mathcal L=\mathbb E_{\epsilon}\big[\nabla_z\log p_\theta(x\mid z)\nabla_\phi z - \nabla_\phi \log q_\phi(z\mid x)\big]
$$
Reparameterization yields low-variance Monte-Carlo estimates with $$\epsilon$$ samples.

#### 4.1.7 Training Pipeline  
1. Mini-batch $$\{x^{(i)}\}_{i=1}^B$$  
2. Forward: compute $$\mu_\phi,\sigma_\phi,\epsilon,z,p_\theta(x\mid z)$$  
3. ELBO estimate  
4. Back-prop through reparameterized graph  
5. Update $$\theta,\phi$$ via Adam/SGD.

#### 4.1.8 Architectural Choices  
• Convolutional, Transformer, or MLP encoders/decoders.  
• Latent dimensionality $$d_z$$ ≈ 2–512 (task dependent).  
• Warm-up schedule to avoid posterior collapse: scale KL term gradually.  

#### 4.1.9 Evaluation Metrics  
• Negative ELBO (nats/bits-per-dim).  
• Importance Weighted Bound (IWAE).  
• Fréchet Inception Distance, reconstruction error, latent traversals.

---

### 4.2 β-VAE

#### 4.2.1 Definition  
β-VAE introduces a scalar $$\beta>0$$ to modulate the KL term, promoting disentangled latent representations.

$$
\mathcal L_{\beta\text{-VAE}}(x)=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-
\beta\,D_{\text{KL}}(q_\phi(z\mid x)\Vert p(z))
$$

#### 4.2.2 Effect of $$\beta$$  
• $$\beta>1$$: stronger pressure towards the prior ⇒ increased independence across latent dimensions, improved disentanglement, but reduced reconstruction fidelity.  
• $$\beta<1$$: looser prior matching, better reconstructions, poorer disentanglement.

#### 4.2.3 Capacity Annealing  
Gradually increase $$\beta$$ or the target KL $$C$$ to balance learning:

$$
\mathcal L =
\mathbb E_{q_\phi}[\log p_\theta(x\mid z)]-
\beta\,|D_{\text{KL}}(q_\phi(z\mid x)\Vert p(z))-C|
$$
Start with $$C=0$$, linearly raise to saturation capacity.

#### 4.2.4 Disentanglement Metrics  
• DCI-Disentanglement  
• MIG  
• SAP  
Measure independence of latent dimensions w.r.t ground-truth generative factors.

---

### 4.3 Conditional VAE (CVAE)

#### 4.3.1 Definition  
CVAE models conditional generation $$p_\theta(x\mid y)$$ by incorporating auxiliary variable $$y$$ (label, class, text prompt, etc.).

Generative process:  
1. $$z\sim p(z)$$  
2. $$x\sim p_\theta(x\mid z,y)$$

#### 4.3.2 Objective  
$$
\mathcal L_{\text{CVAE}}(x,y)=
\mathbb E_{q_\phi(z\mid x,y)}\big[\log p_\theta(x\mid z,y)\big]
-
D_{\text{KL}}\big(q_\phi(z\mid x,y)\Vert p(z)\big)
$$

#### 4.3.3 Encoder / Decoder Conditioning  
Encoder inputs: concat/FiLM/attention on $$[x,y] \rightarrow (\mu,\sigma)$$.  
Decoder: $$p_\theta(x\mid z,y)$$ via concatenation, conditional batch-norm, or cross-attention.

#### 4.3.4 Training & Inference  
• Training: maximize $$\mathcal L_{\text{CVAE}}$$ on paired $$(x,y)$$.  
• Generation: sample $$z\sim p(z)$$ then $$x\sim p_\theta(\cdot\mid z,y^\ast)$$.

#### 4.3.5 Applications  
• Label-controlled image synthesis.  
• Text-to-speech with style tokens.  
• Semi-supervised learning: split ELBO into labelled vs unlabelled terms; augment with classifier.

---

### 4.4 Implementation Checklist

• Data preprocessing → normalized $$x$$, one-hot/embedding $$y$$.  
• Select latent prior ($$p(z)=\mathcal N(0,I)$$ or VampPrior, flow-based).  
• Architecture search: depth, width, residual, attention.  
• Hyper-parameters: $$\beta,d_z,\text{learning rate},\text{KL warm-up steps}$$.  
• Regularization: dropout, weight decay, spectral norm.  
• Monitoring: ELBO, KL, reconstruction loss curve, posterior statistics.  
• Debugging: latent traversal sanity checks, mutual information estimates.