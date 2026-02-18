## Gaussian Mixture Models (GMM)

### 1 Concise Definition  
A Gaussian Mixture Model represents an unknown density $p(x)$ over $x\!\in\!\mathbb{R}^d$ as a convex combination of $K$ multivariate Gaussian components:
$$
p(x\mid\Theta)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}\!\left(x\;\middle|\;\mu_k,\Sigma_k\right),
\qquad
\Theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K},
\qquad
\pi_k>0,\;\sum_{k=1}^{K}\pi_k=1.
$$

---

### 2 Model Specification

#### 2.1 Generative Story  
1. Draw latent component index  
   $$z\;\sim\;\mathrm{Categorical}(\pi_1,\dots,\pi_K),\qquad z\in\{1{:}K\}.$$
2. Draw observation  
   $$x\;\sim\;\mathcal{N}\!\left(\mu_z,\Sigma_z\right).$$  

Conditional independence: $x\;\bot\!\!\!\bot\;x'\mid z,z'$ and $z\;\bot\!\!\!\bot\;z'$.

#### 2.2 Complete–Data Representation  
Introduce $z_{nk}=\mathbb{I}[z_n=k]$ for $n=1{:}N$:
$$
p(x_{1{:}N},z_{1{:}N}\mid\Theta)=\prod_{n=1}^{N}\prod_{k=1}^{K}
\Bigl[\pi_k\,\mathcal{N}(x_n\mid\mu_k,\Sigma_k)\Bigr]^{z_{nk}}.
$$

#### 2.3 Covariance Parameterizations  
• Full: $\Sigma_k\in\mathbb{S}_{++}^d$ – $\tfrac{d(d+1)}{2}$ parameters each.  
• Diagonal: $\Sigma_k=\mathrm{diag}(\sigma_{k1}^2,\dots,\sigma_{kd}^2)$.  
• Isotropic: $\Sigma_k=\sigma_k^2I_d$.  
• Tied / Shared: $\Sigma_k=\Sigma$ for all $k$.  
• Low-rank: $\Sigma_k=W_kW_k^\top+\Psi_k$ (Mixture of Factor Analyzers).

---

### 3 Likelihood Function

Observed-data log-likelihood  
$$
\mathcal{L}(\Theta)=\sum_{n=1}^{N}\log\Bigl[\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x_n\mid\mu_k,\Sigma_k)\Bigr].
$$
Non-convex; direct maximization is intractable ⇒ Expectation–Maximization.

---

### 4 Expectation–Maximization (EM)

#### 4.1 E-Step (Responsibility Computation)  
$$
\gamma_{nk}\;=\;p(z_{nk}=1\mid x_n,\Theta^{\text{old}})=
\frac{\pi_k\,\mathcal{N}(x_n\mid\mu_k,\Sigma_k)}
     {\sum_{j=1}^{K}\pi_j\,\mathcal{N}(x_n\mid\mu_j,\Sigma_j)}.
$$

#### 4.2 M-Step (Parameter Updates)  
Define $N_k=\sum_{n=1}^{N}\gamma_{nk}$.

• Mixing weights  
$$
\pi_k^{\text{new}}=\frac{N_k}{N}.
$$

• Means  
$$
\mu_k^{\text{new}}=\frac{1}{N_k}\sum_{n=1}^{N}\gamma_{nk}\,x_n.
$$

• Covariances  
$$
\Sigma_k^{\text{new}}=\frac{1}{N_k}\sum_{n=1}^{N}\gamma_{nk}
\bigl(x_n-\mu_k^{\text{new}}\bigr)\bigl(x_n-\mu_k^{\text{new}}\bigr)^{\!\top}.
$$

#### 4.3 Convergence  
Iterate until  
$$
\Delta\mathcal{L}<\varepsilon\quad\text{or}\quad
\|\Theta^{(t)}-\Theta^{(t-1)}\|_2<\delta.
$$

#### 4.4 Complexity  
One EM iteration with full covariances:  
$$
\text{Time}= \mathcal{O}(N K d^2),\qquad
\text{Memory}= \mathcal{O}(N K + K d^2).
$$

---

### 5 Bayesian Treatment

#### 5.1 Conjugate Priors  
$$
\pi \sim \mathrm{Dirichlet}(\alpha_1,\dots,\alpha_K),
$$
$$
(\mu_k,\Sigma_k) \sim \mathrm{NIW}\bigl(m_0,\kappa_0,\Psi_0,\nu_0\bigr).
$$

#### 5.2 Posterior Inference  
• Gibbs sampling: integrate out $\pi,\mu_k,\Sigma_k$ ⇒ collapsed sampler for $z$.  
• Variational Bayes (VB): factorized posterior $q(z,\pi,\mu,\Sigma)=q(z)q(\pi)\prod_k q(\mu_k,\Sigma_k)$.  
• Non-parametric: Dirichlet Process Gaussian Mixture (DP-GMM) removes fixed $K$.

---

### 6 Model Selection

• Bayesian Information Criterion  
$$
\text{BIC}=-2\mathcal{L}_{\max}+p\log N,\quad
p=K\Bigl[d+\tfrac{d(d+1)}{2}\Bigr]+K-1.
$$

• Integrated Completed Likelihood (ICL), AIC, cross-validated held-out likelihood.  
• DP-GMM or Pitman–Yor processes for automatic $K$.

---

### 7 Initialization Strategies

• $k$-means / $k$-means++ centers → $\mu_k$, empirical covariances → $\Sigma_k$, cluster sizes → $\pi_k$.  
• Random assignment with Dirichlet draws.  
• Spectral methods (method of moments).  
• Hierarchical agglomerative clustering warm-start.

---

### 8 Numerical Considerations

• Log-sum-exp trick for $\mathcal{L}$ and $\gamma_{nk}$.  
• Regularize $\Sigma_k\leftarrow\Sigma_k+\lambda I_d$ to avoid singularities.  
• Cholesky or eigen-decomposition for $\Sigma_k^{-1}$ and $\det\Sigma_k$.  
• Detect component collapse: if $N_k<\tau$, remove or re-initialize.

---

### 9 Parallelization & Large-Scale Training

| Paradigm | Strategy | Notes |
|----------|----------|-------|
| Data parallel | Split $x_{1{:}N}$, aggregate sufficient statistics $(N_k,\mu_k,\Sigma_k)$ via all-reduce | GPU / MPI |
| Map-Reduce EM | Map: local E-step; Reduce: global M-step | Scales to billions of points |
| Online / Mini-batch EM | Stochastic approximation of sufficient statistics; learning rate $\rho_t$ | $\mathcal{O}(1)$ memory wrt $N$ |
| GPU Kernels | Batched density evaluation, Cholesky on GPU | cuBLAS/cuSOLVER |

---

### 10 Extensions & Variants

• Mixture of Factor Analyzers  
$$
\Sigma_k=W_kW_k^\top+\Psi_k,\;W_k\in\mathbb{R}^{d\times r},\,r\ll d.
$$

• Gaussian Mixture Regression (GMR): predict $y$ given $x$ via conditional mixture.  

• Mixture Density Networks (MDN): neural network outputs $\pi_k,\mu_k,\Sigma_k$.  

• Heteroscedastic GMM, Student-t Mixtures (robust to outliers), Mixture of Gaussians with Normal-Inverse-Wishart priors.

---

### 11 Applications

• Unsupervised clustering and embedding initialization  
• Voice Activity Detection, Speaker Verification (i-Vectors)  
• Background subtraction in vision  
• Anomaly detection via low mixture likelihood  
• Density estimation for generative sampling and importance weighting  
• Semi-supervised learning as class-conditional mixtures

---

### 12 Statistical Properties

• Identifiability up to label permutation.  
• Consistency under correct model and $K$ fixed; convergence rate $\mathcal{O}\!\left(\sqrt{N}\right)$.  
• EM guarantees monotonic likelihood increase but not global optimality; multiple restarts advisable.  

---

### 13 Best-Practice Pipeline (End-to-End)

1. Data preprocessing  
   • Center & possibly whiten data.  
   • Remove gross outliers to stabilize covariance estimates.  

2. Determine candidate $K$  
   • Use BIC, spectral gap, or DP-GMM.  

3. Initialize  
   • $k$-means++ followed by covariance regularization.  

4. Run EM with  
   • Log-domain computations, $\lambda I_d$ regularization, early-stopping on $\Delta\mathcal{L}/|\mathcal{L}|<10^{-6}$.  

5. Multiple restarts ($M\!\sim\!10$), retain highest $\mathcal{L}$.  

6. Validate on held-out set; compute Perplexity  
   $$\text{PP}=\exp\!\Bigl(-\tfrac{1}{N_{\text{val}}}\sum_n\log p(x_n)\Bigr).$$  

7. If needed, refine with Bayesian posterior sampling or merge/split heuristic.  

---

End.