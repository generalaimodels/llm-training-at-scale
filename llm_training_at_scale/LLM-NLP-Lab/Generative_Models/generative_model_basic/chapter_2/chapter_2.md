## Chapter 2 Classical Statistical Generative Models  

---

### 2.1 Gaussian Mixture Models (GMM)

#### Definition  
A Gaussian Mixture Model represents the probability density of a random vector $x\!\in\!\mathbb{R}^d$ as a finite mixture of $K$ multivariate normal components:
$$
p(x|\Theta)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}\!\bigl(x\mid\mu_k,\Sigma_k\bigr),
\quad
\Theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K},
\quad
\sum_{k=1}^{K}\pi_k=1,\;\pi_k\!>\!0.
$$

#### Generative Process  
1. Draw component index $z\sim\mathrm{Categorical}(\pi_1,\dots,\pi_K)$.  
2. Draw observation $x\sim\mathcal{N}(\mu_z,\Sigma_z)$.

#### Complete-Data Likelihood  
Introduce latent assignment indicators $z_{nk}\!=\!\mathbb{I}[z_n\!=\!k]$ for $n\!=\!1{:}N$:
$$
p(\{x_n,z_n\}\mid\Theta) = \prod_{n=1}^N\prod_{k=1}^{K}
\Bigl[\pi_k\,\mathcal{N}(x_n|\mu_k,\Sigma_k)\Bigr]^{z_{nk}}.
$$

#### Expectation–Maximization (EM)  

E-step: responsibilities  
$$
\gamma_{nk}=p(z_{nk}=1\mid x_n,\Theta^{\text{old}})=
\frac{\pi_k\,\mathcal{N}(x_n|\mu_k,\Sigma_k)}
     {\sum_{j=1}^K\pi_j\,\mathcal{N}(x_n|\mu_j,\Sigma_j)}.
$$

M-step: parameter updates  
$$
N_k=\sum_{n=1}^N\gamma_{nk},\qquad
\pi_k^{\text{new}}=\frac{N_k}{N},
$$
$$
\mu_k^{\text{new}}=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}\,x_n,
$$
$$
\Sigma_k^{\text{new}}=\frac{1}{N_k}\sum_{n=1}^N
\gamma_{nk}\,(x_n-\mu_k^{\text{new}})(x_n-\mu_k^{\text{new}})^{\!\top}.
$$

Convergence criterion: non-decreasing log-likelihood  
$
\mathcal{L}=\sum_{n=1}^N\log\bigl[\sum_{k}\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)\bigr].
$

#### Model Selection  
Bayesian Information Criterion (BIC):  
$
\text{BIC}=-2\mathcal{L}_{\max}+p\log N,
$  
where $p$ is the number of free parameters.

#### Computational Complexity  
One EM iteration: $\mathcal{O}(NKd^2)$ for full covariances (matrix inversions cached).

#### Applications  
Density estimation, clustering, voice activity detection, anomaly detection, speaker recognition.

#### Limitations & Extensions  
• Sensitive to initialization; k-means++ or spectral methods recommended.  
• Fixed $K$; non-parametric extension via Dirichlet Process GMM.  
• High-dimensional data → Mixture of Factor Analyzers, tied covariances.



---

### 2.2 Hidden Markov Models (HMM)

#### Definition  
An HMM is a doubly stochastic process with latent states $\{s_t\}_{t=1}^T$, $s_t\!\in\!\{1{:}S\}$, forming a first-order Markov chain and emissions $x_t$ conditionally independent given the state:
$$
p(s_1{:}T,x_1{:}T)=\pi_{s_1}\,b_{s_1}(x_1)\prod_{t=2}^{T}a_{s_{t-1}s_t}\,b_{s_t}(x_t),
$$
where  
• $\pi$: initial distribution, $\pi_i\!\ge\!0$, $\sum_i\pi_i\!=\!1$  
• $A=[a_{ij}]$: state transition matrix, $a_{ij}\!\ge\!0$, $\sum_j a_{ij}\!=\!1$  
• $b_i(\cdot)$: emission probability density/mass (Gaussian, categorical, etc.).

#### Forward–Backward Recursion  

Forward variable  
$$
\alpha_t(i)=p(x_1{:}t,s_t=i)=
\bigl[\sum_{j}\alpha_{t-1}(j)\,a_{ji}\bigr]\,b_i(x_t),
\quad\alpha_1(i)=\pi_i\,b_i(x_1).
$$

Backward variable  
$$
\beta_t(i)=p(x_{t+1{:}T}\mid s_t=i)=
\sum_{j}a_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j),
\quad\beta_T(i)=1.
$$

#### EM (Baum–Welch)

E-step: posterior statistics  
$$
\gamma_t(i)=p(s_t=i\mid x_{1{:}T})=
\frac{\alpha_t(i)\beta_t(i)}{\sum_{j}\alpha_t(j)\beta_t(j)},
$$
$$
\xi_t(i,j)=p(s_t=i,s_{t+1}=j\mid x_{1{:}T})=
\frac{\alpha_t(i)\,a_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j)}
     {\sum_{p,q}\alpha_t(p)\,a_{pq}\,b_q(x_{t+1})\,\beta_{t+1}(q)}.
$$

M-step: parameter re-estimation  
$$
\pi_i^{\text{new}}=\gamma_1(i),
$$
$$
a_{ij}^{\text{new}}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}
                         {\sum_{t=1}^{T-1}\gamma_t(i)},
$$
Emission parameters: maximize  
$
\sum_{t=1}^{T}\gamma_t(i)\,\log b_i(x_t;\theta_i),
$  
e.g. closed-form for Gaussian mean/covariance, or categorical counts.

#### Decoding (Most Likely State Sequence)  
Viterbi algorithm:
$$
\delta_t(i)=\max_{s_1{:}t-1}p(s_1{:}t=i,x_1{:}t),
\qquad
\delta_t(i)=\bigl[\max_j\delta_{t-1}(j)\,a_{ji}\bigr]\,b_i(x_t).
$$
Back-pointer reconstruction gives $\arg\max_{s_1{:}T}p(s_1{:}T|x_{1{:}T})$.

#### Computational Complexity  
Forward-backward: $\mathcal{O}(ST^2)$ naive, $\mathcal{O}(ST)$ optimized.  
Baum-Welch per iteration: $\mathcal{O}(ST)$ memory, $\mathcal{O}(ST)$ time.

#### Applications  
Speech recognition, POS tagging, biosequence analysis, music transcription.

#### Limitations & Extensions  
• First-order assumption; extend to higher order or hierarchical HMMs.  
• Emission independence; coupled HMM, factorial HMM.  
• Discriminative counterparts: Maximum-Entropy Markov Model, CRF.



---

### 2.3 Probabilistic Context-Free Grammars (PCFG)

#### Definition  
A PCFG augments a context-free grammar $G=(N,\Sigma,R,S)$ with rule probabilities $\theta_{A\rightarrow\beta}$ such that:
$$
\forall A\!\in\!N,\quad\sum_{A\rightarrow\beta\in R}\theta_{A\rightarrow\beta}=1,\quad
\theta_{A\rightarrow\beta}>0.
$$
The probability of a parse tree $t$ is the product of its applied rule probabilities.

#### Generative Process  
1. Start with root symbol $S$.  
2. Recursively expand a non-terminal $A$ by sampling production $A\!\rightarrow\!\beta$ with probability $\theta_{A\rightarrow\beta}$.  
3. Continue until only terminals remain; the yield is the generated string.

#### Inside Algorithm (Expectation of Subspans)  
Let $\alpha_{ij}(A)=p(A\!\Rightarrow^\!* x_i{:}x_j)$ for span $(i,j)$:
$$
\alpha_{ij}(A)=
\sum_{A\rightarrow BC}\theta_{A\rightarrow BC}\sum_{k=i+1}^{j-1}
\alpha_{ik}(B)\,\alpha_{kj}(C)
+\sum_{A\rightarrow a}\theta_{A\rightarrow a}\,\mathbb{I}[x_i=a,\,j=i+1].
$$

#### Outside Algorithm  
$\beta_{ij}(A)=p(S\!\Rightarrow^\!* x_1{:}x_{i-1}\,A\,x_{j{:}n})$:
$$
\beta_{ij}(A)=
\sum_{B\rightarrow AC}\theta_{B\rightarrow AC}
\sum_{k=j+1}^{n}\beta_{kj}(B)\,\alpha_{jk}(C)
+
\sum_{B\rightarrow CB}\theta_{B\rightarrow CB}
\sum_{k=1}^{i-1}\beta_{ik}(B)\,\alpha_{ki}(C).
$$

#### EM (Inside–Outside)  

Expected rule counts  
$$
E_{A\rightarrow BC}=\sum_{i<k<j}
\frac{\beta_{ij}(A)\,\theta_{A\rightarrow BC}\,\alpha_{ik}(B)\,\alpha_{kj}(C)}
     {\alpha_{0n}(S)},
$$
$$
E_{A\rightarrow a}=\sum_{i}
\frac{\beta_{i\,i+1}(A)\,\theta_{A\rightarrow a}\,\mathbb{I}[x_i=a]}
     {\alpha_{0n}(S)}.
$$

M-step  
$$
\theta_{A\rightarrow\beta}^{\text{new}}=
\frac{E_{A\rightarrow\beta}}
     {\sum_{\beta'}E_{A\rightarrow\beta'}}.
$$

#### Probabilistic CKY Parsing (Viterbi)  
Replace summations in inside algorithm with maximization to obtain maximum-probability parse.

#### Computational Complexity  
Inside–outside: $\mathcal{O}(|N|\,n^3)$ time, $\mathcal{O}(|N|\,n^2)$ space for sentence length $n$.

#### Smoothing & Sparse Estimation  
• Add-$\lambda$, Good–Turing, or hierarchical Dirichlet priors.  
• Latent-annotation (Petrov) to increase context sensitivity without exploding grammar size.

#### Applications  
Natural-language parsing, code generation, RNA secondary structure prediction.

#### Limitations & Extensions  
• Independence of subtrees ⇒ limited context; addressed by lexicalized PCFG, head-driven models.  
• Symbolic non-terminals; neural PCFGs encode rule probabilities via neural networks enabling gradient-based training.

---

End of Chapter 2