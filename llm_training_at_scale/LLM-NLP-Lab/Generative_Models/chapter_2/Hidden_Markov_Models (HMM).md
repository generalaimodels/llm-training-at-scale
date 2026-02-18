## Hidden Markov Models (HMM)

### 1 Concise Definition
A Hidden Markov Model is a doubly stochastic generative process where an unobserved, first-order Markov chain $ \{s_t\}_{t=1}^{T} $ ($ s_t\in\{1{:}S\} $) produces an observable sequence $ \{x_t\}_{t=1}^{T} $ through conditionally independent emissions. The complete parameter set is  
$$
\Theta=\bigl\{\pi_i,\;a_{ij},\;b_i(\cdot)\bigr\}_{i,j=1}^{S},
$$  
with initial distribution $ \pi $, transition matrix $ A $, and state-indexed emission laws $ \{b_i\} $.

---

### 2 Model Specification

#### 2.1 Components
* Initial probabilities $ \pi_i=p(s_1=i) $, $ \sum_i \pi_i=1 $  
* Transition probabilities $ a_{ij}=p(s_{t}=j\mid s_{t-1}=i) $, $ \sum_j a_{ij}=1 $  
* Emission distribution $ b_i(x)=p(x_t=x\mid s_t=i) $

Both discrete and continuous $ b_i(\cdot) $ are allowed; e.g., categorical or Gaussian, or Gaussian Mixture Models for CD-HMMs.

#### 2.2 Generative Story
1. Sample $ s_1\sim\text{Categorical}(\pi_1,\dots,\pi_S) $.  
2. For $ t=1{:}T $:  
   a. Sample $ x_t\sim b_{s_t}(\cdot) $.  
   b. If $ t<T $, sample $ s_{t+1}\sim\text{Categorical}(a_{s_t1},\dots,a_{s_tS}) $.

#### 2.3 Joint Probability
$$
p(s_{1{:}T},x_{1{:}T}\mid\Theta)
=\pi_{s_1}\,b_{s_1}(x_1)\prod_{t=2}^{T}a_{s_{t-1}s_t}\,b_{s_t}(x_t).
$$

---

### 3 Likelihood Evaluation (Forward Algorithm)

Define forward variables  
$$
\alpha_t(i)=p(x_{1{:}t},s_t=i\mid\Theta).
$$  

Recursion  
$$
\alpha_1(i)=\pi_i\,b_i(x_1),
\qquad
\alpha_t(i)=b_i(x_t)\sum_{j=1}^{S}\alpha_{t-1}(j)\,a_{ji}.
$$  

Total likelihood  
$$
p(x_{1{:}T}\mid\Theta)=\sum_{i=1}^{S}\alpha_T(i).
$$  

Time complexity $ \mathcal{O}(ST) $, space $ \mathcal{O}(S) $ with scaling.

---

### 4 Posterior Inference (Forward–Backward)

Backward variables  
$$
\beta_T(i)=1,
\qquad
\beta_t(i)=\sum_{j=1}^{S} a_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j).
$$  

Posterior state marginals  
$$
\gamma_t(i)=p(s_t=i\mid x_{1{:}T},\Theta)
=\frac{\alpha_t(i)\beta_t(i)}{\sum_{j}\alpha_t(j)\beta_t(j)}.
$$  

Posterior transition marginals  
$$
\xi_t(i,j)=p(s_t=i,s_{t+1}=j\mid x_{1{:}T},\Theta)
=\frac{\alpha_t(i)\,a_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j)}
       {\sum_{p,q}\alpha_t(p)\,a_{pq}\,b_q(x_{t+1})\,\beta_{t+1}(q)}.
$$  

---

### 5 Decoding (Most Probable Path)

Viterbi dynamic programming  
$$
\delta_1(i)=\log\pi_i+\log b_i(x_1),
$$
$$
\delta_t(i)=\log b_i(x_t)+\max_{j}\bigl[\delta_{t-1}(j)+\log a_{ji}\bigr],
$$
with back-pointers for traceback. Complexity $ \mathcal{O}(ST) $.

---

### 6 Parameter Learning (Baum–Welch / EM)

E-step: compute $ \gamma_t(i) $ and $ \xi_t(i,j) $ via forward–backward.

M-step:

Initial distribution
$$
\pi_i^{\text{new}}=\gamma_1(i).
$$

Transitions
$$
a_{ij}^{\text{new}}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}.
$$

Emissions  
*Discrete*:  
$$
b_i^{\text{new}}(v)=\frac{\sum_{t:x_t=v}\gamma_t(i)}{\sum_{t}\gamma_t(i)}.
$$
*Gaussian*:  
$$
\mu_i^{\text{new}}=\frac{\sum_{t}\gamma_t(i)\,x_t}{\sum_{t}\gamma_t(i)},
\qquad
\Sigma_i^{\text{new}}=\frac{\sum_{t}\gamma_t(i)\,(x_t-\mu_i^{\text{new}})(x_t-\mu_i^{\text{new}})^{\!\top}}
                           {\sum_{t}\gamma_t(i)}.
$$

Monotonic increase of log-likelihood is guaranteed; convergence is local.

---

### 7 Numerical Stability

* Use log-space and log-sum-exp for $ \alpha,\beta,\delta $.  
* Scale forward/backward factors: $ c_t=1/\sum_i\alpha_t(i) $, keep running log-likelihood $ -\sum_t\log c_t $.  
* Regularize Gaussian covariances by $ \Sigma_i\leftarrow\Sigma_i+\lambda I $.

---

### 8 Complexity Analysis

| Task | Time | Space |
|------|------|-------|
| Forward / Backward | $ \mathcal{O}(ST) $ | $ \mathcal{O}(S) $ (with scaling) |
| Viterbi | $ \mathcal{O}(ST) $ | $ \mathcal{O}(S) $ |
| One EM iteration | $ \mathcal{O}(ST) $ | $ \mathcal{O}(S) $ |

For Gaussian emissions add $ \mathcal{O}(d^2) $ per state for covariance operations.

---

### 9 Variants and Extensions

* Higher-order HMMs ($ n $th-order Markov in hidden chain).  
* Continuous Density HMM (CD-HMM) with GMM or neural emissions.  
* Left-to-Right (Bakis) topology for speech.  
* Hidden Semi-Markov Model (HSMM): explicit state duration $ p(d) $.  
* Factorial / Coupled HMMs: multiple interacting hidden chains.  
* Input–Output HMM (IO-HMM): transitions conditioned on exogenous input.  
* Hierarchical HMM (HHMM): multi-level abstraction.  
* Bayesian HMM: Dirichlet priors, Gibbs / Variational inference.  
* HDP-HMM (Infinite HMM): non-parametric number of states via Hierarchical Dirichlet Process.  
* Neural HMM: parameterize $ a_{ij},b_i $ with deep nets; train by forward-backward on logits.

---

### 10 Model Selection & Regularization

* Number of states $ S $: choose via BIC, held-out likelihood, or HDP-HMM.  
* Sparse transitions: add Dirichlet-L1 priors or entropy penalty $ \sum_{i,j}a_{ij}\log a_{ij} $.  
* Early-stopping EM on validation likelihood.  
* Merge-Split heuristics for state refinement.

---

### 11 Parallel & Large-Scale Training

* Mini-batch EM / stochastic VB: update sufficient statistics with learning rate $ \rho_t $.  
* Sequence parallelism: distribute sequences across GPUs; aggregate statistics via all-reduce.  
* Vectorized kernels: compute $ b_i(x_t) $ for all $ i,t $ in batched BLAS.  
* FPGA / ASIC for real-time speech decoding.

---

### 12 Applications

* Speech recognition (acoustic modeling).  
* POS tagging, named-entity recognition (NLP).  
* Gene prediction, CpG island detection (bioinformatics).  
* Planetary rover localization (robotics).  
* Financial regime switching.  
* Music score following, tempo tracking.  
* Network intrusion detection.

---

### 13 Statistical Properties

* Identifiability up to state permutation.  
* EM achieves $ \mathcal{O}(\sqrt{N}) $ parameter consistency when model is correct.  
* Mixing time of hidden chain influences variance of estimators.  
* Viterbi path may differ from marginal MAP states; posterior decoding maintains minimal Bayes risk per-frame.

---

### 14 End-to-End Implementation Pipeline

1. **Data preprocessing**  
   • Quantize or MFCC extraction for audio; normalization for continuous data.

2. **Initialize parameters**  
   • $ k $-means on observations → initial state assignments.  
   • Uniform or data-driven $ a_{ij} $.  
   • Randomized restarts to avoid local minima.

3. **EM training**  
   • Iterate until $ |\Delta\log p(x)|<10^{-4} $ or max epochs.  
   • Monitor held-out log-likelihood.

4. **Model validation**  
   • Compute perplexity $ \exp\!\bigl[-\frac{1}{N}\sum_n\log p(x^{(n)})\bigr] $.  
   • Inspect state occupancy $ \sum_t\gamma_t(i) $; prune unused states.

5. **Decoding / inference**  
   • Choose Viterbi for hard labeling, posterior marginals for soft decisions.  
   • For streaming, use online forward filtering with pruning beams.

6. **Deployment**  
   • Convert parameters to fixed-point if required.  
   • Optimize matrix operations, precompute $\log b_i(x)$ tables for discrete symbols.

---

End.