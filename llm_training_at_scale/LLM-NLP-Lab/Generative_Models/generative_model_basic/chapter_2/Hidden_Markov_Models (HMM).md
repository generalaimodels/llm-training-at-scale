---

# Hidden Markov Models (HMM)

---

## Definition

A **Hidden Markov Model (HMM)** is a generative probabilistic model for sequential data, where the system is assumed to be a Markov process with unobserved (hidden) discrete states. Each state emits an observation according to a state-dependent probability distribution. HMMs are foundational in modeling time series, speech, and biological sequences.

---

## Mathematical Formulation

Let $X = (x_1, x_2, \ldots, x_T)$ denote the observed sequence, and $Z = (z_1, z_2, \ldots, z_T)$ the sequence of hidden states, where $z_t \in \{1, \ldots, K\}$ for $K$ possible states.

### Model Parameters

- Initial state distribution: $\pi_k = p(z_1 = k)$
- State transition matrix: $A_{ij} = p(z_{t+1} = j \mid z_t = i)$
- Emission distribution: $B_k(x) = p(x_t = x \mid z_t = k)$

The complete parameter set is $\Theta = \{\pi, A, B\}$.

### Joint Distribution

The joint probability of the observed and hidden sequences is:

$$
p(X, Z \mid \Theta) = \pi_{z_1} B_{z_1}(x_1) \prod_{t=2}^T A_{z_{t-1}, z_t} B_{z_t}(x_t)
$$

The marginal likelihood of the observed sequence is:

$$
p(X \mid \Theta) = \sum_{Z} p(X, Z \mid \Theta)
$$

---

## Step-by-Step Explanation

### 1. **Model Specification**

- **Markov Assumption:** The hidden state at time $t$ depends only on the state at $t-1$:
  $$
  p(z_t \mid z_{1:t-1}) = p(z_t \mid z_{t-1})
  $$
- **Emission Independence:** The observation at time $t$ depends only on the current hidden state:
  $$
  p(x_t \mid z_{1:t}, x_{1:t-1}) = p(x_t \mid z_t)
  $$

---

### 2. **Inference Tasks**

#### a. **Likelihood Computation (Forward Algorithm)**

Compute $p(X \mid \Theta)$ efficiently using dynamic programming.

- **Forward variable:**
  $$
  \alpha_t(k) = p(x_1, \ldots, x_t, z_t = k \mid \Theta)
  $$
- **Recursion:**
  - Initialization:
    $$
    \alpha_1(k) = \pi_k B_k(x_1)
    $$
  - Induction:
    $$
    \alpha_{t}(j) = \left[ \sum_{i=1}^K \alpha_{t-1}(i) A_{ij} \right] B_j(x_t)
    $$
  - Termination:
    $$
    p(X \mid \Theta) = \sum_{k=1}^K \alpha_T(k)
    $$

#### b. **Decoding (Viterbi Algorithm)**

Find the most probable sequence of hidden states:

$$
Z^* = \arg\max_{Z} p(Z \mid X, \Theta)
$$

- **Viterbi variable:**
  $$
  \delta_t(k) = \max_{z_{1:t-1}} p(z_{1:t-1}, z_t = k, x_{1:t} \mid \Theta)
  $$
- **Recursion:**
  - Initialization:
    $$
    \delta_1(k) = \pi_k B_k(x_1)
    $$
  - Induction:
    $$
    \delta_t(j) = \max_{i=1}^K \left[ \delta_{t-1}(i) A_{ij} \right] B_j(x_t)
    $$
  - Backtracking to recover $Z^*$.

#### c. **Posterior Marginals (Forward-Backward Algorithm)**

Compute the posterior probability of each state at each time:

- **Backward variable:**
  $$
  \beta_t(k) = p(x_{t+1}, \ldots, x_T \mid z_t = k, \Theta)
  $$
- **Recursion:**
  - Initialization:
    $$
    \beta_T(k) = 1
    $$
  - Induction:
    $$
    \beta_t(i) = \sum_{j=1}^K A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)
    $$
- **Posterior:**
  $$
  \gamma_t(k) = p(z_t = k \mid X, \Theta) = \frac{\alpha_t(k) \beta_t(k)}{p(X \mid \Theta)}
  $$

---

### 3. **Parameter Estimation (Baum-Welch / EM Algorithm)**

Given observed data $X$, estimate $\Theta$ by maximizing the likelihood.

#### **E-Step**

- Compute expected sufficient statistics using the forward-backward algorithm:
  - **State occupancy:**
    $$
    \gamma_t(k) = p(z_t = k \mid X, \Theta)
    $$
  - **Transition probability:**
    $$
    \xi_t(i, j) = p(z_t = i, z_{t+1} = j \mid X, \Theta) = \frac{\alpha_t(i) A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)}{p(X \mid \Theta)}
    $$

#### **M-Step**

- Update parameters:
  - Initial state:
    $$
    \pi_k^{\text{new}} = \gamma_1(k)
    $$
  - Transition matrix:
    $$
    A_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
    $$
  - Emission probabilities (discrete case):
    $$
    B_k^{\text{new}}(v) = \frac{\sum_{t=1}^T \gamma_t(k) \mathbb{I}[x_t = v]}{\sum_{t=1}^T \gamma_t(k)}
    $$
    For continuous emissions (e.g., Gaussian), update mean and covariance using weighted sums.

- Iterate E and M steps until convergence.

---

### 4. **Extensions**

- **Continuous Emissions:** $B_k(x)$ is a Gaussian or GMM.
- **Higher-Order HMMs:** State depends on more than one previous state.
- **Input-Output HMMs:** Emissions depend on external inputs.
- **Hierarchical HMMs:** States themselves are HMMs.

---

### 5. **Properties and Limitations**

- **Expressiveness:** Models temporal dependencies and hidden structure.
- **Identifiability:** Model is not identifiable up to permutation of states.
- **Local Optima:** EM can converge to suboptimal solutions; initialization is critical.
- **Scalability:** Forward-backward and Viterbi algorithms are $O(TK^2)$.

---

### 6. **Applications**

- **Speech Recognition:** Phoneme/state modeling.
- **Natural Language Processing:** POS tagging, named entity recognition.
- **Bioinformatics:** Gene prediction, protein secondary structure.
- **Time Series Analysis:** Regime switching, anomaly detection.

---

## Summary Table

| Parameter | Description |
|-----------|-------------|
| $K$ | Number of hidden states |
| $\pi_k$ | Initial state probability |
| $A_{ij}$ | State transition probability |
| $B_k(x)$ | Emission probability for state $k$ |
| $\alpha_t(k)$ | Forward variable |
| $\beta_t(k)$ | Backward variable |
| $\gamma_t(k)$ | Posterior state probability |
| $\xi_t(i, j)$ | Posterior transition probability |

---