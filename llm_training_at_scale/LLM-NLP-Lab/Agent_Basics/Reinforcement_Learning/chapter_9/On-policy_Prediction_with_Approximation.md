
# On-Policy Prediction with Approximation

## 9.1 Value-function Approximation

### Definition
Value-function approximation seeks to estimate the value function $v_\pi(s)$ for a policy $\pi$ when the state space is too large for tabular methods. The value function is approximated by a parameterized function $\hat{v}(s, \mathbf{w})$, where $\mathbf{w}$ is a vector of learnable parameters.

### Equation
$$
\hat{v}(s, \mathbf{w}) \approx v_\pi(s)
$$

### Key Principles
- **Generalization:** Approximate value functions generalize across similar states.
- **Parameterization:** $\hat{v}(s, \mathbf{w})$ can be linear or nonlinear in $\mathbf{w}$.
- **Scalability:** Enables RL in high-dimensional or continuous state spaces.

### Detailed Concept Analysis
- **Linear Approximation:** $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$, where $\mathbf{x}(s)$ is a feature vector.
- **Nonlinear Approximation:** Neural networks or other nonlinear models parameterize $\hat{v}$.

### Importance
- **Essential for large-scale RL:** Tabular methods are infeasible for large $|\mathcal{S}|$.
- **Enables function generalization and transfer.**

### Pros vs Cons
- **Pros:** Scalability, generalization, flexibility.
- **Cons:** Instability, bias, convergence issues.

### Cutting-edge Advances
- Deep RL: Use of deep neural networks for $\hat{v}$.
- Meta-learning for adaptive feature construction.

---

## 9.2 The Prediction Objective (VE)

### Definition
The objective is to minimize the mean squared value error (VE) between the true value function and its approximation.

### Equation
$$
\text{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} d(s) \left[ v_\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2
$$
where $d(s)$ is the state distribution under policy $\pi$.

### Key Principles
- **Projection:** The best approximation is the orthogonal projection of $v_\pi$ onto the space spanned by $\hat{v}$.
- **State Distribution Weighting:** Errors are weighted by $d(s)$.

### Detailed Concept Analysis
- **Empirical Estimation:** In practice, $v_\pi(s)$ is unknown; use sample returns or bootstrapped targets.
- **Optimization:** Gradient-based methods minimize VE.

### Importance
- **Defines learning target for function approximation.**

### Pros vs Cons
- **Pros:** Theoretically grounded, aligns with supervised learning.
- **Cons:** $v_\pi(s)$ is unknown; must use estimates.

### Cutting-edge Advances
- Distributional RL: Minimize error over value distributions, not just means.

---

## 9.3 Stochastic-gradient and Semi-gradient Methods

### Definition
Stochastic-gradient methods update $\mathbf{w}$ using sampled data. Semi-gradient methods ignore the dependency of the target on $\mathbf{w}$ in bootstrapped targets.

### Equations
- **Stochastic Gradient:**
  $$
  \mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ G_t - \hat{v}(S_t, \mathbf{w}) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})
  $$
- **Semi-gradient TD(0):**
  $$
  \mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}) \right] \nabla_\mathbf{w} \hat{v}(S_t, \mathbf{w})
  $$

### Key Principles
- **Bootstrapping:** Use of current value estimates as targets.
- **Semi-gradient:** Only differentiate through $\hat{v}(S_t, \mathbf{w})$.

### Detailed Concept Analysis
- **Bias-variance tradeoff:** Bootstrapping introduces bias but reduces variance.
- **Convergence:** Semi-gradient methods may not converge for nonlinear $\hat{v}$.

### Importance
- **Enables online, incremental learning.**

### Pros vs Cons
- **Pros:** Efficient, scalable, online.
- **Cons:** Potential instability, divergence for nonlinear function approximators.

### Cutting-edge Advances
- Adam, RMSProp optimizers for stability.
- Target networks in deep RL.

---

## 9.4 Linear Methods

### Definition
Linear methods use a linear combination of features to approximate the value function.

### Equation
$$
\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)
$$

### Key Principles
- **Feature Engineering:** Choice of $\mathbf{x}(s)$ is critical.
- **Convexity:** Linear methods yield convex optimization problems.

### Detailed Concept Analysis
- **Update Rule:**
  $$
  \mathbf{w} \leftarrow \mathbf{w} + \alpha \delta_t \mathbf{x}(S_t)
  $$
  where $\delta_t$ is the TD error.

### Importance
- **Simplicity and theoretical guarantees.**

### Pros vs Cons
- **Pros:** Convergence guarantees, interpretability.
- **Cons:** Limited expressiveness.

### Cutting-edge Advances
- Sparse coding, regularization for high-dimensional features.

---

## 9.5 Feature Construction for Linear Methods

### 9.5.1 Polynomials

#### Definition
Polynomial features expand the state representation using monomials up to a certain degree.

#### Equation
$$
\mathbf{x}(s) = [1, s, s^2, \ldots, s^d]
$$

#### Key Principles
- **Nonlinear representation via linear weights.**

#### Detailed Concept Analysis
- **Curse of dimensionality:** Number of features grows rapidly with degree and state dimension.

#### Importance
- **Captures nonlinearities in value function.**

#### Pros vs Cons
- **Pros:** Simple, effective for low dimensions.
- **Cons:** Poor scalability.

#### Cutting-edge Advances
- Automatic degree selection, regularization.

---

### 9.5.2 Fourier Basis

#### Definition
Fourier basis uses sinusoidal functions as features.

#### Equation
$$
\mathbf{x}_k(s) = \cos(\pi k s), \quad k = 0, 1, \ldots, n
$$

#### Key Principles
- **Orthogonality:** Fourier features are orthogonal.

#### Detailed Concept Analysis
- **Captures periodic structure in value function.**

#### Importance
- **Efficient for smooth, periodic value functions.**

#### Pros vs Cons
- **Pros:** Compact, expressive.
- **Cons:** May not suit non-periodic domains.

#### Cutting-edge Advances
- Random Fourier features for kernel approximation.

---

### 9.5.3 Coarse Coding

#### Definition
Coarse coding uses overlapping receptive fields to represent states.

#### Equation
$$
\mathbf{x}_i(s) = 
\begin{cases}
1 & \text{if } s \in \text{region } i \\
0 & \text{otherwise}
\end{cases}
$$

#### Key Principles
- **Overlap:** Each state activates multiple features.

#### Detailed Concept Analysis
- **Generalization:** Smooths value estimates across similar states.

#### Importance
- **Efficient for continuous spaces.**

#### Pros vs Cons
- **Pros:** Local generalization.
- **Cons:** Feature design is nontrivial.

#### Cutting-edge Advances
- Adaptive region placement.

---

### 9.5.4 Tile Coding

#### Definition
Tile coding partitions the state space into overlapping grids (tilings).

#### Equation
$$
\mathbf{x}_i(s) = 
\begin{cases}
1 & \text{if } s \text{ is in tile } i \\
0 & \text{otherwise}
\end{cases}
$$

#### Key Principles
- **Multiple tilings:** Each with different offsets.

#### Detailed Concept Analysis
- **Sparse representation:** Each state activates a small subset of features.

#### Importance
- **Widely used in classic RL benchmarks.**

#### Pros vs Cons
- **Pros:** Fast, scalable.
- **Cons:** Manual design, limited expressiveness.

#### Cutting-edge Advances
- Automated tiling selection.

---

### 9.5.5 Radial Basis Functions (RBFs)

#### Definition
RBFs use Gaussian-like functions centered at various points.

#### Equation
$$
\mathbf{x}_i(s) = \exp\left( -\frac{||s - c_i||^2}{2\sigma^2} \right)
$$

#### Key Principles
- **Locality:** Each feature responds to a local region.

#### Detailed Concept Analysis
- **Smooth generalization:** Value function is smooth in state space.

#### Importance
- **Effective for continuous, smooth value functions.**

#### Pros vs Cons
- **Pros:** Flexible, smooth.
- **Cons:** Scalability with dimension.

#### Cutting-edge Advances
- Adaptive centers and widths.

---

## 9.6 Selecting Step-Size Parameters Manually

### Definition
Step-size ($\alpha$) controls the learning rate in parameter updates.

### Equation
$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta_t \mathbf{x}(S_t)
$$

### Key Principles
- **Tradeoff:** Large $\alpha$ speeds learning but risks instability.

### Detailed Concept Analysis
- **Manual tuning:** Empirical selection based on performance.

### Importance
- **Critical for convergence and speed.**

### Pros vs Cons
- **Pros:** Simple.
- **Cons:** Labor-intensive, suboptimal.

### Cutting-edge Advances
- Adaptive step-size algorithms (e.g., Adam, RMSProp).

---

## 9.7 Nonlinear Function Approximation: Artificial Neural Networks

### Definition
Neural networks parameterize $\hat{v}(s, \mathbf{w})$ with nonlinear transformations.

### Equation
$$
\hat{v}(s, \mathbf{w}) = f(s; \mathbf{w})
$$
where $f$ is a neural network.

### Key Principles
- **Universal approximation:** Can represent any continuous function.

### Detailed Concept Analysis
- **Backpropagation:** Used for gradient computation.
- **Overfitting:** Risk with limited data.

### Importance
- **Enables deep RL, high-dimensional state spaces.**

### Pros vs Cons
- **Pros:** High expressiveness.
- **Cons:** Instability, requires large data.

### Cutting-edge Advances
- Deep RL architectures (DQN, DDPG, etc.).
- Regularization, normalization techniques.

---

## 9.8 Least-Squares TD

### Definition
Least-Squares Temporal Difference (LSTD) solves for $\mathbf{w}$ by minimizing the mean squared TD error in closed form.

### Equation
$$
\mathbf{w} = \mathbf{A}^{-1} \mathbf{b}
$$
where
$$
\mathbf{A} = \mathbb{E}[\mathbf{x}(S_t)(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1}))^\top]
$$
$$
\mathbf{b} = \mathbb{E}[R_{t+1} \mathbf{x}(S_t)]
$$

### Key Principles
- **Batch method:** Uses all data at once.

### Detailed Concept Analysis
- **Sample-based estimation:** $\mathbf{A}$ and $\mathbf{b}$ estimated from data.

### Importance
- **Efficient, stable for linear function approximation.**

### Pros vs Cons
- **Pros:** No step-size tuning, stable.
- **Cons:** Computationally expensive for large feature sets.

### Cutting-edge Advances
- Incremental LSTD, regularized LSTD.

---

## 9.9 Memory-based Function Approximation

### Definition
Uses stored experiences to approximate the value function (e.g., k-nearest neighbors).

### Equation
$$
\hat{v}(s) = \frac{1}{k} \sum_{i=1}^k v(s_i)
$$
where $s_i$ are the $k$ nearest stored states to $s$.

### Key Principles
- **Instance-based learning:** No explicit parameterization.

### Detailed Concept Analysis
- **Nonparametric:** Grows with data.

### Importance
- **No need for feature engineering.**

### Pros vs Cons
- **Pros:** Flexible, adaptive.
- **Cons:** Memory and computation scale with data.

### Cutting-edge Advances
- Efficient nearest-neighbor search, memory compression.

---

## 9.10 Kernel-based Function Approximation

### Definition
Uses kernel functions to weigh contributions from stored samples.

### Equation
$$
\hat{v}(s) = \sum_{i=1}^N K(s, s_i) v(s_i)
$$
where $K$ is a kernel function.

### Key Principles
- **Smooth interpolation:** Kernel defines similarity.

### Detailed Concept Analysis
- **Bandwidth selection:** Critical for performance.

### Importance
- **Powerful for smooth value functions.**

### Pros vs Cons
- **Pros:** Nonparametric, flexible.
- **Cons:** Computationally intensive.

### Cutting-edge Advances
- Random feature approximations, scalable kernel methods.

---

## 9.11 Looking Deeper at On-policy Learning: Interest and Emphasis

### Definition
Interest and emphasis modify the learning update to prioritize certain states.

### Equation
Modified TD update:
$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha M_t \delta_t \mathbf{x}(S_t)
$$
where $M_t$ is the emphasis/interest.

### Key Principles
- **Prioritization:** Focus learning on important states.

### Detailed Concept Analysis
- **Emphatic TD:** Adjusts updates based on visitation and interest.

### Importance
- **Improves learning in off-policy and rare-state scenarios.**

### Pros vs Cons
- **Pros:** Better sample efficiency.
- **Cons:** More complex update rules.

### Cutting-edge Advances
- Emphatic TD($\lambda$), prioritized experience replay.

---