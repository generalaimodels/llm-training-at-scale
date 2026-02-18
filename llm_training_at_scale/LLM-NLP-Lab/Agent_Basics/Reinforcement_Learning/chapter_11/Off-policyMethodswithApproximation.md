
# Off-policy Methods with Function Approximation

---

## 11.1 Semi-gradient Methods

### Definition
Semi-gradient methods are a class of algorithms in reinforcement learning (RL) that use the gradient of the value function with respect to the parameters, **ignoring** the dependency of the target on the parameters. This is crucial when using function approximation (e.g., neural networks, linear models) for value estimation.

### Pertinent Equations

- **Value Function Approximation**:  
  $$ v_\pi(s; \mathbf{w}) \approx v_\pi(s) $$
  where $ \mathbf{w} $ are the parameters.

- **Semi-gradient Update (TD(0))**:  
  $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma v(S_{t+1}; \mathbf{w}_t) - v(S_t; \mathbf{w}_t) \right] \nabla_\mathbf{w} v(S_t; \mathbf{w}_t) $$

### Key Principles

- **Partial Derivative**: Only the derivative of the estimated value at the current state is considered, not the target.
- **Bootstrapping**: The target includes the next state's value estimate, introducing bias but reducing variance.

### Detailed Concept Analysis

- **On-policy vs. Off-policy**:  
  Semi-gradient methods can be used in both settings, but off-policy introduces instability due to the mismatch between behavior and target policies.
- **Function Approximation**:  
  When $ v(s; \mathbf{w}) $ is nonlinear (e.g., neural networks), the semi-gradient ignores the effect of $ \mathbf{w} $ on the target.

### Importance

- **Computational Simplicity**: Reduces computational complexity by not backpropagating through the target.
- **Practicality**: Widely used in deep RL (e.g., DQN).

### Pros vs. Cons

- **Pros**:
  - Simpler implementation.
  - Lower computational cost.
- **Cons**:
  - Can diverge, especially off-policy with function approximation (see Deadly Triad).

### Recent Developments

- **Stabilization Techniques**: Target networks, experience replay, and double Q-learning are used to mitigate divergence.

---

## 11.2 Examples of Off-policy Divergence

### Definition

Off-policy divergence refers to the phenomenon where learning algorithms become unstable or diverge when the policy used to generate data (behavior policy) differs from the policy being evaluated or improved (target policy), especially with function approximation.

### Pertinent Equations

- **Off-policy TD(0) Update**:  
  $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \rho_t \delta_t \nabla_\mathbf{w} v(S_t; \mathbf{w}_t) $$
  where $ \rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)} $ is the importance sampling ratio.

### Key Principles

- **Importance Sampling**: Corrects for the difference between behavior and target policies.
- **Instability**: The combination of bootstrapping, off-policy learning, and function approximation can cause divergence.

### Detailed Concept Analysis

- **Baird’s Counterexample**: Demonstrates divergence in linear function approximation with off-policy TD learning.
- **Role of Importance Sampling**: High variance in $ \rho_t $ can exacerbate instability.

### Importance

- **Critical for RL**: Understanding divergence is essential for designing stable off-policy algorithms.

### Pros vs. Cons

- **Pros**:
  - Enables learning from arbitrary data sources.
- **Cons**:
  - High risk of divergence without careful algorithmic design.

### Recent Developments

- **Safe Off-policy Algorithms**: Emphatic TD, Gradient-TD methods, and Retrace($\lambda$) address divergence.

---

## 11.3 The Deadly Triad

### Definition

The "Deadly Triad" refers to the combination of three elements in RL that, when used together, can cause instability or divergence:

1. **Function Approximation**
2. **Bootstrapping**
3. **Off-policy Training**

### Pertinent Equations

- **General TD Update**:  
  $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma v(S_{t+1}; \mathbf{w}_t) - v(S_t; \mathbf{w}_t) \right] \nabla_\mathbf{w} v(S_t; \mathbf{w}_t) $$

### Key Principles

- **Interaction Effects**: Each component is safe alone or in pairs, but all three together can cause divergence.

### Detailed Concept Analysis

- **Function Approximation**: Generalizes across states, can propagate errors.
- **Bootstrapping**: Updates estimates based on other estimates, not just actual returns.
- **Off-policy**: Data distribution mismatch increases instability.

### Importance

- **Central Challenge**: The Deadly Triad is a fundamental barrier to stable, scalable RL.

### Pros vs. Cons

- **Pros**:
  - Each component is essential for practical RL.
- **Cons**:
  - Their combination is hazardous.

### Recent Developments

- **Algorithmic Solutions**: Gradient-TD, Emphatic TD, and target networks.

---

## 11.4 Linear Value-function Geometry

### Definition

Linear value-function approximation represents the value function as a linear combination of features.

### Pertinent Equations

- **Linear Approximation**:  
  $$ v(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) $$
  where $ \mathbf{x}(s) $ is the feature vector for state $ s $.

### Key Principles

- **Projection**: The TD update projects the Bellman target onto the space spanned by the features.

### Detailed Concept Analysis

- **Geometry**: The set of representable value functions forms a subspace; TD learning projects onto this subspace.
- **Oblique Projection**: Off-policy TD does not perform orthogonal projection, leading to bias.

### Importance

- **Analytical Tractability**: Linear models allow for precise analysis of convergence and divergence.

### Pros vs. Cons

- **Pros**:
  - Simplicity, interpretability.
- **Cons**:
  - Limited expressiveness.

### Recent Developments

- **Extensions**: Use of richer feature sets, kernel methods.

---

## 11.5 Gradient Descent in the Bellman Error

### Definition

Gradient descent in the Bellman error seeks to minimize the mean squared Bellman error (MSBE) directly.

### Pertinent Equations

- **Bellman Error**:  
  $$ \text{BE}(s) = r(s) + \gamma \mathbb{E}_{s'}[v(s'; \mathbf{w})] - v(s; \mathbf{w}) $$
- **MSBE Objective**:  
  $$ J(\mathbf{w}) = \mathbb{E}_\mu \left[ \left( r(s) + \gamma \mathbb{E}_{s'}[v(s'; \mathbf{w})] - v(s; \mathbf{w}) \right)^2 \right] $$

### Key Principles

- **Direct Minimization**: Unlike TD, which minimizes projected Bellman error, this approach targets the true Bellman error.

### Detailed Concept Analysis

- **Non-stationarity**: The target depends on $ \mathbf{w} $, making optimization challenging.
- **Double Sampling Problem**: Estimating the gradient unbiasedly requires two independent samples of $ s' $.

### Importance

- **Theoretical Appeal**: Directly addresses the true error.

### Pros vs. Cons

- **Pros**:
  - Conceptually clean.
- **Cons**:
  - Impractical due to double sampling requirement.

### Recent Developments

- **Alternative Objectives**: Projected Bellman error, Gradient-TD methods.

---

## 11.6 The Bellman Error is Not Learnable

### Definition

The Bellman error is not directly learnable with standard stochastic gradient descent due to the double sampling problem.

### Pertinent Equations

- **Gradient of MSBE**:  
  $$ \nabla_\mathbf{w} J(\mathbf{w}) = 2 \mathbb{E}_\mu \left[ \text{BE}(s) \left( \gamma \mathbb{E}_{s'}[\nabla_\mathbf{w} v(s'; \mathbf{w})] - \nabla_\mathbf{w} v(s; \mathbf{w}) \right) \right] $$

### Key Principles

- **Double Sampling Problem**: The expectation over $ s' $ inside the squared term prevents unbiased estimation from a single sample.

### Detailed Concept Analysis

- **Unbiased Gradient Estimation**: Requires two independent next-state samples, which is infeasible in most RL settings.

### Importance

- **Algorithmic Limitation**: Necessitates alternative approaches for stable learning.

### Pros vs. Cons

- **Pros**:
  - None in practice.
- **Cons**:
  - Not implementable in standard RL environments.

### Recent Developments

- **Gradient-TD Methods**: Circumvent the double sampling problem.

---

## 11.7 Gradient-TD Methods

### Definition

Gradient-TD methods are a family of algorithms that perform true stochastic gradient descent on a surrogate objective related to the projected Bellman error, ensuring convergence even off-policy.

### Pertinent Equations

- **Mean Squared Projected Bellman Error (MSPBE)**:  
  $$ \text{MSPBE}(\mathbf{w}) = \mathbb{E}_\mu \left[ \left( v(s; \mathbf{w}) - \Pi T v(s; \mathbf{w}) \right)^2 \right] $$
  where $ \Pi $ is the projection operator.

- **GTD2 Update**:  
  $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left( \delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1} (\mathbf{x}_t^\top \mathbf{h}_t) \right) $$
  $$ \mathbf{h}_{t+1} = \mathbf{h}_t + \beta \left( \delta_t - \mathbf{x}_t^\top \mathbf{h}_t \right) \mathbf{x}_t $$

### Key Principles

- **Two-timescale Updates**: Auxiliary weights $ \mathbf{h} $ estimate expected TD error.
- **Convergence Guarantees**: Proven convergence under general conditions.

### Detailed Concept Analysis

- **Surrogate Objective**: MSPBE is minimized instead of MSBE.
- **Auxiliary Variables**: $ \mathbf{h} $ tracks the expected TD error.

### Importance

- **Stable Off-policy Learning**: Enables safe use of function approximation and off-policy data.

### Pros vs. Cons

- **Pros**:
  - Convergent under general conditions.
- **Cons**:
  - More complex, slower convergence.

### Recent Developments

- **Extensions**: True Online GTD, nonlinear function approximation.

---

## 11.8 Emphatic-TD Methods

### Definition

Emphatic-TD methods modify the TD update by weighting updates according to the "emphasis" of each state, correcting for the distribution mismatch in off-policy learning.

### Pertinent Equations

- **Emphatic Weighting**:  
  $$ F_t = \gamma \rho_{t-1} F_{t-1} + i(S_t) $$
  where $ i(S_t) $ is the interest function.

- **Emphatic TD(0) Update**:  
  $$ \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha F_t \rho_t \delta_t \nabla_\mathbf{w} v(S_t; \mathbf{w}_t) $$

### Key Principles

- **State Emphasis**: States are weighted by their long-term importance under the target policy.

### Detailed Concept Analysis

- **Distribution Correction**: Emphatic weighting aligns the update distribution with the target policy.

### Importance

- **Provable Convergence**: Guarantees convergence even with off-policy data and function approximation.

### Pros vs. Cons

- **Pros**:
  - Stable, convergent.
- **Cons**:
  - More complex, requires tracking emphasis.

### Recent Developments

- **Practical Implementations**: Used in large-scale RL with function approximation.

---

## 11.9 Reducing Variance

### Definition

Variance reduction techniques are methods to decrease the variance of updates in off-policy RL, improving stability and learning speed.

### Pertinent Equations

- **Importance Sampling with Truncation**:  
  $$ \rho_t = \min \left( c, \frac{\pi(A_t|S_t)}{b(A_t|S_t)} \right) $$

- **Retrace($\lambda$) Operator**:  
  $$ Q^{\text{Retrace}}(s, a) = Q(s, a) + \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t \left( \prod_{i=1}^t c_i \right) \delta_t \right] $$
  where $ c_i = \lambda \min \left( 1, \frac{\pi(a_i|s_i)}{b(a_i|s_i)} \right) $.

### Key Principles

- **Truncation/Clipping**: Limits the magnitude of importance weights.
- **Eligibility Traces**: Combines multi-step returns for bias-variance tradeoff.

### Detailed Concept Analysis

- **Bias-Variance Tradeoff**: Truncation introduces bias but reduces variance.
- **Multi-step Methods**: Use more data per update, further reducing variance.

### Importance

- **Practical Stability**: Essential for deep RL and large-scale off-policy learning.

### Pros vs. Cons

- **Pros**:
  - Improved stability, faster learning.
- **Cons**:
  - Potential bias.

### Recent Developments

- **Advanced Operators**: V-trace, Tree-backup, and other safe multi-step off-policy algorithms.

---