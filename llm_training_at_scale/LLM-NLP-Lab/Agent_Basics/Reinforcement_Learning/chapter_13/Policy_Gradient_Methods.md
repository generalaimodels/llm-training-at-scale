
# 13. Policy Gradient Methods

---

## 13.1 Policy Approximation and its Advantages

### Definition

- **Policy Approximation**: The process of representing the policy $\pi(a|s;\theta)$, which maps states $s$ to action probabilities $a$, using a parameterized function (typically a neural network) with parameters $\theta$.

### Pertinent Equations

- **Parameterized Policy**:  
  $$ \pi(a|s; \theta) $$
- **Objective Function (Expected Return)**:  
  $$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] $$
  where $\tau$ is a trajectory and $R(\tau)$ is the cumulative reward.

### Key Principles

- **Direct Policy Optimization**: Optimize the policy directly rather than the value function.
- **Stochastic Policies**: Enables exploration by sampling actions.
- **Function Approximation**: Neural networks or other differentiable models are used for scalability.

### Detailed Concept Analysis

- **Expressiveness**: Neural networks can approximate complex, high-dimensional policies.
- **Generalization**: Parameter sharing across states enables generalization to unseen states.
- **Continuous Actions**: Policy approximation naturally extends to continuous action spaces.

### Importance

- **Scalability**: Handles large or continuous state/action spaces.
- **Flexibility**: Can represent both deterministic and stochastic policies.

### Pros vs Cons

- **Pros**:
  - Handles high-dimensional, continuous spaces.
  - Enables end-to-end learning.
- **Cons**:
  - Prone to high variance in gradient estimates.
  - May require large amounts of data.

### Cutting-Edge Advances

- **Transformer-based Policies**: Improved expressiveness for sequential decision-making.
- **Implicit Policies**: Non-explicit density models for complex action distributions.

---

## 13.2 The Policy Gradient Theorem

### Definition

- **Policy Gradient Theorem**: Provides a formal expression for the gradient of the expected return with respect to policy parameters.

### Pertinent Equations

- **Policy Gradient**:  
  $$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right] $$

### Key Principles

- **Likelihood Ratio Trick**: Converts the gradient of an expectation into an expectation of a gradient.
- **Credit Assignment**: Uses $Q^{\pi_\theta}(s, a)$ to assign credit to actions.

### Detailed Concept Analysis

- **Unbiased Gradient Estimate**: The theorem provides an unbiased estimator for the policy gradient.
- **Sample-based Estimation**: Gradients can be estimated using samples from the policy.

### Importance

- **Foundation for Policy Optimization**: Underpins all policy gradient algorithms.
- **General Applicability**: Valid for any differentiable, parameterized policy.

### Pros vs Cons

- **Pros**:
  - Theoretically sound.
  - Applicable to both discrete and continuous actions.
- **Cons**:
  - High variance in gradient estimates.
  - Requires accurate estimation of $Q^{\pi_\theta}(s, a)$.

### Cutting-Edge Advances

- **Variance Reduction Techniques**: Use of baselines, advantage functions, and control variates.
- **Trust Region Methods**: Constrain policy updates for stability (e.g., TRPO, PPO).

---

## 13.3 REINFORCE: Monte Carlo Policy Gradient

### Definition

- **REINFORCE**: A Monte Carlo policy gradient algorithm that updates policy parameters using complete episodes.

### Pertinent Equations

- **REINFORCE Update Rule**:  
  $$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t $$
  where $G_t$ is the return from time $t$.

### Key Principles

- **Monte Carlo Estimation**: Uses sampled returns $G_t$ as unbiased estimates of $Q^{\pi_\theta}(s_t, a_t)$.
- **Episodic Learning**: Updates occur after complete episodes.

### Detailed Concept Analysis

- **Unbiased but High Variance**: Estimates are unbiased but can have high variance due to episodic returns.
- **No Bootstrapping**: Does not use value function estimates.

### Importance

- **Simplicity**: Conceptually and implementationally simple.
- **Baseline for Comparison**: Serves as a reference for more advanced methods.

### Pros vs Cons

- **Pros**:
  - Unbiased gradient estimates.
  - Simple to implement.
- **Cons**:
  - High variance.
  - Inefficient for long episodes.

### Cutting-Edge Advances

- **Batch REINFORCE**: Uses mini-batches for more stable updates.
- **Variance Reduction**: Incorporation of baselines (see 13.4).

---

## 13.4 REINFORCE with Baseline

### Definition

- **REINFORCE with Baseline**: Extends REINFORCE by subtracting a baseline $b(s_t)$ from the return to reduce variance.

### Pertinent Equations

- **Baseline-augmented Update**:  
  $$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t)) $$

### Key Principles

- **Variance Reduction**: Subtracting a baseline does not introduce bias but reduces variance.
- **Optimal Baseline**: The value function $V^{\pi_\theta}(s_t)$ is often used as the baseline.

### Detailed Concept Analysis

- **Baseline Choice**: $b(s_t)$ can be a constant, moving average, or learned value function.
- **Advantage Function**: $A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$ is a common choice.

### Importance

- **Improved Sample Efficiency**: Lower variance leads to faster learning.
- **Foundation for Actor-Critic**: Baseline learning leads to actor-critic methods.

### Pros vs Cons

- **Pros**:
  - Lower variance.
  - Faster convergence.
- **Cons**:
  - Requires baseline estimation.
  - Baseline bias if not estimated accurately.

### Cutting-Edge Advances

- **Learned Baselines**: Neural networks for adaptive baselines.
- **Generalized Advantage Estimation (GAE)**: Further variance reduction.

---

## 13.5 Actor–Critic Methods

### Definition

- **Actor–Critic**: Combines policy (actor) and value function (critic) learning. The actor updates the policy, while the critic estimates the value function.

### Pertinent Equations

- **Actor Update**:  
  $$ \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t $$
- **Critic Update**:  
  $$ w \leftarrow w + \beta \nabla_w \left( \hat{V}_w(s_t) - G_t \right)^2 $$
  where $\hat{A}_t$ is the estimated advantage.

### Key Principles

- **Two Networks**: Separate parameterizations for actor and critic.
- **Bootstrapping**: Critic uses temporal-difference (TD) learning.

### Detailed Concept Analysis

- **Advantage Estimation**: $\hat{A}_t$ can be computed using TD error or GAE.
- **Stability**: Critic reduces variance, actor improves policy.

### Importance

- **Sample Efficiency**: More efficient than pure policy gradient.
- **Scalability**: Handles large-scale problems.

### Pros vs Cons

- **Pros**:
  - Lower variance.
  - Online, incremental updates.
- **Cons**:
  - Potential instability due to coupled learning.
  - Sensitive to hyperparameters.

### Cutting-Edge Advances

- **Asynchronous Methods (A3C/A2C)**: Parallelized actor-critic.
- **Proximal Policy Optimization (PPO)**: Stable, clipped policy updates.

---

## 13.6 Policy Gradient for Continuing Problems

### Definition

- **Continuing Problems**: Tasks without terminal states, requiring average or discounted reward formulations.

### Pertinent Equations

- **Average Reward Objective**:  
  $$ J(\theta) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T-1} r_t \right] $$
- **Discounted Reward Objective**:  
  $$ J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] $$

### Key Principles

- **Stationarity**: Policy must perform well over an infinite horizon.
- **Discount Factor**: $\gamma$ controls the trade-off between immediate and future rewards.

### Detailed Concept Analysis

- **Ergodicity**: Ensures well-defined long-term averages.
- **Bias-Variance Trade-off**: Discounting introduces bias but reduces variance.

### Importance

- **Real-world Applicability**: Many tasks are continuing (e.g., robotics, finance).
- **Algorithm Adaptation**: Requires modifications to standard episodic algorithms.

### Pros vs Cons

- **Pros**:
  - Models real-world, ongoing tasks.
- **Cons**:
  - Harder to evaluate performance.
  - Requires careful handling of discounting and baselines.

### Cutting-Edge Advances

- **Average Reward Actor-Critic**: Directly optimizes average reward.
- **Emphatic Weightings**: Improved convergence in off-policy settings.

---

## 13.7 Policy Parameterization for Continuous Actions

### Definition

- **Continuous Action Parameterization**: Policies output parameters of continuous distributions (e.g., Gaussian) over actions.

### Pertinent Equations

- **Gaussian Policy**:  
  $$ \pi(a|s; \theta) = \mathcal{N}(a; \mu_\theta(s), \Sigma_\theta(s)) $$
- **Policy Gradient for Continuous Actions**:  
  $$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \mathcal{N}(a; \mu_\theta(s), \Sigma_\theta(s)) Q^{\pi_\theta}(s, a) \right] $$

### Key Principles

- **Differentiable Sampling**: Enables backpropagation through action sampling.
- **Reparameterization Trick**:  
  $$ a = \mu_\theta(s) + \Sigma_\theta(s)^{1/2} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

### Detailed Concept Analysis

- **Expressiveness**: Can model complex, multi-modal action distributions.
- **Exploration**: Stochasticity in action selection aids exploration.

### Importance

- **Robotics and Control**: Many real-world tasks require continuous actions.
- **Generalization**: Parameterization enables learning in high-dimensional spaces.

### Pros vs Cons

- **Pros**:
  - Natural fit for continuous domains.
  - Supports gradient-based optimization.
- **Cons**:
  - May require careful tuning of distribution parameters.
  - Can be sensitive to initialization.

### Cutting-Edge Advances

- **Normalizing Flows**: More expressive policy distributions.
- **Implicit Policy Gradients**: Non-explicit, sample-based policy representations.

---