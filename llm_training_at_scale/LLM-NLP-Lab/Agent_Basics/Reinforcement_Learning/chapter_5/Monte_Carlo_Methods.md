# 5. Monte Carlo Methods

## Definition

Monte Carlo (MC) methods are a class of algorithms that utilize repeated random sampling to obtain numerical results. In reinforcement learning (RL), MC methods estimate value functions and optimize policies by averaging returns from sampled episodes, without requiring knowledge of the environment’s dynamics.

---

## 5.1 Monte Carlo Prediction

### Definition

Monte Carlo prediction estimates the value function $V_\pi(s)$ for a given policy $\pi$ by averaging the returns observed after visiting state $s$ in multiple episodes.

### Pertinent Equations

- **State-value function:**
  $$
  V_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]
  $$
  where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$

- **Empirical estimate:**
  $$
  V(s) \leftarrow \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}
  $$
  where $N(s)$ is the number of times state $s$ is visited.

### Key Principles

- **Episodic Sampling:** MC methods require complete episodes to compute returns.
- **First-visit vs. Every-visit:**  
  - *First-visit MC*: Only the first occurrence of $s$ in an episode is used.
  - *Every-visit MC*: All occurrences of $s$ in an episode are used.

### Detailed Concept Analysis

- **Return ($G_t$):** The total discounted reward from time $t$ onward.
- **Averaging:** The value estimate is the mean of observed returns.
- **Policy Evaluation:** MC prediction is used for policy evaluation, not improvement.

### Importance

- **Model-free:** No need for transition probabilities.
- **Foundation for Control:** Underpins MC control algorithms.

### Pros vs. Cons

- **Pros:**
  - Simple, intuitive.
  - No need for environment model.
- **Cons:**
  - Requires episodic tasks.
  - High variance in estimates.
  - Slow convergence for rarely visited states.

### Recent Developments

- **Variance reduction techniques** (e.g., bootstrapping, eligibility traces).
- **Parallelized MC sampling** for large-scale RL.

---

## 5.2 Monte Carlo Estimation of Action Values

### Definition

Estimates the action-value function $Q_\pi(s, a)$ by averaging returns following state-action pairs under policy $\pi$.

### Pertinent Equations

- **Action-value function:**
  $$
  Q_\pi(s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]
  $$
- **Empirical estimate:**
  $$
  Q(s, a) \leftarrow \frac{1}{N(s, a)} \sum_{i=1}^{N(s, a)} G_t^{(i)}
  $$

### Key Principles

- **Sampling:** Only episodes where $(s, a)$ is visited contribute to $Q(s, a)$.
- **Exploring Starts:** Ensures all $(s, a)$ pairs are sampled.

### Detailed Concept Analysis

- **Exploring Starts:** Each episode starts with a randomly chosen $(s, a)$ to guarantee coverage.
- **First-visit/Every-visit:** Analogous to state-value estimation.

### Importance

- **Policy Improvement:** Enables $\epsilon$-greedy or greedy policy updates.

### Pros vs. Cons

- **Pros:**
  - Direct estimation of $Q$ for control.
- **Cons:**
  - Requires sufficient exploration.
  - High variance.

### Recent Developments

- **Experience replay** to improve sample efficiency.
- **Function approximation** for large state-action spaces.

---

## 5.3 Monte Carlo Control

### Definition

Monte Carlo control methods learn optimal policies by iteratively evaluating and improving policies using MC estimates of $Q(s, a)$.

### Pertinent Equations

- **Policy improvement:**
  $$
  \pi(s) = \arg\max_a Q(s, a)
  $$
- **$\epsilon$-greedy policy:**
  $$
  \pi(a|s) = 
  \begin{cases}
    1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
    \frac{\epsilon}{|\mathcal{A}(s)|} & \text{otherwise}
  \end{cases}
  $$

### Key Principles

- **Generalized Policy Iteration:** Alternates between policy evaluation (MC prediction) and policy improvement.
- **Exploring Starts:** Ensures all $(s, a)$ pairs are visited.

### Detailed Concept Analysis

- **Policy Evaluation:** Use MC to estimate $Q_\pi$.
- **Policy Improvement:** Update policy to be greedy with respect to $Q$.

### Importance

- **Model-free optimal control.**

### Pros vs. Cons

- **Pros:**
  - Simple, model-free.
- **Cons:**
  - Requires exploring starts or sufficient exploration.
  - Slow convergence.

### Recent Developments

- **Soft policy updates** (entropy regularization).
- **Integration with deep RL architectures.**

---

## 5.4 Monte Carlo Control without Exploring Starts

### Definition

Removes the impractical requirement of exploring starts by using stochastic policies (e.g., $\epsilon$-greedy) to ensure sufficient exploration.

### Pertinent Equations

- **$\epsilon$-greedy policy:** (see above)

### Key Principles

- **Stochastic Exploration:** Ensures all actions are sampled with nonzero probability.

### Detailed Concept Analysis

- **Policy Evaluation:** As before, but with stochastic policy.
- **Policy Improvement:** Update towards greedy, but maintain exploration.

### Importance

- **Practical applicability** in real-world RL tasks.

### Pros vs. Cons

- **Pros:**
  - No need for exploring starts.
- **Cons:**
  - May require careful tuning of $\epsilon$.

### Recent Developments

- **Adaptive exploration schedules.**
- **Entropy-based exploration.**

---

## 5.5 Off-policy Prediction via Importance Sampling

### Definition

Estimates $V_\pi$ or $Q_\pi$ using data generated from a different behavior policy $b$, correcting for the distribution mismatch via importance sampling.

### Pertinent Equations

- **Importance sampling ratio:**
  $$
  \rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
  $$
- **Off-policy value estimate:**
  $$
  V(s) \leftarrow \frac{\sum_{i=1}^{N(s)} \rho_t^{(i)} G_t^{(i)}}{\sum_{i=1}^{N(s)} \rho_t^{(i)}}
  $$

### Key Principles

- **Weighted Returns:** Adjusts for the difference between target and behavior policies.
- **Unbiasedness:** Ordinary importance sampling is unbiased; weighted is biased but lower variance.

### Detailed Concept Analysis

- **High Variance:** Especially for long episodes or large policy divergence.
- **Weighted vs. Ordinary:** Weighted normalizes by sum of weights.

### Importance

- **Enables off-policy learning.**

### Pros vs. Cons

- **Pros:**
  - Can learn about any policy from any data.
- **Cons:**
  - High variance, especially with long episodes.

### Recent Developments

- **Variance reduction:** Truncated importance sampling, per-decision weighting.

---

## 5.6 Incremental Implementation

### Definition

Updates value estimates incrementally, avoiding the need to store all returns.

### Pertinent Equations

- **Incremental mean update:**
  $$
  V(s) \leftarrow V(s) + \alpha [G_t - V(s)]
  $$
  where $\alpha = \frac{1}{N(s)}$ or a constant step-size.

### Key Principles

- **Online Updates:** Update after each episode or visit.
- **Memory Efficiency:** No need to store all returns.

### Detailed Concept Analysis

- **Step-size parameter:** Controls learning rate and bias-variance tradeoff.

### Importance

- **Scalability** to large state spaces.

### Pros vs. Cons

- **Pros:**
  - Efficient, online.
- **Cons:**
  - Step-size selection critical.

### Recent Developments

- **Adaptive step-size algorithms.**

---

## 5.7 Off-policy Monte Carlo Control

### Definition

Combines off-policy MC prediction with control, using importance sampling to evaluate and improve a target policy while following a different behavior policy.

### Pertinent Equations

- **Off-policy $Q$ update:**
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha [\rho_t (G_t - Q(s, a))]
  $$

### Key Principles

- **Behavior vs. Target Policy:** Data collected under $b$, learning about $\pi$.
- **Importance Sampling:** Corrects for policy mismatch.

### Detailed Concept Analysis

- **Policy Improvement:** Greedy with respect to $Q$.
- **Exploration:** Behavior policy must cover all actions.

### Importance

- **Experience reuse:** Learn optimal policy from arbitrary data.

### Pros vs. Cons

- **Pros:**
  - Flexible, data-efficient.
- **Cons:**
  - High variance.

### Recent Developments

- **Variance reduction:** Retrace, Tree-backup, V-trace algorithms.

---

## 5.8 *Discounting-aware Importance Sampling

### Definition

Modifies importance sampling to account for discounting, reducing variance in off-policy MC estimates.

### Pertinent Equations

- **Discounted importance sampling ratio:**
  $$
  \rho_t^\gamma = \prod_{k=t}^{T-1} \gamma \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
  $$

### Key Principles

- **Discount-aware weighting:** Emphasizes near-term rewards.

### Detailed Concept Analysis

- **Variance Reduction:** Discounting reduces the impact of large weights from distant future.

### Importance

- **Improved stability** in off-policy MC.

### Pros vs. Cons

- **Pros:**
  - Lower variance.
- **Cons:**
  - Potential bias.

### Recent Developments

- **Theoretical analysis** of bias-variance tradeoff.

---

## 5.9 *Per-decision Importance Sampling

### Definition

Applies importance sampling at each time step, rather than over entire episodes, to further reduce variance.

### Pertinent Equations

- **Per-decision ratio:**
  $$
  \rho_k = \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
  $$
- **Per-decision weighted return:**
  $$
  G_t = R_{t+1} + \gamma \rho_{t+1} [R_{t+2} + \gamma \rho_{t+2} [\cdots]]
  $$

### Key Principles

- **Stepwise Correction:** Adjusts for policy mismatch at each step.

### Detailed Concept Analysis

- **Variance Reduction:** More stable than episode-wise importance sampling.

### Importance

- **Practical off-policy MC learning.**

### Pros vs. Cons

- **Pros:**
  - Significantly reduced variance.
- **Cons:**
  - Slightly more complex implementation.

### Recent Developments

- **Widely adopted** in modern off-policy RL algorithms.

---