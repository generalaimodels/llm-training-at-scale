
## 7. n-step Bootstrapping

### Definition

n-step bootstrapping generalizes one-step temporal-difference (TD) learning by updating value estimates using returns that span $n$ time steps into the future, blending Monte Carlo and TD methods. It enables a trade-off between bias and variance by controlling the lookahead horizon.

---

## 7.1 n-step TD Prediction

### Definition

n-step TD prediction estimates the value function $V(s)$ by updating it with the $n$-step return, which combines actual rewards for $n$ steps and a bootstrapped estimate thereafter.

### Pertinent Equations

- **n-step Return**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$
- **n-step TD Update**:  
  $$ V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t^{(n)} - V(S_t) \right] $$

### Key Principles

- **Bootstrapping**: Combines actual rewards and estimated future value.
- **Trade-off**: $n=1$ is TD(0); $n \to \infty$ approaches Monte Carlo.

### Detailed Concept Analysis

- **Bias-Variance Trade-off**:  
  - Small $n$: Higher bias, lower variance.
  - Large $n$: Lower bias, higher variance.
- **Eligibility Traces**: n-step returns are foundational for TD($\lambda$), which averages over all $n$.

### Importance

- **Flexibility**: Allows tuning of learning dynamics for specific tasks.
- **Foundation**: Underpins advanced RL algorithms.

### Pros vs. Cons

- **Pros**:
  - Adjustable bias/variance.
  - Can accelerate learning in some environments.
- **Cons**:
  - Requires storage of $n$ transitions.
  - Sensitive to $n$ selection.

### Cutting-edge Advances

- **Deep n-step TD**: Used in deep RL (e.g., DQN variants).
- **Adaptive n-step**: Dynamic adjustment of $n$ during training.

---

## 7.2 n-step Sarsa

### Definition

n-step Sarsa extends the Sarsa algorithm to use $n$-step returns for updating the action-value function $Q(s, a)$ in an on-policy manner.

### Pertinent Equations

- **n-step Sarsa Return**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n}) $$
- **n-step Sarsa Update**:  
  $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ G_t^{(n)} - Q(S_t, A_t) \right] $$

### Key Principles

- **On-policy**: Updates use actions actually taken by the policy.
- **n-step Bootstrapping**: Blends real rewards and bootstrapped estimates.

### Detailed Concept Analysis

- **Exploration**: Policy must ensure sufficient exploration (e.g., $\epsilon$-greedy).
- **Trace Decay**: n-step Sarsa is a special case of Sarsa($\lambda$).

### Importance

- **Improved Learning**: Can outperform 1-step Sarsa in some domains.

### Pros vs. Cons

- **Pros**:
  - More robust to delayed rewards.
- **Cons**:
  - Increased memory and computation.

### Cutting-edge Advances

- **Deep n-step Sarsa**: Used in actor-critic and policy gradient methods.

---

## 7.3 n-step Off-policy Learning

### Definition

n-step off-policy learning estimates value functions for a target policy $\pi$ while following a different behavior policy $b$, using importance sampling to correct for the policy mismatch.

### Pertinent Equations

- **Importance Sampling Ratio**:  
  $$ \rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)} $$
- **n-step Off-policy Return**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$
- **n-step Off-policy Update**:  
  $$ V(S_t) \leftarrow V(S_t) + \alpha \rho_{t:t+n-1} \left[ G_t^{(n)} - V(S_t) \right] $$

### Key Principles

- **Off-policy Correction**: Uses importance sampling to ensure unbiased updates.
- **Variance**: Importance sampling can introduce high variance, especially for large $n$.

### Detailed Concept Analysis

- **Bias-Variance Trade-off**:  
  - Larger $n$ increases variance due to product of importance ratios.
- **Truncation**: Techniques like per-decision importance sampling mitigate variance.

### Importance

- **Policy Evaluation**: Enables learning about policies not currently executed.

### Pros vs. Cons

- **Pros**:
  - Flexibility in data collection.
- **Cons**:
  - High variance for large $n$.

### Cutting-edge Advances

- **Variance Reduction**: Use of control variates and weighted importance sampling.

---

## 7.4 Per-decision Methods with Control Variates

### Definition

Per-decision methods with control variates reduce the variance of off-policy n-step updates by applying importance sampling at each step and using control variates to further stabilize learning.

### Pertinent Equations

- **Per-decision Importance Sampling**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma \rho_{t+1} R_{t+2} + \cdots + \gamma^{n-1} \left( \prod_{k=1}^{n-1} \rho_{t+k} \right) R_{t+n} + \gamma^n \left( \prod_{k=1}^{n} \rho_{t+k} \right) V(S_{t+n}) $$
- **Control Variate Adjustment**:  
  $$ \text{Adjusted Update} = \text{Original Update} - \text{Control Variate} $$

### Key Principles

- **Stepwise Correction**: Reduces variance by applying corrections incrementally.
- **Control Variates**: Subtracts a baseline to reduce estimator variance.

### Detailed Concept Analysis

- **Variance Reduction**: Control variates exploit known expectations to stabilize updates.
- **Bias**: Properly chosen control variates do not introduce bias.

### Importance

- **Stability**: Essential for practical off-policy multi-step learning.

### Pros vs. Cons

- **Pros**:
  - Lower variance.
  - More stable learning.
- **Cons**:
  - Increased algorithmic complexity.

### Cutting-edge Advances

- **Learned Control Variates**: Adaptive baselines in deep RL.

---

## 7.5 Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm

### Definition

The n-step Tree Backup algorithm is an off-policy method that avoids importance sampling by backing up expected values under the target policy, rather than sampled actions.

### Pertinent Equations

- **n-step Tree Backup Return**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma \sum_{a_{t+1}} \pi(a_{t+1}|S_{t+1}) \left[ Q(S_{t+1}, a_{t+1}) \right] $$
  - Recursively, for $n$ steps, the return is built by expanding all possible actions at each step, weighted by $\pi$.

### Key Principles

- **Expectation over Actions**: Uses expected value under $\pi$ at each step.
- **No Importance Sampling**: Avoids high variance from importance ratios.

### Detailed Concept Analysis

- **Recursive Expansion**: At each step, sum over all actions weighted by policy probabilities.
- **Computational Cost**: Increases with action space size.

### Importance

- **Variance Reduction**: More stable than importance sampling-based methods.

### Pros vs. Cons

- **Pros**:
  - No importance sampling variance.
- **Cons**:
  - Computationally expensive for large action spaces.

### Cutting-edge Advances

- **Tree Backup in Deep RL**: Used in algorithms for continuous control.

---

## 7.6 A Unifying Algorithm: n-step Q($\sigma$)

### Definition

n-step Q($\sigma$) is a unifying algorithm that interpolates between Sarsa (sampling) and Tree Backup (expectation) using a parameter $\sigma \in [0,1]$ at each step.

### Pertinent Equations

- **n-step Q($\sigma$) Return**:  
  $$ G_t^{(n)} = R_{t+1} + \gamma \left[ \sigma_{t+1} Q(S_{t+1}, A_{t+1}) + (1 - \sigma_{t+1}) \sum_{a} \pi(a|S_{t+1}) Q(S_{t+1}, a) \right] $$
  - Recursively applied for $n$ steps.

### Key Principles

- **Interpolation**: $\sigma=1$ yields Sarsa; $\sigma=0$ yields Tree Backup.
- **Flexibility**: $\sigma$ can be state, time, or step dependent.

### Detailed Concept Analysis

- **Bias-Variance Control**: Adjusting $\sigma$ tunes the trade-off between sampling and expectation.
- **Generalization**: Subsumes many existing algorithms as special cases.

### Importance

- **Unified Framework**: Enables algorithm selection and adaptation within a single paradigm.

### Pros vs. Cons

- **Pros**:
  - Highly flexible.
  - Can optimize learning dynamics for specific tasks.
- **Cons**:
  - Requires tuning or learning $\sigma$.

### Cutting-edge Advances

- **Adaptive $\sigma$**: Algorithms that learn or schedule $\sigma$ for optimal performance.

---