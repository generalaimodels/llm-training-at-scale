# 2. Multi-Armed Bandits

## 2.1 $k$-Armed Bandit Problem

### Definition

A $k$-armed bandit problem is a foundational model in sequential decision-making where an agent repeatedly chooses among $k$ actions (arms), each providing stochastic rewards drawn from an unknown probability distribution. The objective is to maximize the expected cumulative reward over time.

### Pertinent Equations

- **Expected Reward for Arm $i$:**
  $$
  q_*(a_i) = \mathbb{E}[R_t | A_t = a_i]
  $$
- **Action Selection at Time $t$:**
  $$
  A_t \in \{a_1, a_2, ..., a_k\}
  $$
- **Cumulative Reward:**
  $$
  G_T = \sum_{t=1}^T R_t
  $$
- **Regret:**
  $$
  \text{Regret}_T = T \cdot q_*(a^*) - \mathbb{E}\left[\sum_{t=1}^T R_t\right]
  $$
  where $a^* = \arg\max_{a} q_*(a)$

### Key Principles

- **Exploration vs. Exploitation:** Balancing the need to explore less-known arms versus exploiting arms with high estimated rewards.
- **Stationarity:** Reward distributions may be stationary or nonstationary.
- **Regret Minimization:** The goal is to minimize regret over time.

### Detailed Concept Analysis

- **Agent repeatedly selects an arm $a_t$ at each timestep $t$.**
- **Receives reward $R_t$ drawn from the arm's reward distribution.**
- **Estimates $q_*(a)$ for each arm based on observed rewards.**
- **Action selection strategies (e.g., $\epsilon$-greedy, UCB, Thompson sampling) are used to balance exploration and exploitation.**

### Importance

- **Fundamental to online learning, adaptive control, and reinforcement learning.**
- **Models real-world scenarios: clinical trials, ad placement, A/B testing, recommendation systems.**

### Pros vs. Cons

**Pros:**
- Simple, interpretable framework.
- Theoretical guarantees for many algorithms.

**Cons:**
- Assumes independence between arms.
- Does not model state transitions or delayed rewards.

### Cutting-Edge Advances

- **Contextual bandits (see 2.9):** Incorporate side information (context).
- **Nonstationary bandits:** Adapt to changing reward distributions.
- **Best-arm identification:** Focus on identifying the optimal arm efficiently.

---

## 2.2 Action-Value Methods

### Definition

Action-value methods estimate the expected reward (value) of each action and use these estimates to guide action selection.

### Pertinent Equations

- **Sample Average Estimate:**
  $$
  Q_t(a) = \frac{\sum_{i=1}^{N_t(a)} R_i}{N_t(a)}
  $$
  where $N_t(a)$ is the number of times action $a$ has been selected up to time $t$.

- **Incremental Update Rule:**
  $$
  Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a) + 1} [R_n - Q_n(a)]
  $$

### Key Principles

- **Value Estimation:** Maintain and update estimates $Q(a)$ for each arm.
- **Action Selection:** Use $Q(a)$ to select actions (e.g., greedy, $\epsilon$-greedy).

### Detailed Concept Analysis

- **Sample averages converge to true action values as $N_t(a) \to \infty$.**
- **Incremental updates enable efficient online computation.**
- **Action selection can be purely greedy or incorporate exploration.**

### Importance

- **Core to most bandit and RL algorithms.**
- **Enables learning from experience without prior knowledge.**

### Pros vs. Cons

**Pros:**
- Simple, efficient, and online.
- Converges to optimal action in stationary settings.

**Cons:**
- Slow adaptation to nonstationary problems.
- Exploration may be insufficient without explicit mechanisms.

### Cutting-Edge Advances

- **Weighted averages for nonstationary problems.**
- **Bayesian action-value estimation.**

---

## 2.3 The 10-Armed Testbed

### Definition

A standardized experimental setup for evaluating bandit algorithms, typically with $k=10$ arms, each with a stationary reward distribution.

### Pertinent Equations

- **True Action Values:**
  $$
  q_*(a) \sim \mathcal{N}(0, 1)
  $$
- **Observed Rewards:**
  $$
  R_t \sim \mathcal{N}(q_*(a), 1)
  $$

### Key Principles

- **Benchmarking:** Provides a controlled environment for algorithm comparison.
- **Statistical Evaluation:** Multiple runs with different random seeds.

### Detailed Concept Analysis

- **Each arm’s true value is drawn from a normal distribution.**
- **Performance measured by average reward and percentage of optimal action selections.**

### Importance

- **Standardizes empirical evaluation.**
- **Facilitates reproducibility and comparison.**

### Pros vs. Cons

**Pros:**
- Simple, interpretable, widely adopted.

**Cons:**
- Limited to stationary, synthetic settings.

### Cutting-Edge Advances

- **Extensions to nonstationary and contextual testbeds.**
- **Automated benchmarking platforms.**

---

## 2.4 Incremental Implementation

### Definition

An efficient method for updating action-value estimates incrementally, avoiding the need to store all past rewards.

### Pertinent Equations

- **Incremental Update:**
  $$
  Q_{n+1} = Q_n + \alpha [R_n - Q_n]
  $$
  where $\alpha$ is the step-size parameter.

### Key Principles

- **Online Learning:** Update estimates with each new reward.
- **Step-Size Control:** $\alpha$ can be constant or decreasing.

### Detailed Concept Analysis

- **For stationary problems, $\alpha = \frac{1}{N}$ ensures unbiased estimates.**
- **For nonstationary problems, constant $\alpha$ enables tracking.**

### Importance

- **Enables real-time learning in resource-constrained environments.**

### Pros vs. Cons

**Pros:**
- Memory and computation efficient.
- Adaptable to streaming data.

**Cons:**
- Choice of $\alpha$ critical for performance.

### Cutting-Edge Advances

- **Adaptive step-size algorithms.**
- **Variance reduction techniques.**

---

## 2.5 Tracking a Nonstationary Problem

### Definition

Adapting action-value estimates to environments where reward distributions change over time.

### Pertinent Equations

- **Exponential Recency-Weighted Average:**
  $$
  Q_{n+1} = Q_n + \alpha [R_n - Q_n]
  $$
  with constant $\alpha$.

### Key Principles

- **Forgetting Factor:** Recent rewards weighted more heavily.
- **Rapid Adaptation:** Enables tracking of changing action values.

### Detailed Concept Analysis

- **Constant $\alpha$ ensures old data is discounted exponentially.**
- **Trade-off between stability and responsiveness.**

### Importance

- **Essential for real-world, dynamic environments.**

### Pros vs. Cons

**Pros:**
- Responsive to change.
- Simple to implement.

**Cons:**
- Increased variance in estimates.
- Requires tuning of $\alpha$.

### Cutting-Edge Advances

- **Contextual adaptation.**
- **Meta-learning for step-size selection.**

---

## 2.6 Optimistic Initial Values

### Definition

A strategy that initializes action-value estimates optimistically high to encourage exploration.

### Pertinent Equations

- **Initialization:**
  $$
  Q_1(a) = Q_{\text{init}} \gg \max q_*(a)
  $$

### Key Principles

- **Induced Exploration:** High initial values drive the agent to try all actions.
- **Decay of Optimism:** Estimates converge as actions are sampled.

### Detailed Concept Analysis

- **No explicit exploration parameter needed.**
- **Works best in stationary settings.**

### Importance

- **Simple, parameter-free exploration mechanism.**

### Pros vs. Cons

**Pros:**
- Encourages early exploration.
- No need for $\epsilon$ parameter.

**Cons:**
- Ineffective in nonstationary problems.
- May slow convergence if $Q_{\text{init}}$ is too high.

### Cutting-Edge Advances

- **Dynamic optimism adjustment.**
- **Integration with Bayesian priors.**

---

## 2.7 Upper-Confidence-Bound Action Selection

### Definition

A principled exploration strategy that selects actions based on both estimated value and uncertainty.

### Pertinent Equations

- **UCB Action Selection:**
  $$
  A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
  $$
  where $c$ controls exploration.

### Key Principles

- **Optimism in the Face of Uncertainty:** Prefer actions with high uncertainty.
- **Logarithmic Regret Bound:** Theoretical guarantee on performance.

### Detailed Concept Analysis

- **Balances exploitation (high $Q_t(a)$) and exploration (high uncertainty).**
- **Exploration term decays as $N_t(a)$ increases.**

### Importance

- **Strong theoretical guarantees.**
- **Widely used in online learning and RL.**

### Pros vs. Cons

**Pros:**
- Efficient exploration.
- Provable regret bounds.

**Cons:**
- Requires accurate uncertainty estimation.
- Sensitive to choice of $c$.

### Cutting-Edge Advances

- **Bayesian UCB variants.**
- **Contextual UCB algorithms.**

---

## 2.8 Gradient Bandit Algorithms

### Definition

Algorithms that learn a preference for each action and use a softmax distribution to select actions, updating preferences via gradient ascent on expected reward.

### Pertinent Equations

- **Preference Update:**
  $$
  H_{t+1}(a) = H_t(a) + \alpha (R_t - \bar{R}_t) [\mathbb{I}(A_t = a) - \pi_t(a)]
  $$
  where $H_t(a)$ is the preference for action $a$, $\bar{R}_t$ is the average reward, and $\pi_t(a)$ is the softmax probability:
  $$
  \pi_t(a) = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}}
  $$

### Key Principles

- **Policy Gradient:** Directly optimizes the action selection probabilities.
- **Baseline Subtraction:** Reduces variance in updates.

### Detailed Concept Analysis

- **Softmax ensures all actions have nonzero probability.**
- **Preferences updated to increase probability of actions yielding above-average rewards.**

### Importance

- **Foundation for policy gradient methods in RL.**
- **Handles nonstationary reward structures.**

### Pros vs. Cons

**Pros:**
- Flexible, parameterizes policy directly.
- Effective in nonstationary settings.

**Cons:**
- Sensitive to step-size $\alpha$.
- May converge slowly.

### Cutting-Edge Advances

- **Natural gradient methods.**
- **Entropy regularization for exploration.**

---

## 2.9 Associative Search (Contextual Bandits)

### Definition

An extension of the bandit problem where the agent observes a context (feature vector) before selecting an action, and the reward depends on both the action and the context.

### Pertinent Equations

- **Contextual Policy:**
  $$
  \pi(a|x) = P(A_t = a | X_t = x)
  $$
- **Expected Reward:**
  $$
  q_*(x, a) = \mathbb{E}[R_t | X_t = x, A_t = a]
  $$

### Key Principles

- **Contextualization:** Action selection is conditioned on observed context.
- **Function Approximation:** Required for large or continuous context spaces.

### Detailed Concept Analysis

- **Agent observes context $x_t$ at each timestep.**
- **Selects action $a_t$ based on policy $\pi(a|x_t)$.**
- **Receives reward $r_t$ dependent on $(x_t, a_t)$.**
- **Algorithms include LinUCB, contextual Thompson sampling, neural bandits.**

### Importance

- **Models personalized recommendations, targeted advertising, adaptive clinical trials.**

### Pros vs. Cons

**Pros:**
- Captures heterogeneity in environments.
- Enables personalization.

**Cons:**
- Requires efficient context representation.
- Increased computational complexity.

### Cutting-Edge Advances

- **Deep contextual bandits using neural networks.**
- **Meta-contextual bandits for rapid adaptation.**
- **Contextual bandits with structured or relational context (e.g., GNNs).**

---