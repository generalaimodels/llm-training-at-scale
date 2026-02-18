
# 10. On-policy Control with Approximation

---

## 10.1 Episodic Semi-gradient Control

### Definition
Episodic semi-gradient control refers to reinforcement learning (RL) algorithms that learn optimal policies in episodic tasks using function approximation, where the gradient is taken only with respect to the estimated value function, not the target.

### Pertinent Equations

- **Action-value function approximation:**
  $$
  \hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a)
  $$
  where $\mathbf{w}$ are the parameters of the function approximator.

- **Semi-gradient Sarsa update:**
  $$
  \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t) \right] \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w}_t)
  $$

### Key Principles

- **On-policy:** The policy being improved is the same as the policy used to generate data.
- **Semi-gradient:** The gradient is computed only with respect to the estimated value, not the full target.
- **Function Approximation:** Used to generalize across large or continuous state-action spaces.

### Detailed Concept Analysis

- **Episodic Setting:** The environment resets after each episode, and learning is based on complete episodes.
- **Policy Improvement:** $\epsilon$-greedy or softmax policies are typically used for exploration.
- **Convergence:** Semi-gradient methods may not always converge with function approximation, but are widely used due to their simplicity and empirical success.

### Importance

- **Scalability:** Enables RL in high-dimensional or continuous spaces.
- **Generalization:** Learns from limited data by sharing information across similar states/actions.

### Pros vs. Cons

- **Pros:**
  - Handles large/continuous spaces.
  - Simple to implement.
- **Cons:**
  - May diverge with certain function approximators (e.g., non-linear).
  - Biased updates due to semi-gradient.

### Cutting-edge Advances

- **Deep RL:** Use of deep neural networks for $\hat{q}$.
- **Stabilization Techniques:** Target networks, experience replay.
- **Policy Regularization:** To improve stability and convergence.

---

## 10.2 Semi-gradient n-step Sarsa

### Definition
Semi-gradient n-step Sarsa generalizes the one-step Sarsa algorithm to use $n$-step returns, improving learning speed and stability in RL with function approximation.

### Pertinent Equations

- **n-step return:**
  $$
  G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w})
  $$
- **Parameter update:**
  $$
  \mathbf{w}_{t+n} = \mathbf{w}_t + \alpha \left[ G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_t) \right] \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w}_t)
  $$

### Key Principles

- **n-step Bootstrapping:** Combines Monte Carlo and temporal-difference (TD) learning.
- **Semi-gradient:** Gradient only with respect to the estimated value.

### Detailed Concept Analysis

- **Trade-off:** Larger $n$ increases variance but reduces bias.
- **Eligibility Traces:** n-step Sarsa is a special case of TD($\lambda$) with $\lambda$ controlling the mix of n-step returns.

### Importance

- **Sample Efficiency:** Faster learning by leveraging more information per update.
- **Bias-Variance Trade-off:** Adjustable via $n$.

### Pros vs. Cons

- **Pros:**
  - Improved learning speed.
  - Flexibility in bias-variance trade-off.
- **Cons:**
  - Increased memory/computation for large $n$.
  - Still subject to instability with function approximation.

### Cutting-edge Advances

- **Prioritized Experience Replay:** For more efficient sampling.
- **Multi-step Targets in Deep RL:** Used in DQN, Rainbow, etc.

---

## 10.3 Average Reward: A New Problem Setting for Continuing Tasks

### Definition
Average reward RL focuses on maximizing the long-term average reward per time step, rather than the discounted sum of rewards, suitable for continuing (non-episodic) tasks.

### Pertinent Equations

- **Average reward:**
  $$
  r(\pi) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_\pi \left[ \sum_{t=1}^T R_t \right]
  $$
- **Differential value function:**
  $$
  h_\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty (R_{t+1} - r(\pi)) \mid S_0 = s \right]
  $$

### Key Principles

- **No Discount Factor:** Focuses on steady-state performance.
- **Differential Value Functions:** Measures relative value compared to average reward.

### Detailed Concept Analysis

- **Stationarity:** Assumes Markov process reaches steady-state.
- **Policy Evaluation/Improvement:** Requires estimating both $r(\pi)$ and $h_\pi(s)$.

### Importance

- **Natural for Continuing Tasks:** E.g., robotics, process control.
- **Avoids Discounting Artifacts:** More interpretable in some domains.

### Pros vs. Cons

- **Pros:**
  - Directly optimizes steady-state performance.
  - No need to choose $\gamma$.
- **Cons:**
  - Harder to estimate $r(\pi)$ and $h_\pi(s)$.
  - Less common in standard RL libraries.

### Cutting-edge Advances

- **Differential Semi-gradient Methods:** For function approximation.
- **Applications in Operations Research, Robotics.**

---

## 10.4 Deprecating the Discounted Setting

### Definition
Deprecating the discounted setting refers to moving away from the use of a discount factor ($\gamma < 1$) in RL, especially for continuing tasks, in favor of average reward formulations.

### Pertinent Equations

- **Discounted return:**
  $$
  G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
  $$
- **Average reward return:**
  $$
  r(\pi) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_\pi \left[ \sum_{t=1}^T R_t \right]
  $$

### Key Principles

- **Discounting:** Artificially prioritizes immediate rewards.
- **Average Reward:** Focuses on long-term steady-state performance.

### Detailed Concept Analysis

- **Discounting Issues:** Can distort optimal policies in continuing tasks.
- **Average Reward Advantages:** More natural for ongoing processes.

### Importance

- **Theoretical Clarity:** Avoids arbitrary choice of $\gamma$.
- **Practical Relevance:** Better matches real-world objectives in many domains.

### Pros vs. Cons

- **Pros:**
  - More interpretable for continuing tasks.
  - Avoids discounting artifacts.
- **Cons:**
  - Harder to analyze and implement.
  - Less mature algorithmic ecosystem.

### Cutting-edge Advances

- **Average Reward Algorithms:** Differential Sarsa, R-learning.
- **Theoretical Work:** On convergence and stability.

---

## 10.5 Differential Semi-gradient n-step Sarsa

### Definition
Differential semi-gradient n-step Sarsa is an RL algorithm for average reward problems, using n-step returns and function approximation, updating both the value function and the average reward estimate.

### Pertinent Equations

- **Average reward estimate update:**
  $$
  \bar{r}_{t+1} = \bar{r}_t + \beta \left[ R_{t+1} - \bar{r}_t + \hat{h}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{h}(S_t, A_t, \mathbf{w}_t) \right]
  $$
- **n-step differential return:**
  $$
  G_{t:t+n} = \sum_{i=1}^n (R_{t+i} - \bar{r}_t) + \hat{h}(S_{t+n}, A_{t+n}, \mathbf{w}_t)
  $$
- **Parameter update:**
  $$
  \mathbf{w}_{t+n} = \mathbf{w}_t + \alpha \left[ G_{t:t+n} - \hat{h}(S_t, A_t, \mathbf{w}_t) \right] \nabla_{\mathbf{w}} \hat{h}(S_t, A_t, \mathbf{w}_t)
  $$

### Key Principles

- **Differential Value Function:** Measures value relative to average reward.
- **Semi-gradient:** Gradient only with respect to estimated value.
- **n-step Returns:** Combines information from multiple steps.

### Detailed Concept Analysis

- **Simultaneous Estimation:** Both $\bar{r}$ and $\mathbf{w}$ are updated online.
- **Bias-Variance Trade-off:** Controlled by $n$.

### Importance

- **Enables Average Reward RL:** With function approximation and multi-step returns.
- **Sample Efficiency:** Improved by n-step updates.

### Pros vs. Cons

- **Pros:**
  - Handles continuing tasks naturally.
  - More sample efficient than one-step methods.
- **Cons:**
  - More complex to implement.
  - Sensitive to step-size parameters.

### Cutting-edge Advances

- **Deep Differential RL:** Extending to deep neural networks.
- **Adaptive Step-size Methods:** For improved stability.

---