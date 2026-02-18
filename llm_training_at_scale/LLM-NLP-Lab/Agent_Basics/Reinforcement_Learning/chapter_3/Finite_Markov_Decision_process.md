

# Finite Markov Decision Processes (MDPs)

---

## 3.1 The Agent–Environment Interface

### Definition
A **Finite Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making where outcomes are partly random and partly under the control of a decision-maker (agent). The agent interacts with an environment in discrete time steps.

### Key Components
- **States ($S$):** Finite set of environment states.
- **Actions ($A$):** Finite set of actions available to the agent.
- **Transition Probability ($P$):** Probability of moving from one state to another given an action.
- **Reward Function ($R$):** Expected reward received after transitioning between states due to an action.
- **Discount Factor ($\gamma$):** Factor for present value of future rewards, $0 \leq \gamma \leq 1$.

### Equations
- **Transition Probability:**
  $$
  P(s', r \mid s, a) = \Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a\}
  $$
- **State Transition:**
  $$
  S_{t+1} \sim P(\cdot \mid S_t, A_t)
  $$
- **Reward:**
  $$
  R_{t+1} = R(S_t, A_t, S_{t+1})
  $$

### Principles
- **Markov Property:** The future is independent of the past given the present state.
- **Agent-Environment Loop:** At each time step $t$, the agent observes $S_t$, selects $A_t$, receives $R_{t+1}$, and transitions to $S_{t+1}$.

### Significance
- Provides a formalism for sequential decision-making under uncertainty.
- Foundation for reinforcement learning algorithms.

### Pros vs. Cons
- **Pros:** Mathematically tractable, generalizable, supports theoretical analysis.
- **Cons:** Assumes full observability, limited to finite state/action spaces.

### Recent Developments
- Extensions to **Partially Observable MDPs (POMDPs)**.
- Scalable algorithms for large state/action spaces (e.g., Deep RL).

---

## 3.2 Goals and Rewards

### Definition
**Goals** define the agent's objectives, typically encoded via a **reward signal**.

### Equations
- **Reward Function:**
  $$
  R(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
  $$
- **Cumulative Reward (Return):**
  $$
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
  $$

### Principles
- **Reward Hypothesis:** All goals can be described by maximizing expected cumulative reward.

### Detailed Analysis
- **Shaping:** Reward design critically affects learning efficiency and policy quality.
- **Sparse vs. Dense Rewards:** Sparse rewards are challenging for exploration; dense rewards may bias learning.

### Importance
- Central to agent behavior; improper reward design leads to suboptimal or unintended behaviors.

### Pros vs. Cons
- **Pros:** Flexible, general-purpose.
- **Cons:** Reward hacking, misalignment with true objectives.

### Recent Developments
- **Inverse RL:** Inferring reward functions from expert behavior.
- **Reward Learning:** Human-in-the-loop reward specification.

---

## 3.3 Returns and Episodes

### Definition
**Return ($G_t$):** Total accumulated reward from time $t$ onward.

### Equations
- **Episodic Tasks:**
  $$
  G_t = R_{t+1} + R_{t+2} + \cdots + R_T
  $$
- **Continuing Tasks (Discounted):**
  $$
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
  $$

### Principles
- **Episodic:** Tasks with terminal states.
- **Continuing:** Tasks with no natural endpoint.

### Detailed Analysis
- **Discount Factor ($\gamma$):** Controls trade-off between immediate and future rewards.

### Importance
- Defines the objective for policy optimization.

### Pros vs. Cons
- **Pros:** Unified framework for both episodic and continuing tasks.
- **Cons:** Choice of $\gamma$ can be non-trivial.

### Recent Developments
- Adaptive discounting strategies.

---

## 3.4 Unified Notation for Episodic and Continuing Tasks

### Definition
A **unified notation** allows both episodic and continuing tasks to be described using the same mathematical formalism.

### Equations
- **General Return:**
  $$
  G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
  $$
  where $T$ is the terminal time for episodic tasks, $T = \infty$ for continuing tasks.

### Principles
- **Terminal State Convention:** For episodic tasks, rewards after terminal state are zero.

### Detailed Analysis
- Simplifies algorithm design and analysis.

### Importance
- Enables general-purpose RL algorithms.

### Pros vs. Cons
- **Pros:** Reduces notational and implementation complexity.
- **Cons:** May obscure task-specific nuances.

### Recent Developments
- Generalized value functions for multi-timescale prediction.

---

## 3.5 Policies and Value Functions

### Definition
- **Policy ($\pi$):** Mapping from states to action probabilities.
- **Value Function ($v_\pi$):** Expected return from a state under policy $\pi$.
- **Action-Value Function ($q_\pi$):** Expected return from a state-action pair under $\pi$.

### Equations
- **Policy:**
  $$
  \pi(a \mid s) = \Pr\{A_t = a \mid S_t = s\}
  $$
- **State Value Function:**
  $$
  v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
  $$
- **Action Value Function:**
  $$
  q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
  $$

### Principles
- **Bellman Equations:**
  $$
  v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) [r + \gamma v_\pi(s')]
  $$
  $$
  q_\pi(s, a) = \sum_{s', r} P(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a')]
  $$

### Detailed Analysis
- **Value functions** quantify long-term desirability of states/actions.
- **Policy improvement** uses value functions to derive better policies.

### Importance
- Core to most RL algorithms (policy iteration, value iteration, Q-learning).

### Pros vs. Cons
- **Pros:** Enables efficient policy evaluation and improvement.
- **Cons:** Value estimation can be computationally intensive.

### Recent Developments
- Deep value function approximation (Deep Q-Networks, Actor-Critic methods).

---

## 3.6 Optimal Policies and Optimal Value Functions

### Definition
- **Optimal Policy ($\pi^*$):** Policy that maximizes expected return from every state.
- **Optimal Value Function ($v_*$, $q_*$):** Maximum expected return achievable from a state or state-action pair.

### Equations
- **Optimal State Value:**
  $$
  v_*(s) = \max_\pi v_\pi(s)
  $$
- **Optimal Action Value:**
  $$
  q_*(s, a) = \max_\pi q_\pi(s, a)
  $$
- **Bellman Optimality Equations:**
  $$
  v_*(s) = \max_a \sum_{s', r} P(s', r \mid s, a) [r + \gamma v_*(s')]
  $$
  $$
  q_*(s, a) = \sum_{s', r} P(s', r \mid s, a) [r + \gamma \max_{a'} q_*(s', a')]
  $$

### Principles
- **Policy Improvement Theorem:** Greedy policies w.r.t. $v_*$ or $q_*$ are optimal.

### Detailed Analysis
- **Value Iteration:** Iteratively update value estimates using Bellman optimality.
- **Policy Iteration:** Alternate between policy evaluation and improvement.

### Importance
- Defines the goal of RL: finding $\pi^*$.

### Pros vs. Cons
- **Pros:** Guarantees optimality in finite MDPs.
- **Cons:** Computationally expensive for large state/action spaces.

### Recent Developments
- Approximate dynamic programming, deep RL for large-scale problems.

---

## 3.7 Optimality and Approximation

### Definition
**Optimality** refers to achieving the best possible policy/value function. **Approximation** is necessary when exact computation is infeasible.

### Equations
- **Approximate Value Function:**
  $$
  \hat{v}(s; \theta) \approx v_*(s)
  $$
  where $\theta$ are parameters (e.g., neural network weights).

### Principles
- **Function Approximation:** Use parameterized models to estimate value functions/policies.
- **Trade-off:** Bias-variance, generalization vs. accuracy.

### Detailed Analysis
- **Linear/Nonlinear Approximation:** Linear basis functions vs. deep neural networks.
- **Convergence Guarantees:** May be lost with nonlinear function approximation.

### Importance
- Enables RL in high-dimensional/continuous spaces.

### Pros vs. Cons
- **Pros:** Scalability, generalization.
- **Cons:** Instability, divergence, lack of theoretical guarantees.

### Recent Developments
- Deep RL (DQN, DDPG, PPO, SAC).
- Stability techniques: target networks, experience replay, regularization.

