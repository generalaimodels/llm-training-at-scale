### 4 Dynamic Programming (DP)

---

#### 4.1 Policy Evaluation (Prediction)

Definition  
• Compute $v_\pi(s)=\mathbb{E}_\pi\!\left[\sum_{t=0}^\infty\gamma^{t}R_{t+1}\mid S_0=s\right]$ for a fixed policy $\pi$.

Pertinent Equations  
$$v_\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\,v_\pi(s')\right]$$  
Tabular iterative update (“Bellman backup”):  
$$v_{k+1}(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\,v_k(s')\right]$$

Key Principles  
• Bellman expectation operator $\mathcal{T}_\pi$ is a $\gamma$-contraction in $\ell_\infty$ norm.  
• Successive approximation converges to unique fixed point $v_\pi$.  
• Accuracy controlled by sweep count or error threshold $\theta$.

Detailed Concept Analysis  
• Complexity per sweep: $\mathcal{O}(|\mathcal{S}||\mathcal{A}|)$ for tabular MDPs.  
• In large state spaces, use function approximation: $v_\pi\!\approx\!\hat v_\pi(s;\mathbf{w})$ with TD(0) or TD($\lambda$).  
• Stopping criterion: $\max_s|v_{k+1}(s)-v_k(s)|<\theta$.

Importance  
• Baseline for evaluating a policy before improvement.  
• Component of many actor–critic, policy-gradient, and offline RL algorithms.

Pros vs Cons  
+ Guaranteed convergence (tabular, $\gamma<1$)  
+ Provides unbiased value estimates  
– Requires full model $p(s',r\mid s,a)$  
– $\mathcal{O}(|\mathcal{S}|^2)$ memory for naive backups in dense transitions

Cutting-Edge Advances  
• Differential TD learning for average-reward settings.  
• Linear-time “low-rank MDP” solvers exploiting spectral structure.  
• GPU-parallel batched Bellman backups.

---

#### 4.2 Policy Improvement

Definition  
• Construct a better policy $\pi'$ from value function $v_\pi$.

Pertinent Equations  
Greedy improvement:  
$$\pi'(s)=\arg\max_{a}\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\,v_\pi(s')\right]$$

Key Principles  
• Policy-improvement theorem: If $\pi'(s)$ maximizes $q_\pi(s,a)$, then $v_{\pi'}(s)\ge v_\pi(s)\ \forall s$.  
• Deterministic tie-breaking retains monotonicity.

Detailed Concept Analysis  
• Requires $q_\pi(s,a)$; compute via one-step look-ahead or Monte Carlo estimates.  
• Soft/improper improvement: $\epsilon$-greedy or Boltzmann for exploration.

Importance  
• Fundamental step driving convergence toward optimal policy.

Pros vs Cons  
+ Monotone performance increase  
– Relies on accurate $v_\pi$ or $q_\pi$  
– Greedy step may reduce exploration

Cutting-Edge Advances  
• Conservative policy improvement (CPI) to bound performance drop under approximation error.  
• KL-regularized improvement (e.g., TRPO, PPO).

---

#### 4.3 Policy Iteration

Definition  
• Alternating sequence of policy evaluation and policy improvement until convergence to $\pi_\ast$.

Pertinent Equations  
Initialization: $\pi_0$ arbitrary  
Loop:  
$$v_{k}\leftarrow \text{Eval}(\pi_k)$$  
$$\pi_{k+1}\leftarrow \text{Greedy}(v_{k})$$

Key Principles  
• Finite MDP ⇒ finite iterations ($\le |\mathcal{S}||\mathcal{A}|$) to reach optimality.  
• Convergence accelerated via partial evaluation (truncated sweeps).

Detailed Concept Analysis  
• Exact policy evaluation → computationally heavy; use $\kappa$ sweeps or TD to approximate.  
• Variants: modified PI, optimistic PI, aggregated PI.

Importance  
• Canonical DP algorithm, basis for many RL approaches.

Pros vs Cons  
+ Provably optimal  
– High per-iteration cost  
– Requires model

Cutting-Edge Advances  
• Anderson acceleration on policy evaluation stage.  
• Batch PI on distributed clusters.

---

#### 4.4 Value Iteration

Definition  
• Directly iterate Bellman optimality operator on state values.

Pertinent Equations  
$$v_{k+1}(s)=\max_{a}\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\,v_{k}(s')\right]$$  
Policy extraction: $\pi_k(s)=\arg\max_a\{\cdot\}$.

Key Principles  
• Operator $\mathcal{T}_\ast$ is a $\gamma$-contraction ⇒ convergence to $v_\ast$.  
• Combine evaluation & improvement every sweep.

Detailed Concept Analysis  
• Convergence rate: $\|v_k-v_\ast\|_\infty\le\gamma^k \|v_0-v_\ast\|_\infty$.  
• Practical stop rule: stop when $\max_s|v_{k+1}(s)-v_k(s)|<\epsilon\,(1-\gamma)/(2\gamma)$.

Importance  
• Often faster than PI for high-discount problems.  
• Preferred when goal is optimal value only.

Pros vs Cons  
+ Lower memory than PI  
+ Anytime algorithm  
– Oscillatory value estimates before convergence  
– Still needs model

Cutting-Edge Advances  
• Prioritized sweeping orders updates by Bellman residual magnitude.  
• Neural value iteration networks (VIN) for differentiable planning.

---

#### 4.5 Asynchronous Dynamic Programming

Definition  
• Update states in any order instead of synchronous sweeps.

Pertinent Equations  
For selected state $s_t$:  
$$v(s_t)\leftarrow\mathcal{B}(v)(s_t)$$  
where $\mathcal{B}$ is expectation or optimality backup.

Key Principles  
• Convergence preserved if every state is visited infinitely often and steps are stochastic or cyclic.

Detailed Concept Analysis  
• Enables online, real-time DP.  
• Supports Gauss-Seidel style faster convergence.

Importance  
• Bridges DP and temporal-difference learning.  
• Crucial for large or unknown models.

Pros vs Cons  
+ Memory-efficient  
+ Supports partial model knowledge  
– Requires careful scheduling for speed  
– Harder to parallelize

Cutting-Edge Advances  
• GPU warp-level asynchronous value iteration.  
• Event-driven DP for embedded path planners.

---

#### 4.6 Generalized Policy Iteration (GPI)

Definition  
• Any simultaneous or interleaved application of evaluation and improvement processes.

Pertinent Equations  
No fixed form; conceptual framework:  
$$\pi_{k+1} \approx \text{Improve}(v_k),\quad v_{k+1}\approx\text{Eval}(\pi_{k+1})$$

Key Principles  
• Mutual feedback loop; convergence driven by monotone improvement & contraction of evaluation.

Detailed Concept Analysis  
• Covers PI, value iteration, actor–critic, DQN, PPO.  
• Allows approximate, stochastic, or partial steps in either component.

Importance  
• Unifies planning, prediction, and control algorithms.

Pros vs Cons  
+ Flexible trade-off between computation and data  
– Approximation errors may break guarantees

Cutting-Edge Advances  
• Off-policy GPI with importance weighting.  
• Multi-task GPI with shared successor features.

---

#### 4.7 Efficiency of Dynamic Programming

Definition  
• Measure of computational and sample resources required by DP variants.

Pertinent Equations  
Time complexity per full sweep (tabular):  
$$\mathcal{O}(|\mathcal{S}|^2|\mathcal{A}|)$$  
Memory: $$\mathcal{O}(|\mathcal{S}|)$$ for $v$, $$\mathcal{O}(|\mathcal{S}||\mathcal{A}|)$$ for $q$.

Key Principles  
• Efficiency improved by decomposition, caching, prioritization, and approximation.

Detailed Concept Analysis  
• Sparse MDPs reduce cost to $\mathcal{O}(|E|)$ where $|E|$ is number of non-zero transitions.  
• Factored MDPs exploit conditional independence via dynamic Bayesian networks.  
• Function approximation cuts memory but adds bias.

Importance  
• Practical applicability to large-scale planning hinges on efficiency.

Pros vs Cons  
+ Exact optimality (tabular)  
– Curse of dimensionality limits raw DP to small MDPs

Cutting-Edge Advances  
• Tensor-train compression for high-dimensional value functions.  
• Learned model rollouts with neural DP surrogates.