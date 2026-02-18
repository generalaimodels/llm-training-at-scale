# Planning and Learning with Tabular Methods  

---

## 8.1 Models and Planning  

### Definition  
A **model** is any function that, given a state–action pair $(s,a)$, returns a prediction of the resulting next-state distribution $p(\,\cdot\mid s,a)$ and reward distribution $r(\,\cdot\mid s,a)$. **Planning** is the process of improving a decision-making policy by repeatedly querying the model instead of, or in addition to, interacting with the real environment.

### Pertinent Equations  
$$\hat{p}(s',r\mid s,a)=\Pr\big\{S_{t+1}=s',\;R_{t+1}=r\mid S_t=s,\,A_t=a\big\}$$  
$$\hat{r}(s,a)=\mathbb{E}[R_{t+1}\mid S_t=s,A_t=a]$$  
Model-based update (1-step look-ahead):  
$$Q_{t+1}(s,a)\leftarrow (1-\alpha)Q_t(s,a)+\alpha\sum_{s',r}\hat{p}(s',r\mid s,a)\big[r+\gamma\max_{a'}Q_t(s',a')\big]$$  

### Key Principles  
• Separation of data collection (real experience) and hypothetical experience (model rollouts).  
• **Sample efficiency:** fewer real interactions are needed.  
• **Bias–variance trade-off:** accuracy of the learned model controls the bias.

### Detailed Concept Analysis  
1. **Deterministic vs. stochastic models:** deterministic store a single $(s',r)$; stochastic store full distributions or a set of samples.  
2. **Complete vs. partial models:** full transition matrices vs. factored approximations (e.g., independent feature models).  
3. **Tabular storage:** counts $N(s,a)$, empirical means $\bar{r}(s,a)$, and empirical transition probabilities $\bar{p}(s'|s,a)=N(s,a,s')/N(s,a)$.

### Importance  
Models allow re-use of past real data for unlimited virtual experience, drastically reducing sample complexity in domains where real interaction is expensive.

### Pros vs Cons  
Pros  
• Dramatic sample-efficiency gains.  
• Enables look-ahead search and risk assessment.  
• Facilitates transfer and imagination-based exploration.  

Cons  
• Model learning introduces bias.  
• Storage/computation overhead.  
• Brittle when the true environment is non-stationary.

### Cutting-Edge Advances  
• Learned latent-dynamics models (e.g., MuZero).  
• Uncertainty-aware models using Bayesian methods or ensembles.  
• World-model pre-training for few-shot RL.

---

## 8.2 Dyna: Integrated Planning, Acting, and Learning  

### Definition  
**Dyna** is a framework that interleaves direct RL updates from real experience with planning updates from simulated model experience within each time step.

### Pertinent Equations  
Baseline real-experience TD update:  
$$Q\leftarrow Q+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]$$  
For $n$ simulated updates: draw $(\tilde{s},\tilde{a})\sim$ past experience buffer  
$$Q\leftarrow Q+\alpha\big[\tilde{r}+\gamma\max_{a'}Q(\tilde{s}',a')-Q(\tilde{s},\tilde{a})\big]$$  

### Key Principles  
• Unify learning ($\pi$ evaluation), acting (choose $a$), and planning (model rollouts) in one loop.  
• **Any-time improvement:** more planning steps $n$ ⇒ higher data efficiency.  
• Compatible with on-policy or off-policy control.

### Detailed Concept Analysis  
1. **Dyna-Q algorithm:** maintains table $Q(s,a)$ and empirical model $\hat{M}$. Each real step does:  
   a. TD update;  
   b. Save $(s,a,r,s')$ to model;  
   c. Repeat $n$ planning updates using $\hat{M}$.  
2. **Trade-off parameter $n$** controls compute vs. sample efficiency.  
3. **Variants:** Dyna-Q+, Dyna-2, Prioritized Dyna.

### Importance  
Proved that a small number of simulated updates (e.g., $n\!=\!5$) can match performance of pure model-free methods needing orders of magnitude more real samples.

### Pros vs Cons  
Pros  
• Simple, incremental, fully online.  
• Graceful degradation to model-free if the model is poor.  
• Extensible to continuous spaces with function approximators.

Cons  
• Requires careful scheduling of planning steps.  
• Model errors propagate via repeated updates.

### Cutting-Edge Advances  
• Dyna-style imagination in AlphaZero/MuZero.  
• Integrated gradient world models for deep Dyna.  
• Adaptive planning budgets based on uncertainty or value of information.

---

## 8.3 When the Model Is Wrong  

### Definition  
Model imperfection introduces systematic bias into planning updates, which can misguide value estimates and policy improvement.

### Pertinent Equations  
Error term on $Q$ after $k$ simulated steps:  
$$\epsilon_k(s,a)=\gamma^k\mathbb{E}_{\hat{M}}\Big[\sum_{t=0}^{k-1}\gamma^t\big(\hat{r}_t-r_t\big)\Big]$$  

### Key Principles  
• **Model bias accumulation:** repeated rollouts amplify small errors.  
• **Trust region planning:** limit rollout depth or number of updates based on model fidelity.  
• **Uncertainty quantification:** use $\hat{\sigma}(s,a)$ to weight updates.

### Detailed Concept Analysis  
1. **Short-horizon planning:** limit hypothetical trajectory length to reduce bias.  
2. **Stochastic ensemble models:** average multiple predictions to reduce error variance.  
3. **Discrepancy-based learning rates:** shrink $\alpha$ when model prediction disagrees with recent real data.

### Importance  
Robustness to model error is crucial for deploying model-based RL in safety-critical domains.

### Pros vs Cons  
Pros  
• Incorporates realistic scenario where perfect modeling is impossible.  
Cons  
• Requires error-aware mechanisms, increasing algorithmic complexity.

### Cutting-Edge Advances  
• Model-based Value Expansion (MVE) in continuous control.  
• Cross-entropy regularisation of imaginary targets (DreamerV3).  
• Safe policy optimization under model uncertainty (Robust MBRL).

---

## 8.4 Prioritized Sweeping  

### Definition  
A planning method that uses a priority queue to focus simulated backups on state–action pairs whose predecessors are likely to incur large value changes.

### Pertinent Equations  
Priority measure:  
$$P(s,a)=\big| r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big|$$  

### Key Principles  
• **Backward focusing:** propagate unexpected TD errors efficiently.  
• Number of simulated updates determined by queue size rather than fixed $n$.

### Detailed Concept Analysis  
1. **Algorithm:**  
   a. After real transition, compute priority $P(s,a)$; insert predecessors $(\bar{s},\bar{a})$ where $\hat{p}(s|\,\bar{s},\bar{a})>0$.  
   b. Pop highest-priority pair; perform one‐step model backup; recompute priorities for its own predecessors.  
2. **Sparse transitions:** yields $\mathcal{O}(\log N)$ heap operations instead of scanning full table.

### Importance  
Greatly reduces planning updates needed, crucial for high-dimensional tabular tasks.

### Pros vs Cons  
Pros  
• Focused computation; handles non-uniform dynamics elegantly.  
Cons  
• Requires efficient predecessor enumeration; heavy memory in dense graphs.

### Cutting-Edge Advances  
• Distributed prioritized sweeping for large graphs.  
• Neural-network approximation of predecessor sets via conditional generative models.

---

## 8.5 Expected vs. Sample Updates  

### Definition  
Contrast between backups using full expectation over successor states versus single stochastic samples.

### Pertinent Equations  
Expected backup:  
$$V(s)=\sum_{s',r}p(s',r\mid s,a)[\,r+\gamma V(s')\,]$$  
Sample backup (Monte Carlo):  
$$V(s)\approx r+\gamma V(s')$$  

### Key Principles  
• **Computation vs. variance trade-off**: expectation reduces variance but costs more CPU.  
• Sample backups enable incremental online updates.

### Detailed Concept Analysis  
1. In tabular settings, expectation is feasible when transition counts small; otherwise sampling preferred.  
2. Dyna uses sampled backups for planning efficiency.

### Importance  
Selecting correct backup type is critical for balancing computational and statistical efficiency.

### Pros vs Cons  
Expected  
+ Low variance  
− High computation  

Sample  
+ Cheap per update  
− Higher variance requiring more updates  

### Cutting-Edge Advances  
• Variance-reduced sampling via control variates.  
• GPU-accelerated batched expectations in large discrete MDPs.

---

## 8.6 Trajectory Sampling  

### Definition  
Planning method generating entire simulated episodes (or partial rollouts) from the model, then using Monte-Carlo or n-step returns for value updates.

### Pertinent Equations  
n-step target from imaginary trajectory:  
$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V(s_{t+n})$$  

### Key Principles  
• Leverages **multi-step returns** to propagate rewards more deeply per simulated episode.  
• Allows using eligibility traces in imagined experience.

### Detailed Concept Analysis  
1. **Length selection:** longer $n$ improves credit assignment but increases model bias.  
2. **Imagination buffer:** store simulated episodes to reuse for off-policy learning.

### Importance  
Essential for planning in domains with sparse rewards.

### Pros vs Cons  
Pros  
• Deep propagation without full search tree.  
Cons  
• Exponentially growing model error with $n$.

### Cutting-Edge Advances  
• Model-based policy evaluation with importance-weighted imaginary trajectories.

---

## 8.7 Real-Time Dynamic Programming (RTDP)  

### Definition  
An asynchronous dynamic-programming method executing backups only along states reachable from start and guided by current policy or heuristic.

### Pertinent Equations  
One-step RTDP backup:  
$$V(s)\leftarrow \max_a\big[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V(s')\big]$$  

### Key Principles  
• **On-line, goal-directed:** focuses on relevant subset of state space.  
• Converges to optimal $V$ under admissible heuristic and sufficient exploration.

### Detailed Concept Analysis  
1. Each real or simulated move triggers local DP backup.  
2. Enhancements: LRTDP (labelled RTDP) adds convergence detection via residual thresholds.

### Importance  
Allows solving large deterministic MDPs where full DP is impossible.

### Pros vs Cons  
Pros  
• Memory efficient; anytime performance.  
Cons  
• Requires informative heuristic to be effective.

### Cutting-Edge Advances  
• Neural-guided RTDP applying learned heuristics.  
• GPU-based parallel RTDP for grid worlds.

---

## 8.8 Planning at Decision Time  

### Definition  
Planning performed only when an action decision is required, rather than continuously in the background.

### Pertinent Equations  
At decision state $s$: perform depth-$d$ look-ahead tree search; choose  
$$a^*=\arg\max_a \mathbb{E}\big[G_d\mid s,a,\hat{M}\big]$$  

### Key Principles  
• Allocates computation adaptively to critical states.  
• Matches resource constraints in embedded systems.

### Detailed Concept Analysis  
1. Combines reactive cached policy with on-the-fly search refinements.  
2. Methods include MCTS and rollout algorithms.

### Importance  
Key for real-time control in robotics and games.

### Pros vs Cons  
Pros  
• Focuses CPU where it matters.  
Cons  
• Cold-start latency for first decision; requires fast model querying.

### Cutting-Edge Advances  
• Hardware-optimized search (e.g., AlphaGo Zero TPU tree search).  
• Anytime bounded-suboptimal search algorithms.

---

## 8.9 Heuristic Search  

### Definition  
Use of heuristic value estimates $h(s)$ to guide forward search toward promising regions.

### Pertinent Equations  
A* evaluation function in deterministic domains:  
$$f(s)=g(s)+h(s)$$  
where $g$ is path cost so far.

### Key Principles  
• **Admissibility:** $h(s)\le V^*(s)$ guarantees optimality.  
• Heuristic learning via RL (e.g., bootstrapping $h$ with $V$).

### Detailed Concept Analysis  
1. **Heuristic DP:** integrate heuristic into DP backups to accelerate convergence.  
2. **Learning heuristics:** use approximate value functions as informed estimates.

### Importance  
Transforms exponential search into practically tractable problems.

### Pros vs Cons  
Pros  
• Significant pruning of search tree.  
Cons  
• Designing/learning good heuristics is non-trivial.

### Cutting-Edge Advances  
• Neural heuristics for combinatorial optimisation (Graph NN A*).

---

## 8.10 Rollout Algorithms  

### Definition  
Policy improvement method that uses a base policy $\pi_b$ and evaluates one-step look-ahead by rolling out $\pi_b$ to estimate action values.

### Pertinent Equations  
Rollout estimate:  
$$Q^{\text{rol}}(s,a)=\frac{1}{K}\sum_{k=1}^K \Big[r_{0}^{(k)}+\sum_{t=1}^{T-1}\gamma^t r_{t}^{(k)}\Big]$$  

### Key Principles  
• **Policy iteration flavour:** $\pi_{new}(s)=\arg\max_a Q^{\text{rol}}(s,a)$.  
• Uses Monte-Carlo evaluation with controlled horizon.

### Detailed Concept Analysis  
1. **Base policy quality** bounds performance of the rollout policy.  
2. **Parallel rollouts** on GPU/cluster to reduce decision latency.

### Importance  
Forms the core of AmoebaNet architecture search and early game AI agents.

### Pros vs Cons  
Pros  
• Simple; black-box base policy.  
Cons  
• High variance; expensive sampling.

### Cutting-Edge Advances  
• Learned cost models to predict rollout outcome without full simulation.

---

## 8.11 Monte Carlo Tree Search (MCTS)  

### Definition  
Best-first search algorithm that builds a partial search tree using random simulations to evaluate leaf nodes and UCT (Upper Confidence bounds applied to Trees) to balance exploration‐exploitation.

### Pertinent Equations  
UCT action selection at tree node $s$:  
$$a=\arg\max_a \big[ \bar{Q}(s,a)+c\sqrt{\frac{\ln N(s)}{N(s,a)}} \big]$$  

### Key Principles  
• **Four phases:** selection, expansion, simulation, back-propagation.  
• Converges to optimal decision with logarithmic regret under certain assumptions.

### Detailed Concept Analysis  
1. **Statistical guarantees** via bandit theory.  
2. **Virtual loss** and parallel tree search implementations.  
3. Integration with function approximators for prior $P(a|s)$ (AlphaZero).

### Importance  
Dominant method in modern game AI (Go, Chess, Atari) and combinatorial planning tasks.

### Pros vs Cons  
Pros  
• Asymptotically optimal; anytime decision quality.  
• Works without heuristic evaluation.  

Cons  
• Large memory/time for high branching factor.  
• Requires many simulations in stochastic domains.

### Cutting-Edge Advances  
• Neural-guided priors and value networks (MuZero, GATO MCTS).  
• Differentiable tree backup operators enabling end-to-end training.

---