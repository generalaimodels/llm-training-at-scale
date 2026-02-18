# Finite Markov Decision Processes (FMDPs)

---

## 3.1 The Agent–Environment Interface  

### Definition  
An FMDP models interaction between an $agent$ and an $environment$ that evolves over discrete time steps $t=0,1,2,\dots$.  

### Pertinent Equations  
$$S_{t+1}\sim P(\cdot\,|\,S_t=s,A_t=a),\quad R_{t+1}\sim R(\cdot\,|\,S_t=s,A_t=a)$$  
$$\pi(a|s)=\Pr(A_t=a\,|\,S_t=s)$$  

### Key Principles  
- Perception–action loop: $(S_t,A_t,R_{t+1},S_{t+1})$  
- Markov property: $\Pr(S_{t+1},R_{t+1}\,|\,\text{history}) = \Pr(S_{t+1},R_{t+1}\,|\,S_t,A_t)$  

### Detailed Concept Analysis  
- State set $\mathcal S$ and action set $\mathcal A$ are finite.  
- Transition kernel $P:\mathcal S\times\mathcal A\to\Delta(\mathcal S)$ encodes dynamics.  
- Reward kernel $R:\mathcal S\times\mathcal A\to\Delta(\mathbb R)$ encodes stochastic feedback.  
- Interface abstracts hardware / software, enabling algorithm–theory unification.  

### Importance  
- Provides canonical abstraction for reinforcement learning (RL).  
- Enables convergence proofs, sample-complexity analysis, and algorithmic benchmarking.  

### Pros vs Cons  
Pros: mathematically tractable; expressive enough for many tasks.  
Cons: finite assumption limits realism; perfect Markov state seldom available.  

### Cutting-Edge Advances  
- Latent-state inference bridges non-Markov observations to FMDP latent dynamics.  
- Environment simulators leverage learned $P$ and $R$ for planning (world-models).  

---

## 3.2 Goals and Rewards  

### Definition  
Goals encoded via scalar reward signal $R_{t+1}$ delivered by environment to agent.  

### Pertinent Equations  
$$G_t=\sum_{k=0}^\infty \gamma^{k} R_{t+k+1}$$  

### Key Principles  
- Reward hypothesis: any goal can be formulated as reward maximization.  
- Discount factor $\gamma\in[0,1)$ balances immediacy vs. long-term gain.  

### Detailed Concept Analysis  
- Shaping techniques modify $R$ without altering optimal policy.  
- Sparse vs. dense rewards trade off exploration vs. credit assignment.  

### Importance  
- Reward design critically impacts learning stability and sample efficiency.  

### Pros vs Cons  
Pros: unifies diverse tasks; scalar simplicity.  
Cons: mis-specified rewards yield unintended behavior; hard to hand-craft.  

### Cutting-Edge Advances  
- Inverse RL and preference learning infer $R$ from demonstrations.  
- Reward randomization combats overfitting to proxy metrics.  

---

## 3.3 Returns and Episodes  

### Definition  
Return $G_t$ is cumulative, possibly discounted, future reward; episodes are finite trajectories terminated by absorbing state $S_T$.  

### Pertinent Equations  
Episodic: $$G_t=\sum_{k=0}^{T-t-1} R_{t+k+1}$$  
Continuing: $$G_t=\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}$$  

### Key Principles  
- Episode length distribution impacts variance of return estimates.  
- Discounting ensures bounded returns for continuing tasks.  

### Detailed Concept Analysis  
- Risk-sensitive metrics (e.g., CVaR) applied to $G_t$ for safety-critical RL.  

### Importance  
- Defines objective optimized by value-based or policy-gradient algorithms.  

### Pros vs Cons  
Pros: flexible horizon modeling; discount ensures mathematical convergence.  
Cons: discount selection subjective; episodic resets may be unrealistic.  

### Cutting-Edge Advances  
- Variable-horizon methods learn adaptive $\gamma_t$.  
- Episodic-memory agents reuse past $S_T$ for credit assignment.  

---

## 3.4 Unified Notation for Episodic and Continuing Tasks  

### Definition  
Employ generalized discounting $\gamma_t\in[0,1]$ and termination indicator $\tau_t$ to fuse episodic and continuing formulations.  

### Pertinent Equations  
$$G_t=\sum_{k=0}^{\infty}\left(\prod_{j=0}^{k-1}\gamma_{t+j}\right)R_{t+k+1},\quad \gamma_{t}=1-\tau_t$$  

### Key Principles  
- Termination encoded as $\gamma_t=0$.  
- Allows algorithms to operate without special-case code paths.  

### Detailed Concept Analysis  
- Enables generalized advantage estimation and off-policy corrections across task types.  

### Importance  
- Simplifies theoretical analysis and software libraries.  

### Pros vs Cons  
Pros: unified APIs; streamlined proofs.  
Cons: dynamic $\gamma_t$ complicates eligibility traces in TD($\lambda$).  

### Cutting-Edge Advances  
- Meta-learning $\gamma_t$ for non-stationary environments.  

---

## 3.5 Policies and Value Functions  

### Definition  
Policy $\pi$ maps states to probability distributions over actions; value functions quantify expected return under $\pi$.  

### Pertinent Equations  
State-value: $$v_\pi(s)=\mathbb E_\pi[G_t\,|\,S_t=s]$$  
Action-value: $$q_\pi(s,a)=\mathbb E_\pi[G_t\,|\,S_t=s,A_t=a]$$  
Bellman:  
$$v_\pi(s)=\sum_{a}\pi(a|s)\sum_{s',r}P(s',r|s,a)\big[r+\gamma v_\pi(s')\big]$$  

### Key Principles  
- Bellman expectation operator $\mathcal T_\pi$.  
- Contraction mapping under $\gamma<1$.  

### Detailed Concept Analysis  
- Function approximation replaces tabular $v_\pi$ with parametric $v_\pi(s;\theta)$.  
- Policy gradient theorem uses $q_\pi$ for $\nabla J(\theta)$.  

### Importance  
- Central to prediction (policy evaluation) and control (policy improvement).  

### Pros vs Cons  
Pros: decomposes temporal credit; facilitates bootstrapping.  
Cons: high-dimensional $s$ causes approximation errors; bootstrapping can destabilize.  

### Cutting-Edge Advances  
- Distributional value functions model full return distribution $Z_\pi(s,a)$.  
- Implicit policies via energy-based models extend $\pi$.  

---

## 3.6 Optimal Policies and Optimal Value Functions  

### Definition  
Optimal state-value $v_*(s)=\max_\pi v_\pi(s)$; optimal action-value $q_*(s,a)=\max_\pi q_\pi(s,a)$.  

### Pertinent Equations  
Bellman optimality:  
$$v_*(s)=\max_{a}\sum_{s',r}P(s',r|s,a)\big[r+\gamma v_*(s')\big]$$  
$$q_*(s,a)=\sum_{s',r}P(s',r|s,a)\big[r+\gamma \max_{a'} q_*(s',a')\big]$$  

### Key Principles  
- Greedy policy $\pi_*(s)=\arg\max_a q_*(s,a)$.  
- Monotonic improvement via policy iteration.  

### Detailed Concept Analysis  
- Value-iteration: repeated application of optimality operator $\mathcal T_*$.  
- Approximate dynamic programming handles large $\mathcal S$.  

### Importance  
- Defines gold-standard performance bound for RL algorithms.  

### Pros vs Cons  
Pros: principled objective; contraction guarantees convergence (finite case).  
Cons: intractable for huge state spaces; max operator induces non-smoothness.  

### Cutting-Edge Advances  
- Soft-optimality ($\max\to\text{log-sum-exp}$) yields entropy-regularized policies.  
- Neural MCTS integrates value nets and search to approximate $q_*$.  

---

## 3.7 Optimality and Approximation  

### Definition  
Examines deviation between approximate solutions $\hat v,\hat\pi$ and optimal counterparts $v_*,\pi_*$.  

### Pertinent Equations  
Performance loss bound:  
$$J(\pi_*)-J(\hat\pi)\le \frac{2\gamma}{(1-\gamma)^2}\max_s |v_*(s)-\hat v(s)|$$  

### Key Principles  
- Bias–variance trade-off in function approximation.  
- Concentration coefficients link sample complexity to visitation distribution.  

### Detailed Concept Analysis  
- Projected Bellman error $\|\Pi \mathcal T_\pi \hat v - \hat v\|$ guides algorithm design (e.g., TD(0)).  
- Over-parameterized networks can interpolate and still generalize (neural Tangent Kernel view).  

### Importance  
- Quantifies reliability of deployed RL systems.  

### Pros vs Cons  
Pros: explicit error guarantees; informs sample allocation.  
Cons: bounds often loose; require strong assumptions (e.g., realizability).  

### Cutting-Edge Advances  
- Double sampling and gradient correction reduce deadly triad instability.  
- Representation learning aligns approximation spaces with task-relevant structure.