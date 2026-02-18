## 6 Temporal-Difference Learning  

### 6.1 TD Prediction  
**Definition**  
Predict the value function $V_\pi(s)$ by bootstrapping from successive state estimates while following policy $\pi$.  

**Pertinent Equations**  
$$V_\pi(s)\!\leftarrow\!V_\pi(s)+\alpha\bigl[R_{t+1}+\gamma V_\pi(S_{t+1})-V_\pi(s)\bigr]$$  
TD-error: $$\delta_t=R_{t+1}+\gamma V_\pi(S_{t+1})-V_\pi(S_t)$$  

**Key Principles**  
• Bootstrapping (updating from current estimate)  
• Online, incremental updates  
• One-step return mixes Monte-Carlo target ($G_t$) and dynamic-programming bootstrapping  

**Detailed Concept Analysis**  
• TD uses single-sample transition—low variance relative to Monte-Carlo.  
• Converges to $V_\pi$ for tabular representation when $\alpha$ diminishes appropriately and $\pi$ explores every state.  
• Eligibility traces extend TD(0) to TD($\lambda$), weighting $n$-step returns geometrically.  

**Importance**  
• Foundation of modern RL (SARSA, Q-learning, actor–critic, Deep RL).  
• Enables real-time learning without full episodes.  

**Pros vs Cons**  
Pros: low memory, lower variance, online; Cons: biased target, sensitive to $\alpha$, bootstrapping can diverge with function approximation.  

**Cutting-Edge Advances**  
• Differential TD for continuing tasks;  
• Emphatic TD for off-policy stability;  
• Deep successor features linking TD to representation learning.  

---

### 6.2 Advantages of TD Prediction Methods  
**Definition**  
Comparative benefits over Monte-Carlo and dynamic-programming methods.  

**Pertinent Equations**  
Mean-squared-error decomposition:  
$$\text{MSE}=\text{Bias}^2+\text{Var}$$  

**Key Principles**  
• TD’s bias–variance trade-off more favorable in stochastic environments.  
• Computes during episode ($t\!<\!T$); no need for terminal reward.  

**Detailed Concept Analysis**  
• DP requires full model $P(s',r|s,a)$; TD does not.  
• MC methods unbiased but high variance; TD reduces variance by bootstrapping.  
• Eligibility traces bridge MC ($\lambda\!=\!1$) and TD(0).  

**Importance**  
Enables scalable RL in environments lacking models and with long horizons.  

**Pros vs Cons**  
Pros: data efficiency, temporal credit assignment;  
Cons: potential instability with non-linear function approximation.  

**Cutting-Edge Advances**  
Variance-reduced TD ($\mathrm{ACER}$, V-trace), adaptive $\lambda$ schedules, and uncertainty-aware TD for exploration.  

---

### 6.3 Optimality of TD(0)  
**Definition**  
Conditions under which TD(0) converges to unique fixed point $V_\pi$ minimizing MSPBE.  

**Pertinent Equations**  
MSPBE objective:  
$$\text{MSPBE}(\mathbf{w})=||\Pi T_\pi V_{\mathbf{w}}-V_{\mathbf{w}}||_D^2$$  
Projected Bellman operator $\Pi T_\pi$ is a $\gamma$-contraction.  

**Key Principles**  
• Contraction mapping guarantees unique fixed point.  
• For linear function approximation $V_{\mathbf w}=\mathbf{w}^\top\phi(s)$, TD(0) performs stochastic semi-gradient descent on MSPBE.  

**Detailed Concept Analysis**  
• Proof uses ordinary differential equation (ODE) method and stochastic approximation theorems.  
• Convergence requires: step-size $\alpha_t$ fulfilling $\sum\alpha_t=\infty,\sum\alpha_t^2<\infty$; feature matrix full rank; on-policy sampling.  

**Importance**  
Establishes theoretical backbone for using TD(0) in value estimation and policy evaluation with large state spaces.  

**Pros vs Cons**  
Pros: provable convergence (tabular/linear on-policy); Cons: divergent off-policy or with non-linear networks.  

**Cutting-Edge Advances**  
Gradient-TD algorithms (GTD2, TDC) and true-online TD($\lambda$) maintaining convergence beyond conventional assumptions.  

---

### 6.4 SARSA: On-Policy TD Control  
**Definition**  
On-policy algorithm updating action-value $Q_\pi(s,a)$ using quintuple $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$.  

**Pertinent Equations**  
$$Q(S_t,A_t)\!\leftarrow\!Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\bigr]$$  

**Key Principles**  
• Policy improvement embedded via $\epsilon$-greedy or softmax; learning and behavior policies coincide.  

**Detailed Concept Analysis**  
• Converges to $Q^\*$ when exploration decays ($\epsilon\!\to\!0$) and step-sizes diminish;  
• $\lambda$ extension: SARSA($\lambda$) with eligibility traces.  

**Importance**  
Robust where exploratory actions must be accounted for (e.g., windy grid-world).  

**Pros vs Cons**  
Pros: safe learning; captures exploratory risk;  
Cons: slower convergence than off-policy, policy-evaluation coupling restricts reuse of data.  

**Cutting-Edge Advances**  
• N-step SARSA in DeepMind’s Rainbow;  
• Distributional SARSA for full return distributions.  

---

### 6.5 Q-learning: Off-Policy TD Control  
**Definition**  
Off-policy method learning optimal $Q^\*(s,a)$ independent of behavior policy.  

**Pertinent Equations**  
$$Q(S_t,A_t)\!\leftarrow\!Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\bigr]$$  

**Key Principles**  
• Bootstraps toward greedy action;  
• Sample-efficient via off-policy reuse.  

**Detailed Concept Analysis**  
• Proven to converge in tabular case under sufficient exploration.  
• With function approximation, maximization bias and deadly triad (off-policy + bootstrapping + function approximation) can cause divergence.  

**Importance**  
Backbone of many Deep RL systems (DQN).  

**Pros vs Cons**  
Pros: decoupled data collection, faster optimality;  
Cons: over-estimation bias, instability with non-linear approximators.  

**Cutting-Edge Advances**  
Double DQN, dueling networks, noisy nets, prioritized replay tackling bias, exploration, and sample efficiency.  

---

### 6.6 Expected SARSA  
**Definition**  
Replaces sampled next action with expectation over behavior policy, reducing variance.  

**Pertinent Equations**  
$$Q(S_t,A_t)\!\leftarrow\!Q(S_t,A_t)+\alpha\Bigl[R_{t+1}+\gamma\!\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')-Q(S_t,A_t)\Bigr]$$  

**Key Principles**  
• Intermediate between SARSA (fully on-policy) and Q-learning (fully greedy).  

**Detailed Concept Analysis**  
• Lower variance due to expectation; bias same order as SARSA.  
• Useful in continuous action spaces where max operator expensive.  

**Importance**  
Foundational in algorithms like IQN and Rainbow’s “Expected SARSA” component.  

**Pros vs Cons**  
Pros: decreased variance, smoother updates;  
Cons: computation increases with action count, requires full distribution over actions.  

**Cutting-Edge Advances**  
Implicit quantile networks deploy expected SARSA on quantile distributions; soft Q-learning uses entropy-regularized expectation.  

---

### 6.7 Maximization Bias and Double Learning  
**Definition**  
Positive bias arises when $\max_{a} \hat Q$ is used as an estimator for $\max_{a} Q$; Double learning mitigates it.  

**Pertinent Equations**  
Double Q update:  
$$Q^{(1)}(S_t,A_t)\!\leftarrow\!Q^{(1)}(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma Q^{(2)}(S_{t+1},\arg\max_{a'}Q^{(1)}(S_{t+1},a'))-Q^{(1)}(S_t,A_t)\bigr]$$  
(and vice-versa swapping indices).  

**Key Principles**  
• Two independent estimators decouple action selection from evaluation.  

**Detailed Concept Analysis**  
• Reduces over-estimation error $\mathbb E[\max_a \hat Q - \max_a Q]$.  
• Empirically boosts performance in stochastic domains.  

**Importance**  
Foundation of Double DQN, pervasive in deep off-policy control.  

**Pros vs Cons**  
Pros: unbiased estimates, increased stability;  
Cons: double memory, mild under-estimation risk.  

**Cutting-Edge Advances**  
• Averaged DQNs and ensemble Q-learning generalize double learning;  
• Bootstrapped DQN leverages multiple heads for uncertainty.  

---

### 6.8 Games, Afterstates, and Other Special Cases  
**Definition**  
Domain-specific TD variants exploiting structure (e.g., deterministic moves, two-player games).  

**Pertinent Equations**  
Afterstate value: $$V^{\text{after}}(s')=\mathbb E\bigl[ G_t \mid S_{t+1}=s' \bigr]$$  

**Key Principles**  
• Afterstates collapse $(s,a)$ into post-decision state, halving branching factor.  
• Temporal-difference backups tailored to game plies.  

**Detailed Concept Analysis**  
• TD-Gammon applied TD($\lambda$) on afterstates of backgammon, achieving expert play.  
• For deterministic board games, TD learning combined with minimax search (AlphaZero).  

**Importance**  
Demonstrates TD’s potency when aligned with domain heuristics and search.  

**Pros vs Cons**  
Pros: reduced state dimensionality, faster learning;  
Cons: requires domain insight, may not generalize outside structured environments.  

**Cutting-Edge Advances**  
• AlphaGo/AlphaZero integrate TD with Monte-Carlo tree search;  
• MuZero predicts dynamics latent-state TD, obviating explicit environment model.