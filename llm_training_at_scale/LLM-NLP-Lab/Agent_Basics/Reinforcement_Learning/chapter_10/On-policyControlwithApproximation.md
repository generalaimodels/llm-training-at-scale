## 10 On-policy Control with Approximation  

### 10.1 Episodic Semi-gradient Control  

**Definition**  
On-policy control method that updates an approximate action-value function $\hat{q}(s,a,\mathbf{w})$ after complete episodes using semi-gradient descent toward the episodic return.

**Pertinent Equations**  
$$G_t = \sum_{k=t}^{T-1} \gamma^{\,k-t} R_{k+1}$$  
$$\delta_t = G_t - \hat{q}(S_t,A_t,\mathbf{w}_t)$$  
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \,\delta_t \,\nabla_{\mathbf{w}} \hat{q}(S_t,A_t,\mathbf{w}_t)$$  

**Key Principles**  
• Monte-Carlo target $G_t$ is unbiased but high-variance.  
• Semi-gradient: gradient computed wrt $\hat{q}$ only; dependence of $G_t$ on $\mathbf{w}$ is ignored.  
• $\epsilon$-greedy policy ensures on-policy exploration while improving greediness w.r.t. $\hat{q}$.  

**Detailed Concept Analysis**  
• Episodic returns enable proper credit assignment in finite-horizon tasks.  
• Weight vector $\mathbf{w}$ adjusted once per state-action pair per episode, reducing update frequency vs. TD.  
• Function approximators: linear $\hat{q}(s,a,\mathbf{w})=\mathbf{w}^\top \mathbf{x}(s,a)$ or non-linear networks.  

**Importance**  
• Bridges Monte-Carlo learning and function approximation.  
• Foundational for policy-based methods relying on episodic returns (REINFORCE).  

**Pros vs Cons**  
+ Simple, unbiased targets, easy to parallelize.  
− High variance, slow convergence, unsuitable for continuing tasks.  

**Cutting-Edge Advances**  
• Bootstrap return blending to lower variance: $G^\lambda$.  
• Episodic backward replay buffers accelerate convergence with off-policy corrections.  


### 10.2 Semi-gradient $n$-step Sarsa  

**Definition**  
Temporal-difference control method using an $n$-step return while ignoring the gradient w.r.t. bootstrapped portions.

**Pertinent Equations**  
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^{\,k} R_{t+k+1} + \gamma^{\,n}\hat{q}(S_{t+n},A_{t+n},\mathbf{w}_t)$$  
$$\delta_t^{(n)} = G_t^{(n)} - \hat{q}(S_t,A_t,\mathbf{w}_t)$$  
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \,\delta_t^{(n)} \,\nabla_{\mathbf{w}} \hat{q}(S_t,A_t,\mathbf{w}_t)$$  

**Key Principles**  
• Interpolates between 1-step Sarsa ($n=1$) and episodic Monte-Carlo ($n=T$).  
• On-policy requirement via $\epsilon$-greedy behavior.  
• Eligibility-trace formulation yields Sarsa($\lambda$).  

**Detailed Concept Analysis**  
• Bias-variance trade-off controlled by $n$ (or $\lambda$).  
• Eligibility vector $\mathbf{e}_t = \gamma\lambda\mathbf{e}_{t-1} + \nabla_{\mathbf{w}}\hat{q}(S_t,A_t,\mathbf{w}_t)$ simplifies $n$-step accumulation.  

**Importance**  
• Enables scalable control in large state spaces with continuous operation.  
• Basis for Deep RL algorithms (e.g., Deep Sarsa, Rainbow’s multi-step return).  

**Pros vs Cons**  
+ Lower variance than MC, faster learning.  
− Requires careful $n$ or $\lambda$ tuning; still on-policy so sample-inefficient vs. off-policy Q-learning.  

**Cutting-Edge Advances**  
• Meta-gradient tuning of $n$/$\lambda$.  
• Distributional $n$-step returns integrated with quantile critics.  


### 10.3 Average Reward: A New Problem Setting for Continuing Tasks  

**Definition**  
Formulation optimizing the steady-state average reward $\rho$ instead of discounted cumulative reward.

**Pertinent Equations**  
$$\rho_{\pi} = \lim_{T\to\infty}\frac{1}{T}\mathbb{E}_{\pi}\Bigl[\sum_{t=1}^{T} R_t\Bigr]$$  
$$h_{\pi}(s) = \mathbb{E}_{\pi}\Bigl[\sum_{k=0}^{\infty} \bigl(R_{t+k+1}-\rho_{\pi}\bigr)\,\big|\,S_t=s\Bigr]$$  

**Key Principles**  
• Removes dependence on $\gamma$, avoiding arbitrary time-scale settings.  
• Differential value function $h_{\pi}(s)$ replaces $v_{\pi}(s)$.  

**Detailed Concept Analysis**  
• Average-reward optimality criterion aligns with ergodic Markov chains.  
• Stationary distribution $d_{\pi}(s)$ determines weighting in performance.  
• Policy improvement: maximize $\rho$ via $h_{\pi}(s)$.  

**Importance**  
• Critical for control in continuous processes (telecom, robotics) where discounting distorts objectives.  

**Pros vs Cons**  
+ Time-scale invariant, principled for continuing domains.  
− Implementation harder: must estimate $\rho$ online; theoretical guarantees weaker under function approximation.  

**Cutting-Edge Advances**  
• Actor-critic algorithms with average-reward baselines (e.g., A3C-average).  
• Connection to entropy-regularized RL via soft-average reward.  


### 10.4 Deprecating the Discounted Setting  

**Definition**  
Argument and methodologies advocating replacement of discounted returns with average-reward or finite-horizon objectives.

**Pertinent Equations**  
Relationship: $$v_{\gamma}(s) = h(s) + \frac{\rho}{1-\gamma}$$  

**Key Principles**  
• Discount factor $\gamma$ complicates hyper-parameter search and ethical reward scaling.  
• For $\gamma\uparrow 1$, learning becomes unstable; near-singularity in Bellman operator.  

**Detailed Concept Analysis**  
• Bias introduced by $\gamma<1$ shrinks long-term rewards, conflicting with certain domains (energy grids).  
• Average-reward algorithms circumvent contraction issues by focusing on steady-state values.  

**Importance**  
• Influences design of future RL benchmarks and evaluation metrics.  

**Pros vs Cons**  
+ Eliminates arbitrary temporal discounting; clearer interpretability.  
− Tooling ecosystem, theoretical foundations & convergence proofs richer for discounted case; migration costs.  

**Cutting-Edge Advances**  
• Discount-free policy gradients using Poisson equation solutions.  
• Blackout-time transformations permitting off-policy evaluation without $\gamma$.  


### 10.5 Differential Semi-gradient $n$-step Sarsa  

**Definition**  
Average-reward counterpart of semi-gradient $n$-step Sarsa updating both average reward estimate $\hat{\rho}$ and differential action-value function $\hat{q}$.

**Pertinent Equations**  
$$G_t^{(n)} = \sum_{k=0}^{n-1} \bigl(R_{t+k+1}-\hat{\rho}_t\bigr) + \hat{q}(S_{t+n},A_{t+n},\mathbf{w}_t)$$  
$$\delta_t^{(n)} = G_t^{(n)} - \hat{q}(S_t,A_t,\mathbf{w}_t)$$  
$$\hat{\rho}_{t+1} = \hat{\rho}_t + \beta \,\delta_t^{(n)}$$  
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \,\delta_t^{(n)} \,\nabla_{\mathbf{w}}\hat{q}(S_t,A_t,\mathbf{w}_t)$$  

**Key Principles**  
• Simultaneous estimation of $\rho$ and differential values.  
• On-policy sampling with eligibility traces in differential form.  

**Detailed Concept Analysis**  
• $\beta$ step-size governs average reward tracking; often $\beta \ll \alpha$.  
• Convergence proven for linear function approximation under ergodicity.  

**Importance**  
• Enables scalable average-reward control in high-dimensional spaces (autonomous driving).  

**Pros vs Cons**  
+ Discount-free; supports continuing tasks.  
− Extra hyper-parameter $\beta$; slower reward-rate adaptation under non-stationarity.  

**Cutting-Edge Advances**  
• Differential off-policy corrections via Emphatic traces.  
• Deep differential critics leveraging continual landscape adaptation.