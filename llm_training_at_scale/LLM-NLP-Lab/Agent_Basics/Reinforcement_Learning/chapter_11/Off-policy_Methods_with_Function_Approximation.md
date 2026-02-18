### 11.1 Semi-gradient Methods  
**Definition**  
Off-policy temporal-difference (TD) learning with a parametric value function $v_\theta(s)\!\approx\!v_\pi(s)$, where only the gradient w.r.t. the approximation is used, treating the bootstrapped target as a constant.

**Pertinent Equations**  
$$\delta_t = R_{t+1} + \gamma \, v_\theta(S_{t+1}) - v_\theta(S_t)$$  
$$\theta_{t+1} = \theta_t + \alpha \, \rho_t \, \delta_t \, \nabla_\theta v_\theta(S_t)$$  
$$\rho_t = \frac{\pi(A_t\!|\!S_t)}{b(A_t\!|\!S_t)} \quad (\text{importance sampling ratio})$$

**Key Principles**  
- Uses an off-policy behavior policy $b$ and target policy $\pi$.  
- Bootstrapped target introduces bias; ignoring $\nabla_\theta$ of the target removes computational burden but breaks true gradient property.  

**Detailed Concept Analysis**  
- Semi-gradient TD(0) converges for on-policy tabular case but **may diverge under function approximation + off-policy sampling**.  
- Extension to multi-step: replace $\delta_t$ with $n$-step or $\lambda$-return, each weighted by products of $\rho$.  

**Importance**  
- Computationally cheap, forms the backbone of early deep off-policy agents (e.g., DQN uses semi-gradient Q-learning).  

**Pros vs Cons**  
+ Simple, one-step update, linear complexity.  
− No true objective being minimized ⇒ instability and potential divergence.  

**Cutting-edge Advances**  
- Usage of target networks and replay buffers (DQN) to mitigate divergence.  
- Residual and distributional variants (QR-DQN, C51) remain semi-gradient in nature.

---

### 11.2 Examples of Off-policy Divergence  
**Definition**  
Empirical demonstrations where semi-gradient TD with linear function approximation fails to converge.

**Pertinent Equations**  
Canonical 2-state counterexample (Baird’s star):  
$$\theta_{t+1} = \theta_t + \alpha \, \delta_t \, \phi(S_t)$$  
with crafted feature matrix $\Phi$ s.t. $\Vert\theta_t\Vert \to \infty$.

**Key Principles**  
- Divergence arises when the projected Bellman operator is not a contraction under off-policy distribution.  

**Detailed Concept Analysis**  
- In Baird’s counterexample, seven states share six features ⇒ insufficient representational power.  
- Importance sampling ratios amplify certain directions, pushing $\theta$ outward indefinitely.  

**Importance**  
- Motivated research into stable gradient-based TD methods (GTDa/GTDb, TDC).  

**Pros vs Cons**  
+ Clarifies theoretical limits.  
− Demonstrates catastrophic failure modes in real systems.  

**Cutting-edge Advances**  
- Benchmark suites (e.g., Mountain-Car with noisy features) specifically constructed to test divergence guards.

---

### 11.3 The Deadly Triad  
**Definition**  
Three factors whose simultaneous presence can cause divergence:  
1. Function approximation  
2. Bootstrapping  
3. Off-policy learning

**Pertinent Equations**  
Projected Bellman operator:  
$$\Pi_d T_\pi v_\theta$$  
Non-contraction condition when sampling distribution $d_b \!\neq\! d_\pi$.

**Key Principles**  
- Removing any single element restores convergence guarantees.  

**Detailed Concept Analysis**  
- $T_\pi$ is a $\gamma$-contraction in max-norm, but $\Pi_d$ projection undermines this.  
- Off-policy introduces mismatch $d_b \to \Pi_{d_b}$, breaking fixed-point properties.  

**Importance**  
- Central conceptual lens for RL stability; guides algorithm design (e.g., actor-critic with compatible features).  

**Pros vs Cons**  
+ Highlights why divergence happens.  
− Does not itself provide a remedy.  

**Cutting-edge Advances**  
- Soft updates, target networks, and trust-region methods reduce effective bootstrapping strength.  
- Implicit regularization via dropout/weight decay shown to soften the triad’s impact in deep settings.

---

### 11.4 Linear Value-function Geometry  
**Definition**  
Analysis of TD learning when $v_\theta(s) = \theta^\top \phi(s)$ with $\phi(s)\in\mathbb{R}^k$.

**Pertinent Equations**  
Mean TD update:  
$$\mathbb{E}[\Delta\theta] = \alpha (A_\pi \theta + b_\pi)$$  
$$A_\pi = \Phi^\top D_b (I - \gamma P_\pi) \Phi$$  
$$b_\pi = \Phi^\top D_b r_\pi$$  

**Key Principles**  
- Convergence when $A_\pi$ is negative definite; off-policy can make $A_\pi$ indefinite.  

**Detailed Concept Analysis**  
- Eigen-structure of $A_\pi$ dictates learning dynamics; positive eigenvalues → divergence.  
- Optimal projected value vector:  
$$\theta^\ast = -A_\pi^{-1} b_\pi$$  
exists only when $A_\pi$ invertible and negative definite.  

**Importance**  
- Linear algebraic view enables precise characterisation and inspires gradient-TD algorithms.  

**Pros vs Cons**  
+ Analytically tractable; allows proofs.  
− Limited to linear features; deep nets break assumptions.  

**Cutting-edge Advances**  
- Use of random Fourier features and tile coding in combination with GTD to retain linear guarantees while scaling representational capacity.

---

### 11.5 Gradient Descent in the Bellman Error  
**Definition**  
Minimise mean-squared projected Bellman error (MSPBE) or mean-squared TD error (MSTDE).  

**Pertinent Equations**  
Bellman error: $$\mathcal{E}(\theta) = \Vert T_\pi v_\theta - v_\theta \Vert_{d_b}^2$$  
Gradient:  
$$\nabla_\theta \text{MSPBE} = 2 A_\pi^\top C^{-1} (A_\pi \theta + b_\pi)$$  
$C = \Phi^\top D_b \Phi$ (feature covariance).

**Key Principles**  
- Direct gradient needs double sampling (two independent next-states) → impractical.  

**Detailed Concept Analysis**  
- Biased single-sample estimator causes divergence.  
- Auxiliary variables (e.g., $w$) introduced to obtain unbiased linear-in-parameter form.  

**Importance**  
- Serves as theoretical target objective for stable methods.  

**Pros vs Cons**  
+ Yields principled optimisation.  
− Naïve implementation unattainable in MDPs without double models.  

**Cutting-edge Advances**  
- Stochastic meta-gradient techniques approximate second-order terms online.

---

### 11.6 The Bellman Error is Not Learnable  
**Definition**  
Statement that stochastic gradients of Bellman error require unattainable double samples, making direct minimisation infeasible.

**Pertinent Equations**  
Single-sample bias:  
$$\mathbb{E}[\delta_t^2] \neq \Vert T_\pi v_\theta - v_\theta \Vert_{d_b}^2$$  

**Key Principles**  
- Correlation between $R_{t+1}+\gamma v_\theta(S_{t+1})$ and $v_\theta(S_t)$ introduces bias.  

**Detailed Concept Analysis**  
- Double sampling would need two independent $S_{t+1}$ from same $S_t, A_t$.  
- Highlighting necessity for surrogate objectives (MSPBE) addressable via GTD.  

**Importance**  
- Clarifies why many naïve off-policy algorithms fail.  

**Pros vs Cons**  
+ Provides conceptual clarity.  
− Implies fundamental hardness without creative reformulation.  

**Cutting-edge Advances**  
- Model-based learners create virtual double samples via learned transition models (MBPO, Dreamer-V3).

---

### 11.7 Gradient-TD Methods  
**Definition**  
Family of algorithms performing true stochastic gradient descent on MSPBE using auxiliary weights.

**Pertinent Equations**  
GTD2 update:  
$$
\begin{aligned}
\delta_t &= R_{t+1} + \gamma \theta_t^\top \phi_{t+1} - \theta_t^\top \phi_t \\
w_{t+1} &= w_t + \beta (\delta_t - \phi_t^\top w_t) \phi_t \\
\theta_{t+1} &= \theta_t + \alpha \big( \phi_t - \gamma \phi_{t+1} \big) (\phi_t^\top w_t)
\end{aligned}
$$  

**Key Principles**  
- Two-time-scale stochastic approximation: $\beta \gg \alpha$.  
- Converges under off-policy sampling for linear function approximation.

**Detailed Concept Analysis**  
- $w$ estimates $C^{-1} (A_\pi \theta + b_\pi)$ online.  
- Proofs rely on ODE methods showing asymptotic stability.

**Importance**  
- First practical, provably convergent off-policy TD algorithms with function approximation.

**Pros vs Cons**  
+ Guaranteed convergence; unbiased gradient.  
− Extra memory ($w$), hyper-parameter tuning, slower learning.

**Cutting-edge Advances**  
- GTD variants extended to actor-critic (e.g., Off-Policy Actor-Critic, ACE).  
- DeepGTD applies target networks and Adam to scale with CNN features.

---

### 11.8 Emphatic-TD Methods  
**Definition**  
TD algorithms weighting updates by emphasis to correct distribution mismatch between $d_b$ and $d_\pi$.

**Pertinent Equations**  
Emphatic weight:  
$$F_t = i(S_t) + \gamma \, \rho_{t-1} \, F_{t-1}$$  
where $i(s)$ is interest.  
ETD(λ) update:  
$$\theta_{t+1} = \theta_t + \alpha \, \rho_t \, F_t \, \delta_t \, e_t$$  
Eligibility:  
$$e_t = \rho_t \big(\gamma \lambda e_{t-1} + M_t \phi(S_t)\big), \quad M_t = \lambda i(S_t) + (1-\lambda)F_t$$

**Key Principles**  
- Emphasis sequence re-weights to approximate $d_\pi$.  

**Detailed Concept Analysis**  
- Convergent for linear $\phi$ with arbitrary off-policy $\rho$.  
- Interest function allows prioritising states (e.g., start distribution).

**Importance**  
- Handles high-variance ratios without importance-sampling clipping, crucial for continuing tasks.  

**Pros vs Cons**  
+ Theoretically sound; adaptive weighting.  
− Additional recursion (computational overhead), can still exhibit high variance.

**Cutting-edge Advances**  
- Combination with variance-reduction (ETD-v2).  
- Deep Emphatic-TD integrates per-state interest using auxiliary networks.

---

### 11.9 Reducing Variance  
**Definition**  
Techniques to manage variance introduced by importance sampling in off-policy TD.

**Pertinent Equations**  
Clipped ratio: $$\bar\rho_t = \min(\bar c, \rho_t)$$  
Retrace $n$-step target:  
$$G_t^{\text{Retrace}} = v_\theta(S_t) + \sum_{k=0}^{\infty} (\gamma \lambda)^k \bigg( \prod_{i=1}^{k} c_{t+i} \bigg) \delta_{t+k}$$  
with $c_{t} = \lambda \min\big(1,\rho_t\big)$.

**Key Principles**  
- Trade-off bias and variance via clipping, truncation, or control variates.

**Detailed Concept Analysis**  
- Weighted importance sampling (WIS) normalises ratios:  
$$\hat v = \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}$$  
eliminates worst-case explosion.  
- Control variates (V-trace) subtract baseline to shrink variance while bounding bias.

**Importance**  
- Essential for deep off-policy actor-critic (IMPALA, R2D3, ReAct).  

**Pros vs Cons**  
+ Stabilises training; enables large replay buffers.  
− Introduces bias; hyper-parameters (clip values) environment dependent.

**Cutting-edge Advances**  
- Adaptive IS clipping via learned coefficients.  
- Doubly-robust estimators merging model-based predictions with IS for minimal variance.