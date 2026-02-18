## 7 n-step Bootstrapping  

### 7.1 n-step TD Prediction  

**Definition**  
Estimate $V_\pi(s)$ using the $n$-step return $G_t^{(n)}$ instead of the 1-step target.  

**Pertinent Equations**  
$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^{k}R_{t+k+1}+\gamma^{n}V_\pi(S_{t+n})$$  
$$V_\pi(S_t)\leftarrow V_\pi(S_t)+\alpha\bigl[G_t^{(n)}-V_\pi(S_t)\bigr]$$  
TD-error (accumulated): $$\delta_t^{(n)}=G_t^{(n)}-V_\pi(S_t)$$  

**Key Principles**  
• Interpolates between Monte-Carlo ($n\!=\!T-t$) and TD(0) ($n\!=\!1$).  
• Controls bias–variance via choice of $n$.  
• Update performed after $n$ steps (or online with eligibility traces).  

**Detailed Concept Analysis**  
• Variance decreases as $n$ shrinks; bias decreases as $n$ grows.  
• For linear function approximation, convergence proved when $n$ finite, $\alpha_t$ diminishes, and on-policy sampling.  
• Equivalence to TD($\lambda$) with $\lambda=\frac{1-\gamma^n}{1-\gamma^{n+1}}$ for geometric weighting.  

**Importance**  
Enables designer to tune sample-efficiency and stability. Foundation for multi-step Deep RL (e.g., A3C, IMPALA).  

**Pros vs Cons**  
Pros: better credit assignment for delayed rewards; faster propagation.  
Cons: storage of interim transitions; delayed updates increase latency.  

**Cut-ting-Edge Advances**  
• Retrace($\lambda$) uses truncated importance weights with multi-step targets.  
• GAE (generalized advantage estimation) applies exponentially-weighted $n$-step returns in policy-gradient methods.  

---  

### 7.2 n-step SARSA  

**Definition**  
On-policy control updating $Q_\pi(s,a)$ with $n$-step action-value return.  

**Pertinent Equations**  
$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^{k}R_{t+k+1}+\gamma^{n}Q_\pi(S_{t+n},A_{t+n})$$  
$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[G_t^{(n)}-Q(S_t,A_t)\bigr]$$  

**Key Principles**  
• Uses trajectory $(S_t,A_t,\dots,S_{t+n},A_{t+n})$.  
• Policy both generates and is evaluated by data (ε-greedy, softmax).  

**Detailed Concept Analysis**  
• Converges to $Q^\*$ when $n$ finite, exploration decays, and $\alpha_t$ diminishes.  
• With eligibility traces, yields SARSA($\lambda$) where $\lambda$ relates to $n$ as above.  

**Importance**  
Improves learning speed in sparse-reward tasks (e.g., Atari) versus 1-step.  

**Pros vs Cons**  
Pros: reduced variance compared to Monte-Carlo; faster reward propagation.  
Cons: still on-policy—sample inefficiency; sensitive to $n$ choice.  

**Cut-ting-Edge Advances**  
• Rainbow incorporates 3-step SARSA target;  
• N-step Actor-Critic extends to continuous actions via deterministic policy gradients.  

---  

### 7.3 n-step Off-policy Learning  

**Definition**  
Learn about target policy $\pi$ while following behavior policy $b$ using $n$-step importance sampling (IS).  

**Pertinent Equations**  
Cumulative IS ratio: $$\rho_{t:t+n-1}=\prod_{k=t}^{t+n-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$  
Off-policy return:  
$$G_t^{(n)}=\rho_{t:t+n-1}\Bigl[\sum_{k=0}^{n-1}\gamma^{k}R_{t+k+1}\Bigr]+\gamma^{n}\rho_{t:t+n-1}V_\pi(S_{t+n})$$  
Update: $$V(S_t)\leftarrow V(S_t)+\alpha\bigl[G_t^{(n)}-V(S_t)\bigr]$$  

**Key Principles**  
• IS corrects distribution mismatch; variance grows with $|\rho|$.  
• Truncation or renormalization required for stability.  

**Detailed Concept Analysis**  
• Expected squared error $\propto\mathbb E[\rho^2]$ grows exponentially with $n$ under stochastic $b$.  
• Weight capping/clipping introduces bias but bounded variance.  

**Importance**  
Allows experience replay, multi-task data sharing, and distributed RL.  

**Pros vs Cons**  
Pros: data reuse; decoupled exploration.  
Cons: high variance; possible divergence with function approximation.  

**Cut-ting-Edge Advances**  
• V-trace, IMPALA truncate $\rho$ to stabilize deep off-policy multi-step updates.  
• R-trace and AB-trace provide bias-corrected truncation schemes.  

---  

### 7.4 Per-decision Methods with Control Variates (*)  

**Definition**  
Reduce variance of off-policy multi-step targets by applying per-decision IS ratios multiplied by control variates.  

**Pertinent Equations**  
Per-decision weighting: $$\rho_k=\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$  
Return with control variate $c_{k}$:  
$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^{k}\Bigl(\rho_{t:k}R_{k+1}+ (1-\rho_{t:k})c_{k}\Bigr)+\gamma^{n}\rho_{t:t+n-1}V_\pi(S_{t+n})$$  

**Key Principles**  
• Control variate $c_k$ chosen to have zero mean wrt variance source; typical choice $c_k=V_\pi(S_k)$.  
• Each step corrects only the immediate mismatch, limiting weight explosion.  

**Detailed Concept Analysis**  
• Variance decomposed per timestep; expectation remains unbiased.  
• Implementation aligns with Emphatic TD and ordinary importance-weighted eligibility traces.  

**Importance**  
Enables stable, deep off-policy learning (e.g., ACER).  

**Pros vs Cons**  
Pros: dramatic variance reduction; unbiased under proper $c_k$.  
Cons: added computation; requires accurate $c_k$ estimate.  

**Cut-ting-Edge Advances**  
• Q-prop uses analytic control variates in continuous action policy gradients.  
• Stochastic weight averaging of control variates for adaptive variance control.  

---  

### 7.5 Off-policy Learning Without Importance Sampling: The n-step Tree-Backup Algorithm  

**Definition**  
Computes expected values under target policy at each step, eliminating IS ratios entirely.  

**Pertinent Equations**  
Recursive target:  
$$G_t^{\text{TB}} = R_{t+1} + \gamma \bigl[ \pi(A_{t+1}|S_{t+1})G_{t+1}^{\text{TB}} + \sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q(S_{t+1},a) \bigr]$$  
Compact $n$-step form:  
$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^{k}R_{t+k+1}+\gamma^{n}\sum_{a}\pi(a|S_{t+n})Q(S_{t+n},a)$$  

**Key Principles**  
• Backs up expected value over all actions at non-terminal step—“tree backup”.  
• Guarantees bounded updates irrespective of $n$.  

**Detailed Concept Analysis**  
• Equivalent to expectation of Double-Q update with averaging over $\pi$.  
• Computational cost grows with $|A|$, but parallelizable.  

**Importance**  
Underpins algorithms like Expected SARSA and V-trace (limiting variance).  

**Pros vs Cons**  
Pros: zero IS variance; stable with function approximation.  
Cons: biased when $Q$ inaccurate; expensive for large or continuous $A$.  

**Cut-ting-Edge Advances**  
• QT-Opt uses tree-backup-style Bellman operator with sampled-action expectation for robot grasping.  
• Soft-actor critic employs entropy-regularized tree backup in continuous spaces.  

---  

### 7.6 A Unifying Algorithm: n-step Q(σ) (*)  

**Definition**  
Generalizes SARSA ($\sigma\!=\!1$), Expected SARSA ($\sigma\!=\!0$), and Tree-Backup ($\sigma\!=\!0$ at inner nodes) via mixing parameter $\sigma\in[0,1]$.  

**Pertinent Equations**  
Mixed target at step $k$:  
$$\hat G_k = \sigma_k \bigl[R_{k+1}+\gamma Q(S_{k+1},A_{k+1})\bigr]+ (1-\sigma_k)\bigl[R_{k+1}+\gamma \sum_{a}\pi(a|S_{k+1})Q(S_{k+1},a)\bigr]$$  
n-step return: $$G_t^{(n)}=\hat G_t + \gamma \hat G_{t+1}+ \dots + \gamma^{n-1} \hat G_{t+n-1} + \gamma^{n}\sum_{a}\pi(a|S_{t+n})Q(S_{t+n},a)$$  
Update: $$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[G_t^{(n)}-Q(S_t,A_t)\bigr]$$  

**Key Principles**  
• $\sigma_k$ can vary per step or be annealed during learning.  
• Provides continuous spectrum between sampling (high variance, low bias) and expectation (low variance, high bias).  

**Detailed Concept Analysis**  
• Optimal $\sigma$ depends on reward stochasticity and function approximator.  
• Adaptive schemes set $\sigma_k$ proportional to TD-error magnitude or uncertainty.  

**Importance**  
Gives practitioners unified interface to tune bias–variance without algorithm switch.  

**Pros vs Cons**  
Pros: flexible; can outperform fixed methods; facilitates curriculum from exploratory to greedy updates.  
Cons: extra hyper-parameter; scheduler complexity.  

**Cut-ting-Edge Advances**  
• DeepMind’s R2D2 anneals $\sigma$ with prioritized replay.  
• Ensemble Q(σ) uses Bayesian uncertainty to set $\sigma$ per head, boosting exploration-exploitation trade-off.