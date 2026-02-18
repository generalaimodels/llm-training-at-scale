# 2 Multi-armed Bandits  

## 2.1 K-armed Bandit Problem  
### Definition  
Sequential decision task with $K$ independent arms; pulling arm $k$ yields stochastic reward $R_t\!\sim\!P_k$ with unknown mean $\mu_k$. Objective: maximize expected cumulative reward or minimize regret.  
### Pertinent Equations  
• Expected return $$\mathbb E\left[\sum_{t=1}^{T}R_t\right]$$  
• Regret $$\mathcal R_T=\!T\mu^\star-\sum_{t=1}^{T}\mu_{A_t},\quad\mu^\star=\max_k\mu_k$$  
### Key Principles  
• Exploration vs. exploitation trade-off • Stochastic vs. adversarial reward models • Regret minimization metrics.  
### Detailed Concept Analysis  
• Bernoulli, Gaussian, heavy-tailed reward families.  
• Optimal policy unknown; algorithm updates estimate $\hat\mu_k$.  
• Horizon-aware vs. horizon-free strategies, e.g., Gittins index optimal for discounted reward.  
### Importance  
Forms theoretical backbone for online learning, A/B testing, ad allocation, adaptive routing.  
### Pros vs Cons  
+ Minimal assumptions, interpretable metrics.  
– Ignores state dynamics, limited to immediate reward.  
### Cutting-Edge Advances  
• Best-arm identification under fixed confidence.  
• Bandits with knapsacks, combinatorial pulls, causal bandits.  



## 2.2 Action-value Methods  
### Definition  
Algorithms estimating $\hat Q_t(k)\approx\mu_k$ and selecting $A_t=\arg\max_k \hat Q_t(k)$ mod exploration.  
### Pertinent Equations  
• Sample average $$\hat Q_t(k)=\frac{\sum_{i=1}^{t-1}\mathbb 1\{A_i=k\}R_i}{\sum_{i=1}^{t-1}\mathbb 1\{A_i=k\}}$$  
### Key Principles  
• Law of large numbers ensures $\hat Q_t(k)\!\to\!\mu_k$.  
• $\epsilon$-greedy exploration: pick random arm w.p. $\epsilon$.  
### Detailed Concept Analysis  
• Bias–variance interplay: sample mean unbiased but high variance early.  
• Importance sampling for non-uniform exploration.  
### Importance  
Baseline for more sophisticated bandit algorithms; simple, parameter-free.  
### Pros vs Cons  
+ Easy to implement, convergent.  
– Slow adaptation to non-stationarity, no uncertainty quantification.  
### Cutting-Edge Advances  
• Bootstrap-based uncertainty on $\hat Q_t(k)$ for efficient exploration.  



## 2.3 The 10-armed Testbed  
### Definition  
Benchmark with $K=10$, $\mu_k\!\sim\!\mathcal N(0,1)$, rewards $R_t\!\sim\!\mathcal N(\mu_k,1)$.  
### Pertinent Equations  
• Performance metric $$\text{AvgReturn}=\frac{1}{N}\sum_{n=1}^{N}\frac{1}{T}\sum_{t=1}^{T}R_t^{(n)}$$  
### Key Principles  
• Monte-Carlo averaging over $N$ runs removes outcome variance.  
### Detailed Concept Analysis  
• Highlights effects of $\epsilon$, UCB $c$, optimistic starts.  
• Provides qualitative curves: early exploration cost vs. long-term gain.  
### Importance  
Canonical empirical testbed popularized by Sutton & Barto; enables fair comparison.  
### Pros vs Cons  
+ Simple, reproducible.  
– Low dimension, stationary; limited realism.  
### Cutting-Edge Advances  
• Extensions: non-stationary, contextual, heavy-tailed variants for robustness studies.  



## 2.4 Incremental Implementation  
### Definition  
Online update eliminating need to store past rewards.  
### Pertinent Equations  
• Update rule $$\hat Q_{t+1}(k)=\hat Q_t(k)+\alpha_t(k)\bigl[R_t-\hat Q_t(k)\bigr]$$  
### Key Principles  
• Step-size $\alpha_t(k)=1/N_t(k)$ recovers sample average; constant $\alpha$ enables non-stationary tracking.  
### Detailed Concept Analysis  
• Computational $\mathcal O}(1)$ per step.  
• Stochastic approximation convergence if $\sum_t\alpha_t=\infty,\,\sum_t\alpha_t^2<\infty$.  
### Importance  
Crucial for real-time applications with limited memory.  
### Pros vs Cons  
+ Memory-efficient, anytime estimates.  
– Requires step-size tuning.  
### Cutting-Edge Advances  
• Adaptive step-sizes via meta-gradients minimizing squared TD-error.  



## 2.5 Tracking a Nonstationary Problem  
### Definition  
Reward means drift over time: $\mu_{k,t+1}=\mu_{k,t}+\zeta_{k,t},\;\zeta\!\sim\!\mathcal N(0,\sigma^2_\zeta)$.  
### Pertinent Equations  
• Exponential recency-weighted estimate $$\hat Q_{t+1}(k)=\hat Q_t(k)+\alpha\,(R_t-\hat Q_t(k))$$  
• Effective window $$T_{\text{eff}}=\tfrac{1}{\alpha}$$  
### Key Principles  
• Bias-variance trade-off controlled by $\alpha$.  
• Concept drift detection may trigger exploration reset.  
### Detailed Concept Analysis  
• Kalman filter view: estimate and drift variance jointly.  
• Bayesian change-point models provide posterior over $\mu_{k,t}$.  
### Importance  
Reflects real-world volatility in finance, recommender systems.  
### Pros vs Cons  
+ Maintains relevance under drift.  
– High variance; choosing $\alpha$ non-trivial.  
### Cutting-Edge Advances  
• Sliding-window UCB, discounted Thompson sampling.  
• Meta-RL agents learning optimal $\alpha$ schedule.  



## 2.6 Optimistic Initial Values  
### Definition  
Initialize $\hat Q_0(k)=Q_{\text{high}}\!>\!\max_k\mu_k$ to encourage exploration.  
### Pertinent Equations  
• Greedy selection saturates exploration until $$\hat Q_t(k)\downarrow\mu_k$$  
### Key Principles  
• Exploration emerges without stochastic choice; deterministic optimism in face of uncertainty.  
### Detailed Concept Analysis  
• Decay rate depends on reward variance and pull count.  
• Ineffective in non-stationary settings where optimism vanishes.  
### Importance  
Simple strategy eliminating $\epsilon$ parameter.  
### Pros vs Cons  
+ No randomness, easy to reason about.  
– Hyper-parameter $Q_{\text{high}}$ sensitive; diminished benefit in large horizon.  
### Cutting-Edge Advances  
• Dynamic optimism via prior-dependent value functions (Bayesian UCB).  



## 2.7 Upper-Confidence-Bound Action Selection  
### Definition  
Select arm maximizing upper confidence estimate.  
### Pertinent Equations  
• UCB1 $$A_t=\arg\max_k\Bigl[\hat\mu_k+\sqrt{\frac{2\ln t}{N_t(k)}}\Bigr]$$  
• Regret bound $$\mathcal R_T\le\!\sum_{k:\mu_k<\mu^\star}\frac{8\ln T}{\Delta_k}+O(1)$$ where $\Delta_k=\mu^\star-\mu_k$.  
### Key Principles  
• Optimism in the face of uncertainty based on Hoeffding bounds.  
### Detailed Concept Analysis  
• Exploration term shrinks $\propto\sqrt{\ln t/N_t(k)}$.  
• Extensions: UCB-Tuned, KL-UCB, Gaussian UCB.  
### Importance  
Theoretically near-optimal $\mathcal O}(\ln T)$ regret.  
### Pros vs Cons  
+ Parameter-free, strong guarantees.  
– Over-explores with heavy-tailed rewards; assumes sub-Gaussian noise.  
### Cutting-Edge Advances  
• Bayesian UCB with posterior quantiles.  
• NeuralUCB leveraging feature embeddings for contextual setting.  



## 2.8 Gradient Bandit Algorithms  
### Definition  
Policy-gradient methods optimize preference vector $H_t(k)$ producing softmax probabilities.  
### Pertinent Equations  
• Policy $$\pi_t(k)=\frac{e^{H_t(k)}}{\sum_{j}e^{H_t(j)}}$$  
• Update $$H_{t+1}(k)=H_t(k)+\alpha\,(R_t-\bar R_t)\bigl[\mathbb 1\{k=A_t\}-\pi_t(k)\bigr]$$  
### Key Principles  
• Uses baseline $\bar R_t$ to reduce variance; equivalent to SGD on expected reward.  
### Detailed Concept Analysis  
• Converges to optimal arm in stationary case; step-size controls stochasticity.  
• Connection to REINFORCE in RL.  
### Importance  
Extends naturally to large, differentiable action embeddings.  
### Pros vs Cons  
+ Handles continuous/parameterized actions.  
– Sensitive to $\alpha$; slower asymptotic convergence than UCB/TS.  
### Cutting-Edge Advances  
• Natural-gradient bandits, mirror descent with entropy regularization.  
• Softmax exploration combined with meta-learned temperature schedules.  



## 2.9 Associative Search (Contextual Bandits)  
### Definition  
Each round observes context $x_t\!\in\!\mathcal X$; reward distribution $\mu_k(x_t)$; aim to learn $\pi(a\mid x)$.  
### Pertinent Equations  
• Contextual regret $$\mathcal R_T=\sum_{t=1}^{T}\bigl[\mu_{a^\star(x_t)}(x_t)-\mu_{A_t}(x_t)\bigr]$$  
• LinUCB score $$\text{UCB}_k(x_t)=\theta_k^\top x_t+\alpha\sqrt{x_t^\top A_k^{-1}x_t}$$  
### Key Principles  
• Leverages side-information to reduce exploration.  
• Parametric (linear) vs. non-parametric (kernel, neural) reward models.  
### Detailed Concept Analysis  
• Thompson Sampling with Bayesian linear regression: posterior $p(\theta\mid\mathcal D_t)$ guides sampling.  
• Reduction to supervised learning via inverse-propensity weighting (IPS) for offline evaluation.  
### Importance  
Critical for recommendation, personalized medicine, adaptive tutoring.  
### Pros vs Cons  
+ Exploits rich features; lower regret than non-contextual.  
– Susceptible to covariate shift; computationally heavier.  
### Cutting-Edge Advances  
• Neural contextual bandits with representation learning.  
• Counterfactual bandits using causal inference to debias logging policies.  
• Large-language-model-based policy networks for text-rich contexts.  