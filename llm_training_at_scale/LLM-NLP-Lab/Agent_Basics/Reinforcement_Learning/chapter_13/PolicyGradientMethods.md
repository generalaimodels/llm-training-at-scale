### 13 Policy Gradient Methods  

---

#### 13.1 Policy Approximation and Its Advantages  

**Definition**  
A differentiable parametric family of policies $\pi_{\theta}(a|s)$, where $\theta\in\mathbb{R}^d$, used to directly optimize expected return without constructing an explicit value table.

**Pertinent Equations**  
$$\pi_{\theta}(a|s)=\text{softmax}\left( f_{\theta}(s,a)\right),\qquad J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\big[R(\tau)\big]$$  

**Key Principles**  
- Function approximation (NNs, linear, RBF) for high‐dimensional spaces.  
- Stochasticity for exploration.  
- Direct differentiation of $J(\theta)$ to update $\theta$.

**Detailed Concept Analysis**  
- Eliminates need for $\epsilon$‐greedy; exploration implicit in $\pi_{\theta}$.  
- Scales to continuous action via e.g. Gaussian $\pi_{\theta}(a|s)=\mathcal{N}(\mu_{\theta}(s),\Sigma_{\theta}(s))$.  

**Importance**  
- Handles large/continuous action spaces where value‐based methods struggle.  

**Pros vs Cons**  
+ Continuous actions, smooth updates, on‐policy consistency.  
− High variance, potentially slow convergence, sensitivity to step size.

**Cutting-Edge Advances**  
- Normalizing flow policies; implicit distributions; transformer‐based policies enabling long‐horizon memory.

---

#### 13.2 The Policy Gradient Theorem  

**Definition**  
Provides an exact expression for $\nabla_{\theta}J(\theta)$ requiring only policy and action‐value estimates.

**Pertinent Equations**  
$$\nabla_{\theta}J(\theta)=\mathbb{E}_{s\sim d^{\pi},\,a\sim\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a|s)\,Q^{\pi}(s,a)\big]$$  

**Key Principles**  
- $d^{\pi}(s)$: discounted state distribution.  
- Uses log‐derivative trick: $\nabla_{\theta}\pi=\pi\nabla_{\theta}\log\pi$.  

**Detailed Concept Analysis**  
- Separates dynamics from optimization; model‐free.  
- $Q^{\pi}$ may be Monte Carlo, critic, or bootstrapped estimate.

**Importance**  
Foundation behind REINFORCE, actor–critic, PPO, TRPO, SAC.

**Pros vs Cons**  
+ Exact gradient in expectation; unbiased.  
− Requires large sample size; $Q^{\pi}$ estimation quality critical.

**Cutting-Edge Advances**  
- Compatibility conditions for natural gradients; variance‐reduced estimators (e.g., SVRG‐PG).

---

#### 13.3 REINFORCE: Monte Carlo Policy Gradient  

**Definition**  
Stochastic gradient ascent using full‐trajectory returns as $Q^{\pi}$.

**Pertinent Equations**  
$$\theta_{t+1}=\theta_t+\alpha\,G_t\,\nabla_{\theta}\log\pi_{\theta}(A_t|S_t),\qquad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$$  

**Key Principles**  
- Episodic rollouts; unbiased but high‐variance estimates.  
- No bootstrapping; delayed updates at episode end.

**Detailed Concept Analysis**  
- Sampling efficiency poor; variance $\propto$ episode length.  
- Learning rate tuning crucial.

**Importance**  
Historical baseline; conceptual simplicity; basis for modern variants.

**Pros vs Cons**  
+ Unbiased, conceptually clean, trivial implementation.  
− Very high variance, slow in long‐horizon tasks.

**Cutting-Edge Advances**  
- Control variates, importance‐weighted REINFORCE for off‐policy correction.

---

#### 13.4 REINFORCE with Baseline  

**Definition**  
Adds baseline $b(s)$ to reduce variance without bias.

**Pertinent Equations**  
$$\nabla_{\theta}J(\theta)=\mathbb{E}\!\big[(G_t-b(S_t))\nabla_{\theta}\log\pi_{\theta}(A_t|S_t)\big]$$  

**Key Principles**  
- Optimal $b(s)=\mathbb{E}[G_t|S_t=s]$ minimizes variance.  
- Frequently $b(s)\approx V^{\pi}(s)$ learned via regression.

**Detailed Concept Analysis**  
- Variance reduction $\approx\mathrm{Var}(G_t)-\mathrm{Var}(G_t-b)$.  
- Enables larger step sizes.

**Importance**  
Bridge to actor–critic by learning $b(s)$ as critic.

**Pros vs Cons**  
+ Significant variance savings; retains unbiasedness.  
− Additional critic network; extra hyper‐parameters.

**Cutting-Edge Advances**  
- Generalized Advantage Estimator (GAE): $$\hat{A}^{\text{GAE}}_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$ achieving bias–variance trade-off.

---

#### 13.5 Actor–Critic Methods  

**Definition**  
Dual‐network architecture: actor $\pi_{\theta}$ and critic $V_{w}$ (or $Q_{w}$) updated jointly.

**Pertinent Equations**  
Actor update: $$\theta\leftarrow\theta+\alpha\,\hat{A}_t\,\nabla_{\theta}\log\pi_{\theta}(A_t|S_t)$$  
Critic update (TD learning): $$w\leftarrow w+\beta\,\delta_t\,\nabla_{w}V_{w}(S_t),\quad\delta_t=R_{t+1}+\gamma V_{w}(S_{t+1})-V_{w}(S_t)$$  

**Key Principles**  
- Temporal Difference bootstrapping for critic.  
- Online, incremental; supports continuing tasks.

**Detailed Concept Analysis**  
- Bias introduced by TD compensated by variance reduction.  
- Stability hinges on compatible function approximation and step‐size ratio $\beta\gg\alpha$.

**Importance**  
Foundation for A2C, A3C, DDPG, TD3, SAC.

**Pros vs Cons**  
+ Lower variance vs pure MC; online updates; scalable.  
− Bias from critic approximation; potential actor–critic oscillations.

**Cutting-Edge Advances**  
- Entropy‐regularized actor–critic (SAC) optimizing $\max J(\theta)+\alpha_\text{ent}\mathbb{E}[\mathcal{H}(\pi_{\theta}(\cdot|s))]$.  
- Distributed actor–critic (IMPALA, R2D2) with off‐policy corrections (V‐trace).

---

#### 13.6 Policy Gradient for Continuing Problems  

**Definition**  
Extends policy gradients to infinite‐horizon tasks without episodes using average‐reward or discounted formulations.

**Pertinent Equations**  
Average‐reward objective: $$\rho(\theta)=\lim_{T\to\infty}\frac1T\mathbb{E}\Big[\sum_{t=0}^{T-1}R_{t+1}\Big]$$  
Gradient (average‐reward): $$\nabla_{\theta}\rho(\theta)=\mathbb{E}_{s\sim d^{\pi}}\!\big[\!\!\sum_a\nabla_{\theta}\pi_{\theta}(a|s)Q^{\pi}(s,a)\big]$$  

**Key Principles**  
- Stationary state distribution $d^{\pi}$; ergodicity required.  
- Differential value functions solve Poisson equation for bias removal.

**Detailed Concept Analysis**  
- Typically recast with discount $\gamma\lesssim1$ for practicality.  
- Average‐reward methods (e.g., RVI actor–critic) subtract running reward baseline.

**Importance**  
Necessary for robotics/control tasks lacking natural termination.

**Pros vs Cons**  
+ Aligns with perpetual real‐time operation.  
− Estimating average reward $\rho$ introduces extra variance/bias.

**Cutting-Edge Advances**  
- Bias‐corrected estimators (Emphatic PG); adaptive $\gamma$ annealing.

---

#### 13.7 Policy Parameterization for Continuous Actions  

**Definition**  
Choose differentiable distributions over $\mathcal{A}=\mathbb{R}^n$ enabling analytic $\nabla_{\theta}\log\pi$.

**Pertinent Equations**  
Gaussian policy: $$\pi_{\theta}(a|s)=\mathcal{N}\big(a\,;\,\mu_{\theta}(s),\text{diag}(\sigma_{\theta}^2(s))\big),\quad \nabla_{\theta}\log\pi=\frac{(a-\mu)}{\sigma^2}\nabla_{\theta}\mu+\Big[-1+\frac{(a-\mu)^2}{\sigma^2}\Big]\nabla_{\theta}\log\sigma$$  

**Key Principles**  
- Reparameterization trick: $a=\mu_{\theta}(s)+\sigma_{\theta}(s)\odot\epsilon,\;\epsilon\sim\mathcal{N}(0,I)$.  
- Squashing (tanh) to respect bounds.

**Detailed Concept Analysis**  
- Covariance can be full, diagonal, or low‐rank; trade complexity vs stability.  
- Entropy regularization mitigates premature variance collapse.

**Importance**  
Enables high‐precision continuous control (MuJoCo, robotics).

**Pros vs Cons**  
+ Smooth gradients; natural exploration.  
− Covariance explosion in high‐dimensional spaces; sensitivity to scaling.

**Cutting-Edge Advances**  
- Implicit policy gradients via score matching; normalizing‐flows for expressive $\pi_{\theta}$ (e.g., Tanh‐squashed RealNVP).  

---