## Chapter 1 Foundations of Probabilistic Generative Modeling  

### 1 Probability Spaces, Random Variables, and Likelihood  

**Definition**  
A probability space is the triple $$(\Omega,\mathcal F,P)$$ where  
• $\Omega$ – sample space,  
• $\mathcal F$ – $\sigma$-algebra on $\Omega$,  
• $P:\mathcal F\!\rightarrow[0,1]$ – probability measure with $P(\Omega)=1$.  

A (scalar or vector-valued) random variable is a measurable map $$X:\Omega\!\to\!(\mathcal X,\mathcal B_{\mathcal X})$$.  
For continuous $X$, the density $$p_X(x)=\frac{dP_X}{dx}$$ satisfies $\int_{\mathcal X}p_X(x)dx=1$.  
For discrete $X$, $$p_X(x)=P[X=x]$$.

**Joint, Marginal, Conditional**  
For $(X,Y)$ with joint density $p(x,y)$:  
$$p(x)=\int p(x,y)\,dy,\quad p(y|x)=\frac{p(x,y)}{p(x)}.$$

**Likelihood**  
Given i.i.d. observations $$\mathcal D=\{x_i\}_{i=1}^{N}$$ and model parameters $\theta\!\in\!\Theta$, the likelihood is  
$$\mathcal L(\theta;\mathcal D)=p(\mathcal D|\theta)=\prod_{i=1}^{N}p(x_i|\theta).$$  
Log-likelihood: $$\ell(\theta)=\sum_{i=1}^{N}\log p(x_i|\theta).$$  

---

### 2 Maximum-Likelihood Estimation (MLE)  

**Objective**  
$$\hat\theta_{\text{MLE}}=\arg\max_{\theta\in\Theta}\;\ell(\theta).$$  

**Optimization**  
1. Compute $\nabla_\theta \ell(\theta)=\sum_{i}\nabla_\theta\!\log p(x_i|\theta)$.  
2. Solve $\nabla_\theta \ell(\theta)=0$ via analytical root or numerical optimizer (e.g., Newton, L-BFGS, SGD).  

**Statistical Properties**  
• Consistency: $\hat\theta_{\text{MLE}}\xrightarrow{p}\theta^\star$ under regularity.  
• Asymptotic normality: $$\sqrt N(\hat\theta_{\text{MLE}}-\theta^\star)\xrightarrow{d}\mathcal N\!\bigl(0,\;I(\theta^\star)^{-1}\bigr),$$  
where the Fisher information $$I(\theta)=\mathbb E\!\bigl[-\nabla_{\!\theta}^2\!\log p(X|\theta)\bigr].$$  

---

### 3 Bayesian Inference  

**Prior and Posterior**  
Let prior $$p(\theta)$$. Observations update belief via Bayes’ rule  
$$p(\theta|\mathcal D)=\frac{p(\mathcal D|\theta)\,p(\theta)}{p(\mathcal D)},\qquad p(\mathcal D)=\int p(\mathcal D|\theta)p(\theta)\,d\theta.$$  

**Posterior Predictive**  
$$p(x_{\text{new}}|\mathcal D)=\int p(x_{\text{new}}|\theta)\,p(\theta|\mathcal D)\,d\theta.$$  

**Decision-Theoretic Optimality**  
Given loss $L(\theta,a)$, Bayes action $$a^\star=\arg\min_a\mathbb E_{p(\theta|\mathcal D)}[L(\theta,a)].$$  

---

### 4 Latent-Variable Models and Marginal Likelihood  

**Model Structure**  
Observed $x$, latent $z$: joint $$p(x,z|\theta)=p(z|\theta)\,p(x|z,\theta).$$  

**Marginal Likelihood**  
$$p(x|\theta)=\int p(x,z|\theta)\,dz.$$  
The intractability of this integral (high-dimensional $z$ or complex conditionals) motivates EM and variational methods.

---

### 5 Expectation–Maximization (EM) Algorithm  

**Goal** Maximize $$\log p(\mathcal D|\theta)=\sum_i\log\!\int p(x_i,z_i|\theta)\,dz_i.$$

#### 5.1 Derivation via Jensen’s Inequality  
Introduce any density $q(z_i)$:  
$$\log p(x_i|\theta)=\log\!\int q(z_i)\frac{p(x_i,z_i|\theta)}{q(z_i)}\,dz_i
\ge \int q(z_i)\log\!\frac{p(x_i,z_i|\theta)}{q(z_i)}\,dz_i.$$
Equality holds when $$q(z_i)=p(z_i|x_i,\theta).$$  

Define the complete-data log-likelihood $$\log p(x_i,z_i|\theta).$$  

#### 5.2 Iterative Procedure  
E-step (Expectation):  
$$Q(\theta|\theta^{(t)})=\sum_{i=1}^{N}\mathbb E_{z_i\sim p(z_i|x_i,\theta^{(t)})}\bigl[\log p(x_i,z_i|\theta)\bigr].$$  

M-step (Maximization):  
$$\theta^{(t+1)}=\arg\max_{\theta} Q(\theta|\theta^{(t)}).$$  

#### 5.3 Convergence  
EM monotonically increases the data log-likelihood:  
$$\log p(\mathcal D|\theta^{(t+1)})\ge \log p(\mathcal D|\theta^{(t)}).$$  
Limit points are stationary points of $\log p(\mathcal D|\theta)$.

---

### 6 Variational Inference (VI) Primer  

**6.1 Problem Statement**  
Target posterior $$p(z,\theta|\mathcal D)$$ intractable. Approximate with tractable family $$\mathcal Q=\{q_{\lambda}(z,\theta)\}.$$  

**6.2 Evidence Lower Bound (ELBO)**  
$$\begin{aligned}
\log p(\mathcal D) &= \log\!\int q_{\lambda}(z,\theta)\frac{p(\mathcal D,z,\theta)}{q_{\lambda}(z,\theta)}\,dz\,d\theta \\
&\ge \mathcal L(\lambda) = \mathbb E_{q_{\lambda}}\bigl[\log p(\mathcal D,z,\theta)-\log q_{\lambda}(z,\theta)\bigr].
\end{aligned}$$  
Gap is KL divergence:  
$$\log p(\mathcal D)-\mathcal L(\lambda)=\text{KL}\bigl(q_{\lambda}(z,\theta)\,\Vert\,p(z,\theta|\mathcal D)\bigr)\,\ge 0.$$

**6.3 Optimization Schemes**  

• Coordinate Ascent VI (CAVI)  
 If $q$ factorizes, update each factor by  
 $$\log q^\star_j(z_j)=\mathbb E_{q_{-j}}\![\log p(\mathcal D,z)] + \text{const}.$$  

• Stochastic VI / SVI  
 Use mini-batches and noisy gradients $$\nabla_\lambda \mathcal L(\lambda)\approx \frac{| \mathcal D |}{|\mathcal B|}\sum_{x_i\in\mathcal B}\nabla_\lambda\!\log p(x_i,z_i,\theta)-\nabla_\lambda\!\log q_\lambda.$$  

• Reparameterization Gradient  
 For differentiable latent $$z=g(\epsilon,\lambda),\; \epsilon\!\sim\!p(\epsilon)$$:  
 $$\nabla_\lambda\mathcal L(\lambda)=\mathbb E_{\epsilon}\bigl[\nabla_\lambda \log p(\mathcal D,g(\epsilon,\lambda))-\nabla_\lambda\log q_\lambda(g(\epsilon,\lambda))\bigr].$$  

**6.4 Choice of Variational Family**  
• Mean-field: $q(z,\theta)=\prod_j q_j(z_j)\prod_k q_k(\theta_k)$ – tractable, but ignores posterior correlations.  
• Structured: covariance or hierarchical couplings to capture dependencies.  
• Amortized: $q_\lambda(z|x)=\mathcal N\bigl(\mu_\lambda(x),\Sigma_\lambda(x)\bigr)$ learned by neural encoder (VAE setting).  

---

### 7 Interplay & Practical Guidance  

• Use MLE for large-data, low-dimensional $\theta$ when integrating over $\theta$ is unnecessary.  
• Bayesian inference preferred for uncertainty quantification; approximate via MCMC or VI.  
• Latent-variable models:  
 – Closed-form posteriors → EM (e.g., GMM).  
 – Non-conjugate → VI or Monte-Carlo EM.  
• Monitor ELBO or log-likelihood; exploit automatic differentiation for gradients.  

---

End of Chapter 1