## 1 Maximum-Likelihood Estimation (MLE)  

### 1.1 Definition  
Given observations $x_{1:n}$ i.i.d.\ from an unknown distribution within a parametric family $\{p_\theta(x)\}_{\theta\in\Theta}$, MLE seeks  
$$
\hat{\theta}_{\text{MLE}}\;=\;\arg\max_{\theta\in\Theta}\;\mathcal{L}(\theta;\,x_{1:n}),\qquad
\mathcal{L}(\theta;\,x_{1:n})=\prod_{i=1}^{n}p_\theta(x_i).
$$  
Equivalently maximize the log-likelihood  
$$
\ell(\theta)=\sum_{i=1}^{n}\log p_\theta(x_i).
$$  

### 1.2 First-Order Condition  
For differentiable $\ell$  
$$
\nabla_\theta\ell(\theta)=0,\qquad\widehat{\theta}_{\text{MLE}}\;\text{solves gradient root.}
$$  

### 1.3 Asymptotic Properties  
Assuming regularity:  
* Consistency: $\widehat{\theta}_{\text{MLE}}\xrightarrow{p}\theta^\star$.  
* Asymptotic normality:  
  $$
  \sqrt{n}\,(\widehat{\theta}_{\text{MLE}}-\theta^\star)\;\xrightarrow{d}\;\mathcal{N}\!\bigl(0,\mathcal{I}(\theta^\star)^{-1}\bigr),
  $$  
  where the Fisher information  
  $$
  \mathcal{I}(\theta)=\mathbb{E}_{x\sim p_\theta}\!\bigl[\nabla_\theta\log p_\theta(x)\,\nabla_\theta\log p_\theta(x)^{\!\top}\bigr].
  $$  
* Invariance: $g(\widehat{\theta}_{\text{MLE}})$ is MLE of $g(\theta)$.  

### 1.4 Optimization in High-Dimensional Generative Models  
* Minimize negative log-likelihood (NLL) with stochastic gradient descent:  
  $$
  \min_\theta \; \mathbb{E}_{x\sim\hat{p}}\!\bigl[-\log p_\theta(x)\bigr].
  $$  
* Distributed training: data parallelism, gradient accumulation, mixed precision.  
* Regularization: weight decay, dropout, early stopping.  

---

## 2 Bayesian Inference  

### 2.1 Definition  
Place prior $p(\theta)$; after observing $x_{1:n}$ compute posterior  
$$
p(\theta\mid x_{1:n})=\frac{\mathcal{L}(\theta;\,x_{1:n})\,p(\theta)}{\underbrace{\int_\Theta \mathcal{L}(\vartheta;\,x_{1:n})\,p(\vartheta)\,d\vartheta}_{p(x_{1:n})}}.
$$  

### 2.2 Posterior Predictive  
$$
p(x_\text{new}\mid x_{1:n})=\int_\Theta p_\theta(x_\text{new})\,p(\theta\mid x_{1:n})\,d\theta.
$$  

### 2.3 Point Estimators  
* MAP: $\hat{\theta}_{\text{MAP}}=\arg\max_\theta \bigl[\ell(\theta)+\log p(\theta)\bigr]$.  
* Bayesian decision theory: minimize posterior expected loss.  

### 2.4 Computation  
* Conjugate families: closed-form (e.g.\ Dirichlet–Multinomial).  
* Intractable posteriors: Markov Chain Monte Carlo, variational inference (Section 5).  

---

## 3 Latent Variables and Marginal Likelihood  

### 3.1 Generative Model Structure  
Introduce latent $z\in\mathcal{Z}$ with joint  
$$
p_\theta(x,z)=p_\theta(z)\,p_\theta(x\mid z).
$$  

### 3.2 Marginal Likelihood  
$$
p_\theta(x)=\int_{\mathcal{Z}} p_\theta(x,z)\,dz.
$$  
Integrating out $z$ yields the observable distribution; maximization of $\log p_\theta(x)$ trains models with hidden structure (e.g.\ VAEs, mixture models, HMMs, diffusion models with latent noise).  

### 3.3 Challenges  
* Integration often intractable for continuous high-dimensional $z$.  
* Requires approximate inference (EM, VI).  

---

## 4 Expectation–Maximization (EM) Algorithm  

### 4.1 Objective  
Maximize $\log p_\theta(x_{1:n})$ when $z_{1:n}$ are unobserved.  

### 4.2 Lower-Bound via Jensen  
For any distribution $q(z_{1:n})$  
$$
\log p_\theta(x_{1:n}) \;=\; 
\underbrace{\mathbb{E}_{q}\!\bigl[\log p_\theta(x_{1:n},z_{1:n})-\log q(z_{1:n})\bigr]}_{\mathcal{F}(q,\theta)} \;+\;
\mathbb{KL}\bigl[q(z_{1:n})\,\Vert\,p_\theta(z_{1:n}\mid x_{1:n})\bigr].
$$  
$\mathcal{F}$ is the evidence lower bound (ELBO). Setting $q$ to the true posterior maximizes $\mathcal{F}$.  

### 4.3 Iterative Scheme  
Initialize $\theta^{(0)}$. For $t=0,1,\dots$  

E-step:  
$$
q^{(t+1)}(z_{1:n}) := p_{\theta^{(t)}}(z_{1:n}\mid x_{1:n})
\quad\Longrightarrow\quad
Q(\theta\mid\theta^{(t)}) = \mathbb{E}_{z\sim q^{(t+1)}}\!\bigl[\log p_\theta(x_{1:n},z_{1:n})\bigr].
$$  

M-step:  
$$
\theta^{(t+1)} := \arg\max_{\theta} Q(\theta\mid\theta^{(t)}).
$$  

Guarantee: $\log p_{\theta^{(t+1)}}(x_{1:n})\;\ge\;\log p_{\theta^{(t)}}(x_{1:n})$.  

### 4.4 Examples  
* Gaussian mixture: closed-form updates for means, covariances, mixing coefficients.  
* HMM: Baum-Welch (forward–backward in E-step).  

### 4.5 Limitations  
* Requires exact posterior in E-step; infeasible in complex neural models.  
* Converges to local maxima; sensitive to initialization.  

---

## 5 Variational Inference (VI) Primer  

### 5.1 Problem Statement  
Approximate intractable posterior $p_\theta(z\mid x)$ by choosing a tractable family $\mathcal{Q}=\{q_\phi(z\mid x)\}$ and minimizing  
$$
\phi^\star=\arg\min_\phi\;\mathbb{KL}\bigl[q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\bigr].
$$  

### 5.2 Evidence Lower Bound (ELBO)  
$$
\mathcal{L}_{\text{ELBO}}(\theta,\phi) = 
\mathbb{E}_{q_\phi(z\mid x)}\!\bigl[\log p_\theta(x,z)-\log q_\phi(z\mid x)\bigr],
\qquad
\log p_\theta(x) \;\ge\; \mathcal{L}_{\text{ELBO}}(\theta,\phi).
$$  
Joint optimization  
$$
\max_{\theta,\phi}\;\mathcal{L}_{\text{ELBO}}.
$$  

### 5.3 Mean-Field Factorization  
Assume $q_\phi(z) = \prod_{k} q_{\phi_k}(z_k)$; update each factor by coordinate ascent (CAVI)  
$$
\log q_{\phi_k}(z_k) \propto \mathbb{E}_{j\neq k}\!\bigl[\log p_\theta(x,z)\bigr].
$$  

### 5.4 Stochastic Gradient VI (SGVI) for Continuous Latents  
Reparameterization trick (Kingma & Welling): for latent $z = g_\phi(\epsilon,x),\; \epsilon\sim p(\epsilon)$  
$$
\nabla_{\theta,\phi}\mathcal{L}_{\text{ELBO}} =
\mathbb{E}_{\epsilon}\!\bigl[\nabla_{\theta,\phi}\,
\bigl(\log p_\theta(x,g_\phi(\epsilon,x))-\log q_\phi(g_\phi(\epsilon,x)\mid x)\bigr)\bigr].
$$  
Enables low-variance Monte-Carlo gradients, used in VAEs, diffusion latent models.  

### 5.5 Advanced Extensions  
* Importance-Weighted ELBO: tighter bounds $\mathcal{L}_{\text{IWAE}}$.  
* Amortized inference: share $\phi$ across dataset via neural encoder $q_\phi(z\mid x)$.  
* Normalizing flows: increase expressiveness of $q_\phi$ with invertible transforms $f_\psi$.  
* Variational families on manifolds, Stein VI, SVGD.  

---

## 6 Inter-Relationship of Concepts  

1. MLE is a special case of Bayesian inference with a uniform prior or MAP with Dirac prior.  
2. Latent-variable models convert integration in $\log p_\theta(x)$ into an inference problem; EM provides deterministic coordinate ascent, VI provides stochastic, scalable alternatives.  
3. Modern generative frameworks (VAE, diffusion, autoregressive flows) are trained by maximizing exact or approximate ELBO/NLL; efficient implementation leverages large-scale optimization, parallel hardware, and advanced variational techniques.