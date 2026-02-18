## 1 Probability Spaces

### 1.1 Definition  
A probability space is an ordered triple  
$$
\bigl(\Omega,\;\mathcal{F},\; \mathbb{P}\bigr)
$$  
where  

* $\Omega$ — **sample space**: the set of all elementary outcomes.  
* $\mathcal{F}$ — **$\sigma$-algebra** on $\Omega$: a non-empty collection of subsets of $\Omega$ that is closed under complementation and countable unions, i.e.  
  $$
  A\in\mathcal{F}\;\Longrightarrow\;A^c\in\mathcal{F},\qquad
  \{A_k\}_{k=1}^\infty\subset\mathcal{F}\;\Longrightarrow\;\bigcup_{k=1}^\infty A_k\in\mathcal{F}.
  $$  
* $\mathbb{P}$ — **probability measure**: a countably additive set function  
  $$
  \mathbb{P}:\mathcal{F}\to[0,1],\qquad
  \mathbb{P}(\Omega)=1,\qquad
  \forall \{A_k\}\subset\mathcal{F},\;A_i\cap A_j=\varnothing\;(i\neq j)\;\Longrightarrow\;
  \mathbb{P}\!\Bigl(\bigcup_{k=1}^\infty A_k\Bigr)=\sum_{k=1}^\infty\mathbb{P}(A_k).
  $$

### 1.2 Measurability and Dominating Measure  
For continuous distributions it is convenient to assume $\mathbb{P}$ is absolutely continuous w.r.t. a dominating measure $\mu$ (e.g.\ Lebesgue). By the Radon–Nikodým theorem there exists a density  
$$
p:\Omega\to\mathbb{R}_{\ge0},\qquad
\mathbb{P}(A)=\int_A p(\omega)\,d\mu(\omega).
$$

---

## 2 Random Variables

### 2.1 Definition  
A random variable (RV) is a measurable mapping  
$$
X:\bigl(\Omega,\mathcal{F}\bigr)\;\longrightarrow\;\bigl(\mathcal{X},\mathcal{B}(\mathcal{X})\bigr),
$$  
where $\mathcal{B}(\mathcal{X})$ denotes the Borel $\sigma$-algebra on the state space $\mathcal{X}$.

### 2.2 Distribution, CDF, PMF, PDF  
* **Distribution** of $X$ is the push-forward measure  
  $$
  \mathbb{P}_X(B)=\mathbb{P}\bigl(X^{-1}(B)\bigr),\qquad B\in\mathcal{B}(\mathcal{X}).
  $$  
* **Cumulative distribution function (CDF)**  
  $$
  F_X(x)=\mathbb{P}(X\le x).
  $$  
* **Probability mass function (PMF)** for discrete $\mathcal{X}$  
  $$
  p_X(x)=\mathbb{P}(X=x),\qquad \sum_{x\in\mathcal{X}}p_X(x)=1.
  $$  
* **Probability density function (PDF)** for continuous $\mathcal{X}$ w.r.t.\ Lebesgue measure  
  $$
  p_X(x)=\frac{d\mathbb{P}_X}{d\lambda}(x),\qquad \int_{\mathcal{X}}p_X(x)\,dx=1.
  $$

### 2.3 Joint, Marginal, Conditional  
Given $(X,Y)$ with joint density $p_{X,Y}(x,y)$:  
$$
p_X(x)=\int p_{X,Y}(x,y)\,dy,\qquad
p_{Y|X}(y|x)=\frac{p_{X,Y}(x,y)}{p_X(x)}.
$$

### 2.4 Expectations and Moments  
For any integrable $g:\mathcal{X}\to\mathbb{R}$  
$$
\mathbb{E}[g(X)] = \int_{\mathcal{X}} g(x)\,p_X(x)\,d\mu(x).
$$  
Variance, covariance, higher-order cumulants, characteristic and moment-generating functions are obtained by selecting appropriate $g$.

---

## 3 Likelihood $$p(x\mid\theta)$$

### 3.1 Generative Model Specification  
Assume a parametric family $\{p_\theta\}_{\theta\in\Theta}$ on data space $\mathcal{X}$.  
* $\theta$ — unknown parameters (weights of a neural generative model, HMM transition probs, etc.).  
* $x$ — single observation or dataset $x_{1:n}$.  

A **probabilistic generative model** defines a stochastic procedure  
$$
\theta\sim p(\theta)\quad(\text{optional hierarchical prior}),
\qquad x\sim p_\theta(x).
$$

### 3.2 Likelihood Function  
Given observed $x$, the **likelihood** as a function of $\theta$ is  
$$
\mathcal{L}(\theta\,;\,x)=p_\theta(x),\qquad
\theta\mapsto\mathcal{L}(\theta\,;\,x).
$$  
For i.i.d.\ dataset $x_{1:n}$  
$$
\mathcal{L}(\theta\,;\,x_{1:n}) = \prod_{i=1}^{n} p_\theta\bigl(x_i\bigr),
\qquad
\ell(\theta) := \log\mathcal{L}(\theta)=\sum_{i=1}^{n}\log p_\theta(x_i).
$$

### 3.3 Maximum Likelihood Estimation (MLE)  
$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta\in\Theta}\; \ell(\theta).
$$  
Necessary condition (for differentiable $\ell$):  
$$
\nabla_\theta \ell(\theta)=0,\qquad
\nabla^2_\theta \ell(\theta)\;\text{negative-definite}.
$$

### 3.4 Fisher Information  
For regular models, the expected curvature of $\ell$ at the true $\theta^\star$ is  
$$
\mathcal{I}(\theta^\star)=\mathbb{E}_{x\sim p_{\theta^\star}}\!\Bigl[
\bigl(\nabla_\theta\log p_\theta(x)\bigr)\bigl(\nabla_\theta\log p_\theta(x)\bigr)^{\!\top}
\Bigr]_{\theta=\theta^\star}.
$$  
Cramér–Rao lower bound: any unbiased estimator $\tilde{\theta}$ satisfies  
$$
\mathrm{Cov}(\tilde{\theta}) \succeq \mathcal{I}(\theta^\star)^{-1}.
$$

### 3.5 Bayesian Perspective  
With prior $p(\theta)$ the posterior is  
$$
p(\theta\mid x_{1:n})=\frac{\mathcal{L}(\theta\,;\,x_{1:n})\,p(\theta)}{\int_\Theta \mathcal{L}(\vartheta\,;\,x_{1:n})\,p(\vartheta)\,d\vartheta}.
$$  
MAP estimator  
$$
\hat{\theta}_{\text{MAP}}=\arg\max_{\theta}\;\bigl[\ell(\theta)+\log p(\theta)\bigr].
$$

---

## 4 Connection to Probabilistic Generative Modeling

1. **Modeling Assumption**: Data are realized from an unknown distribution within a parameterized family $p_\theta(x)$.  
2. **Training Objective**: Estimate $\theta$ by maximizing likelihood (equivalently minimizing $\mathbb{KL}[ \hat{p}(x)\;\|\;p_\theta(x) ]$ where $\hat{p}$ is empirical).  
   $$
   \min_\theta\;\mathbb{E}_{x\sim\hat{p}}\!\bigl[-\log p_\theta(x)\bigr] = \mathbb{KL}\bigl[\hat{p}\,\Vert\,p_\theta\bigr] + H(\hat{p})
   $$  
   (entropy term $H(\hat{p})$ constant wrt $\theta$).  
3. **Inference**: Generate new samples by ancestral sampling, rejection sampling, MCMC, or latent-variable sampling depending on model class (e.g.\ VAEs, normalizing flows, diffusion models).  
4. **Evaluation Metrics**: NLL, bits-per-dim, FID (for images), precision/recall on generated vs real.  

---

## 5 Worked Example

Consider a univariate Gaussian with unknown mean $\mu$ and known variance $\sigma^2$:  

* Model:  
  $$
  p_\mu(x)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^{2}}\Bigr).
  $$  
* Dataset: $x_{1:n}$.  
* Log-likelihood:  
  $$
  \ell(\mu)= -\frac{n}{2}\log(2\pi\sigma^{2}) - \frac{1}{2\sigma^{2}}\sum_{i=1}^{n}(x_i-\mu)^2.
  $$  
* MLE:  
  $$
  \frac{d\ell}{d\mu}= \frac{1}{\sigma^{2}}\sum_{i=1}^{n}(x_i-\mu)=0
  \;\;\Longrightarrow\;\;
  \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}x_i.
  $$  
* Fisher information:  
  $$
  \mathcal{I}(\mu)=\frac{1}{\sigma^{2}}n.
  $$  
* CRLB: $\mathrm{Var}(\hat{\mu}_{\text{unbiased}})\ge\sigma^{2}/n$; sample mean attains equality ⇒ efficient.

---

## 6 Summary of Key Takeaways

* A probability space $(\Omega,\mathcal{F},\mathbb{P})$ provides the formal foundation for any stochastic generative process.  
* Random variables are measurable maps enabling transformation from sample space to data space; their distributions (pmf/pdf) define observable statistics.  
* The likelihood $p(x\mid\theta)$ reinterprets the pdf/pmf as a function of the parameters, forming the core objective for parameter estimation in generative modeling.  
* MLE aligns the model distribution with empirical data via KL minimization; Fisher information quantifies estimator efficiency.  
* These constructs underpin modern generative architectures (VAEs, flows, diffusion, transformers) through explicit or implicit likelihood maximization.