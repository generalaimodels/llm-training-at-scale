---

# Maximum-Likelihood Estimation (MLE) & Bayesian Inference

## Definition

- **Maximum-Likelihood Estimation (MLE):** A method for estimating the parameters $\theta$ of a probabilistic model by maximizing the likelihood function $p(x|\theta)$ given observed data $x$.
- **Bayesian Inference:** A framework for updating the probability distribution over model parameters $\theta$ given observed data $x$, using Bayes’ theorem.

---

## Mathematical Formulation

### Maximum-Likelihood Estimation (MLE)

Given data $x_{1:N} = \{x_1, \ldots, x_N\}$ and model $p(x|\theta)$:

$$
\theta^* = \arg\max_\theta p(x_{1:N}|\theta)
$$

For i.i.d. data:

$$
p(x_{1:N}|\theta) = \prod_{i=1}^N p(x_i|\theta)
$$

Log-likelihood:

$$
\ell(\theta) = \log p(x_{1:N}|\theta) = \sum_{i=1}^N \log p(x_i|\theta)
$$

### Bayesian Inference

Given prior $p(\theta)$ and likelihood $p(x|\theta)$, the posterior is:

$$
p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}
$$

where

$$
p(x) = \int p(x|\theta) p(\theta) d\theta
$$

---

## Step-by-Step Explanation

### MLE: End-to-End

1. **Model Specification:** Choose a parametric family $p(x|\theta)$.
2. **Likelihood Construction:** For observed data $x_{1:N}$, write the likelihood $p(x_{1:N}|\theta)$.
3. **Log-Likelihood:** Take the logarithm for numerical stability and analytical tractability.
4. **Optimization:** Solve
   $$
   \theta^* = \arg\max_\theta \ell(\theta)
   $$
   using analytical or numerical optimization (e.g., gradient ascent).
5. **Interpretation:** $\theta^*$ is the parameter value under which the observed data is most probable.

### Bayesian Inference: End-to-End

1. **Prior Selection:** Specify a prior $p(\theta)$ encoding beliefs about $\theta$ before observing data.
2. **Likelihood Construction:** As above, define $p(x|\theta)$.
3. **Posterior Computation:** Apply Bayes’ theorem:
   $$
   p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}
   $$
4. **Marginal Likelihood (Evidence):** Compute
   $$
   p(x) = \int p(x|\theta) p(\theta) d\theta
   $$
   (often intractable; approximations may be required).
5. **Posterior Analysis:** Use $p(\theta|x)$ for prediction, uncertainty quantification, or further inference.

---

# Latent Variables, Marginal Likelihood $p(x) = \int p(x, z)\,dz$

## Definition

- **Latent Variables:** Unobserved (hidden) variables $z$ introduced to model complex dependencies or structure in data $x$.
- **Marginal Likelihood:** The probability of observed data $x$ under the model, integrating out latent variables $z$.

---

## Mathematical Formulation

- **Joint Distribution:** $p(x, z|\theta)$
- **Marginal Likelihood:**
  $$
  p(x|\theta) = \int p(x, z|\theta) dz
  $$
- **Posterior over Latents:**
  $$
  p(z|x, \theta) = \frac{p(x, z|\theta)}{p(x|\theta)}
  $$

---

## Step-by-Step Explanation

1. **Model Construction:** Define $p(x, z|\theta)$, where $z$ are latent variables.
2. **Marginalization:** Compute $p(x|\theta)$ by integrating out $z$:
   $$
   p(x|\theta) = \int p(x, z|\theta) dz
   $$
   - For discrete $z$, sum over all possible values.
   - For continuous $z$, integrate over the latent space.
3. **Posterior Inference:** Compute $p(z|x, \theta)$ for tasks such as clustering, imputation, or representation learning.
4. **Learning:** Maximize $p(x|\theta)$ (or its lower bound) with respect to $\theta$.

---

# EM Algorithm, Variational Inference Primer

## EM Algorithm

### Definition

The **Expectation-Maximization (EM) algorithm** is an iterative method for maximum-likelihood estimation in models with latent variables.

---

### Mathematical Formulation

Given observed data $x$, latent variables $z$, and parameters $\theta$:

- **E-step:** Compute the expected complete-data log-likelihood under the current posterior:
  $$
  Q(\theta, \theta^{(t)}) = \mathbb{E}_{p(z|x, \theta^{(t)})}[\log p(x, z|\theta)]
  $$
- **M-step:** Maximize $Q$ with respect to $\theta$:
  $$
  \theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})
  $$

---

### Step-by-Step Explanation

1. **Initialization:** Set initial parameter $\theta^{(0)}$.
2. **E-step:** Compute $p(z|x, \theta^{(t)})$ and evaluate $Q(\theta, \theta^{(t)})$.
3. **M-step:** Update parameters:
   $$
   \theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})
   $$
4. **Convergence:** Repeat E and M steps until convergence (e.g., change in likelihood below threshold).

---

## Variational Inference Primer

### Definition

**Variational Inference (VI)** is a family of techniques for approximating intractable posteriors $p(z|x)$ by optimizing over a tractable family of distributions $q(z)$.

---

### Mathematical Formulation

- **Evidence Lower Bound (ELBO):**
  $$
  \log p(x) \geq \mathbb{E}_{q(z)}[\log p(x, z) - \log q(z)] = \mathcal{L}(q)
  $$
- **KL Divergence:**
  $$
  \mathrm{KL}(q(z) \| p(z|x)) = \mathbb{E}_{q(z)}\left[\log \frac{q(z)}{p(z|x)}\right]
  $$
- **Optimization:**
  $$
  q^*(z) = \arg\max_{q \in \mathcal{Q}} \mathcal{L}(q)
  $$

---

### Step-by-Step Explanation

1. **Intractable Posterior:** Recognize $p(z|x)$ is intractable due to complex marginalization.
2. **Variational Family:** Choose a tractable family $\mathcal{Q}$ (e.g., mean-field) for $q(z)$.
3. **ELBO Construction:** Write the ELBO:
   $$
   \mathcal{L}(q) = \mathbb{E}_{q(z)}[\log p(x, z) - \log q(z)]
   $$
4. **Optimization:** Maximize $\mathcal{L}(q)$ with respect to $q(z)$, often via coordinate ascent or stochastic gradient methods.
5. **Approximate Posterior:** Use $q^*(z)$ as an approximation to $p(z|x)$ for inference and learning.

---

# Summary Table

| Concept                | Mathematical Notation         | Key Properties/Usage                                  |
|------------------------|------------------------------|-------------------------------------------------------|
| MLE                    | $\theta^* = \arg\max_\theta p(x|\theta)$ | Point estimate maximizing likelihood                  |
| Bayesian Inference     | $p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}$ | Posterior over parameters                             |
| Latent Variables       | $z$                          | Unobserved variables, introduce structure             |
| Marginal Likelihood    | $p(x) = \int p(x, z) dz$     | Probability of data, integrating out latents          |
| EM Algorithm           | $Q(\theta, \theta^{(t)})$    | Iterative MLE with latent variables                   |
| Variational Inference  | $\mathcal{L}(q)$, $\mathrm{KL}(q(z)\|p(z|x))$ | Approximate inference via optimization                |

---