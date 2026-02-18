
# Chapter 1 Foundations of Probabilistic Generative Modeling

---

## 1. Probability Spaces, Random Variables, Likelihood $p(x|\theta)$

### Definition

- **Probability Space**: A probability space is a mathematical construct $(\Omega, \mathcal{F}, P)$ where:
  - $\Omega$: Sample space (set of all possible outcomes)
  - $\mathcal{F}$: $\sigma$-algebra of events (subsets of $\Omega$)
  - $P$: Probability measure, $P: \mathcal{F} \rightarrow [0,1]$, with $P(\Omega) = 1$

- **Random Variable**: A random variable $X$ is a measurable function $X: \Omega \rightarrow \mathbb{R}$ mapping outcomes to real values.

- **Likelihood**: The likelihood function $p(x|\theta)$ is the probability (for discrete $x$) or probability density (for continuous $x$) of observing data $x$ given model parameters $\theta$.

### Mathematical Formulation

- **Probability Mass Function (PMF)** (discrete):
  $$
  P(X = x) = p(x|\theta)
  $$
- **Probability Density Function (PDF)** (continuous):
  $$
  P(a < X < b) = \int_a^b p(x|\theta) dx
  $$
- **Likelihood Function**:
  $$
  \mathcal{L}(\theta; x) = p(x|\theta)
  $$

### Step-by-Step Explanation

1. **Define the sample space $\Omega$**: Enumerate all possible outcomes.
2. **Specify the $\sigma$-algebra $\mathcal{F}$**: Identify all measurable events.
3. **Assign the probability measure $P$**: Ensure $P$ is countably additive and $P(\Omega) = 1$.
4. **Introduce random variables**: Map outcomes to real values for modeling.
5. **Model the data generation process**: Specify $p(x|\theta)$, the likelihood of data $x$ under parameters $\theta$.

---

## 2. Maximum-Likelihood Estimation (MLE) & Bayesian Inference

### Definition

- **Maximum-Likelihood Estimation (MLE)**: A method to estimate parameters $\theta$ by maximizing the likelihood of observed data.
- **Bayesian Inference**: Updates the probability distribution of parameters $\theta$ given observed data $x$ using Bayes’ theorem.

### Mathematical Formulation

- **MLE Objective**:
  $$
  \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} p(x|\theta)
  $$
  For $N$ i.i.d. samples $x_1, \ldots, x_N$:
  $$
  \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^N p(x_i|\theta)
  $$
  Or, equivalently, maximizing the log-likelihood:
  $$
  \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^N \log p(x_i|\theta)
  $$

- **Bayesian Posterior**:
  $$
  p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}
  $$
  where $p(\theta)$ is the prior, $p(x|\theta)$ is the likelihood, and $p(x)$ is the marginal likelihood:
  $$
  p(x) = \int p(x|\theta) p(\theta) d\theta
  $$

### Step-by-Step Explanation

#### MLE

1. **Specify the likelihood $p(x|\theta)$**.
2. **Formulate the log-likelihood for observed data**.
3. **Optimize $\theta$**: Use analytical or numerical optimization to maximize the log-likelihood.

#### Bayesian Inference

1. **Specify the prior $p(\theta)$**: Encodes beliefs about $\theta$ before observing data.
2. **Compute the likelihood $p(x|\theta)$**.
3. **Apply Bayes’ theorem**: Compute the posterior $p(\theta|x)$.
4. **Marginalize if necessary**: Integrate over $\theta$ for predictions or model evidence.

---

## 3. Latent Variables, Marginal Likelihood $p(x)=\int p(x,z)\,dz$

### Definition

- **Latent Variables**: Unobserved variables $z$ that explain observed data $x$ via a joint distribution $p(x, z)$.
- **Marginal Likelihood**: The probability of observed data $x$ marginalized over latent variables $z$.

### Mathematical Formulation

- **Joint Distribution**:
  $$
  p(x, z) = p(x|z) p(z)
  $$
- **Marginal Likelihood**:
  $$
  p(x) = \int p(x, z) dz = \int p(x|z) p(z) dz
  $$

### Step-by-Step Explanation

1. **Define latent variable model**: Specify $p(z)$ (prior over $z$) and $p(x|z)$ (likelihood of $x$ given $z$).
2. **Formulate the joint distribution $p(x, z)$**.
3. **Marginalize over $z$**: Integrate (or sum, if $z$ is discrete) to obtain $p(x)$.
4. **Use $p(x)$ for model evaluation**: The marginal likelihood is central for model selection and evidence computation.

---

## 4. EM Algorithm, Variational Inference Primer

### EM Algorithm

#### Definition

- **Expectation-Maximization (EM) Algorithm**: An iterative method for MLE in models with latent variables.

#### Mathematical Formulation

- **Observed data**: $x$
- **Latent variables**: $z$
- **Parameters**: $\theta$
- **Complete-data log-likelihood**: $\log p(x, z|\theta)$

- **E-step**: Compute the expected complete-data log-likelihood under the current posterior $q(z) = p(z|x, \theta^{(t)})$:
  $$
  Q(\theta, \theta^{(t)}) = \mathbb{E}_{z \sim p(z|x, \theta^{(t)})} [\log p(x, z|\theta)]
  $$
- **M-step**: Maximize $Q$ with respect to $\theta$:
  $$
  \theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})
  $$

#### Step-by-Step Explanation

1. **Initialize parameters $\theta^{(0)}$**.
2. **E-step**: Compute $p(z|x, \theta^{(t)})$ and evaluate $Q(\theta, \theta^{(t)})$.
3. **M-step**: Update parameters by maximizing $Q$.
4. **Iterate**: Repeat E and M steps until convergence.

---

### Variational Inference Primer

#### Definition

- **Variational Inference (VI)**: Approximates intractable posteriors $p(z|x)$ with a tractable family $q(z)$ by minimizing the Kullback-Leibler (KL) divergence.

#### Mathematical Formulation

- **Evidence Lower Bound (ELBO)**:
  $$
  \log p(x) \geq \mathbb{E}_{q(z)}[\log p(x, z) - \log q(z)] = \mathcal{L}(q)
  $$
- **KL Divergence**:
  $$
  \mathrm{KL}(q(z) \| p(z|x)) = \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{p(z|x)} \right]
  $$
- **Optimization Objective**:
  $$
  q^*(z) = \arg\min_{q \in \mathcal{Q}} \mathrm{KL}(q(z) \| p(z|x))
  $$

#### Step-by-Step Explanation

1. **Define variational family $\mathcal{Q}$**: Choose a tractable family of distributions $q(z)$.
2. **Formulate the ELBO**: Express $\log p(x)$ as a sum of ELBO and KL divergence.
3. **Optimize ELBO**: Maximize $\mathcal{L}(q)$ with respect to $q(z)$, typically via gradient-based methods.
4. **Approximate posterior**: Use optimized $q^*(z)$ as a surrogate for $p(z|x)$ in downstream inference.

---