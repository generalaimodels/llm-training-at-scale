---

# Chapter 1 Foundations of Probabilistic Generative Modeling

---

## 1. Probability Spaces

### Definition

A **probability space** is a mathematical construct that provides a formal model for randomness and uncertainty. It is defined as a triple $(\Omega, \mathcal{F}, P)$, where:

- $\Omega$: **Sample space** — the set of all possible outcomes.
- $\mathcal{F}$: **Sigma-algebra** — a collection of subsets of $\Omega$ (called events), closed under complementation and countable unions.
- $P$: **Probability measure** — a function $P: \mathcal{F} \rightarrow [0,1]$ assigning probabilities to events, satisfying:
  - $P(\Omega) = 1$
  - $P(\emptyset) = 0$
  - For any countable collection of disjoint events $\{A_i\}$, $P\left(\bigcup_i A_i\right) = \sum_i P(A_i)$

### Mathematical Formulation

$$
(\Omega, \mathcal{F}, P)
$$

#### Properties

- **Non-negativity**: $P(A) \geq 0$ for all $A \in \mathcal{F}$
- **Normalization**: $P(\Omega) = 1$
- **Countable Additivity**: For disjoint $A_i \in \mathcal{F}$,
  $$
  P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)
  $$

---

## 2. Random Variables

### Definition

A **random variable** is a measurable function $X: \Omega \rightarrow \mathcal{X}$, where $\mathcal{X}$ is the set of possible values (e.g., $\mathbb{R}$ for real-valued random variables).

### Types

- **Discrete random variable**: $\mathcal{X}$ is countable.
- **Continuous random variable**: $\mathcal{X}$ is uncountable (typically $\mathbb{R}^n$).

### Probability Distributions

- **Probability Mass Function (PMF)** for discrete $X$:
  $$
  p_X(x) = P(X = x)
  $$
- **Probability Density Function (PDF)** for continuous $X$:
  $$
  p_X(x) = \frac{d}{dx} P(X \leq x)
  $$

### Expectation

- **Discrete**:
  $$
  \mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \, p_X(x)
  $$
- **Continuous**:
  $$
  \mathbb{E}[X] = \int_{\mathcal{X}} x \, p_X(x) \, dx
  $$

---

## 3. Likelihood $p(x|\theta)$

### Definition

The **likelihood** is a function of the model parameters $\theta$ given observed data $x$. It quantifies how probable the observed data is under a specific parameterization of the model.

### Mathematical Formulation

Given a parametric family of distributions $\{p(x|\theta)\}$, the likelihood function is:
$$
\mathcal{L}(\theta; x) = p(x|\theta)
$$

- $x$: observed data
- $\theta$: model parameters

### Log-Likelihood

For computational convenience, the log-likelihood is often used:
$$
\ell(\theta; x) = \log p(x|\theta)
$$

For $N$ i.i.d. samples $x_1, \ldots, x_N$:
$$
\ell(\theta; x_{1:N}) = \sum_{i=1}^N \log p(x_i|\theta)
$$

### Maximum Likelihood Estimation (MLE)

The MLE seeks the parameter $\theta^*$ that maximizes the likelihood:
$$
\theta^* = \arg\max_\theta \mathcal{L}(\theta; x) = \arg\max_\theta p(x|\theta)
$$

Or equivalently, maximizing the log-likelihood:
$$
\theta^* = \arg\max_\theta \ell(\theta; x)
$$

---

## 4. Step-by-Step Explanation

### 4.1. Constructing a Probability Space

- **Step 1**: Define the sample space $\Omega$ (all possible outcomes).
- **Step 2**: Specify the sigma-algebra $\mathcal{F}$ (events of interest).
- **Step 3**: Assign a probability measure $P$ to events in $\mathcal{F}$, ensuring axioms are satisfied.

### 4.2. Defining Random Variables

- **Step 1**: Specify a measurable function $X: \Omega \rightarrow \mathcal{X}$.
- **Step 2**: Determine the distribution of $X$ (PMF or PDF).
- **Step 3**: Compute expectations, variances, and higher moments as required.

### 4.3. Likelihood in Generative Modeling

- **Step 1**: Choose a parametric family $p(x|\theta)$ to model the data.
- **Step 2**: Given observed data $x$, compute the likelihood $\mathcal{L}(\theta; x)$.
- **Step 3**: For multiple observations, use the product of likelihoods (or sum of log-likelihoods).
- **Step 4**: Optimize $\theta$ (e.g., via MLE) to fit the model to data.

### 4.4. Role in Generative Modeling

- **Generative models** define a joint distribution $p(x, z|\theta)$ over observed data $x$ and latent variables $z$.
- The marginal likelihood is:
  $$
  p(x|\theta) = \int p(x, z|\theta) dz
  $$
- Learning involves maximizing $p(x|\theta)$ with respect to $\theta$.

---

## 5. Summary Table

| Concept                | Mathematical Notation         | Key Properties/Usage                                  |
|------------------------|------------------------------|-------------------------------------------------------|
| Probability Space      | $(\Omega, \mathcal{F}, P)$   | Foundation for all probabilistic modeling             |
| Random Variable        | $X: \Omega \rightarrow \mathcal{X}$ | Maps outcomes to values, induces distributions        |
| Likelihood             | $p(x|\theta)$                | Probability of data given parameters, used for MLE    |
| Log-Likelihood         | $\log p(x|\theta)$           | Sum over data, numerically stable for optimization    |

---

## 6. End-to-End Example

**Suppose**: $x_1, \ldots, x_N$ are i.i.d. samples from a Gaussian $\mathcal{N}(\mu, \sigma^2)$.

- **Probability space**: $\Omega = \mathbb{R}^N$, $\mathcal{F}$ = Borel sets, $P$ = product measure.
- **Random variable**: $X_i: \Omega \rightarrow \mathbb{R}$, $X_i(\omega) = x_i$.
- **Likelihood**:
  $$
  p(x_{1:N}|\mu, \sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
  $$
- **Log-likelihood**:
  $$
  \ell(\mu, \sigma^2; x_{1:N}) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (x_i - \mu)^2
  $$
- **MLE**:
  $$
  \mu^* = \frac{1}{N} \sum_{i=1}^N x_i, \quad (\sigma^2)^* = \frac{1}{N} \sum_{i=1}^N (x_i - \mu^*)^2
  $$

---

**This covers the foundational mathematical and conceptual framework for probabilistic generative modeling, including probability spaces, random variables, and the likelihood function $p(x|\theta)$.**