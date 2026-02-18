---

# Gaussian Mixture Models (GMM)

---

## Definition

A **Gaussian Mixture Model (GMM)** is a probabilistic generative model that represents the distribution of data as a convex combination of multiple Gaussian (normal) distributions, each with its own mean and covariance. GMMs are widely used for unsupervised learning tasks such as clustering, density estimation, and as a building block in more complex models.

---

## Mathematical Formulation

Let $x \in \mathbb{R}^d$ denote a $d$-dimensional observed data point. The GMM models the probability density function as:

$$
p(x \mid \Theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

where:
- $K$ is the number of mixture components (clusters).
- $\pi_k$ is the mixing coefficient for component $k$, with $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \geq 0$.
- $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ is the multivariate Gaussian distribution:
  $$
  \mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) \right)
  $$
- $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ denotes all model parameters.

---

## Latent Variable Representation

Introduce a latent variable $z \in \{1, \ldots, K\}$ indicating the component assignment for each data point. The generative process is:

1. Sample $z \sim \text{Categorical}(\pi_1, \ldots, \pi_K)$.
2. Sample $x \sim \mathcal{N}(\mu_z, \Sigma_z)$.

The joint distribution is:

$$
p(x, z \mid \Theta) = \pi_z \, \mathcal{N}(x \mid \mu_z, \Sigma_z)
$$

The marginal distribution over $x$ is obtained by summing over $z$:

$$
p(x \mid \Theta) = \sum_{z=1}^K p(x, z \mid \Theta)
$$

---

## Parameter Estimation: Expectation-Maximization (EM) Algorithm

Given a dataset $X = \{x_i\}_{i=1}^N$, the goal is to estimate the parameters $\Theta$ that maximize the likelihood:

$$
\mathcal{L}(\Theta) = \prod_{i=1}^N p(x_i \mid \Theta)
$$

Or, equivalently, maximize the log-likelihood:

$$
\log \mathcal{L}(\Theta) = \sum_{i=1}^N \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)
$$

Direct maximization is intractable due to the sum inside the log. The EM algorithm is used:

### E-Step (Expectation)

Compute the posterior probability (responsibility) that component $k$ generated $x_i$:

$$
\gamma_{ik} = p(z_i = k \mid x_i, \Theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}
$$

### M-Step (Maximization)

Update the parameters using the responsibilities:

- Effective number of points assigned to component $k$:
  $$
  N_k = \sum_{i=1}^N \gamma_{ik}
  $$

- Update mixing coefficients:
  $$
  \pi_k^{(t+1)} = \frac{N_k}{N}
  $$

- Update means:
  $$
  \mu_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i
  $$

- Update covariances:
  $$
  \Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^\top
  $$

Repeat E and M steps until convergence (e.g., log-likelihood change below a threshold).

---

## Inference

Given a new data point $x^*$, the posterior probability that it belongs to component $k$ is:

$$
p(z^* = k \mid x^*, \Theta) = \frac{\pi_k \mathcal{N}(x^* \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x^* \mid \mu_j, \Sigma_j)}
$$

---

## Model Selection

- The number of components $K$ is typically selected using model selection criteria such as:
  - Bayesian Information Criterion (BIC):
    $$
    \text{BIC} = -2 \log \mathcal{L}(\hat{\Theta}) + p \log N
    $$
    where $p$ is the number of free parameters.
  - Akaike Information Criterion (AIC):
    $$
    \text{AIC} = -2 \log \mathcal{L}(\hat{\Theta}) + 2p
    $$

---

## Properties and Limitations

- **Expressiveness:** GMMs can approximate any continuous density given enough components.
- **Identifiability:** The model is not identifiable up to permutation of components.
- **Singularities:** Covariance matrices can become singular if a component collapses onto a single data point.
- **Initialization Sensitivity:** EM can converge to local optima; initialization (e.g., k-means) is critical.

---

## Applications

- **Clustering:** Soft assignment of data points to clusters.
- **Density Estimation:** Modeling complex, multimodal distributions.
- **Anomaly Detection:** Low likelihood under the model indicates outliers.
- **Speaker Recognition:** Modeling speaker-specific feature distributions.
- **Image Segmentation:** Pixel clustering in color or feature space.

---

## Extensions

- **Bayesian GMM:** Place priors over parameters, infer posterior distributions.
- **Infinite GMM (Dirichlet Process Mixture):** Nonparametric extension allowing an unbounded number of components.
- **Mixtures of Factor Analyzers:** For high-dimensional data, model each component with a lower-dimensional factor model.

---

## Summary Table

| Parameter | Description |
|-----------|-------------|
| $K$ | Number of mixture components |
| $\pi_k$ | Mixing coefficient for component $k$ |
| $\mu_k$ | Mean vector of component $k$ |
| $\Sigma_k$ | Covariance matrix of component $k$ |
| $\gamma_{ik}$ | Responsibility of component $k$ for data point $x_i$ |

---