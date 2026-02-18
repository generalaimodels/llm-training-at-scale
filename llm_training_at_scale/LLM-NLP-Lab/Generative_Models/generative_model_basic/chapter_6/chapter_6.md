---

# Chapter 6 Flow-Based Generative Models

Flow-based generative models are a class of deep generative models that learn invertible mappings between simple base distributions (e.g., Gaussian) and complex data distributions. These models enable exact log-likelihood computation and efficient sampling via invertible transformations.

## Mathematical Foundation

Given data $\mathbf{x} \in \mathbb{R}^D$, flow-based models define an invertible function $f: \mathbb{R}^D \rightarrow \mathbb{R}^D$ such that:
$$
\mathbf{z} = f(\mathbf{x}), \quad \mathbf{x} = f^{-1}(\mathbf{z})
$$
where $\mathbf{z}$ is sampled from a simple prior $p_{\mathbf{z}}(\mathbf{z})$ (e.g., $\mathcal{N}(0, I)$).

The change of variables formula gives the data likelihood:
$$
p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f(\mathbf{x})) \left| \det \left( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right) \right|
$$
or, in log-space:
$$
\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(f(\mathbf{x})) + \log \left| \det \left( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right) \right|
$$

The core challenge is designing $f$ such that both $f$ and $f^{-1}$ are tractable and the Jacobian determinant is efficiently computable.

---

## Non-linear Independent Components Estimation (NICE)

### Definition

NICE is the foundational flow-based model introducing volume-preserving, invertible transformations using additive coupling layers.

### Mathematical Formulation

- **Coupling Layer**: Partition $\mathbf{x}$ into two subsets: $\mathbf{x}_1, \mathbf{x}_2$.
- **Transformation**:
  $$
  \begin{align*}
  \mathbf{y}_1 &= \mathbf{x}_1 \\
  \mathbf{y}_2 &= \mathbf{x}_2 + m(\mathbf{x}_1)
  \end{align*}
  $$
  where $m$ is a neural network.

- **Inverse**:
  $$
  \begin{align*}
  \mathbf{x}_1 &= \mathbf{y}_1 \\
  \mathbf{x}_2 &= \mathbf{y}_2 - m(\mathbf{y}_1)
  \end{align*}
  $$

- **Jacobian**:
  $$
  J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
  \begin{bmatrix}
  I & 0 \\
  \frac{\partial m}{\partial \mathbf{x}_1} & I
  \end{bmatrix}
  $$
  $\Rightarrow \det J = 1$

### Step-by-Step Explanation

- **Invertibility**: Guaranteed by the structure of the coupling layer.
- **Volume Preservation**: $\det J = 1$; no scaling of probability mass.
- **Stacking**: Multiple coupling layers with permutations between layers to ensure all dimensions are transformed.
- **Limitations**: No scaling; limited expressivity.

---

## Real NVP (Real-valued Non-Volume Preserving)

### Definition

Real NVP extends NICE by introducing scaling in the coupling layers, enabling non-volume-preserving transformations.

### Mathematical Formulation

- **Affine Coupling Layer**:
  $$
  \begin{align*}
  \mathbf{y}_1 &= \mathbf{x}_1 \\
  \mathbf{y}_2 &= \mathbf{x}_2 \odot \exp(s(\mathbf{x}_1)) + t(\mathbf{x}_1)
  \end{align*}
  $$
  where $s, t$ are neural networks, $\odot$ denotes elementwise multiplication.

- **Inverse**:
  $$
  \begin{align*}
  \mathbf{x}_1 &= \mathbf{y}_1 \\
  \mathbf{x}_2 &= (\mathbf{y}_2 - t(\mathbf{y}_1)) \odot \exp(-s(\mathbf{y}_1))
  \end{align*}
  $$

- **Jacobian Determinant**:
  $$
  \log \left| \det \left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right) \right| = \sum_j s_j(\mathbf{x}_1)
  $$

### Step-by-Step Explanation

- **Expressivity**: Scaling allows for richer transformations.
- **Efficient Inversion**: Only requires forward/backward passes through $s, t$.
- **Efficient Jacobian**: Diagonal, so determinant is sum of $s$ outputs.
- **Stacking**: Multiple layers with permutations for full coverage.
- **Applications**: Image modeling, density estimation.

---

## Glow

### Definition

Glow is a flow-based model that improves upon Real NVP with invertible $1 \times 1$ convolutions and actnorm layers, enhancing expressivity and training stability.

### Mathematical Formulation

- **Flow Step**: Each step consists of:
  1. **ActNorm**: Per-channel affine transformation.
     $$
     \mathbf{y} = \mathbf{x} \odot \mathbf{s} + \mathbf{b}
     $$
  2. **Invertible $1 \times 1$ Convolution**:
     $$
     \mathbf{y} = W \mathbf{x}
     $$
     where $W$ is a learned invertible matrix.
  3. **Affine Coupling Layer** (as in Real NVP).

- **Jacobian Determinant**:
  $$
  \log |\det J| = \log |\det W| + \sum_j s_j(\mathbf{x}_1) + \log |\det \text{ActNorm}|
  $$

### Step-by-Step Explanation

- **ActNorm**: Data-dependent initialization, stabilizes training.
- **Invertible $1 \times 1$ Convolution**: Permutes channels, increases mixing, determinant via LU decomposition.
- **Affine Coupling**: As in Real NVP.
- **Multi-scale Architecture**: Splits off part of the representation at each level, reducing computational cost.
- **Training**: Maximum likelihood via exact log-likelihood.
- **Sampling**: Invertible, enables efficient generation.

---

## Masked Autoregressive Flow (MAF)

### Definition

MAF is a flow-based model where each transformation is an autoregressive mapping, enabling flexible density estimation.

### Mathematical Formulation

- **Autoregressive Transformation**:
  $$
  \mathbf{y}_i = \mu_i(\mathbf{x}_{1:i-1}) + \sigma_i(\mathbf{x}_{1:i-1}) \cdot \mathbf{x}_i
  $$
  where $\mu_i, \sigma_i$ are neural networks.

- **Inverse**: Sequential, as each $\mathbf{x}_i$ depends on previous $\mathbf{y}_{1:i-1}$.

- **Jacobian Determinant**:
  $$
  \log \left| \det \left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right) \right| = \sum_{i=1}^D \log \sigma_i(\mathbf{x}_{1:i-1})
  $$

### Step-by-Step Explanation

- **Autoregressive Property**: Each output depends only on previous inputs.
- **Masking**: Enforced via masked neural networks (e.g., MADE).
- **Efficient Density Estimation**: Parallelizable in density computation, sequential in sampling.
- **Expressivity**: Highly flexible, can model complex dependencies.
- **Limitations**: Slow sampling due to sequential inversion.

---

## Neural Spline Flow (NSF)

### Definition

NSF generalizes affine coupling by using monotonic rational-quadratic splines for more expressive, invertible transformations.

### Mathematical Formulation

- **Spline Coupling Layer**:
  $$
  \begin{align*}
  \mathbf{y}_1 &= \mathbf{x}_1 \\
  \mathbf{y}_2 &= \text{Spline}(\mathbf{x}_2; \theta(\mathbf{x}_1))
  \end{align*}
  $$
  where $\theta(\mathbf{x}_1)$ parameterizes the spline (knot positions, heights, derivatives).

- **Inverse**: Invert spline numerically or analytically (for rational-quadratic).

- **Jacobian Determinant**:
  $$
  \log \left| \det \left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right) \right| = \sum_{j} \log \left| \frac{\partial \text{Spline}(x_{2j}; \theta(\mathbf{x}_1))}{\partial x_{2j}} \right|
  $$

### Step-by-Step Explanation

- **Spline Parameterization**: Neural network outputs spline parameters conditioned on $\mathbf{x}_1$.
- **Monotonicity**: Ensured by construction, guaranteeing invertibility.
- **Expressivity**: Can approximate complex, non-linear functions.
- **Efficient Inversion**: Rational-quadratic splines allow fast inversion.
- **Stacking**: Multiple spline coupling layers for full coverage.

---

# Summary Table

| Model | Transformation | Invertibility | Jacobian | Expressivity | Sampling |
|-------|----------------|--------------|----------|-------------|----------|
| NICE  | Additive       | Trivial      | 1        | Limited     | Fast     |
| Real NVP | Affine      | Trivial      | Diagonal | Moderate    | Fast     |
| Glow  | Affine + $1\times1$ Conv | Trivial | Efficient | High | Fast |
| MAF   | Autoregressive | Sequential   | Triangular | High      | Slow     |
| NSF   | Spline         | Trivial      | Diagonal | Very High  | Fast     |

---