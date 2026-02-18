

# Stochastic Gradient Descent (SGD) and Adam Optimizer

---

## 1. Definition

### Stochastic Gradient Descent (SGD)
- **SGD** is an iterative optimization algorithm for minimizing an objective function $f(\theta)$, typically a loss function in machine learning, by updating parameters $\theta$ in the direction of the negative gradient computed from a randomly selected subset (mini-batch) of the data.

### Adam (Adaptive Moment Estimation)
- **Adam** is an adaptive learning rate optimization algorithm that computes individual learning rates for different parameters using estimates of first and second moments of the gradients.

---

## 2. Mathematical Equations

### SGD Update Rule
- **Parameter Update:**
  $$
  \theta_{t+1} = \theta_t - \eta \nabla_\theta f_i(\theta_t)
  $$
  - $\theta_t$: Parameters at iteration $t$
  - $\eta$: Learning rate
  - $\nabla_\theta f_i(\theta_t)$: Gradient of loss for sample $i$ (or mini-batch) at $\theta_t$

### Adam Update Rule
- **Moment Estimates:**
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  $$
  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  $$
  - $g_t = \nabla_\theta f(\theta_t)$: Gradient at $t$
  - $m_t$: First moment (mean) estimate
  - $v_t$: Second moment (uncentered variance) estimate
  - $\beta_1, \beta_2$: Exponential decay rates for the moment estimates

- **Bias Correction:**
  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
  $$
  $$
  \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$

- **Parameter Update:**
  $$
  \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$
  - $\epsilon$: Small constant for numerical stability

---

## 3. Pseudo-Algorithm

### SGD Pseudo-Algorithm

1. **Initialize** parameters $\theta_0$
2. **For** $t = 0$ to $T-1$:
   - Sample mini-batch $\mathcal{B}_t$ from data
   - Compute gradient: $g_t = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla_\theta f_i(\theta_t)$
   - Update parameters: $\theta_{t+1} = \theta_t - \eta g_t$

### Adam Pseudo-Algorithm

1. **Initialize** parameters $\theta_0$, $m_0 = 0$, $v_0 = 0$
2. **For** $t = 1$ to $T$:
   - Sample mini-batch $\mathcal{B}_t$ from data
   - Compute gradient: $g_t = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla_\theta f_i(\theta_t)$
   - Update biased first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - Update biased second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - Compute bias-corrected moments:
     - $\hat{m}_t = m_t / (1 - \beta_1^t)$
     - $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - Update parameters: $\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

---

## 4. Underlying Principles and Mechanisms

### SGD
- **Stochasticity:** Uses random mini-batches, introducing noise that can help escape local minima and saddle points.
- **Efficiency:** Reduces computation per iteration compared to full-batch gradient descent.
- **Convergence:** Sensitive to learning rate; may oscillate or diverge if not properly tuned.

### Adam
- **Adaptive Learning Rates:** Maintains per-parameter learning rates, adapting based on historical gradients.
- **Moment Estimation:** Utilizes exponentially decaying averages of past gradients (first moment) and squared gradients (second moment).
- **Bias Correction:** Compensates for initialization bias in moment estimates.

---

## 5. Significance and Use Cases

### SGD
- **Significance:** Foundation of modern deep learning optimization; simple, memory-efficient.
- **Use Cases:** Training neural networks, logistic regression, SVMs, and other large-scale machine learning models.

### Adam
- **Significance:** Default optimizer for many deep learning frameworks; robust to hyperparameter settings.
- **Use Cases:** Deep neural networks (CNNs, RNNs, Transformers), especially where gradients are sparse or data is noisy.

---

## 6. Advantages and Disadvantages

### SGD

**Advantages:**
- Simple to implement
- Low memory footprint
- Can generalize well with proper regularization

**Disadvantages:**
- Sensitive to learning rate
- May get stuck in saddle points or poor local minima
- Slow convergence, especially on ill-conditioned problems

### Adam

**Advantages:**
- Fast convergence
- Handles sparse gradients well
- Less sensitive to initial learning rate
- Per-parameter learning rate adaptation

**Disadvantages:**
- Can lead to poor generalization compared to SGD in some cases
- May converge to sharp minima
- Requires more memory (stores moments for each parameter)

---

## 7. Variants, Extensions, and Recent Developments

### SGD Variants
- **SGD with Momentum:** Adds a velocity term to accelerate convergence.
- **Nesterov Accelerated Gradient (NAG):** Looks ahead before updating parameters.
- **SGD with Learning Rate Schedules:** Decays learning rate over time (e.g., step decay, cosine annealing).

### Adam Variants
- **AdamW:** Decouples weight decay from gradient update, improving generalization.
- **AMSGrad:** Modifies second moment estimate to ensure non-increasing step sizes, addressing convergence issues.
- **AdaBound:** Interpolates between Adam and SGD by bounding learning rates.

### Recent Developments
- **Decoupled Weight Decay (AdamW):** Now standard in many frameworks.
- **Rectified Adam (RAdam):** Rectifies variance of adaptive learning rates for more stable training.
- **Lookahead Optimizer:** Combines fast and slow weights for improved stability.

---

## 8. Best Practices and Common Pitfalls

### Best Practices
- **SGD:**
  - Use momentum for faster convergence.
  - Employ learning rate schedules.
  - Tune batch size and learning rate jointly.
- **Adam:**
  - Default $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ work well in most cases.
  - Consider AdamW for better generalization.
  - Monitor validation loss to avoid overfitting.

### Common Pitfalls
- **SGD:**
  - Too large learning rate causes divergence.
  - Too small learning rate leads to slow convergence.
  - Ignoring momentum or learning rate schedules.
- **Adam:**
  - Overfitting due to aggressive adaptation.
  - Poor generalization on some tasks compared to SGD.
  - Not decoupling weight decay (use AdamW).

---