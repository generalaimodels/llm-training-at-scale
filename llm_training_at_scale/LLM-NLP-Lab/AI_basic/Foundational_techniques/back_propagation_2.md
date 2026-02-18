
## Backpropagation

---

### Definition

Backpropagation is the algorithmic framework for efficiently computing gradients of a loss function with respect to model parameters in feedforward and recurrent neural networks, enabling gradient-based optimization.

---

### Pertinent Equations

#### 1. **Forward Pass**

- For a layer $l$:
  - $$ a^{(l)} = f^{(l)}(z^{(l)}) $$
  - $$ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} $$
  - Where $a^{(0)} = x$ (input), $f^{(l)}$ is the activation function.

#### 2. **Loss Function**

- For output $\hat{y}$ and target $y$:
  - $$ L = \mathcal{L}(\hat{y}, y) $$

#### 3. **Backward Pass (Gradients)**

- Output layer error:
  - $$ \delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \odot f'^{(L)}(z^{(L)}) $$
- Hidden layer error (for $l = L-1, ..., 1$):
  - $$ \delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'^{(l)}(z^{(l)}) $$
- Gradients w.r.t. weights and biases:
  - $$ \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T $$
  - $$ \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} $$

---

### Key Principles

- **Chain Rule:** Central to propagating gradients backward through composite functions.
- **Local Gradients:** Each layer computes gradients using only local information.
- **Parameter Sharing:** Efficient for deep and recurrent architectures.

---

### Detailed Concept Analysis

- **Forward Pass:** Computes activations and stores intermediate values ($z^{(l)}$, $a^{(l)}$) for use in the backward pass.
- **Backward Pass:** Recursively computes gradients from output to input using the chain rule.
- **Gradient Accumulation:** Gradients are accumulated for each parameter, enabling batch or stochastic updates.

---

### Importance

- **Enables Deep Learning:** Without backpropagation, training deep networks would be computationally infeasible.
- **Efficiency:** Reduces computational complexity from exponential to linear in the number of layers.
- **Generalizability:** Applicable to arbitrary computation graphs (e.g., CNNs, RNNs, GNNs).

---

### Pros vs. Cons

| Aspect                | Pros                                                      | Cons                                                      |
|-----------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| Computationally Efficient | Scales linearly with depth                              | Susceptible to vanishing/exploding gradients              |
| General Applicability | Works for any differentiable architecture                 | Requires differentiable operations                        |
| Exact Gradients       | Provides precise gradients for optimization               | Memory-intensive due to storage of intermediate activations|

---

### Cutting-Edge Advances

- **Automatic Differentiation:** Modern frameworks (e.g., PyTorch, TensorFlow) automate backpropagation via dynamic/static computation graphs.
- **Gradient Checkpointing:** Reduces memory usage by recomputing activations during backward pass.
- **Second-Order Methods:** Extensions (e.g., Hessian-vector products) for advanced optimization.
- **Backpropagation Through Time (BPTT):** Specialized for RNNs, unrolling through temporal steps.

---

## Model Architecture Breakdown

### 1. **Pre-processing**

- **Input Normalization:**
  - $$ x' = \frac{x - \mu}{\sigma} $$
  - Where $\mu$ and $\sigma$ are dataset mean and standard deviation.

### 2. **Core Model (Feedforward Example)**

- **Layer $l$ computation:**
  - $$ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} $$
  - $$ a^{(l)} = f^{(l)}(z^{(l)}) $$

### 3. **Post-Training Procedures**

- **Weight Averaging (e.g., SWA):**
  - $$ W_{avg} = \frac{1}{K} \sum_{k=1}^K W^{(k)} $$
- **Quantization/Pruning:** Reduces model size for deployment.

---

## Step-by-Step Training Pseudo-Algorithm

```python
# Pseudo-algorithm for one training iteration

for batch in data_loader:
    # Pre-processing
    x, y = preprocess(batch)  # e.g., normalization

    # Forward pass
    activations = [x]
    zs = []
    for l in range(1, L+1):
        z = W[l] @ activations[-1] + b[l]
        zs.append(z)
        a = f[l](z)
        activations.append(a)
    y_pred = activations[-1]

    # Compute loss
    L = loss_fn(y_pred, y)

    # Backward pass
    delta = dL/da * f'[L](zs[-1])
    grads_W = []
    grads_b = []
    for l in reversed(range(1, L+1)):
        grad_W = delta @ activations[l-1].T
        grad_b = delta
        grads_W.insert(0, grad_W)
        grads_b.insert(0, grad_b)
        if l > 1:
            delta = (W[l].T @ delta) * f'[l-1](zs[l-2])

    # Parameter update (e.g., SGD)
    for l in range(1, L+1):
        W[l] -= eta * grads_W[l-1]
        b[l] -= eta * grads_b[l-1]
```

---

## Evaluation Phase

### Metrics (SOTA and Domain-Specific)

#### 1. **Loss Functions**

- **Cross-Entropy (Classification):**
  - $$ L = -\sum_{i=1}^N y_i \log(\hat{y}_i) $$
- **Mean Squared Error (Regression):**
  - $$ L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 $$

#### 2. **Accuracy**
- $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}} $$

#### 3. **F1 Score**
- $$ \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

#### 4. **AUC-ROC**
- Area under the ROC curve, computed as:
  - $$ \text{AUC} = \int_0^1 TPR(FPR^{-1}(x)) dx $$

#### 5. **Task-Specific Metrics**
- **BLEU, ROUGE (NLP)**
- **mAP (Object Detection)**

---

### Best Practices & Pitfalls

- **Best Practices:**
  - Use validation/test sets for unbiased evaluation.
  - Monitor both loss and task-specific metrics.
  - Employ early stopping based on validation performance.

- **Pitfalls:**
  - Overfitting to training data.
  - Ignoring gradient flow issues (vanishing/exploding).
  - Failing to check gradient correctness (use gradient checking).

---