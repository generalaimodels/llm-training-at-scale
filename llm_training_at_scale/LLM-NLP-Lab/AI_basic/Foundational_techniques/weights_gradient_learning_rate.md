
## Weight Initialization

### Definition
Weight initialization refers to the strategy for assigning initial values to the parameters ($W$, $b$) of a neural network before training begins. Proper initialization is critical for stable and efficient convergence.

---

### Mathematical Formulations

#### 1. **Random Initialization**
- $$ W_{ij} \sim \mathcal{U}(-a, a) $$
- $$ W_{ij} \sim \mathcal{N}(0, \sigma^2) $$

#### 2. **Xavier/Glorot Initialization**
- For a layer with $n_{in}$ inputs and $n_{out}$ outputs:
- $$ W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right) $$
- $$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right) $$

#### 3. **He Initialization (for ReLU)**
- $$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right) $$

#### 4. **Orthogonal Initialization**
- $$ W = Q $$
- Where $Q$ is an orthogonal matrix from the QR decomposition of a random Gaussian matrix.

---

### Key Principles

- **Variance Preservation:** Prevents vanishing/exploding activations.
- **Activation Function Compatibility:** Initialization should match the nonlinearity (e.g., He for ReLU).
- **Symmetry Breaking:** Ensures different neurons learn different features.

---

### Detailed Concept Analysis

- **Vanishing/Exploding Gradients:** Poor initialization can cause gradients to shrink or grow exponentially through layers.
- **Layerwise Considerations:** Deep networks are more sensitive to initialization due to compounding effects.

---

### Importance

- **Convergence Speed:** Good initialization accelerates training.
- **Model Performance:** Reduces risk of poor local minima.
- **Stability:** Prevents numerical instabilities.

---

### Pros vs. Cons

| Method         | Pros                                    | Cons                                  |
|----------------|-----------------------------------------|---------------------------------------|
| Random         | Simple                                  | May cause instability                 |
| Xavier/Glorot  | Good for tanh/sigmoid                   | Not optimal for ReLU                  |
| He             | Optimal for ReLU                        | Not ideal for other activations       |
| Orthogonal     | Preserves norm, good for RNNs           | Computationally expensive             |

---

### Recent Developments

- **LSUV (Layer-sequential unit-variance):** Data-driven adjustment post-initialization.
- **Meta-initialization:** Learning initialization schemes via meta-learning.

---

### Best Practices & Pitfalls

- Match initialization to activation function.
- Avoid zero or identical initialization.
- Monitor for vanishing/exploding gradients during early training.

---

## Gradient Clipping

### Definition

Gradient clipping is a technique to limit (clip) the magnitude of gradients during backpropagation to prevent exploding gradients, especially in deep or recurrent networks.

---

### Mathematical Formulations

#### 1. **Norm Clipping**
- If $||g||_2 > \tau$:
  - $$ g \leftarrow g \cdot \frac{\tau}{||g||_2} $$
- Where $g$ is the gradient vector, $\tau$ is the threshold.

#### 2. **Value Clipping**
- $$ g_i \leftarrow \text{clip}(g_i, -\tau, \tau) $$

---

### Key Principles

- **Stabilization:** Prevents parameter updates from becoming excessively large.
- **Selective Clipping:** Only clips when necessary, preserving learning signal.

---

### Detailed Concept Analysis

- **Norm Clipping:** Preserves direction, only scales magnitude.
- **Value Clipping:** Can distort direction, less commonly used.

---

### Importance

- **Prevents Divergence:** Especially critical in RNNs and deep networks.
- **Enables Higher Learning Rates:** Without risk of instability.

---

### Pros vs. Cons

| Method         | Pros                                    | Cons                                  |
|----------------|-----------------------------------------|---------------------------------------|
| Norm Clipping  | Preserves gradient direction            | May mask underlying issues            |
| Value Clipping | Simple                                  | Can distort optimization trajectory   |

---

### Recent Developments

- **Adaptive Clipping:** Dynamic thresholds based on training statistics.
- **Per-layer Clipping:** Different thresholds for different layers.

---

### Best Practices & Pitfalls

- Set $\tau$ based on empirical gradient norms.
- Monitor for frequent clipping (may indicate deeper issues).
- Combine with proper initialization and normalization.

---

## Learning Rate Scheduling

### Definition

Learning rate scheduling refers to dynamically adjusting the learning rate ($\eta$) during training to improve convergence and generalization.

---

### Mathematical Formulations

#### 1. **Step Decay**
- $$ \eta_t = \eta_0 \cdot \gamma^{\left\lfloor \frac{t}{T} \right\rfloor} $$
- Where $\eta_0$ is initial rate, $\gamma$ is decay factor, $T$ is step interval.

#### 2. **Exponential Decay**
- $$ \eta_t = \eta_0 \cdot e^{-\lambda t} $$

#### 3. **Cosine Annealing**
- $$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right) $$

#### 4. **Cyclical Learning Rates**
- $$ \eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \text{triangular}(t) $$

#### 5. **Warmup**
- $$ \eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}} \quad \text{for} \quad t < T_{warmup} $$

---

### Key Principles

- **Exploration vs. Exploitation:** High initial rates for exploration, lower rates for fine-tuning.
- **Avoiding Local Minima:** Schedules can help escape shallow minima.

---

### Detailed Concept Analysis

- **Step/Exponential Decay:** Simple, widely used.
- **Cosine Annealing:** Smooth, effective for large-scale training.
- **Warmup:** Prevents instability at start of training.

---

### Importance

- **Faster Convergence:** Adapts learning to training phase.
- **Better Generalization:** Reduces overfitting risk.

---

### Pros vs. Cons

| Method         | Pros                                    | Cons                                  |
|----------------|-----------------------------------------|---------------------------------------|
| Step/Exp Decay | Simple, effective                       | Hyperparameter tuning required        |
| Cosine         | Smooth, SOTA results                    | More complex implementation           |
| Warmup         | Stabilizes early training                | Adds scheduling complexity            |

---

### Recent Developments

- **One-cycle Policy:** Combines warmup, annealing, and decay for optimal results.
- **Adaptive Schedulers:** Adjust rates based on validation metrics.

---

### Best Practices & Pitfalls

- Tune schedule parameters for each task.
- Use warmup for very deep or transformer models.
- Monitor for learning rate plateaus or divergence.

---

## Training Pseudo-Algorithm (with Mathematical Justification)

```python
# Pseudo-algorithm for a single training epoch

for batch in data_loader:
    # Preprocessing
    x, y = preprocess(batch)  # e.g., normalization, augmentation

    # Forward pass
    y_pred = model(x; W, b)

    # Compute loss
    L = loss_fn(y_pred, y)

    # Backward pass
    g = ∇_{W,b} L

    # Gradient Clipping
    if ||g||_2 > τ:
        g = g * (τ / ||g||_2)

    # Parameter Update (SGD example)
    W = W - η_t * g_W
    b = b - η_t * g_b

    # Learning Rate Scheduling
    η_t = scheduler(t)
```

---

## Evaluation Metrics

### 1. **Loss Functions**

- **Cross-Entropy Loss (Classification):**
  - $$ L = -\sum_{i=1}^N y_i \log(\hat{y}_i) $$
- **MSE (Regression):**
  - $$ L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 $$

### 2. **Domain-Specific Metrics**

- **Accuracy:**
  - $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}} $$
- **F1 Score:**
  - $$ \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
- **AUC-ROC:**
  - Area under the ROC curve.

### 3. **SOTA Metrics**

- **Top-1/Top-5 Accuracy (ImageNet):**
- **BLEU, ROUGE (NLP):**
- **mAP (Object Detection):**

---

### Best Practices & Pitfalls

- Use validation set for metric computation.
- Monitor both loss and task-specific metrics.
- Beware of overfitting to metrics; use multiple metrics for robust evaluation.

---