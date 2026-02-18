
# Perceptron

## Definition

- The **Perceptron** is a single-layer, feedforward neural unit functioning as a linear binary classifier. It computes a weighted sum of input features, adds a bias, and applies a threshold activation.

---

## Pertinent Equations

- **Input:** $ \mathbf{x} \in \mathbb{R}^n $
- **Weights:** $ \mathbf{w} \in \mathbb{R}^n $
- **Bias:** $ b \in \mathbb{R} $
- **Linear Combination:**  
  $$
  z = \mathbf{w}^\top \mathbf{x} + b
  $$
- **Activation (Heaviside Step):**  
  $$
  \hat{y} = f(z) = 
  \begin{cases}
    1 & \text{if } z \geq 0 \\
    0 & \text{if } z < 0
  \end{cases}
  $$

---

## Key Principles

- **Linear Separability:** Only classifies data that is linearly separable.
- **Binary Output:** $ \hat{y} \in \{0, 1\} $.
- **Iterative Weight Update:** Adjusts weights based on misclassifications.

---

## Detailed Concept Analysis

### Pre-processing

- **Feature Normalization:**  
  $$
  x_i' = \frac{x_i - \mu_i}{\sigma_i}
  $$
  where $ \mu_i $ and $ \sigma_i $ are the mean and standard deviation of feature $ i $.

### Model Construction

- **Parameter Initialization:**  
  $$
  \mathbf{w} \sim \mathcal{N}(0, \sigma^2), \quad b = 0
  $$

### Training Algorithm (Pseudo-code)

1. **Initialize** $ \mathbf{w}, b $
2. **For** each epoch:
   - **For** each sample $ (\mathbf{x}^{(i)}, y^{(i)}) $:
     - $ z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b $
     - $ \hat{y}^{(i)} = f(z^{(i)}) $
     - **Update:**
       $$
       \mathbf{w} \leftarrow \mathbf{w} + \eta (y^{(i)} - \hat{y}^{(i)}) \mathbf{x}^{(i)}
       $$
       $$
       b \leftarrow b + \eta (y^{(i)} - \hat{y}^{(i)})
       $$
     - $ \eta $: learning rate

### Loss Function

- **Perceptron Loss:**  
  $$
  L(\mathbf{w}, b) = -\sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)}) z^{(i)}
  $$
  (Used for update logic, not for gradient-based optimization.)

### Post-Training

- **No explicit post-training;** model is ready after convergence.

---

## Importance

- **Foundation for neural networks.**
- **Efficient for linearly separable problems.**

---

## Pros vs. Cons

- **Pros:**
  - Simple, interpretable.
  - Fast convergence for linearly separable data.
- **Cons:**
  - Cannot solve non-linear problems (e.g., XOR).
  - Limited to binary classification.

---

## Cutting-Edge Advances

- **Kernel Perceptron:**  
  $$
  f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^N \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b\right)
  $$
  where $ K $ is a kernel function.
- **Online/Incremental Perceptron:** Adapted for streaming data.

---

## Evaluation Metrics

- **Accuracy:**  
  $$
  \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} = y^{(i)})
  $$
- **Precision:**  
  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$
- **Recall:**  
  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
- **F1-score:**  
  $$
  \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

---

## Best Practices & Pitfalls

- **Best Practices:**
  - Normalize features.
  - Shuffle data each epoch.
- **Pitfalls:**
  - Fails on non-linearly separable data.
  - Sensitive to feature scaling.

---

# Multi-Layer Perceptron (MLP)

## Definition

- The **Multi-Layer Perceptron (MLP)** is a feedforward neural network with one or more hidden layers, enabling the modeling of non-linear functions.

---

## Pertinent Equations

- **Input:** $ \mathbf{x} \in \mathbb{R}^{d_0} $
- **Hidden Layers:** $ L $ layers, each with $ d_l $ units.
- **Layer $ l $ Linear Transformation:**  
  $$
  \mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
  $$
- **Activation:**  
  $$
  \mathbf{a}^{(l)} = \phi^{(l)}(\mathbf{z}^{(l)})
  $$
- **Output:**  
  $$
  \hat{\mathbf{y}} = \mathbf{a}^{(L+1)}
  $$

---

## Key Principles

- **Universal Approximation:** Can approximate any continuous function with sufficient width.
- **Non-linearity:** Hidden layers with non-linear activations.
- **Backpropagation:** Gradient-based optimization.

---

## Detailed Concept Analysis

### Pre-processing

- **Feature Normalization:**  
  $$
  x_i' = \frac{x_i - \mu_i}{\sigma_i}
  $$
- **One-hot Encoding:** For categorical variables.

### Model Construction

- **Parameter Initialization:**  
  $$
  \mathbf{W}^{(l)} \sim \mathcal{N}(0, \frac{2}{d_{l-1}})
  $$
  (He initialization for ReLU activations.)

### Training Algorithm (Pseudo-code)

1. **Initialize** $ \mathbf{W}^{(l)}, \mathbf{b}^{(l)} $
2. **For** each epoch:
   - **For** each batch:
     - **Forward Pass:**  
       $ \mathbf{a}^{(l)} = \phi^{(l)}(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}) $
     - **Compute Loss:** $ L(\hat{\mathbf{y}}, \mathbf{y}) $
     - **Backward Pass:**  
       Compute gradients $ \frac{\partial L}{\partial \mathbf{W}^{(l)}}, \frac{\partial L}{\partial \mathbf{b}^{(l)}} $
     - **Update Parameters:**  
       $$
       \mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}
       $$
       $$
       \mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}
       $$

### Loss Functions

- **Binary Cross-Entropy:**  
  $$
  L(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  $$
- **Categorical Cross-Entropy:**  
  $$
  L(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{k=1}^K y_k \log(\hat{y}_k)
  $$
- **Mean Squared Error:**  
  $$
  L(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2
  $$

### Post-Training Procedures

- **Regularization:**  
  $$
  L_{\text{reg}} = L + \lambda \sum_{l} \|\mathbf{W}^{(l)}\|_2^2
  $$
- **Pruning/Quantization:** Reduce model size.
- **Ensembling:** Combine multiple MLPs.

---

## Importance

- **Versatile:** Handles classification, regression, and representation learning.
- **Foundation for deep learning architectures.**

---

## Pros vs. Cons

- **Pros:**
  - Models non-linear relationships.
  - Flexible architecture.
- **Cons:**
  - Prone to overfitting.
  - Requires careful hyperparameter tuning.
  - Not optimal for spatial/temporal data.

---

## Cutting-Edge Advances

- **Residual Connections:**  
  $$
  \mathbf{a}^{(l)} = \phi^{(l)}(\mathbf{z}^{(l)}) + \mathbf{a}^{(l-1)}
  $$
- **Layer/Batch Normalization:**  
  $$
  \text{BN}(\mathbf{z}) = \gamma \frac{\mathbf{z} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
  $$
- **Advanced Optimizers:** Adam, RMSProp.

---

## Evaluation Metrics

- **Classification:**
  - **Accuracy:**  
    $$
    \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} = y^{(i)})
    $$
  - **Precision, Recall, F1-score, AUC-ROC.**
- **Regression:**
  - **MSE:**  
    $$
    \text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2
    $$
  - **MAE:**  
    $$
    \text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}^{(i)} - y^{(i)}|
    $$

---

## Best Practices & Pitfalls

- **Best Practices:**
  - Normalize inputs.
  - Use dropout/regularization.
  - Monitor validation metrics.
- **Pitfalls:**
  - Overfitting on small data.
  - Vanishing/exploding gradients.
  - Poor initialization.

---