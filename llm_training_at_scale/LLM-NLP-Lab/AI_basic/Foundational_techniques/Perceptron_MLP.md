
# Perceptron

## Definition

A **Perceptron** is the fundamental unit of a neural network, representing a binary linear classifier. It computes a weighted sum of input features, applies a bias, and passes the result through an activation function (typically the Heaviside step function).

---

## Mathematical Formulation

### Input and Output

Given input vector $ \mathbf{x} \in \mathbb{R}^n $, weight vector $ \mathbf{w} \in \mathbb{R}^n $, and bias $ b \in \mathbb{R} $:

- **Linear Combination:**
  $$
  z = \mathbf{w}^\top \mathbf{x} + b
  $$

- **Activation (Heaviside Step):**
  $$
  y = f(z) = 
  \begin{cases}
    1 & \text{if } z \geq 0 \\
    0 & \text{otherwise}
  \end{cases}
  $$

---

## Key Principles

- **Linearity:** The perceptron can only separate data that is linearly separable.
- **Binary Classification:** Outputs are binary (0 or 1).
- **Learning Rule:** Weights are updated using the perceptron learning algorithm.

---

## Detailed Concept Analysis

### Pre-processing

- **Feature Scaling:** Standardize or normalize input features to ensure stable convergence.
  $$
  x_i' = \frac{x_i - \mu_i}{\sigma_i}
  $$
  where $ \mu_i $ and $ \sigma_i $ are the mean and standard deviation of feature $ i $.

### Training Algorithm (Pseudo-code)

1. **Initialize** $ \mathbf{w} \leftarrow 0 $, $ b \leftarrow 0 $
2. **For** each epoch:
   - **For** each training sample $ (\mathbf{x}^{(i)}, y^{(i)}) $:
     - Compute $ z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b $
     - Predict $ \hat{y}^{(i)} = f(z^{(i)}) $
     - Update:
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
  (Not differentiable; updates are based on misclassifications.)

---

## Importance

- **Foundation of Neural Networks:** Basis for more complex architectures.
- **Efficient for Linearly Separable Data:** Fast convergence in such cases.

---

## Pros vs. Cons

- **Pros:**
  - Simple, interpretable.
  - Fast training for small datasets.

- **Cons:**
  - Cannot solve non-linearly separable problems (e.g., XOR).
  - Limited to binary classification.

---

## Recent Developments

- **Kernel Perceptron:** Extends to non-linear boundaries using kernel trick.
- **Online Learning:** Adaptations for streaming data.

---

## Evaluation Metrics

- **Accuracy:**
  $$
  \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} = y^{(i)})
  $$
- **Precision, Recall, F1-score:** Standard for binary classification.

---

# Multi-Layer Perceptron (MLP)

## Definition

A **Multi-Layer Perceptron (MLP)** is a feedforward artificial neural network with one or more hidden layers, enabling the modeling of non-linear relationships.

---

## Mathematical Formulation

### Architecture

- **Input Layer:** $ \mathbf{x} \in \mathbb{R}^{d_0} $
- **Hidden Layers:** $ L $ layers, each with $ d_l $ units.
- **Output Layer:** $ \mathbf{y} \in \mathbb{R}^{d_{L+1}} $

### Forward Pass

For layer $ l $ ($ l = 1, \ldots, L+1 $):

- **Linear Transformation:**
  $$
  \mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
  $$
  where $ \mathbf{a}^{(0)} = \mathbf{x} $

- **Activation:**
  $$
  \mathbf{a}^{(l)} = \phi^{(l)}(\mathbf{z}^{(l)})
  $$
  where $ \phi^{(l)} $ is the activation function (e.g., ReLU, sigmoid, tanh).

- **Output:**
  $$
  \hat{\mathbf{y}} = \mathbf{a}^{(L+1)}
  $$

---

## Key Principles

- **Universal Approximation:** MLPs with sufficient width can approximate any continuous function.
- **Non-linearity:** Hidden layers enable modeling of complex, non-linear relationships.
- **Backpropagation:** Gradient-based optimization for training.

---

## Detailed Concept Analysis

### Pre-processing

- **Feature Normalization:** Zero-mean, unit-variance scaling.
- **One-hot Encoding:** For categorical variables.

### Training Algorithm (Pseudo-code)

1. **Initialize** weights $ \mathbf{W}^{(l)} $, biases $ \mathbf{b}^{(l)} $ (e.g., Xavier/He initialization).
2. **For** each epoch:
   - **For** each batch:
     - **Forward Pass:** Compute $ \mathbf{a}^{(l)} $ for all layers.
     - **Compute Loss:** $ L(\hat{\mathbf{y}}, \mathbf{y}) $
     - **Backward Pass:** Compute gradients via backpropagation.
     - **Update Parameters:**
       $$
       \mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}
       $$
       $$
       \mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}
       $$

### Loss Functions

- **Binary Cross-Entropy (for binary classification):**
  $$
  L(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  $$
- **Categorical Cross-Entropy (for multi-class):**
  $$
  L(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{k=1}^K y_k \log(\hat{y}_k)
  $$
- **Mean Squared Error (for regression):**
  $$
  L(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2
  $$

### Post-Training Procedures

- **Regularization:** $ L_2 $ penalty:
  $$
  L_{\text{reg}} = L + \lambda \sum_{l} \|\mathbf{W}^{(l)}\|_2^2
  $$
- **Pruning/Quantization:** Reduce model size for deployment.
- **Ensembling:** Combine multiple MLPs for improved generalization.

---

## Importance

- **Versatility:** Applicable to classification, regression, and representation learning.
- **Foundation for Deep Learning:** Precursor to modern deep architectures.

---

## Pros vs. Cons

- **Pros:**
  - Can model non-linear relationships.
  - Flexible architecture.

- **Cons:**
  - Prone to overfitting without regularization.
  - Requires careful hyperparameter tuning.
  - Not ideal for spatial/temporal data (e.g., images, sequences) compared to CNNs/RNNs.

---

## Recent Developments

- **Residual Connections:** Improve gradient flow in deep MLPs.
- **Layer Normalization/Batch Normalization:** Stabilize training.
- **Advanced Optimizers:** Adam, RMSProp, etc.

---

## Evaluation Metrics

- **Classification:**
  - **Accuracy:** $ \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}^{(i)} = y^{(i)}) $
  - **Precision, Recall, F1-score:** Standard for imbalanced datasets.
  - **AUC-ROC:** Area under the ROC curve.
- **Regression:**
  - **MSE:** $ \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2 $
  - **MAE:** $ \frac{1}{N} \sum_{i=1}^N |\hat{y}^{(i)} - y^{(i)}| $

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use appropriate activation functions (ReLU for hidden, softmax/sigmoid for output).
  - Apply regularization and dropout to prevent overfitting.
  - Normalize inputs for stable training.
  - Monitor validation metrics to avoid overfitting.

- **Pitfalls:**
  - Overfitting with small datasets.
  - Vanishing/exploding gradients in deep MLPs.
  - Poor initialization can hinder convergence.

---