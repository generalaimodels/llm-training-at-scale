---

# Convolutional Neural Networks (CNNs): Comprehensive Technical Breakdown

---

## 1. **Definition**

A **Convolutional Neural Network (CNN)** is a class of deep neural networks primarily used for processing data with grid-like topology (e.g., images, audio, video). CNNs leverage convolutional layers to extract hierarchical features, enabling efficient learning of spatial and temporal dependencies.

---

## 2. **Core Model Architecture**

### 2.1 **Input Data Types**
- **1D CNN:** Sequential data (e.g., audio, time series)
- **2D CNN:** Images (height × width × channels)
- **3D CNN:** Video or volumetric data (depth × height × width × channels)

---

### 2.2 **Convolutional Layer**

#### **Mathematical Formulation**

- **1D Convolution:**
  $$
  y[i] = \sum_{k=0}^{K-1} x[i + k] \cdot w[k] + b
  $$
  - $x$: Input sequence
  - $w$: Kernel/filter of size $K$
  - $b$: Bias term

- **2D Convolution:**
  $$
  y[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x[i + m, j + n] \cdot w[m, n] + b
  $$
  - $x$: Input image
  - $w$: 2D kernel of size $M \times N$

- **3D Convolution:**
  $$
  y[i, j, k] = \sum_{d=0}^{D-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x[i + d, j + m, k + n] \cdot w[d, m, n] + b
  $$
  - $x$: Input volume
  - $w$: 3D kernel of size $D \times M \times N$

---

### 2.3 **Key Parameters**

- **Kernel Size ($K$, $M \times N$, $D \times M \times N$):** Size of the convolutional filter.
- **Stride ($S$):** Step size for moving the kernel.
- **Padding ($P$):** Number of pixels/values added to the input borders.
- **Dilation ($D$):** Spacing between kernel elements.
- **Number of Filters ($F$):** Number of output channels.

---

### 2.4 **Output Shape Calculation**

#### **General Formula (for each dimension):**
$$
O = \left\lfloor \frac{I + 2P - D \cdot (K - 1) - 1}{S} + 1 \right\rfloor
$$
- $O$: Output size
- $I$: Input size
- $P$: Padding
- $K$: Kernel size
- $S$: Stride
- $D$: Dilation

#### **Example (2D):**
$$
O_{height} = \left\lfloor \frac{I_{height} + 2P_{height} - D_{height} \cdot (K_{height} - 1) - 1}{S_{height}} + 1 \right\rfloor
$$

---

### 2.5 **Pooling Layer**

#### **Max Pooling Equation (2D):**
$$
y[i, j] = \max_{0 \leq m < M, 0 \leq n < N} x[S \cdot i + m, S \cdot j + n]
$$
- $M \times N$: Pool size
- $S$: Stride

#### **Output Shape (Pooling):**
$$
O = \left\lfloor \frac{I - K}{S} + 1 \right\rfloor
$$
- $K$: Pool size

---

### 2.6 **Padding Types**

- **Valid:** No padding ($P=0$)
- **Same:** Padding such that output size equals input size
  $$
  P = \left\lfloor \frac{(S-1) \cdot I - S + K}{2} \right\rfloor
  $$

---

### 2.7 **Activation Functions**

- **ReLU:** $f(x) = \max(0, x)$
- **Leaky ReLU:** $f(x) = \max(\alpha x, x)$, $\alpha \ll 1$
- **Softmax (for classification):**
  $$
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
  $$

---

## 3. **Pre-processing Steps**

- **Normalization:** $x' = \frac{x - \mu}{\sigma}$
- **Resizing:** Interpolation to standard input size
- **Augmentation:** Random crop, flip, rotation, color jitter

---

## 4. **Training Procedure**

### 4.1 **Loss Functions**

- **Cross-Entropy Loss (Classification):**
  $$
  \mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
  $$
  - $C$: Number of classes
  - $y_i$: Ground truth (one-hot)
  - $\hat{y}_i$: Predicted probability

- **MSE Loss (Regression):**
  $$
  \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
  $$

---

### 4.2 **Optimization**

- **SGD Update:**
  $$
  \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
  $$
  - $\eta$: Learning rate

- **Adam Update:**
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
  \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
  \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$

---

### 4.3 **Training Pseudo-Algorithm (PyTorch-like)**

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        x = preprocess(x)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. **Post-Training Procedures**

- **Quantization:** Reducing precision of weights/activations
- **Pruning:** Removing redundant weights
- **Knowledge Distillation:** Training a smaller model using teacher outputs

---

## 6. **Evaluation Phase**

### 6.1 **Metrics (SOTA & Domain-Specific)**

- **Accuracy:**
  $$
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
  $$

- **Precision:**
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **Recall:**
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

- **F1 Score:**
  $$
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

- **IoU (Intersection over Union, for segmentation):**
  $$
  IoU = \frac{|A \cap B|}{|A \cup B|}
  $$

- **AUC-ROC:**
  $$
  \text{AUC} = \int_{0}^{1} TPR(FPR^{-1}(x)) dx
  $$

---

## 7. **Best Practices & Pitfalls**

### **Best Practices**
- Use batch normalization for stable training.
- Employ dropout to mitigate overfitting.
- Monitor validation metrics to prevent overfitting.
- Use data augmentation for generalization.

### **Pitfalls**
- Incorrect padding/stride can cause shape mismatches.
- Overfitting due to excessive model capacity.
- Vanishing/exploding gradients in deep networks.

---

## 8. **Recent Developments**

- **Depthwise Separable Convolutions (e.g., MobileNet):** Reduces computation.
- **Dilated Convolutions:** Expands receptive field without increasing parameters.
- **Residual Connections (ResNet):** Mitigates vanishing gradients.
- **Attention Mechanisms in CNNs:** Improves feature selection.

---

## 9. **Parameter Calculation Example**

### **2D Convolutional Layer Parameters:**
$$
\text{Params} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
$$
- $K_h, K_w$: Kernel height/width
- $C_{in}$: Input channels
- $C_{out}$: Output channels

---

## 10. **Implementation (PyTorch Example)**

```python
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3, 
    out_channels=64, 
    kernel_size=3, 
    stride=1, 
    padding=1
)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

---

**All variables are defined in context. For further domain-specific metrics or advanced architectures, refer to the latest literature and framework documentation.**