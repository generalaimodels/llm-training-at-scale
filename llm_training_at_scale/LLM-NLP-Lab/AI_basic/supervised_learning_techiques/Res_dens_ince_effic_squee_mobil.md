
# Residual Networks (ResNet)

## Definition
Residual Networks (ResNet) are deep convolutional neural network (CNN) architectures that utilize *residual connections* (skip connections) to enable the training of extremely deep networks by mitigating the vanishing gradient problem.

---

## Mathematical Equations

- **Residual Block Output:**
  $$
  \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
  $$
  - $\mathbf{x}$: Input to the block
  - $\mathcal{F}$: Residual function (e.g., two stacked convolutional layers)
  - $\{W_i\}$: Weights of the layers in the block
  - $\mathbf{y}$: Output of the block

---

## Pseudo-Algorithm

1. **Input:** $\mathbf{x}$ (input tensor)
2. **Compute Residual:** $\mathbf{r} = \mathcal{F}(\mathbf{x}, \{W_i\})$
3. **Add Skip Connection:** $\mathbf{y} = \mathbf{r} + \mathbf{x}$
4. **Apply Activation (optional):** $\mathbf{y} = \sigma(\mathbf{y})$
5. **Output:** $\mathbf{y}$

---

## Principles and Mechanisms

- **Skip Connections:** Directly add input $\mathbf{x}$ to the output of a stack of layers, allowing gradients to flow unimpeded.
- **Identity Mapping:** If dimensions differ, use a linear projection (e.g., $1 \times 1$ convolution) to match dimensions.
- **Mitigates Degradation:** Prevents accuracy degradation in very deep networks.

---

## Significance and Use Cases

- **Significance:** Enabled the training of networks with hundreds or thousands of layers, leading to state-of-the-art results in image classification, detection, and segmentation.
- **Use Cases:** ImageNet classification, object detection (e.g., Faster R-CNN), semantic segmentation (e.g., DeepLab).

---

## Pros vs. Cons

- **Pros:**
  - Enables very deep architectures
  - Alleviates vanishing/exploding gradient issues
  - Simple to implement
- **Cons:**
  - Increased memory usage due to skip connections
  - Diminishing returns with extreme depth

---

## Variants and Recent Developments

- **ResNeXt:** Aggregated residual transformations (grouped convolutions)
- **Wide ResNet:** Increased width instead of depth
- **Pre-activation ResNet:** BatchNorm and ReLU before convolutions

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use batch normalization after each convolution
  - Employ identity mapping when possible
- **Pitfalls:**
  - Mismatched dimensions in skip connections
  - Overfitting with excessive depth

---

# DenseNet

## Definition
DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer in a feed-forward fashion, enhancing feature reuse and gradient flow.

---

## Mathematical Equations

- **Dense Block Output:**
  $$
  \mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])
  $$
  - $H_l$: Composite function (BN, ReLU, Conv)
  - $[\cdot]$: Concatenation operation

---

## Pseudo-Algorithm

1. **Input:** $\mathbf{x}_0$
2. **For** $l = 1$ to $L$:
   - Concatenate all previous outputs: $\mathbf{z}_l = [\mathbf{x}_0, ..., \mathbf{x}_{l-1}]$
   - Compute: $\mathbf{x}_l = H_l(\mathbf{z}_l)$
3. **Output:** $\mathbf{x}_L$

---

## Principles and Mechanisms

- **Dense Connectivity:** Each layer receives all preceding feature maps as input.
- **Feature Reuse:** Promotes efficient parameter usage and mitigates vanishing gradients.

---

## Significance and Use Cases

- **Significance:** Achieves high accuracy with fewer parameters; efficient feature propagation.
- **Use Cases:** Image classification, medical imaging, object detection.

---

## Pros vs. Cons

- **Pros:**
  - Improved gradient flow
  - Parameter efficiency
  - Implicit deep supervision
- **Cons:**
  - High memory/computation cost due to concatenation
  - Slower inference

---

## Variants and Recent Developments

- **DenseNet-BC:** Bottleneck and compression layers for efficiency
- **3D DenseNet:** For volumetric data

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use bottleneck layers to reduce computation
  - Apply transition layers to control feature map size
- **Pitfalls:**
  - Memory bottlenecks with large input sizes

---

# Inception Networks

## Definition
Inception Networks utilize parallel convolutional filters of varying sizes within the same layer, enabling multi-scale feature extraction.

---

## Mathematical Equations

- **Inception Module Output:**
  $$
  \mathbf{y} = [f_1(\mathbf{x}), f_3(\mathbf{x}), f_5(\mathbf{x}), p(\mathbf{x})]
  $$
  - $f_k$: $k \times k$ convolution
  - $p$: pooling operation
  - $[\cdot]$: Concatenation

---

## Pseudo-Algorithm

1. **Input:** $\mathbf{x}$
2. **Apply parallel convolutions:** $f_1(\mathbf{x}), f_3(\mathbf{x}), f_5(\mathbf{x})$
3. **Apply pooling:** $p(\mathbf{x})$
4. **Concatenate outputs:** $\mathbf{y} = [f_1, f_3, f_5, p]$
5. **Output:** $\mathbf{y}$

---

## Principles and Mechanisms

- **Multi-scale Processing:** Simultaneous extraction of features at different scales.
- **Dimensionality Reduction:** $1 \times 1$ convolutions reduce computation.

---

## Significance and Use Cases

- **Significance:** Efficiently increases network width and depth; state-of-the-art in ILSVRC.
- **Use Cases:** Image classification, object detection, video analysis.

---

## Pros vs. Cons

- **Pros:**
  - Multi-scale feature extraction
  - Computational efficiency via $1 \times 1$ convolutions
- **Cons:**
  - Complex architecture
  - Manual design of modules

---

## Variants and Recent Developments

- **Inception-v2/v3:** Factorized convolutions, batch normalization
- **Inception-v4/ResNet:** Hybrid with residual connections

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use $1 \times 1$ convolutions for dimensionality reduction
  - Balance filter sizes for target application
- **Pitfalls:**
  - Overly complex modules can hinder optimization

---

# EfficientNet

## Definition
EfficientNet is a family of models that scale network depth, width, and resolution using a compound scaling method for optimal accuracy and efficiency.

---

## Mathematical Equations

- **Compound Scaling:**
  $$
  \begin{align*}
  \text{depth:} & \quad d = \alpha^\phi \\
  \text{width:} & \quad w = \beta^\phi \\
  \text{resolution:} & \quad r = \gamma^\phi \\
  \text{subject to:} & \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha, \beta, \gamma > 0
  \end{align*}
  $$
  - $\phi$: Compound coefficient
  - $\alpha, \beta, \gamma$: Scaling constants

---

## Pseudo-Algorithm

1. **Input:** Baseline network, scaling coefficient $\phi$
2. **Scale depth, width, resolution:** Apply compound scaling equations
3. **Construct scaled network**
4. **Train and evaluate**

---

## Principles and Mechanisms

- **Compound Scaling:** Jointly scales all dimensions for balanced model growth.
- **MBConv Blocks:** Mobile inverted bottleneck convolution with squeeze-and-excitation.

---

## Significance and Use Cases

- **Significance:** State-of-the-art accuracy/efficiency trade-off; widely used in mobile and cloud applications.
- **Use Cases:** Image classification, transfer learning, edge deployment.

---

## Pros vs. Cons

- **Pros:**
  - Superior accuracy-efficiency trade-off
  - Scalable to various resource constraints
- **Cons:**
  - Requires careful tuning of scaling coefficients
  - More complex training pipeline

---

## Variants and Recent Developments

- **EfficientNetV2:** Faster training, improved parameter efficiency
- **Noisy Student Training:** Semi-supervised learning for further gains

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use pre-trained weights for transfer learning
  - Select appropriate $\phi$ for hardware constraints
- **Pitfalls:**
  - Over-scaling can lead to diminishing returns

---

# SqueezeNet

## Definition
SqueezeNet is a lightweight CNN architecture that achieves AlexNet-level accuracy with 50x fewer parameters, primarily via "fire modules."

---

## Mathematical Equations

- **Fire Module:**
  $$
  \mathbf{y} = [\text{Conv}_{1 \times 1}(\mathbf{x}), \text{Conv}_{3 \times 3}(\mathbf{x})]
  $$
  - Squeeze: $1 \times 1$ conv
  - Expand: $1 \times 1$ and $3 \times 3$ conv

---

## Pseudo-Algorithm

1. **Input:** $\mathbf{x}$
2. **Squeeze:** $s = \text{Conv}_{1 \times 1}(\mathbf{x})$
3. **Expand:** $e_1 = \text{Conv}_{1 \times 1}(s)$, $e_3 = \text{Conv}_{3 \times 3}(s)$
4. **Concatenate:** $\mathbf{y} = [e_1, e_3]$
5. **Output:** $\mathbf{y}$

---

## Principles and Mechanisms

- **Parameter Reduction:** Replace $3 \times 3$ convs with $1 \times 1$ convs where possible.
- **Fire Modules:** Squeeze (reduce channels), then expand (increase channels).

---

## Significance and Use Cases

- **Significance:** Enables deployment on resource-constrained devices.
- **Use Cases:** Mobile/embedded vision, IoT, robotics.

---

## Pros vs. Cons

- **Pros:**
  - Extremely small model size
  - Competitive accuracy
- **Cons:**
  - Lower accuracy than larger models
  - Limited scalability

---

## Variants and Recent Developments

- **SqueezeNet 1.1:** Faster, smaller model
- **SqueezeNext:** Further parameter reduction

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use for memory-constrained applications
  - Combine with quantization/pruning
- **Pitfalls:**
  - Not suitable for high-accuracy requirements

---

# MobileNet

## Definition
MobileNet is a family of efficient CNN architectures optimized for mobile and embedded vision applications, primarily using depthwise separable convolutions.

---

## Mathematical Equations

- **Depthwise Separable Convolution:**
  $$
  \text{Cost}_{\text{standard}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
  $$
  $$
  \text{Cost}_{\text{depthwise}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F
  $$
  - $D_K$: Kernel size
  - $M$: Input channels
  - $N$: Output channels
  - $D_F$: Feature map size

---

## Pseudo-Algorithm

1. **Input:** $\mathbf{x}$
2. **Depthwise Conv:** Apply $D_K \times D_K$ conv to each channel
3. **Pointwise Conv:** Apply $1 \times 1$ conv to combine channels
4. **Output:** Feature map

---

## Principles and Mechanisms

- **Depthwise Separable Convolutions:** Factorize standard conv into depthwise and pointwise steps, reducing computation.
- **Width/Resolution Multipliers:** Control model size and computation.

---

## Significance and Use Cases

- **Significance:** Enables real-time inference on mobile/edge devices.
- **Use Cases:** Mobile vision, AR/VR, robotics.

---

## Pros vs. Cons

- **Pros:**
  - Highly efficient
  - Tunable for resource constraints
- **Cons:**
  - Lower accuracy than larger models
  - Sensitive to quantization

---

## Variants and Recent Developments

- **MobileNetV2:** Inverted residuals, linear bottlenecks
- **MobileNetV3:** NAS-optimized, squeeze-and-excitation

---

## Best Practices and Pitfalls

- **Best Practices:**
  - Use width/resolution multipliers for target hardware
  - Combine with quantization for further efficiency
- **Pitfalls:**
  - Over-reduction of width/resolution degrades accuracy

---