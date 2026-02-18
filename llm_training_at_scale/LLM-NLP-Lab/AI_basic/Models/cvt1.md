### CvT (Convolutional vision Transformer)

#### 1. Model Architecture

##### 1.1. Definition
The Convolutional vision Transformer (CvT) introduces convolutions into the Vision Transformer (ViT) architecture to leverage the benefits of both CNNs (desirable properties like shift, scale, and distortion invariance) and Transformers (dynamic attention, global context, and better generalization). CvT achieves this through two primary modifications: a hierarchy of Transformers containing Convolutional Token Embedding, and Convolutional Projection for the Query, Key, and Value matrices within the self-attention mechanism.

##### 1.2. Core Components

###### 1.2.1. Convolutional Token Embedding
*   **Definition:** This process replaces the standard ViT's patch flattening and linear projection with a convolutional layer. It aims to model local spatial context from the early stages and allows for overlapping "patches" or tokens, providing richer local information. This is applied at the beginning of each stage.
*   **Pertinent Equations:**
    Let $X_{s-1} \in \mathbb{R}^{H_{s-1} \times W_{s-1} \times C_{s-1}}$ be the output feature map from the previous stage (or the input image for $s=1$). The Convolutional Token Embedding for stage $s$ is:
    $$ X_{s, token\_embed} = \text{Conv2d}(X_{s-1}; K_s, S_s, P_s, C_s) $$
    Where:
    *   $K_s$: Kernel size for the convolution at stage $s$.
    *   $S_s$: Stride for the convolution at stage $s$.
    *   $P_s$: Padding for the convolution at stage $s$.
    *   $C_s$: Number of output channels for the convolution at stage $s$.
    The output $X_{s, token\_embed} \in \mathbb{R}^{H_s \times W_s \times C_s}$ is then typically followed by a Layer Normalization and reshaped into a sequence of tokens.
    $$ T_s = \text{Reshape}(\text{LayerNorm}(X_{s, token\_embed})) $$
    So, $T_s \in \mathbb{R}^{N_s \times C_s}$, where $N_s = H_s \times W_s$.
*   **Key Principles:**
    *   **Local Context Modeling:** Convolutions inherently capture local spatial relationships.
    *   **Overlapping Patches:** Strided convolutions can naturally create overlapping receptive fields for tokens, unlike the disjoint patches in ViT.
    *   **Hierarchical Feature Learning:** Applied at each stage, it allows for progressively coarser tokenization with richer semantic information.
*   **Detailed Concept Analysis:**
    In standard ViT, image patches are independently projected. Convolutional Token Embedding introduces local spatial dependencies early on by processing neighboring pixels through convolutional filters before they are treated as tokens. This helps the model learn low-level features more effectively. For stage 1, $X_0$ is the input image. For subsequent stages ($s > 1$), $X_{s-1}$ is the reshaped output from the previous stage's Transformer blocks. The convolution operation effectively performs both patch embedding and patch merging (downsampling) if $S_s > 1$.
    For example, in PyTorch, this would be implemented using `nn.Conv2d` followed by `nn.LayerNorm` and reshaping.
    ```python
    # Illustrative PyTorch-like snippet
    # self.token_embedding = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    # x = self.token_embedding(input_feature_map)
    # x = x.flatten(2).transpose(1, 2) # (B, C, H, W) -> (B, H*W, C)
    # x = self.norm_layer(x) # LayerNorm typically applied after reshaping for Transformers
    ```
    In CvT, the LayerNorm might be applied before flattening for consistency with CNN practices or after flattening as per Transformer norms depending on specific implementation choices aiming for optimal performance. The original paper applies it on the 2D feature map before flattening.

###### 1.2.2. Convolutional Projection for Query, Key, Value
*   **Definition:** Instead of using a simple linear projection (1x1 convolution equivalent) to generate Query ($Q$), Key ($K$), and Value ($V$) matrices from input tokens for the self-attention mechanism, CvT employs depth-wise separable convolutions. This allows the projection process itself to model local spatial context and potentially reduce parameters.
*   **Pertinent Equations:**
    Given input tokens (feature map from the previous layer, reshaped to 2D spatial form) $X_{att} \in \mathbb{R}^{H' \times W' \times C_{in}}$ for an attention head:
    The Query ($Q$), Key ($K$), and Value ($V$) are generated as:
    $$ Q = \text{ConvProj}_Q(X_{att}) = \text{DWConv2d}_Q(\text{PointwiseConv2d}_Q(X_{att}); S_Q, K_Q) $$
    $$ K = \text{ConvProj}_K(X_{att}) = \text{DWConv2d}_K(\text{PointwiseConv2d}_K(X_{att}); S_K, K_K) $$
    $$ V = \text{ConvProj}_V(X_{att}) = \text{DWConv2d}_V(\text{PointwiseConv2d}_V(X_{att}); S_V, K_V) $$
    Where:
    *   $\text{PointwiseConv2d}$ is a $1 \times 1$ convolution.
    *   $\text{DWConv2d}$ is a depth-wise convolution with kernel size $K_X$ and stride $S_X$ (where $X \in \{Q, K, V\}$).
    *   The output of $\text{ConvProj}_X$ is reshaped into a sequence for attention: $Q \in \mathbb{R}^{N_Q \times D_h}$, $K \in \mathbb{R}^{N_K \times D_h}$, $V \in \mathbb{R}^{N_V \times D_h}$, where $D_h$ is the dimension per head. The sequence lengths $N_Q, N_K, N_V$ can differ if strides $S_Q, S_K, S_V$ are greater than 1, effectively downsampling the key/value sequences.
*   **Key Principles:**
    *   **Spatial Context in Projections:** Allows attention to be conditioned on local neighborhood information even before the dot-product attention.
    *   **Efficiency:** Depth-wise separable convolutions are more parameter-efficient than standard convolutions for the same receptive field.
    *   **Sequence Length Reduction (Optional):** Using a stride $S_K > 1$ or $S_V > 1$ can reduce the number of key/value tokens, decreasing computational cost in self-attention, especially for high-resolution inputs.
*   **Detailed Concept Analysis:**
    This is a key innovation. Standard Transformers use linear layers (position-wise feed-forward networks) for $Q,K,V$ projections. By using depth-wise separable convolutions, CvT can capture local spatial patterns in the tokens that are being transformed into $Q, K, V$. For instance, for $K$ and $V$, using a stride of 2 for their convolutional projections halves their spatial dimensions, reducing the $N_K \times N_V$ complexity in the attention matrix computation. This is akin to how pooling or strided convolutions reduce feature map sizes in CNNs.
    PyTorch implementation would involve `nn.Conv2d` with `groups=in_channels` for depth-wise, and `nn.Conv2d` with `kernel_size=1` for pointwise, typically bundled.
    ```python
    # Illustrative PyTorch-like snippet
    # self.q_conv = nn.Conv2d(in_dim, head_dim, kernel_size_q, stride_q, padding_q, groups=groups_q, bias=False) # Potentially depth-wise separable
    # self.k_conv = nn.Conv2d(in_dim, head_dim, kernel_size_k, stride_k, padding_k, groups=groups_k, bias=False)
    # self.v_conv = nn.Conv2d(in_dim, head_dim, kernel_size_v, stride_v, padding_v, groups=groups_v, bias=False)
    # q = self.q_conv(x_reshaped_spatial).flatten(2).transpose(1,2) # (B, H*W, C_head)
    ```
    The input $X_{att}$ is the output of LayerNorm (from the previous block or token embedding), reshaped from $B \times N \times C_{in}$ to $B \times C_{in} \times H' \times W'$.

###### 1.2.3. Self-Attention with Convolutional Projections
*   **Definition:** Standard multi-head self-attention (MHSA) mechanism, but utilizing the $Q, K, V$ generated by the Convolutional Projections.
*   **Pertinent Equations:**
    For a single head, with $Q \in \mathbb{R}^{N_Q \times d_k}$, $K \in \mathbb{R}^{N_K \times d_k}$, $V \in \mathbb{R}^{N_V \times d_v}$ (where $d_k = d_v = D_h$, the dimension per head), and assuming $N_K = N_V$:
    $$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
    The output is then concatenated across heads and linearly projected:
    $$ \text{MHSA}(X_{att}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O $$
    Where $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$, and $W_O$ is the output projection matrix.
*   **Key Principles:**
    *   **Global Context:** Retains the Transformer's ability to model long-range dependencies.
    *   **Dynamic Weighting:** Attention scores are dynamically computed based on input.
*   **Detailed Concept Analysis:**
    The core attention mechanism remains the scaled dot-product attention. The novelty lies in *how* $Q, K, V$ are derived. The spatial information captured by the convolutional projections for $Q, K, V$ allows the attention mechanism to implicitly consider local structure when determining global relationships. If $S_K > 1$ or $S_V > 1$, then $N_K < N_Q$ and $N_V < N_Q$, which makes the attention map $A \in \mathbb{R}^{N_Q \times N_K}$. This is handled by ensuring matrix multiplications are conformant.

###### 1.2.4. CvT Block
*   **Definition:** A standard Transformer block structure, incorporating the modified self-attention.
*   **Pertinent Equations:**
    Let $Z_{l-1} \in \mathbb{R}^{N \times C}$ be the input sequence to the $l$-th CvT block.
    1.  Layer Normalization:
        $$ Z'_{l-1} = \text{LayerNorm}(Z_{l-1}) $$
    2.  Multi-Head Self-Attention (with Convolutional Projection):
        Input $Z'_{l-1}$ is reshaped to spatial form $X_{att}$ for ConvProjs.
        $$ A_l = \text{MHSA}_{\text{ConvProj}}(Z'_{l-1}) $$
    3.  Residual Connection:
        $$ Z''_l = A_l + Z_{l-1} $$
    4.  Layer Normalization:
        $$ Z'''_l = \text{LayerNorm}(Z''_l) $$
    5.  MLP (Feed-Forward Network):
        $$ F_l = \text{Linear}(\text{GELU}(\text{Linear}(Z'''_l))) $$
        Typically, the MLP expands the dimension $C$ by a factor (e.g., 4) and then projects it back.
        $W_1 \in \mathbb{R}^{C \times C_{mlp}}$, $b_1 \in \mathbb{R}^{C_{mlp}}$, $W_2 \in \mathbb{R}^{C_{mlp} \times C}$, $b_2 \in \mathbb{R}^{C}$.
        $$ \text{MLP}(X) = (XW_1 + b_1)\text{GELU} W_2 + b_2 $$
        (Often implemented as two `nn.Linear` layers.)
    6.  Residual Connection:
        $$ Z_l = F_l + Z''_l $$
    $Z_l$ is the output of the $l$-th CvT block.
*   **Key Principles:**
    *   **Residual Connections:** Essential for training deep networks by mitigating vanishing gradients.
    *   **Layer Normalization:** Stabilizes training and feature distributions.
*   **Detailed Concept Analysis:**
    This structure is largely similar to a standard ViT block. The critical difference is within the MHSA component, which uses convolutional projections. The sequence of operations (Norm -> Attention -> Add -> Norm -> MLP -> Add) is standard.
    In PyTorch, this is a module composed of sub-modules for MHSA (customized), MLP (`nn.Linear`, `nn.GELU`), and `nn.LayerNorm`.

###### 1.2.5. Hierarchical Stages & Overall Architecture
*   **Definition:** CvT is typically organized into multiple stages (e.g., 3 stages). Each stage consists of a Convolutional Token Embedding layer followed by a stack of CvT blocks. The token embedding layer of later stages also performs spatial downsampling.
*   **Pertinent Equations & Structure:**
    **Stage $s$ (for $s=1, \dots, S_{total}$):**
    1.  **Input:** $X_{s-1}$ (output feature map from stage $s-1$, or input image $X_0$ for $s=1$).
    2.  **Convolutional Token Embedding:**
        $X_{s, token\_embed} = \text{Conv2d}(X_{s-1}; K_s, S_s, P_s, C_s)$
        $T_s = \text{Reshape}(\text{LayerNorm}(X_{s, token\_embed}))$
        where $T_s \in \mathbb{R}^{N_s \times C_s}$.
    3.  **CvT Blocks:**
        $Z_{s,0} = T_s$
        For $l = 1, \dots, L_s$ (number of CvT blocks in stage $s$):
        $Z_{s,l} = \text{CvTBlock}_l(Z_{s,l-1})$
    4.  **Output of Stage $s$:** $Z_{s, L_s}$. This is reshaped back to a 2D spatial form $X_s \in \mathbb{R}^{H_s \times W_s \times C_s}$ to serve as input to the next stage's Convolutional Token Embedding. $H_s \times W_s = N_s$.

    **Final Classification Head:**
    After the final stage $S_{total}$:
    1.  The output sequence $Z_{S_{total}, L_{S_{total}}} \in \mathbb{R}^{N_{S_{total}} \times C_{S_{total}}}$ is processed.
    2.  Often, a global average pooling is applied over the sequence dimension, or the representation of a special [CLS] token (if used, though CvT often avoids it for image classification by pooling) is taken.
        $$ Z_{pool} = \text{GlobalAvgPool}(Z_{S_{total}, L_{S_{total}}}) \in \mathbb{R}^{C_{S_{total}}} $$
        (Pooling over the $N_{S_{total}}$ dimension).
    3.  A final linear layer (classifier) maps $Z_{pool}$ to class scores:
        $$ \text{Logits} = \text{Linear}(Z_{pool}) \in \mathbb{R}^{N_{classes}} $$
*   **Key Principles:**
    *   **Pyramidal Structure:** Feature maps decrease in spatial resolution and increase in channel depth through stages, similar to CNNs.
    *   **Multi-scale Feature Learning:** Different stages capture features at different scales.
*   **Detailed Concept Analysis:**
    This hierarchical design allows CvT to process information at multiple resolutions. Stage 1 might operate on fine-grained tokens from high-resolution input, capturing low-level details. Subsequent stages operate on coarser, more semantically rich tokens derived from downsampled feature maps of previous stages. This structure is more aligned with conventional CNN architectures like ResNet and is known to be beneficial for vision tasks.
    Example Configuration (CvT-13):
    *   Stage 1: $H_0/4 \times W_0/4$ tokens, $C_1=64$ channels, 1 CvT block. ConvTokenEmbed: $K_1=7, S_1=4, P_1=...$. ConvProjs for QKV: $S_Q=1, S_K=2, S_V=2$.
    *   Stage 2: $H_0/8 \times W_0/8$ tokens, $C_2=192$ channels, 2 CvT blocks. ConvTokenEmbed: $K_2=3, S_2=2, P_2=...$. ConvProjs for QKV: $S_Q=1, S_K=2, S_V=2$.
    *   Stage 3: $H_0/16 \times W_0/16$ tokens, $C_3=384$ channels, 10 CvT blocks. ConvTokenEmbed: $K_3=3, S_3=2, P_3=...$. ConvProjs for QKV: $S_Q=1, S_K=1, S_V=1$. (No downsampling for K,V in later stages).

##### 1.3. Importance
CvT bridges the gap between CNNs and Transformers by effectively integrating convolutional structures. This leads to:
*   Improved performance and efficiency on vision tasks compared to earlier ViT variants.
*   Better model scalability and transfer learning capabilities.
*   The ability to capture both local (via convolutions) and global (via self-attention) image features effectively.

##### 1.4. Pros vs. Cons
*   **Pros:**
    *   Combines strengths of CNNs (inductive biases like translation equivariance, locality) and Transformers (global context, dynamic attention).
    *   Achieves SOTA performance on various vision benchmarks.
    *   More efficient than ViT in terms of parameters and FLOPs for comparable performance, partly due to convolutional projections and hierarchical structure.
    *   Flexible architecture allowing for different stage configurations and convolutional parameters.
*   **Cons:**
    *   More complex architecture than a pure ViT or a simple CNN.
    *   Design choices (kernel sizes, strides for convolutional embeddings and projections) require careful tuning.
    *   While more efficient than ViT, it can still be computationally intensive for very high-resolution images or many stages/blocks.

##### 1.5. Cutting-Edge Advances
*   **Variants and Extensions:** CvT's principles have influenced subsequent architectures. For example, CoAtNet, MaxViT, and others explore different ways to combine convolutions and attention.
*   **Applications:** CvT and similar hybrid models are being applied to downstream tasks like object detection and segmentation, often by using the CvT backbone for feature extraction.
*   **Efficiency Improvements:** Ongoing research focuses on further optimizing the attention mechanism and convolutional components for speed and memory.

#### 2. Data Pre-processing

##### 2.1. Definition
Standard image pre-processing steps are applied to raw input images to prepare them for the CvT model. This typically includes resizing, normalization, and data augmentation.

##### 2.2. Mathematical Formulations & Procedures
*   **Input Image:** $I_{raw} \in [0, 255]^{H_{raw} \times W_{raw} \times C_{raw}}$ (typically $C_{raw}=3$ for RGB).
*   **Resizing/Cropping:**
    *   Images are typically resized to a fixed input size, e.g., $224 \times 224$ or $384 \times 384$.
    *   During training: `RandomResizedCrop` is common. An area is randomly cropped from the image and resized.
    *   During evaluation: `Resize` to a slightly larger dimension (e.g., 256 for 224 input) and then `CenterCrop` to the target size.
    *   Mathematically, this involves interpolation algorithms (bilinear, bicubic).
*   **Data Augmentation (Training only):**
    *   **Random Horizontal Flip:**
        $$ I_{flipped}(x, y, c) = I(W - 1 - x, y, c) $$
        Applied with a probability (e.g., 0.5).
    *   **Color Jitter:** Randomly adjust brightness, contrast, saturation, hue.
        $I_{jittered} = f_{hue}(f_{sat}(f_{cont}(f_{bright}(I))))$ where $f$ are random transformations.
    *   **Normalization:** Standardize pixel values.
        For each channel $c$:
        $$ I_{norm}(x, y, c) = \frac{I_{processed}(x, y, c)/255.0 - \mu_c}{\sigma_c} $$
        Where $\mu = [\mu_R, \mu_G, \mu_B]$ and $\sigma = [\sigma_R, \sigma_G, \sigma_B]$ are per-channel means and standard deviations (e.g., ImageNet means/stds: $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
*   **Output:** Pre-processed image tensor $X_0 \in \mathbb{R}^{H_0 \times W_0 \times 3}$ (channel-last) or $\mathbb{R}^{3 \times H_0 \times W_0}$ (channel-first, common in PyTorch).

##### 2.3. Significance
*   **Standardization:** Ensures model receives inputs in a consistent format and range.
*   **Improved Generalization:** Data augmentation helps the model learn to be invariant to irrelevant transformations, reducing overfitting.
*   **Performance:** Proper normalization can speed up convergence and improve model stability.

#### 3. Training Procedure

##### 3.1. Initialization
*   **Weights:**
    *   Convolutional layers: Kaiming (He) initialization (`torch.nn.init.kaiming_normal_` or `kaiming_uniform_`) is common for layers followed by ReLU-like activations (GELU is similar).
    *   Linear layers: Xavier (Glorot) initialization (`torch.nn.init.xavier_uniform_` or `xavier_normal_`) or Kaiming. Truncated normal distribution is often used for Transformer weights (e.g., `trunc_normal_` from `timm` library).
    *   Biases: Typically initialized to zero.
    *   LayerNorm parameters: $\gamma$ (scale) initialized to 1, $\beta$ (shift) initialized to 0.
*   **Positional Embeddings (if explicit ones were used, CvT aims to reduce reliance via convolutions):** If any absolute positional embeddings are used (less central in CvT due to conv properties), they are often initialized from a truncated normal distribution or learned. CvT's convolutional nature implicitly handles relative spatial information.

##### 3.2. Forward Pass
Given a batch of pre-processed images $X_{batch} \in \mathbb{R}^{B \times 3 \times H_0 \times W_0}$.
1.  **Stage 1:**
    a.  $X_{1, token\_embed} = \text{Conv2d}_1(X_{batch}; K_1, S_1, P_1, C_1)$
    b.  $T_1 = \text{Reshape}(\text{LayerNorm}(\text{BatchNorm2d}(X_{1, token\_embed})))$ (Paper sometimes includes BN before LN for ConvTokenEmbed)
        $T_1 \in \mathbb{R}^{B \times N_1 \times C_1}$
    c.  $Z_{1,0} = T_1$
    d.  For $l = 1, \dots, L_1$: $Z_{1,l} = \text{CvTBlock}_l(Z_{1,l-1})$. Each CvTBlock involves:
        i.   $Z'_{1,l-1} = \text{LayerNorm}(Z_{1,l-1})$
        ii.  Reshape $Z'_{1,l-1}$ to $B \times C_1 \times H_1 \times W_1$.
        iii. $Q_i, K_i, V_i$ from $Z'_{1,l-1}$ via ConvProjs (depth-wise separable conv).
        iv.  $A_l = \text{MHSA}_{\text{ConvProj}}(Q_i, K_i, V_i)$ (output reshaped to $B \times N_1 \times C_1$).
        v.   $Z''_l = A_l + Z_{1,l-1}$
        vi.  $Z'''_l = \text{LayerNorm}(Z''_l)$
        vii. $F_l = \text{MLP}(Z'''_l)$
        viii. $Z_{1,l} = F_l + Z''_l$
    e.  Output of Stage 1: $X_1 = \text{ReshapeTo2D}(Z_{1,L_1}) \in \mathbb{R}^{B \times C_1 \times H_1 \times W_1}$.

2.  **Stage 2, ..., Stage $S_{total}$:** Repeat analogously.
    a.  $X_{s, token\_embed} = \text{Conv2d}_s(X_{s-1}; K_s, S_s, P_s, C_s)$
    b.  $T_s = \text{Reshape}(\text{LayerNorm}(\text{BatchNorm2d}(X_{s, token\_embed})))$
        $T_s \in \mathbb{R}^{B \times N_s \times C_s}$
    c.  $Z_{s,0} = T_s$
    d.  For $l = 1, \dots, L_s$: $Z_{s,l} = \text{CvTBlock}_l(Z_{s,l-1})$
    e.  Output of Stage $s$: $X_s = \text{ReshapeTo2D}(Z_{s,L_s}) \in \mathbb{R}^{B \times C_s \times H_s \times W_s}$.

3.  **Classification Head:**
    a.  $Z_{final} = Z_{S_{total}, L_{S_{total}}} \in \mathbb{R}^{B \times N_{S_{total}} \times C_{S_{total}}}$
    b.  $Z_{pool} = \text{GlobalAvgPool}(Z_{final}, \text{dim}=1) \in \mathbb{R}^{B \times C_{S_{total}}}$
        (Average over the $N_{S_{total}}$ sequence tokens for each item in batch).
    c.  $\text{Logits} = \text{Linear}(Z_{pool}) \in \mathbb{R}^{B \times N_{classes}}$

##### 3.3. Loss Function
*   **Cross-Entropy Loss:** For classification tasks.
    Let $y \in \{0, \dots, N_{classes}-1\}^B$ be the true class labels (one-hot encoded as $Y \in \{0,1\}^{B \times N_{classes}}$).
    Let $p_j = \text{Softmax}(\text{Logits}_j)$ for $j=0, \dots, N_{classes}-1$.
    $$ \mathcal{L}_{CE} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=0}^{N_{classes}-1} Y_{i,j} \log(p_{i,j}) $$
*   **Label Smoothing (Optional):**
    The target $Y_{i,j}$ can be modified for label smoothing:
    $$ Y'_{i,j} = (1 - \epsilon) Y_{i,j} + \frac{\epsilon}{N_{classes}} $$
    Where $\epsilon$ is a small constant (e.g., 0.1). The loss is then computed with $Y'$.

##### 3.4. Backward Pass (Backpropagation)
*   Compute gradients of the loss $\mathcal{L}$ with respect to all model parameters $\theta$ (weights and biases of all Conv layers, Linear layers, LayerNorm parameters).
    $$ \nabla_{\theta} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta} $$
*   This is done by applying the chain rule recursively, starting from the output layer and propagating gradients backward through the network. Frameworks like PyTorch (`loss.backward()`) automate this.

##### 3.5. Parameter Update
*   **Optimizers:**
    *   **AdamW:** Commonly used for Transformers. Adam with decoupled weight decay.
        For a parameter $\theta_t$ at step $t$:
        1.  Weight decay: $\theta_t \leftarrow \theta_t - \eta \lambda \theta_{t-1}$ (applied before momentum update)
        2.  Compute gradients: $g_t = \nabla_{\theta_t} \mathcal{L}_t$
        3.  Update biased first moment estimate: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
        4.  Update biased second raw moment estimate: $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
        5.  Compute bias-corrected first moment estimate: $\hat{m}_t = m_t / (1-\beta_1^t)$
        6.  Compute bias-corrected second raw moment estimate: $\hat{v}_t = v_t / (1-\beta_2^t)$
        7.  Update parameters: $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon_{opt}}$
        Where $\eta$ is learning rate, $\lambda$ is weight decay rate, $\beta_1, \beta_2$ are exponential decay rates for moment estimates, $\epsilon_{opt}$ is a small constant for numerical stability.
    *   **SGD with Momentum:**
        $v_t = \gamma v_{t-1} + \eta g_t$
        $\theta_{t+1} = \theta_t - v_t$ (Weight decay is often added directly to gradient: $g_t \leftarrow g_t + \lambda \theta_t$)
*   **Learning Rate Schedule:**
    *   **Cosine Annealing:**
        $$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right) $$
        Where $T_{cur}$ is current iteration, $T_{max}$ is total iterations.
    *   **Warmup:** Linearly increase learning rate from a small value to $\eta_{max}$ over a few initial epochs.
*   **Gradient Clipping (Optional):**
    To prevent exploding gradients, clip the L2 norm of gradients:
    If $\|\nabla_{\theta} \mathcal{L}\| > \text{max\_norm}$, then $\nabla_{\theta} \mathcal{L} \leftarrow \nabla_{\theta} \mathcal{L} \cdot \frac{\text{max\_norm}}{\|\nabla_{\theta} \mathcal{L}\|}$.

##### Training Loop Pseudo-Algorithm:
```
Initialize CvT model parameters θ
Initialize Optimizer (e.g., AdamW) with learning rate η, β1, β2, weight_decay λ
Initialize Learning Rate Scheduler (e.g., CosineAnnealingLR with Warmup)

For epoch = 1 to N_epochs:
  model.train()
  For each batch (images, labels) in training_dataloader:
    1. Pre-process images (augmentations, normalization)
    2. Optimizer.zero_grad()                                  // Clear previous gradients
    3. logits = model.forward(images)                         // Forward pass
       // Mathematical Justification: Compute model's prediction based on current parameters.
    4. loss = LossFunction(logits, labels)                    // Calculate loss
       // Mathematical Justification: Quantify discrepancy between prediction and true labels.
    5. loss.backward()                                        // Backward pass (compute gradients)
       // Mathematical Justification: Apply chain rule to find ∂L/∂θ for all parameters.
    6. (Optional) ClipGradients(model.parameters(), max_norm)
       // Mathematical Justification: Stabilize training by preventing overly large gradient updates.
    7. Optimizer.step()                                       // Update parameters
       // Mathematical Justification: Adjust θ in direction that minimizes L, using optimizer's rule.
    8. Scheduler.step()                                     // Update learning rate

  model.eval()
  Evaluate model on validation_dataloader (see Evaluation Phase)
  Save model checkpoint if validation performance improves
```

#### 4. Post-Training Procedures

##### 4.1. Fine-tuning
*   **Definition:** Adapting a pre-trained CvT model (e.g., trained on ImageNet) to a new, often smaller, target dataset or task.
*   **Mathematical Underpinning:**
    1.  Replace the final classification head: The old head (e.g., 1000 classes for ImageNet) is replaced with a new one matching $N_{classes, target}$.
        $$ \text{Logits}_{new} = \text{Linear}_{new}(Z_{pool}) \in \mathbb{R}^{N_{classes, target}} $$
        The weights of $\text{Linear}_{new}$ are randomly initialized.
    2.  Continue training: The entire model, or parts of it (e.g., only the new head and later stages), are trained on the new dataset using a smaller learning rate. The loss function and optimizer are similar to the pre-training phase but adapted for the new task.
        The objective is to minimize $\mathcal{L}_{target}$ on the new dataset.
*   **Significance:** Leverages learned features from large datasets, often leading to better performance and faster convergence on smaller datasets.

##### 4.2. Other Procedures (Briefly)
*   **Knowledge Distillation:** Training a smaller student model to mimic a larger, pre-trained teacher model (like CvT). Loss includes a term for matching student's output distribution to teacher's.
    $$ \mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y_{true}, p_{student}) + (1-\alpha) \mathcal{L}_{KL}(p_{teacher}^{\tau} || p_{student}^{\tau}) $$
    where $p^{\tau}$ are softmax outputs with temperature $\tau$.
*   **Model Pruning/Quantization:** For deployment on resource-constrained devices, reducing model size and computational cost by removing less important weights (pruning) or using lower-precision numerical formats (quantization).

#### 5. Evaluation Phase

##### 5.1. Metrics (SOTA for Image Classification)

*   **Top-1 Accuracy:**
    *   **Definition:** The proportion of test samples for which the predicted class with the highest probability is the correct class.
    *   **Equation:**
        $$ \text{Top-1 Accuracy} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{I}(\hat{y}_i = y_i) $$
        Where:
        *   $N_{test}$ is the total number of samples in the test set.
        *   $y_i$ is the true label for sample $i$.
        *   $\hat{y}_i = \arg\max_j (\text{Logits}_{i,j})$ is the predicted label for sample $i$.
        *   $\mathbb{I}(\cdot)$ is the indicator function (1 if true, 0 if false).

*   **Top-5 Accuracy:**
    *   **Definition:** The proportion of test samples for which the true class is among the top 5 predicted classes (those with the 5 highest probabilities).
    *   **Equation:**
        $$ \text{Top-5 Accuracy} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{I}(y_i \in \{\hat{y}_{i,1}, \hat{y}_{i,2}, \hat{y}_{i,3}, \hat{y}_{i,4}, \hat{y}_{i,5}\}) $$
        Where:
        *   $\{\hat{y}_{i,1}, \dots, \hat{y}_{i,5}\}$ are the labels corresponding to the 5 highest scores in $\text{Logits}_i$.

##### 5.2. Loss Functions (During Evaluation)
*   The same loss function used for training (e.g., Cross-Entropy) is typically monitored on the validation set to check for overfitting and model convergence, but accuracy is the primary metric for reporting performance.

##### 5.3. Evaluation Pseudo-Algorithm:
```
Load trained CvT model parameters
Set model to evaluation mode: model.eval()
  // Mathematical Justification: Disables dropout, uses running stats for BatchNorm (if any outside LN scope).

Initialize accumulator for correct_top1 = 0, correct_top5 = 0, total_samples = 0
Initialize accumulator for total_loss = 0

For each batch (images, labels) in evaluation_dataloader:
  1. Pre-process images (typically resize and center crop, normalization)
  2. With torch.no_grad():                                     // Disable gradient computation
     // Mathematical Justification: Speeds up inference, saves memory as gradients are not needed.
     logits = model.forward(images)
     loss = LossFunction(logits, labels)
     total_loss += loss.item() * images.size(0)

  3. Get top-1 predicted class: predicted_top1 = argmax(logits, dim=1)
  4. Get top-5 predicted classes: predicted_top5_indices = argsort(logits, dim=1, descending=True)[:, :5]

  5. Update correct_top1 += sum(predicted_top1 == labels)
  6. For each sample i in batch:
       If labels[i] in predicted_top5_indices[i]:
         correct_top5 += 1
  7. total_samples += images.size(0)

avg_loss = total_loss / total_samples
accuracy_top1 = correct_top1 / total_samples
accuracy_top5 = correct_top5 / total_samples

Print/Log avg_loss, accuracy_top1, accuracy_top5
```

##### 5.4. Best Practices & Potential Pitfalls
*   **Consistent Pre-processing:** Ensure evaluation pre-processing matches what the model was trained with (e.g., image size, normalization stats). Slight variations (like center crop vs. random crop) are standard.
*   **No Data Augmentation:** Do not use training-time data augmentation (like random flips, color jitter) during evaluation, as the goal is to assess performance on unmodified data.
*   **Batch Size:** Evaluation batch size can often be larger than training, limited by GPU memory. It does not affect model output values, only throughput.
*   **Reproducibility:** Set random seeds if stochastic elements are somehow part of an advanced evaluation (e.g., Monte Carlo dropout), though typically evaluation is deterministic.
*   **Pitfall - Training Mode:** Forgetting `model.eval()` can lead to incorrect results due to active dropout layers or BatchNorm layers updating their running statistics.
*   **Pitfall - Gradient Calculation:** Forgetting `torch.no_grad()` (or equivalent) will consume unnecessary memory and computation.