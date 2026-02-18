**SwAV: Swapping Assignments between multiple Views of the same image**

### Definition
SwAV (Swapping Assignments between multiple Views of the same image) is a self-supervised learning algorithm for visual representations. It leverages an online clustering-based approach where it enforces consistency between cluster assignments produced for different augmented views (crops) of the same image. Instead of comparing features directly like contrastive methods, SwAV predicts the cluster assignment (code) of one view from the representation of another view. The cluster assignments are computed online using prototype vectors.

### Pertinent Equations
1.  **Feature Representation from Encoder $f_\theta$ and Projector $h_\theta$:**
    $$ z = h_\theta(f_\theta(x)) $$
    After projection, $z$ is L2-normalized: $\bar{z} = z / \|z\|_2$.
2.  **Prototype Vectors (Learnable):**
    A set of $K$ prototype vectors (cluster centroids), denoted by the matrix $C = [c_1, c_2, \dots, c_K]$, where each $c_k \in \mathbb{R}^D$.
3.  **Scores for Prototype Assignment (Similarity):**
    For a feature $\bar{z}_i$, its scores with respect to all prototypes are given by $\bar{z}_i^T C$.
4.  **Assignment Code Computation (via Optimal Transport / Sinkhorn-Knopp):**
    The assignment matrix $Q = [q_1, \dots, q_B]$ for a batch of $B$ features $Z = [\bar{z}_1, \dots, \bar{z}_B]$ is found by optimizing:
    $$ Q^* = \text{argmax}_{Q \in \mathcal{Q}} \text{Tr}(Q^T C^T Z) + \epsilon H(Q) $$
    where $H(Q) = -\sum_{ij} Q_{ij} \log Q_{ij}$ is an entropy regularization term, and $\mathcal{Q}$ is the transportation polytope:
    $$ \mathcal{Q} = \{ Q \in \mathbb{R}_+^{K \times B} | Q \mathbf{1}_B = \frac{1}{K} \mathbf{1}_K, Q^T \mathbf{1}_K = \frac{1}{B} \mathbf{1}_B \} $$
    This enforces that each prototype is selected at least $B/K$ times on average in a batch (equipartiotion constraint).
5.  **Swapped Prediction Loss:**
    For two augmented views $x_t$ and $x_s$ of the same image, with features $z_t, z_s$ and assignments $q_t, q_s$:
    $$ \mathcal{L}(z_t, z_s) = \ell(z_t, q_s) + \ell(z_s, q_t) $$
    where the individual loss term is:
    $$ \ell(z, q) = -\sum_{k=1}^K q_k \log p_k(z) $$
    and $p_k(z)$ is the probability of feature $z$ belonging to prototype $k$, computed via a softmax over dot products with prototypes:
    $$ p_k(z) = \frac{\exp(\frac{1}{\tau_p} z^T c_k)}{\sum_{k'=1}^K \exp(\frac{1}{\tau_p} z^T c_{k'})} $$
    Here, $q_k$ is the $k$-th component of the assignment vector $q$ (target code), and $\tau_p$ is a temperature parameter.

### Key Principles
*   **Online Clustering:** SwAV implicitly performs clustering by assigning image features to a set of learnable prototype vectors. This assignment is done online for each mini-batch.
*   **Swapped Prediction Task:** The model is trained to predict the cluster assignment (code) of one augmented view of an image using the feature representation of another augmented view of the same image.
*   **Multi-Crop Augmentation:** SwAV employs a multi-crop strategy, generating several low-resolution views and two standard-resolution views from each image. This increases the number of view pairs without a significant increase in computational cost.
*   **Equipartition Constraint via Optimal Transport:** The Sinkhorn-Knopp algorithm is used to compute the assignment codes, ensuring that data points are assigned relatively uniformly across the prototypes, preventing a trivial solution where all features map to a single prototype.
*   **No Direct Contrastive Comparison:** Unlike methods like SimCLR or MoCo, SwAV does not directly contrast positive pairs against negative pairs using a InfoNCE-like loss. Instead, it focuses on consistency of cluster assignments.

### Detailed Concept Analysis

#### Pre-processing Steps: Multi-Crop Data Augmentation
SwAV utilizes a "multi-crop" strategy to generate various views from a single image $x$.
1.  **Standard Resolution Crops:** Two standard resolution crops (e.g., $224 \times 224$) are generated, $x_1$ and $x_2$. These are obtained using standard augmentations:
    *   Random Resized Crop (scale: [0.14, 1.0] for $224 \times 224$)
    *   Random Horizontal Flip
    *   Color Jitter (brightness, contrast, saturation, hue)
    *   Grayscale Conversion
    *   Gaussian Blur
2.  **Low Resolution Crops:** $V$ additional low-resolution crops (e.g., $96 \times 96$) are generated, $\{x_v\}_{v=3}^{V+2}$. These are obtained using:
    *   Random Resized Crop (scale: [0.05, 0.14] for $96 \times 96$)
    *   Same augmentations as above (horizontal flip, color jitter, etc.).
Mathematically, if $T_i$ represents the $i$-th stochastic augmentation pipeline:
$$ x_i = T_i(x) $$
The set of augmented views for one image is $\{x_1, x_2, x_3, \dots, x_{V+2}\}$.

#### Model Architecture
1.  **Encoder ($f_\theta$):**
    *   **Definition:** A convolutional neural network (e.g., ResNet-50) that extracts feature maps from an input image. The parameters are $\theta$.
    *   **Output:** For an input view $x_i$, it outputs a representation $y_i = f_\theta(x_i)$. This is typically the output of the average pooling layer.

2.  **Projector ($h_\theta$):**
    *   **Definition:** An MLP (e.g., 2-layer) that maps the encoder's output $y_i$ to a lower-dimensional embedding space. The parameters are also part of $\theta$.
    *   **Mathematical Formulation:**
        $$ z_i = h_\theta(y_i) = W^{(p_2)} \text{ReLU}(BN(W^{(p_1)} y_i)) $$
        The output $z_i \in \mathbb{R}^D$ (e.g., $D=128$) is then L2-normalized: $\bar{z}_i = z_i / \|z_i\|_2$.

3.  **Prototypes ($C$):**
    *   **Definition:** A set of $K$ learnable prototype vectors, $C = [c_1, \dots, c_K]$, where each $c_k \in \mathbb{R}^D$. These vectors act as learnable cluster centroids.
    *   **Initialization:** Can be initialized randomly or from representations of random samples.
    *   **Learning:** The prototypes $C$ are part of the model's parameters and are learned jointly with $\theta$ via backpropagation.

#### Online Clustering and Assignment Computation
For a mini-batch of $B$ features $Z = [\bar{z}_1, \dots, \bar{z}_B]$, where each $\bar{z}_j$ is the L2-normalized projection of an augmented view:
1.  **Compute Scores:** Calculate the dot products between all features in the batch and all prototypes: $S = Z^T C$. $S_{ji} = \bar{z}_j^T c_i$.
2.  **Optimal Transport for Codes ($Q$):** The assignment codes $Q = [q_1, \dots, q_B] \in \mathbb{R}^{K \times B}$ are computed by solving an optimal transport problem. This ensures that each feature is assigned to prototypes and that prototypes are used roughly equally often across the batch (equipartiotion). The optimization problem is:
    $$ Q^* = \underset{Q \in \mathcal{Q}}{\text{argmax}} \left( \text{Tr}(Q^T S) + \epsilon H(Q) \right) $$
    where $S_{ij}$ is the similarity between feature $j$ and prototype $i$, and $H(Q)$ is an entropy term. $\mathcal{Q}$ is the transportation polytope:
    $$ \mathcal{Q} = \left\{ Q \in \mathbb{R}_+^{K \times B} \, \middle| \, Q \mathbf{1}_B = \frac{1}{K} \mathbf{1}_K, Q^T \mathbf{1}_K = \frac{1}{B} \mathbf{1}_B \right\} $$
    This problem is solved efficiently using the iterative Sinkhorn-Knopp algorithm.
    *   **Sinkhorn-Knopp Algorithm:**
        Initialize $Q = \exp(S/\epsilon)$. Iteratively normalize rows and columns:
        For $iter = 1 \dots N_{\text{sk}}$:
        1.  $Q \leftarrow Q / (Q \mathbf{1}_B \mathbf{1}_K^T)$ (Normalize columns sum to $1/B$, assuming $Q$ is $B \times K$ here for row/col consistency. In paper's $K \times B$ notation, normalize rows to sum to $1/K$). SwAV code often uses $Z C^T$, so $S$ is $B \times K$. Then $Q$ is $B \times K$.
            $Q \leftarrow Q / (\mathbf{1}_K \mathbf{1}_B^T Q)$ (Normalize rows to sum to $1/K$)
            $Q \leftarrow Q / (Q \mathbf{1}_K \mathbf{1}_B^T)$ (Normalize columns to sum to $1/B$)
        A simplified version using row and column scaling vectors $u, v$:
        Let $M = \exp(Z C^T / \epsilon)$.
        Initialize $v = \mathbf{1}_K$.
        For $iter = 1 \dots N_{\text{sk}}$:
          $u = (1/B) \mathbf{1}_B / (Mv)$
          $v = (1/K) \mathbf{1}_K / (M^T u)$
        $Q = \text{diag}(u) M \text{diag}(v)$.
    *   Each column $q_j$ of $Q^*$ is the "soft" assignment code for feature $\bar{z}_j$. It represents a probability distribution over the $K$ prototypes.
    *   The $\text{stop_gradient}$ operation is applied to $Q$ when computing the loss for updating $\theta$, similar to target networks in other methods.

### Training Procedure

#### Loss Function
The total loss over all pairs of views generated from an image is:
$$ \mathcal{L}_{\text{total}} = \sum_{i=1}^{N_{views}} \sum_{j \neq i, j=1}^{N_{views}} \mathcal{L}(z_i, z_j) $$
where $N_{views} = V+2$ is the total number of views for one image.
The swapped prediction loss for a pair of features $(z_t, z_s)$ and their corresponding codes $(q_t, q_s)$ is:
$$ \mathcal{L}(z_t, z_s) = \ell(z_t, q_s) + \ell(z_s, q_t) $$
The individual loss term $\ell(z, q)$ is a cross-entropy like loss:
$$ \ell(z, q) = -\sum_{k=1}^K q_k \log p_k(z) $$
where $q_k$ is the $k$-th component of the target code $q$ (obtained from another view via Sinkhorn-Knopp, and treated as fixed for this term), and $p_k(z)$ is the softmax probability of feature $z$ being assigned to prototype $c_k$:
$$ p_k(z) = \frac{\exp(z^T c_k / \tau_p)}{\sum_{k'=1}^K \exp(z^T c_{k'} / \tau_p)} $$
The temperature $\tau_p$ controls the sharpness of the predicted distribution.

#### Training Algorithm
For each training iteration:
1.  **Sample Mini-batch:** Sample a mini-batch of $B$ images $\{x^{(b)}\}_{b=1}^B$.
2.  **Multi-Crop Augmentation:** For each image $x^{(b)}$:
    *   Generate 2 standard-resolution crops $x^{(b)}_1, x^{(b)}_2$.
    *   Generate $V$ low-resolution crops $\{x^{(b)}_v\}_{v=3}^{V+2}$.
    *   This results in $B \times (V+2)$ augmented views in total.
3.  **Compute Features and Projections:** For every augmented view $x_j$ in the batch:
    *   $y_j = f_\theta(x_j)$
    *   $z_j = h_\theta(y_j)$
    *   $\bar{z}_j = z_j / \|z_j\|_2$
    Let $Z_{batch}$ be the matrix of all $\bar{z}_j$.
4.  **Compute Assignment Codes $Q_{batch}$:**
    *   With current prototypes $C$, compute similarities $S = Z_{batch}^T C$.
    *   Solve for $Q_{batch}$ using the Sinkhorn-Knopp algorithm applied to $S$ with equipartition constraint.
    *   $Q_{batch}$ contains the assignment code $q_j$ for each $\bar{z}_j$. These codes $q_j$ are treated as targets (constants, use stop_gradient).
5.  **Compute Swapped Prediction Loss:**
    *   Initialize total loss $\mathcal{L}_{\text{iter}} = 0$.
    *   For each original image $x^{(b)}$ in the mini-batch:
        *   Let its set of projected features be $\{\bar{z}^{(b)}_1, \dots, \bar{z}^{(b)}_{V+2}\}$ and corresponding codes $\{q^{(b)}_1, \dots, q^{(b)}_{V+2}\}$.
        *   For each pair of distinct views $(i, j)$ from this set:
            *   Predict code $q^{(b)}_j$ using feature $\bar{z}^{(b)}_i$: Compute $p_k(\bar{z}^{(b)}_i)$ for all $k$.
            *   Loss term: $\ell(\bar{z}^{(b)}_i, q^{(b)}_j) = -\sum_{k=1}^K q^{(b)}_{jk} \log p_k(\bar{z}^{(b)}_i)$.
            *   Add to total loss: $\mathcal{L}_{\text{iter}} \leftarrow \mathcal{L}_{\text{iter}} + \ell(\bar{z}^{(b)}_i, q^{(b)}_j)$.
6.  **Gradient Update:**
    *   The gradients are computed for $\mathcal{L}_{\text{iter}}$ with respect to network parameters $\theta$ (of $f_\theta, h_\theta$) and prototype parameters $C$.
    *   Update $\theta$ and $C$ using an optimizer (e.g., LARS for $\theta$, SGD for $C$ with specific learning rates):
        $$ \theta \leftarrow \theta - \eta_\theta \nabla_\theta \mathcal{L}_{\text{iter}} $$
        $$ C \leftarrow C - \eta_C \nabla_C \mathcal{L}_{\text{iter}} $$
    *   **Important:** When computing gradients for $\theta$ (for $\ell(z_i, q_j)$), $q_j$ is detached from the computation graph. When computing gradients for $C$ (for $\ell(z_i, q_j)$), $q_j$ is also detached, but $C$ itself is part of $p_k(z_i)$.

**Best Practices / Pitfalls:**
*   **Learning Rates:** Separate learning rates for the encoder/projector and the prototypes can be beneficial.
*   **Number of Prototypes $K$:** A moderately large $K$ (e.g., 3000-5000 for ImageNet) is typical.
*   **Sinkhorn Iterations:** A small number of iterations (e.g., 3) for Sinkhorn-Knopp is usually sufficient.
*   **Temperature $\tau_p$:** Needs careful tuning. SwAV authors used $\tau_p = 0.1$.
*   **Warm-up:** A learning rate warm-up schedule is common.
*   **Multi-crop:** Crucial for performance and efficiency. Processing smaller crops is faster. Low-resolution crops provide local views, high-resolution crops provide global views.

### Post-Training Procedures
The pre-trained encoder $f_\theta$ is used as a feature extractor.
1.  **Linear Evaluation Protocol:**
    *   Freeze $f_\theta$. Train a linear classifier on top of its output features $y = f_\theta(x)$.
    *   Minimize $\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{N_{classes}} y_c^{\text{label}} \log(\text{softmax}(W^T y + b)_c)$.
2.  **Fine-tuning Protocol:**
    *   Initialize an encoder with $f_\theta$. Fine-tune all parameters (or a subset) on the downstream task.

### Evaluation Phase

#### Metrics (SOTA and Standard)
1.  **ImageNet Linear Classification Accuracy:**
    *   **Top-1 Accuracy:**
        $$ \text{Acc}_1 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\text{argmax}_c p_{i,c} == y_i^{\text{label}}) $$
    *   **Top-5 Accuracy:**
        $$ \text{Acc}_5 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(y_i^{\text{label}} \in \{\text{top 5 predicted classes for sample } i\}) $$
    *   SwAV achieved SOTA on this benchmark at its publication.

#### Transfer Learning Metrics
1.  **Object Detection (PASCAL VOC, COCO):** Mean Average Precision (mAP).
2.  **Semantic Segmentation (PASCAL VOC, Cityscapes):** Mean Intersection over Union (mIoU).
3.  **Instance Segmentation (COCO):** Mask mAP.
4.  **Other Classification Tasks:** Top-1 Accuracy on datasets like iNaturalist, Places205.

#### Loss Function (Monitoring during Training)
*   **SwAV Loss ($\mathcal{L}_{\text{total}}$):** The average swapped prediction loss. Monitoring its decrease indicates learning.
*   **Entropy of Assignments:** The entropy of the average assignment distribution over prototypes $ (1/B) \sum_j q_j $ can be monitored. Ideally, it should be high, indicating that all prototypes are being utilized.

### Importance
*   **Highly Efficient Self-Supervised Learning:** SwAV, especially with multi-crop, demonstrated a strong trade-off between performance and computational cost. It can achieve excellent results with smaller batch sizes compared to some contrastive methods.
*   **Online Clustering without Explicit Negative Pairs:** Provided an effective alternative to contrastive learning by framing self-supervision as an online clustering and assignment prediction task.
*   **State-of-the-Art Results:** Surpassed previous self-supervised methods and even supervised pre-training on several downstream tasks.
*   **Robustness to Batch Size:** More robust to smaller batch sizes than methods heavily reliant on in-batch negatives.

### Pros versus Cons

#### Pros
*   **High Performance:** Achieved SOTA results on various benchmarks.
*   **Computational Efficiency:** The multi-crop strategy allows for many comparisons per image with manageable compute. Not as demanding on batch size as some contrastive methods.
*   **No Need for Large Memory Bank or Huge Batches:** Unlike MoCo (queue) or SimCLR (large batches for negatives), SwAV is more flexible.
*   **Online Prototype Learning:** Adapts cluster centroids dynamically during training.

#### Cons
*   **Complexity of Assignment:** The Sinkhorn-Knopp algorithm adds a layer of complexity to the training loop compared to simpler objectives, though it is computationally efficient.
*   **Hyperparameter Sensitivity:** Performance can be sensitive to the number of prototypes $K$, temperature $\tau_p$, and the Sinkhorn-Knopp parameters (e.g., $\epsilon$, number of iterations).
*   **Potential for Cluster Degeneracy:** While Sinkhorn-Knopp helps, careful initialization and learning rate schedules for prototypes are important to prevent all features from mapping to a few prototypes (though the equipartition constraint is designed to mitigate this).
*   **Training Stability:** Can sometimes be less stable than simpler methods if hyperparameters are not well-tuned.

### Cutting-Edge Advances

1.  **DINO (Self-Distillation with No Labels, 2021):**
    *   Builds upon similar ideas of self-distillation (student-teacher) without explicit negative pairs, but uses a teacher network updated with EMA (like BYOL/MoCo). It also uses a centering and sharpening mechanism on the teacher outputs to prevent collapse. DINO showed excellent results with Vision Transformers (ViTs). SwAV's online clustering is distinct, but the theme of avoiding direct negative pair comparison is shared.

2.  **MSN (Masked Siamese Networks for Label-Efficient Learning, 2022):**
    *   Combines ideas from siamese networks (like SwAV, BYOL) with masked image modeling. One view is randomly masked, and its features are used to predict the features of an unmasked view, using prototype-based matching similar in spirit to SwAV.

3.  **EsViT (Efficient Self-supervised Vision Transformers, 2021):**
    *   Focuses on improving efficiency for ViTs in self-supervised settings, sometimes incorporating multi-crop ideas similar to SwAV or clustering-based objectives.

4.  **DenseCL (Dense Contrastive Learning for Self-Supervised Visual Pre-Training, 2021):**
    *   While contrastive, DenseCL extends ideas from instance-level discrimination to pixel or region-level, which can be seen as a finer-grained version of what clustering aims to achieve at an instance level. SwAV's prototypes can be thought of as capturing semantic clusters, while DenseCL focuses on local feature matching.

SwAV's core contribution of online clustering with swapped assignments and multi-crop augmentation remains influential, inspiring further research into efficient and high-performing self-supervised learning techniques.