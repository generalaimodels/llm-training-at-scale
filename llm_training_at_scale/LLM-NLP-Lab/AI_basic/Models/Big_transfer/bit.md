### BiT: Big Transfer

#### I. Definition
BiT (Big Transfer) refers to a set of pre-trained ResNet-v2 architectures and a specific transfer learning methodology. The core idea is to pre-train large ResNet models on extensive datasets (e.g., ImageNet-21k, JFT-300M) and then provide a simple, effective heuristic (BiT-HyperRule) for fine-tuning these models on smaller downstream tasks with minimal hyperparameter tuning.

#### II. Pre-processing

##### A. Image Normalization
1.  **Definition**: Standard normalization of input image pixel values.
2.  **Equation**:
    For an input image $x$ with pixel values in $[0, 255]$, each channel $c$ is normalized as:
    $x_{norm}^{(c)} = \frac{x^{(c)}/255.0 - \mu_c}{\sigma_c}$
    *   $\mu_c$: Mean for channel $c$ (e.g., ImageNet means: $[0.485, 0.456, 0.406]$).
    *   $\sigma_c$: Standard deviation for channel $c$ (e.g., ImageNet stds: $[0.229, 0.224, 0.225]$).
    *   BiT typically uses means and stds derived from the specific large-scale pre-training dataset, or standard ImageNet statistics if appropriate. The paper states pixel values are in $[0,1]$ and no dataset-specific channel-wise normalization is performed during pre-training; input values are just rescaled from $[0,255]$ to $[-0.5, 0.5]$ or $[0,1]$. For fine-tuning, it's crucial to match the pre-training normalization. If pre-training used $[0,1]$, then for fine-tuning:
    $x_{norm}^{(c)} = x^{(c)}/255.0$
    If pre-training used $[-0.5, 0.5]$:
    $x_{norm}^{(c)} = x^{(c)}/255.0 - 0.5$

##### B. Image Resizing and Cropping
1.  **During Pre-training**:
    *   Images are resized such that the shorter side is a certain length (e.g., 256 pixels).
    *   A random crop of a fixed size (e.g., $224 \times 224$ pixels) is taken.
    *   Random horizontal flipping is applied.
2.  **During Fine-tuning (and Evaluation)**:
    *   Images are resized to a specific resolution (e.g., $256 \times 256$ or $384 \times 384$ for evaluation, fine-tuning resolution might vary).
    *   A central crop of the target fine-tuning/evaluation resolution (e.g., $224 \times 224$) is typically used for evaluation. For fine-tuning, random crops are used.
    *   BiT proposes a specific fine-tuning resolution based on the downstream dataset size using the BiT-HyperRule.

#### III. Model Architecture (ResNet-v2 with Group Normalization and Weight Standardization)

BiT utilizes ResNet-v2 architectures of varying depths and widths. The key modifications are the use of Group Normalization (GN) instead of Batch Normalization (BN) and Weight Standardization (WS) in convolutional layers.

##### A. ResNet-v2 Backbone
1.  **General Structure**:
    *   **Stem**: Initial convolutional layer followed by max pooling.
        *   Convolution: e.g., $7 \times 7$ conv, stride 2.
        *   Max Pooling: e.g., $3 \times 3$ pool, stride 2.
    *   **Stages**: Sequence of residual blocks (typically 4 stages). Each stage reduces spatial resolution (except possibly the first) and increases channel depth.
    *   **Residual Block (Pre-activation)**: The ResNet-v2 block applies normalization and activation *before* the convolutional layers.
        *   Bottleneck block structure: GN $\rightarrow$ ReLU $\rightarrow$ WS-Conv ($1 \times 1$) $\rightarrow$ GN $\rightarrow$ ReLU $\rightarrow$ WS-Conv ($3 \times 3$) $\rightarrow$ GN $\rightarrow$ ReLU $\rightarrow$ WS-Conv ($1 \times 1$).
        *   Identity shortcut: Connects input of the block to the output. If channel dimensions or spatial dimensions change, a projection WS-Conv ($1 \times 1$) is used in the shortcut.
        $y = \mathcal{F}(x, \{W_i\}) + x$ (identity shortcut)
        $y = \mathcal{F}(x, \{W_i\}) + W_s x$ (projection shortcut)

##### B. Group Normalization (GN)
1.  **Definition**: Normalizes features within groups of channels.
2.  **Equation**:
    For a feature map tensor $X \in \mathbb{R}^{N \times C \times H \times W}$ (Batch, Channels, Height, Width), GN divides $C$ channels into $G$ groups of $C/G$ channels each.
    For each sample $n$ and each group $g$:
    $\mu_{n,g} = \frac{1}{(C/G)HW} \sum_{c \in S_g} \sum_{h=1}^H \sum_{w=1}^W X_{n,c,h,w}$
    $\sigma_{n,g}^2 = \frac{1}{(C/G)HW} \sum_{c \in S_g} \sum_{h=1}^H \sum_{w=1}^W (X_{n,c,h,w} - \mu_{n,g})^2$
    $\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu_{n,g_c}}{\sqrt{\sigma_{n,g_c}^2 + \epsilon}} \quad \text{where } c \in S_{g_c}$
    $Y_{n,c,h,w} = \gamma_c \hat{X}_{n,c,h,w} + \beta_c$
    *   $S_g$: Set of channel indices in group $g$.
    *   $\epsilon$: Small constant for numerical stability.
    *   $\gamma_c, \beta_c$: Learnable scale and shift parameters per channel.
3.  **Industrial Standard**: `torch.nn.GroupNorm` in PyTorch. $G$ is typically set to 32.

##### C. Weight Standardization (WS)
1.  **Definition**: Standardizes the weights of convolutional layers. Applied in conjunction with GN.
2.  **Equation**:
    For a weight tensor $W \in \mathbb{R}^{C_{out} \times C_{in} \times k_H \times k_W}$ of a convolutional layer:
    For each output channel $i$ (filter $i$):
    $\mu_{W_i} = \frac{1}{C_{in} k_H k_W} \sum_{j=1}^{C_{in}} \sum_{x=1}^{k_H} \sum_{y=1}^{k_W} W_{i,j,x,y}$
    $\sigma_{W_i}^2 = \frac{1}{C_{in} k_H k_W} \sum_{j=1}^{C_{in}} \sum_{x=1}^{k_H} \sum_{y=1}^{k_W} (W_{i,j,x,y} - \mu_{W_i})^2$
    $\hat{W}_{i,j,x,y} = \frac{W_{i,j,x,y} - \mu_{W_i}}{\sqrt{\sigma_{W_i}^2 + \epsilon}}$
    The convolution then uses $\hat{W}$ instead of $W$.
3.  **Industrial Standard**: Implemented by modifying the `forward` pass of a standard `torch.nn.Conv2d` layer to include these calculations for its `weight` attribute before the convolution operation. The original `weight` tensor remains the one updated by the optimizer.

##### D. Model Variants (BiT-S, BiT-M, BiT-L)
BiT models are typically ResNet-v2 variants like R50x1, R101x1, R152x1, R50x3, R101x3, R152x3, where "RxY" means a ResNet with $X$ layers and $Y$ times the width multiplier.
*   **BiT-S**: Models pre-trained on ImageNet-1k (e.g., R50x1, R101x1, R152x1).
*   **BiT-M**: Models pre-trained on ImageNet-21k (e.g., R50x1, R101x1, R152x1, R50x3, R101x3, R152x3).
*   **BiT-L**: Models pre-trained on JFT-300M (e.g., R50x1, R101x1, R152x1, R50x3, R101x3, R152x3, R152x4).

##### E. Final Layer
*   After the last stage, Global Average Pooling (GAP) is applied to the feature maps.
    $f = \text{GAP}(H_L)$ where $H_L$ is the output of the last stage. $f \in \mathbb{R}^{D_{feat}}$.
*   A fully connected (linear) layer maps these features to the number of classes for the pre-training task.
    $\text{logits}_{\text{pretrain}} = f W_{\text{pretrain}} + b_{\text{pretrain}}$
    *   $W_{\text{pretrain}} \in \mathbb{R}^{D_{feat} \times N_{\text{classes, pretrain}}}$.

#### IV. Training (Pre-training and Fine-tuning)

##### A. Upstream Pre-training
1.  **Dataset**: Large-scale datasets like ImageNet-21k (~14 million images, ~21k classes) or JFT-300M (~300 million images, ~18k classes).
2.  **Optimizer**: SGD with momentum.
    $v_{t+1} = \mu v_t - \eta \nabla \mathcal{L}(W_t)$
    $W_{t+1} = W_t + v_{t+1}$
    *   $\mu$: momentum (e.g., 0.9).
    *   $\eta$: learning rate.
3.  **Learning Rate Schedule**: Linear warmup followed by polynomial decay or cosine decay.
    *   Example: Warmup for first ~5-10% of epochs, then decay.
4.  **Batch Size**: Large batch sizes are used (e.g., 4096, distributed across multiple TPU/GPU cores).
5.  **Regularization**:
    *   Weight decay (L2 regularization) is applied.
        $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} ||W||_2^2$
    *   Label smoothing is often used for the cross-entropy loss.
        $q_i' = (1 - \alpha) q_i + \alpha / K$
        *   $q_i$: one-hot ground truth label.
        *   $q_i'$: smoothed label.
        *   $\alpha$: smoothing factor (e.g., 0.1).
        *   $K$: number of classes.
6.  **Loss Function**: Cross-Entropy Loss (with label smoothing).
    $\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{K} q_i' \log(p_i)$
    *   $p_i = \text{softmax}(\text{logits}_{\text{pretrain}})_i$.
7.  **Duration**: Pre-training for a fixed number of epochs (e.g., 90 epochs for ImageNet-21k, fewer for JFT-300M due to its size, like 7-14 epochs).

##### B. Downstream Fine-tuning (BiT-HyperRule)
BiT provides a specific, simple recipe for fine-tuning.
1.  **Replace Head**: Remove the pre-training classification head ($W_{\text{pretrain}}, b_{\text{pretrain}}$) and replace it with a new, randomly initialized linear layer for the downstream task.
    $\text{logits}_{\text{task}} = f W_{\text{task}} + b_{\text{task}}$
    *   $W_{\text{task}} \in \mathbb{R}^{D_{feat} \times N_{\text{classes, task}}}$.
2.  **BiT-HyperRule Heuristics**:
    *   **Training Schedule (Number of Steps/Epochs)**:
        *   500 steps for datasets with < 20k examples.
        *   2500 steps for datasets with < 500k examples.
        *   10000 steps for datasets with > 500k examples.
        *   (Alternatively, train for a fixed number of epochs if step counts are not directly comparable, e.g., 20 epochs on medium-sized datasets).
    *   **Resolution**:
        *   Train at $128 \times 128$ resolution.
        *   (Or, for better performance, use higher resolutions like $224 \times 224$ or $384 \times 384$, especially for larger BiT models). The paper also suggests higher resolution fine-tuning benefits from models pre-trained at higher resolutions.
    *   **Data Augmentation**:
        *   Resize to $(R_{train}, R_{train})$.
        *   Random crop to $(R_{crop}, R_{crop})$.
        *   Random horizontal flip.
        *   No mixup or strong augmentations are used by default in the BiT-HyperRule, aiming for simplicity.
    *   **Optimizer**: SGD with momentum (e.g., 0.9).
    *   **Learning Rate Schedule**:
        *   Base learning rate (e.g., 0.001, 0.003, 0.01, 0.03 - typically needs light tuning or can be fixed).
        *   Cosine decay or decay at fixed steps (e.g., by a factor of 10 at 30%, 60%, 90% of total steps).
    *   **Batch Size**: 512 (can be adjusted based on hardware, with learning rate scaling if necessary, e.g., linear scaling: $\eta' = \eta \cdot B' / B$).
    *   **Weight Decay**: Typically set to 0 or a very small value, as GN+WS already provides some regularization.

##### C. Pre-training Pseudo-algorithm (Simplified)
1.  Initialize ResNet-v2 (with GN+WS) model parameters $W$.
2.  Set pre-training hyperparameters: learning rate $\eta$, momentum $\mu$, weight decay $\lambda$, batch size $B$, number of epochs $E_{pre}$.
3.  **For** epoch = 1 to $E_{pre}$:
    *   **For** each batch of images $\{x^{(b)}\}$ and labels $\{y^{(b)}\}$ from the pre-training dataset:
        1.  **Pre-processing**: Apply normalization, random resize, random crop, random flip to $x^{(b)}$ to get $x_{proc}^{(b)}$.
        2.  **Forward Pass**:
            $f^{(b)} = \text{ResNet-v2-GN-WS-Backbone}(x_{proc}^{(b)})$
            $\text{logits}^{(b)} = f^{(b)} W_{\text{pretrain}} + b_{\text{pretrain}}$
        3.  **Loss Calculation**:
            Apply label smoothing to $y^{(b)}$ to get $y_{smooth}^{(b)}$.
            $\mathcal{L}_{\text{CE}}^{(b)} = - \sum_{i} y_{smooth,i}^{(b)} \log(\text{softmax}(\text{logits}^{(b)})_i)$
            $\mathcal{L}_{\text{total}}^{(b)} = \mathcal{L}_{\text{CE}}^{(b)} + \frac{\lambda}{2} ||W||_2^2$ (considering all trainable weights $W$)
        4.  **Backward Pass & Optimization**:
            Compute gradients $\nabla \mathcal{L}_{\text{total}}^{(b)}$.
            Update $W$, $W_{\text{pretrain}}$, $b_{\text{pretrain}}$ using SGD with momentum.
    *   Adjust learning rate according to schedule.
4.  Store the pre-trained backbone weights $W$ (and potentially $W_{\text{pretrain}}, b_{\text{pretrain}}$ if the head is to be used as initialization).

##### D. Fine-tuning Pseudo-algorithm (BiT-HyperRule)
1.  Load pre-trained ResNet-v2 (GN+WS) backbone parameters $W_{pt}$. Initialize a new task-specific head $W_{task}, b_{task}$.
2.  Set fine-tuning hyperparameters based on BiT-HyperRule: total steps $S_{ft}$, base learning rate $\eta_{ft}$, batch size $B_{ft}$, input resolution $R_{ft}$.
3.  **For** step = 1 to $S_{ft}$:
    *   Sample a batch of images $\{x^{(b)}\}$ and labels $\{y^{(b)}\}$ from the downstream task dataset.
    1.  **Pre-processing**: Apply normalization (consistent with pre-training), resize to $(R_{ft, aug}, R_{ft, aug})$, random crop to $(R_{ft}, R_{ft})$, random flip to $x^{(b)}$ to get $x_{proc}^{(b)}$.
    2.  **Forward Pass**:
        $f^{(b)} = \text{ResNet-v2-GN-WS-Backbone}(x_{proc}^{(b)}; W_{pt})$
        $\text{logits}^{(b)} = f^{(b)} W_{\text{task}} + b_{\text{task}}$
    3.  **Loss Calculation**: (No label smoothing by default in BiT-HyperRule for fine-tuning)
        $\mathcal{L}_{\text{CE}}^{(b)} = - \sum_{i} y_i^{(b)} \log(\text{softmax}(\text{logits}^{(b)})_i)$
        (Weight decay might be applied, but often minimal or zero in BiT fine-tuning).
    4.  **Backward Pass & Optimization**:
        Compute gradients $\nabla \mathcal{L}_{\text{CE}}^{(b)}$.
        Update $W_{pt}$ (all layers are fine-tuned), $W_{\text{task}}$, $b_{\text{task}}$ using SGD with momentum and $\eta_{ft}$.
    *   Adjust learning rate according to schedule (e.g., cosine decay over $S_{ft}$ steps).
4.  The fine-tuned model ($W_{pt}$ updated, $W_{task}, b_{task}$) is ready for evaluation.

#### V. Post-training Procedures
For BiT, "post-training" largely refers to the fine-tuning stage itself, as it's designed for transfer. No specific post-training quantization or pruning steps are part of the core BiT methodology, though such techniques could be applied subsequently. The primary "post-training" consideration is the selection of the checkpoint from the fine-tuning run, typically the one performing best on a validation set or simply the final checkpoint if following the fixed-schedule BiT-HyperRule.

#### VI. Evaluation Phase

##### A. Loss Functions (Recap)
1.  **Pre-training Loss**: Cross-Entropy Loss with Label Smoothing.
    $\mathcal{L}_{\text{CE-pretrain}} = - \sum_{i=1}^{K_{\text{pretrain}}} q_i' \log(p_i)$
2.  **Fine-tuning Loss**: Standard Cross-Entropy Loss.
    $\mathcal{L}_{\text{CE-finetune}} = - \sum_{i=1}^{K_{\text{task}}} y_i \log(p_i)$

##### B. Metrics (SOTA and Domain-Specific)

1.  **Primary Metric: Top-1 Accuracy** (for classification tasks)
    $\text{Top-1 Acc} = \frac{\text{Number of samples where argmax}(p_i) \text{ is the correct class}}{\text{Total number of samples}}$
    *   BiT models are evaluated on a wide range of downstream datasets, including:
        *   ImageNet-1k (as a downstream task if pre-trained on a larger set).
        *   CIFAR-10, CIFAR-100.
        *   Oxford-IIIT Pets, Oxford Flowers-102.
        *   VTAB-1k (Visual Task Adaptation Benchmark, a suite of 19 diverse vision tasks).
    *   **SOTA**: BiT models, especially BiT-L pre-trained on JFT-300M, achieved SOTA on many of these benchmarks at the time of publication, demonstrating excellent transfer learning capabilities. For instance, BiT-L (R152x4) achieved 87.5% Top-1 on ImageNet-1k.

2.  **Few-Shot Accuracy**
    *   Often, Top-1 accuracy is reported when fine-tuning on subsets of downstream datasets with a small number of examples per class (e.g., 1, 5, 10, 25 shots). BiT shows strong performance in few-shot settings.

3.  **Mean Per-Class Accuracy**
    *   For datasets with class imbalance, mean per-class accuracy can be a more informative metric.
    $\text{Mean Per-Class Acc} = \frac{1}{K_{\text{task}}} \sum_{c=1}^{K_{\text{task}}} \text{Accuracy}_c$

##### C. Evaluation Protocol
*   **Image Pre-processing**: Resize the image such that the shorter side is $R_{eval}$ (e.g., if fine-tuning at $224 \times 224$, $R_{eval}$ could be 256). Then take a center crop of size $(R_{crop\_eval}, R_{crop\_eval})$ (e.g., $224 \times 224$).
*   **No Test-Time Augmentation (TTA)**: BiT evaluations typically report scores without TTA for simplicity, though TTA could further improve results.

##### D. Pitfalls and Best Practices
*   **Consistency in Pre-processing**: Crucial to use the same normalization and input range for fine-tuning as used during pre-training.
*   **BiT-HyperRule as a Starting Point**: While effective, the BiT-HyperRule might not be optimal for all datasets. Some hyperparameter tuning (especially learning rate and schedule length) can yield further gains.
*   **Resolution**: Fine-tuning at higher resolutions generally improves performance but increases computational cost. The choice of resolution should align with the capacity of the pre-trained model.
*   **Data Leakage**: Ensure strict separation between training, validation, and test sets for downstream tasks. When reporting on standard benchmarks, use official splits.

#### VII. Importance & Significance
*   **State-of-the-Art Transfer Learning**: Demonstrated that large-scale pre-training combined with a simple fine-tuning recipe can yield exceptional results across many vision tasks.
*   **Democratizing Large Model Usage**: Provided robust pre-trained models and a straightforward fine-tuning guide, making powerful models more accessible.
*   **Effectiveness of GN+WS**: Showcased that Group Normalization with Weight Standardization can be a strong alternative to Batch Normalization, especially beneficial for very large batch pre-training and stable fine-tuning across varying batch sizes.
*   **Benchmark for Transferability**: BiT models and the VTAB-1k benchmark became standard references for evaluating the transfer learning capabilities of vision models.

#### VIII. Pros and Cons

##### A. Pros
*   **Excellent Performance**: Achieves high accuracy on a wide range of downstream tasks.
*   **Simple Fine-tuning**: The BiT-HyperRule simplifies the transfer learning process, reducing the need for extensive hyperparameter search.
*   **Robustness**: Models are relatively robust to variations in downstream dataset size and characteristics.
*   **Good Few-Shot Learner**: Performs well even when fine-tuned on very few examples.
*   **Publicly Available Models**: Google released many BiT pre-trained models, facilitating research and application.
*   **GN+WS Benefits**: Avoids issues with Batch Normalization related to small batch sizes during fine-tuning or differing statistics between pre-training and fine-tuning.

##### B. Cons
*   **High Pre-training Cost**: Pre-training BiT-M and BiT-L models requires massive datasets and computational resources, making it inaccessible for most researchers/organizations to replicate from scratch.
*   **Large Model Size**: The larger BiT models (e.g., R152x4) are computationally intensive for both fine-tuning and inference.
*   **HyperRule is a Heuristic**: While generally good, it's not universally optimal; some tasks may require more specific tuning.
*   **Focus on Supervised Pre-training**: Relies on labeled pre-training datasets, which can be expensive to curate at scale (though ImageNet-21k and JFT-300M labels are somewhat noisy).

#### IX. Cutting-Edge Advances and BiT's Influence
*   **Vision Transformers (ViTs)**: While BiT pushed the boundaries of CNN-based transfer learning, Vision Transformers subsequently emerged, demonstrating even stronger scaling properties and performance on similar large-scale pre-training regimes. ViT papers often compare against BiT baselines.
*   **Improved Pre-training Strategies**: Self-supervised learning (e.g., SimCLR, MoCo, DINO, MAE, BEiT) has become a major focus, aiming to reduce reliance on large labeled pre-training datasets. Many of these compare their transfer learning capabilities to BiT.
*   **Efficient Fine-tuning**: Techniques like LoRA (Low-Rank Adaptation), prompt tuning, and adapter modules have been developed to make fine-tuning large models more parameter-efficient, which can be applied to BiT models as well.
*   **Data Curation for Pre-training**: Research continues into curating better and larger pre-training datasets (e.g., LAION).
*   **BiT as a Strong Baseline**: BiT remains a formidable baseline for supervised transfer learning in computer vision. Its methodology and findings influenced how subsequent large models are trained and transferred.
*   **Refinement of Normalization Layers**: Continued exploration of normalization techniques beyond BN and GN, or combinations thereof, for large-scale training.