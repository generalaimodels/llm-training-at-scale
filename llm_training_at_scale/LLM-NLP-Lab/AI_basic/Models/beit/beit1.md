### BEiT: 

#### I. Definition
BEiT  is a self-supervised pre-training framework for vision Transformers. It adapts the Masked Language Modeling (MLM) paradigm from NLP (specifically BERT) to the visual domain, termed Masked Image Modeling (MIM). BEiT pre-trains a Vision Transformer (ViT) to reconstruct original visual tokens corresponding to masked image patches. The visual tokens are obtained from a discrete Variational Autoencoder (dVAE).

#### II. Pre-processing and Visual Tokenization

##### A. Image Patching
1.  **Definition**: The input image is partitioned into a sequence of non-overlapping or minimally overlapping fixed-size patches.
2.  **Equation**:
    Given an input image $x \in \mathbb{R}^{H \times W \times C}$ (Height, Width, Channels), it is reshaped into a sequence of $N_p$ flattened 2D patches $x_p \in \mathbb{R}^{N_p \times (P^2 \cdot C)}$.
    *   $P$: Patch size (e.g., $16 \times 16$ pixels).
    *   $N_p = \frac{H \cdot W}{P^2}$: Number of patches.
    *   Each patch $x_p^i$ is a vector of dimension $P^2 \cdot C$.
3.  **Principle**: This transforms a 2D image into a 1D sequence, suitable for Transformer architectures.

##### B. Visual Tokenization via discrete Variational Autoencoder (dVAE)
1.  **Definition**: A dVAE is pre-trained to reconstruct image patches using a discrete codebook. Each image patch is then represented by the index of the closest codebook vector (visual token). This dVAE acts as an image tokenizer.
2.  **dVAE Architecture**:
    *   **Encoder ($E_{dVAE}$)**: Maps an image patch $x_p^i$ to a continuous representation $z_e^i$. Typically a convolutional network.
    *   **Quantizer ($Q_{dVAE}$)**: Maps $z_e^i$ to a discrete visual token $v_i \in \{1, \dots, |V|\}$ by finding the nearest neighbor in a codebook $\mathcal{C} = \{c_k \in \mathbb{R}^{D_{token}}\}_{k=1}^{|V|}$.
        $z_q^i = c_k \text{ where } k = \text{argmin}_j ||z_e^i - c_j||_2$
    *   **Decoder ($D_{dVAE}$)**: Reconstructs the image patch $\hat{x}_p^i$ from the quantized representation $z_q^i$. Typically a deconvolutional network.
3.  **dVAE Training**:
    *   **Objective**: Minimize reconstruction loss and regularize the codebook.
    *   **Loss Function**: A common choice for VQ-VAE style models (which dVAE is closely related to):
        $\mathcal{L}_{\text{dVAE}} = \mathbb{E}_{x_p \sim \text{Dataset}} \left[ ||x_p - D_{dVAE}(Q_{dVAE}(E_{dVAE}(x_p)))||_2^2 + ||\text{sg}[E_{dVAE}(x_p)] - Q_{dVAE}(E_{dVAE}(x_p))||_2^2 + \beta ||E_{dVAE}(x_p) - \text{sg}[Q_{dVAE}(E_{dVAE}(x_p))]||_2^2 \right]$
        *   $\text{sg}[\cdot]$ denotes the stop-gradient operator.
        *   The first term is reconstruction loss.
        *   The second term (codebook loss) updates the codebook embeddings $c_k$.
        *   The third term (commitment loss) ensures encoder outputs commit to codebook vectors. $\beta$ is a hyperparameter.
    *   **Industrial Standard**: Implemented using standard CNN components (e.g., `torch.nn.Conv2d`, `torch.nn.ConvTranspose2d` in PyTorch). The codebook is an `torch.nn.Embedding` layer. Training involves freezing BEiT and only training the dVAE on image patches.

4.  **Obtaining Visual Tokens for BEiT Pre-training**:
    *   For each image patch $x_p^i$ from the original image, its corresponding visual token $v_i$ is obtained by:
        $v_i = \text{index}(Q_{dVAE}(E_{dVAE}(x_p^i)))$
    *   The dVAE is trained once and then frozen. Its sole purpose during BEiT pre-training is to provide these target visual tokens $v_i$ for the masked patches.

#### III. BEiT Model Architecture (Transformer Encoder)

##### A. Input Representation for BEiT Pre-training
1.  **Masking**: A subset of image patches $x_p^i$ is randomly selected for masking. Common strategy: blockwise masking, where blocks of contiguous patches are masked.
    *   Let $\mathcal{M}$ be the set of indices of masked patches.
    *   For $i \in \mathcal{M}$, the patch $x_p^i$ is replaced by a special, learnable embedding $E_{[\text{MASK}]} \in \mathbb{R}^{D_{model}}$.
    *   For $i \notin \mathcal{M}$, unmasked patches are linearly projected: $E_i = x_p^i W_e + b_e$, where $W_e \in \mathbb{R}^{(P^2 \cdot C) \times D_{model}}$.
2.  **CLS Token**: A learnable classification token $E_{[\text{CLS}]}$ is prepended to the sequence of patch embeddings.
3.  **Positional Embeddings**: Learnable 1D positional embeddings $E_{pos} \in \mathbb{R}^{(N_p+1) \times D_{model}}$ are added to the patch embeddings to retain spatial information.
4.  **Input Sequence ($H_0$)**:
    $H_0 = [E_{[\text{CLS}]}; E_1'; E_2'; \dots; E_{N_p}'] + E_{pos}$
    *   $E_i' = E_{[\text{MASK}]}$ if patch $i$ is masked.
    *   $E_i' = \text{LinearProjection}(x_p^i)$ if patch $i$ is not masked.
    *   $D_{model}$ is the hidden dimension of the Transformer.

##### B. Transformer Encoder Layers
The BEiT model uses a standard Transformer encoder architecture, consisting of $L$ identical layers. Each layer has two sub-layers: Multi-Head Self-Attention (MHSA) and a position-wise Feed-Forward Network (FFN).

1.  **Multi-Head Self-Attention (MHSA)**
    *   **Definition**: Allows the model to jointly attend to information from different representation subspaces at different positions.
    *   **Equations**:
        For an input sequence $Z \in \mathbb{R}^{(N_p+1) \times D_{model}}$:
        Queries ($Q$), Keys ($K$), Values ($V$) for each head $h$:
        $Q_h = Z W_Q^h$, $K_h = Z W_K^h$, $V_h = Z W_V^h$
        *   $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{D_{model} \times d_k}$ are weight matrices for head $h$.
        *   $d_k = D_{model} / N_h$ (dimension per head, $N_h$ is number of heads).
        Attention scores for head $h$:
        $\text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h$
        Multi-head output:
        $\text{MHSA}(Z) = \text{Concat}(\text{Attention}_1, \dots, \text{Attention}_{N_h}) W_O$
        *   $W_O \in \mathbb{R}^{D_{model} \times D_{model}}$ is the output projection matrix.
    *   **Industrial Standard**: Implemented using `torch.nn.MultiheadAttention` or custom scaled dot-product attention layers.

2.  **Feed-Forward Network (FFN)**
    *   **Definition**: Applied to each position independently, consists of two linear transformations with an activation function in between.
    *   **Equations**:
        $\text{FFN}(Y) = \text{GELU}(Y W_1 + b_1) W_2 + b_2$
        *   $W_1 \in \mathbb{R}^{D_{model} \times D_{ffn}}$, $b_1 \in \mathbb{R}^{D_{ffn}}$
        *   $W_2 \in \mathbb{R}^{D_{ffn} \times D_{model}}$, $b_2 \in \mathbb{R}^{D_{model}}$
        *   $D_{ffn}$ is the inner FFN dimension (typically $4 \cdot D_{model}$). GELU is the Gaussian Error Linear Unit activation.
    *   **Industrial Standard**: Implemented using `torch.nn.Linear` and `torch.nn.GELU`.

3.  **Layer Norm and Residual Connections**
    *   Each sub-layer (MHSA, FFN) is preceded by Layer Normalization (LN) and followed by a residual connection.
    *   For layer $l$:
        $Y_l' = \text{MHSA}(\text{LN}(H_{l-1})) + H_{l-1}$
        $H_l = \text{FFN}(\text{LN}(Y_l')) + Y_l'$
    *   $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \gamma + \beta$, where $\mu, \sigma^2$ are mean/variance across features, $\gamma, \beta$ are learnable parameters.

##### C. Prediction Head (for MIM Pre-training)
1.  **Definition**: A linear layer that takes the final hidden state $H_L^i$ corresponding to a masked patch $i$ and predicts the probability distribution over the visual token vocabulary.
2.  **Equation**:
    For each masked patch $i \in \mathcal{M}$, let $H_L^i$ be its output representation from the last Transformer layer.
    $\text{logits}_i = H_L^i W_p + b_p$
    $P(v_i | x_{\text{masked}}) = \text{softmax}(\text{logits}_i)$
    *   $W_p \in \mathbb{R}^{D_{model} \times |V|}$, $b_p \in \mathbb{R}^{|V|}$.
    *   $|V|$ is the size of the dVAE codebook (vocabulary size).

#### IV. Pre-training: Masked Image Modeling (MIM)

##### A. Objective
To predict the original visual tokens (from the frozen dVAE) for the masked image patches based on the corrupted image (unmasked patches and `$[MASK]$` tokens).

##### B. Masking Strategy
*   **Blockwise Masking**: Randomly select blocks of patches to mask. This encourages learning of local context and high-level structure. Typically, around 40% of patches are masked.
*   For each image $x$:
    1.  Divide $x$ into $N_p$ patches $\{x_p^1, \dots, x_p^{N_p}\}$.
    2.  Obtain corresponding visual tokens $\{v^1, \dots, v^{N_p}\}$ using the pre-trained dVAE.
    3.  Randomly select a set of indices $\mathcal{M}$ for masking (e.g., 75 patches for a $14 \times 14$ grid of patches, which is $\approx 38\%$).
    4.  The input to BEiT is $x_{\text{corrupted}} = \{ (x_p^j \text{ if } j \notin \mathcal{M} \text{ else } E_{[\text{MASK}]}) \}_{j=1}^{N_p}$.
    5.  The target is to predict $\{v^j \text{ for } j \in \mathcal{M}\}$.

##### C. Loss Function for MIM
*   **Cross-Entropy Loss**: The model is trained to minimize the cross-entropy loss between the predicted visual token distribution and the true visual token (one-hot encoded) for each masked patch.
*   **Equation**:
    $\mathcal{L}_{\text{MIM}} = - \sum_{i \in \mathcal{M}} \sum_{k=1}^{|V|} \mathbb{I}(v_i = k) \log P(v_i = k | x_{\text{corrupted}})$
    *   $\mathcal{M}$ is the set of masked patch indices.
    *   $v_i$ is the true visual token for the $i$-th masked patch.
    *   $P(v_i = k | x_{\text{corrupted}})$ is the predicted probability that the $i$-th masked patch corresponds to visual token $k$.
    *   $\mathbb{I}(\cdot)$ is the indicator function.
*   **Industrial Standard**: Implemented using `torch.nn.CrossEntropyLoss`.

##### D. Pre-training Pseudo-algorithm
1.  **Phase 1: dVAE Training** (performed once, then dVAE is frozen)
    *   Initialize dVAE ($E_{dVAE}, Q_{dVAE}, D_{dVAE}$) with codebook $\mathcal{C}$.
    *   **For** each training epoch:
        *   **For** each batch of image patches $\{x_p^j\}$:
            *   $z_e^j = E_{dVAE}(x_p^j)$
            *   $z_q^j = Q_{dVAE}(z_e^j)$ (quantized representation using $\mathcal{C}$)
            *   $\hat{x}_p^j = D_{dVAE}(z_q^j)$ (reconstructed patch)
            *   Compute $\mathcal{L}_{\text{dVAE}}$ using $x_p^j, \hat{x}_p^j, z_e^j, z_q^j$.
            *   Update dVAE parameters and codebook $\mathcal{C}$ by backpropagating $\mathcal{L}_{\text{dVAE}}$.
    *   Store the trained dVAE (especially $E_{dVAE}$, $Q_{dVAE}$, and $\mathcal{C}$).

2.  **Phase 2: BEiT Pre-training**
    *   Initialize BEiT Transformer ($L$ layers, $E_{[\text{CLS}]}$, $E_{[\text{MASK}]}$, $E_{pos}$, prediction head).
    *   Load frozen, pre-trained dVAE tokenizer.
    *   **For** each training epoch:
        *   **For** each batch of images $\{x^{(b)}\}$:
            1.  **Image Patching**: $x^{(b)} \rightarrow \{x_p^{b,1}, \dots, x_p^{b,N_p}\}$.
            2.  **Visual Token Generation (Targets)**:
                For each patch $x_p^{b,j}$, obtain its visual token $v^{b,j}$ using the frozen dVAE:
                $v^{b,j} = \text{index}(Q_{dVAE}(E_{dVAE}(x_p^{b,j})))$.
            3.  **Masking**:
                *   Randomly select a set of patch indices $\mathcal{M}^{(b)}$ for masking.
                *   Create input sequence $H_0^{(b)}$:
                    *   For $j \in \mathcal{M}^{(b)}$, use $E_{[\text{MASK}]}$.
                    *   For $j \notin \mathcal{M}^{(b)}$, use linear projection of $x_p^{b,j}$.
                    *   Prepend $E_{[\text{CLS}]}$ and add $E_{pos}$.
            4.  **Forward Pass**:
                $H_L^{(b)} = \text{BEiT-Transformer-Encoder}(H_0^{(b)})$.
                Get outputs for masked positions: $\{H_L^{b,j} \text{ for } j \in \mathcal{M}^{(b)}\}$.
            5.  **Prediction**:
                For each $j \in \mathcal{M}^{(b)}$:
                $\text{logits}^{b,j} = H_L^{b,j} W_p + b_p$.
                $P(\cdot | x^{(b)}_{\text{corrupted}}) = \text{softmax}(\text{logits}^{b,j})$.
            6.  **Loss Calculation**:
                Compute $\mathcal{L}_{\text{MIM}}^{(b)}$ using predicted probabilities and target visual tokens $\{v^{b,j} \text{ for } j \in \mathcal{M}^{(b)}\}$.
            7.  **Backward Pass & Optimization**:
                Compute gradients $\nabla \mathcal{L}_{\text{MIM}}^{(b)}$.
                Update BEiT parameters (Transformer weights, $E_{[\text{CLS}]}$, $E_{[\text{MASK}]}$, $E_{pos}$, prediction head weights $W_p, b_p$) using an optimizer (e.g., AdamW).
    *   Store the pre-trained BEiT model weights.

#### V. Post-training: Fine-tuning for Downstream Tasks

##### A. Procedure
1.  **Replace/Modify Head**: Remove the MIM prediction head ($W_p, b_p$). Add a task-specific head (e.g., a linear classifier for image classification, a segmentation decoder for semantic segmentation).
2.  **Initialize**: Initialize the new head randomly. Load pre-trained BEiT weights for the Transformer backbone.
3.  **Training**: Train the entire model (or a subset of its layers, e.g., fine-tuning only the last few layers or just the new head) on a task-specific labeled dataset.
    *   The learning rate is typically smaller than during pre-training.
    *   Layer-wise learning rate decay (LLRD) is often beneficial, where layers closer to the input have smaller learning rates.

##### B. Mathematical Formulation (Example: Image Classification)
1.  **Task-specific Head**: A linear layer.
    Input: Output representation of the `$[CLS]$` token from the pre-trained BEiT encoder, $H_L^{[\text{CLS}]}$.
    $\text{logits}_{\text{task}} = H_L^{[\text{CLS}]} W_{\text{task}} + b_{\text{task}}$
    *   $W_{\text{task}} \in \mathbb{R}^{D_{model} \times N_{\text{classes}}}$, $b_{\text{task}} \in \mathbb{R}^{N_{\text{classes}}}$.
    *   $N_{\text{classes}}$ is the number of classes for the downstream task.
2.  **Loss Function**: Standard cross-entropy loss for classification.
    $\mathcal{L}_{\text{task}} = - \sum_{c=1}^{N_{\text{classes}}} y_c \log (\text{softmax}(\text{logits}_{\text{task}})_c)$
    *   $y$ is the one-hot encoded true label.

##### C. Best Practices
*   **Data Augmentation**: Use standard image augmentations (random crops, flips, color jittering) during fine-tuning.
*   **Regularization**: Dropout, weight decay.
*   **Optimizer**: AdamW is common.
*   **Learning Rate Schedule**: Cosine annealing or linear warmup followed by decay.

#### VI. Evaluation Phase

##### A. Loss Functions (Recap)
1.  **Pre-training Loss (MIM)**:
    $\mathcal{L}_{\text{MIM}} = - \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(v_i | x_{\text{corrupted}})$ (simplified from above, assuming $v_i$ is the target token class index)
    *   Measures the model's ability to reconstruct masked visual tokens.

2.  **Fine-tuning Loss (Task-dependent)**:
    *   **Image Classification**: Cross-Entropy Loss (as defined in V.B.2).
    *   **Semantic Segmentation**: Typically pixel-wise Cross-Entropy Loss or Dice Loss.
        $\mathcal{L}_{\text{seg}} = - \frac{1}{N_{\text{pixels}}} \sum_j \sum_c y_{j,c} \log p_{j,c}$ (Pixel-wise CE)
        *   $y_{j,c}$: one-hot true class for pixel $j$.
        *   $p_{j,c}$: predicted probability for pixel $j$ belonging to class $c$.
    *   **Object Detection**: Combination of classification loss (e.g., Focal Loss) for object classes and regression loss (e.g., Smooth L1 Loss) for bounding box coordinates.

##### B. Metrics (SOTA and Domain-Specific)

1.  **Image Classification**
    *   **Top-1 Accuracy**:
        $\text{Top-1 Acc} = \frac{\text{Number of samples where highest probability class is correct}}{\text{Total number of samples}}$
    *   **Top-5 Accuracy**:
        $\text{Top-5 Acc} = \frac{\text{Number of samples where true class is among top 5 highest probability classes}}{\text{Total number of samples}}$
    *   **SOTA**: Evaluated on datasets like ImageNet-1K. BEiT achieves competitive results compared to other self-supervised and supervised methods.

2.  **Semantic Segmentation**
    *   **Mean Intersection over Union (mIoU)**:
        $\text{mIoU} = \frac{1}{N_c} \sum_{c=1}^{N_c} \frac{TP_c}{TP_c + FP_c + FN_c}$
        *   $N_c$: Number of classes (including background).
        *   $TP_c$: True Positives for class $c$.
        *   $FP_c$: False Positives for class $c$.
        *   $FN_c$: False Negatives for class $c$.
    *   **Pixel Accuracy (PA)**:
        $\text{PA} = \frac{\sum_c TP_c}{\text{Total number of pixels}}$
    *   **SOTA**: Evaluated on datasets like ADE20K, Cityscapes. BEiT backbones (e.g., integrated into UPerNet or Mask2Former) show strong performance.

3.  **Object Detection and Instance Segmentation**
    *   **Average Precision (AP)**: Calculated based on the precision-recall curve. AP is usually averaged over multiple IoU thresholds (e.g., AP@[.5:.95]) and/or object categories (mAP).
        $\text{AP} = \sum_k (R_k - R_{k-1}) P_k$
        *   $P_k, R_k$ are precision and recall at the $k$-th threshold.
    *   **SOTA**: Evaluated on datasets like COCO. BEiT used as a backbone in detectors like Mask R-CNN shows significant improvements.

##### C. Pitfalls and Best Practices in Evaluation
*   **Consistent Pre-processing**: Ensure evaluation pre-processing matches training/fine-tuning pre-processing (e.g., image normalization, resizing).
*   **Hyperparameter Sensitivity**: Fine-tuning results can be sensitive to hyperparameters (learning rate, weight decay, batch size). Thorough hyperparameter search is often needed.
*   **Dataset Bias**: Evaluate on diverse datasets to assess generalization. Pre-training on larger, more diverse datasets generally leads to better downstream performance.
*   **Computational Cost**: Large Transformer models like BEiT are computationally expensive to pre-train and fine-tune. Monitor resource usage.
*   **Reproducibility**: Clearly document all settings, seeds, and software versions for reproducibility. Use standard evaluation scripts provided by benchmarks.

#### VII. Importance & Significance
*   **Bridging NLP and Vision**: Successfully adapts BERT's MLM pre-training to vision, demonstrating the potential of unified self-supervised learning approaches.
*   **Effective Self-Supervision**: Achieves strong performance on various downstream tasks, often outperforming supervised pre-training, especially in low-data regimes for fine-tuning.
*   **Scalability**: Vision Transformers, including BEiT, scale well with model size and data, leading to continued performance improvements.
*   **Foundation Model Potential**: Pre-trained BEiT can serve as a powerful backbone for a wide array of vision tasks.

#### VIII. Pros and Cons

##### A. Pros
*   **Strong Performance**: Achieves state-of-the-art or competitive results on major vision benchmarks.
*   **Label Efficiency**: Reduces reliance on large labeled datasets for pre-training.
*   **Generalization**: Learns robust representations that transfer well to diverse downstream tasks.
*   **Scalability**: Benefits from larger model sizes and more pre-training data.
*   **Unified Approach**: Aligns vision pre-training with successful paradigms in NLP.

##### B. Cons
*   **High Computational Cost**: Pre-training (both dVAE and BEiT itself) requires significant computational resources (GPUs/TPUs, time).
*   **Complexity**: Two-stage pre-training (dVAE tokenizer training then BEiT MIM) adds complexity.
*   **Discrete Tokenization Dependency**: Performance is linked to the quality of the dVAE visual tokenizer. Suboptimal tokenization can hinder learning.
*   **Masking Strategy**: The choice of masking strategy (e.g., blockwise vs. random) can impact performance and may require tuning.

#### IX. Cutting-Edge Advances and Variants
*   **BEiT v2**: Improves upon BEiT by introducing a shared (for dVAE and BEiT) backbone and a more efficient pre-training scheme where the model predicts features from a teacher model instead of discrete tokens, along with patch aggregation. This makes the visual tokenizer part of the main model.
*   **MaskFeat**: Similar to BEiT, but predicts Histogram of Oriented Gradients (HOG) features for masked patches, avoiding the need for a separate dVAE.
*   **MAE (Masked Autoencoders)**: A concurrent and similar approach that reconstructs raw pixel values of masked patches, simplifying the pre-training target. MAE often uses a lightweight decoder.
*   **SimMIM (Simplified Masked Image Modeling)**: Further simplifies MAE by using a very shallow decoder (e.g., a single linear layer) to predict raw pixel values.
*   **PeCo (Perceptual Codebook)**: Learns a perceptual codebook jointly with the MIM objective, aiming to improve the quality of visual tokens.
*   **VQ-GAN / ViT-VQGAN**: Advances in visual tokenization that could potentially improve the dVAE stage for BEiT-like models.
*   **Integration with Multimodal Models**: BEiT-like vision backbones are being integrated into vision-language models.