## I. Visual Token (ViT) and Image Transformer Backbone

### A. Definition
The Vision Transformer (ViT) model represents a paradigm shift in computer vision, applying the Transformer architecture, originally successful in Natural Language Processing (NLP), directly to sequences of image patches. Instead of relying on convolutional neural networks (CNNs) to extract features, ViT treats an image as a sequence of flattened patches, analogous to tokens (words) in a sentence. These patches are linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder.

### B. Model Architecture

#### 1. Patch and Position Embedding
An input image $x \in \mathbb{R}^{H \times W \times C}$ (Height, Width, Channels) is reshaped into a sequence of $N$ flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the resolution of each patch, and $N = HW/P^2$ is the number of patches.

*   **Patching:**
    The image $x$ is divided into $N$ patches. Each patch $x_p^i$ has dimensions $P \times P \times C$.
*   **Linear Projection:**
    Each flattened patch is linearly projected into a $D$-dimensional embedding space.
    $$ z_0^i = E x_p^i + e_{pos}^i \quad \text{for } i=1, \dots, N $$
    where $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding projection matrix (a learnable linear layer).
*   **Class Token (Optional, for classification):**
    Similar to BERT's `[CLS]` token, a learnable embedding $x_{class}$ is prepended to the sequence of embedded patches. Its state at the output of the Transformer encoder serves as the image representation.
    $$ z_0 = [x_{class}; E x_p^1 + e_{pos}^1; E x_p^2 + e_{pos}^2; \dots; E x_p^N + e_{pos}^N] $$
    where $x_{class} \in \mathbb{R}^D$ is the learnable class embedding and $e_{pos}^0$ is its corresponding position embedding.
*   **Position Embeddings:**
    Learnable 1D position embeddings $E_{pos} = [e_{pos}^0, e_{pos}^1, \dots, e_{pos}^N] \in \mathbb{R}^{(N+1) \times D}$ are added to the patch embeddings to retain positional information.
    The input to the Transformer encoder is $z_0 \in \mathbb{R}^{(N+1) \times D}$.

#### 2. Transformer Encoder
The Transformer encoder consists of $L$ identical layers. Each layer has two sub-layers: Multi-Head Self-Attention (MHSA) and a position-wise Feed-Forward Network (FFN). Layer Normalization (LN) is applied before each sub-layer, and residual connections are used after each sub-layer.

Let $z_{l-1}$ be the input to layer $l$.

*   **a. Multi-Head Self-Attention (MHSA)**
    Input $z_{l-1}$ is linearly projected into queries ($Q$), keys ($K$), and values ($V$) for each of $k$ attention heads.
    For a single head $h$:
    $$ Q_h = z_{l-1} W_h^Q, \quad K_h = z_{l-1} W_h^K, \quad V_h = z_{l-1} W_h^V $$
    where $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{D \times D_k}$ are learnable weight matrices for head $h$, and $D_k = D/k$ is the dimension of queries, keys, and values for each head.

    Scaled Dot-Product Attention for head $h$:
    $$ \text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{D_k}}\right) V_h $$
    The outputs of $k$ heads are concatenated and projected:
    $$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_k) W^O $$
    where $\text{head}_h = \text{Attention}(Q_h, K_h, V_h)$ and $W^O \in \mathbb{R}^{kD_k \times D} = \mathbb{R}^{D \times D}$ is another learnable projection matrix.

    The MHSA block output with LayerNorm and residual connection:
    $$ z'_l = \text{MHSA}(\text{LN}(z_{l-1})) + z_{l-1} $$

*   **b. Feed-Forward Network (FFN)**
    The FFN is a two-layer MLP with a GELU non-linearity.
    $$ \text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2 $$
    where $W_1 \in \mathbb{R}^{D \times D_{ff}}$, $b_1 \in \mathbb{R}^{D_{ff}}$, $W_2 \in \mathbb{R}^{D_{ff} \times D}$, $b_2 \in \mathbb{R}^{D}$. $D_{ff}$ is the inner-layer dimensionality, typically $4D$.

    The FFN block output with LayerNorm and residual connection:
    $$ z_l = \text{FFN}(\text{LN}(z'_l)) + z'_l $$

*   **c. Layer Normalization (LN) and Residual Connections**
    Layer Normalization is applied before each sub-layer.
    $$ \text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$
    where $\mu$ and $\sigma^2$ are the mean and variance of $x$ computed over the feature dimension $D$, and $\gamma, \beta$ are learnable affine transformation parameters. $\epsilon$ is a small constant for numerical stability.
    Residual connections $x + \text{Sublayer}(\text{LN}(x))$ are used around each of the two sub-layers.

The final output of the $L$-layer encoder is $z_L$. For classification, the representation $z_L^0$ (corresponding to the `[CLS]` token) is typically passed to an MLP head.

### C. Key Principles
*   **Sequence Processing:** Images are treated as sequences, leveraging the Transformer's strength in modeling long-range dependencies.
*   **Self-Attention:** The self-attention mechanism allows the model to weigh the importance of different image patches relative to each other, capturing global contextual information.
*   **Scalability:** Transformers scale effectively with model size (depth $L$, width $D$, number of heads $k$) and dataset size.
*   **Inductive Bias:** ViTs have less image-specific inductive bias (e.g., locality, translation equivariance) compared to CNNs. This makes them more data-hungry but allows them to learn more general patterns from large datasets.

### D. Detailed Concept Analysis
The core innovation of ViT is the direct application of the Transformer to image data with minimal modifications.
1.  **Patching as Tokenization:** Dividing the image into patches and linearly embedding them serves as the "tokenization" step. This converts spatial information into a sequence format suitable for Transformers.
2.  **Positional Embeddings:** Since self-attention is permutation-invariant, positional embeddings are crucial for the model to understand the spatial relationships between patches. Learnable 1D embeddings are common, but 2D-aware or relative positional embeddings can also be used.
3.  **Global Receptive Field:** From the very first layer, self-attention allows every patch to interact with every other patch, providing a global receptive field. This contrasts with CNNs where the receptive field grows gradually with depth.
4.  **Computational Complexity:** The self-attention mechanism has a quadratic complexity with respect to the number of patches $N$, i.e., $O(N^2 D)$. For high-resolution images, $N$ can be large, making ViTs computationally expensive.

### E. Importance
*   **Challenging CNN Dominance:** ViT demonstrated that a pure Transformer architecture could achieve state-of-the-art results on image recognition tasks, challenging the long-standing dominance of CNNs.
*   **Unified Architecture for Multi-Modal Learning:** The success of Transformers in both NLP and vision paves the way for unified architectures for multi-modal tasks.
*   **Foundation for Large-Scale Pre-training:** ViTs benefit significantly from pre-training on massive datasets (e.g., JFT-300M, ImageNet-21k), often outperforming CNNs when sufficient data is available. This has spurred research into self-supervised pre-training methods for ViTs, like BEiT.

### F. Pros and Cons
*   **Pros:**
    *   Excellent performance, especially when pre-trained on large datasets.
    *   Captures global relationships effectively due to self-attention.
    *   Scalable architecture in terms of model and data size.
    *   Less image-specific inductive bias can be an advantage if enough data is available to learn appropriate patterns.
*   **Cons:**
    *   Requires large amounts of training data to outperform CNNs; performs worse than CNNs on smaller datasets without strong regularization or pre-training.
    *   Computationally intensive, especially for high-resolution images, due to $O(N^2)$ complexity of self-attention.
    *   Less interpretable in its early layers compared to CNNs' localized feature learning.
    *   Patching can disrupt local structures at patch boundaries.

### G. Recent Developments
*   **Hierarchical ViTs (e.g., Swin Transformer, PVT):** Introduce hierarchical structures and local attention windows to reduce computational cost and incorporate some inductive biases helpful for vision.
*   **Efficient Attention Mechanisms:** Linear attention, sparse attention, and other approximations to reduce the quadratic complexity of self-attention.
*   **Hybrid Architectures:** Combining convolutional layers with Transformer blocks (e.g., CoAtNet, CvT).
*   **Advanced Pre-training Strategies:** Beyond supervised pre-training, methods like BEiT (MIM), MAE, MoCo v3, DINO have significantly improved ViT performance, especially in low-data regimes.

## II. Pre-Training BEiT: Masked Image Modeling (MIM)

### A. Definition
BEiT (Bidirectional Encoder representation from Image Transformers) is a self-supervised pre-training method for Vision Transformers. It adapts the Masked Language Modeling (MLM) paradigm from NLP to vision by introducing Masked Image Modeling (MIM). In MIM, a certain percentage of image patches are masked, and the BEiT model is trained to predict the discrete "visual tokens" corresponding to these masked patches. These visual tokens are obtained from a pre-trained image tokenizer, typically a discrete Variational Autoencoder (dVAE) or a VQ-VAE.

### B. Visual Tokenizer (dVAE/VQ-VAE Perspective)

#### 1. Definition
A visual tokenizer is a model that maps an image patch or an entire image into a sequence of discrete tokens from a finite vocabulary (codebook). For BEiT, a common choice is a dVAE (discrete Variational Autoencoder) or VQ-VAE (Vector Quantized Variational Autoencoder). This tokenizer is pre-trained separately to reconstruct images.

#### 2. Mathematical Formulation (VQ-VAE focus)
A VQ-VAE consists of an encoder $E_{tok}$, a discrete codebook $\mathcal{V} = \{v_1, \dots, v_{|\mathcal{V}|}\} \subset \mathbb{R}^{D_{vq}}$, and a decoder $D_{tok}$.
*   **Encoder:** For an image $x$ (or image patch), the encoder outputs continuous latent vectors $z_e(x) = E_{tok}(x)$.
*   **Quantization:** Each $z_e(x)$ is mapped to the closest codebook vector $v_k \in \mathcal{V}$.
    $$ z_q(x) = v_k \quad \text{where } k = \arg\min_j ||z_e(x) - v_j||_2 $$
    The index $k$ is the discrete visual token.
*   **Decoder:** The decoder reconstructs the image from the quantized latent vectors: $\hat{x} = D_{tok}(z_q(x))$.
*   **Training Loss (VQ-VAE):**
    $$ L_{VQ-VAE} = ||x - \hat{x}||_2^2 + ||\text{sg}[z_e(x)] - z_q(x)||_2^2 + \beta ||z_e(x) - \text{sg}[z_q(x)]||_2^2 $$
    where $\text{sg}[\cdot]$ denotes the stop-gradient operator. The first term is the reconstruction loss, the second is the codebook loss (moves codebook vectors towards encoder outputs), and the third is the commitment loss (encourages encoder output to stay close to chosen codebook vector). $\beta$ is a hyperparameter.
    *In PyTorch, this involves an `nn.Embedding` for the codebook and finding nearest neighbors.*

#### 3. Key Principles
*   **Image Discretization:** Reduces continuous pixel space to a discrete, manageable set of visual tokens.
*   **Semantic Compression:** The visual tokens aim to capture high-level semantic information rather than fine-grained pixel details.
*   **Pre-computation:** The visual tokenizer is trained once and then used to generate targets for BEiT pre-training.

### C. BEiT Pre-training Process

#### 1. Image Tokenization
For each input image $x$, it is first passed through the pre-trained visual tokenizer (e.g., dVAE encoder $E_{tok}$) to obtain a grid of discrete visual tokens $z = [z_{ij}] \in \{1, \dots, |\mathcal{V}|\}^{H' \times W'}$. These tokens $z_{ij}$ represent the "ground truth" for the MIM task. $H', W'$ are the dimensions of the token grid.

#### 2. Masking Strategy (Blockwise Masking)
A random subset of image patches (not visual tokens) is selected for masking.
*   Approximately 40-50% of the image patches are masked.
*   BEiT uses **blockwise masking**: random blocks of patches are masked. This encourages learning of high-level structures.
*   Let $M$ be the set of indices of the masked patches.

#### 3. Input to BEiT Backbone (Image Transformer)
The input image $x$ is split into patches $x_p = [x_p^1, \dots, x_p^N]$ as in standard ViT.
*   For unmasked patches ($i \notin M$): Linearly project $x_p^i$ to get $E x_p^i$.
*   For masked patches ($i \in M$): Replace the patch embedding with a shared, learnable `[MASK]` embedding $e_{[MASK]} \in \mathbb{R}^D$.
*   Add positional embeddings $E_{pos}$ to all patch representations.
The input sequence to the BEiT Transformer is:
$$ z_0 = [(E x_p^i \text{ if } i \notin M \text{ else } e_{[MASK]}) + e_{pos}^i \text{ for } i=0, \dots, N-1] $$
(Note: BEiT originally does not use a `[CLS]` token during pre-training, focusing on patch representations.)

#### 4. Prediction Head
The BEiT model (an Image Transformer) processes this input sequence $z_0$ to produce output embeddings $H = [H^0, \dots, H^{N-1}]$.
For each masked position $i \in M$, the corresponding output embedding $H^i$ is fed into a linear layer (softmax classifier) to predict the visual token $z_i$ for that patch.
$$ P(z_i | x_{\text{unmasked}}) = \text{softmax}(W_{pred} H^i + b_{pred}) $$
where $W_{pred} \in \mathbb{R}^{D \times |\mathcal{V}|}$ and $b_{pred} \in \mathbb{R}^{|\mathcal{V}|}$.

#### 5. Loss Function
The pre-training objective is to minimize the cross-entropy loss between the predicted visual token distributions and the ground-truth visual tokens $z_i$ (from the tokenizer) for all masked patches.
$$ L_{MIM} = - \sum_{x \in \text{Dataset}} \sum_{i \in M_x} \log P(z_i | x_{\text{unmasked}}) $$
This is effectively a sum of classification losses over the vocabulary of visual tokens for each masked patch.
*In PyTorch, this is `nn.CrossEntropyLoss` where logits are $W_{pred} H^i + b_{pred}$ and targets are the integer visual token indices $z_i$. The loss is typically averaged over masked patches and then over the batch.*

### D. Pre-Training Setup (Industrial Standard)
*   **Dataset:** Large-scale unlabeled or labeled (labels not used for MIM) image datasets, e.g., ImageNet-21k (14 million images), JFT-300M, or domain-specific large datasets.
*   **Visual Tokenizer:**
    *   Often a dVAE trained on a large and diverse dataset like OpenImages or the DALL-E dataset (250 million text-image pairs, image part used).
    *   Codebook size $|\mathcal{V}|$ is typically 8192.
*   **BEiT Backbone:** Usually ViT-Base or ViT-Large architecture.
    *   ViT-Base: $L=12$ layers, $D=768$ hidden size, $k=12$ heads, $D_{ff}=3072$.
    *   ViT-Large: $L=24$ layers, $D=1024$ hidden size, $k=16$ heads, $D_{ff}=4096$.
*   **Optimizer:** AdamW (Adam with weight decay).
    *   Learning rate: e.g., $1.5 \times 10^{-3}$ for ViT-Large, scaled by batch size.
    *   Weight decay: e.g., 0.05.
    *   Betas: ($\beta_1=0.9, \beta_2=0.999$).
*   **Learning Rate Schedule:** Cosine decay with linear warmup (e.g., 10 epochs warmup).
*   **Batch Size:** Large, e.g., 2048 or 4096, distributed across multiple GPUs.
*   **Epochs:** 300-800 epochs on ImageNet-21k.
*   **Data Augmentation:** Random resized crop, horizontal flip.
*   **Regularization:** Stochastic depth (drop path), layer-wise learning rate decay (optional).

### E. Importance
*   **Effective Self-Supervision for ViTs:** BEiT provided one of the first highly successful self-supervised learning (SSL) frameworks for ViTs, rivaling supervised pre-training.
*   **Reduces Label Dependency:** Enables learning strong visual representations without human-annotated labels for the pre-training stage.
*   **Improved Transferability:** Representations learned by BEiT transfer well to various downstream tasks like image classification and semantic segmentation.
*   **BERT-like Pre-training for Vision:** Successfully translates the "predict masked elements" idea from NLP to vision, demonstrating architectural and methodological convergence.

### F. Pros and Cons
*   **Pros:**
    *   Achieves state-of-the-art performance on many vision benchmarks.
    *   Learns robust and generalizable visual features.
    *   Scales well with model size and pre-training data.
    *   Conceptually simple and aligned with MLM in NLP.
*   **Cons:**
    *   Requires a pre-trained visual tokenizer, which itself can be complex and computationally expensive to train. The quality of the tokenizer impacts BEiT's performance.
    *   Pre-training BEiT is computationally intensive (large models, large datasets, many epochs).
    *   The discrete visual tokens might lead to information loss compared to reconstructing raw pixels or continuous features (addressed by MAE, which predicts pixels).

### G. Recent Developments
*   **MAE (Masked Autoencoders):** A concurrent work that masks patches but predicts raw pixel values of masked patches, simplifying the pre-training by removing the need for a visual tokenizer. MAE often achieves comparable or better performance with a simpler setup.
*   **SimMIM (Simplified MIM):** Further simplifies MIM by using a very light prediction head (e.g., a single linear layer) to predict raw pixels of masked patches, showing that complex decoders are not always necessary.
*   **PeCo, MVP:** Other variants exploring different aspects of MIM or combining MIM with contrastive learning.
*   **BEiT v2:** Improves upon BEiT by using a vector-quantized feature map from a teacher model (e.g., pre-trained CLIP ViT) as targets, and a more sophisticated masking strategy.

## III. Training Pseudo-Algorithm: BEiT Pre-training

This algorithm outlines the steps for pre-training the BEiT model using Masked Image Modeling.

### A. Initialization
1.  **Initialize BEiT Model:** Instantiate the Image Transformer backbone $\theta_{BEiT}$ (e.g., ViT-Base/Large) with random weights or weights from a simpler pre-training if applicable.
    *   `model = ViTEncoder(...)` (PyTorch-like)
2.  **Initialize Prediction Head:** Instantiate the linear layer $\theta_{pred}$ for predicting visual tokens.
    *   `prediction_head = nn.Linear(D, |\mathcal{V}|)`
3.  **Load Pre-trained Visual Tokenizer:** Load the weights for the visual tokenizer's encoder $E_{tok}$ and its codebook $\mathcal{V}$. This tokenizer is frozen during BEiT pre-training.
    *   `tokenizer = Pretrained_dVAE_Encoder()`
    *   `tokenizer.eval()`
4.  **Optimizer:** Initialize AdamW optimizer for $\theta_{BEiT}$ and $\theta_{pred}$.
    *   `optimizer = torch.optim.AdamW(list(model.parameters()) + list(prediction_head.parameters()), lr=..., weight_decay=...)`
5.  **Learning Rate Scheduler:** Initialize learning rate scheduler (e.g., cosine decay with warmup).

### B. Epoch Loop
```
for epoch in range(num_epochs):
    model.train()
    prediction_head.train()
    for batch_idx, images_batch in enumerate(dataloader): # images_batch: [B, C, H, W]
```

#### 1. Batch Iteration

##### a. Data Pre-processing (within Dataloader or at start of loop)
*   For each image $x \in \text{images_batch}$:
    *   Apply data augmentations: `RandomResizedCrop(target_size)`, `RandomHorizontalFlip()`.
    *   Normalize pixel values (e.g., to $[0,1]$ or standard ImageNet mean/std).
    *   `images_batch` now contains augmented, normalized images.

##### b. Visual Tokenization (Target Generation)
*   With `torch.no_grad()`:
    *   For each image $x \in \text{images_batch}$:
        *   Pass $x$ through the visual tokenizer $E_{tok}$ to get patch-wise latent representations.
        *   Quantize these representations using codebook $\mathcal{V}$ to get discrete visual tokens $z_x \in \mathbb{Z}^{N_t}$, where $N_t$ is the number of visual tokens per image.
        *   `visual_tokens_batch = tokenizer.encode_to_tokens(images_batch)`
        *   These $z_x$ are the ground truth targets for the MIM task.

##### c. Masked Image Construction
*   For each image $x \in \text{images_batch}$ and its corresponding visual tokens $z_x$:
    1.  **Patchify Image:** Divide $x$ into $N$ patches $x_p = [x_p^1, \dots, x_p^N]$.
        *   `patches = image_to_patches(x, patch_size=P)`
    2.  **Apply Masking Strategy:**
        *   Randomly select a set of patch indices $M_x$ to mask (e.g., blockwise masking, ~40-50% of patches).
        *   Store the ground truth visual tokens $z_x[i]$ for $i \in M_x$.
    3.  **Embed Patches and Apply Mask Token:**
        *   `input_embeddings = torch.zeros(N, D)`
        *   For $i = 0, \dots, N-1$:
            *   If $i \in M_x$: `input_embeddings[i] = learnable_mask_embedding`
            *   Else: `input_embeddings[i] = patch_projection_layer(flatten(patches[i]))`
    4.  **Add Positional Embeddings:**
        *   `input_sequence = input_embeddings + positional_embeddings`
        *   Collect `input_sequence` for all images in the batch. Let this be `batch_input_sequences`.
        *   Collect target visual tokens for masked positions: `batch_target_tokens_masked`.

##### d. Forward Pass
*   Pass the `batch_input_sequences` through the BEiT model (Transformer encoder):
    *   `output_embeddings = model(batch_input_sequences)`
    *   `output_embeddings` will have shape `[B, N, D]`.

##### e. Loss Calculation
*   Extract output embeddings corresponding to masked positions from `output_embeddings`. Let these be `masked_output_embeddings`.
*   Predict logits for visual tokens using the prediction head:
    *   `predicted_token_logits = prediction_head(masked_output_embeddings)`
    *   `predicted_token_logits` will have shape `[Num_Masked_Total, |\mathcal{V}|]`.
*   Calculate Cross-Entropy loss:
    *   `loss = nn.CrossEntropyLoss(reduction='mean')(predicted_token_logits, batch_target_tokens_masked)`
    *   (Ensure `batch_target_tokens_masked` are flat and correspond to `predicted_token_logits`)

##### f. Parameter Update
1.  `optimizer.zero_grad()`
2.  `loss.backward()`
3.  `optimizer.step()`
4.  `scheduler.step()` (if an LR scheduler is used per step/batch)

```
    # End of batch loop
    # scheduler.step() (if an LR scheduler is used per epoch)
    # Log metrics (loss, learning rate, etc.)
# End of epoch loop
# Save pre-trained model weights (model.state_dict())
```

**Mathematical Justification at Each Stage:**
*   **Visual Tokenization:** $z_x = \text{Quantize}(E_{tok}(x), \mathcal{V})$. This provides discrete targets that abstract away low-level pixel noise, focusing on semantic concepts.
*   **Masking & Input Construction:** $z_0 = [(E x_p^i \text{ if } i \notin M \text{ else } e_{[MASK]}) + e_{pos}^i]$. The `[MASK]` token forces the model to use context from unmasked patches to infer the content of masked ones. Positional embeddings are crucial as self-attention is permutation invariant.
*   **Forward Pass:** $H = \text{TransformerEncoder}(z_0)$. The Transformer aggregates global context via self-attention layers.
*   **Loss Calculation:** $L_{MIM} = -\sum \log P(z_i | x_{\text{unmasked}})$. This maximum likelihood objective trains the model to predict the correct discrete visual category for masked regions.
*   **Parameter Update:** $\theta \leftarrow \theta - \eta \nabla_{\theta} L_{MIM}$. Standard gradient descent-based optimization.

## IV. Fine-Tuning BEiT on Downstream Vision Tasks

### A. General Fine-Tuning Procedure
1.  **Load Pre-trained BEiT:** Initialize the Image Transformer backbone with weights from BEiT pre-training.
2.  **Task-Specific Head:** Remove the MIM prediction head. Add a new, randomly initialized head suitable for the downstream task (e.g., a linear layer for classification, a decoder for segmentation).
3.  **Dataset:** Use the labeled dataset for the specific downstream task.
4.  **Training:** Fine-tune the entire model (backbone + new head) or parts of it (e.g., only the head, or with a lower learning rate for the backbone – differential learning rates).
    *   Optimizer: AdamW is common.
    *   Learning rate: Typically smaller than pre-training LR (e.g., $5 \times 10^{-5}$ to $5 \times 10^{-4}$).
    *   Schedule: Cosine decay, possibly with warmup.
    *   Data Augmentations: Task-specific augmentations (e.g., RandAugment, Mixup, CutMix for classification).

### B. Image Classification

#### 1. Model Architecture
*   **Backbone:** Pre-trained BEiT (Image Transformer).
*   **Classification Head:**
    *   BEiT papers often describe using global average pooling (GAP) over the output patch embeddings, followed by a single linear layer.
    *   $H_{patches} = [H^1, \dots, H^N]$ (output embeddings from BEiT, excluding `[CLS]` token if not used in pre-training, or after removing it if it was).
    *   $\bar{H} = \text{GlobalAveragePool}(H_{patches}) = \frac{1}{N} \sum_{i=1}^N H^i$.
    *   $y_{logits} = W_{cls} \bar{H} + b_{cls}$, where $W_{cls} \in \mathbb{R}^{D \times N_{classes}}$, $b_{cls} \in \mathbb{R}^{N_{classes}}$.
    *   Alternatively, if a `[CLS]` token was used and fine-tuned: $y_{logits} = W_{cls} H^{[CLS]} + b_{cls}$.
*   **Output Activation:** Softmax for multi-class classification.
    $$ y_{pred} = \text{softmax}(y_{logits}) $$

#### 2. Data Pre-processing
*   **Resize/Crop:** Images are typically resized to the same resolution used during pre-training (e.g., 224x224 or 384x384). Random crops and flips are common during training. Test-time: center crop.
*   **Normalization:** Using the same mean/std statistics as pre-training.
*   **Augmentations:**
    *   `RandomResizedCrop`, `RandomHorizontalFlip`.
    *   Advanced augmentations: `RandAugment`, `Mixup`, `CutMix`.
    *   Mixup: $x_{mix} = \lambda x_i + (1-\lambda) x_j$, $y_{mix} = \lambda y_i + (1-\lambda) y_j$, where $\lambda \sim \text{Beta}(\alpha, \alpha)$.
    *   CutMix: A patch from image $x_j$ is pasted onto $x_i$. Labels are mixed proportionally to the area: $y_{mix} = \lambda y_i + (1-\lambda) y_j$.

#### 3. Training Pseudo-Algorithm (Image Classification Fine-tuning)
1.  Initialize BEiT backbone with pre-trained weights $\theta_{BEiT}^*$.
2.  Initialize classification head $\theta_{head}$ (e.g., `nn.Linear(D, num_classes)`).
3.  Setup optimizer (e.g., AdamW for $\theta_{BEiT}^* \cup \theta_{head}$, potentially with lower LR for backbone).
4.  `for epoch in range(num_fine_tune_epochs):`
    `    for images_batch, labels_batch in enumerate(classification_dataloader):`
    `        # Pre-process images_batch (augment, patchify, add pos_embed)`
    `        input_sequences = preprocess_for_vit(images_batch)`
    `        # Forward pass`
    `        output_embeddings = be_it_backbone(input_sequences)`
    `        # Global Average Pooling (if used)`
    `        pooled_output = torch.mean(output_embeddings, dim=1) # Assuming [B, N, D]`
    `        logits = classification_head(pooled_output)`
    `        # Loss Calculation`
    `        loss = nn.CrossEntropyLoss()(logits, labels_batch)`
    `        # Backward pass and optimize`
    `        optimizer.zero_grad()`
    `        loss.backward()`
    `        optimizer.step()`
    `        # (Update LR scheduler)`

#### 4. Evaluation Metrics and Loss Function
*   **Loss Function (Training):**
    *   **Cross-Entropy Loss:** For multi-class classification.
        $$ L_{CE} = - \sum_{c=1}^{N_{classes}} y_{true,c} \log(y_{pred,c}) $$
        where $y_{true,c}$ is 1 if true class is $c$ and 0 otherwise, $y_{pred,c}$ is the predicted probability for class $c$.
*   **Metrics (SOTA and Standard):**
    *   **Accuracy (Top-1):**
        $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Samples}} $$
    *   **Top-k Accuracy:** A prediction is correct if the true label is among the top $k$ predicted labels with highest probabilities.
        $$ \text{Top-k Accuracy} = \frac{\text{Number of Samples where True Label is in Top-k Predictions}}{\text{Total Number of Samples}} $$
    *   **Precision, Recall, F1-Score (especially for imbalanced classes or per-class analysis):**
        For a class $c$:
        $$ \text{Precision}_c = \frac{TP_c}{TP_c + FP_c} $$
        $$ \text{Recall}_c = \frac{TP_c}{TP_c + FN_c} $$
        $$ \text{F1-Score}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} $$
        (TP: True Positives, FP: False Positives, FN: False Negatives)
        Often reported as macro-averaged (average of per-class scores) or micro-averaged (aggregate TPs, FPs, FNs).

### C. Semantic Segmentation

#### 1. Model Architecture (with UPerNet considerations)
*   **Backbone:** Pre-trained BEiT, acting as a feature extractor. It outputs a sequence of patch embeddings $H_{patches} \in \mathbb{R}^{N \times D}$. These are reshaped to form a 2D feature map, e.g., $H_f \in \mathbb{R}^{(H/P) \times (W/P) \times D}$.
*   **Decoder:** A segmentation decoder is attached to the BEiT backbone to produce pixel-wise predictions. UPerNet is a common choice.
    *   **UPerNet Decoder:**
        1.  **Feature Pyramid Network (FPN):** BEiT, especially if non-hierarchical, provides features from its final layer. To get multi-scale features for FPN, one might take intermediate layer outputs (if available and compatible) or use techniques to generate a pyramid from the final feature map (e.g., dilated convolutions, pooling). For standard ViT/BEiT, features are extracted from different blocks (e.g., every 3 layers for ViT-B, resulting in 4 feature maps).
            Let $H^{(s)}$ be the feature map from stage $s$ of BEiT (reshaped spatially).
            FPN combines features: $F_{out}^{(s)} = \text{Conv}(H^{(s)}) + \text{Upsample}(\text{Conv}(F_{out}^{(s+1)}))$.
        2.  **Pyramid Pooling Module (PPM):** The finest FPN level features are fed into a PPM, which applies pooling at multiple scales, concatenates the results, and uses a convolution to fuse them.
            $P_{PPM} = \text{Concat}(\text{Pool}_1(F_{FPN}^{top}), \dots, \text{Pool}_k(F_{FPN}^{top}))$.
            $F_{fused} = \text{Conv}(\text{Concat}(F_{FPN}^{top}, \text{Upsample}(P_{PPM})))$.
        3.  **Final Prediction:** The fused features are progressively upsampled (often with skip connections from FPN layers) and a final convolutional layer predicts the class logits for each pixel.
            $S_{logits} = \text{Conv}(\text{Upsample}(\dots \text{Upsample}(F_{fused})\dots)) \in \mathbb{R}^{H_{img} \times W_{img} \times N_{classes}}$.
    *   *Implementation in PyTorch/TensorFlow involves taking feature maps from various layers of the BEiT backbone, reshaping them spatially, and feeding them to the corresponding UPerNet modules (`mmsegmentation` provides UPerNet implementations).*

#### 2. Data Pre-processing
*   **Resize/Crop:** Images and their corresponding segmentation masks are resized to a fixed input size (e.g., 512x512, 640x640). Random cropping with constraints to maintain object presence, random horizontal flipping (masks must be flipped too).
*   **Normalization:** Image pixel values normalized.
*   **Augmentations:** Random scaling, random rotation.

#### 3. Post-processing
*   The output segmentation map logits $S_{logits}$ are passed through an argmax operation per pixel to get the predicted class labels.
    $$ S_{pred}(i,j) = \arg\max_c S_{logits}(i,j,c) $$
*   If input size was different from original image size, the predicted map $S_{pred}$ is resized to the original image dimensions using nearest-neighbor interpolation to maintain discrete class labels.

#### 4. Training Pseudo-Algorithm (Semantic Segmentation Fine-tuning)
1.  Initialize BEiT backbone with pre-trained weights $\theta_{BEiT}^*$.
2.  Initialize segmentation decoder (e.g., UPerNet) $\theta_{decoder}$.
3.  Setup optimizer (e.g., AdamW for $\theta_{BEiT}^* \cup \theta_{decoder}$).
4.  `for epoch in range(num_fine_tune_epochs):`
    `    for images_batch, masks_batch in enumerate(segmentation_dataloader):`
    `        # Pre-process images_batch (augment, patchify, add pos_embed)`
    `        input_sequences = preprocess_for_vit(images_batch)`
    `        # Forward pass through BEiT backbone`
    `        # Extract features from multiple stages/layers if decoder requires (e.g., UPerNet)`
    `        # For a simple ViT, take final layer patch embeddings`
    `        # Reshape patch embeddings to spatial feature map(s)`
    `        backbone_features = be_it_backbone.get_features(input_sequences)`
    `        # Forward pass through decoder`
    `        segmentation_logits = decoder(backbone_features) # Output: [B, H_out, W_out, N_classes]`
    `        # Upsample logits to match mask_batch resolution if necessary`
    `        segmentation_logits_resized = F.interpolate(segmentation_logits.permute(0,3,1,2), size=masks_batch.shape[1:], mode='bilinear', align_corners=False)`
    `        # Loss Calculation (pixel-wise cross-entropy)`
    `        loss = nn.CrossEntropyLoss(ignore_index=ignore_label_value)(segmentation_logits_resized, masks_batch.long())`
    `        # Backward pass and optimize`
    `        optimizer.zero_grad()`
    `        loss.backward()`
    `        optimizer.step()`
    `        # (Update LR scheduler)`

#### 5. Evaluation Metrics and Loss Function
*   **Loss Function (Training):**
    *   **Pixel-wise Cross-Entropy Loss:**
        $$ L_{SegCE} = - \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \sum_{c=1}^{N_{classes}} M_{ijc} \log(P_{ijc}) $$
        where $M_{ijc}$ is 1 if pixel $(i,j)$ belongs to class $c$, and $P_{ijc}$ is the predicted probability of pixel $(i,j)$ being class $c$. Often an `ignore_index` is used for unlabeled pixels.
    *   **Dice Loss:** Can be beneficial for imbalanced classes.
        $$ L_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i^2 + \sum_i g_i^2 + \epsilon} $$ (for binary, can be extended to multi-class)
    *   **Focal Loss:** To down-weight well-classified examples and focus on hard ones.
        $$ L_{Focal} = - \alpha_t (1-p_t)^\gamma \log(p_t) $$
*   **Metrics (SOTA and Standard):**
    *   **Mean Intersection over Union (mIoU):** The primary metric for semantic segmentation.
        For a class $c$: $\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$
        $$ \text{mIoU} = \frac{1}{N_{classes}} \sum_{c=1}^{N_{classes}} \text{IoU}_c $$
    *   **Pixel Accuracy (PA):** Overall percentage of correctly classified pixels.
        $$ \text{PA} = \frac{\sum_c TP_c}{\text{Total Number of Pixels}} = \frac{\sum_c TP_c}{\sum_c (TP_c + FP_c)} $$ (sum over classes in denominator if considering only valid classes)
    *   **Mean Pixel Accuracy (MPA) / Mean Class Accuracy (mAcc):** Average of per-class pixel accuracies.
        $$ \text{MPA} = \frac{1}{N_{classes}} \sum_{c=1}^{N_{classes}} \frac{TP_c}{TP_c + FN_c} $$

### D. Intermediate Fine-tuning

#### 1. Definition
Intermediate fine-tuning is a step between the initial self-supervised pre-training (like BEiT's MIM) and the final downstream task fine-tuning. It involves fine-tuning the pre-trained model on a large, labeled dataset from a related, but usually more general, supervised task (e.g., ImageNet-1k classification).

#### 2. Rationale and Mathematical Underpinnings
*   **Rationale:**
    *   **Adaptation:** MIM learns general visual representations. Intermediate fine-tuning on a supervised task like ImageNet classification helps adapt these features to be more discriminative for semantic categories present in natural images.
    *   **Bridging Domain Gaps:** If the downstream task dataset is small or has a different data distribution than the MIM pre-training data, intermediate fine-tuning on a large, relevant dataset can bridge this gap.
    *   **Improved Initialization:** Provides a better weight initialization for the final downstream task, often leading to faster convergence and higher performance.
*   **Mathematical Underpinnings:** The process is identical to standard supervised fine-tuning for the intermediate task (e.g., image classification as described in IV.B).
    *   Model: BEiT backbone + classification head.
    *   Loss: Cross-Entropy Loss on the intermediate dataset labels.
    *   Objective: $\min_{\theta_{BEiT}, \theta_{head\_interm}} L_{CE}(\text{Model}(X_{interm}), Y_{interm})$.
    The resulting $\theta_{BEiT}^*$ (from this intermediate stage) is then used for the final downstream task.

#### 3. Procedure
1.  **BEiT Pre-training:** Train BEiT using MIM on a large (unlabeled) dataset (e.g., ImageNet-21k without labels, or custom large dataset). Obtain $\theta_{BEiT\_MIM}$.
2.  **Intermediate Supervised Fine-tuning:**
    *   Load $\theta_{BEiT\_MIM}$.
    *   Add a classification head.
    *   Fine-tune on a large labeled dataset (e.g., ImageNet-1k classification task).
    *   Save the backbone weights after this stage: $\theta_{BEiT\_interm}$.
3.  **Final Downstream Task Fine-tuning:**
    *   Load $\theta_{BEiT\_interm}$.
    *   Add the task-specific head for the final task (e.g., segmentation decoder).
    *   Fine-tune on the target downstream dataset (e.g., ADE20K for segmentation).

#### 4. Importance
*   **Performance Boost:** Often significantly improves performance on downstream tasks, especially when the target dataset is small or dissimilar to the MIM pre-training data.
*   **Common Practice:** For many SOTA results using ViTs/BEiT, intermediate fine-tuning (e.g., on ImageNet-1k or ImageNet-21k with labels) is a standard step.
*   **Robustness:** Can make the model more robust and better at generalizing.

**Best Practices and Potential Pitfalls:**
*   **Learning Rates:** Use smaller learning rates for fine-tuning compared to pre-training. Differential learning rates (smaller for backbone, larger for head) can be beneficial.
*   **Regularization:** Continue using weight decay. Stochastic depth is often used during ViT fine-tuning.
*   **Resolution Mismatch:** Fine-tuning at a higher resolution than pre-training is possible but may require adjusting positional embeddings (e.g., via interpolation).
*   **Dataset Size for Fine-tuning:** If the downstream dataset is very small, extensive fine-tuning of the backbone might lead to overfitting. Consider freezing more layers or using stronger regularization.
*   **Tokenizer Consistency:** For BEiT, the visual tokenizer is fixed. The patch size and other fundamental architectural choices (e.g., $D$) must be consistent between pre-training and fine-tuning of the backbone.