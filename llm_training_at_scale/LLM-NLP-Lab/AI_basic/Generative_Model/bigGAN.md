### BigGAN

#### I. Definition
BigGAN (Brock et al., 2018) is a class-conditional Generative Adversarial Network (GAN) designed for synthesizing high-resolution, high-fidelity images. It achieved then state-of-the-art results on large-scale datasets like ImageNet by scaling up existing GAN architectures and introducing several key architectural and training modifications, including Conditional Batch Normalization, Self-Attention, Spectral Normalization, a projection-based discriminator, and the "truncation trick" for improved sample quality at the cost of variety.

#### II. Pertinent Equations (and Core Model Architecture)

**A. Generator (G) Architecture Overview**

The Generator $G$ maps a latent noise vector $z$ and a class embedding $c$ to an image $G(z, c)$.

*   **1. Input:**
    *   Latent Vector $z \in \mathbb{R}^{N_z}$: Sampled from a distribution, typically $\mathcal{N}(0, I)$. $N_z$ is the dimension of the noise vector (e.g., 128).
    *   Class Label $y$: An integer representing the class.
*   **2. Shared Class Embeddings:**
    The class label $y$ is mapped to a dense embedding vector $c_{emb} \in \mathbb{R}^{N_c}$ via a learnable embedding layer $E$:
    $$c_{emb} = E(y)$$
    This embedding $c_{emb}$ is used in multiple ways:
    *   Linearly projected to obtain gains $\gamma$ and biases $\beta$ for Conditional Batch Normalization layers.
    *   Optionally split and concatenated with chunks of $z$ at different resolution blocks. For BigGAN, $c_{emb}$ is typically concatenated with $z$ only at the initial block after $z$ is processed.
*   **3. Conditional Batch Normalization (CBN):**
    CBN modulates the normalized activations $x_{norm}$ using class-conditional gains $\gamma(c_{emb})$ and biases $\beta(c_{emb})$. Given an intermediate activation map $h_i$ in a layer $i$:
    *   Standard Batch Normalization:
        $$\mu_B = \frac{1}{M} \sum_{m=1}^{M} h_{i,m}$$
        $$\sigma_B^2 = \frac{1}{M} \sum_{m=1}^{M} (h_{i,m} - \mu_B)^2$$
        $$\hat{h}_{i,m} = \frac{h_{i,m} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
        where $M$ is the batch size and $\epsilon$ is a small constant for numerical stability.
    *   CBN output:
        $$ \text{CBN}(h_i, c_{emb}) = \gamma_i(c_{emb}) \cdot \hat{h}_i + \beta_i(c_{emb}) $$
        where $\gamma_i(c_{emb})$ and $\beta_i(c_{emb})$ are per-channel scaling and shifting parameters learned by projecting the class embedding $c_{emb}$ using separate linear layers for each CBN layer $i$:
        $$ \gamma_i(c_{emb}) = W_{\gamma,i} c_{emb} + b_{\gamma,i} $$
        $$ \beta_i(c_{emb}) = W_{\beta,i} c_{emb} + b_{\beta,i} $$
*   **4. Residual Blocks (Upsampling):**
    The Generator uses a series of residual blocks. Each block typically consists of:
    *   Upsampling (e.g., nearest-neighbor or bilinear interpolation).
    *   Convolutional layers (e.g., $3 \times 3$ Conv).
    *   Conditional Batch Normalization.
    *   Non-linear activation (e.g., ReLU).
    A typical ResBlock structure:
    $$ h_{out} = \text{Conv}(\text{ReLU}(\text{CBN}(\text{Conv}(\text{ReLU}(\text{CBN}(\text{Upsample}(h_{in}), c_{emb}))), c_{emb}))) + \text{Shortcut}(h_{in}) $$
    The shortcut connection $\text{Shortcut}(h_{in})$ might involve an upsampling and a $1 \times 1$ convolution to match dimensions if they change within the block.
*   **5. Self-Attention Module (G):**
    Incorporated at specific resolutions (e.g., $64 \times 64$ feature maps) to model long-range dependencies.
    Given input feature maps $x \in \mathbb{R}^{C \times H \times W}$:
    *   Queries: $q(x) = W_q x$
    *   Keys: $k(x) = W_k x$
    *   Values: $v(x) = W_v x$
    (where $W_q, W_k, W_v$ are $1 \times 1$ convolutional layers).
    *   Attention scores: $s_{ij} = q(x_i)^T k(x_j)$
    *   Attention map $\beta_{j,i}$:
        $$ \beta_{j,i} = \frac{\exp(s_{ij})}{\sum_{l=1}^{N} \exp(s_{il})} $$
        where $N=H \times W$ is the number of feature locations.
    *   Attention output: $o_j = \sum_{i=1}^{N} \beta_{j,i} v(x_i)$
    *   Final output: $y_i = \alpha_{SA} o_i + x_i$, where $\alpha_{SA}$ is a learnable scalar parameter, initialized to 0.

**B. Discriminator (D) Architecture Overview**

The Discriminator $D$ takes an image $x$ (real or generated) and outputs a scalar score indicating its realness, conditioned on the class label $y$.

*   **1. Input:**
    *   Image $x \in \mathbb{R}^{C_{img} \times H_{img} \times W_{img}}$ (real or $G(z,c)$).
    *   Class label $y$.
*   **2. Residual Blocks (Downsampling):**
    Mirrors the generator structure but uses downsampling (e.g., average pooling or strided convolutions) within residual blocks. These blocks typically do not use batch normalization but rely on Spectral Normalization for stability.
    A typical ResBlock structure:
    $$ h_{out} = \text{Conv}(\text{ReLU}(\text{Conv}(\text{ReLU}(h_{in})))) + \text{Shortcut}(\text{Downsample}(h_{in})) $$
    All convolutional layers are spectrally normalized.
*   **3. Self-Attention Module (D):**
    Similar to the generator's self-attention module, applied at specific feature map resolutions (e.g., $64 \times 64$). The formulation is identical.
*   **4. Projection Discriminator:**
    The discriminator's output is conditioned on the class label $y$.
    Let $\phi(x)$ be the feature representation of image $x$ from the penultimate layer of the discriminator.
    Let $E_D(y) \in \mathbb{R}^{N_D}$ be a learned embedding for class $y$ in the discriminator (distinct from $E(y)$ in G).
    The discriminator output $D(x, y)$ is computed as:
    $$ D(x, y) = W_{out}^T \phi(x) + b_{out} + (W_{proj} \phi(x))^T E_D(y) $$
    The term $W_{out}^T \phi(x) + b_{out}$ is the "unconditional" part, and $(W_{proj} \phi(x))^T E_D(y)$ is the "conditional" part, projecting $\phi(x)$ into the same space as $E_D(y)$ before taking the inner product. Often, $W_{proj}$ is an identity or omitted, and $E_D(y)$ is directly multiplied with $\phi(x)$, after $\phi(x)$ has been globally summed or averaged.
    More simply, for a hinge loss variant:
    The main branch output: $f_D(\phi(x))$ (e.g., a linear layer on global sum-pooled $\phi(x)$).
    The conditional term: $E_D(y)^T \phi(x)$ (where $\phi(x)$ is the global sum-pooled features).
    $$ D_{score}(x, y) = f_D(\phi(x)) + E_D(y)^T (\text{GlobalSumPool}(\phi(x))) $$
    (Note: The exact formulation by Miyato & Koyama for projection D uses $D(x,y) = \sigma(v^T \phi(x) + \mathbf{e}_y^T \phi(x))$ for sigmoid loss. BigGAN adapts this to hinge loss.)
    For hinge loss, the logits are directly used.
    The output from the discriminator's backbone $\phi(x)$ is globally sum-pooled. This vector is then used for two terms:
    1.  Unconditional term: $h_u = W_u (\text{GlobalSumPool}(\phi(x)))$ (a scalar).
    2.  Conditional term: $h_c = E_D(y)^T (\text{GlobalSumPool}(\phi(x)))$ (a scalar).
    The final logit is $D_{logit}(x, y) = h_u + h_c$.

**C. Spectral Normalization (SN)**

Applied to all weight matrices $W$ in both G and D (except CBN gains/biases and class embeddings).
$$ W_{SN} = \frac{W}{\sigma(W)} $$
where $\sigma(W)$ is the spectral norm (largest singular value) of $W$. $\sigma(W)$ is estimated efficiently using one iteration of the power iteration method:
Initialize $u_0$ randomly.
$$ v_1 = \frac{W^T u_0}{||W^T u_0||_2} $$
$$ u_1 = \frac{W v_1}{||W v_1||_2} $$
$$ \sigma(W) \approx u_1^T W v_1 $$
During training, $u$ is stored and updated.

**D. Loss Functions**

BigGAN employs the hinge version of the adversarial loss.

*   **1. Hinge Adversarial Loss:**
    *   Discriminator Loss $L_D$:
        $$ L_D = -\mathbb{E}_{(x,y) \sim p_{data}}[\min(0, -1 + D_{logit}(x, y))] - \mathbb{E}_{z \sim p_z, y' \sim p_{class}}[\min(0, -1 - D_{logit}(G(z, c_{emb}(y')), y'))] $$
        where $p_{data}$ is the real data distribution, $p_z$ is the noise distribution, and $p_{class}$ is the distribution of class labels (typically uniform over training classes or sampled from data). $c_{emb}(y')$ is the class embedding for label $y'$.
    *   Generator Loss $L_G$:
        $$ L_G = -\mathbb{E}_{z \sim p_z, y' \sim p_{class}}[D_{logit}(G(z, c_{emb}(y')), y')] $$
        (Optionally, Orthogonal Regularization is added to $L_G$).

#### III. Key Principles

*   **A. Scaling Hypothesis:** Larger models (more parameters, deeper architectures) and larger batch sizes significantly improve GAN performance, particularly image fidelity and Inception Score. BigGAN systematically explored this.
*   **B. Architectural Stability:**
    *   **Spectral Normalization (SN):** Crucial for stabilizing training of large GANs by constraining the Lipschitz constant of the discriminator (and generator in BigGAN).
    *   **Residual Blocks:** Allow for training very deep networks by facilitating gradient flow.
*   **C. Explicit Conditioning:**
    *   **Conditional Batch Normalization (CBN):** Allows class information to modulate feature maps throughout the generator, leading to better class-conditional generation.
    *   **Projection Discriminator:** Provides a direct way for the discriminator to leverage class information, improving its ability to distinguish real from fake samples conditioned on class.
*   **D. Fidelity-Variety Trade-off (Truncation Trick):** A post-hoc technique to improve individual sample quality (fidelity) by truncating the latent space, at the expense of overall sample diversity (variety).
*   **E. Orthogonal Regularization for Generator Conditioning:** Applying orthogonal regularization to the generator's weights improves the conditioning of the weight matrices, making the generator less sensitive to small changes in input $z$, which aids in training stability and sample quality.

#### IV. Detailed Concept Analysis

**A. Data Pre-processing**

*   **1. Image Resizing and Normalization:**
    *   Images are resized to the target resolution (e.g., 128x128, 256x256, 512x512) using high-quality interpolation (e.g., bilinear or bicubic).
    *   Pixel values are normalized to the range $[-1, 1]$. If original pixels are in $[0, 255]$, the transformation is $x' = (x / 127.5) - 1$.
*   **2. Class Label Preparation:**
    *   Class labels $y$ are provided as integers. These are used to look up embeddings $c_{emb} = E(y)$.

**B. Generator (G) In-Depth**

*   **1. Latent Vector Processing ($z$ splitting, linear projection):**
    *   The input noise vector $z \in \mathbb{R}^{N_z}$ (e.g., $N_z=128$) is typically split into chunks. For example, if $N_z = 120$ and there are 6 residual blocks, $z$ might be split into 6 chunks of dimension 20.
    *   The first chunk is typically processed by a linear layer and reshaped to form the initial $4 \times 4$ feature map.
    *   Subsequent chunks of $z$ are concatenated with the class embedding $c_{emb}$ and fed into different residual blocks, effectively creating a hierarchical latent space. However, the primary BigGAN architecture concatenates $c_{emb}$ with the full $z$ (or a projection of $z$) at the input block. Simpler variants use $c_{emb}$ only for CBN. BigGAN's specific choice was to have a shared embedding $c_{shared}$ (dimension 128) derived from $y$, which is then split. One part (20 dimensions) is concatenated to each 20-dimension chunk of $z$. The remaining $c_{shared}$ (e.g., $128 - 20 = 108$ if $N_z=120$ split into one chunk for initial block, and $c_{shared}$ has 20 dims for concatenation, remainder 108 for CBN projection) is projected to get CBN parameters.
    *   The actual BigGAN-deep architecture: $z \in \mathbb{R}^{128}$. $c_{emb} \in \mathbb{R}^{128}$. $z$ is projected to $4 \times 4 \times (16 \cdot \text{ch})$ features. $c_{emb}$ is fed to all CBNs.
*   **2. Class Embeddings for CBN ($\gamma(c_{emb})$, $\beta(c_{emb})$ generation):**
    The shared class embedding $c_{emb}$ is projected by learned linear layers $W_{\gamma,i}, b_{\gamma,i}$ and $W_{\beta,i}, b_{\beta,i}$ for each Conditional Batch Norm layer $i$ to produce per-channel scaling factors $\gamma_i$ and biases $\beta_i$. This allows fine-grained control over feature modulation based on class.
*   **3. Hierarchical Latent Space:**
    By injecting (parts of) $z$ and $c_{emb}$ at different depths of the generator, the model can learn to use different components of the latent code to control features at different scales. BigGAN is less explicit about this than StyleGAN, primarily using $z$ at the input and $c_{emb}$ for CBN throughout. The "chunking" mechanism for $z$ and $c_{emb}$ in the original BigGAN paper is a specific detail: $z \in \mathbb{R}^{120}$, $c_{emb} \in \mathbb{R}^{128}$. $z$ is input. For each ResBlock, $c_{emb}$ is split into two parts, one for CBN parameters and one for concatenation with the hidden state input to the ResBlock.

**C. Discriminator (D) In-Depth**

*   **1. Feature Extraction Backbone:**
    A deep ResNet-style architecture, typically mirroring the generator's structure but with downsampling instead of upsampling and no batch normalization. Spectral Normalization is applied to all convolutional layers.
*   **2. Projection Mechanism Details:**
    The projection term $E_D(y)^T (\text{GlobalSumPool}(\phi(x)))$ allows the discriminator to explicitly check if the image features $\phi(x)$ are consistent with the claimed class $y$. $E_D(y)$ are learnable class embeddings for the discriminator. This formulation provides a strong class-conditional signal to both D and G.

**D. Training Procedure**

*   **1. Adam Optimizer with Custom Betas:**
    Adam optimizer with $\beta_1=0.0, \beta_2=0.999$ and learning rate $lr_G = 5 \times 10^{-5}$, $lr_D = 2 \times 10^{-4}$ (for ImageNet 128x128). These can vary based on resolution and batch size.
*   **2. Two Discriminator Updates per Generator Update ($N_D_steps = 2, N_G_steps = 1$):**
    This helps ensure the discriminator remains optimal or near-optimal relative to the generator.
*   **3. Moving Averages of G's Weights:**
    An exponential moving average (EMA) of the generator's weights is often kept during training. This EMA copy ($G_{EMA}$) tends to produce better quality samples and is used for evaluation/inference.
    $$ \theta_{G_{EMA}}^{(t)} = \text{decay} \cdot \theta_{G_{EMA}}^{(t-1)} + (1 - \text{decay}) \cdot \theta_G^{(t)} $$
    Typical decay is $0.9999$.
*   **4. Training Pseudo-algorithm (BigGAN):**
    ```pseudo
    Initialize G parameters $\theta_G$, D parameters $\theta_D$.
    Initialize EMA G parameters $\theta_{G_{EMA}} \leftarrow \theta_G$.
    For each training iteration:
      // Discriminator Update Phase
      For N_D_steps iterations:
        1. Sample minibatch of N real images {x_1, ..., x_N} with labels {y_1, ..., y_N} from p_data.
        2. Sample minibatch of N noise vectors {z_1, ..., z_N} from p_z.
        3. Sample minibatch of N class labels {y'_1, ..., y'_N} from p_class.
        4. Obtain class embeddings c_emb(y') for G.
        5. Generate fake images: x_f_i = G(z_i, c_emb(y'_i); \theta_G).
        6. Compute D_logit_real_i = D(x_i, y_i; \theta_D).
        7. Compute D_logit_fake_i = D(x_f_i, y'_i; \theta_D).
        8. Compute L_D = mean(max(0, 1 - D_logit_real_i)) + mean(max(0, 1 + D_logit_fake_i)).
           (Equation from paper: L_D = -E[min(0, -1+D(x,y))] - E[min(0, -1-D(G(z,y'),y'))] )
        9. Update $\theta_D \leftarrow \text{AdamUpdate}(\theta_D, \nabla_{\theta_D} L_D, lr_D)$.

      // Generator Update Phase
      For N_G_steps iterations:
        1. Sample minibatch of N noise vectors {z_1, ..., z_N} from p_z.
        2. Sample minibatch of N class labels {y'_1, ..., y'_N} from p_class.
        3. Obtain class embeddings c_emb(y') for G.
        4. Generate fake images: x_f_i = G(z_i, c_emb(y'_i); \theta_G).
        5. Compute D_logit_fake_i = D(x_f_i, y'_i; \theta_D).
        6. Compute L_G_adv = -mean(D_logit_fake_i).
           (Equation from paper: L_G = -E[D(G(z,y'),y')] )
        7. If using Orthogonal Regularization:
           Compute R_ortho for G weights.
           L_G = L_G_adv + \lambda_{ortho} R_{ortho}.
        8. Update $\theta_G \leftarrow \text{AdamUpdate}(\theta_G, \nabla_{\theta_G} L_G, lr_G)$.
        9. Update $\theta_{G_{EMA}} \leftarrow \text{decay} \cdot \theta_{G_{EMA}} + (1 - \text{decay}) \cdot \theta_G$.
    ```
*   **5. Orthogonal Regularization (Detailed):**
    Applied to the weight matrices $W$ of the generator (excluding biases, gains, and embeddings).
    $$ R_{ortho}(W) = \beta_{reg} ||W^T W - I||_F^2 $$
    where $|| \cdot ||_F^2$ is the squared Frobenius norm, and $I$ is the identity matrix. This encourages the rows (and columns) of $W$ to be orthogonal, which can improve conditioning and training stability. The hyperparameter $\beta_{reg}$ controls the strength of this regularization (e.g., $10^{-4}$).

**E. Post-Training Procedures**

*   **1. Truncation Trick (Mathematical Formulation & Impact):**
    To improve sample fidelity, $z$ vectors are sampled from a truncated normal distribution. For a given truncation threshold $\psi > 0$:
    Sample $z_i \sim \mathcal{N}(0, 1)$. If $|z_i| > \psi$, resample $z_i$ until $|z_i| \le \psi$.
    This is done for each component of $z$.
    The distribution is $p_{trunc}(z; \psi) = \frac{1}{Z(\psi)} \prod_{i=1}^{N_z} \mathbb{I}(|z_i| < \psi) \mathcal{N}(z_i | 0, 1)$, where $Z(\psi)$ is the normalization constant.
    *   **Impact:**
        *   Smaller $\psi$ (e.g., 0.04, 0.5) values lead to higher fidelity (sharper, more prototypical images) but lower diversity (fewer variations, mode concentration).
        *   $\psi=1.0$ approximates sampling from the original $\mathcal{N}(0,1)$ with minimal truncation.
        *   $\psi=0$ would mean $z=0$, yielding the "average" image for each class.
    The generator $G_{EMA}$ (with moving averaged weights) is used for generating samples with the truncation trick.

**F. Evaluation Phase**

Metrics used to assess the performance of BigGAN, primarily focusing on image quality and diversity.

*   **1. Inception Score (IS):**
    Measures both fidelity (are individual images recognizable and sharp?) and diversity (are various classes represented?).
    $$ IS = \exp \left( \mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) || p(y))] \right) $$
    *   $x \sim p_g$: An image sampled from the generator $G$.
    *   $p(y|x)$: The conditional class distribution predicted by a pre-trained Inception network for image $x$. High fidelity samples should have low entropy $p(y|x)$.
    *   $p(y) = \int_x p(y|x) p_g(x) dx$: The marginal class distribution over all generated samples. High diversity samples should have high entropy $p(y)$.
    *   $D_{KL}$: Kullback-Leibler divergence.
    *   Higher IS is better. Calculated on a large set of generated samples (e.g., 50k).
*   **2. Fréchet Inception Distance (FID):**
    Measures the similarity between the distribution of generated images and real images in the feature space of a pre-trained Inception network.
    $$ FID(p_r, p_g) = ||\mu_r - \mu_g||_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}) $$
    *   $\mu_r, \Sigma_r$: Mean and covariance matrix of Inception activations for real images from $p_r$.
    *   $\mu_g, \Sigma_g$: Mean and covariance matrix of Inception activations for generated images from $p_g$.
    *   Tr: Trace of a matrix.
    *   Lower FID is better, indicating generated image distribution is closer to real image distribution. Typically more robust and correlates better with human perception than IS.
*   **3. Loss Values:**
    During training, monitoring $L_D$ and $L_G$ provides insights into training stability and convergence, but they are not direct measures of image quality.

#### V. Importance

*   **State-of-the-Art Image Generation:** BigGAN significantly advanced the SOTA in class-conditional image generation, producing images on ImageNet at resolutions up to $512 \times 512$ with unprecedented fidelity and diversity (pre-truncation).
*   **Demonstration of Scaling Benefits:** It provided strong evidence that scaling up GAN model capacity (parameters) and batch sizes leads to substantial improvements in generation quality. This influenced subsequent research in large-scale generative modeling.
*   **Introduction/Popularization of Techniques:** While some components like Self-Attention and Spectral Normalization were adapted from prior work (e.g., SAGAN), BigGAN demonstrated their effectiveness at a much larger scale and introduced/popularized the truncation trick and orthogonal regularization in this context.
*   **Benchmark for Future Research:** BigGAN became a critical benchmark for subsequent GAN research and other generative models aiming to achieve high-fidelity image synthesis.

#### VI. Pros versus Cons

**Pros:**

*   **High Fidelity:** Generates exceptionally realistic and detailed images, especially when using the truncation trick.
*   **High Resolution:** Capable of generating images at resolutions like $256 \times 256$ and $512 \times 512$.
*   **Class Controllability:** Effective class-conditional generation due to CBN and projection discriminator.
*   **Good Diversity (pre-truncation):** Can generate a wide variety of samples within each class when not heavily truncating the latent space.
*   **Stable Training at Scale:** Achieved stable training for very large models, which was a significant challenge.

**Cons:**

*   **Computationally Expensive:** Requires substantial computational resources (GPUs, TPUs) and long training times due to large model size and batch sizes.
*   **Truncation Trade-off:** The truncation trick, while improving fidelity, significantly reduces sample diversity.
*   **Mode Collapse/Dropping:** While mitigated, GANs including BigGAN can still suffer from mode collapse (failing to capture all modes of the data distribution) or mode dropping (missing entire classes or sub-classes), especially at very high resolutions or complex datasets without sufficient scale.
*   **Sensitivity to Hyperparameters:** Performance can be sensitive to choices of learning rates, regularization strengths, and architectural details.
*   **Complexity:** The architecture and training regimen are complex, involving multiple specialized components.

#### VII. Cutting-Edge Advances (Post-BigGAN)

Since BigGAN, the field of generative modeling has seen further significant advancements:

*   **StyleGAN Series (StyleGAN, StyleGAN2, StyleGAN2-ADA, StyleGAN3):**
    *   Introduced a style-based generator architecture with improved disentanglement of latent factors, new normalization techniques (Adaptive Instance Normalization - AdaIN), and progressive focus on equivariances.
    *   StyleGAN2-ADA significantly improved performance with limited training data.
*   **Diffusion Models (e.g., DDPM, IDDPM, GLIDE, DALL-E 2, Imagen, Stable Diffusion):**
    *   Emerged as a dominant paradigm, often surpassing GANs in image fidelity and diversity.
    *   Work by iteratively denoising an image, guided by a learned model.
    *   Computationally intensive for sampling but can achieve outstanding results.
*   **Transformers for Image Generation (e.g., VQ-VAE + Transformer, ViT-VQGAN, Parti):**
    *   Leverage the power of transformers, often in conjunction with discrete VAEs (VQ-VAE/VQGAN), to model image distributions autoregressively or via diffusion in latent space.
    *   Capable of generating highly coherent and diverse images.
*   **Improved GAN Training Techniques:**
    *   Contrastive losses (e.g., ContraGAN).
    *   Data augmentation strategies (e.g., Differentiable Augmentation).
    *   Further regularization methods and architectural refinements.
*   **Neural Radiance Fields (NeRF) and 3D-Aware GANs:**
    *   Focus on generating 3D-consistent representations and views of objects/scenes.