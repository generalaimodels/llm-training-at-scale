## Diffusion Models

### 1. Definition
Diffusion Models are a class of probabilistic generative models that learn to synthesize data by simulating a dual process: a fixed **forward diffusion process** that gradually perturbs data with noise until it resembles pure noise, and a learned **reverse diffusion process** that iteratively denoises samples, starting from noise, to produce data samples. They are defined by a Markov chain where each forward step adds a small amount of Gaussian noise, and the reverse process aims to undo these additions.

### 2. Pertinent Equations

#### 2.1 Forward Diffusion Process ($q$)
The forward process $q$ gradually adds Gaussian noise to a data sample $x_0 \sim q(x_0)$ over $T$ discrete timesteps.
*   **Single step transition:**
    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$
    where $\{\beta_t\}_{t=1}^T$ is a variance schedule (hyperparameters, e.g., linearly increasing from $\beta_1 \approx 10^{-4}$ to $\beta_T \approx 0.02$).
*   Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
*   **Sampling $x_t$ directly from $x_0$:**
    $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
    This can be written as:
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

#### 2.2 Reverse Diffusion Process ($p_\theta$)
The reverse process $p_\theta$ is learned to approximate the true posterior $q(x_{t-1} | x_t, x_0)$. It starts from $p(x_T) = \mathcal{N}(x_T; 0, I)$ and generates data by iteratively sampling.
*   **Single step transition:**
    $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$
*   The true posterior $q(x_{t-1} | x_t, x_0)$ is tractable and Gaussian:
    $$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$
    where:
    $$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$
    $$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$
*   The model $\epsilon_\theta(x_t, t)$ is trained to predict the noise $\epsilon$ from $x_t$. The mean $\mu_\theta$ is then parameterized as:
    $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
*   The covariance $\Sigma_\theta(x_t, t)$ is often set to be time-dependent constants: $\Sigma_\theta(x_t, t) = \sigma_t^2 I$. Common choices are $\sigma_t^2 = \beta_t$ or $\sigma_t^2 = \tilde{\beta}_t$.

#### 2.3 Objective Function (Loss)
The model is trained by maximizing the Evidence Lower Bound (ELBO) on the log-likelihood:
$$L_{VLB} = \mathbb{E}_{q(x_0)} \left[ D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1) \right]$$
A simplified training objective, empirically found to work well (DDPM, Ho et al. 2020), focuses on predicting the noise component:
$$L_{simple}(\theta) = \mathbb{E}_{t \sim U(1,T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]$$
where $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$.

### 3. Key Principles
*   **Markovian Structure:** Both forward and reverse processes are Markov chains, meaning the current state depends only on the previous state.
*   **Fixed Forward Process:** The noising process $q$ is predefined and does not involve learnable parameters. Its schedule $\beta_t$ is fixed.
*   **Learned Reverse Process:** The denoising process $p_\theta$ is parameterized by a neural network (typically a UNet architecture) and learned from data.
*   **Variational Inference:** The parameters $\theta$ are learned by optimizing a variational lower bound on the data log-likelihood. The simplified objective corresponds to a reweighted VLB.
*   **Noise Prediction:** Instead of directly predicting the mean $\mu_\theta(x_t, t)$ of the denoised $x_{t-1}$, models typically predict the noise $\epsilon$ that was added to $x_0$ to obtain $x_t$. This has proven to be a more stable and effective parameterization.
*   **Gaussian Assumptions:** Both conditional distributions in the forward process and the parameterized reverse process are assumed to be Gaussian.

### 4. Detailed Concept Analysis

#### 4.1 Data Pre-processing
*   **Normalization:** Input data $x_0$ (e.g., images) are typically normalized to a standard range. For images with pixel values in $[0, 255]$, a common normalization is to scale them to $[-1, 1]$:
    $$x_{0,norm} = (x_0 / 127.5) - 1$$
    This ensures that the input data has a similar scale to the noise $\epsilon \sim \mathcal{N}(0,I)$ added during the diffusion process, particularly at later timesteps $t \approx T$.

#### 4.2 Model Architecture (Core Model $\epsilon_\theta$)
The neural network $\epsilon_\theta(x_t, t)$ aims to predict the noise $\epsilon$ given the noisy input $x_t$ and the timestep $t$. A UNet architecture is commonly used.

*   **Input:**
    *   Noisy data $x_t$ (e.g., an image tensor of shape $(C, H, W)$).
    *   Timestep $t$ (an integer from $1$ to $T$).
*   **Timestep Embedding:** The discrete timestep $t$ is converted into a continuous embedding vector $t_{emb}$.
    *   **Sinusoidal Positional Embedding (inspired by Transformers):**
        Let $d$ be the embedding dimension. For $i = 0, ..., d/2 - 1$:
        $$PE(t, 2i) = \sin(t / 10000^{2i/d})$$
        $$PE(t, 2i+1) = \cos(t / 10000^{2i/d})$$
    *   This $PE(t)$ vector is often further processed by a small Multi-Layer Perceptron (MLP):
        $$t_{emb} = MLP(PE(t))$$
*   **UNet Architecture:**
    *   **Encoder:**
        *   Consists of multiple downsampling blocks. Each block typically includes:
            *   Convolutional layers (often two or more).
            *   Normalization layers (e.g., Group Normalization).
            *   Activation functions (e.g., SiLU/Swish: $x \cdot \sigma(x)$, GeLU).
            *   Residual connections.
            *   Timestep embedding $t_{emb}$ is incorporated, usually by adding it (after a linear projection) to the feature maps or concatenating it.
            *   Downsampling operations (e.g., strided convolution, average pooling).
            *   Self-attention blocks are often interspersed, especially at lower resolutions, to capture global context.
        *   Let $h_0^{enc} = x_t$. For block $l=0, \dots, L_{enc}-1$:
            $h_{l+1}^{enc} = Downsample_l(Block_{enc,l}(h_l^{enc}, t_{emb}))$
            A $Block_{enc,l}$ might be:
            $h' = Conv(GroupNorm(Activation(h_l^{enc})))$
            $h' = h' + Linear_{time,l}(t_{emb})$ (broadcast and add)
            $h'' = Conv(GroupNorm(Activation(h')))$
            $h_{out} = h'' + ResConv_l(h_l^{enc})$ (residual connection, $ResConv_l$ for channel matching)
            If attention is used: $h_{out} = Attention_l(h_{out})$
    *   **Bottleneck:**
        *   One or more blocks similar to encoder blocks, but without downsampling. Connects encoder to decoder.
        *   $h_{bottleneck} = Block_{bottleneck}(h_{L_{enc}}^{enc}, t_{emb})$
    *   **Decoder:**
        *   Symmetric to the encoder, consists of multiple upsampling blocks. Each block typically includes:
            *   Convolutional layers, normalization, activation, residual connections, timestep embedding integration.
            *   Upsampling operations (e.g., transposed convolution, nearest-neighbor upsampling followed by convolution).
            *   **Skip Connections:** Feature maps from corresponding encoder blocks are concatenated with the upsampled feature maps in the decoder. This is crucial for preserving low-level details.
        *   Let $h_0^{dec} = h_{bottleneck}$. For block $l=0, \dots, L_{dec}-1$:
            $h_{l+1}^{dec} = Block_{dec,l}(Upsample_l(h_l^{dec}) \oplus h_{L_{enc}-1-l}^{enc}, t_{emb})$
            (where $\oplus$ denotes channel-wise concatenation).
    *   **Output Layer:**
        *   A final convolutional layer maps the output features from the last decoder block to the desired noise prediction shape (same as $x_t$).
        *   $\epsilon_\theta(x_t, t) = Conv_{out}(h_{L_{dec}}^{dec})$
        *   The activation function for this layer is typically linear.

#### 4.3 Forward Diffusion Process (Detailed)
This is a fixed process defined by the variance schedule $\beta_t$.
1.  Start with clean data $x_0$.
2.  For $t=1, \dots, T$:
    Sample $\epsilon_t \sim \mathcal{N}(0, I)$.
    $x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$.
    As $t \to T$, if $\sum \beta_t$ is large enough, $x_T$ approaches an isotropic Gaussian distribution $\mathcal{N}(0,I)$. The closed-form $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is derived by recursively applying the single-step update and utilizing properties of Gaussian distributions.

#### 4.4 Reverse Diffusion Process (Detailed)
The core task is to learn $p_\theta(x_{t-1} | x_t)$ to approximate $q(x_{t-1} | x_t, x_0)$.
*   The mean of $p_\theta(x_{t-1} | x_t)$ is $\mu_\theta(x_t, t)$. Using the parameterization $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ in $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$:
    First, predict $x_0$ from $x_t$:
    $$\hat{x}_0(x_t, t) = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t))$$
    Then, substitute this $\hat{x}_0$ into the equation for $\tilde{\mu}_t(x_t, x_0)$:
    $$\mu_\theta(x_t, t) = \tilde{\mu}_t(x_t, \hat{x}_0(x_t,t)) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_0(x_t,t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$
    This simplifies to the commonly used form:
    $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
*   The variance $\Sigma_\theta(x_t, t) = \sigma_t^2 I$ is typically fixed, e.g., $\sigma_t^2 = \beta_t$ or $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$. The choice $\sigma_t^2 = \tilde{\beta}_t$ corresponds to an optimal setting when $p_\theta(x_0|x_1)$ is perfectly learned, while $\sigma_t^2 = \beta_t$ is simpler and works well.

#### 4.5 Post-Training Procedures: Sampling
Given a trained model $\epsilon_\theta(x_t, t)$:
1.  Start with $x_T \sim \mathcal{N}(0, I)$.
2.  For $t = T, T-1, \dots, 1$:
    a.  Sample $z \sim \mathcal{N}(0, I)$ if $t > 1$, else $z=0$.
    b.  Compute $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$.
    c.  Set $\sigma_t^2$ (e.g., $\beta_t$ or $\tilde{\beta}_t$).
    d.  $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$.
3.  The final output $x_0$ is the generated sample.

*   **Denoising Diffusion Implicit Models (DDIM) Sampling:** (Song et al., 2020)
    Allows faster sampling using fewer steps ($S < T$) and a parameter $\eta$ controlling stochasticity.
    The update rule for a step from $x_{\tau_i}$ to $x_{\tau_{i-1}}$ (where $\{\tau_i\}$ is a sub-sequence of $\{1, ..., T\}$):
    $$\hat{x}_0(x_{\tau_i}, \tau_i) = \frac{x_{\tau_i} - \sqrt{1-\bar{\alpha}_{\tau_i}}\epsilon_\theta(x_{\tau_i}, \tau_i)}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
    $$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \hat{x}_0(x_{\tau_i}, \tau_i) + \sqrt{1-\bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i} z$$
    where $\sigma_{\tau_i}^2 = \eta \frac{(1-\bar{\alpha}_{\tau_{i-1}})}{(1-\bar{\alpha}_{\tau_i})} (1 - \frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}})$.
    If $\eta=0$, sampling is deterministic. If $\eta=1$, it resembles DDPM sampling with a modified variance.
*   **Classifier-Free Guidance:** (Ho & Salimans, 2022)
    For conditional generation $p(x|y)$, the model $\epsilon_\theta(x_t, t, y)$ is trained with conditioning $y$ (e.g., class label, text embedding).
    During training, $y$ is randomly replaced with a null token $\emptyset$ (unconditional) with some probability.
    Modified noise prediction at sampling:
    $$\hat{\epsilon}_\theta(x_t, t, y) = (1+w)\epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t, \emptyset)$$
    or equivalently (and more commonly):
    $$\hat{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$
    where $w$ is the guidance scale. $w=0$ gives unconditional samples, $w=1$ uses standard conditional prediction, $w>1$ amplifies guidance. This $\hat{\epsilon}_\theta$ replaces $\epsilon_\theta$ in the sampling step.

#### 4.6 Training Pseudo-algorithm (DDPM with $L_{simple}$)

1.  **Input:** Training dataset $D = \{x_0^{(i)}\}$, number of diffusion steps $T$, variance schedule $\beta_1, \dots, \beta_T$.
2.  **Initialization:**
    *   Initialize neural network parameters $\theta$ for $\epsilon_\theta(x_t, t)$.
    *   Precompute $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ for all $t \in \{1, \dots, T\}$.
3.  **Training Loop:**
    Repeat for a desired number of iterations or until convergence:
    a.  Sample a mini-batch of clean data $\{x_0^{(j)}\}$ from $D$.
    b.  For each sample $x_0^{(j)}$ in the mini-batch:
        i.  Sample a timestep $t^{(j)} \sim Uniform(\{1, \dots, T\})$.
        ii. Sample noise $\epsilon^{(j)} \sim \mathcal{N}(0, I)$.
        iii. Compute the noisy sample $x_t^{(j)}$ using the direct sampling formula:
            $$x_t^{(j)} = \sqrt{\bar{\alpha}_{t^{(j)}}} x_0^{(j)} + \sqrt{1 - \bar{\alpha}_{t^{(j)}}} \epsilon^{(j)}$$
    c.  Pass the batch of noisy samples $\{x_t^{(j)}\}$ and their corresponding timesteps $\{t^{(j)}\}$ to the network to get predicted noise $\{\hat{\epsilon}^{(j)} = \epsilon_\theta(x_t^{(j)}, t^{(j)})\}$.
    d.  Calculate the loss for the mini-batch:
        $$L = \frac{1}{\text{batch_size}} \sum_j || \epsilon^{(j)} - \hat{\epsilon}^{(j)} ||^2$$
    e.  Compute gradients $\nabla_\theta L$.
    f.  Update parameters $\theta$ using an optimizer (e.g., Adam, AdamW):
        $\theta \leftarrow \theta - \eta_{lr} \nabla_\theta L$ (where $\eta_{lr}$ is the learning rate).

    **Mathematical Justification for $L_{simple}$:**
    The term $D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t))$ in $L_{VLB}$ can be written as:
    $$L_{t-1} = \mathbb{E}_{q(x_0, \epsilon)} \left[ \frac{1}{2\sigma_t^2} || \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) ||^2 \right] + C$$
    Using $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)/\sqrt{\bar{\alpha}_t}$ and substituting into $\tilde{\mu}_t$, and using the parameterization for $\mu_\theta$:
    $$\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$
    $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
    So, $L_{t-1} = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} (\epsilon - \epsilon_\theta(x_t,t)) \right\|^2 \right] + C$.
    This is proportional to $\mathbb{E} [ ||\epsilon - \epsilon_\theta(x_t,t)||^2 ]$. $L_{simple}$ effectively drops the weighting term $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}$ and sums/averages these losses over $t$. This simplification has been shown to lead to better sample quality.

### 5. Importance
*   **State-of-the-Art Generative Performance:** Diffusion models currently achieve leading results in generating high-fidelity and diverse samples for various data modalities, especially images, audio, and video.
*   **Stable Training Dynamics:** Compared to Generative Adversarial Networks (GANs), their training is generally more stable and less prone to issues like mode collapse.
*   **Theoretical Foundation:** They are built upon a well-defined probabilistic framework, allowing for tractable likelihood computation (though often optimized with a simplified objective).
*   **Controllability and Flexibility:** The iterative nature and conditioning mechanisms (like classifier-free guidance) offer significant control over the generation process.
*   **Wide Range of Applications:** Successfully applied to text-to-image synthesis, image inpainting, super-resolution, image editing, audio synthesis, video generation, molecule design, and more.

### 6. Pros versus Cons

*   **Pros:**
    *   **Exceptional Sample Quality:** Often produce samples of higher fidelity and diversity than other generative model classes.
    *   **Training Stability:** Optimization is robust; less hyperparameter sensitivity than GANs.
    *   **Strong Mode Coverage:** Less susceptible to mode collapse.
    *   **Principled Probabilistic Framework:** Allows for (approximate) likelihood evaluation.
    *   **Amenable to Conditioning:** Effectively incorporates various forms of conditioning information.
    *   **Progressive Generation:** The step-by-step denoising is interpretable and allows for interventions.

*   **Cons:**
    *   **Slow Sampling Speed:** The iterative denoising process requires many sequential evaluations of the neural network (typically $T=1000$ to $4000$ steps), making sampling much slower than single-pass models like GANs or VAEs.
    *   **Computationally Intensive:** Training and inference can be demanding, especially for high-resolution data and large $T$.
    *   **Guidance Artifacts:** Strong classifier-free guidance can sometimes lead to oversaturation or unnatural artifacts, trading off diversity for adherence to the prompt.
    *   **Mathematical Complexity:** While the core idea is elegant, the full derivation of the VLB and advanced sampling techniques can be mathematically involved.

### 7. Cutting-Edge Advances

*   **Accelerated Sampling:**
    *   **Denoising Diffusion Implicit Models (DDIM):** (Song et al., 2020) Formulates a non-Markovian forward process, leading to deterministic sampling paths and significantly fewer steps (e.g., 20-100).
    *   **Probability Flow ODE Solvers:** (Song et al., 2020; Lu et al., 2022 - DPM-Solver/DPM-Solver++) Re-frames diffusion as solving an ODE, allowing use of advanced numerical solvers for faster and more accurate sampling.
    *   **Progressive Distillation / Consistency Models:** (Salimans & Ho, 2022; Song et al., 2023) Train a student model to perform multiple denoising steps of a teacher diffusion model in a single step, or learn a "consistency function" $f(x_t, t) \approx x_0$, enabling 1-to-few step generation.
*   **Latent Diffusion Models (LDMs) / Stable Diffusion:** (Rombach et al., 2022)
    *   Apply the diffusion process in a lower-dimensional latent space learned by an autoencoder (e.g., VQ-VAE or KL-regularized autoencoder).
    *   Data $x_0$ is first encoded to $z_0 = \mathcal{E}(x_0)$. Diffusion happens on $z_t$. Generated $z_0$ is decoded by $\mathcal{D}(z_0)$.
    *   This drastically reduces computational cost, allowing for efficient training and generation of high-resolution images.
    *   Loss: $\mathbb{E}_{\mathcal{E}(x_0), \epsilon, t, c} [ || \epsilon - \epsilon_\theta(z_t, t, c) ||^2 ]$, where $c$ is conditioning (e.g., text embeddings).
*   **Improved Control and Conditioning:**
    *   **ControlNet:** (Zhang & Agrawala, 2023) Adds fine-grained spatial control to pre-trained text-to-image diffusion models using conditions like edge maps, depth maps, segmentation, human pose. It involves training a "copy" of the UNet's encoder blocks that integrates spatial conditions, whose outputs are added to the original UNet's skip connections.
        $$h_{out,l}^{ControlNet} = Block_{ControlNet,l}(h_{in,l}^{UNet}, c_{spatial}, t_{emb})$$
        $$h_{skip,l}^{modified} = h_{skip,l}^{UNet} + \gamma_l \cdot h_{out,l'}^{ControlNet}$$
*   **Flow Matching / Rectified Flow:** (Lipman et al., 2023; Liu et al., 2022)
    *   New frameworks for learning continuous normalizing flows (CNFs) by directly regressing vector fields of ODEs that transport noise to data.
    *   Rectified Flow specifically learns to map noise to data along straight paths in expectation.
    *   Loss (Rectified Flow): $\mathbb{E}_{t \sim U[0,1], x_0 \sim p_{data}, x_1 \sim p_{noise}} [ || (x_0 - x_1) - v_\theta((1-t)x_1 + t x_0, t) ||^2 ]$. (Note: some formulations map $x_0$ to $x_1$, others $x_1$ to $x_0$).
    *   These can achieve high-quality generation in very few steps.
*   **Video, Audio, and 3D Diffusion:**
    *   Architectures are being extended for temporal data (Video Diffusion Models, e.g., by Ho et al., 2022; Blattmann et al., 2023) and 3D data (e.g., Point-E, GET3D, Shap-E).
    *   For video, this often involves adding temporal convolutions and temporal attention layers to the UNet.

### 8. Evaluation Phase

#### 8.1 Loss Functions (During Training)
*   **Primary Loss (Simplified $L_2$ Loss):**
    $$L_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} [ || \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) ||^2 ]$$
    This is the most common objective, focusing on the accuracy of noise prediction.
*   **Alternative (Simplified $L_1$ Loss):**
    $$L_{simple, L1}(\theta) = \mathbb{E}_{t, x_0, \epsilon} [ || \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) ||_1 ]$$
    Sometimes used for potentially sharper results or robustness to outliers.
*   **Full Variational Lower Bound (VLB):**
    $$L_{VLB} = \mathbb{E}_q \left[ L_T + \sum_{t=2}^T L_{t-1} + L_0 \right]$$
    where $L_T = D_{KL}(q(x_T|x_0) || p(x_T))$, $L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t))$, and $L_0 = -\log p_\theta(x_0|x_1)$.
    Optimizing the full VLB can provide NLL estimates but $L_{simple}$ often yields better perceptual quality.

#### 8.2 Metrics (State-of-the-Art - SOTA)
These are primarily for image generation quality assessment.
*   **Fréchet Inception Distance (FID):**
    Measures the Wasserstein-2 distance between distributions of InceptionV3 feature activations for real ($x$) and generated ($g$) images. Lower is better.
    $$FID(x, g) = || \mu_x - \mu_g ||^2 + Tr(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})$$
    where $(\mu_x, \Sigma_x)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of activations.
    *   **Best Practice:** Use at least 10,000-50,000 samples. Standardized implementations are crucial.
*   **Inception Score (IS):**
    Measures sample quality (low entropy $p(y|x)$) and diversity (high entropy $p(y)$). Higher is better.
    $$IS(G) = \exp(\mathbb{E}_{x \sim G} [D_{KL}(p(y|x) || p(y))])$$
    where $p(y|x)$ is class probability from InceptionV3 for image $x$, $p(y)$ is marginal class probability.
    *   **Pitfall:** Can be gamed; less reliable than FID for modern high-quality models.
*   **Precision and Recall (for distributions):** (Kynkäänniemi et al., 2019)
    Measures fidelity (Precision: are generated samples realistic?) and diversity (Recall: does the model cover all modes of real data?). Calculated using k-NN in a feature space (e.g., VGG-16).
    Let $M_{real}, M_{gen}$ be sets of feature vectors.
    $$Precision = \frac{1}{|M_{gen}|} \sum_{y \in M_{gen}} \mathbb{I}(\exists x \in M_{real} \text{ s.t. } y \text{ is a k-NN of } x \text{ in } M_{real})$$
    $$Recall = \frac{1}{|M_{real}|} \sum_{x \in M_{real}} \mathbb{I}(\exists y \in M_{gen} \text{ s.t. } x \text{ is a k-NN of } y \text{ in } M_{gen})$$
    (Slight variations in definition exist based on which set k-NN is computed over).
*   **CLIP Score:** (Radford et al., 2021)
    For text-to-image models, measures cosine similarity between CLIP embeddings of generated images and corresponding text prompts. Higher is better.
    $$CLIPScore(I, T) = \text{cosine_similarity}(CLIP_{img\_emb}(I), CLIP_{txt\_emb}(T))$$
    Averaged over a set of (Image, Text) pairs.
*   **Negative Log-Likelihood (NLL):**
    Reported in bits per dimension (BPD). Only meaningful if the model is trained to optimize $L_{VLB}$. Lower is better.
    $$NLL = -\frac{1}{N \cdot D_{dim}} \sum_{i=1}^N \log p_\theta(x_0^{(i)})$$

#### 8.3 Domain-Specific Metrics
*   **Audio Generation:**
    *   **Fréchet Audio Distance (FAD):** FID equivalent using audio embeddings (e.g., VGGish, PANNs).
    *   **Objective Audio Quality Metrics:** PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility), SI-SNR (Scale-Invariant Signal-to-Noise Ratio).
*   **Video Generation:**
    *   **Fréchet Video Distance (FVD):** FID equivalent using features from a pre-trained video classifier (e.g., I3D).
    *   **Kernel Video Distance (KVD):**
*   **Molecular Generation:**
    *   **Validity:** % of generated molecules that are chemically valid.
    *   **Uniqueness:** % of unique valid molecules among generated ones.
    *   **Novelty:** % of unique valid molecules not found in the training set.
    *   **QED (Quantitative Estimate of Drug-likeness), SA Score (Synthetic Accessibility).**

#### 8.4 Best Practices and Potential Pitfalls in Evaluation
*   **Standardized Metric Implementation:** Use official or widely adopted code for metrics to ensure comparability.
*   **Sufficient Sample Size:** Metrics like FID require a large number of samples (e.g., 50k) for stable estimates.
*   **Avoiding Overfitting to Metrics:** Do not solely optimize for metrics; human perceptual evaluation is often crucial.
*   **Computational Cost:** Some metrics are resource-intensive to compute.
*   **Dataset Bias:** Metrics based on pre-trained models (FID, IS, CLIP Score) inherit biases from the datasets these models were trained on.
*   **Reproducibility:** Clearly state evaluation parameters, model checkpoints, and metric computation details.
*   **Human Evaluation:** For subjective quality (e.g., art, creative text-to-image), human studies (e.g., pairwise comparisons, MOS - Mean Opinion Score) are invaluable but costly and time-consuming.
    *   **MOS Equation (conceptual):**
        $$MOS = \frac{\sum_{n=1}^N S_n}{N}$$
        where $S_n$ is the rating given by evaluator $n$, and $N$ is the total number of evaluators. Ratings are typically on a 1-5 scale.

**Potential Pitfalls:**
*   **Metric Hacking:** Models can be fine-tuned to perform well on specific metrics without genuine improvement in perceptual quality.
*   **Misinterpretation:** A single metric rarely tells the whole story. A suite of metrics is usually more informative.
*   **Slow Evaluation Cycles:** If metrics are too slow to compute, it can hinder rapid model development and iteration.