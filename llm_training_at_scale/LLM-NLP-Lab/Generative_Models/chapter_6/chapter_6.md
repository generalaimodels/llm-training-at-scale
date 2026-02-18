# Chapter 6 Flow-Based Generative Models  

---

## 6.1 General Framework

### Definition  
Flow-based generative models transform a simple base distribution $p_{\mathbf{z}}(\mathbf{z})$ (typically standard Gaussian) into a complex data distribution $p_{\mathbf{x}}(\mathbf{x})$ via a sequence of $K$ **invertible, differentiable mappings** (“normalizing flows”)  
$$
\mathbf{x}=f_K\circ f_{K-1}\circ\cdots\circ f_1(\mathbf{z}),\quad
\mathbf{z}\sim p_{\mathbf{z}}.
$$  
The log-likelihood follows from the change-of-variables formula  
$$
\log p_{\mathbf{x}}(\mathbf{x})=\log p_{\mathbf{z}}\!\bigl(f^{-1}(\mathbf{x})\bigr)+
\sum_{k=1}^K\log\Bigl|\det\bigl(\tfrac{\partial f_k}{\partial\mathbf{h}_{k-1}}\bigr)\Bigr|, \qquad \mathbf{h}_0=\mathbf{z},\;\mathbf{h}_K=\mathbf{x}.
$$  
Training maximizes the exact likelihood; sampling draws $\mathbf{z}\sim p_{\mathbf{z}}$ and applies the forward flow.

### Design Requirements  
• **Invertibility** ⇒ exact inverse and tractable determinant.  
• **Expressivity** ⇒ highly non-linear transformations.  
• **Parallelization** ⇒ enable batched GPU/TPU execution.  
• **Scalable Jacobian determinants** ⇒ $\mathcal{O}(D)$, not $\mathcal{O}(D^3)$.

---

## 6.2 Non-linear Independent Components Estimation (NICE)

### Concise Definition  
NICE (Dinh et al., 2014) introduces **additive coupling layers** yielding unit-Jacobian determinants for cost-free likelihood computation.

### Mathematical Core  

1. Partition variables: $\mathbf{h}=(\mathbf{h}_{1:d},\mathbf{h}_{d+1:D})$.  
2. Additive coupling  
$$
\mathbf{y}_{1:d}= \mathbf{h}_{1:d},\qquad
\mathbf{y}_{d+1:D}= \mathbf{h}_{d+1:D}+m(\mathbf{h}_{1:d};\theta),
$$
where $m$ is an arbitrary NN.  
3. Jacobian  
$$
\frac{\partial\mathbf{y}}{\partial\mathbf{h}}=
\begin{bmatrix}
\mathbf{I}_d & \mathbf{0}\\
\frac{\partial m}{\partial\mathbf{h}_{1:d}} & \mathbf{I}_{D-d}
\end{bmatrix},\quad
\det=1.
$$  

### Step-by-Step Mechanics  

1. **Architecture**  
   • Stack $K$ additive coupling layers with alternating partitions (e.g., checkerboard & channel-wise).  
   • Conclude with a scaling diagonal layer $s\in\mathbb{R}^D$: $\mathbf{x}=\exp(s)\odot\mathbf{h}_K$ to match data variance.

2. **Training**  
   • Objective: maximize $$\sum_{n=1}^N \log p_{\mathbf{z}}\bigl(f^{-1}(\mathbf{x}^{(n)})\bigr)+\|s\|_1.$$

3. **Inference / Sampling**  
   • Sampling: draw $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, apply forward flow.  
   • Inference: exact inversion by reversing layer order, exploiting additive coupling invertibility.

4. **Limitations**  
   • Unit determinant → volume-preserving restricts expressivity; remedy: Real NVP introduces scaling inside coupling.  

---

## 6.3 Real NVP (Real-valued Non-Volume Preserving)

### Concise Definition  
Real NVP (Dinh et al., 2017) extends NICE by **affine coupling layers** and a **multi-scale architecture** to capture complex density variations with non-unit Jacobian determinants.

### Affine Coupling Layer  

Partition $\mathbf{h}=(\mathbf{h}_a,\mathbf{h}_b)$:  

$$
\begin{aligned}
\mathbf{y}_a &= \mathbf{h}_a,\\
\mathbf{y}_b &= \mathbf{h}_b \odot \exp\bigl(s(\mathbf{h}_a)\bigr)+t(\mathbf{h}_a),
\end{aligned}
$$  
where $s,t:\mathbb{R}^{|a|}\!\to\mathbb{R}^{|b|}$ are CNNs/FCNs.  

Jacobian determinant  
$$
\log\bigl|\det\tfrac{\partial\mathbf{y}}{\partial\mathbf{h}}\bigr|
= \sum_i s_i(\mathbf{h}_a).
$$  

### Multi-Scale Architecture  

After several coupling layers, half of the variables are **factored out** to the prior:  
$$
\mathbf{h}^{(l)}=(\mathbf{h}_{\text{keep}}^{(l)},\mathbf{z}^{(l)}), \quad
\mathbf{z}^{(l)}\sim\mathcal{N}(0,I),
$$  
reducing memory and increasing depth.

### End-to-End Workflow  

1. **Data Preprocessing**:  
   • Dequantize integers via $u = (x + \epsilon)/256$, $\epsilon\sim\mathcal{U}[0,1)$.  
   • Logit-transform before flow.

2. **Layer Composition**:  
   • Sequence: [Coupling]→[Permutation (channel shuffle)]→[Coupling]→…  
   • Permutation ensures each dimension eventually conditions on every other.

3. **Training Objective**:  
   $$\mathcal{L} = -\frac{1}{N}\sum_{n}\bigl[\log p_{\mathbf{z}}(\mathbf{z}^{(n)}) + \sum_{k}\log|\det J_k^{(n)}|\bigr].$$  

4. **Parallel Computation**: Affine coupling computes $s,t$ only once per layer; determinant costs $\mathcal{O}(D)$.

5. **Advantages**  
   • Exact log-likelihood & sampling.  
   • Non-volume preserving enhances flexibility.  

---

## 6.4 Glow (Generative Flow with Invertible 1×1 Convolutions)

### Concise Definition  
Glow (Kingma & Dhariwal, 2018) refines Real NVP via **actnorm**, **invertible $1\times1$ convolutions** (learnable channel permutations), and **efficient architecture** enabling high-fidelity $256\!\times\!256$ image synthesis.

### Building Blocks  

1. **ActNorm**  
   Per-channel affine transform with trainable $(s,b)$ initialized via data statistics:  
   $$\mathbf{y}_{c,h,w}=s_c\,\mathbf{h}_{c,h,w}+b_c,\quad
   \log|\det| = H W \sum_c \log|s_c|.$$

2. **Invertible $1\times1$ Convolution**  
   $$\mathbf{y}_{c} = W\,\mathbf{h}_{c},\qquad
   \log|\det| = H W\log|\det W|,$$  
   with $W\in\mathrm{GL}(C,\mathbb{R})$. Parameterizations:  
   • Direct weight matrix with LU-decomposition for $\mathcal{O}(C)$ determinant.  
   • Or orthogonal initialization.

3. **Affine Coupling** (same as Real NVP).

### Flow Step  
[ActNorm] → [Invertible $1×1$ Conv] → [Affine Coupling]. Repeat $L$ times per level, then multi-scale factor-out.

### Training/Tuning Details  

• **Path length**: up to $K\sim 32$ steps × 3 levels.  
• **Bits-per-dimension** metric on images.  
• **Parallel Decoding**: one forward pass generates megapixel images in milliseconds.

### Advantages vs. Real NVP  

• Learnable channel mixing vs. fixed permutations.  
• ActNorm avoids batch-norm dependency, stable on small batches.  
• Empirically sharper likelihoods and samples.

---

## 6.5 Masked Autoregressive Flow (MAF)

### Concise Definition  
MAF (Papamakarios et al., 2017) leverages **autoregressive transformations** inside flows, reversing the usual sampling/likelihood trade-off: fast density estimation, slower sampling.

### Transformation  

Autoregressive mapping:  
$$
\mathbf{y}_i = \sigma_i(\mathbf{h}_{<i})\;\mathbf{h}_i + \mu_i(\mathbf{h}_{<i}),\quad i=1,\dots,D,
$$  
with $\mu,\sigma$ parameterized by a **masked autoregressive neural network (MADE)**.  

Jacobian determinant: triangular ⇒  
$$
\log|\det| = \sum_{i=1}^D \log\sigma_i(\mathbf{h}_{<i}).
$$  

### Key Properties  

• **Likelihood Evaluation**: parallelizable; feed full $\mathbf{h}$, get all $\mu,\sigma$ in one pass.  
• **Sampling**: sequential due to autoregressive dependency (order $D$).  
• **Expressivity**: stacking $K$ MAF layers increases flexibility; interleave random permutations.

### Training Regime  

Same MLE objective; base distribution often Gaussian.

### Comparison  

• MAF ≈ inverse of **Inverse Autoregressive Flow (IAF)** used in variational inference; swapping forward/inverse directions trades speed characteristics.  

---

## 6.6 Neural Spline Flow (NSF)

### Concise Definition  
NSF (Durkan et al., 2019) employs **monotonic rational-quadratic splines** within coupling or autoregressive layers, enabling flexible, invertible, non-linear piecewise transforms with analytically tractable inverses and log-determinants.

### Rational-Quadratic Spline  

Given $K_{\text{bin}}$ bins with widths $\Delta x_j$, heights $\Delta y_j$, and derivatives $d_j$:

1. **Forward Mapping**  
   • Identify bin $j$ s.t. $x\in[x_{j},x_{j+1}]$.  
   • Compute  
   $$
   y = y_{j} + \frac{(a\,b + c)}{(d\,b + e)}, \;\text{where}\;
   b = \frac{x - x_j}{\Delta x_j},
   $$  
   coefficients $\{a,c,d,e\}$ are functions of $\Delta x_j,\Delta y_j,d_j,d_{j+1}$ (see paper).

2. **Log-determinant**  
   $$\log\left|\tfrac{dy}{dx}\right| = \log\bigl(\frac{d\,e - a\,c}{(d\,b+e)^2}\bigr).$$

### Coupling Layer with Splines  

Partition $\mathbf{h}=(\mathbf{h}_a,\mathbf{h}_b)$; neural nets output bin parameters conditioned on $\mathbf{h}_a$ to transform $\mathbf{h}_b$ via splines.  

Advantages:  
• Higher-order non-linearities vs. affine.  
• Exact inverse via root-finding closed-form.  
• Empirically improves likelihood on tabular and image data.

### Implementation Steps  

1. **Parameter Output**  
   • Network predicts unconstrained parameters; apply softmax for widths/heights, softplus for derivatives, ensuring monotonicity.  

2. **Boundary Handling**  
   • Outside learned bounds, apply linear tails to maintain invertibility.  

3. **Stacking Strategy**  
   • Replace affine couplings in Real NVP/Glow or autoregressive transforms in MAF with spline layers.  
   • Maintain $\mathcal{O}(D)$ complexity.

---

## 6.7 Practical Considerations

• **Parallelization**: Favor coupling/IAF for fast sampling; favor MAF for fast likelihood in density estimation tasks.  
• **Memory**: Multi-scale factoring reduces activation footprint.  
• **Numerical Stability**: Use log-scaling for $\sigma$, LU-decomposed conv kernels, and constrain spline derivatives.  
• **Hybrid Flows**: Combine coupling, autoregressive, and continuous-time (e.g., FFJORD) flows for task-specific trade-offs.  

---

End of Chapter 6