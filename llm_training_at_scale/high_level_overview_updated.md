

# Scaling Distributed Training: Foundations and First Principles

---

## 1. High-Level Overview: The Three Fundamental Challenges

Every technique in large-scale model training addresses one or more of three orthogonal resource constraints. These constraints form the **scaling trilemma** of distributed deep learning.

---

### 1.1 Memory Usage — The Hard Constraint

Memory is a **binary gate**: if the aggregate memory required by a single training step exceeds the available GPU high-bandwidth memory (HBM), training **cannot proceed at all** — there is no graceful degradation, only an Out-Of-Memory (OOM) crash.

During a single training step, four principal memory occupants coexist on the GPU:

| Memory Occupant | Description |
|---|---|
| **Model Parameters** | The learnable weight tensors $W$ of the network |
| **Gradients** | $\nabla_{W}\mathcal{L}$, partial derivatives of the loss w.r.t. each parameter |
| **Optimizer States** | Auxiliary running statistics (e.g., first and second moment estimates in Adam) |
| **Activations** | Intermediate tensors stored during the forward pass, required for gradient computation in the backward pass |

Additionally, there are minor but non-negligible constant overheads:

- **CUDA context/kernels**: typically $1$–$2$ GB upon first CUDA tensor allocation.
- **Fragmentation & temporary buffers**: memory that exists but cannot be utilized due to allocator fragmentation or short-lived intermediate results.

---

### 1.2 Compute Efficiency — Maximizing Arithmetic Throughput

Modern accelerators (e.g., NVIDIA H100 SXM5) deliver peak throughput of approximately $989$ TFLOPS in BF16 Tensor Core operations. However, **achieved throughput** is invariably lower due to:

- **Memory-bound operations**: kernels whose execution time is dominated by data movement (loads/stores) rather than arithmetic.
- **Kernel launch overhead**: CPU-side latency to dispatch GPU kernels.
- **Pipeline bubbles**: idle time when certain stages of computation must wait for others.

The goal is to maximize the **compute utilization ratio**:

$$
\text{Compute Utilization} = \frac{\text{Achieved FLOPS}}{\text{Peak Hardware FLOPS}}
$$

Every second the GPU spends waiting — for data transfers from CPU to GPU, for memory allocation, or for synchronization barriers — directly reduces this ratio.

---

### 1.3 Communication Overhead — Minimizing GPU Idle Time

In multi-GPU and multi-node settings, GPUs must exchange data (gradients, activations, parameters). Communication occurs over interconnects with heterogeneous bandwidths:

| Interconnect Type | Example Technology | Typical Bandwidth |
|---|---|---|
| **Intra-node** (fast) | NVLink 4.0 (H100) | $900$ GB/s bidirectional |
| **Inter-node** (slower) | InfiniBand NDR400 | $400$ Gb/s ($\approx 50$ GB/s) per port |

Communication overhead keeps GPUs **idle** (not performing useful arithmetic). Two primary strategies mitigate this:

1. **Overlap communication with computation**: launch asynchronous collective operations (e.g., `AllReduce`) concurrently with backward-pass gradient computation on independent layers.
2. **Topology-aware placement**: assign communication-heavy operations to the fast intra-node links and minimize the volume of data traversing slow inter-node links.

---

### 1.4 The Scaling Trilemma: Trading Off Resources

These three resources — **memory**, **compute**, and **communication** — are not independent. Optimizing one often comes at the cost of another. Two canonical examples:

| Technique | Saves | Costs |
|---|---|---|
| **Activation Recomputation** | Memory (activations discarded) | Compute (activations recomputed during backward) |
| **Tensor Parallelism** | Memory (model sharded across GPUs) | Communication (intermediate activations exchanged) |

Formally, if we denote total training step wall-clock time as $T$, we can decompose it as:

$$
T = T_{\text{compute}} + T_{\text{communication}} + T_{\text{idle}}
$$

The objective of efficient scaling is:

$$
\min_{(\text{parallelism strategy, memory strategy})} \; T \quad \text{subject to} \quad M_{\text{peak}} \leq M_{\text{GPU}}
$$

where $M_{\text{peak}}$ is the peak memory during a training step and $M_{\text{GPU}}$ is the physical HBM capacity.

---

## 2. First Steps: Training on One GPU

### 2.1 The Three Phases of a Training Step

A single training step on one GPU consists of three sequential phases:

---

#### Phase 1: Forward Pass

The input batch $X \in \mathbb{R}^{bs \times seq \times d_{\text{input}}}$ is propagated through $L$ successive layers of the model. For a generic layer $\ell$ with parameters $W_\ell$, the forward computation is:

$$
a^{(\ell)} = f_\ell\!\left(a^{(\ell-1)};\; W_\ell\right), \quad \ell = 1, 2, \ldots, L
$$

where $a^{(0)} = X$ is the input embedding and $a^{(\ell)}$ is the **activation** (hidden state) output of layer $\ell$. The final output $\hat{y} = a^{(L)}$ is used to compute the scalar loss:

$$
\mathcal{L} = \mathcal{L}\!\left(\hat{y},\; y_{\text{target}}\right)
$$

**Critical point**: Every intermediate activation $a^{(\ell)}$ must be **stored in GPU memory** because it is needed for the backward pass (see Phase 2).

---

#### Phase 2: Backward Pass (Backpropagation)

Using the chain rule, gradients of the loss with respect to each layer's parameters are computed **in reverse order** ($\ell = L, L-1, \ldots, 1$):

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} = \frac{\partial \mathcal{L}}{\partial a^{(\ell)}} \cdot \frac{\partial a^{(\ell)}}{\partial W_\ell}
$$

The upstream gradient $\frac{\partial \mathcal{L}}{\partial a^{(\ell)}}$ is propagated backward through:

$$
\frac{\partial \mathcal{L}}{\partial a^{(\ell-1)}} = \frac{\partial \mathcal{L}}{\partial a^{(\ell)}} \cdot \frac{\partial a^{(\ell)}}{\partial a^{(\ell-1)}}
$$

As each layer $\ell$ completes its gradient computation, the stored activation $a^{(\ell)}$ is **freed** from memory, while the gradient tensor $\nabla_{W_\ell}\mathcal{L}$ is **allocated**.

---

#### Phase 3: Optimizer Step

The optimizer uses the accumulated gradients to update all parameters. For the Adam optimizer, the update rule for each parameter tensor $\theta$ at step $t$ is:

$$
m_t = \beta_1 \, m_{t-1} + (1 - \beta_1)\, g_t
$$

$$
v_t = \beta_2 \, v_{t-1} + (1 - \beta_2)\, g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $g_t = \nabla_\theta \mathcal{L}$, $m_t$ is the first moment (momentum), $v_t$ is the second moment (variance), $\eta$ is the learning rate, and $\epsilon$ is a numerical stability constant (typically $10^{-8}$).

After the optimizer step, the gradient buffers are zeroed and the cycle repeats.

---

### 2.2 Batch Size: Definitions, Impact, and Practical Ranges

#### Definition

The **batch size** ($bs$) is the number of independent input samples processed in a single forward–backward pass before the optimizer updates the parameters.

In the LLM pretraining community, batch sizes are reported in **tokens** ($bst$) to decouple from the choice of sequence length ($seq$):

$$
bst = bs \times seq
$$

---

#### Impact on Convergence

| Regime | Gradient Estimate Quality | Convergence Behavior |
|---|---|---|
| **Small** $bs$ | High variance (noisy) | Fast early exploration; difficulty converging to sharp optima |
| **Large** $bs$ | Low variance (accurate) | Diminishing returns per token; slower convergence per token seen |

OpenAI's seminal study on large batch training [1] demonstrated the existence of a **critical batch size** $B_{\text{crit}}$ for each model and dataset, below which doubling the batch size nearly halves the number of required optimization steps, and above which diminishing returns set in rapidly.

**Key practical insight**: The sensitivity of final model performance to the exact batch size is **low** around the optimum — i.e., batch size can be varied within a broad range near $B_{\text{crit}}$ without significant degradation.

---

#### Real-World Examples

| Model | Batch Size (tokens) | Total Training Tokens | Notes |
|---|---|---|---|
| Llama 1 | $\sim 4$M | $1.4$T | Fixed batch size throughout |
| DeepSeek-V3/R1 | $3{,}072 \to 15{,}360$ sequences ($\sim$60M tokens) | $14.8$T | Batch size ramped during first $469$B tokens |

The sweet spot for contemporary LLM pretraining is typically:

$$
bst \in [4 \times 10^6,\; 60 \times 10^6] \text{ tokens per global batch}
$$

---

## 3. Memory Usage in Transformers: Detailed Breakdown

### 3.1 The Four Principal Memory Occupants

| # | Occupant | Depends on Batch Size? | Depends on Sequence Length? |
|---|---|---|---|
| 1 | Model weights $W$ | No | No |
| 2 | Gradients $\nabla_W \mathcal{L}$ | No | No |
| 3 | Optimizer states ($m$, $v$ for Adam) | No | No |
| 4 | Activations $a^{(\ell)}$ | **Yes** (linear) | **Yes** (quadratic) |

Items 1–3 are **static** with respect to batch size and sequence length — they depend only on the model architecture (parameter count $N$). Item 4 is **dynamic** and is the dominant memory consumer for large batch sizes and long sequences.

---

### 3.2 Numerical Precision Formats

| Format | Bytes per Value | Exponent Bits | Mantissa Bits | Dynamic Range | Precision |
|---|---|---|---|---|---|
| FP32 | $4$ | $8$ | $23$ | $\sim 10^{\pm 38}$ | High |
| BF16 | $2$ | $8$ | $7$ | $\sim 10^{\pm 38}$ | Reduced mantissa |
| FP16 | $2$ | $5$ | $10$ | $\sim 6.5 \times 10^4$ | Narrower range |
| FP8 (E4M3) | $1$ | $4$ | $3$ | $\sim 448$ | Very low |

The memory footprint of any tensor is:

$$
\text{Memory (bytes)} = \text{number of elements} \times \text{bytes per element}
$$

---

## 4. Memory for Weights, Gradients, and Optimizer States

### 4.1 Parameter Count of a Transformer LLM

For a decoder-only transformer with:

- $h$: hidden dimension
- $v$: vocabulary size
- $L$: number of layers
- No fixed positional embeddings (e.g., using RoPE)

The total parameter count is:

$$
N = h \cdot v + L \cdot (12h^2 + 13h) + 2h
$$

**Breakdown of the per-layer term** $12h^2 + 13h$:

| Component | Parameters | Count |
|---|---|---|
| Self-attention: $W_Q, W_K, W_V$ | $3 \times h \times h$ | $3h^2$ |
| Self-attention: $W_O$ (output projection) | $h \times h$ | $h^2$ |
| Feed-forward: $W_1$ (up-projection) | $h \times 4h$ | $4h^2$ |
| Feed-forward: $W_2$ (down-projection) | $4h \times h$ | $4h^2$ |
| LayerNorm (×2 per layer, each with $\gamma, \beta$) | $2 \times 2h$ | $4h$ |
| Biases in attention and FFN (varies) | — | $\sim 9h$ |
| **Total per layer** | — | $12h^2 + 13h$ |

The term $h \cdot v$ accounts for the **token embedding matrix**, and the final $2h$ accounts for the **final LayerNorm** (gain and bias).

**Scaling observation**: As $h$ grows, the dominant term is $12Lh^2$, which grows **quadratically** in the hidden dimension. The linear terms ($13Lh$, $hv$, $2h$) become negligible for very large models.

---

### 4.2 Full Precision (FP32) Training Memory

In pure FP32 training, every value — parameters, gradients, optimizer states — is stored in 32-bit floating point ($4$ bytes):

$$
m_{\text{params}} = 4N \text{ bytes}
$$

$$
m_{\text{grad}} = 4N \text{ bytes}
$$

$$
m_{\text{opt}} = (4 + 4) \cdot N = 8N \text{ bytes} \quad \text{(Adam: momentum } m_t \text{ + variance } v_t\text{)}
$$

**Total (FP32)**:

$$
m_{\text{total}}^{\text{FP32}} = 4N + 4N + 8N = 16N \text{ bytes}
$$

---

### 4.3 Mixed Precision (BF16 + FP32 Master Weights) Training Memory

In standard mixed precision training:

- **Forward/backward** computations are done in BF16 ($2$ bytes per value).
- A **master copy** of weights is maintained in FP32 for numerical stability during the optimizer update.
- **Optimizer states** (Adam $m_t$, $v_t$) are stored in FP32.

| Component | Precision | Bytes per Parameter |
|---|---|---|
| Parameters (BF16 working copy) | BF16 | $2$ |
| Gradients (BF16) | BF16 | $2$ |
| Master weights (FP32 copy) | FP32 | $4$ |
| Adam momentum $m_t$ | FP32 | $4$ |
| Adam variance $v_t$ | FP32 | $4$ |

$$
m_{\text{params}} = 2N
$$

$$
m_{\text{grad}} = 2N
$$

$$
m_{\text{params\_fp32}} = 4N \quad \text{(master weights)}
$$

$$
m_{\text{opt}} = (4 + 4) \cdot N = 8N
$$

**Total (mixed precision, BF16 gradients)**:

$$
m_{\text{total}}^{\text{mixed}} = 2N + 2N + 4N + 8N = 16N \text{ bytes}
$$

**Total (mixed precision, FP32 gradient accumulation)**:

If gradients are accumulated in FP32 (for stability with small gradient values in BF16), an additional $4N$ bytes is required:

$$
m_{\text{total}}^{\text{mixed, FP32 grad}} = 2N + 2N + 4N + 4N + 8N = 20N \text{ bytes}
$$

> **Key insight**: Mixed precision training does **not** reduce the total memory for weights + gradients + optimizer states compared to full FP32 training (both are $16N$ bytes without FP32 gradient accumulation). The advantages of mixed precision are:
> 1. **Faster computation**: BF16 Tensor Core operations are $2\times$ faster than FP32.
> 2. **Reduced activation memory**: activations stored in BF16 use half the memory of FP32.

---

### 4.4 Practical Memory Requirements Table

| Model Size ($N$) | FP32 or BF16 (no FP32 grad acc) — $16N$ | BF16 with FP32 grad acc — $20N$ |
|---|---|---|
| $1$B | $16$ GB | $20$ GB |
| $7$B | $112$ GB | $140$ GB |
| $70$B | $1{,}120$ GB | $1{,}400$ GB |
| $405$B | $6{,}480$ GB | $8{,}100$ GB |

For reference, an NVIDIA H100 SXM5 has **80 GB** HBM3. At $7$B parameters, the weights + gradients + optimizer states alone ($112$–$140$ GB) already exceed a single GPU's capacity — **before** accounting for activations.

---

## 5. Memory for Activations

### 5.1 Why Activations Must Be Stored

During the backward pass, computing $\frac{\partial \mathcal{L}}{\partial W_\ell}$ requires the **input activation** $a^{(\ell-1)}$ to layer $\ell$:

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} = \left(\frac{\partial \mathcal{L}}{\partial a^{(\ell)}}\right)^\top \cdot \frac{\partial a^{(\ell)}}{\partial W_\ell}
$$

For a linear layer $a^{(\ell)} = W_\ell \, a^{(\ell-1)}$, this becomes:

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} = \left(\frac{\partial \mathcal{L}}{\partial a^{(\ell)}}\right)^\top \cdot a^{(\ell-1)}
$$

Without the stored activation $a^{(\ell-1)}$, this gradient **cannot be computed**. Hence, all intermediate activations must be retained in memory from the forward pass until they are consumed in the backward pass.

---

### 5.2 Activation Memory Formula

For a transformer model in mixed precision (BF16 activations), the total activation memory is:

$$
\boxed{m_{\text{act}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)}
$$

where:

| Symbol | Meaning |
|---|---|
| $L$ | Number of transformer layers |
| $seq$ | Sequence length (number of tokens per sample) |
| $bs$ | Batch size (number of samples) |
| $h$ | Hidden dimension |
| $n_{\text{heads}}$ | Number of attention heads |

---

#### Derivation Sketch (per layer, per sample)

Within each transformer layer, the following intermediate tensors must be stored for backpropagation:

| Activation Tensor | Shape | Bytes (BF16, 2 bytes) |
|---|---|---|
| Input to attention LayerNorm | $seq \times h$ | $2 \cdot seq \cdot h$ |
| $Q, K, V$ projections | $3 \times seq \times h$ | $6 \cdot seq \cdot h$ |
| Attention scores $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$ | $n_{\text{heads}} \times seq \times seq$ | $2 \cdot n_{\text{heads}} \cdot seq^2$ |
| Attention output before $W_O$ | $seq \times h$ | $2 \cdot seq \cdot h$ |
| Dropout masks (attention + FFN) | — | $\sim 2 \cdot seq \cdot h$ (1 byte each × 2) |
| Input to FFN LayerNorm | $seq \times h$ | $2 \cdot seq \cdot h$ |
| FFN intermediate ($4h$ hidden) | $seq \times 4h$ | $8 \cdot seq \cdot h$ |
| FFN output | $seq \times h$ | $2 \cdot seq \cdot h$ |
| Residual connections, GeLU input, etc. | — | remaining terms |

Summing all per-layer contributions for a single sample and then scaling by $L$ layers and $bs$ samples yields the formula above. The full accounting is detailed in the NVIDIA recomputation paper [4].

**Critical scaling behavior**:

- **Linear** in $bs$ (batch size)
- **Linear** in $L$ (depth)
- **Quadratic** in $seq$ (due to the $n_{\text{heads}} \cdot seq^2 / h$ attention score term)
- The attention score memory $\propto seq^2$ dominates for long sequences

This means:

$$
m_{\text{act}} = \mathcal{O}(L \cdot bs \cdot seq^2 \cdot n_{\text{heads}}) \quad \text{for large } seq
$$

For short sequences, the $34 \cdot seq \cdot h$ term dominates and memory is approximately linear in $seq$. For long sequences, the quadratic term takes over and activation memory **explodes**.

---

## 6. Activation Recomputation (Gradient Checkpointing / Rematerialization)

### 6.1 Core Idea

**Activation recomputation** trades **compute** for **memory**: instead of storing all intermediate activations during the forward pass, we **discard** most of them and **recompute** them on-the-fly during the backward pass from a small set of saved **checkpoints**.

Without recomputation, memory for activations is $\mathcal{O}(L)$ (all $L$ layers' activations stored). With full recomputation, if we checkpoint only every $k$-th layer, activation memory becomes $\mathcal{O}(L/k)$, but we pay an additional forward-pass compute cost to recompute the discarded activations.

---

### 6.2 Strategies

#### Strategy 1: Full Recomputation

- **What is stored**: only the activation at the **boundary** of each transformer layer (i.e., $a^{(\ell)}$ for $\ell = 0, 1, \ldots, L$, but none of the intermediate tensors within each layer).
- **Recomputation cost**: during the backward pass, for each layer $\ell$, the **entire forward pass through layer $\ell$** must be re-executed to reconstruct internal activations (attention scores, FFN intermediates, etc.).
- **Net effect**: approximately **one additional full forward pass** is performed during each backward pass.
- **Compute overhead**: typically $+30\%$ to $+40\%$ wall-clock time increase per training step.
- **Memory savings**: maximal — all intra-layer activations are freed.

#### Strategy 2: Selective Recomputation

The key observation from the NVIDIA paper [4] is that not all activations are equally costly to store or cheap to recompute:

| Activation Type | Memory Footprint | Recompute FLOPS Cost |
|---|---|---|
| **Attention scores** ($QK^\top / \sqrt{d_k}$) | Large ($\propto n_{\text{heads}} \cdot seq^2$) | Low (matrix multiply, softmax) |
| **FFN intermediates** | Moderate ($\propto seq \cdot 4h$) | High (large matrix multiplications) |

**Selective strategy**: discard and recompute only the **attention-related** activations (which are large but cheap to recompute), while **retaining** the FFN activations (which are expensive to recompute).

For GPT-3 (175B):
- **Activation memory reduction**: $\sim 70\%$
- **Compute overhead**: only $\sim 2.7\%$

This is a dramatically better trade-off than full recomputation.

**Example — DeepSeek-V3**: uses **Multi-Head Latent Attention (MLA)**, which compresses the $Q$, $K$, $V$ representations into a low-rank latent space. This reduces the attention activation memory even further, making selective checkpointing even more effective:

$$
\tilde{K}, \tilde{V} \in \mathbb{R}^{seq \times d_c}, \quad d_c \ll h
$$

where $d_c$ is the compressed latent dimension, drastically reducing the stored activation size for key-value pairs.

---

### 6.3 FlashAttention and Native Recomputation

**FlashAttention** (Dao et al.) is a hardware-aware exact attention algorithm that:

1. Computes attention in **tiled blocks** that fit in GPU SRAM (on-chip memory), avoiding materialization of the full $seq \times seq$ attention matrix in HBM.
2. **Natively integrates selective recomputation**: during the backward pass, attention scores are **recomputed from $Q$, $K$, $V$** rather than loaded from HBM.

Since FlashAttention is now the default in most training frameworks, practitioners using it are **already benefiting from selective recomputation** without explicit gradient checkpointing configuration for the attention layers.

---

### 6.4 Hardware FLOPS Utilization (HFU) vs. Model FLOPS Utilization (MFU)

Recomputation adds "extra" floating-point operations that are not part of the theoretical minimum computation for a forward + backward pass. This creates an important distinction in efficiency metrics:

**Hardware FLOPS Utilization (HFU)**:

$$
\text{HFU} = \frac{F_{\text{forward}} + F_{\text{backward}} + F_{\text{recompute}}}{\Delta t \cdot \text{Peak FLOPS}}
$$

This measures how effectively the hardware arithmetic units are utilized, **including** recomputation. A high HFU means the GPU is kept busy, but some of that work is "redundant."

**Model FLOPS Utilization (MFU)**:

$$
\text{MFU} = \frac{F_{\text{forward}} + F_{\text{backward}}}{\Delta t \cdot \text{Peak FLOPS}}
$$

This measures how much **useful** (non-redundant) computation the hardware performs per unit time. MFU is the better metric for **comparing different hardware** or **different training configurations**, because it rewards setups that can avoid recomputation (e.g., by having more memory available).

For an ideal training setup with no recomputation: $\text{HFU} = \text{MFU}$.

For a setup with full recomputation: $\text{HFU} > \text{MFU}$ (because the denominator is the same but HFU's numerator includes the extra forward pass).

---

## 7. Gradient Accumulation

### 7.1 Core Mechanism

Gradient accumulation decouples the **global batch size** (which determines the gradient quality and convergence behavior) from the **micro-batch size** (which determines the activation memory per forward–backward pass).

The procedure for a single optimizer step with $G$ gradient accumulation steps:

$$
\text{For } i = 1, 2, \ldots, G:
$$

$$
\quad \text{1. Forward pass on micro-batch } \mathcal{B}_i \text{ of size } mbs
$$

$$
\quad \text{2. Backward pass: compute } \nabla_W \mathcal{L}(\mathcal{B}_i)
$$

$$
\quad \text{3. Accumulate: } \bar{g} \leftarrow \bar{g} + \nabla_W \mathcal{L}(\mathcal{B}_i)
$$

$$
\text{4. Average: } \bar{g} \leftarrow \frac{\bar{g}}{G}
$$

$$
\text{5. Optimizer step: } W \leftarrow \text{Adam}(W, \bar{g})
$$

$$
\text{6. Zero: } \bar{g} \leftarrow 0
$$

---

### 7.2 Batch Size Relationship

$$
\boxed{gbs = mbs \times grad\_acc}
$$

where:

| Symbol | Meaning |
|---|---|
| $gbs$ | **Global batch size** — total number of samples per optimizer step |
| $mbs$ | **Micro-batch size** — number of samples per single forward–backward pass |
| $grad\_acc$ ($G$) | **Gradient accumulation steps** — number of sequential forward–backward passes per optimizer step |

In tokens:

$$
gbs_t = gbs \times seq = mbs \times grad\_acc \times seq
$$

---

### 7.3 Memory Trade-Off

| Without Gradient Accumulation | With Gradient Accumulation |
|---|---|
| $m_{\text{act}} \propto gbs \cdot seq$ (full batch in memory) | $m_{\text{act}} \propto mbs \cdot seq$ (only one micro-batch in memory) |
| Single forward–backward pass | $G$ sequential forward–backward passes |

**Activation memory reduction factor**: $\frac{gbs}{mbs} = G$

**Drawback**: Gradient accumulation requires $G$ sequential forward–backward passes per optimizer step. The wall-clock time per optimizer step increases approximately linearly with $G$:

$$
T_{\text{step}} \approx G \cdot (T_{\text{fwd}} + T_{\text{bwd}}) + T_{\text{opt}}
$$

**Subtle memory note**: Gradient accumulation requires a **persistent gradient buffer** of size $m_{\text{grad}}$ that persists across all $G$ micro-batch passes. Without gradient accumulation, gradients can be computed and immediately consumed during the backward pass, enabling slightly lower peak memory through operator fusion. With accumulation, gradients from previous micro-batches must persist, creating a small memory overhead.

---

### 7.4 The Parallelism Opportunity

The $G$ micro-batch forward–backward passes are **independent** computations (different input samples, no inter-dependencies). This independence is precisely what enables **Data Parallelism**: distribute the $G$ micro-batches across $G$ GPUs, perform forward–backward passes simultaneously, and synchronize gradients via an `AllReduce` collective before the optimizer step. This is covered in the next section of the book.

---

## 8. Profiling GPU Compute and Communication

### 8.1 The PyTorch Profiler

PyTorch provides a built-in profiler that traces CPU and GPU activity at the kernel level. The API:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    with_stack=True
) as prof:
    for step in range(steps):
        train_step()
        prof.step()
```

**Parameters explained**:

| Parameter | Meaning |
|---|---|
| `activities` | Which devices to trace (CPU thread activity, CUDA kernel launches and execution) |
| `schedule(wait=1, warmup=1, active=3)` | Skip 1 step (cold start), warm up for 1 step (JIT compilation, allocator), then actively trace 3 steps |
| `on_trace_ready` | Callback to export the trace (here, to TensorBoard format) |
| `with_stack=True` | Capture Python call stacks for each operation (enables source-code-level attribution) |

---

### 8.2 Anatomy of a Profiler Trace

A profiler trace visualized in TensorBoard or Chrome's `chrome://tracing` reveals multiple concurrent timelines:

| Timeline | What It Shows |
|---|---|
| **CPU threads** | Python execution, kernel launch commands, data loading, synchronization calls |
| **CUDA streams** (compute) | GPU kernel execution: GEMM (matrix multiplies), element-wise ops, softmax, LayerNorm, etc. |
| **CUDA streams** (communication) | NCCL collective operations: `AllReduce`, `AllGather`, `ReduceScatter` |
| **Memory** | Allocation and deallocation events, peak memory watermark |

**Key patterns to identify**:

1. **Sequential compute and communication**: If a gradient `AllReduce` operation appears **after** the backward pass completes (rather than overlapping with it), there is unnecessary GPU idle time. The fix is to launch `AllReduce` asynchronously as soon as each layer's gradients are ready.

2. **Idle GPU time (gaps between kernels)**: May indicate:
   - CPU bottleneck (kernel launch overhead, data preprocessing)
   - Synchronization barriers (`torch.cuda.synchronize()`, `dist.barrier()`)
   - Memory allocation stalls (fragmentation forcing the CUDA caching allocator to search or defragment)

3. **CUDA Syncs and CPU↔GPU data transfers**: `cudaMemcpy` (host-to-device or device-to-host) operations appear as blocking periods. These should be minimized or overlapped with computation using pinned memory and non-blocking transfers.

4. **First step anomaly**: The first training step exhibits a markedly different profile due to:
   - **CUDA context initialization**: loading CUDA kernels, JIT-compiling fused operations.
   - **PyTorch caching allocator warm-up**: the allocator performs trial allocations to build a free-block cache (see Zach DeVito's blog on the PyTorch CUDA caching allocator). Subsequent steps reuse these cached blocks, making allocation nearly free.
   - **Optimizer state initialization**: Adam's $m_t$ and $v_t$ tensors are allocated after the first backward pass and persist for all subsequent steps.

> **Practical consequence**: A model that fits in memory during step 1 may OOM at step 2, because the optimizer states ($8N$ bytes for Adam in FP32) are allocated only after the first backward pass, permanently increasing the memory baseline.

---

## 9. Summary: The Landscape Before Multi-GPU Scaling

| Concept | Key Formula / Insight |
|---|---|
| Training step | Forward → Backward → Optimizer update |
| Batch size (tokens) | $bst = bs \times seq$ |
| Parameter count (Transformer) | $N = hv + L(12h^2 + 13h) + 2h$ |
| Memory: params + grads + opt (FP32) | $16N$ bytes |
| Memory: params + grads + opt (mixed, no FP32 grad) | $16N$ bytes |
| Memory: params + grads + opt (mixed, FP32 grad) | $20N$ bytes |
| Memory: activations (mixed precision) | $m_{\text{act}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)$ |
| Activation recomputation | Trade compute (+30–40% full, +2.7% selective) for memory (up to 70% activation reduction) |
| Gradient accumulation | $gbs = mbs \times grad\_acc$; constant activation memory regardless of $gbs$ |
| Scaling trilemma | Memory ↔ Compute ↔ Communication — optimizing one often costs another |

These single-GPU foundations — memory anatomy, activation recomputation, and gradient accumulation — form the building blocks upon which all multi-GPU parallelism strategies (Data Parallelism, Tensor Parallelism, Pipeline Parallelism, ZeRO, etc.) are constructed.






-------------------
# Corrected Production-Accurate Memory Analysis for Transformer Training

---

## 0. The Fundamental Error in the Naïve "$16N$ Bytes" Estimate

The statement "$16N$ bytes for FP32 training" commits a critical analytical error: it treats all four memory occupants — parameters, gradients, optimizer states, and activations — as if they coexist simultaneously at their full sizes throughout the entire training step. In reality:

1. **Gradients do not exist before the backward pass begins.** They are allocated layer-by-layer during backpropagation.
2. **Activations do not vanish the instant the backward pass starts.** They are freed progressively as each layer's backward computation consumes them.
3. The **true peak memory** occurs at a specific transient moment — the **start of the backward pass** — when gradients begin accumulating while nearly all activations are still resident.

The corrected analysis requires tracing memory occupancy **as a function of the training phase**, not collapsing it to a single static number.

---

## 1. Notation and Assumptions

### 1.1 Symbols

| Symbol | Definition |
|---|---|
| $N$ | Total number of learnable parameters |
| $L$ | Number of transformer layers |
| $h$ | Hidden dimension |
| $v$ | Vocabulary size |
| $n_{\text{heads}}$ | Number of attention heads |
| $seq$ | Sequence length (tokens per sample) |
| $bs$ | Batch size (samples) |
| $A_\ell$ | Activation memory (bytes) stored for layer $\ell$ during forward pass |
| $A_{\text{tot}}$ | Total activation memory across all layers: $A_{\text{tot}} = \sum_{\ell=1}^{L} A_\ell$ |
| $A_{\text{rem}}(k)$ | Remaining (unreleased) activation memory when backward pass has reached layer $k$ |

### 1.2 Precision Assumption (FP32 Baseline)

All tensors — parameters, gradients, optimizer states — stored in FP32 ($4$ bytes per scalar). Adam optimizer stores two auxiliary states per parameter: first moment $m_t$ and second moment $v_t$.

### 1.3 Generalized Parameter Count

The formula $N = hv + L(12h^2 + 13h) + 2h$ is **architecture-specific** (standard GPT-style decoder with $4h$ FFN intermediate dimension, bias terms in all projections, two LayerNorms per layer, and no tied embeddings). A **generalized** parameter count for an arbitrary transformer should account for:

$$
N = \underbrace{N_{\text{embed}}}_{\text{embedding}} + \underbrace{\sum_{\ell=1}^{L} N_\ell}_{\text{per-layer}} + \underbrace{N_{\text{head}}}_{\text{output head + final norm}}
$$

where each layer $\ell$ contributes:

$$
N_\ell = \underbrace{n_{\text{heads}} \cdot d_{\text{head}} \cdot h \cdot 3}_{\text{QKV projections}} + \underbrace{n_{\text{heads}} \cdot d_{\text{head}} \cdot h}_{W_O} + \underbrace{h \cdot d_{\text{ff}} + d_{\text{ff}} \cdot h}_{\text{FFN up + down}} + \underbrace{N_{\text{norm}}^\ell + N_{\text{bias}}^\ell}_{\text{norms, biases}}
$$

with $d_{\text{head}} = h / n_{\text{heads}}$ and $d_{\text{ff}}$ being the FFN intermediate dimension (often $4h$, but $\frac{8h}{3}$ in SwiGLU-based models like Llama). Whether biases exist, whether embeddings are tied to the output head, whether GQA (grouped-query attention) is used — all change $N$. **The specific formula must be derived per architecture; no single formula is universal.**

For the remainder of this analysis, we treat $N$ as a **known constant** for a given model and focus on the **memory dynamics** during training, which are architecture-independent in structure.

---

## 2. Fixed Memory: Parameters and Optimizer States

These components depend only on $N$ and are **independent** of batch size, sequence length, and training phase (once initialized):

| Component | Memory (bytes) | When Allocated | When Freed |
|---|---|---|---|
| Parameters $W$ | $4N$ | Model initialization | Never (persistent) |
| Adam first moment $m_t$ | $4N$ | After first backward pass | Never (persistent) |
| Adam second moment $v_t$ | $4N$ | After first backward pass | Never (persistent) |
| Gradients $\nabla_W \mathcal{L}$ | $4N$ | During backward pass | Zeroed after optimizer step |

$$
\boxed{m_{\text{param+opt}} = 4N + 4N + 4N = 12N \text{ bytes (persistent baseline after step 1)}}
$$

$$
\boxed{m_{\text{grad}} = 4N \text{ bytes (transient, exists only during backward + optimizer)}}
$$

> **Critical observation**: The gradient tensor's $4N$ bytes is **not** always resident. Before the backward pass, no gradient memory is allocated (or it is zero-filled from the previous step's clearing). It builds up during the backward pass. This temporal behavior is what makes the simple "$16N$" summary misleading.

---

## 3. Phase-by-Phase Memory Analysis

### 3.1 Phase 0: Before Forward Pass (Steady-State Baseline)

At this point in the training loop, the previous step's optimizer update has completed, gradients have been zeroed, and activations from the previous step have been fully freed.

| Component | Memory |
|---|---|
| Parameters $W$ | $4N$ |
| Adam $m_t$ | $4N$ |
| Adam $v_t$ | $4N$ |
| Activations | $0$ |
| Gradients | $0$ |

$$
\boxed{M_{\text{phase 0}} = 12N}
$$

> **Note on Step 0 vs. Step 1+**: At the very first step, the optimizer states $m_0$ and $v_0$ do not yet exist (they are initialized to zero tensors only after the first backward pass). Therefore, memory before the first forward pass is only $4N$ (parameters alone). After the first optimizer step, the persistent baseline jumps to $12N$. **This explains why a model that fits in GPU memory at step 0 can OOM at step 1.**

---

### 3.2 Phase 1: During the Forward Pass

As the forward pass proceeds through layers $\ell = 1, 2, \ldots, L$, each layer produces intermediate activations $a^{(\ell)}$ that must be retained for the backward pass. The activation memory **monotonically increases** as more layers are processed.

After completing the forward pass through layer $k$ (i.e., layers $1$ through $k$ are done):

$$
M_{\text{fwd}}(k) = \underbrace{12N}_{\text{params + optimizer}} + \underbrace{\sum_{\ell=1}^{k} A_\ell}_{\text{accumulated activations}} + \underbrace{0}_{\text{no gradients yet}}
$$

---

### 3.3 Phase 2: End of Forward Pass (All Activations Stored)

When the forward pass is complete ($k = L$), **all** layer activations are simultaneously resident:

$$
\boxed{M_{\text{end-fwd}} = 12N + A_{\text{tot}}, \quad \text{where } A_{\text{tot}} = \sum_{\ell=1}^{L} A_\ell}
$$

At this point, the loss $\mathcal{L}$ is computed, and the backward pass is about to begin. No gradients have been allocated yet.

For the activation memory formula in mixed precision (BF16):

$$
A_{\text{tot}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)
$$

For FP32, all activation bytes double (replace the constant $34$ and coefficient $5$ with their FP32 equivalents derived from the same accounting with 4-byte storage per element instead of 2-byte).

---

### 3.4 Phase 3: During the Backward Pass — The True Peak

The backward pass proceeds in **reverse** layer order: $\ell = L, L-1, \ldots, 1$. At each backward step $k$ (processing layer $k$ in reverse):

1. The activation $a^{(k-1)}$ (input to layer $k$) is **read** to compute $\frac{\partial \mathcal{L}}{\partial W_k}$.
2. The gradient $\nabla_{W_k} \mathcal{L}$ is **written** to the gradient buffer ($4N_k$ bytes, where $N_k$ is the parameter count of layer $k$).
3. After layer $k$'s backward computation completes, activation $a^{(k)}$ is **freed**.

Define the **remaining activation memory** when the backward pass has just reached layer $k$ (i.e., layers $L, L-1, \ldots, k+1$ have completed their backward, and layer $k$ is about to begin):

$$
A_{\text{rem}}(k) = \sum_{\ell=k}^{L} A_\ell
$$

And the **accumulated gradient memory** at this point (gradients for layers $L, L-1, \ldots, k+1$ have been computed, and layer $k$'s gradient is about to be computed):

$$
G_{\text{acc}}(k) = \sum_{\ell=k+1}^{L} 4 N_\ell \approx 4N \cdot \frac{L - k}{L}
$$

(Approximate, assuming equal parameter distribution across layers.)

The total memory at backward step $k$:

$$
\boxed{M_{\text{bwd}}(k) = \underbrace{4N}_{\text{params}} + \underbrace{8N}_{\text{Adam } m,v} + \underbrace{A_{\text{rem}}(k)}_{\text{unfree'd activations}} + \underbrace{G_{\text{acc}}(k)}_{\text{accumulated gradients}}}
$$

---

### 3.5 Identifying the True Peak: Start of Backward ($k = L$)

The peak memory occurs at the **very beginning of the backward pass**, when:

- **All activations are still resident**: $A_{\text{rem}}(1) \approx A_{\text{tot}}$
- **Gradients are beginning to be allocated**: $G_{\text{acc}}$ is initially small but the gradient buffer for the full model is typically pre-allocated

In most frameworks (PyTorch, JAX), the full gradient buffer of size $4N$ is **allocated at once** when `.backward()` is called, not incrementally. Therefore, at the first instant of the backward pass:

$$
\boxed{M_{\text{peak}} = 4N + 8N + 4N + A_{\text{tot}} = 16N + A_{\text{tot}}}
$$

This is the **worst-case peak memory** — the absolute maximum GPU memory required during training.

**This is strictly greater than $16N$:**

$$
M_{\text{peak}} = 16N + A_{\text{tot}} > 16N \quad \text{since } A_{\text{tot}} > 0 \text{ for any non-trivial model/batch}
$$

---

### 3.6 Memory Decrease During Backward Pass

As the backward pass progresses from layer $L$ down to layer $1$:

- $A_{\text{rem}}(k)$ **decreases** (activations freed after each layer's backward)
- $G_{\text{acc}}(k)$ **increases** (gradients accumulate) — but since the gradient buffer is pre-allocated, this doesn't actually change total memory

Net effect: memory **decreases monotonically** during the backward pass, from the peak of $16N + A_{\text{tot}}$ toward $16N$ (all activations freed, all gradients computed).

---

### 3.7 Phase 4: Optimizer Step

After the backward pass completes:

| Component | Memory |
|---|---|
| Parameters $W$ | $4N$ |
| Gradients $\nabla_W \mathcal{L}$ | $4N$ |
| Adam $m_t$ | $4N$ |
| Adam $v_t$ | $4N$ |
| Activations | $0$ |

$$
M_{\text{opt}} = 16N
$$

The optimizer reads $\nabla_W \mathcal{L}$, updates $m_t$, $v_t$, and $W$ in-place. After the update, gradients are zeroed (or freed and re-allocated next step).

---

### 3.8 Phase 5: After Optimizer Step (Return to Baseline)

$$
\boxed{M_{\text{post-opt}} = 12N}
$$

Parameters ($4N$) + Adam states ($8N$) persist. Gradient buffers are zeroed/freed. The cycle repeats.

---

## 4. Complete Phase-Accurate Memory Table (FP32 + Adam)

| Phase | Parameters | Gradients | Adam $m, v$ | Activations | **Total** |
|---|---|---|---|---|---|
| Before forward (step $\geq 1$) | $4N$ | $0$ | $8N$ | $0$ | $\mathbf{12N}$ |
| End of forward | $4N$ | $0$ | $8N$ | $A_{\text{tot}}$ | $\mathbf{12N + A_{\text{tot}}}$ |
| Start of backward (**TRUE PEAK**) | $4N$ | $4N$ | $8N$ | $A_{\text{tot}}$ | $\mathbf{16N + A_{\text{tot}}}$ |
| During backward (layer $k$) | $4N$ | $4N$ | $8N$ | $A_{\text{rem}}(k)$ | $\mathbf{16N + A_{\text{rem}}(k)}$ |
| End of backward | $4N$ | $4N$ | $8N$ | $0$ | $\mathbf{16N}$ |
| After optimizer (steady state) | $4N$ | $0$ | $8N$ | $0$ | $\mathbf{12N}$ |

---

## 5. The System-Level Invariant

$$
\boxed{M_{\text{peak}}^{\text{training}} = 16N + A_{\text{tot}}}
$$

**In words**: *The peak training memory occurs at the start of the backward pass and equals the full parameter + gradient + optimizer state memory ($16N$) plus all unreleased activations ($A_{\text{tot}}$).*

This is an **exact invariant** (modulo constant CUDA context overhead and allocator fragmentation) that holds for:

- Any transformer architecture (encoder, decoder, encoder-decoder)
- Any depth $L$, hidden dimension $h$, batch size $bs$, sequence length $seq$
- FP32 training with the Adam optimizer

For **mixed precision (BF16 + FP32 master weights, BF16 gradients)**:

$$
M_{\text{peak}}^{\text{mixed}} = \underbrace{2N}_{\text{BF16 params}} + \underbrace{2N}_{\text{BF16 grads}} + \underbrace{4N}_{\text{FP32 master}} + \underbrace{8N}_{\text{Adam } m,v \text{ (FP32)}} + A_{\text{tot}}^{\text{BF16}} = 16N + A_{\text{tot}}^{\text{BF16}}
$$

For **mixed precision with FP32 gradient accumulation**:

$$
M_{\text{peak}}^{\text{mixed, FP32 grad}} = 2N + 2N + 4N + 4N + 8N + A_{\text{tot}}^{\text{BF16}} = 20N + A_{\text{tot}}^{\text{BF16}}
$$

---

## 6. Corrected Practical Memory Requirements

### 6.1 Activation Memory Estimates

Using the activation formula for mixed precision:

$$
A_{\text{tot}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)
$$

Example for Llama-2 7B ($L=32$, $h=4096$, $n_{\text{heads}}=32$, $bs=1$, $seq=4096$):

$$
A_{\text{tot}} = 32 \times 4096 \times 1 \times 4096 \times \left(34 + \frac{5 \times 32 \times 4096}{4096}\right)
$$

$$
= 32 \times 4096 \times 4096 \times (34 + 160)
$$

$$
= 32 \times 4096 \times 4096 \times 194
$$

$$
= 32 \times 4096 \times 794{,}624
$$

$$
\approx 104.1 \times 10^9 \text{ bytes} \approx 96.9 \text{ GB}
$$

### 6.2 Corrected Total Peak Memory Table (Mixed Precision, BF16 Gradients, $bs=1$, $seq=4096$)

| Model | $N$ | Params+Grads+Opt ($16N$) | $A_{\text{tot}}$ (est.) | **True Peak** ($16N + A_{\text{tot}}$) | H100 80GB Fits? |
|---|---|---|---|---|---|
| 1B | $10^9$ | 16 GB | $\sim$6 GB | **$\sim$22 GB** | ✅ |
| 7B | $7 \times 10^9$ | 112 GB | $\sim$97 GB | **$\sim$209 GB** | ❌ |
| 70B | $70 \times 10^9$ | 1,120 GB | $\sim$970 GB | **$\sim$2,090 GB** | ❌ |
| 405B | $405 \times 10^9$ | 6,480 GB | $\sim$5,600 GB | **$\sim$12,080 GB** | ❌ |

> The naïve "$16N$" table dramatically **underestimates** the true memory requirement by ignoring $A_{\text{tot}}$, which can be comparable to or larger than the parameter-related memory, especially for long sequences and large batch sizes.

---

## 7. Why This Correction Matters in Practice

### 7.1 OOM Debugging

When a training run crashes with OOM, practitioners often check only whether "$16N$ fits in GPU memory." This is insufficient. The true check must be:

$$
16N + A_{\text{tot}}(bs, seq) \leq M_{\text{GPU}} - M_{\text{CUDA context}}
$$

where $M_{\text{CUDA context}} \approx 1\text{–}2$ GB.

### 7.2 Batch Size / Sequence Length Selection

Since $A_{\text{tot}} \propto bs \cdot seq$ (linearly) and $A_{\text{tot}} \propto seq^2$ (quadratically, through the attention term), the maximum feasible $bs$ and $seq$ are **not** determined by $16N$ alone but by the **residual memory** after subtracting the persistent $12N$ baseline:

$$
A_{\text{tot}}^{\text{max}} = M_{\text{GPU}} - 16N - M_{\text{overhead}}
$$

$$
bs_{\text{max}} = \frac{A_{\text{tot}}^{\text{max}}}{L \cdot seq \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)}
$$

### 7.3 Impact on Activation Recomputation Decisions

With the corrected peak $16N + A_{\text{tot}}$, the value of activation recomputation becomes quantifiable. Selective recomputation reduces $A_{\text{tot}}$ by $\sim 70\%$:

$$
M_{\text{peak}}^{\text{selective}} = 16N + 0.3 \cdot A_{\text{tot}}
$$

Full recomputation reduces $A_{\text{tot}}$ to near zero (only layer-boundary checkpoints remain):

$$
M_{\text{peak}}^{\text{full}} \approx 16N + \mathcal{O}(L \cdot seq \cdot bs \cdot h)
$$

The memory saved is:

$$
\Delta M = A_{\text{tot}} - A_{\text{tot}}^{\text{recomp}} \approx 0.7 \cdot A_{\text{tot}} \quad \text{(selective)}
$$

This saving can convert an infeasible training configuration into a feasible one.

---

## 8. Summary: The Corrected System-Level Memory Invariant

$$
\boxed{M_{\text{peak}}^{\text{train}} = \underbrace{16N}_{\substack{\text{params (4N) +}\\\text{grads (4N) +}\\\text{Adam } m,v \text{ (8N)}}} + \underbrace{A_{\text{tot}}(bs, seq, L, h, n_{\text{heads}})}_{\substack{\text{all activations still}\\\text{resident at backward start}}} + \underbrace{M_{\text{overhead}}}_{\substack{\text{CUDA context,}\\\text{fragmentation,}\\\text{buffers}}}}
$$

The steady-state baseline (between steps) is:

$$
M_{\text{baseline}} = 12N
$$

The peak-to-baseline ratio:

$$
\frac{M_{\text{peak}}}{M_{\text{baseline}}} = \frac{16N + A_{\text{tot}}}{12N} = \frac{4}{3} + \frac{A_{\text{tot}}}{12N}
$$

For large batch sizes or long sequences where $A_{\text{tot}} \gg N$, the peak can be **many times** the baseline, making the naïve $16N$ estimate not just slightly wrong but **qualitatively misleading** about whether a training configuration will fit in memory.