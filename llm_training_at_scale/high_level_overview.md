# High-Level Overview of Distributed Training: Foundations, Memory Analysis, and First-Step Techniques

---

## 1. The Three Fundamental Challenges of Scalable Training

Every technique in large-scale model training addresses one or more of three orthogonal resource constraints:

### 1.1 Memory Usage (Hard Constraint)

Memory is a **binary gate**: if the aggregate of model parameters, gradients, optimizer states, and activations exceeds available GPU VRAM, the training step **cannot execute at all**. There is no graceful degradation — the process terminates with an Out-Of-Memory (OOM) error.

### 1.2 Compute Efficiency (Utilization Constraint)

Hardware accelerators achieve peak throughput only when arithmetic logic units (ALUs) are saturated with floating-point operations. Any time spent on:

- Memory reads/writes (bandwidth-bound operations)
- Kernel launch overhead
- Waiting for synchronization barriers

represents **wasted compute capacity**. The goal is to maximize the ratio:

$$
\text{Compute Utilization} = \frac{\text{Time spent on FLOPs}}{\text{Total wall-clock time}}
$$

### 1.3 Communication Overhead (Coordination Constraint)

In multi-GPU settings, GPUs must exchange data (gradients, activations, parameters). Communication has two distinct regimes:

| Link Type | Typical Bandwidth | Latency |
|-----------|-------------------|---------|
| **Intra-node** (NVLink, NVSwitch) | 450–900 GB/s per GPU (H100) | ~1–5 $\mu$s |
| **Inter-node** (InfiniBand, RoCE) | 50–400 Gb/s per NIC | ~1–10  $\mu$s + propagation |

Any time a GPU is **idle** waiting for a remote tensor constitutes communication overhead. The primary mitigation strategies are:

1. **Overlap** communication with computation (hide latency behind useful work)
2. **Minimize** total bytes transferred
3. **Prefer** intra-node links over inter-node links when possible

### 1.4 The Fundamental Trade-off Triangle

These three resources are **fungible** — one can be traded for another:

| Technique | Saves | Costs |
|-----------|-------|-------|
| Activation Recomputation | Memory | Compute |
| Tensor Parallelism | Memory | Communication |
| Gradient Accumulation | Memory (activations) | Compute (sequential passes) |
| Gradient Compression | Communication | Compute + slight accuracy |

Finding the optimal operating point within this triangle is the central systems-level challenge of distributed training.

---

## 2. Training on One GPU: The Canonical Three-Step Loop

### 2.1 The Three Phases

All neural network training, regardless of scale, consists of three atomic operations per optimization step:

**Phase 1 — Forward Pass:**
Given a mini-batch of inputs $\mathbf{X} \in \mathbb{R}^{bs \times seq \times d_{\text{input}}}$, compute the model output by successively applying each layer $f_\ell$ with parameters $\theta_\ell$:

$$
\mathbf{a}_0 = \mathbf{X}, \quad \mathbf{a}_\ell = f_\ell(\mathbf{a}_{\ell-1};\, \theta_\ell), \quad \ell = 1, \dots, L
$$

The final output $\mathbf{a}_L$ is used to compute the scalar loss:

$$
\mathcal{L} = \mathcal{L}(\mathbf{a}_L,\, \mathbf{y})
$$

where $\mathbf{y}$ denotes the target labels.

**Phase 2 — Backward Pass:**
Compute gradients via the chain rule (backpropagation). For each layer $\ell = L, L-1, \dots, 1$:

$$
\frac{\partial \mathcal{L}}{\partial \theta_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_\ell} \cdot \frac{\partial \mathbf{a}_\ell}{\partial \theta_\ell}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}_{\ell-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_\ell} \cdot \frac{\partial \mathbf{a}_\ell}{\partial \mathbf{a}_{\ell-1}}
$$

This requires the stored activations $\mathbf{a}_{\ell-1}$ from the forward pass — a critical fact for memory analysis.

**Phase 3 — Optimizer Step:**
Update parameters using the computed gradients. For the Adam optimizer:

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

where $\mathbf{g}_t = \nabla_\theta \mathcal{L}$, $\beta_1, \beta_2$ are decay rates, $\eta$ is the learning rate, and $\epsilon$ is a numerical stability constant.

---

## 3. Batch Size: Convergence, Throughput, and Token-Based Reporting

### 3.1 Definition and Impact on Convergence

The **batch size** ($bs$) is the number of independent training samples processed before a single parameter update. It has two opposing effects:

- **Small $bs$**: Gradient estimates have high variance $\text{Var}[\hat{\mathbf{g}}] \propto \frac{\sigma^2}{bs}$, causing noisy updates. Useful early in training for rapid exploration of the loss landscape, but impedes precise convergence later.
- **Large $bs$**: Gradient estimates approach the true gradient $\mathbb{E}[\nabla_\theta \mathcal{L}]$, but each token contributes less unique information per optimizer step — convergence in terms of tokens consumed becomes slower, potentially wasting compute.

The critical batch size $B_{\text{crit}}$, defined in the OpenAI scaling work, characterizes the transition point:

$$
B_{\text{crit}} = \frac{B_{\text{noise}}}{1} \quad \text{where} \quad B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{G^T G}
$$

Here, $G = \mathbb{E}[\nabla_\theta \mathcal{L}]$ is the true gradient and $\Sigma$ is the covariance of per-sample gradients. Below $B_{\text{crit}}$, increasing batch size improves both time efficiency and sample efficiency. Above $B_{\text{crit}}$, only time efficiency improves (each step processes more data, but more data is needed for the same loss reduction).

### 3.2 Practical Batch Size Schedules

Modern LLM training frequently uses **batch size warm-up**:

| Training | Initial $bs$ (sequences) | Final $bs$ (sequences) | Transition Point |
|----------|--------------------------|------------------------|------------------|
| DeepSeek-V3/R1 | 3,072 | 15,360 | After 469B tokens |

### 3.3 Token-Based Batch Size

Because input sequences may vary in length across training configurations, the community reports batch size in tokens:

$$
bs_t = bs \times seq
$$

where $seq$ is the input sequence length and $bs$ is the batch size in samples (number of sequences).

**Typical ranges for modern LLM pretraining:**

| Model | $bs_t$ (tokens per batch) | Total Training Tokens |
|-------|---------------------------|----------------------|
| Llama 1 | ~4M | 1.4T |
| DeepSeek-V3 | ~60M | 14T |

### 3.4 Sensitivity

An important empirical observation: final model performance exhibits **low sensitivity** to the exact batch size value in a neighborhood around the optimum. This provides practical flexibility when tuning batch size for hardware constraints.

---

## 4. Memory Usage in Transformer Training

### 4.1 The Four Memory Occupants

During training, GPU VRAM must simultaneously hold:

1. **Model weights** $\theta$
2. **Gradients** $\nabla_\theta \mathcal{L}$
3. **Optimizer states** (e.g., Adam's $\mathbf{m}$ and $\mathbf{v}$)
4. **Activations** $\{\mathbf{a}_0, \mathbf{a}_1, \dots, \mathbf{a}_L\}$ and all intermediate tensors needed for gradient computation

**Additional (constant) overhead:**
- CUDA kernel context: ~1–2 GB (verified via `import torch; torch.ones((1,1)).to("cuda")` then `nvidia-smi`)
- Internal buffers and memory fragmentation (typically small)

### 4.2 Numerical Precision and Bytes Per Element

| Format | Bits | Bytes per element | Exponent bits | Mantissa bits |
|--------|------|-------------------|---------------|---------------|
| FP32 | 32 | 4 | 8 | 23 |
| BF16 | 16 | 2 | 8 | 7 |
| FP16 | 16 | 2 | 5 | 10 |
| FP8 (E4M3) | 8 | 1 | 4 | 3 |

---

## 5. Parameter Count for Transformer LLMs

For a standard decoder-only transformer **without** fixed positional embeddings, the total parameter count is:

$$
N = h \times v + L \times (12h^2 + 13h) + 2h
$$

where:

| Symbol | Meaning |
|--------|---------|
| $h$ | Hidden dimension |
| $v$ | Vocabulary size |
| $L$ | Number of transformer layers |

**Derivation of the per-layer term $12h^2 + 13h$:**

Each transformer layer contains:

| Sub-module | Parameters |
|------------|-----------|
| Self-attention QKV projection | $3 \times h \times h = 3h^2$ |
| Self-attention output projection | $h \times h = h^2$ |
| MLP up-projection (typically $4h$) | $h \times 4h = 4h^2$ |
| MLP down-projection | $4h \times h = 4h^2$ |
| **Total weight matrices** | $12h^2$ |
| LayerNorm (×2, each with scale + bias) | $2 \times 2h = 4h$ |
| Bias terms in QKV, output, MLP layers | Various, summing to $9h$ |
| **Total biases + norms** | $13h$ |

The **embedding layer** contributes $h \times v$ parameters. The **final LayerNorm** before the output head contributes $2h$.

**Scaling insight:** As $h$ grows, the $12Lh^2$ term dominates quadratically, making the model size approximately:

$$
N \approx 12Lh^2 \quad \text{(for large } h \text{)}
$$

---

## 6. Memory for Weights, Gradients, and Optimizer States

### 6.1 Full Precision (FP32) Training

All tensors stored in FP32 (4 bytes each):

$$
m_{\text{params}} = 4N \text{ bytes}
$$

$$
m_{\text{grad}} = 4N \text{ bytes}
$$

For the Adam optimizer, which maintains first moment $\mathbf{m}$ and second moment $\mathbf{v}$:

$$
m_{\text{opt}} = (4 + 4) \times N = 8N \text{ bytes}
$$

**Total (FP32):**

$$
m_{\text{total}}^{\text{FP32}} = 4N + 4N + 8N = 16N \text{ bytes}
$$

### 6.2 Mixed Precision (BF16 + FP32 Master Weights) Training

The standard mixed precision scheme:

| Component | Precision | Bytes per parameter |
|-----------|-----------|-------------------|
| Working parameters (forward/backward) | BF16 | 2 |
| Gradients (forward/backward) | BF16 | 2 |
| Master weights (optimizer copy) | FP32 | 4 |
| Adam first moment $\mathbf{m}$ | FP32 | 4 |
| Adam second moment $\mathbf{v}$ | FP32 | 4 |

$$
m_{\text{params}} = 2N
$$

$$
m_{\text{grad}} = 2N
$$

$$
m_{\text{params\_fp32}} = 4N
$$

$$
m_{\text{opt}} = (4 + 4) \times N = 8N
$$

**Total (mixed precision without FP32 grad accumulation):**

$$
m_{\text{total}}^{\text{mixed}} = 2N + 2N + 4N + 8N = 16N \text{ bytes}
$$

**Total (mixed precision with FP32 gradient accumulation):**

Some libraries (e.g., Nanotron) store an additional FP32 copy of gradients for numerical stability:

$$
m_{\text{total}}^{\text{mixed+fp32grad}} = 2N + 2N + 4N + 4N + 8N = 20N \text{ bytes}
$$

### 6.3 Key Insight

Mixed precision does **not** reduce total weight/gradient/optimizer memory; the total bytes per parameter remain $16N$ (or $20N$ with FP32 gradient accumulation). The advantage lies elsewhere:

1. **Faster arithmetic**: BF16 matrix multiplications achieve 2× or greater throughput on tensor cores compared to FP32.
2. **Reduced activation memory**: Activations during forward/backward are stored in BF16 (2 bytes instead of 4), which is the dominant memory consumer at scale.

### 6.4 Practical Memory Table

| Model Size ($N$) | FP32 or BF16 (without FP32 grad acc) | BF16 (with FP32 grad acc) |
|---------------------|---------------------------------------|---------------------------|
| 1B | $16N = $ **16 GB** | $20N = $ **20 GB** |
| 7B | **112 GB** | **140 GB** |
| 70B | **1,120 GB** | **1,400 GB** |
| 405B | **6,480 GB** | **8,100 GB** |

For reference, a single NVIDIA H100 SXM has **80 GB** of HBM3 VRAM. At 7B parameters, the weight/gradient/optimizer memory alone ($112$–$140$ GB) already exceeds a single GPU's capacity — before even accounting for activations.

---

## 7. Memory for Activations

### 7.1 Why Activations Must Be Stored

During the backward pass, computing $\frac{\partial \mathcal{L}}{\partial \theta_\ell}$ requires the input activation $\mathbf{a}_{\ell-1}$. Similarly, operations like softmax, layer normalization, and GeLU require their own inputs to compute local Jacobians. Therefore, **all intermediate activations** between learnable operations must be retained from the forward pass until consumed during the backward pass.

### 7.2 Activation Memory Formula

For a transformer model in mixed precision (BF16 activations), the total activation memory is:

$$
m_{\text{act}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)
$$

where:

| Symbol | Meaning |
|--------|---------|
| $L$ | Number of transformer layers |
| $seq$ | Sequence length |
| $bs$ | Batch size (in samples) |
| $h$ | Hidden dimension |
| $n_{\text{heads}}$ | Number of attention heads |

**Derivation sketch** (following Korthikanti et al., 2022):

Within each transformer layer, the intermediate activations that must be stored include:

| Operation | Stored Activation | Size (elements) | Bytes (BF16) |
|-----------|-------------------|-----------------|--------------|
| Input to self-attention LayerNorm | $\mathbf{a}_{\text{ln1}}$ | $bs \times seq \times h$ | $2 \cdot bs \cdot seq \cdot h$ |
| Q, K, V projections (3 matrices) | $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | $3 \times bs \times seq \times h$ | $6 \cdot bs \cdot seq \cdot h$ |
| Attention scores (pre-softmax) | $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$ | $bs \times n_{\text{heads}} \times seq \times seq$ | $2 \cdot bs \cdot n_{\text{heads}} \cdot seq^2$ |
| Attention weights (post-softmax) | $\mathbf{A} = \text{softmax}(\mathbf{S})$ | $bs \times n_{\text{heads}} \times seq \times seq$ | $2 \cdot bs \cdot n_{\text{heads}} \cdot seq^2$ |
| Dropout mask (attention) | binary | $bs \times n_{\text{heads}} \times seq \times seq$ | $bs \cdot n_{\text{heads}} \cdot seq^2$ |
| Attention output projection input | $\mathbf{a}_{\text{attn\_out}}$ | $bs \times seq \times h$ | $2 \cdot bs \cdot seq \cdot h$ |
| Residual + LayerNorm input to MLP | $\mathbf{a}_{\text{ln2}}$ | $bs \times seq \times h$ | $2 \cdot bs \cdot seq \cdot h$ |
| MLP intermediate (up-projected) | $\mathbf{a}_{\text{mlp\_up}}$ | $bs \times seq \times 4h$ | $8 \cdot bs \cdot seq \cdot h$ |
| GeLU/activation function input | $\mathbf{a}_{\text{gelu}}$ | $bs \times seq \times 4h$ | $8 \cdot bs \cdot seq \cdot h$ |
| MLP down-projection input | | $bs \times seq \times 4h$ | $8 \cdot bs \cdot seq \cdot h$ |
| Dropout masks (×2) | binary | $2 \times bs \times seq \times h$ | $2 \cdot bs \cdot seq \cdot h$ |

Summing the terms proportional to $bs \cdot seq \cdot h$ yields the constant factor $34$, and summing the attention-score terms proportional to $bs \cdot n_{\text{heads}} \cdot seq^2$ yields the factor $5 \cdot n_{\text{heads}} \cdot seq / h$.

### 7.3 Scaling Behavior

Two critical observations:

1. **Linear scaling with $bs$:** Doubling the batch size doubles activation memory.
2. **Quadratic scaling with $seq$:** The attention score matrices scale as $O(seq^2)$ per layer.

$$
m_{\text{act}} = O(L \cdot bs \cdot seq \cdot h) + O(L \cdot bs \cdot n_{\text{heads}} \cdot seq^2)
$$

For short sequences, the $O(seq \cdot h)$ term dominates and activation memory is modest. For long sequences ($seq \gtrsim 2\text{k}$–$4\text{k}$), the quadratic term dominates, and **activations become the largest single memory consumer**, dwarfing parameters, gradients, and optimizer states combined.

By contrast, $m_{\text{params}}, m_{\text{grad}}, m_{\text{opt}}$ are **independent** of $bs$ and $seq$.

---

## 8. Activation Recomputation (Gradient Checkpointing / Rematerialization)

### 8.1 Core Idea

**Trade compute for memory.** Instead of storing all intermediate activations during the forward pass, discard most of them and **recompute** them on-the-fly during the backward pass from a small set of saved "checkpoint" activations.

Formally, without recomputation, we store:

$$
\{\mathbf{a}_0, \mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_L\} \quad \text{and all sub-layer intermediates}
$$

With recomputation, we store only:

$$
\{\mathbf{a}_0, \mathbf{a}_{c_1}, \mathbf{a}_{c_2}, \dots, \mathbf{a}_{c_k}\} \quad \text{where } \{c_1, \dots, c_k\} \subset \{1, \dots, L\}
$$

When the backward pass requires $\mathbf{a}_\ell$ for some $\ell \notin \{c_1, \dots, c_k\}$, we recompute it by running a partial forward pass from the nearest preceding checkpoint $\mathbf{a}_{c_j}$ where $c_j < \ell$.

### 8.2 Strategies

#### 8.2.1 Full Recomputation

Checkpoint **only at layer boundaries**: store $\{\mathbf{a}_0, \mathbf{a}_1, \dots, \mathbf{a}_L\}$ at the layer level but discard all sub-layer intermediates.

- **Memory saved:** All intra-layer activation intermediates (the dominant component).
- **Compute cost:** Essentially one additional full forward pass during the backward pass.
- **Overhead:** Typically **30–40%** increase in wall-clock time.

Activation memory with full recomputation reduces to approximately:

$$
m_{\text{act}}^{\text{full\_recomp}} = L \cdot bs \cdot seq \cdot h \cdot 2 \quad \text{(only layer-boundary activations in BF16)}
$$

#### 8.2.2 Selective Recomputation

Observation from Korthikanti et al. (2022): the **attention score matrices** ($bs \times n_{\text{heads}} \times seq \times seq$) are the largest activations but are **cheapest** to recompute (they involve only a batched matrix multiplication $\mathbf{Q}\mathbf{K}^T$ and a softmax, both of which are relatively inexpensive in FLOPs relative to the memory they consume).

Strategy: **Discard** attention scores and softmax outputs; **keep** the expensive feedforward (MLP) activations.

- **GPT-3 (175B) empirical result:** ~**70%** activation memory reduction at only **2.7%** compute cost.
- **FlashAttention** natively implements this strategy: it recomputes attention scores in the backward pass from $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ blocks, never materializing the full $seq \times seq$ attention matrix.

#### 8.2.3 DeepSeek-V3 / Multi-Head Latent Attention (MLA)

MLA compresses the key-value cache into a low-rank latent space, reducing the activation footprint of attention even further beyond standard selective recomputation.

### 8.3 FLOPS Utilization Metrics

Recomputation changes the total number of floating-point operations performed, which affects how we measure hardware efficiency:

**Hardware FLOPS Utilization (HFU):**

$$
\text{HFU} = \frac{\text{Hardware FLOPs (including recomputation)}}{\text{Step duration (seconds)} \times \text{Peak accelerator FLOPS}}
$$

**Model FLOPS Utilization (MFU):**

$$
\text{MFU} = \frac{\text{Model FLOPs (forward + backward only, no recomputation)}}{\text{Step duration (seconds)} \times \text{Peak accelerator FLOPS}}
$$

MFU is the preferred metric for comparing accelerators and training configurations, because it measures **useful** work per unit time, independent of implementation-level choices like recomputation.

### 8.4 A Counter-Intuitive Performance Effect

Although recomputation increases total FLOPs, it **reduces memory traffic**. On bandwidth-limited hardware (which GPUs often are), fewer memory accesses can make overall execution **faster** despite more arithmetic — a net win in both memory and speed.

---

## 9. Gradient Accumulation

### 9.1 Problem Statement

Even with activation recomputation, activation memory scales linearly with $bs$. Modern LLM training targets $bs_t \sim 4$M–$60$M tokens. For $seq = 4096$, this implies $bs \sim 1000$–$15000$ samples — far too many to fit in a single GPU's memory simultaneously.

### 9.2 Mechanism

Split the global batch into $\text{grad\_acc}$ **micro-batches**, each of size $mbs$:

$$
gbs = mbs \times \text{grad\_acc}
$$

$$
bs_t = gbs \times seq = mbs \times \text{grad\_acc} \times seq
$$

Execute the training loop as:

```
gradient_buffer = 0
for i in range(grad_acc):
    micro_batch = get_micro_batch(i)
    loss = forward(micro_batch) / grad_acc   # normalize
    loss.backward()                            # accumulates into gradient_buffer
optimizer.step()
optimizer.zero_grad()
```

The division by $\text{grad\_acc}$ ensures the accumulated gradient equals the **mean** gradient over the global batch:

$$
\nabla_\theta \mathcal{L}_{\text{global}} = \frac{1}{\text{grad\_acc}} \sum_{i=1}^{\text{grad\_acc}} \nabla_\theta \mathcal{L}_i
$$

This makes the result **mathematically identical** to processing the full global batch at once (assuming no batch normalization or similar batch-dependent operations).

### 9.3 Memory Analysis

| Component | Without Gradient Accumulation | With Gradient Accumulation |
|-----------|-------------------------------|----------------------------|
| Parameters | $m_{\text{params}}$ | $m_{\text{params}}$ (unchanged) |
| Gradients | $m_{\text{grad}}$ | $m_{\text{grad}}$ (persistent buffer) |
| Optimizer | $m_{\text{opt}}$ | $m_{\text{opt}}$ (unchanged) |
| Activations | $m_{\text{act}}(gbs)$ | $m_{\text{act}}(mbs) \ll m_{\text{act}}(gbs)$ |

The activation memory is reduced by a factor of $\text{grad\_acc}$:

$$
m_{\text{act}}^{\text{GA}} = \frac{m_{\text{act}}^{\text{full}}}{\text{grad\_acc}}
$$

**Trade-off:** The gradient buffer must persist across all micro-batch iterations (it is not freed until after `optimizer.step()`), creating a small additional memory overhead compared to the non-accumulation case where gradients are computed and freed layer-by-layer during the backward pass. However, this is vastly outweighed by the activation memory savings.

### 9.4 Compute Cost

Gradient accumulation is **sequential**: $\text{grad\_acc}$ forward-backward passes execute one after another. Total compute per optimizer step is identical to processing the full batch, but wall-clock time increases proportionally because there is no parallelism across micro-batches on a single GPU:

$$
t_{\text{step}}^{\text{GA}} \approx \text{grad\_acc} \times t_{\text{micro-step}}
$$

### 9.5 The Path to Data Parallelism

A critical observation: the $\text{grad\_acc}$ micro-batch forward-backward passes are **independent** computations (they share parameters but operate on disjoint data). This independence is precisely what enables **data parallelism** — distributing micro-batches across multiple GPUs and computing them simultaneously, then synchronizing gradients before the optimizer step.

---

## 10. Profiling GPU Compute and Communication

### 10.1 PyTorch Profiler

The PyTorch profiler instruments both CPU and CUDA activity, generating traces viewable in TensorBoard or Chrome's `chrome://tracing`:

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

| Parameter | Purpose |
|-----------|---------|
| `wait=1` | Skip 1 step (cold start) |
| `warmup=1` | Profile 1 step but discard (cache warming) |
| `active=3` | Record 3 steps for analysis |
| `with_stack=True` | Include Python call stacks in trace |

### 10.2 What the Trace Reveals

The profiler trace shows multiple concurrent tracks:

1. **CPU thread(s):** Launching CUDA kernels asynchronously, managing data loading, executing Python logic.
2. **CUDA compute stream(s):** Executing matrix multiplications, activations, normalization kernels.
3. **CUDA communication stream(s):** Executing NCCL collectives (AllReduce, AllGather, ReduceScatter) for gradient synchronization.

### 10.3 Key Bottleneck Patterns to Identify

| Pattern | Symptom in Trace | Root Cause |
|---------|-----------------|------------|
| Sequential compute + communication | Communication kernel starts only after all backward kernels finish | Missing overlap of gradient sync with backward pass |
| GPU idle gaps | Empty regions in CUDA compute stream | CPU-side bottleneck (data loading, Python overhead) or CUDA synchronization barriers |
| Excessive `cudaMemcpy` | Large H2D/D2H blocks | Data not pre-pinned or pre-staged on GPU |
| Kernel launch overhead | Many tiny CUDA kernels with gaps between them | Operator fusion needed (e.g., via `torch.compile`) |
| First step anomaly | Longer first iteration with memory allocation plateaus | PyTorch caching allocator warming up memory pools |

### 10.4 First-Step vs. Steady-State Behavior

The profiler reveals a characteristic difference:

- **Step 1:** Activations ramp up, then plateau as the PyTorch CUDA caching allocator pre-allocates memory blocks. Optimizer states do not yet exist.
- **Step 2+:** Optimizer states ($\mathbf{m}$ and $\mathbf{v}$ for Adam) are allocated after the first `optimizer.step()`, **permanently increasing** the memory baseline by $8N$ bytes. This explains why **training can succeed on step 1 but OOM on step 2**.

---

## 11. Memory Budget Summary

For a transformer with $N$ parameters, in mixed precision (BF16 compute, FP32 master weights, Adam optimizer), the total GPU memory requirement is:

$$
\boxed{m_{\text{total}} = \underbrace{2N}_{\text{BF16 params}} + \underbrace{2N}_{\text{BF16 grads}} + \underbrace{4N}_{\text{FP32 master weights}} + \underbrace{8N}_{\text{Adam states}} + \underbrace{m_{\text{act}}(L, seq, mbs, h, n_{\text{heads}})}_{\text{Activations}} + \underbrace{\sim 2 \text{ GB}}_{\text{CUDA context}}}
$$

$$
= 16N + m_{\text{act}} + O(1) \quad \text{(without FP32 grad accumulation)}
$$

$$
= 20N + m_{\text{act}} + O(1) \quad \text{(with FP32 grad accumulation)}
$$

where

$$
m_{\text{act}} = L \cdot seq \cdot mbs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)
$$

and $mbs$ replaces $bs$ when gradient accumulation is used.

With **full activation recomputation**, $m_{\text{act}}$ is reduced to approximately $2 \cdot L \cdot seq \cdot mbs \cdot h$ bytes (storing only layer-boundary activations in BF16).

With **selective recomputation**, $m_{\text{act}}$ is reduced by ~70% (discarding the $O(n_{\text{heads}} \cdot seq^2)$ attention terms) at ~2.7% compute overhead.

---

## 12. Conceptual Map: From Single GPU to Distributed Training

```
Single-GPU Training
├── Forward → Backward → Optimize
├── Memory constraints
│   ├── Weights + Grads + Optimizer: 16N–20N bytes (fixed per model)
│   └── Activations: O(L · bs · seq · h + L · bs · n_heads · seq²) (variable)
├── Memory mitigation
│   ├── Activation Recomputation (trade compute ↔ memory)
│   └── Gradient Accumulation (trade time ↔ memory)
└── Next step: Data Parallelism
    └── Parallelize independent micro-batch computations across GPUs
```

The independent micro-batch computations identified through gradient accumulation form the natural entry point to **data parallelism**, where multiple GPUs execute forward-backward passes on different micro-batches simultaneously, synchronize gradients via collective communication (AllReduce), and perform a unified optimizer step — the subject of the next stage of scaling.