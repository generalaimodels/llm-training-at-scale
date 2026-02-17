

# Data Parallelism (DP): A Comprehensive Technical Treatment

---

## 1. Foundational Concept

**Data Parallelism (DP)** is a distributed training strategy in which the **entire model** is replicated across $N_d$ GPUs (called **model instances** or **replicas**), and the global mini-batch is partitioned into $N_d$ disjoint **micro-batches**, one per GPU. Each replica independently executes the forward and backward passes on its own micro-batch **in parallel**. Because different micro-batches produce different gradients, all replicas must **synchronize** their gradients before the optimizer step to ensure every replica remains an identical copy of the model.

**Core invariant of DP:** at every training step, after synchronization, all $N_d$ replicas hold **exactly the same** parameters $\theta$.

---

## 2. Distributed Communication Primitives

Before analyzing DP optimizations, three collective communication primitives are essential:

| Primitive | Description |
|---|---|
| **Broadcast** | One rank sends a tensor to all other ranks. |
| **All-Reduce** | Every rank contributes a tensor; the element-wise reduction (e.g., sum) is computed and the result is distributed back to **all** ranks. |
| **Reduce-Scatter** | Every rank contributes a tensor; the element-wise reduction is computed, but the result is **scattered** — each rank receives only a $\frac{1}{N_d}$ slice of the reduced tensor. |
| **All-Gather** | Each rank holds a $\frac{1}{N_d}$ slice; slices are gathered so that **every** rank ends up with the full tensor. |

A critical identity linking these primitives:

$$
\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}
$$

This decomposition is the mathematical backbone of ZeRO optimizations.

---

## 3. Naive Data Parallelism

### 3.1 Algorithm

Given $N_d$ GPUs, parameters $\theta$, and a global batch $\mathcal{B}$ split into micro-batches $\{b_1, b_2, \ldots, b_{N_d}\}$:

1. **Forward pass** on GPU $i$: compute loss $\mathcal{L}_i = \mathcal{L}(\theta; b_i)$.
2. **Backward pass** on GPU $i$: compute local gradients $g_i = \nabla_\theta \mathcal{L}_i$.
3. **All-Reduce** across all $N_d$ ranks:
$$
\bar{g} = \frac{1}{N_d} \sum_{i=1}^{N_d} g_i
$$
4. **Optimizer step** on every GPU: $\theta \leftarrow \theta - \eta \cdot \text{Optimizer}(\bar{g})$.

### 3.2 The Problem: Sequential Computation → Communication

In the naive implementation, the entire backward pass must **complete** before the all-reduce is launched. During the all-reduce, **all GPUs are idle** — no computation overlaps with communication. This is a critical inefficiency:

$$
T_{\text{naive}} = T_{\text{forward}} + T_{\text{backward}} + T_{\text{all-reduce}} + T_{\text{optimizer}}
$$

The term $T_{\text{all-reduce}}$ is purely wasted time where GPUs sit idle. At scale, this becomes a dominant bottleneck.

---

## 4. Three Key Optimizations

### 4.1 First Optimization: Overlap Gradient Synchronization with Backward Pass

**Key insight:** Gradients for layer $l$ are available as soon as the backward pass through layer $l$ is complete. We do **not** need to wait for gradients from earlier layers $l-1, l-2, \ldots, 1$.

**Mechanism:** Attach a **post-accumulate gradient hook** to each parameter. As soon as the gradient for parameter $p$ is computed, an all-reduce is **immediately triggered** for that gradient, while the backward pass continues computing gradients for earlier layers.

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model 
    that require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

**Result:** The effective training step time becomes:

$$
T_{\text{overlap}} = T_{\text{forward}} + \max\!\Big(T_{\text{backward}},\; T_{\text{all-reduce}}\Big) + T_{\text{optimizer}}
$$

When $T_{\text{backward}} \geq T_{\text{all-reduce}}$, the communication is **fully hidden** behind computation, and the all-reduce cost effectively becomes zero.

---

### 4.2 Second Optimization: Bucketing Gradients

**Problem:** Launching an independent all-reduce for every individual parameter tensor results in many small communication operations. GPUs and interconnects are **far more efficient** on large, contiguous tensors than on many small ones due to:

- Per-operation **launch latency**
- Suboptimal **bandwidth utilization** for small messages

**Solution — Bucketing:** Group multiple parameter gradients into **buckets** of a fixed size (e.g., 25 MB in PyTorch DDP). A **single** all-reduce is launched per bucket rather than per parameter.

$$
\text{Communication operations:} \quad \frac{|\theta|}{\text{bucket\_size}} \ll |\text{parameters}|
$$

**Analogy:** Shipping a few large boxes is more efficient than shipping many small packages — the fixed overhead per shipment is amortized over more data.

**Combined with Optimization 1:** Buckets are filled as gradients become available during the backward pass, and an all-reduce is triggered when a bucket is full, overlapping with ongoing backward computation.

---

### 4.3 Third Optimization: Interplay with Gradient Accumulation

**Gradient accumulation** performs $G$ sequential forward-backward passes before a single optimizer step, effectively simulating a larger batch:

$$
g_{\text{accumulated}} = \sum_{j=1}^{G} \nabla_\theta \mathcal{L}(\theta; b_j)
$$

**Problem:** In a naive DP + gradient accumulation setup, an all-reduce is triggered after **every** backward pass (all $G$ of them). This is wasteful — only the **final** accumulated gradient needs synchronization.

**Solution:** Use a **no-sync context manager** (`model.no_sync()` in PyTorch) to disable gradient synchronization for the first $G-1$ backward passes. Only on the $G$-th (final) backward pass is the all-reduce enabled:

$$
\text{Communication cost} = 1 \times T_{\text{all-reduce}} \quad \text{instead of} \quad G \times T_{\text{all-reduce}}
$$

---

### 4.4 Memory Contiguity Note

Communication operations require tensors to be **contiguous in memory**. In practice, **preallocated contiguous buffers** of size equal to the activations or model parameters are reserved exclusively for communication. This:

- **Speeds up** communication (avoids redundant memory copies)
- **Increases peak memory** usage (the buffer itself consumes GPU memory)

---

## 5. Global Batch Size Equation

With data parallelism and gradient accumulation, the **global batch size** (in samples) is:

$$
\boxed{bs = gbs = mbs \times grad\_acc \times dp}
$$

where:

| Symbol | Meaning |
|---|---|
| $gbs$ | Global batch size (total samples per training step) |
| $mbs$ | Micro-batch size (samples per GPU per forward-backward pass) |
| $grad\_acc$ | Number of gradient accumulation steps |
| $dp$ | Data parallelism degree (number of GPU replicas) |

**In tokens:**

$$
\text{Global batch size (tokens)} = gbs \times S
$$

where $S$ is the sequence length.

### Practical Prioritization

Given a target $gbs$:

$$
grad\_acc = \frac{gbs}{mbs \times dp}
$$

**Maximize $dp$ first, then use $grad\_acc$ to fill the remainder.** This is because:

- Data parallelism is **inherently parallel** — all micro-batches process simultaneously
- Gradient accumulation is **inherently sequential** — each accumulation step runs one after another

---

## 6. Practical Recipe for 1D Data-Parallel Training

**Step-by-step procedure:**

1. **Determine optimal $gbs$ (in tokens):** Consult literature (e.g., Chinchilla scaling laws) or run convergence experiments.

2. **Select sequence length $S$:** Typically $S \in [2048, 8192]$ tokens for pretraining. Most web documents are shorter than this range, making longer sequences yield diminishing returns during main pretraining.

3. **Compute $gbs$ in samples:**
$$
gbs = \frac{\text{Global batch size in tokens}}{S}
$$

4. **Find maximum $mbs$:** Increase $mbs$ on a single GPU until GPU memory is exhausted.

5. **Determine $dp$:** Set $dp$ to the number of available GPUs.

6. **Compute $grad\_acc$:**
$$
grad\_acc = \frac{gbs}{mbs \times dp}
$$

### Concrete Example

| Parameter | Value |
|---|---|
| Target $gbs$ (tokens) | 4,194,304 (4M) |
| Sequence length $S$ | 4,096 |
| $gbs$ (samples) | $\frac{4{,}194{,}304}{4{,}096} = 1{,}024$ |
| $mbs$ (memory-limited) | 2 |

**With 128 GPUs:**

$$
grad\_acc = \frac{1{,}024}{2 \times 128} = 4
$$

**With 512 GPUs:**

$$
grad\_acc = \frac{1{,}024}{2 \times 512} = 1
$$

The second scenario is faster because gradient accumulation is eliminated (no sequential steps), and all computation is fully parallel.

**GPU-rich case ($grad\_acc < 1$):** If $mbs \times dp > gbs$, options include:

- Not using all GPUs
- Increasing $gbs$ (if training dynamics allow)
- Reducing $mbs$ below its maximum to prioritize throughput over per-GPU compute efficiency

---

## 7. Scaling Limitations of Data Parallelism

### 7.1 Communication Overhead at Scale

As $dp$ grows (hundreds to thousands of GPUs), the all-reduce communication overhead grows due to:

- **Ring latency:** The minimum time for a signal to propagate once around the communication ring, which is proportional to $N_d$.
- **Bandwidth saturation:** Inter-node network bandwidth becomes the bottleneck.

At large $dp$, the all-reduce can **no longer be fully overlapped** with backward computation:

$$
T_{\text{all-reduce}} > T_{\text{backward}} \implies \text{GPUs idle during communication}
$$

**Empirically:** Throughput (tokens/sec/GPU) begins to drop significantly beyond a scaling threshold (often around $dp \approx 512$), while **memory usage per GPU remains constant** — DP does not reduce per-GPU memory.

### 7.2 Memory Limitation

Data parallelism requires **each GPU to hold the entire model** (parameters, gradients, optimizer states) plus activations for at least one micro-batch ($mbs = 1$). For large models:

$$
\text{Minimum memory} \approx 2\Psi \text{ bytes (parameters in BF16)}
$$

**Quick heuristic:** A model with $\Psi$ billion parameters requires approximately $2\Psi$ GB of memory just for parameters. For example, a 70B-parameter model needs $\approx 140$ GB $\approx 133$ GiB **for parameters alone**, before accounting for gradients, optimizer states, or activations. This often exceeds single-GPU capacity.

---

## 8. Zero Redundancy Optimizer (ZeRO)

### 8.1 Motivation

In vanilla DP, **every** GPU holds a full copy of:

- Parameters $\theta$
- Gradients $g$
- Optimizer states (e.g., Adam's first and second moments $m, v$, and FP32 master weights)

This is **massively redundant** — $N_d$ identical copies exist across the cluster.

**ZeRO** eliminates this redundancy by **partitioning** (sharding) these objects across the $N_d$ DP ranks, with each rank storing only a $\frac{1}{N_d}$ slice. Full tensors are reconstructed on-demand via communication when needed.

**Important:** Activations **cannot** be sharded by ZeRO because each DP replica processes a **different** micro-batch, so activations are already unique per rank (not duplicated).

### 8.2 Memory Usage Analysis (Baseline)

Let $\Psi$ denote the number of model parameters. In **mixed-precision training** (BF16/FP16 forward/backward, FP32 optimizer) with Adam:

| Component | Precision | Memory |
|---|---|---|
| Parameters | BF16 | $2\Psi$ bytes |
| Gradients | BF16 | $2\Psi$ bytes |
| FP32 master weights | FP32 | $4\Psi$ bytes |
| Adam first moment $m$ | FP32 | $4\Psi$ bytes |
| Adam second moment $v$ | FP32 | $4\Psi$ bytes |
| (Optional) FP32 gradients | FP32 | $4\Psi$ bytes |

**Without FP32 gradient accumulation:**

$$
M_{\text{total}} = \underbrace{2\Psi}_{\text{params}} + \underbrace{2\Psi}_{\text{grads}} + \underbrace{12\Psi}_{\text{optimizer states}} = 16\Psi \text{ bytes}
$$

**With FP32 gradient accumulation:**

$$
M_{\text{total}} = 2\Psi + \underbrace{(2\Psi + 4\Psi)}_{\text{BF16 + FP32 grads}} + 12\Psi = 20\Psi \text{ bytes}
$$

Define the **optimizer state memory multiplier** $k$:

$$
k = 12 \quad \text{(for Adam: } 4\Psi_{\text{FP32 weights}} + 4\Psi_{m} + 4\Psi_{v}\text{)}
$$

The baseline (without FP32 gradient accumulation) can be written as:

$$
\boxed{M_{\text{baseline}} = 2\Psi + 2\Psi + k\Psi}
$$

---

### 8.3 ZeRO-1: Optimizer State Partitioning

**What is partitioned:** Only the optimizer states (FP32 master weights, Adam $m$ and $v$).

**Mechanism:**

1. **Forward pass:** Each replica uses the full set of BF16 parameters $\theta$ on its micro-batch.
2. **Backward pass:** Each replica computes full gradients $g_i$.
3. **Reduce-Scatter on gradients:** Instead of all-reduce, perform a reduce-scatter. Each rank $i$ receives only the $\frac{1}{N_d}$ slice of the summed gradients corresponding to its optimizer state partition.
4. **Optimizer step (local):** Each rank updates only its $\frac{1}{N_d}$ slice of the FP32 parameters using its local optimizer states and gradient slice.
5. **All-Gather on BF16 parameters:** Each rank broadcasts its updated $\frac{1}{N_d}$ BF16 parameter slice; all ranks reconstruct the full parameter set.

**Memory per rank:**

$$
\boxed{M_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{k\Psi}{N_d}}
$$

| Component | Memory per rank |
|---|---|
| BF16 parameters (full) | $2\Psi$ |
| BF16 gradients (full) | $2\Psi$ |
| Optimizer states (sharded) | $\frac{k\Psi}{N_d} = \frac{12\Psi}{N_d}$ |

**Communication volume per rank (per training step):**

| Operation | Volume | When |
|---|---|---|
| Reduce-Scatter (gradients) | $\Psi$ | Backward pass |
| All-Gather (BF16 parameters) | $\Psi$ | After optimizer step |
| **Total** | $2\Psi$ | — |

This is **identical** to vanilla DP's all-reduce communication volume ($2\Psi$), since $\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$.

**Overlap strategies for the new all-gather:**

- **During the optimizer step:** Initiate all-gather as soon as the first parameter slice is updated; overlap with remaining optimizer computations.
- **During the forward pass:** Prefetch all-gathered parameters for the next layer while computing the forward pass of the current layer.

---

### 8.4 ZeRO-2: Optimizer State + Gradient Partitioning

**What is additionally partitioned:** Gradients are now also sharded across ranks.

**Key insight:** Since each rank only updates $\frac{1}{N_d}$ of the parameters (in its optimizer step), it only **needs** $\frac{1}{N_d}$ of the gradients. Storing the full gradient tensor on every rank is wasteful.

**Mechanism:**

1. **Forward pass:** Same as ZeRO-1 (full BF16 parameters on each replica).
2. **Backward pass + Reduce-Scatter:** As gradients are computed, a reduce-scatter is performed. Each rank retains only its $\frac{1}{N_d}$ slice of the reduced gradients; the remaining gradient memory is **immediately freed**.
3. **Optimizer step (local):** Each rank uses its gradient slice and optimizer state slice to update its $\frac{1}{N_d}$ parameter slice.
4. **All-Gather on BF16 parameters:** Same as ZeRO-1.

**Memory per rank:**

$$
\boxed{M_{\text{ZeRO-2}} = 2\Psi + \frac{(2 + k)\Psi}{N_d} = 2\Psi + \frac{2\Psi + k\Psi}{N_d}}
$$

| Component | Memory per rank |
|---|---|
| BF16 parameters (full) | $2\Psi$ |
| BF16 gradients (sharded) | $\frac{2\Psi}{N_d}$ |
| Optimizer states (sharded) | $\frac{k\Psi}{N_d}$ |

With FP32 gradient accumulation, add $\frac{4\Psi}{N_d}$ to the gradient term.

**As $N_d \to \infty$:**

$$
M_{\text{ZeRO-2}} \to 2\Psi
$$

This represents up to an **8× memory reduction** compared to the baseline $16\Psi$.

**Communication volume:** Identical to ZeRO-1 — one reduce-scatter ($\Psi$) and one all-gather ($\Psi$), totaling $2\Psi$. Therefore:

> **ZeRO-2 has no communication overhead compared to ZeRO-1 or vanilla DP, but provides strictly greater memory savings.** ZeRO-2 dominates ZeRO-1.

---

### 8.5 ZeRO-3: Full Partitioning — Parameters + Gradients + Optimizer States (FSDP)

**What is additionally partitioned:** Model parameters themselves are sharded across ranks.

> **PyTorch's native implementation of ZeRO-3 is called FSDP (Fully Sharded Data Parallelism).**

**Mechanism — Forward Pass:**

For each layer $l = 1, 2, \ldots, L$:

1. **All-Gather** the full parameters of layer $l$ from all $N_d$ ranks (each rank contributes its $\frac{1}{N_d}$ shard).
2. **Compute** the forward pass through layer $l$.
3. **Discard** (free) the full parameters of layer $l$ from GPU memory — only the local $\frac{1}{N_d}$ shard is retained.

**Mechanism — Backward Pass:**

For each layer $l = L, L-1, \ldots, 1$:

1. **All-Gather** the full parameters of layer $l$ (needed again since they were discarded after the forward pass).
2. **Compute** the backward pass through layer $l$ to produce gradients.
3. **Reduce-Scatter** the gradients of layer $l$ — each rank keeps only its $\frac{1}{N_d}$ gradient shard.
4. **Discard** the full parameters and non-local gradient portions.

After all layers:

4. **Optimizer step (local):** Each rank updates its $\frac{1}{N_d}$ parameter shard using its gradient and optimizer state shards.

**Memory per rank:**

$$
\boxed{M_{\text{ZeRO-3}} = \frac{(2 + 2 + k)\Psi}{N_d} = \frac{(4 + k)\Psi}{N_d}}
$$

| Component | Memory per rank |
|---|---|
| BF16 parameters (sharded) | $\frac{2\Psi}{N_d}$ |
| BF16 gradients (sharded) | $\frac{2\Psi}{N_d}$ |
| Optimizer states (sharded) | $\frac{k\Psi}{N_d}$ |

**As $N_d \to \infty$, $M_{\text{ZeRO-3}} \to 0$ for model-related memory.** However, activation memory is **not** reduced by ZeRO (activation checkpointing and gradient accumulation address that separately).

**Communication Cost Analysis:**

| Operation | Volume per rank | Occurrences |
|---|---|---|
| All-Gather (forward pass, parameters) | $\Psi$ | Once over all layers |
| All-Gather (backward pass, parameters) | $\Psi$ | Once over all layers |
| Reduce-Scatter (backward pass, gradients) | $\Psi$ | Once over all layers |
| **Total** | $3\Psi$ | — |

Compared to ZeRO-2's $2\Psi$, ZeRO-3 incurs an **additional $\Psi$ communication cost** (the extra all-gather during forward pass).

**Prefetching for Overlap:**

The additional all-gathers can be **overlapped** with computation via prefetching:

- **Forward pass:** While computing layer $n$, initiate the all-gather for layer $n+1$'s parameters.
- **Backward pass:** While computing layer $n$, initiate the all-gather for layer $n-1$'s parameters.

This overlap is effective as long as computation time per layer exceeds the all-gather latency. The rule of thumb is that this works well for $dp \leq 512$.

---

## 9. Comparative Summary

### 9.1 Memory Comparison Table

| Method | Parameters | Gradients | Optimizer States | Total Memory per Rank |
|---|---|---|---|---|
| **Vanilla DP** | $2\Psi$ | $2\Psi$ | $k\Psi$ | $2\Psi + 2\Psi + k\Psi$ |
| **ZeRO-1** | $2\Psi$ | $2\Psi$ | $\frac{k\Psi}{N_d}$ | $2\Psi + 2\Psi + \frac{k\Psi}{N_d}$ |
| **ZeRO-2** | $2\Psi$ | $\frac{2\Psi}{N_d}$ | $\frac{k\Psi}{N_d}$ | $2\Psi + \frac{2\Psi + k\Psi}{N_d}$ |
| **ZeRO-3** | $\frac{2\Psi}{N_d}$ | $\frac{2\Psi}{N_d}$ | $\frac{k\Psi}{N_d}$ | $\frac{2\Psi + 2\Psi + k\Psi}{N_d}$ |

With $k = 12$ (Adam) and ignoring FP32 gradient accumulation:

| Method | Total (bytes) | With $N_d = 64$ |
|---|---|---|
| **Vanilla DP** | $16\Psi$ | $16\Psi$ |
| **ZeRO-1** | $4\Psi + \frac{12\Psi}{N_d}$ | $4.1875\Psi$ |
| **ZeRO-2** | $2\Psi + \frac{14\Psi}{N_d}$ | $2.21875\Psi$ |
| **ZeRO-3** | $\frac{16\Psi}{N_d}$ | $0.25\Psi$ |

### 9.2 Communication Comparison Table

| Method | Communication Volume per Rank per Step |
|---|---|
| **Vanilla DP** (All-Reduce) | $2\Psi$ |
| **ZeRO-1** (Reduce-Scatter + All-Gather) | $2\Psi$ |
| **ZeRO-2** (Reduce-Scatter + All-Gather) | $2\Psi$ |
| **ZeRO-3** (2× All-Gather + Reduce-Scatter) | $3\Psi$ |

> ZeRO-1 and ZeRO-2 are **communication-equivalent** to vanilla DP. ZeRO-3 incurs a 50% communication increase ($3\Psi$ vs. $2\Psi$), which is mitigated by prefetching.

---

## 10. Fundamental Limitations and Transition to Other Parallelism Axes

### 10.1 Limitations of Data Parallelism + ZeRO

| Limitation | Explanation |
|---|---|
| **Layer must fit in a single GPU** | Even ZeRO-3 must reconstruct the full parameters of a layer on a single GPU for the forward/backward computation. If a single layer exceeds GPU memory, DP/ZeRO alone is insufficient. |
| **Activation memory is not sharded** | Activations depend on micro-batch content (unique per rank) and thus cannot be deduplicated. Activation memory scales as $\mathcal{O}(mbs \times S \times h \times L)$ where $S$ is sequence length, $h$ is hidden dimension, and $L$ is the number of layers. Only activation checkpointing and gradient accumulation (reducing $mbs$) address this. |
| **Communication overhead at large $N_d$** | Ring latency and bandwidth saturation degrade compute efficiency as $N_d$ grows beyond ~512. The overlap of communication with computation breaks down. |
| **Throughput ceiling** | Adding more DP ranks eventually yields diminishing returns — throughput per GPU drops while aggregate throughput plateaus. |

### 10.2 Transition to Tensor Parallelism

To overcome these limitations, an **orthogonal** parallelism axis is needed — **Tensor Parallelism (TP)**. Unlike ZeRO-3 (which communicates full parameters on demand), TP **partitions parameters, gradients, optimizer states, AND activations** across devices. No GPU ever needs the full parameter set of a layer; instead, each GPU computes a **shard of the layer's output** using a **shard of the layer's weights**, with only activation-sized communication between GPUs.

This makes TP complementary to DP: DP scales across micro-batches (the data dimension), while TP scales within each layer (the model dimension). They can be composed, leading to **multi-dimensional parallelism** (2D, 3D, and beyond).





# Data Parallelism: Communication Primitives — Implementation in PyTorch & Triton

---

## 1. Distributed Communication Primitives: Mathematical Definitions

Before any code, we rigorously define each primitive. Let $N_d$ denote the number of participating ranks (GPUs). Each rank $i \in \{0, 1, \ldots, N_d - 1\}$ holds a tensor $x_i \in \mathbb{R}^{D}$.

---

### 1.1 Broadcast

One designated root rank $r$ sends its tensor to all other ranks:

$$
\forall\, i \in \{0, \ldots, N_d - 1\}: \quad x_i \leftarrow x_r
$$

---

### 1.2 Reduce

All ranks contribute tensors; the element-wise reduction (e.g., sum) is deposited on a single destination rank $r$:

$$
x_r \leftarrow \bigoplus_{i=0}^{N_d - 1} x_i = \sum_{i=0}^{N_d - 1} x_i
$$

All other ranks' tensors are unchanged.

---

### 1.3 All-Reduce

Every rank contributes a tensor; the element-wise reduction is computed and the result is placed on **all** ranks:

$$
\forall\, i: \quad x_i \leftarrow \sum_{j=0}^{N_d - 1} x_j
$$

**Ring All-Reduce complexity** for a tensor of size $D$:

$$
T_{\text{all-reduce}} = 2(N_d - 1) \cdot \alpha + 2 \cdot \frac{N_d - 1}{N_d} \cdot \frac{D}{\beta}
$$

where $\alpha$ is the per-message latency and $\beta$ is the per-link bandwidth.

---

### 1.4 Reduce-Scatter

All ranks contribute tensors; the element-wise reduction is computed, then the result is **scattered** — rank $i$ receives the $i$-th chunk of the reduced tensor:

$$
x_i^{(\text{chunk})} \leftarrow \left(\sum_{j=0}^{N_d - 1} x_j\right)\Bigg[\frac{iD}{N_d} : \frac{(i+1)D}{N_d}\Bigg]
$$

**Communication volume per rank:**

$$
V_{\text{reduce-scatter}} = \frac{N_d - 1}{N_d} \cdot D \approx D \quad (\text{for large } N_d)
$$

---

### 1.5 All-Gather

Each rank holds a chunk $x_i^{(\text{chunk})} \in \mathbb{R}^{D/N_d}$; all chunks are gathered so every rank holds the full tensor:

$$
\forall\, i: \quad x_i \leftarrow \text{concat}\!\Big(x_0^{(\text{chunk})}, x_1^{(\text{chunk})}, \ldots, x_{N_d-1}^{(\text{chunk})}\Big) \in \mathbb{R}^D
$$

**Fundamental decomposition identity:**

$$
\boxed{\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}}
$$

This identity is the algorithmic basis for ZeRO optimizations.

---

### 1.6 Scatter

Root rank $r$ partitions its tensor into $N_d$ chunks and sends chunk $i$ to rank $i$:

$$
x_i \leftarrow x_r\!\left[\frac{iD}{N_d} : \frac{(i+1)D}{N_d}\right]
$$

---

### 1.7 Gather

Each rank sends its tensor to a designated root rank $r$, which concatenates them:

$$
x_r \leftarrow \text{concat}(x_0, x_1, \ldots, x_{N_d-1})
$$

---

### 1.8 Summary Table

| Primitive | Input per rank | Output per rank | Comm. Volume per rank |
|---|---|---|---|
| **Broadcast** | $x_r$ (root only) | $x_r$ (all) | $D$ |
| **Reduce** | $x_i$ (all) | $\sum x_i$ (root only) | $\frac{N_d - 1}{N_d} D$ |
| **All-Reduce** | $x_i$ (all) | $\sum x_i$ (all) | $2\frac{N_d-1}{N_d}D$ |
| **Reduce-Scatter** | $x_i$ (all) | chunk of $\sum x_i$ | $\frac{N_d-1}{N_d}D$ |
| **All-Gather** | chunk $x_i^{(c)}$ | full concat | $\frac{N_d-1}{N_d}D$ |
| **Scatter** | full (root) | chunk | $\frac{N_d-1}{N_d}D$ |
| **Gather** | $x_i$ (all) | concat (root) | $\frac{N_d-1}{N_d}D$ |

---

## 2. PyTorch Distributed: Complete Implementations

### 2.1 Environment Setup and Process Group Initialization

```python
import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

def init_distributed():
    """
    Initialize the distributed process group.
    Expects environment variables: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    to be set (e.g., by torchrun).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the device for this rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group with NCCL backend (optimal for GPU-GPU comm)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size, device


def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()
```

**Launch command:**

```bash
torchrun --nproc_per_node=4 --nnodes=1 script.py
```

---

### 2.2 Broadcast

```python
def broadcast_example(rank: int, world_size: int, device: torch.device):
    """
    Root rank 0 broadcasts a tensor to all other ranks.
    
    Mathematically:
        ∀ i ∈ {0, ..., N_d - 1}:  x_i ← x_0
    """
    if rank == 0:
        tensor = torch.arange(8, dtype=torch.float32, device=device)
    else:
        tensor = torch.zeros(8, dtype=torch.float32, device=device)

    print(f"[Rank {rank}] Before broadcast: {tensor}")

    # In-place broadcast from src=0
    dist.broadcast(tensor, src=0)

    print(f"[Rank {rank}] After broadcast:  {tensor}")
    # All ranks now hold [0, 1, 2, 3, 4, 5, 6, 7]
    return tensor
```

---

### 2.3 Reduce

```python
def reduce_example(rank: int, world_size: int, device: torch.device):
    """
    All ranks contribute tensors; element-wise sum is deposited on dst=0.

    Mathematically:
        x_0 ← Σ_{i=0}^{N_d - 1} x_i
    """
    tensor = torch.ones(4, dtype=torch.float32, device=device) * (rank + 1)
    print(f"[Rank {rank}] Before reduce: {tensor}")

    dist.reduce(tensor, dst=0, op=ReduceOp.SUM)

    if rank == 0:
        # tensor now contains sum: [1+2+...+N_d] for each element
        print(f"[Rank {rank}] After reduce (SUM): {tensor}")
    return tensor
```

---

### 2.4 All-Reduce

```python
def all_reduce_example(rank: int, world_size: int, device: torch.device):
    """
    All ranks contribute tensors; element-wise sum is placed on ALL ranks.

    Mathematically:
        ∀ i:  x_i ← Σ_{j=0}^{N_d - 1} x_j
    
    Communication cost (Ring algorithm):
        T = 2(N_d - 1)α + 2·(N_d - 1)/N_d · D/β
    """
    # Each rank has its own gradient-like tensor
    gradient = torch.randn(1024, dtype=torch.float32, device=device)
    gradient_copy = gradient.clone()

    print(f"[Rank {rank}] Before all-reduce, sum of local grad: {gradient.sum():.4f}")

    # In-place all-reduce: sum across all ranks
    dist.all_reduce(gradient, op=ReduceOp.SUM)

    print(f"[Rank {rank}] After all-reduce, sum of global grad: {gradient.sum():.4f}")

    # To compute the AVERAGE (standard in data parallelism):
    gradient_avg = gradient_copy.clone()
    dist.all_reduce(gradient_avg, op=ReduceOp.SUM)
    gradient_avg /= world_size
    # Equivalent shorthand:
    # dist.all_reduce(gradient_avg, op=ReduceOp.AVG)  # PyTorch >= 1.12

    return gradient_avg
```

---

### 2.5 Reduce-Scatter

```python
def reduce_scatter_example(rank: int, world_size: int, device: torch.device):
    """
    All ranks contribute tensors; element-wise sum is computed, 
    then the result is scattered — rank i receives the i-th chunk.

    Mathematically:
        x_i^(chunk) ← (Σ_{j} x_j)[iD/N_d : (i+1)D/N_d]

    This is the FIRST HALF of all-reduce and is critical for ZeRO-1/2/3.
    Communication volume per rank: (N_d - 1)/N_d · D
    """
    D = 1024  # Total tensor size (must be divisible by world_size)
    chunk_size = D // world_size

    # Each rank has a full gradient tensor
    full_gradient = torch.randn(D, dtype=torch.float32, device=device)
    print(f"[Rank {rank}] Full gradient shape: {full_gradient.shape}")

    # Output: each rank only stores its chunk
    output_chunk = torch.zeros(chunk_size, dtype=torch.float32, device=device)

    # Prepare input as list of chunks (one per rank)
    input_list = list(full_gradient.chunk(world_size))

    dist.reduce_scatter(output_chunk, input_list, op=ReduceOp.SUM)

    print(f"[Rank {rank}] Received chunk shape: {output_chunk.shape}")
    # output_chunk contains the reduced i-th chunk of the global sum
    return output_chunk
```

**Tensor variant (more memory-efficient, avoids list allocation):**

```python
def reduce_scatter_tensor_example(rank: int, world_size: int, device: torch.device):
    """
    reduce_scatter_tensor operates on contiguous tensors directly.
    More efficient than the list-based API.
    """
    D = 1024
    chunk_size = D // world_size

    full_gradient = torch.randn(D, dtype=torch.float32, device=device)
    output_chunk = torch.zeros(chunk_size, dtype=torch.float32, device=device)

    # Single contiguous input tensor (preferred for NCCL)
    dist.reduce_scatter_tensor(output_chunk, full_gradient, op=ReduceOp.SUM)

    return output_chunk
```

---

### 2.6 All-Gather

```python
def all_gather_example(rank: int, world_size: int, device: torch.device):
    """
    Each rank holds a chunk; all chunks are gathered so every rank 
    holds the full tensor.

    Mathematically:
        ∀ i: x_i ← concat(x_0^(c), x_1^(c), ..., x_{N_d-1}^(c))
    
    This is the SECOND HALF of all-reduce and is critical for 
    ZeRO-1/2/3 parameter reconstruction.
    """
    chunk_size = 256
    local_chunk = torch.randn(chunk_size, dtype=torch.float32, device=device)

    # List-based API: prepare output list
    gathered_list = [
        torch.zeros(chunk_size, dtype=torch.float32, device=device)
        for _ in range(world_size)
    ]

    dist.all_gather(gathered_list, local_chunk)

    full_tensor = torch.cat(gathered_list, dim=0)
    print(f"[Rank {rank}] Gathered full tensor shape: {full_tensor.shape}")
    return full_tensor


def all_gather_tensor_example(rank: int, world_size: int, device: torch.device):
    """
    Tensor-based all-gather: more memory-efficient, single contiguous output.
    """
    chunk_size = 256
    local_chunk = torch.randn(chunk_size, dtype=torch.float32, device=device)
    full_tensor = torch.zeros(
        chunk_size * world_size, dtype=torch.float32, device=device
    )

    dist.all_gather_into_tensor(full_tensor, local_chunk)

    return full_tensor
```

---

### 2.7 Scatter and Gather

```python
def scatter_example(rank: int, world_size: int, device: torch.device):
    """
    Root rank partitions its tensor and sends chunk i to rank i.
    
    Mathematically:
        x_i ← x_r[iD/N_d : (i+1)D/N_d]
    """
    chunk_size = 128
    output = torch.zeros(chunk_size, dtype=torch.float32, device=device)

    if rank == 0:
        # Root prepares chunks for each rank
        scatter_list = [
            torch.ones(chunk_size, dtype=torch.float32, device=device) * i
            for i in range(world_size)
        ]
    else:
        scatter_list = None

    dist.scatter(output, scatter_list=scatter_list, src=0)
    print(f"[Rank {rank}] Received: {output[0].item()}")
    return output


def gather_example(rank: int, world_size: int, device: torch.device):
    """
    Each rank sends its tensor to root rank 0, which concatenates them.
    
    Mathematically:
        x_0 ← concat(x_0, x_1, ..., x_{N_d-1})
    """
    local_tensor = torch.ones(64, dtype=torch.float32, device=device) * rank

    if rank == 0:
        gather_list = [
            torch.zeros(64, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]
    else:
        gather_list = None

    dist.gather(local_tensor, gather_list=gather_list, dst=0)

    if rank == 0:
        full = torch.cat(gather_list, dim=0)
        print(f"[Rank 0] Gathered: {full.shape}")
    return gather_list if rank == 0 else None
```

---

## 3. Complete Data Parallelism Implementations

### 3.1 Naive Data Parallelism (No Overlap)

```python
import torch
import torch.nn as nn
import torch.distributed as dist


class NaiveDataParallel:
    """
    Naive DP: Sequential backward → all-reduce → optimizer step.
    
    T_total = T_fwd + T_bwd + T_all_reduce + T_opt
    
    GPU is IDLE during T_all_reduce. This is suboptimal.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.world_size = dist.get_world_size()

    def sync_gradients(self):
        """
        All-reduce all gradients: ∀ param p:
            p.grad ← (1/N_d) Σ_{i=0}^{N_d-1} p.grad_i
        """
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size

    def train_step(self, batch, loss_fn, optimizer):
        # Forward pass
        output = self.model(batch["input"])
        loss = loss_fn(output, batch["target"])

        # Backward pass (compute local gradients)
        loss.backward()

        # *** IDLE: waiting for communication ***
        self.sync_gradients()

        # Optimizer step (identical on all ranks)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

---

### 3.2 Overlapped Data Parallelism (Hook-Based)

```python
class OverlappedDataParallel:
    """
    Optimization 1: Overlap gradient all-reduce with backward pass.
    
    As soon as a parameter's gradient is computed, the all-reduce 
    is triggered immediately via a hook, while backward continues 
    for earlier layers.
    
    T_total = T_fwd + max(T_bwd, T_all_reduce) + T_opt
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.world_size = dist.get_world_size()
        self._handles = []
        self._register_hooks()

    def _all_reduce_hook(self, param: torch.Tensor):
        """
        Post-accumulate gradient hook: triggered immediately 
        when param.grad is ready.
        """
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((handle, param))

    def _register_hooks(self):
        """Attach hooks to all trainable parameters."""
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._all_reduce_hook)

    def _finalize_gradients(self):
        """Wait for all async all-reduce operations to complete."""
        for handle, param in self._handles:
            handle.wait()
            param.grad /= self.world_size
        self._handles.clear()

    def train_step(self, batch, loss_fn, optimizer):
        output = self.model(batch["input"])
        loss = loss_fn(output, batch["target"])

        # Backward pass: hooks fire all-reduce as gradients become available
        loss.backward()

        # Wait for any remaining async operations
        self._finalize_gradients()

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

---

### 3.3 Bucketed Data Parallelism

```python
class BucketedDataParallel:
    """
    Optimization 2: Group gradients into buckets before all-reduce.
    
    Instead of N_params individual all-reduces, perform 
    ceil(total_param_size / bucket_size_bytes) bucket-level all-reduces.
    
    Benefits:
        - Higher bandwidth utilization (large contiguous transfers)
        - Reduced per-operation launch latency overhead
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        bucket_size_mb: float = 25.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.world_size = dist.get_world_size()
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        # Build buckets: group parameters into contiguous buffers
        self.buckets = self._build_buckets()
        self._handles = []
        self._register_hooks()

    def _build_buckets(self):
        """
        Group parameters into buckets of approximately bucket_size_bytes.
        Parameters are iterated in REVERSE order (matching backward pass order).
        """
        buckets = []
        current_bucket = []
        current_size = 0

        # Reverse order: last layer's params first (they finish backward first)
        for param in reversed(list(self.model.parameters())):
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()
            if current_size + param_size > self.bucket_size_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_size

        if current_bucket:
            buckets.append(current_bucket)

        # Create contiguous gradient buffers for each bucket
        bucket_info = []
        for bucket_params in buckets:
            total_numel = sum(p.numel() for p in bucket_params)
            buffer = torch.zeros(
                total_numel, dtype=bucket_params[0].dtype, device=self.device
            )
            bucket_info.append({
                "params": bucket_params,
                "buffer": buffer,
                "num_ready": 0,
                "num_params": len(bucket_params),
            })

        return bucket_info

    def _make_hook(self, bucket_idx: int):
        """Create a hook closure for the given bucket."""
        def hook(param: torch.Tensor):
            bucket = self.buckets[bucket_idx]
            bucket["num_ready"] += 1

            if bucket["num_ready"] == bucket["num_params"]:
                # All gradients in this bucket are ready
                # Pack gradients into contiguous buffer
                offset = 0
                for p in bucket["params"]:
                    numel = p.numel()
                    bucket["buffer"][offset:offset + numel].copy_(p.grad.flatten())
                    offset += numel

                # Launch single all-reduce for entire bucket
                handle = dist.all_reduce(
                    bucket["buffer"], op=dist.ReduceOp.SUM, async_op=True
                )
                self._handles.append((handle, bucket))
        return hook

    def _register_hooks(self):
        """Map each parameter to its bucket and attach hook."""
        param_to_bucket = {}
        for idx, bucket in enumerate(self.buckets):
            for p in bucket["params"]:
                param_to_bucket[p] = idx

        for p in self.model.parameters():
            if p.requires_grad and p in param_to_bucket:
                bucket_idx = param_to_bucket[p]
                p.register_post_accumulate_grad_hook(self._make_hook(bucket_idx))

    def _finalize_and_unpack(self):
        """Wait for all bucket all-reduces and unpack back to param.grad."""
        for handle, bucket in self._handles:
            handle.wait()
            bucket["buffer"] /= self.world_size
            # Unpack from contiguous buffer back to individual param.grad
            offset = 0
            for p in bucket["params"]:
                numel = p.numel()
                p.grad.copy_(
                    bucket["buffer"][offset:offset + numel].view_as(p.grad)
                )
                offset += numel
            bucket["num_ready"] = 0  # Reset for next step

        self._handles.clear()

    def train_step(self, batch, loss_fn, optimizer):
        output = self.model(batch["input"])
        loss = loss_fn(output, batch["target"])
        loss.backward()
        self._finalize_and_unpack()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

---

### 3.4 Gradient Accumulation with DP (no_sync)

```python
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP


def train_step_with_grad_accum(
    model: DDP,
    batches: list,       # List of G micro-batches
    loss_fn,
    optimizer,
    grad_acc_steps: int,  # G
):
    """
    Optimization 3: Disable gradient sync for first G-1 steps.
    Only the final backward pass triggers all-reduce.
    
    Communication cost: 1 × T_all_reduce  (instead of G × T_all_reduce)
    
    Global batch size:
        gbs = mbs × grad_acc × dp
    """
    optimizer.zero_grad()
    total_loss = 0.0

    for step_idx, micro_batch in enumerate(batches):
        is_last_step = (step_idx == grad_acc_steps - 1)

        # Disable gradient sync for all but the last accumulation step
        context = (
            contextlib.nullcontext()
            if is_last_step
            else model.no_sync()
        )

        with context:
            output = model(micro_batch["input"])
            # Scale loss by accumulation steps to maintain correct gradient magnitude:
            # ∇L_total = (1/G) Σ_{j=1}^{G} ∇L_j
            loss = loss_fn(output, micro_batch["target"]) / grad_acc_steps
            loss.backward()
            total_loss += loss.item()

    # At this point, gradients are accumulated AND synchronized (from last step)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss * grad_acc_steps  # Undo the scaling for logging
```

---

### 3.5 PyTorch DistributedDataParallel (DDP) — Production Best Practice

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def production_ddp_training(rank, world_size):
    """
    Production-grade DDP training loop incorporating all three optimizations:
    1. Overlapped gradient sync (built into DDP)
    2. Bucketing (built into DDP, configurable via bucket_cap_mb)
    3. Gradient accumulation with no_sync
    """
    # --- Setup ---
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # --- Model ---
    model = nn.TransformerEncoderLayer(
        d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True
    ).to(device)

    # Wrap with DDP:
    #   - bucket_cap_mb: controls bucket size (Optimization 2)
    #   - gradient_as_bucket_view: avoids extra gradient copies
    #   - static_graph: enables advanced comm optimizations if graph doesn't change
    ddp_model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        bucket_cap_mb=25,                # Bucket size in MB
        gradient_as_bucket_view=True,     # Memory optimization
        static_graph=False,               # Set True if computation graph is fixed
    )

    # --- Data ---
    dataset = ...  # Your dataset
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,  # Ensures equal batch sizes across ranks
    )
    dataloader = DataLoader(
        dataset, batch_size=2, sampler=sampler, pin_memory=True, num_workers=4
    )

    # --- Training config ---
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    grad_acc_steps = 4  # G

    # --- Global batch size ---
    mbs = 2
    dp = world_size
    gbs = mbs * grad_acc_steps * dp
    if rank == 0:
        print(f"Global batch size: {gbs} samples = {gbs * 4096} tokens (seq_len=4096)")

    # --- Training loop ---
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Critical for proper shuffling
        micro_batch_buffer = []

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            micro_batch_buffer.append(batch)

            if len(micro_batch_buffer) == grad_acc_steps:
                # Execute one full training step with gradient accumulation
                train_step_with_grad_accum(
                    ddp_model, micro_batch_buffer, loss_fn, optimizer, grad_acc_steps
                )
                micro_batch_buffer = []

    cleanup()
```

---

### 3.6 FSDP (ZeRO-3) — Production Implementation

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools


def production_fsdp_training(rank, world_size):
    """
    ZeRO-3 / FSDP: Parameters, gradients, and optimizer states 
    are all sharded across N_d ranks.
    
    Memory per rank: (2Ψ + 2Ψ + kΨ) / N_d
    Communication per step: 3Ψ (2× all-gather + 1× reduce-scatter)
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # --- Model ---
    # Build model on meta device first (avoids OOM during init for large models)
    with torch.device("meta"):
        model = build_large_transformer_model()  # Your model factory

    # --- FSDP Wrapping Policy ---
    # Wrap each TransformerEncoderLayer as a separate FSDP unit
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )

    # --- Mixed Precision ---
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,      # BF16 for forward/backward
        reduce_dtype=torch.bfloat16,      # BF16 for gradient reduction
        buffer_dtype=torch.bfloat16,
    )

    # --- FSDP Configuration ---
    # ShardingStrategy options:
    #   FULL_SHARD  = ZeRO-3 (shard params + grads + optimizer states)
    #   SHARD_GRAD_OP = ZeRO-2 (shard grads + optimizer states)
    #   NO_SHARD    = DDP equivalent
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=rank,
        use_orig_params=True,        # Needed for torch.compile compatibility
        forward_prefetch=True,        # Prefetch layer n+1 during forward of layer n
        backward_prefetch_limit=1,    # Prefetch during backward pass
        limit_all_gathers=True,       # Prevents excessive memory from queued all-gathers
    )

    # --- Optimizer (after FSDP wrapping) ---
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    # --- Training loop with gradient accumulation ---
    grad_acc_steps = 4
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            is_accumulating = ((step + 1) % grad_acc_steps != 0)

            # no_sync for accumulation steps (same pattern as DDP)
            context = fsdp_model.no_sync() if is_accumulating else contextlib.nullcontext()

            with context:
                output = fsdp_model(batch["input"])
                loss = loss_fn(output, batch["target"]) / grad_acc_steps
                loss.backward()

            if not is_accumulating:
                # Gradient clipping (works with FSDP)
                fsdp_model.clip_grad_norm_(max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

    cleanup()
```

---

## 4. Triton Kernels for Communication-Adjacent Operations

Triton operates at the **single-GPU kernel level** — it does not directly implement inter-GPU communication (that is NCCL's domain). However, Triton is invaluable for the **computation kernels** that run alongside communication, such as gradient packing/unpacking, buffer operations, and fused reduction operations within a single GPU.

### 4.1 Fused Gradient Packing into Contiguous Bucket Buffer

```python
import triton
import triton.language as tl


@triton.jit
def _pack_gradients_kernel(
    src_ptr,        # Pointer to source gradient tensor (non-contiguous possible)
    dst_ptr,        # Pointer to destination contiguous buffer
    src_stride,     # Stride of source
    N: tl.constexpr,  # Number of elements to copy
    BLOCK_SIZE: tl.constexpr,
):
    """
    Packs a gradient tensor into a contiguous communication buffer.
    
    This is the operation that happens BEFORE an all-reduce/reduce-scatter:
    gradients must be contiguous in memory for efficient NCCL operations.
    
    Each Triton program instance handles BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load from (potentially non-contiguous) source
    vals = tl.load(src_ptr + offsets * src_stride, mask=mask, other=0.0)
    # Store into contiguous destination buffer
    tl.store(dst_ptr + offsets, vals, mask=mask)


def pack_gradients(src: torch.Tensor, dst: torch.Tensor):
    """
    Pack gradient tensor into contiguous buffer for NCCL communication.
    
    Args:
        src: gradient tensor (any shape)
        dst: pre-allocated contiguous buffer (flat)
    """
    N = src.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _pack_gradients_kernel[grid](
        src.data_ptr(),
        dst.data_ptr(),
        src.stride(0) if src.dim() > 0 else 1,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

---

### 4.2 Fused Gradient Averaging (Post All-Reduce)

```python
@triton.jit
def _average_gradients_kernel(
    grad_ptr,
    N: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    After all-reduce (SUM), divide by N_d to compute gradient average.
    
    Mathematically:
        ḡ = (1/N_d) Σ_{i=0}^{N_d-1} g_i
    
    This is a simple element-wise division that can be fused with 
    other post-processing (e.g., gradient clipping).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    grad = grad / world_size
    tl.store(grad_ptr + offsets, grad, mask=mask)


def average_gradients_inplace(grad: torch.Tensor, world_size: int):
    """Divide gradient by world_size in-place using fused Triton kernel."""
    N = grad.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _average_gradients_kernel[grid](
        grad.data_ptr(), N, world_size, BLOCK_SIZE=BLOCK_SIZE
    )
```

---

### 4.3 Fused Gradient Scaling + Clipping (Pre-Communication)

```python
@triton.jit
def _scale_and_clip_kernel(
    grad_ptr,
    scale: tl.constexpr,    # 1.0 / grad_acc_steps
    max_norm_sq,             # Pointer to pre-computed squared norm
    clip_coeff,              # Pointer to clip coefficient
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: scale gradient by 1/G (for gradient accumulation) 
    AND apply gradient clipping, before communication.
    
    Gradient clipping:
        if ||g||_2 > max_norm:
            g ← g × (max_norm / ||g||_2)
    
    Combined with scaling:
        g ← g × scale × clip_coeff
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    coeff = tl.load(clip_coeff)
    grad = grad * scale * coeff
    tl.store(grad_ptr + offsets, grad, mask=mask)


def fused_scale_and_clip(
    grad: torch.Tensor,
    grad_acc_steps: int,
    max_norm: float,
    grad_norm: torch.Tensor,  # Pre-computed ||g||_2
):
    """
    Apply gradient accumulation scaling and gradient clipping in one pass.
    
    This reduces memory traffic: instead of two passes over the gradient 
    tensor, we do one.
    """
    N = grad.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Compute clip coefficient: min(1.0, max_norm / ||g||_2)
    clip_coeff = torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0)

    _scale_and_clip_kernel[grid](
        grad.data_ptr(),
        1.0 / grad_acc_steps,
        (grad_norm ** 2).data_ptr(),
        clip_coeff.data_ptr(),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

---

### 4.4 Fused Multi-Tensor Gradient Norm Computation

```python
@triton.jit
def _partial_squared_norm_kernel(
    input_ptr,
    output_ptr,     # Partial sum output (one per program)
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute partial squared L2 norm of a gradient tensor.
    
    ||g||_2^2 = Σ_i g_i^2
    
    Used before all-reduce of norms across DP ranks for global gradient clipping:
        ||g_global||_2^2 = Σ_{rank} ||g_local||_2^2
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(vals * vals)
    tl.store(output_ptr + pid, partial_sum)


def compute_grad_norm_squared(grad: torch.Tensor) -> torch.Tensor:
    """
    Compute ||grad||_2^2 using Triton.
    
    In distributed setting, follow with:
        dist.all_reduce(norm_sq, op=ReduceOp.SUM)
        total_norm = norm_sq.sqrt()
    """
    N = grad.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=grad.device)

    _partial_squared_norm_kernel[grid](
        grad.data_ptr(),
        partial_sums.data_ptr(),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return partial_sums.sum()  # Final reduction on GPU
```

---

### 4.5 Fused BF16 ↔ FP32 Conversion (ZeRO Optimizer Step)

```python
@triton.jit
def _bf16_to_fp32_kernel(
    src_ptr,    # BF16 source
    dst_ptr,    # FP32 destination
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Convert BF16 parameters to FP32 for optimizer step.
    
    In ZeRO, each rank converts only its 1/N_d shard:
        θ_fp32[shard_i] ← cast_fp32(θ_bf16[shard_i])
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    vals_fp32 = vals.to(tl.float32)
    tl.store(dst_ptr + offsets, vals_fp32, mask=mask)


@triton.jit
def _fp32_to_bf16_kernel(
    src_ptr,    # FP32 source (updated by optimizer)
    dst_ptr,    # BF16 destination (for forward pass)
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Convert FP32 updated parameters back to BF16 for all-gather.
    
    After optimizer step on local shard:
        θ_bf16[shard_i] ← cast_bf16(θ_fp32[shard_i])
    Then all-gather to reconstruct full θ_bf16.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(dst_ptr + offsets, vals_bf16, mask=mask)


def cast_bf16_to_fp32(src: torch.Tensor, dst: torch.Tensor):
    """BF16 → FP32 for optimizer step (per-shard in ZeRO)."""
    N = src.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _bf16_to_fp32_kernel[grid](src.data_ptr(), dst.data_ptr(), N, BLOCK_SIZE=BLOCK_SIZE)


def cast_fp32_to_bf16(src: torch.Tensor, dst: torch.Tensor):
    """FP32 → BF16 after optimizer step, before all-gather."""
    N = src.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fp32_to_bf16_kernel[grid](src.data_ptr(), dst.data_ptr(), N, BLOCK_SIZE=BLOCK_SIZE)
```

---

## 5. Ring All-Reduce: Algorithmic Implementation

The **Ring All-Reduce** algorithm is the most common implementation of the all-reduce primitive. It proceeds in two phases:

### 5.1 Phase 1: Reduce-Scatter (Ring Reduce)

Each rank sends its chunk to the next rank in the ring, accumulating partial sums. After $N_d - 1$ steps, each rank holds the fully reduced version of one chunk.

### 5.2 Phase 2: All-Gather (Ring Broadcast)

Each rank sends its fully reduced chunk to the next rank in the ring. After $N_d - 1$ steps, every rank holds all fully reduced chunks.

```python
def ring_all_reduce_simulation(
    local_tensor: torch.Tensor,
    rank: int,
    world_size: int,
):
    """
    Simulates Ring All-Reduce on a single device for educational purposes.
    
    Total communication volume per rank:
        V = 2 · (N_d - 1)/N_d · D
    
    Total latency:
        T = 2(N_d - 1) · α + 2 · (N_d - 1)/N_d · D/β
    
    This is OPTIMAL: it matches the bandwidth lower bound.
    """
    N_d = world_size
    D = local_tensor.numel()
    chunk_size = D // N_d

    # Split tensor into N_d chunks
    chunks = list(local_tensor.chunk(N_d))

    # ========================
    # Phase 1: Reduce-Scatter
    # ========================
    # After N_d - 1 steps, chunks[rank] contains the fully reduced chunk
    for step in range(N_d - 1):
        send_idx = (rank - step) % N_d
        recv_idx = (rank - step - 1) % N_d

        # In real implementation: send chunks[send_idx] to rank (rank+1) % N_d
        # and receive into chunks[recv_idx] from rank (rank-1) % N_d
        # Here we simulate the accumulation:
        # chunks[recv_idx] += received_chunk  (element-wise sum)
        pass  # Actual send/recv would use dist.send / dist.recv

    # After Phase 1: chunks[rank] = Σ_{i=0}^{N_d-1} tensor_i[rank_chunk]
    # This is exactly reduce-scatter!

    # ========================
    # Phase 2: All-Gather
    # ========================
    # After N_d - 1 steps, every rank has all fully reduced chunks
    for step in range(N_d - 1):
        send_idx = (rank - step + 1) % N_d
        recv_idx = (rank - step) % N_d

        # Send chunks[send_idx] to rank (rank+1) % N_d
        # Receive into chunks[recv_idx] from rank (rank-1) % N_d
        pass  # Actual send/recv

    # Reconstruct full tensor
    result = torch.cat(chunks)
    return result
```

---

## 6. ZeRO Stages: Communication Pattern Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Communication Patterns                          │
├──────────┬───────────────────┬──────────────────┬──────────────────┤
│          │   Backward Pass   │ After Optimizer  │  Forward Pass    │
├──────────┼───────────────────┼──────────────────┼──────────────────┤
│ Vanilla  │   All-Reduce(g)   │       —          │       —          │
│ DP       │      2Ψ           │                  │                  │
├──────────┼───────────────────┼──────────────────┼──────────────────┤
│ ZeRO-1   │ Reduce-Scatter(g) │ All-Gather(θ)    │       —          │
│          │      Ψ            │      Ψ           │                  │
├──────────┼───────────────────┼──────────────────┼──────────────────┤
│ ZeRO-2   │ Reduce-Scatter(g) │ All-Gather(θ)    │       —          │
│          │      Ψ            │      Ψ           │                  │
├──────────┼───────────────────┼──────────────────┼──────────────────┤
│ ZeRO-3   │ Reduce-Scatter(g) │       —          │ All-Gather(θ)    │
│ (FSDP)   │ + All-Gather(θ)   │                  │ All-Gather(θ)    │
│          │      2Ψ           │                  │      Ψ           │
│          │           Total: 3Ψ                                     │
└──────────┴───────────────────┴──────────────────┴──────────────────┘
```

---

## 7. Best Practices Summary

| Practice | Rationale |
|---|---|
| **Use `DistributedSampler` with `drop_last=True`** | Ensures uniform batch sizes across ranks; prevents hangs from uneven data |
| **Call `sampler.set_epoch(epoch)`** | Ensures different shuffling per epoch; otherwise every epoch sees same order |
| **Use NCCL backend** | Optimized for GPU-GPU communication; supports all collective ops |
| **Set `gradient_as_bucket_view=True` in DDP** | Avoids extra memory copy; gradients alias bucket buffer directly |
| **Preallocate contiguous comm buffers** | Avoids runtime memory allocation; tensors must be contiguous for NCCL |
| **Maximize $dp$ before $grad\_acc$** | DP is parallel; gradient accumulation is sequential |
| **Use `model.no_sync()` during accumulation** | Avoids $G$ all-reduces when only 1 is needed |
| **Use `forward_prefetch=True` in FSDP** | Overlaps all-gather of layer $n+1$ with compute of layer $n$ |
| **Keep $dp \leq 512$ for ZeRO-3** | Beyond this, ring latency dominates and overlap degrades |
| **Pin memory (`pin_memory=True`)** | Enables async CPU→GPU transfers via DMA |
| **Use `torch.compile` with `use_orig_params=True`** | Enables graph-level optimizations with FSDP |
| **Bucket size 25 MB (default)** | Empirically optimal tradeoff between latency amortization and overlap granularity |