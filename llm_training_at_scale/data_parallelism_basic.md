# Data Parallelism: A Comprehensive Technical Treatment

---

## 1. Foundational Concept

Data Parallelism (DP) is the most fundamental distributed training strategy for deep learning. The core idea is straightforward: **replicate the entire model** on $N_d$ GPUs (each replica is called a **model instance**), partition the global mini-batch into $N_d$ **micro-batches**, and execute forward and backward passes **concurrently** across all GPUs. Since each GPU processes a different micro-batch, it computes different gradients. To maintain **parameter consistency** across all replicas, the gradients are **averaged** via a distributed collective communication operation called **all-reduce** before the optimizer step.

### Formal Setup

Let the model parameters be denoted $\theta$. Let the global batch $\mathcal{B}$ be partitioned into $N_d$ disjoint micro-batches $\{\mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_{N_d}\}$. On GPU $i$, the local gradient is:

$$
g_i = \frac{1}{|\mathcal{B}_i|} \sum_{x \in \mathcal{B}_i} \nabla_\theta \mathcal{L}(\theta; x)
$$

The synchronized (averaged) gradient used for the parameter update is:

$$
\bar{g} = \frac{1}{N_d} \sum_{i=1}^{N_d} g_i
$$

This averaged gradient $\bar{g}$ is mathematically equivalent to the gradient computed over the entire global batch $\mathcal{B}$:

$$
\bar{g} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \nabla_\theta \mathcal{L}(\theta; x)
$$

The parameter update then proceeds identically on every GPU:

$$
\theta_{t+1} = \theta_t - \eta \cdot \text{OptimizerStep}(\bar{g})
$$

Because every GPU applies the same averaged gradient to the same parameters, all replicas remain **synchronized** after every step.

---

## 2. The All-Reduce Communication Primitive

The operation that computes $\bar{g}$ across all GPUs is **all-reduce**. Formally, given $N_d$ GPUs each holding a tensor $g_i \in \mathbb{R}^{|\theta|}$:

$$
\texttt{all-reduce}(g_1, g_2, \dots, g_{N_d}) \rightarrow \bar{g} = \frac{1}{N_d}\sum_{i=1}^{N_d} g_i \quad \text{(on every GPU)}
$$

After the all-reduce completes, **every GPU** holds the identical averaged gradient $\bar{g}$.

### Communication Cost of All-Reduce (Ring All-Reduce)

For a tensor of size $|\theta|$ (in bytes) across $N_d$ GPUs, the ring all-reduce algorithm has communication volume:

$$
\text{Communication Volume} = 2 \cdot |\theta| \cdot \frac{N_d - 1}{N_d}
$$

This decomposes into a **reduce-scatter** phase (volume $|\theta| \cdot \frac{N_d - 1}{N_d}$) followed by an **all-gather** phase (same volume). As $N_d \to \infty$, the per-GPU communication approaches $2|\theta|$, which is **independent of $N_d$** — making ring all-reduce bandwidth-optimal.

---

## 3. Naive Data Parallelism and Its Inefficiency

A **naive implementation** proceeds sequentially:

1. **Forward pass** on each GPU (computation)
2. **Backward pass** on each GPU (computation)
3. **All-reduce** over gradients (communication)
4. **Optimizer step** (computation)

The critical inefficiency: during step 3, all GPUs are **idle** — they have finished computation and are waiting for communication to complete. This sequential dependency between computation and communication is fundamentally wasteful.

---

## 4. Three Key Optimizations

### 4.1 First Optimization: Overlapping Gradient Synchronization with the Backward Pass

**Key Insight:** In the backward pass, gradients are computed **layer by layer**, starting from the last layer and moving toward the first. The gradient $\nabla_{\theta_L} \mathcal{L}$ for the last layer $L$ is available **before** the gradients for earlier layers $L-1, L-2, \dots, 1$ have been computed.

Therefore, we can **trigger the all-reduce for layer $L$'s gradients immediately**, while the backward pass continues computing gradients for earlier layers.

**Implementation Mechanism:** In PyTorch, this is achieved by registering a **post-accumulate gradient hook** on each parameter:

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

When the gradient for parameter $p$ is computed, the hook fires immediately, launching an asynchronous all-reduce for that parameter's gradient **while** gradients for other parameters are still being computed.

**Result:** The all-reduce communication is **overlapped** with backward computation. In the ideal case, by the time the backward pass finishes computing $\nabla_{\theta_1} \mathcal{L}$, all other gradient all-reduces have already completed, and only the all-reduce for $\nabla_{\theta_1} \mathcal{L}$ remains.

---

### 4.2 Second Optimization: Bucketing Gradients

**Key Insight:** GPU kernels and network operations are significantly more efficient on **large contiguous tensors** than on many small tensors. Launching an independent all-reduce for each individual parameter tensor incurs excessive kernel launch overhead and underutilizes network bandwidth.

**Solution:** Group gradients into **buckets** of a fixed size (e.g., 25 MB in PyTorch DDP). A single all-reduce is launched for the entire bucket once all gradients within that bucket have been computed.

**Procedure:**

1. Assign model parameters to buckets in **reverse** computation order (so that the last-computed gradients fill the first bucket).
2. When all gradients in a bucket are ready, launch a **single all-reduce** for that bucket.
3. This reduces the number of communication operations from $O(P)$ (where $P$ is the number of parameters) to $O\left(\frac{|\theta|}{\text{bucket\_size}}\right)$.

**Analogy:** Instead of shipping many small packages individually, pack items into a few large boxes and ship those — reducing per-item shipping overhead.

---

### 4.3 Third Optimization: Interplay with Gradient Accumulation

**Gradient accumulation** performs $K$ forward-backward passes before a single optimizer step, effectively multiplying the batch size by $K$ without increasing memory:

$$
g_{\text{accumulated}} = \sum_{k=1}^{K} g^{(k)}
$$

**Problem with naive combination:** If DP is active, an all-reduce is triggered after **every** backward pass. But during gradient accumulation, we only need the synchronized gradient after the **final** accumulation step — the intermediate all-reduces are **wasteful**.

**Solution:** Use a **no-sync context manager** to disable gradient synchronization during the first $K-1$ backward passes:

```python
for k in range(K):
    if k < K - 1:
        with model.no_sync():  # Disable all-reduce
            loss = model(micro_batch[k])
            loss.backward()
    else:
        loss = model(micro_batch[k])  # All-reduce triggered here
        loss.backward()
optimizer.step()
```

This reduces communication overhead by a factor of $K$ during gradient accumulation.

---

### Memory Contiguity Note

Communication operations require tensors to be **contiguous in memory**. In practice, **pre-allocated contiguous communication buffers** are used to avoid redundant memory copies. While this accelerates communication, it contributes to **peak memory usage** during training.

---

## 5. Global Batch Size Equation

With data parallelism and gradient accumulation, the relationship between batch sizes is:

$$
\boxed{bs = gbs = mbs \times grad\_acc \times dp}
$$

Where:

| Symbol | Definition |
|--------|-----------|
| $gbs$ | **Global batch size** — total number of samples processed per optimizer step |
| $mbs$ | **Micro-batch size** — number of samples per forward pass on a single GPU |
| $grad\_acc$ | **Gradient accumulation steps** — number of sequential forward-backward passes before an optimizer step |
| $dp$ | **Data parallel degree** — number of GPU replicas |

### Key Practical Principle

**Maximize $dp$ over $grad\_acc$** whenever possible:

- Data parallelism is **inherently parallel** — all $dp$ GPUs compute simultaneously
- Gradient accumulation is **inherently sequential** — $grad\_acc$ steps execute one after another

Therefore, $grad\_acc$ is used only to **fill the gap** when the available GPU count is insufficient to achieve the target $gbs$ through $dp$ alone.

---

## 6. Practical Recipe for 1D Data-Parallel Training

### Step-by-Step Procedure

1. **Determine the optimal global batch size** $gbs$ (in tokens) — from literature or convergence experiments.
2. **Select the training sequence length** $seq\_len$ — typically 2,048 to 8,192 tokens works reliably for current evaluation benchmarks. (Longer documents are rare on the web; shorter sequences suffice for most pretraining.)
3. **Convert to samples:**
$$
gbs_{\text{samples}} = \frac{gbs_{\text{tokens}}}{seq\_len}
$$
4. **Find the maximum micro-batch size** $mbs$ by increasing it on a single GPU until out-of-memory.
5. **Determine available GPUs** $dp$.
6. **Compute the required gradient accumulation steps:**
$$
grad\_acc = \frac{gbs_{\text{samples}}}{mbs \times dp}
$$

### Concrete Example

- Target: $gbs = 4\text{M tokens}$, $seq\_len = 4096$
- Batch size in samples: $gbs_{\text{samples}} = \frac{4{,}000{,}000}{4096} = 976 \approx 1024$ (nearest power of 2)
- Observation: a single GPU fits $mbs = 2$

| GPUs ($dp$) | $grad\_acc = \frac{1024}{2 \times dp}$ | Behavior |
|:-----------:|:--------------------------------------:|----------|
| 128 | $\frac{1024}{256} = 4$ | 4 sequential accumulation steps |
| 512 | $\frac{1024}{1024} = 1$ | No accumulation needed — **faster training** |
| 1024+ | $< 1$ | GPU-rich: reduce $mbs$, explore larger $gbs$, or leave GPUs idle |

---

## 7. Scaling Limits of Data Parallelism

### Communication Overhead at Scale

At large $dp$ (hundreds to thousands of GPUs), the all-reduce overhead grows due to:

1. **Ring latency:** The minimum time for a signal to traverse all $N_d$ nodes in a ring topology scales as $O(N_d)$.
2. **Network bandwidth saturation:** The aggregate gradient traffic approaches network capacity limits.
3. **Overlap breakdown:** The backward pass computation can no longer fully mask the growing communication time.

As a result, **throughput per GPU decreases** with each additional DP rank beyond a critical point (empirically around $dp \approx 512$), while **memory usage per GPU remains constant** — DP does not reduce per-GPU memory.

### Memory Limitation

Data parallelism requires that **at least one complete layer** (and ideally one full forward pass with $mbs = 1$) fits in a single GPU's memory. For large models, this is not possible:

**Quick memory estimate for parameters alone:**

$$
\text{Memory}_{\text{params}} \approx 2 \times N_{\text{params}} \text{ bytes (in half precision)}
$$

For example, a 70B parameter model requires approximately $2 \times 70 \times 10^9 = 140\text{ GB}$ just for parameters in BF16/FP16, exceeding the capacity of a single 80 GB GPU.

---

## 8. Zero Redundancy Optimizer (ZeRO)

### Motivation

In vanilla DP, every GPU stores a **complete copy** of:
- Model parameters
- Gradients
- Optimizer states (e.g., Adam's first and second moments)

This is **massively redundant**. ZeRO eliminates this redundancy by **partitioning** (sharding) these tensors across DP ranks, reconstructing them on demand when needed.

### Memory Baseline (Mixed-Precision Training with Adam)

Let $\Psi$ denote the number of model parameters. In mixed-precision training with Adam:

| Component | Precision | Memory |
|-----------|-----------|--------|
| Parameters | BF16/FP16 | $2\Psi$ |
| Gradients | BF16/FP16 | $2\Psi$ |
| FP32 master copy of parameters | FP32 | $4\Psi$ |
| Adam first moment ($m$) | FP32 | $4\Psi$ |
| Adam second moment ($v$) | FP32 | $4\Psi$ |

The optimizer states memory multiplier is $k = 12$ (FP32 params + two Adam states: $4\Psi + 4\Psi + 4\Psi$).

**Without FP32 gradient accumulation:**

$$
\text{Total Memory} = \underbrace{2\Psi}_{\text{BF16 params}} + \underbrace{2\Psi}_{\text{BF16 grads}} + \underbrace{12\Psi}_{\text{optimizer states}} = 16\Psi
$$

**With FP32 gradient accumulation** (optional additional $4\Psi$):

$$
\text{Total Memory} = 2\Psi + 6\Psi + 12\Psi = 20\Psi
$$

---

### 8.1 ZeRO Stage 1: Optimizer State Partitioning

**What is sharded:** Optimizer states only (FP32 master weights, Adam $m$ and $v$).

**How it works:**

1. **Forward pass:** Each GPU uses the full BF16 parameters $\theta$ (identical across replicas) on its own micro-batch.
2. **Backward pass:** Each GPU computes full gradients $g_i$ on its micro-batch.
3. **Reduce-scatter on gradients:** Instead of all-reduce, perform a **reduce-scatter**. After this operation, GPU $i$ holds only the $\frac{1}{N_d}$-th shard of the summed gradients — precisely the shard corresponding to its optimizer state partition.
4. **Local optimizer step:** Each GPU updates only its $\frac{1}{N_d}$ shard of optimizer states and produces $\frac{1}{N_d}$ of the updated FP32 parameters, which are cast back to BF16.
5. **All-gather on BF16 parameters:** Reconstruct the full BF16 parameter set on every GPU for the next forward pass.

**Memory per GPU:**

$$
\boxed{\text{ZeRO-1 Memory} = 2\Psi + 2\Psi + \frac{k\Psi}{N_d} = 4\Psi + \frac{12\Psi}{N_d}}
$$

As $N_d \to \infty$, memory approaches $4\Psi$ (parameters + gradients only), compared to $16\Psi$ in vanilla DP.

**Communication:**

| Operation | Volume per GPU | When |
|-----------|---------------|------|
| Reduce-scatter (gradients) | $\Psi \cdot \frac{N_d - 1}{N_d} \approx \Psi$ | After backward pass |
| All-gather (BF16 parameters) | $\Psi \cdot \frac{N_d - 1}{N_d} \approx \Psi$ | After optimizer step |

**Note on reduce-scatter vs. all-reduce:** A reduce-scatter has **half** the communication volume of an all-reduce ($\Psi$ vs. $2\Psi$). However, ZeRO-1 adds an all-gather ($\Psi$), so total communication is $\Psi + \Psi = 2\Psi$, **equivalent** to vanilla DP.

#### Overlapping Strategies for the All-Gather

The all-gather of BF16 parameters (step 5) is a **new** communication cost not present in vanilla DP. Two strategies exist to overlap it:

1. **During the optimizer step:** Initiate the all-gather as soon as the first shard is updated, overlapping with updates of remaining shards.
2. **During the forward pass:** Prefetch parameters layer-by-layer — all-gather layer $n+1$'s parameters while computing the forward pass for layer $n$.

---

### 8.2 ZeRO Stage 2: Optimizer State + Gradient Partitioning

**Key Insight:** Since each GPU only needs $\frac{1}{N_d}$ of the gradients (the shard corresponding to its optimizer states), there is no need to store the full gradient tensor.

**What is sharded:** Optimizer states **and** gradients.

**How it works:**

The procedure is identical to ZeRO-1, except:

- After the reduce-scatter in the backward pass, each GPU **retains only its gradient shard** and **discards the rest**.
- Gradients are released from memory on the fly as they are scattered.

**Memory per GPU:**

$$
\boxed{\text{ZeRO-2 Memory} = 2\Psi + \frac{2\Psi + k\Psi}{N_d} = 2\Psi + \frac{14\Psi}{N_d}}
$$

As $N_d \to \infty$, memory approaches $2\Psi$ (parameters only).

Compared to the baseline $16\Psi$, ZeRO-2 achieves up to **8× memory reduction** at large $N_d$.

**Communication:** Identical to ZeRO-1:

$$
\text{Total Communication} = \underbrace{\Psi}_{\text{reduce-scatter (grads)}} + \underbrace{\Psi}_{\text{all-gather (params)}} = 2\Psi
$$

**Practical Note:** ZeRO-2 has **no communication overhead** relative to ZeRO-1 while providing strictly better memory savings. Therefore, ZeRO-2 is generally preferred over ZeRO-1.

---

### 8.3 ZeRO Stage 3: Full Partitioning (FSDP)

**What is sharded:** Optimizer states, gradients, **and** parameters.

> PyTorch's native implementation of ZeRO-3 is called **FSDP** (Fully Sharded Data Parallelism).

**How it works:**

Each GPU stores only $\frac{1}{N_d}$ of the model parameters. Full parameters are **reconstructed on demand** via all-gather, used, and then **immediately discarded**.

#### Forward Pass

For each layer $\ell = 1, 2, \dots, L$:

1. **All-gather** the full parameters $\theta_\ell$ for layer $\ell$ from all $N_d$ GPUs.
2. Compute the forward pass for layer $\ell$.
3. **Discard** the non-local parameter shards (free memory).

#### Backward Pass

For each layer $\ell = L, L-1, \dots, 1$:

1. **All-gather** the full parameters $\theta_\ell$ again (they were discarded after the forward pass).
2. Compute the backward pass for layer $\ell$, producing gradients.
3. **Reduce-scatter** the gradients to retain only the local shard.
4. **Discard** the non-local parameter shards.

**Memory per GPU:**

$$
\boxed{\text{ZeRO-3 Memory} = \frac{2\Psi + 2\Psi + k\Psi}{N_d} = \frac{16\Psi}{N_d}}
$$

As $N_d \to \infty$, the memory for model-related tensors approaches **zero** — hence the name "Zero Redundancy Optimizer."

#### Communication Cost Analysis

| Operation | Count per Step | Volume per Operation | Total |
|-----------|---------------|---------------------|-------|
| All-gather (forward pass) | $L$ layers | $\Psi/L$ each $\Rightarrow \Psi$ total | $\Psi$ |
| All-gather (backward pass) | $L$ layers | $\Psi/L$ each $\Rightarrow \Psi$ total | $\Psi$ |
| Reduce-scatter (gradients) | 1 | $\Psi$ | $\Psi$ |

$$
\boxed{\text{ZeRO-3 Total Communication} = 3\Psi}
$$

This is a **1.5× increase** over ZeRO-2's $2\Psi$.

#### Prefetching to Overlap Communication

The additional all-gathers can be **overlapped** with computation via **prefetching**:

- **Forward pass:** While computing layer $n$, initiate all-gather for layer $n+1$'s parameters.
- **Backward pass:** While computing layer $n$, initiate all-gather for layer $n-1$'s parameters.

This overlap is effective as long as $dp$ does not become excessively large (rule of thumb: $dp \lesssim 512$).

#### Critical Limitation

ZeRO-3 partitions parameters, gradients, and optimizer states but **cannot** partition **activations**. Since each DP replica processes a different micro-batch, the activations are **unique** to each GPU (not duplicated) and therefore cannot be sharded across DP ranks.

Activation memory scales as:

$$
\text{Activation Memory} \propto mbs \times seq\_len \times h \times L
$$

where $h$ is the hidden dimension and $L$ is the number of layers. This remains a bottleneck that requires **activation checkpointing** (recomputation) or **tensor/context parallelism** to address.

---

## 9. Comparative Summary of ZeRO Stages

| Stage | What is Sharded | Memory per GPU | Communication Volume |
|-------|----------------|---------------|---------------------|
| Vanilla DP | Nothing | $2\Psi + 2\Psi + k\Psi = 16\Psi$ | $2\Psi$ (all-reduce) |
| ZeRO-1 | Optimizer states | $4\Psi + \frac{k\Psi}{N_d}$ | $2\Psi$ (reduce-scatter + all-gather) |
| ZeRO-2 | Optimizer states + gradients | $2\Psi + \frac{(2+k)\Psi}{N_d}$ | $2\Psi$ (reduce-scatter + all-gather) |
| ZeRO-3 (FSDP) | Optimizer states + gradients + parameters | $\frac{(4+k)\Psi}{N_d}$ | $3\Psi$ (2× all-gather + reduce-scatter) |

With $k = 12$ (Adam, mixed precision, no FP32 grad accumulation):

| Stage | Memory per GPU | Memory as $N_d \to \infty$ |
|-------|---------------|---------------------------|
| Vanilla DP | $16\Psi$ | $16\Psi$ |
| ZeRO-1 | $4\Psi + \frac{12\Psi}{N_d}$ | $4\Psi$ |
| ZeRO-2 | $2\Psi + \frac{14\Psi}{N_d}$ | $2\Psi$ |
| ZeRO-3 | $\frac{16\Psi}{N_d}$ | $\to 0$ |

---

## 10. Transition to Further Parallelism Dimensions

Data parallelism with ZeRO provides powerful memory savings for model states (parameters, gradients, optimizer states) but faces two fundamental limits:

1. **Communication overhead** grows with $N_d$, eventually dominating computation time.
2. **Activation memory** is not shardable via ZeRO because activations differ across DP replicas.
3. **Single-layer memory** must fit on one GPU — ZeRO-3 gathers full layer parameters, so each layer must fit in GPU memory.

These limitations motivate orthogonal parallelism dimensions:

- **Tensor Parallelism (TP):** Shards parameters, gradients, optimizer states, **and activations** across devices within a layer — without communicating full model parameters between GPUs.
- **Pipeline Parallelism (PP):** Distributes different layers across different GPUs.
- **Context/Sequence Parallelism (CP/SP):** Shards along the sequence length dimension.

Data parallelism constitutes the **first dimension** of parallelism (1D parallelism), upon which these additional dimensions are composed to enable training of models that exceed single-GPU or single-node capacity.