

# Pipeline Parallelism: Comprehensive Technical Exposition

---

## 1. Motivation: Why Pipeline Parallelism?

### 1.1 The Inter-Node Communication Bottleneck

Tensor Parallelism (TP) partitions individual layer weight matrices across GPUs within a single node. However, scaling TP beyond the GPUs on a single node (typically $N_{\text{intra}} = 4$ or $8$) forces communication across **inter-node** network links. These links operate at significantly lower bandwidth than intra-node interconnects (e.g., NVLink at $\sim$900 GB/s vs. InfiniBand at $\sim$50–400 GB/s), degrading performance during collective operations such as **all-reduce**, **all-gather**, and **reduce-scatter**.

> **Key empirical observation:** When benchmarking all-reduce across multiple nodes (each with 8 GPUs), median bandwidth drops substantially as node count increases, and variance (5th–95th percentile spread) widens—confirming that inter-node communication is a primary scaling bottleneck for TP.

### 1.2 The Model Size Problem

For large models ($\geq 70\text{B}$ parameters), the memory footprint of weights alone can exceed the aggregate GPU memory of a single node. Specifically, for a model with $P$ parameters stored in mixed precision (e.g., 2 bytes per parameter for BF16):

$$
\text{Weight Memory} = P \times 2 \;\text{bytes}
$$

For $P = 70 \times 10^9$:

$$
\text{Weight Memory} = 70 \times 10^9 \times 2 = 140 \;\text{GB}
$$

This exceeds the memory capacity of 4–8 GPUs (e.g., 4 × 80 GB = 320 GB leaves little room for optimizer states, gradients, and activations). Sequence parallelism and context parallelism address long-sequence memory pressure but do **not** address the fundamental weight memory constraint.

**Pipeline Parallelism (PP)** resolves this by partitioning the model **along the depth (layer) dimension** across multiple GPUs.

---

## 2. Core Concept: Partitioning Layers Across GPUs

### 2.1 Definition

Pipeline parallelism splits a model's $L$ transformer layers into $p$ contiguous groups called **stages**, distributing each stage to a separate GPU (or device). If we have $p$ GPUs and $L$ layers:

$$
\text{Layers per stage} = \frac{L}{p}
$$

**Example:** For $L = 32$ layers and $p = 8$ GPUs:

| GPU | Layers |
|-----|--------|
| GPU 0 | Layers 1–4 |
| GPU 1 | Layers 5–8 |
| GPU 2 | Layers 9–12 |
| GPU 3 | Layers 13–16 |
| GPU 4 | Layers 17–20 |
| GPU 5 | Layers 21–24 |
| GPU 6 | Layers 25–28 |
| GPU 7 | Layers 29–32 |

### 2.2 Memory Reduction for Model Parameters

Each GPU stores only $\frac{1}{p}$ of the model's parameters. For an 8B parameter model with $p = 4$:

$$
\text{Parameters per GPU} = \frac{8 \times 10^9}{4} = 2 \times 10^9
$$

$$
\text{Weight memory per GPU} = \frac{P \times 2}{p} \;\text{bytes}
$$

### 2.3 Activation Memory: No Savings

A critical and initially counterintuitive observation: **activation memory is NOT reduced** by pipeline parallelism.

**Explanation:** Each GPU handles $\frac{1}{p}$ of the layers, so the activation memory per micro-batch per stage is $\frac{\text{activs}}{p}$. However, in any PP schedule, each GPU must perform $p$ (or more) forward passes on successive micro-batches **before** beginning the first backward pass. Therefore, the total activation memory stored simultaneously on each GPU is:

$$
\text{Activation memory per GPU} = p \times \frac{\text{activs}}{p} = \text{activs}
$$

where $\text{activs}$ denotes the total activation memory for a single micro-batch across all layers. The activation memory per GPU thus remains approximately equal to the non-parallelized case.

### 2.4 Communication Pattern

Unlike data parallelism (which communicates **gradients**) or ZeRO-3 (which communicates **parameters**), pipeline parallelism communicates **activation tensors** sequentially between adjacent stages. Between stage $s$ and stage $s+1$, the output activation tensor of the last layer on GPU $s$ is sent as input to the first layer on GPU $s+1$.

$$
\text{PP Communication} = \text{point-to-point send/recv of activation tensors at stage boundaries}
$$

**Advantage over TP:** Communication occurs only at $p - 1$ stage boundaries (once per stage transition), rather than multiple times within every layer. The volume per communication is moderate (one hidden-state tensor of shape $[B_{\mu}, S, H]$ where $B_{\mu}$ is micro-batch size, $S$ is sequence length, $H$ is hidden dimension).

---

## 3. The Pipeline Bubble: Fundamental Inefficiency

### 3.1 Naive Scheduling (Single Micro-Batch)

When a single batch is processed through $p$ stages sequentially, only **one GPU is active at any given time**. All other GPUs are idle.

**Timing definitions:**
- $t_f$: Time for a forward pass through one stage for one micro-batch
- $t_b$: Time for a backward pass through one stage for one micro-batch
- Common approximation: $t_b \approx 2 \times t_f$ (backward involves computing gradients w.r.t. both inputs and weights)

**Ideal time** (if perfectly parallelized):

$$
t_{\text{ideal}} = t_f + t_b
$$

**Pipeline bubble time** (additional idle time):

$$
t_{\text{bubble}} = (p - 1) \times (t_f + t_b)
$$

This represents the cumulative idle time across all stages where GPUs wait while others compute.

**Bubble ratio** (fraction of wasted time relative to ideal):

$$
\boxed{r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{t_f + t_b} = p - 1}
$$

For $p = 8$: $r_{\text{bubble}} = 7$, meaning the bubble time is $7\times$ the ideal compute time. This is catastrophically inefficient.

---

## 4. All Forward, All Backward (AFAB) Schedule

### 4.1 Concept

To mitigate the bubble, we split the global batch into $m$ **micro-batches**. The schedule proceeds as:

1. **All Forward Phase:** Process all $m$ micro-batches through the forward pass sequentially across stages.
2. **All Backward Phase:** Process all $m$ micro-batches through the backward pass sequentially.

When GPU $s+1$ begins processing micro-batch $i$, GPU $s$ can immediately start processing micro-batch $i+1$. This creates a pipelined overlap.

### 4.2 Bubble Analysis

**Ideal time** for $m$ micro-batches:

$$
t_{\text{ideal}} = m \times (t_f + t_b)
$$

**Bubble time** remains:

$$
t_{\text{bubble}} = (p - 1) \times (t_f + t_b)
$$

**Bubble ratio:**

$$
\boxed{r_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{m}}
$$

By increasing $m$, the bubble ratio decreases inversely. For $p = 8$, $m = 32$:

$$
r_{\text{bubble}} = \frac{7}{32} = 0.21875 \approx 21.9\%
$$

### 4.3 Memory Problem

AFAB requires storing activations for **all** $m$ micro-batches simultaneously, because no backward pass begins until all forward passes complete. The activation memory requirement on each GPU is:

$$
\text{Activation memory}_{\text{AFAB}} = m \times \frac{\text{activs}}{p}
$$

Since we want large $m$ to reduce the bubble, this creates a **memory explosion**—activations for many micro-batches must be retained until the backward phase begins.

---

## 5. One Forward, One Backward (1F1B) Schedule

### 5.1 Concept

The 1F1B schedule addresses the activation memory problem by **starting backward passes as early as possible**. The schedule has three phases:

1. **Warm-up phase:** Each GPU performs successive forward passes to fill the pipeline (the number of warm-up forward passes depends on the stage position).
2. **Steady-state phase:** Each GPU alternates between one forward pass and one backward pass (hence "1F1B").
3. **Cool-down phase:** Each GPU drains remaining backward passes.

### 5.2 Bubble Analysis

The bubble size in 1F1B is **identical** to AFAB:

$$
\boxed{r_{\text{bubble}} = \frac{p - 1}{m}}
$$

The bubble is not reduced because the total amount of idle time at the start (warm-up) and end (cool-down) remains the same—it is simply rearranged.

### 5.3 Memory Advantage

The critical improvement is in activation memory. In 1F1B, each GPU stores activations for at most $p$ micro-batches (not $m$ micro-batches as in AFAB):

$$
\text{Activation memory}_{\text{1F1B}} = p \times \frac{\text{activs}}{p} = \text{activs}
$$

In contrast to AFAB where activation memory was $m \times \frac{\text{activs}}{p}$, 1F1B limits it to $p \times \frac{\text{activs}}{p}$. Since typically $m \gg p$ in practical configurations, this is a **substantial memory reduction**.

**Because** 1F1B uses less activation memory, we can **increase** $m$ further (without running out of memory), which **indirectly** reduces the bubble ratio $\frac{p-1}{m}$.

### 5.4 Practical Scaling Behavior

Empirical benchmarks reveal two regimes:

| Configuration | Behavior |
|---|---|
| $m \leq p - 1$ | Bubble dominates; performance **degrades** as $p$ increases |
| $m = 32 \gg p - 1$ | Performance improves at low $p$; still limited at very large $p$ |

**Cross-node scaling advantage:** When scaling from one node ($p = 8$) to two nodes ($p = 16$), PP shows only $\sim$14% performance drop, compared to $\sim$43% for TP. This is because PP communicates only point-to-point activation tensors at stage boundaries, whereas TP requires bandwidth-intensive collective operations (all-reduce) within every layer.

### 5.5 Implementation Complexity

In 1F1B, forward and backward passes are **no longer globally sequential**. Different GPUs execute forward and backward passes for different micro-batches concurrently. This requires:

- **Per-device scheduling logic** to decide when to switch between forward and backward execution.
- Extensive modifications to both training loop code and model code.
- Careful management of micro-batch indexing and gradient accumulation.

---

## 6. Interleaved Stages

### 6.1 Concept

Instead of assigning **contiguous** layer blocks to each GPU, interleaved PP assigns **non-contiguous** layer subsets. Each GPU hosts $v$ **model chunks** (also called virtual stages), where $v$ is the number of chunks per GPU.

**Example** with $L = 8$ layers, $p = 2$ GPUs, $v = 2$ chunks per GPU:

| GPU | Chunks | Layers |
|-----|--------|--------|
| GPU 0 | Chunk 0, Chunk 2 | Layers 1–2, Layers 5–6 |
| GPU 1 | Chunk 1, Chunk 3 | Layers 3–4, Layers 7–8 |

A micro-batch now **loops** through the GPUs multiple times during a single forward pass: GPU 0 → GPU 1 → GPU 0 → GPU 1 → ...

### 6.2 Bubble Reduction

Each forward and backward pass through a single chunk is $v$ times shorter than a full stage pass. The pipeline bubble time becomes:

$$
t_{\text{bubble}} = \frac{(p - 1) \times (t_f + t_b)}{v}
$$

The bubble ratio:

$$
\boxed{r_{\text{bubble}} = \frac{1}{v} \cdot \frac{(p - 1) \times (t_f + t_b)}{m \times (t_f + t_b)} = \frac{p - 1}{v \cdot m}}
$$

where:
- $p$ = pipeline parallelism degree (number of GPUs)
- $m$ = number of micro-batches
- $v$ = number of model chunks (virtual stages) per GPU

### 6.3 Communication Trade-off

The number of point-to-point communications increases by a factor of $v$, since each micro-batch traverses each GPU $v$ times instead of once. This introduces a direct trade-off:

$$
\text{Communication volume} \propto v \times (p - 1)
$$

$$
\text{Bubble size} \propto \frac{p - 1}{v \cdot m}
$$

The optimal $v$ balances reduced idle time against increased communication overhead.

### 6.4 Scheduling Policies: Depth-First vs. Breadth-First

With interleaved stages, a scheduling decision arises at each time step for each GPU: should it prioritize:

| Policy | Description | Effect |
|--------|-------------|--------|
| **Depth-first** | Advance earlier micro-batches through later layers first | Minimizes per-micro-batch latency; completes individual micro-batches faster, freeing activation memory sooner |
| **Breadth-first** | Advance later micro-batches through earlier layers first | Maximizes pipeline filling; keeps all stages busy |

The Llama 3.1 training pipeline uses a **1F1B schedule with interleaved stages**, with a tunable priority parameter between depth-first and breadth-first policies.

### 6.5 Special Cases Summary

| $m$ | $v$ | Schedule Type |
|-----|-----|--------------|
| $1$ | $1$ | Naive PP (single micro-batch, single chunk) |
| $m > 1$ | $1$ | AFAB or 1F1B |
| $m > 1$ | $v > 1$ | Interleaved 1F1B |

---

## 7. Zero Bubble Pipeline Parallelism

### 7.1 Key Insight: Decomposing the Backward Pass

The backward pass through a linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$ involves **two independent gradient computations**:

1. **Input gradient ($B$):** Gradient w.r.t. the input activations $\mathbf{X}$, needed to propagate gradients to earlier layers.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \cdot \mathbf{W}^{\top}
$$

2. **Weight gradient ($W$):** Gradient w.r.t. the weight matrix $\mathbf{W}$, needed for the optimizer update.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^{\top} \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}
$$

**Critical observation:** Operation $B$ must complete before the backward pass of the preceding stage can begin (it is on the critical path). However, operation $W$ has **no such dependency**—it only needs to complete before the optimizer step. Therefore:

$$
W \text{ can be scheduled anywhere after the corresponding } B \text{ of the same stage}
$$

### 7.2 Exploiting the Decomposition

By splitting the coarse-grained backward pass into fine-grained $B$ and $W$ operations, we gain scheduling flexibility. The $W$ operations can be **strategically placed into bubble slots** that would otherwise be idle.

**Timing decomposition:**

$$
t_b = t_B + t_W
$$

where $t_B$ is the time for the input gradient computation and $t_W$ is the time for the weight gradient computation.

### 7.3 ZB-H1 and ZB-H2 Schedules

The Zero Bubble paper proposes two schedules:

| Schedule | Description | Bubble |
|----------|-------------|--------|
| **ZB-H1** | Handcrafted schedule with $B$/$W$ decomposition | Significantly reduced |
| **ZB-H2** | Optimized schedule filling all bubbles with $W$ operations | **Theoretically zero** |

In ZB-H2, every idle slot is filled with a $W$ computation, achieving:

$$
\boxed{r_{\text{bubble}} \approx 0}
$$

### 7.4 Optimal Scheduling via Integer Linear Programming (ILP)

Finding the optimal placement of $F$ (forward), $B$ (input backward), and $W$ (weight backward) operations across $p$ stages and $m$ micro-batches is formulated as an **Integer Linear Programming** problem:

**Objective:**

$$
\min \; t_{\text{bubble}}
$$

**Subject to constraints:**
1. **Data dependency:** $F_i^{s+1}$ cannot start before $F_i^{s}$ completes (forward propagation order).
2. **Data dependency:** $B_i^{s}$ cannot start before $B_i^{s+1}$ completes (backward propagation order).
3. **Ordering:** $B_i^{s}$ cannot start before $F_i^{s}$ completes.
4. **Flexible scheduling:** $W_i^{s}$ must occur after $B_i^{s}$ but can occur at any later time.
5. **Optimizer constraint:** All $W_i^{s}$ for all micro-batches $i$ must complete before the optimizer step.
6. **Non-overlap:** No two operations on the same GPU can overlap in time.

Here, $F_i^{s}$ denotes the forward pass for micro-batch $i$ at stage $s$, and similarly for $B_i^{s}$ and $W_i^{s}$.

---

## 8. DualPipe (DeepSeek-V3/R1)

### 8.1 Concept

DualPipe extends the zero-bubble decomposition by introducing **two concurrent pipeline streams** propagating from **both ends** of the pipeline dimension simultaneously:

- **Stream 1:** Micro-batches flow forward from stage 0 → stage $p-1$ (left to right)
- **Stream 2:** Micro-batches flow forward from stage $p-1$ → stage 0 (right to left)

These bidirectional streams are **interleaved** on each GPU, ensuring that when one stream encounters a dependency stall, the other stream can utilize the idle compute cycles.

### 8.2 Fine-Grained Operation Decomposition

DualPipe further decomposes operations beyond $F$, $B$, $W$ to include **communication operations** (all-to-all for MoE expert routing in DeepSeek-V3). The scheduling interleaves:

- Forward computation ($F$)
- Input backward computation ($B$)  
- Weight backward computation ($W$)
- All-to-all communication (for expert parallelism)

By overlapping communication with computation from the opposing stream, DeepSeek-V3 achieved:

$$
\text{near-zero all-to-all communication overhead}
$$

### 8.3 Complexity

The DualPipe schedule is significantly more complex than 1F1B or interleaved schedules. Its design requires:

1. Precise profiling of individual operation durations ($t_F$, $t_B$, $t_W$, $t_{\text{comm}}$).
2. Solving an ILP or heuristic optimization problem for operation placement.
3. Bidirectional pipeline infrastructure with careful synchronization.

---

## 9. Comparative Summary of Pipeline Schedules

| Schedule | Bubble Ratio $r_{\text{bubble}}$ | Activation Memory per GPU | Communication Volume | Implementation Complexity |
|----------|----------------------------------|--------------------------|----------------------|--------------------------|
| **Naive** (single micro-batch) | $p - 1$ | $\text{activs}$ | $(p-1)$ sends | Low |
| **AFAB** ($m$ micro-batches) | $\dfrac{p - 1}{m}$ | $m \cdot \dfrac{\text{activs}}{p}$ | $(p-1)$ sends per micro-batch | Low |
| **1F1B** ($m$ micro-batches) | $\dfrac{p - 1}{m}$ | $p \cdot \dfrac{\text{activs}}{p} = \text{activs}$ | $(p-1)$ sends per micro-batch | Medium |
| **Interleaved 1F1B** ($v$ chunks) | $\dfrac{p - 1}{v \cdot m}$ | $\text{activs}$ (reduced per chunk) | $v \cdot (p-1)$ sends per micro-batch | High |
| **Zero Bubble (ZB-H2)** | $\approx 0$ | Similar to 1F1B | Similar to 1F1B | Very High |
| **DualPipe** | $\approx 0$ | Similar to 1F1B | Bidirectional + overlapped | Very High |

---

## 10. Key Mathematical Relations: Consolidated Reference

### Pipeline Bubble (General)

$$
\boxed{r_{\text{bubble}} = \frac{p - 1}{v \cdot m}}
$$

where:
- $p$ = number of pipeline stages (GPUs allocated to PP)
- $m$ = number of micro-batches
- $v$ = number of interleaved chunks per GPU ($v = 1$ for non-interleaved)

### Backward Pass Decomposition

$$
t_b = t_B + t_W
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \cdot \mathbf{W}^{\top} \quad (B: \text{on critical path})
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^{\top} \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \quad (W: \text{flexibly schedulable})
$$

### Memory per GPU (Parameters)

$$
\text{Param memory per GPU} = \frac{P \times \text{bytes\_per\_param}}{p}
$$

### Communication Advantage over TP

$$
\text{PP: } (p - 1) \times v \text{ point-to-point transfers per micro-batch}
$$

$$
\text{TP: Multiple all-reduce operations per layer per micro-batch}
$$

---

## 11. Practical Design Principles

1. **Choose $m \gg p - 1$** to minimize the bubble ratio, subject to the constraint that $m$ divides the global batch size.

2. **PP excels across nodes** because point-to-point activation transfers tolerate lower inter-node bandwidth far better than TP's collective all-reduce operations ($\sim$14% degradation vs. $\sim$43% for TP when crossing node boundaries).

3. **Interleaving ($v > 1$) trades communication for compute efficiency.** Increase $v$ only when intra-stage communication bandwidth is sufficient to absorb the $v$-fold increase in transfers.

4. **1F1B is strictly preferred over AFAB** when activation memory is the binding constraint, as it reduces peak activation storage from $\mathcal{O}(m)$ to $\mathcal{O}(p)$ micro-batches.

5. **Zero-bubble methods require fine-grained profiling and ILP-based scheduling,** making them implementation-intensive but near-optimal for large-scale deployments (e.g., DeepSeek-V3/R1).

6. **PP does not reduce activation memory per se**—it reduces parameter memory. Activation recomputation (gradient checkpointing) remains the primary tool for activation memory reduction and is orthogonal to PP.