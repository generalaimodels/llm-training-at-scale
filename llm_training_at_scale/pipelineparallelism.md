

# Pipeline Parallelism: A Comprehensive Technical Exposition

---

## 1. Motivation and Context

### 1.1 The Inter-Node Communication Bottleneck

In tensor parallelism (TP), every transformer layer requires collective communication operations (all-reduce, all-gather, reduce-scatter) **within** each layer's computation. When TP is confined to a single node—typically housing 4 or 8 GPUs connected via high-bandwidth interconnects such as NVLink (up to ~900 GB/s bidirectional on modern hardware)—the overhead remains tolerable. However, scaling TP **across** nodes forces communication over lower-bandwidth fabrics (InfiniBand, typically 25–100 GB/s per link), producing severe performance degradation.

Empirical measurements on multi-node clusters reveal a characteristic pattern: as the number of nodes increases, the effective bandwidth for collective operations (all-reduce, all-gather, reduce-scatter) drops substantially due to:

- **Network topology constraints**: inter-node links have lower bandwidth and higher latency than intra-node NVLink/NVSwitch.
- **Congestion and contention**: multiple simultaneous all-reduce operations compete for shared network resources.
- **Protocol overhead**: TCP/IP or RDMA stack latency accumulates across hops.

### 1.2 Why Not Simply Use Sequence/Context Parallelism or ZeRO?

| Parallelism Strategy | What It Partitions | Limitation |
|---|---|---|
| Sequence/Context Parallelism | Activation tensors along the sequence dimension | Helps only when sequence length is the memory bottleneck, not model size |
| ZeRO-3 (Data Parallelism) | Model parameters, gradients, optimizer states across DP ranks | Requires all-gather of full parameters before every forward/backward computation; high communication volume |
| **Pipeline Parallelism** | **Model layers across pipeline stages** | **Introduces pipeline bubbles; requires careful scheduling** |

For models with 70B+ parameters, the weight memory alone can exceed the aggregate capacity of 4–8 GPUs on a single node. Pipeline parallelism addresses this by distributing **layers** (not shards of every layer) across devices.

---

## 2. Fundamental Concept of Pipeline Parallelism

### 2.1 Layer Partitioning

Given a model with $L$ transformer layers and $p$ pipeline stages (each mapped to one GPU or a group of GPUs), we assign approximately $\lceil L/p \rceil$ consecutive layers to each stage:

$$
\text{Stage } k \text{ holds layers } \left\{ \ell : \left\lfloor \frac{(k-1) \cdot L}{p} \right\rfloor + 1 \leq \ell \leq \left\lfloor \frac{k \cdot L}{p} \right\rfloor \right\}, \quad k = 1, 2, \ldots, p
$$

**Example**: For $L = 32$ layers and $p = 4$ GPUs:
- GPU 0: Layers 1–8
- GPU 1: Layers 9–16
- GPU 2: Layers 17–24
- GPU 3: Layers 25–32

### 2.2 Memory Decomposition Under Pipeline Parallelism

For a model with total parameter count $\Phi$, the memory per GPU for **model parameters** (in mixed precision with optimizer states) scales as:

$$
M_{\text{params}}^{(\text{per GPU})} = \frac{\Phi}{p} \cdot \kappa
$$

where $\kappa$ is the per-parameter memory multiplier (e.g., $\kappa = 18$ bytes for AdamW with fp32 master weights + fp32 momentum + fp32 variance + fp16 parameters + fp16 gradients, or $\kappa = 2$ bytes for inference-only fp16 weights).

**Critical observation**: While parameter memory is divided by $p$, **activation memory is NOT reduced**. The reason is subtle and requires careful analysis.

### 2.3 Why Activation Memory Remains Constant

Each pipeline stage processes $1/p$ of the model's layers per micro-batch, so the activation memory for a single micro-batch through one stage is:

$$
A_{\text{single}} = \frac{A_{\text{total}}}{p}
$$

where $A_{\text{total}}$ is the total activation memory for all layers on a single micro-batch.

However, in standard pipeline schedules, each GPU must complete forward passes on $p$ micro-batches before any backward pass begins (as we will see in the AFAB schedule). Therefore, the total activation memory stored on each GPU is:

$$
A_{\text{per GPU}} = p \times \frac{A_{\text{total}}}{p} = A_{\text{total}}
$$

This cancellation is a fundamental property of naive pipeline parallelism: the memory savings from partitioning layers are exactly offset by the need to buffer multiple micro-batches' activations.

### 2.4 Communication Pattern

Unlike tensor parallelism (which communicates **within** layers via all-reduce/all-gather) or ZeRO-3 (which communicates **parameters**), pipeline parallelism communicates **activation tensors** sequentially between adjacent stages:

$$
\text{Stage } k \xrightarrow{\text{send } \mathbf{h}^{(k)}} \text{Stage } k+1
$$

where $\mathbf{h}^{(k)} \in \mathbb{R}^{b \times s \times d}$ is the hidden state tensor (batch size $b$, sequence length $s$, hidden dimension $d$). During the backward pass, gradients flow in the reverse direction:

$$
\text{Stage } k+1 \xrightarrow{\text{send } \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(k)}}} \text{Stage } k
$$

**Key advantage**: These are point-to-point (P2P) communications occurring only at $p - 1$ stage boundaries, not collective operations at every layer. The communication volume per boundary is:

$$
V_{\text{P2P}} = b \times s \times d \times \text{bytes\_per\_element}
$$

This is dramatically less frequent than TP's per-layer collectives, making PP **particularly well-suited for inter-node scaling**.

---

## 3. The Pipeline Bubble Problem

### 3.1 Naive Sequential Execution

In the simplest implementation, a single batch passes through all stages sequentially: Stage 1 computes the forward pass, sends activations to Stage 2, which computes, sends to Stage 3, and so on. Then the backward pass reverses direction.

During forward computation on Stage $k$, all stages $\{1, \ldots, k-1, k+1, \ldots, p\}$ are **idle**. This idle time constitutes the **pipeline bubble**.

### 3.2 Bubble Quantification (Single Micro-Batch)

Let:
- $t_f$ = time for one forward pass through one pipeline stage for one micro-batch
- $t_b$ = time for one backward pass through one pipeline stage for one micro-batch
- A common empirical approximation: $t_b \approx 2 \cdot t_f$ (backward requires computing gradients w.r.t. both inputs and weights)

**Ideal time** (perfect parallelization, no bubble):

$$
t_{\text{ideal}} = t_f + t_b
$$

**Actual pipeline bubble time** (naive schedule):

$$
t_{\text{bubble}} = (p - 1) \cdot (t_f + t_b)
$$

**Bubble ratio** (bubble time relative to ideal time):

$$
r_{\text{bubble}} = \frac{t_{\text{bubble}}}{t_{\text{ideal}}} = \frac{(p - 1)(t_f + t_b)}{t_f + t_b} = p - 1
$$

This is devastating: with $p = 8$ pipeline stages, we waste $7\times$ the useful computation time. The **pipeline efficiency** is:

$$
\eta_{\text{pipeline}} = \frac{t_{\text{ideal}}}{t_{\text{ideal}} + t_{\text{bubble}}} = \frac{1}{1 + (p-1)} = \frac{1}{p}
$$

With $p = 8$, efficiency is only $12.5\%$.

---

## 4. All Forward All Backward (AFAB) Schedule

### 4.1 Micro-Batching

The first mitigation strategy borrows from data parallelism: split the global batch into $m$ **micro-batches**. When Stage 2 processes micro-batch 1, Stage 1 can begin processing micro-batch 2. This creates a "pipeline fill" phase, a "steady state," and a "pipeline drain" phase.

### 4.2 AFAB Schedule Description

In AFAB:
1. **Forward phase**: All $m$ micro-batches execute their forward passes through all stages. As soon as Stage $k$ finishes the forward pass for micro-batch $i$, it begins micro-batch $i+1$.
2. **Backward phase**: After all forward passes complete, all $m$ micro-batches execute their backward passes in reverse order.

### 4.3 Bubble Analysis with Micro-Batching

**Ideal time** for $m$ micro-batches:

$$
t_{\text{ideal}} = m \cdot (t_f + t_b)
$$

**Bubble time** remains $(p-1)(t_f + t_b)$ since the pipeline fill and drain phases are unchanged:

$$
t_{\text{bubble}} = (p - 1)(t_f + t_b)
$$

**Bubble ratio**:

$$
\boxed{r_{\text{bubble}}^{\text{AFAB}} = \frac{(p-1)(t_f + t_b)}{m(t_f + t_b)} = \frac{p - 1}{m}}
$$

By increasing $m \gg p$, we can make the bubble arbitrarily small. For example, with $p = 8$ and $m = 32$:

$$
r_{\text{bubble}} = \frac{7}{32} = 21.875\%
$$

**Pipeline efficiency**:

$$
\eta_{\text{AFAB}} = \frac{m}{m + p - 1}
$$

### 4.4 Memory Problem in AFAB

AFAB requires storing activations for **all** $m$ micro-batches simultaneously during the forward phase, since no backward pass begins until all forwards complete:

$$
M_{\text{activations}}^{\text{AFAB}} = m \times \frac{A_{\text{total}}}{p}
$$

As $m$ increases to reduce the bubble, activation memory grows linearly—a direct and painful trade-off.

---

## 5. One Forward One Backward (1F1B) Schedule

### 5.1 Core Idea

The 1F1B schedule addresses AFAB's memory explosion by starting backward passes **as soon as possible**. Instead of completing all $m$ forward passes before any backward, 1F1B enters a **steady state** where each GPU alternates between one forward pass and one backward pass.

### 5.2 Schedule Phases

For a pipeline with $p$ stages and $m$ micro-batches ($m \geq p$):

1. **Warmup phase**: Stage $k$ (0-indexed) performs $p - k - 1$ forward passes to fill the pipeline.
2. **Steady state**: Each stage alternates: one forward pass, one backward pass (1F1B).
3. **Cooldown phase**: Remaining backward passes drain the pipeline.

### 5.3 Bubble Analysis

The bubble size in 1F1B is **identical** to AFAB:

$$
\boxed{r_{\text{bubble}}^{\text{1F1B}} = \frac{p - 1}{m}}
$$

The bubble is not reduced. The improvement is purely in **memory**.

### 5.4 Activation Memory Improvement

In 1F1B, the maximum number of in-flight micro-batches (those whose activations must be stored) at any given stage is at most $p$, not $m$:

$$
\boxed{M_{\text{activations}}^{\text{1F1B}} = p \times \frac{A_{\text{total}}}{p} = A_{\text{total}}}
$$

Compare to AFAB:

$$
M_{\text{activations}}^{\text{AFAB}} = m \times \frac{A_{\text{total}}}{p}
$$

Since typically $m \gg p$, 1F1B achieves a memory reduction factor of $m/p$. This is significant: with $m = 32$ and $p = 4$, AFAB stores $8\times$ more activations than 1F1B per stage.

**Crucially**, because 1F1B decouples memory from $m$, we can freely increase $m$ to reduce the bubble **without** proportionally increasing memory.

### 5.5 Implementation Complexity

1F1B breaks the clean separation between forward and backward phases. Each device independently schedules forward and backward operations according to its position in the pipeline. This requires:

- **Asynchronous scheduling logic**: Each stage must track which micro-batches have completed forward passes and which are ready for backward passes.
- **Modified training loops**: The standard `forward() → loss() → backward() → step()` paradigm must be replaced with a state-machine-like scheduler.
- **Model code modifications**: The model must be sliceable into stages, each independently callable.

### 5.6 Empirical Scaling Behavior

Benchmark results reveal two regimes:

| Configuration | Behavior |
|---|---|
| $m \leq p - 1$ | Performance degrades with increasing $p$; bubble dominates |
| $m \gg p - 1$ (e.g., $m = 32$) | Reasonable scaling at low $p$; still limited at very large $p$ |

**Inter-node scaling advantage**: When crossing node boundaries (e.g., $p = 8$ on one node to $p = 16$ across two nodes), PP shows only ~14% throughput degradation versus ~43% for TP. This is because PP communicates **point-to-point activation tensors at stage boundaries** rather than **all-reduce across all GPUs within every layer**.

---

## 6. Interleaved Stages (Interleaved 1F1B)

### 6.1 Concept: Non-Contiguous Layer Assignment

Instead of assigning contiguous blocks of layers to each GPU, we assign $v$ **chunks** (also called virtual stages) per GPU. Each GPU holds $v$ non-contiguous groups of layers, and a micro-batch "loops" through GPUs multiple times during a single forward pass.

**Example** with $L = 16$, $p = 4$, $v = 2$:

| GPU | Chunk 1 (layers) | Chunk 2 (layers) |
|---|---|---|
| GPU 0 | 1–2 | 9–10 |
| GPU 1 | 3–4 | 11–12 |
| GPU 2 | 5–6 | 13–14 |
| GPU 3 | 7–8 | 15–16 |

A micro-batch traverses: GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0 → GPU 1 → GPU 2 → GPU 3 during the forward pass.

### 6.2 Bubble Reduction

Each forward and backward pass through a single chunk takes $t_f/v$ and $t_b/v$ respectively (since each chunk has $1/v$-th of the layers). The bubble time becomes:

$$
t_{\text{bubble}}^{\text{interleaved}} = \frac{(p-1)(t_f + t_b)}{v}
$$

**Bubble ratio**:

$$
\boxed{r_{\text{bubble}}^{\text{interleaved}} = \frac{p - 1}{v \cdot m}}
$$

### 6.3 Communication Trade-Off

Interleaving increases the number of P2P communications by a factor of $v$. Previously, each micro-batch required $p - 1$ send/receive pairs per direction. With interleaving, each micro-batch requires $v \cdot p - 1$ communication steps (since it traverses the full pipeline $v$ times).

**Total communication volume per micro-batch**:

$$
V_{\text{comm}}^{\text{interleaved}} = v \cdot (p - 1) \cdot V_{\text{P2P}} = v \cdot (p - 1) \cdot b \cdot s \cdot d \cdot \text{bytes\_per\_element}
$$

This is a factor of $v$ increase over non-interleaved PP.

### 6.4 Depth-First vs. Breadth-First Scheduling

With interleaved stages, a scheduling ambiguity arises. At any given moment on a GPU, we must choose between:

- **Depth-first**: Prioritize advancing **earlier micro-batches** through **later chunks**. This minimizes the end-to-end latency for individual micro-batches, releasing their activations sooner (lower memory, lower time-to-first-backward).

- **Breadth-first**: Prioritize advancing **later micro-batches** through **earlier chunks**. This fills the pipeline more uniformly, potentially improving steady-state utilization.

The optimal choice depends on the specific $p$, $v$, $m$ configuration and the relative costs of computation versus communication.

### 6.5 Configuration Space

The design space for pipeline parallelism can be parameterized as $(p, m, v)$:

| $m$ | $v$ | Schedule Type |
|---|---|---|
| 1 | 1 | Naive PP (single batch, single chunk) |
| $m > 1$ | 1 | AFAB or 1F1B |
| $m > 1$ | $v > 1$ | Interleaved 1F1B |

Llama 3.1's training infrastructure uses an interleaved 1F1B schedule with a tunable depth-first/breadth-first priority.

---

## 7. Zero Bubble and DualPipe Schedules

### 7.1 Decomposing the Backward Pass

The breakthrough enabling near-zero bubble schedules is the observation that the backward pass through a linear (matrix multiplication) layer decomposes into **two independent operations**:

Given a linear layer computing $\mathbf{Y} = \mathbf{X}\mathbf{W}$ with incoming gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$:

1. **Input gradient ($B$)**:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \cdot \mathbf{W}^{\top}
$$

This is required by the **preceding** pipeline stage to continue its backward pass.

2. **Weight gradient ($W$)**:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^{\top} \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}
$$

This is required only for the **optimizer step** at the end of the iteration. It has **no downstream dependency** in the pipeline.

### 7.2 Dependency Analysis

The critical insight is the **asymmetry in dependencies**:

| Operation | Depends On | Required By |
|---|---|---|
| Forward ($F$) of stage $k$ | $F$ of stage $k-1$ | $F$ of stage $k+1$, $B$ of stage $k$ |
| Input backward ($B$) of stage $k$ | $B$ of stage $k+1$ | $B$ of stage $k-1$ |
| Weight backward ($W$) of stage $k$ | $B$ of stage $k$ (same stage) | Only the optimizer step |

Since $W$ has no inter-stage dependency, it can be **scheduled anywhere** after the corresponding $B$ of the same stage and before the optimizer step. This flexibility allows $W$ operations to be placed into what would otherwise be bubble slots.

### 7.3 Zero Bubble Schedule (ZB-H2)

By decomposing backward passes into $B$ and $W$ and solving a scheduling optimization problem, it is possible to construct schedules where the bubble is **theoretically zero**:

$$
\boxed{r_{\text{bubble}}^{\text{ZB-H2}} \approx 0}
$$

The total time per iteration approaches the ideal:

$$
t_{\text{total}} \approx m \cdot (t_f + t_B + t_W)
$$

where $t_B$ and $t_W$ are the times for the input-backward and weight-backward operations respectively, with $t_B + t_W = t_b$.

### 7.4 Scheduling as Integer Linear Programming

Finding the optimal placement of $F$, $B$, and $W$ operations across $p$ stages and $m$ micro-batches is formulated as an **Integer Linear Programming (ILP)** problem:

**Decision variables**: Start time $s_{i,k}^{(\text{op})}$ for each operation type $\text{op} \in \{F, B, W\}$, micro-batch $i \in \{1, \ldots, m\}$, stage $k \in \{1, \ldots, p\}$.

**Objective**:

$$
\min \sum_{k=1}^{p} \left( t_{\text{end}}^{(k)} - t_{\text{start}}^{(k)} - \sum_{i=1}^{m} (t_f + t_B + t_W) \right)
$$

This minimizes total idle (bubble) time across all stages.

**Constraints**:
1. **Dependency constraints**: $B$ of stage $k$ cannot start before $B$ of stage $k+1$ finishes (for the same micro-batch).
2. **Non-overlap**: No two operations on the same GPU overlap in time.
3. **Ordering**: $W_{i,k}$ must follow $B_{i,k}$.
4. **All $W$ complete before optimizer step**.

### 7.5 DualPipe (DeepSeek-V3/R1)

DualPipe extends the zero-bubble concept with **bidirectional pipeline streams**:

- **Stream A**: Micro-batches propagate forward from Stage 1 → Stage $p$.
- **Stream B**: Micro-batches propagate forward from Stage $p$ → Stage 1.

Both streams are interleaved on each GPU, with the $F$, $B$, $W$ decomposition applied to both. The key benefits:

1. **Doubled pipeline utilization**: Two independent forward-backward chains fill each other's bubbles.
2. **Near-zero all-to-all communication overhead**: As reported in the DeepSeek-V3 technical report, the overlap between computation and communication is nearly perfect.

The resulting schedule is substantially more complex—each GPU may be simultaneously handling:
- Forward of micro-batch $i$ from Stream A (chunk $j$)
- Input-backward of micro-batch $i'$ from Stream B (chunk $j'$)
- Weight-backward of micro-batch $i''$ from Stream A (chunk $j''$)

---

## 8. Summary of Pipeline Parallelism Schedules

| Schedule | Bubble Ratio $r_{\text{bubble}}$ | Peak Activation Memory per Stage | Communication Overhead | Implementation Complexity |
|---|---|---|---|---|
| Naive (1 micro-batch) | $p - 1$ | $A_{\text{total}}$ | Minimal | Trivial |
| AFAB ($m$ micro-batches) | $\dfrac{p-1}{m}$ | $\dfrac{m \cdot A_{\text{total}}}{p}$ | Minimal | Low |
| 1F1B ($m$ micro-batches) | $\dfrac{p-1}{m}$ | $A_{\text{total}}$ | Minimal | Moderate |
| Interleaved 1F1B ($v$ chunks) | $\dfrac{p-1}{v \cdot m}$ | $\leq A_{\text{total}}$ | $v \times$ baseline | High |
| Zero Bubble (ZB-H2) | $\approx 0$ | $\leq A_{\text{total}}$ | Baseline + scheduling | Very High |
| DualPipe | $\approx 0$ | $\leq A_{\text{total}}$ | Bidirectional streams | Extremely High |

### Pipeline Efficiency Summary

$$
\boxed{\eta_{\text{pipeline}} = \frac{1}{1 + r_{\text{bubble}}} = \frac{v \cdot m}{v \cdot m + p - 1}}
$$

For the zero-bubble case:

$$
\eta_{\text{pipeline}}^{\text{ZB}} \approx 1
$$

---

## 9. Pipeline Parallelism vs. Other Parallelism Strategies

### 9.1 PP vs. TP for Inter-Node Scaling

| Property | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| Communication type | All-reduce / all-gather (collective) | Point-to-point (P2P) |
| Communication frequency | Multiple times per layer | Once per stage boundary |
| Communication volume per operation | $O(b \cdot s \cdot d)$ per layer | $O(b \cdot s \cdot d)$ per stage boundary |
| Sensitivity to bandwidth | Very high (many operations) | Low (few operations) |
| Empirical inter-node degradation | ~43% throughput loss | ~14% throughput loss |

### 9.2 PP vs. ZeRO-3

| Property | ZeRO-3 | Pipeline Parallelism |
|---|---|---|
| What is partitioned | Parameters, gradients, optimizer states | Model layers |
| Communication pattern | All-gather parameters before each forward/backward | P2P activations at stage boundaries |
| Activation memory | Unchanged | Unchanged (for 1F1B); reduced (for interleaved) |
| Compute efficiency | No bubble (but high communication) | Bubble exists (but low communication) |

---

## 10. Practical Design Considerations

### 10.1 Choosing $p$, $m$, and $v$

The practitioner must balance:

1. **Bubble minimization**: Requires $m \gg p - 1$ and large $v$.
2. **Memory constraints**: 1F1B caps activation memory at $\sim A_{\text{total}}$; AFAB grows with $m$.
3. **Communication overhead**: Interleaving ($v > 1$) multiplies P2P volume by $v$.
4. **Global batch size constraint**: $m$ is bounded above by $B_{\text{global}} / (b \times \text{DP\_degree})$, where $b$ is the micro-batch size.

### 10.2 Optimal Stage Partitioning

Layers may have heterogeneous computation costs (e.g., the embedding layer, the final language modeling head). Optimal partitioning minimizes the maximum per-stage computation time:

$$
\min_{P} \max_{k=1,\ldots,p} \sum_{\ell \in P_k} c_\ell
$$

where $P = \{P_1, \ldots, P_p\}$ is a partition of layers and $c_\ell$ is the computation cost of layer $\ell$. Imbalanced stages exacerbate the bubble, as the slowest stage becomes the bottleneck.

---

This completes the technical exposition of pipeline parallelism, from its fundamental motivation through naive schedules, micro-batch–based AFAB and 1F1B schedules, interleaved stages, and the frontier zero-bubble / DualPipe methods that approach theoretically optimal GPU utilization through fine-grained backward-pass decomposition and ILP-based schedule optimization.