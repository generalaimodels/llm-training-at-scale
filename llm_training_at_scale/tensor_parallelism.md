

# Tensor Parallelism (TP) and Sequence Parallelism (SP)

---

## 1. Motivation and Problem Statement

ZeRO-style data parallelism shards **parameters**, **gradients**, and **optimizer states** across GPUs. However, as model scale increases, **activation memory** becomes the dominant bottleneck. ZeRO requires an **all-gather** to reconstruct the full parameter tensor *before* each computation step, which limits scalability.

**Tensor Parallelism (TP)** addresses this by sharding **weights, gradients, optimizer states, and activations simultaneously**, performing computation directly on the shards *without* gathering the full tensors beforehand.

---

## 2. Mathematical Foundations of Tensor Parallelism

Tensor parallelism exploits two fundamental properties of matrix multiplication. Given matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:

### 2.1 Column-wise Partitioning (Equation 1)

Partition $B$ into $N$ column blocks:

$$
B = \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{k \times \frac{n}{N}}
$$

Then:

$$
A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots & AB_N \end{bmatrix}
$$

Each partial product $AB_i \in \mathbb{R}^{m \times \frac{n}{N}}$ can be computed **independently** on GPU $i$. The full result is recovered via **concatenation** (all-gather along the column dimension).

### 2.2 Row-wise Partitioning (Equation 2)

Partition $A$ into $N$ column blocks and $B$ into $N$ corresponding row blocks:

$$
A = \begin{bmatrix} A_1 & A_2 & \cdots & A_N \end{bmatrix}, \quad A_i \in \mathbb{R}^{m \times \frac{k}{N}}
$$

$$
B = \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{\frac{k}{N} \times n}
$$

Then:

$$
A \cdot B = \sum_{i=1}^{N} A_i B_i
$$

Each partial product $A_i B_i \in \mathbb{R}^{m \times n}$ is computed independently on GPU $i$. The full result requires **summation** across GPUs (all-reduce).

### 2.3 Neural Network Convention

In neural networks, the standard linear layer computes:

$$
Y = X \cdot W + b
$$

where:
- $X \in \mathbb{R}^{b \times s \times h_{\text{in}}}$ — input activations (batch $b$, sequence length $s$, hidden dimension $h_{\text{in}}$)
- $W \in \mathbb{R}^{h_{\text{in}} \times h_{\text{out}}}$ — weight matrix
- $b \in \mathbb{R}^{h_{\text{out}}}$ — bias vector
- $Y \in \mathbb{R}^{b \times s \times h_{\text{out}}}$ — output activations

---

## 3. Column-Linear (Column-Parallel) Sharding

### 3.1 Mechanism

Given $N$ GPUs (TP degree $= N$), partition the weight matrix $W$ along its **output (column) dimension**:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{h_{\text{in}} \times \frac{h_{\text{out}}}{N}}
$$

### 3.2 Forward Pass

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | **Broadcast** | Copy the full input $X \in \mathbb{R}^{b \times s \times h_{\text{in}}}$ to all $N$ GPUs |
| 2 | **Local matmul** | GPU $i$ computes $Y_i = X \cdot W_i \in \mathbb{R}^{b \times s \times \frac{h_{\text{out}}}{N}}$ |
| 3 | **All-gather** | Concatenate $\{Y_1, Y_2, \ldots, Y_N\}$ to reconstruct $Y \in \mathbb{R}^{b \times s \times h_{\text{out}}}$ |

### 3.3 Communication Primitives

- **Broadcast**: $O(h_{\text{in}} \cdot b \cdot s)$ — replicate full input
- **All-gather**: $O\!\left(\frac{h_{\text{out}}}{N} \cdot b \cdot s \cdot (N-1)\right)$ — reconstruct output

### 3.4 Key Property

Each GPU stores only $\frac{1}{N}$ of the weight columns and produces $\frac{1}{N}$ of the output columns. The **input** is replicated, but the **output activations** are sharded.

---

## 4. Row-Linear (Row-Parallel) Sharding

### 4.1 Mechanism

Partition the weight matrix $W$ along its **input (row) dimension**:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{h_{\text{in}}}{N} \times h_{\text{out}}}
$$

This requires a corresponding partition of the input along the hidden dimension:

$$
X = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}, \quad X_i \in \mathbb{R}^{b \times s \times \frac{h_{\text{in}}}{N}}
$$

### 4.2 Forward Pass

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | **Scatter** | Split and distribute input $X$ so GPU $i$ receives $X_i$ |
| 2 | **Local matmul** | GPU $i$ computes $Y_i = X_i \cdot W_i \in \mathbb{R}^{b \times s \times h_{\text{out}}}$ |
| 3 | **All-reduce** | Sum $\{Y_1, Y_2, \ldots, Y_N\}$ element-wise: $Y = \sum_{i=1}^{N} Y_i$ |

### 4.3 Communication Primitives

- **Scatter**: $O(b \cdot s \cdot h_{\text{in}})$ — distribute input chunks
- **All-reduce**: $O(b \cdot s \cdot h_{\text{out}})$ — sum partial results (equivalent to reduce-scatter + all-gather)

### 4.4 Key Property

Each GPU stores only $\frac{1}{N}$ of the weight rows. The **input** is sharded, and the **output** is full-sized but requires summation for correctness.

---

## 5. Tensor Parallelism in the Transformer Block

A standard Transformer decoder layer consists of two primary sub-blocks:

$$
\text{Transformer Layer} = \text{LayerNorm} \rightarrow \text{MHA} \rightarrow \text{LayerNorm} \rightarrow \text{MLP}
$$

### 5.1 MLP Block

The feedforward MLP in a Transformer consists of two linear projections with a nonlinearity (e.g., GeLU or SiLU):

$$
\text{MLP}(X) = \sigma(X W_1) W_2
$$

where:
- $W_1 \in \mathbb{R}^{h \times 4h}$ (up-projection, or gate projection)
- $W_2 \in \mathbb{R}^{4h \times h}$ (down-projection)
- $\sigma$ is an element-wise activation function (GeLU, SiLU, etc.)

**TP Strategy for MLP:**

| Layer | Parallelism Type | Rationale |
|-------|-----------------|-----------|
| $W_1$ (FC1 / up-projection) | **Column-linear** | Splits output hidden dim; input is broadcast (or already synced) |
| $\sigma$ (activation) | **Local** | Applied element-wise on sharded activations |
| $W_2$ (FC2 / down-projection) | **Row-linear** | Takes sharded input, produces full output; requires all-reduce |

**Why Column-Linear → Row-Linear (not vice versa)?**

If we used Row-Linear → Column-Linear, we would need an **intermediate all-reduce** between the two layers (to get correct row-linear output) followed by another communication for the column-linear input. The Column-Linear → Row-Linear ordering requires:
- **Forward pass**: One broadcast (often a no-op if inputs are already synced) + one all-reduce
- **Backward pass**: One all-reduce + one no-op

This yields **one all-reduce per MLP sub-block per direction** — the minimal communication.

### 5.2 Multi-Head Attention (MHA) Block

Multi-head attention computes:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_{n_h}) \cdot W^O
$$

where each attention head $i$ computes:

$$
\text{head}_i = \text{Attention}(X W^Q_i, \; X W^K_i, \; X W^V_i)
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

with $d_k = \frac{h}{n_h}$ being the per-head dimension.

**TP Strategy for MHA:**

| Component | Parallelism Type | Rationale |
|-----------|-----------------|-----------|
| $W^Q, W^K, W^V$ projections | **Column-linear** | Naturally partition along the $n_h$ (num\_heads) dimension; each GPU handles $\frac{n_h}{N}$ heads |
| Attention computation | **Local** | Each GPU computes attention for its assigned heads independently |
| $W^O$ (output projection) | **Row-linear** | Recombines head outputs; requires all-reduce |

**Natural interpretation**: With TP degree $N$, GPU $i$ computes attention for heads $\left\{\frac{(i-1) \cdot n_h}{N} + 1, \ldots, \frac{i \cdot n_h}{N}\right\}$.

### 5.3 Constraints on TP Degree

**Hard constraint:**

$$
N_{\text{TP}} \leq n_h \quad \text{(number of query attention heads)}
$$

since we partition along the head dimension, and each GPU must receive at least one head.

**GQA/MQA constraint**: In Grouped Query Attention:
- $n_q$ query heads, $n_{kv}$ key/value heads, with $n_q \geq n_{kv}$
- TP degree can go up to $n_q$, but when $N_{\text{TP}} > n_{kv}$, multiple GPUs share the same K/V heads
- Requires careful **K/V head replication** and synchronization

**Example — Llama-3 8B:**
- $n_q = 32$ query heads, $n_{kv} = 8$ key/value heads
- Maximum $N_{\text{TP}} = 32$ theoretically, but practical implementations typically use $N_{\text{TP}} \leq 8$ to avoid K/V synchronization overhead and inter-node communication

---

## 6. Communication Analysis in Tensor Parallelism

### 6.1 Communication Primitives Summary

| Primitive | Description | Volume |
|-----------|-------------|--------|
| **Broadcast** | Replicate data from one GPU to all | $O(D)$ |
| **Scatter** | Split data and distribute chunks | $O(D)$ |
| **All-gather** | Each GPU contributes a chunk; all receive the full tensor | $O\!\left(\frac{N-1}{N} \cdot D\right)$ |
| **Reduce-scatter** | Reduce (sum) and scatter result chunks | $O\!\left(\frac{N-1}{N} \cdot D\right)$ |
| **All-reduce** | Sum across all GPUs; all receive the full result | $O\!\left(2 \cdot \frac{N-1}{N} \cdot D\right)$ |

**Critical identity:**

$$
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
$$

### 6.2 Per-Transformer-Layer Communication (Vanilla TP)

Each Transformer layer requires:

| Sub-block | Forward | Backward |
|-----------|---------|----------|
| MHA | 1 × all-reduce | 1 × all-reduce |
| MLP | 1 × all-reduce | 1 × all-reduce |
| **Total** | **2 × all-reduce** | **2 × all-reduce** |

### 6.3 Conjugate Operator Pairs ($f$, $f^*$)

Tensor parallelism uses conjugate operator pairs that swap roles between forward and backward passes:

$$
\begin{aligned}
&\textbf{Forward:} \quad f = \text{no-op}, \quad f^* = \text{all-reduce} \\
&\textbf{Backward:} \quad f = \text{all-reduce}, \quad f^* = \text{no-op}
\end{aligned}
$$

**Rationale**: In the forward pass, inputs are already replicated across TP ranks (making $f$ a no-op), but outputs from row-linear layers must be summed ($f^*$ = all-reduce). In the backward pass, gradient flow reverses: gradients arriving at $f^*$ are already correct (no-op), while gradients at $f$ must be synchronized (all-reduce).

### 6.4 Critical Path and Overlap Limitations

Unlike ZeRO, where communication (all-gather of parameters) can be overlapped with computation, TP places communication **directly on the critical path**:

$$
T_{\text{layer}} = T_{\text{compute}} + T_{\text{all-reduce}}^{\text{exposed}}
$$

The all-reduce at the end of each MLP and MHA block **cannot** begin until the local matrix multiplication completes, and the subsequent LayerNorm **cannot** begin until the all-reduce finishes. This creates a **synchronization barrier** on every layer.

**Partial mitigation (Megatron-LM, Nanotron):** Overlap all-gather with FC1 computation using block/chunked matrix multiplication:
- Divide the weight into chunks
- As each chunk's all-gather completes, begin matmul for that chunk while the next chunk's all-gather proceeds asynchronously

**Advanced mitigation (Domino):** Novel scheduling techniques to maximize overlap of communication and computation within TP regions.

---

## 7. Scaling Behavior and Trade-offs

### 7.1 Intra-node vs. Inter-node Communication

| Regime | Interconnect | Bandwidth (typical) | Latency |
|--------|-------------|---------------------|---------|
| $N_{\text{TP}} \leq 8$ (intra-node) | NVLink / NVSwitch | 600–900 GB/s (bidirectional, per GPU) | ~µs |
| $N_{\text{TP}} > 8$ (inter-node) | InfiniBand / EFA | 50–400 GB/s (per node) | ~10–100 µs |

**Empirical observation:** Throughput drops **significantly** at $N_{\text{TP}} = 16$ (crossing node boundary) and **precipitously** at $N_{\text{TP}} = 32$.

### 7.2 Memory Reduction

For a model with $P$ total parameters, with TP degree $N$:

$$
\text{Parameters per GPU} = \frac{P}{N}
$$

$$
\text{Gradients per GPU} = \frac{P}{N}
$$

$$
\text{Optimizer states per GPU} = \frac{S \cdot P}{N}
$$

where $S$ is the optimizer state multiplier (e.g., $S = 12$ bytes per parameter for Adam in mixed precision: 4 bytes fp32 master weights + 4 bytes first moment + 4 bytes second moment).

**Activation memory** in TP regions:

$$
\text{Activation per GPU (TP region)} = \frac{b \cdot s \cdot h}{N} \quad \text{(intermediate activations)}
$$

**However**, operations like **LayerNorm** and **Dropout** still require the **full** activation tensor $(b, s, h)$, partially negating activation memory savings.

### 7.3 Throughput–Memory Trade-off

$$
\text{Effective throughput per GPU} \propto \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{communication}}}
$$

As $N$ increases:
- $T_{\text{compute}}$ decreases (less work per GPU)
- $T_{\text{communication}}$ increases (more participants, potentially crossing node boundaries)
- Net effect: **diminishing returns** beyond $N = 8$ (single node)

The **benefit** is that reduced memory per GPU allows larger batch sizes, which can compensate for per-GPU throughput loss at the **system** level.

---

## 8. Sequence Parallelism (SP)

### 8.1 Motivation

Even with TP, operations outside the attention and MLP sub-blocks — specifically **LayerNorm** and **Dropout** — require the **full hidden dimension** $h$ and therefore cannot be sharded along $h$. These operations still store activations of shape $(b, s, h)$ on each GPU, creating memory bottlenecks.

**Sequence Parallelism** shards these operations along the **sequence dimension** $s$ instead:

$$
\text{SP shard on GPU } i: \quad X_i^{*} \in \mathbb{R}^{b \times \frac{s}{N} \times h}
$$

### 8.2 Why LayerNorm Requires Full Hidden Dimension

LayerNorm computes:

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where the statistics are computed **across the hidden dimension** $h$:

$$
\mu = \frac{1}{h} \sum_{j=1}^{h} x_j, \qquad \sigma^2 = \frac{1}{h} \sum_{j=1}^{h} (x_j - \mu)^2
$$

Since $\mu$ and $\sigma^2$ require access to **all** $h$ elements, we **cannot** shard LayerNorm along $h$. However, different sequence positions are **independent** for LayerNorm, so we **can** shard along $s$.

### 8.3 Conjugate Operator Pairs for SP ($g$, $g^*$)

$$
\begin{aligned}
&\textbf{Forward:} \quad g = \text{all-gather}, \quad g^* = \text{reduce-scatter} \\
&\textbf{Backward:} \quad g = \text{reduce-scatter}, \quad g^* = \text{all-gather}
\end{aligned}
$$

These replace the $f$/$f^*$ operators at the **boundaries** between SP regions and TP regions.

### 8.4 Forward Pass Through a TP+SP Transformer Layer (Step-by-Step)

Consider an MLP sub-block with TP+SP on $N=2$ GPUs:

---

**Step 1: Initial LayerNorm (SP Region)**

Each GPU $i$ holds $X_i^* \in \mathbb{R}^{b \times \frac{s}{2} \times h}$ (sharded along sequence dimension).

LayerNorm is computed independently per sequence position:

$$
Y_i^* = \text{LayerNorm}(X_i^*) \in \mathbb{R}^{b \times \frac{s}{2} \times h}
$$

---

**Step 2: Transition SP → TP ($g$ = all-gather)**

Gather sequence chunks to reconstruct the full sequence on each GPU:

$$
Y = \text{AllGather}(Y_1^*, Y_2^*) \in \mathbb{R}^{b \times s \times h}
$$

This is necessary because the column-linear layer (FC1) needs the full input.

---

**Step 3: First Linear Layer — Column-Linear (TP Region)**

GPU $i$ holds column shard $W_1^{(i)} \in \mathbb{R}^{h \times \frac{4h}{N}}$:

$$
Z_i^* = \sigma(Y \cdot W_1^{(i)}) \in \mathbb{R}^{b \times s \times \frac{4h}{N}}
$$

where $\sigma$ is the activation function (GeLU/SiLU), applied element-wise.

---

**Step 4: Second Linear Layer — Row-Linear (TP Region)**

GPU $i$ holds row shard $W_2^{(i)} \in \mathbb{R}^{\frac{4h}{N} \times h}$:

$$
\hat{Y}_i = Z_i^* \cdot W_2^{(i)} \in \mathbb{R}^{b \times s \times h}
$$

---

**Step 5: Transition TP → SP ($g^*$ = reduce-scatter)**

$$
\hat{Y}_j^* = \text{ReduceScatter}(\hat{Y}_1, \hat{Y}_2) \in \mathbb{R}^{b \times \frac{s}{N} \times h}
$$

This simultaneously:
1. **Reduces** (sums) the partial results from the row-linear operation (required for correctness: $\hat{Y} = \sum_i \hat{Y}_i$)
2. **Scatters** the result along the sequence dimension (returning to SP sharding)

---

### 8.5 Activation Shape Summary Table

| Region | Vanilla TP | TP + SP |
|--------|-----------|---------|
| **Enter TP** (column-linear input) | $h$: full, $s$: full | $h$: full, $s$: all-gather to full |
| **TP region** (between FC1 and FC2) | $h$: sharded ($\frac{h}{N}$), $s$: full | $h$: sharded ($\frac{h}{N}$), $s$: full |
| **Exit TP** (row-linear output) | $h$: full + all-reduce, $s$: full | $h$: full + reduce-scatter, $s$: reduce-scatter to sharded |
| **SP region** (LayerNorm, Dropout) | $h$: full, $s$: full | $h$: full, $s$: sharded ($\frac{s}{N}$) |

### 8.6 Maximum Activation Size

**Vanilla TP:**

$$
\text{Max activation per GPU} = b \cdot s \cdot h
$$

(Full activations required in LayerNorm/Dropout regions.)

**TP + SP:**

$$
\text{Max activation per GPU} = \frac{b \cdot s \cdot h}{N}
$$

At every point in the computation, activations are sharded along **either** $h$ (in TP regions) **or** $s$ (in SP regions), never requiring the full $(b, s, h)$ tensor on any single GPU.

### 8.7 Communication Cost Equivalence

**Vanilla TP per transformer layer:** 2 × all-reduce (forward), 2 × all-reduce (backward)

**TP+SP per transformer layer:** 2 × all-gather + 2 × reduce-scatter (forward), 2 × reduce-scatter + 2 × all-gather (backward)

Since:

$$
\text{cost}(\text{all-reduce}) = \text{cost}(\text{all-gather}) + \text{cost}(\text{reduce-scatter})
$$

and TP+SP replaces each all-reduce with one all-gather and one reduce-scatter (at different points in the computation):

$$
\text{Total communication}(\text{TP+SP}) = \text{Total communication}(\text{Vanilla TP})
$$

**SP achieves strictly better activation memory with no additional communication overhead.**

### 8.8 Gradient Synchronization Notes

**LayerNorm weights in SP regions:** Since each TP rank sees **different** sequence positions (but the same LayerNorm parameters $\gamma, \beta$), gradients for $\gamma$ and $\beta$ will differ across ranks. An **all-reduce** of LayerNorm gradients is required during the backward pass:

$$
\nabla_\gamma = \text{AllReduce}\!\left(\nabla_\gamma^{(1)}, \nabla_\gamma^{(2)}, \ldots, \nabla_\gamma^{(N)}\right)
$$

This overhead is negligible since LayerNorm has only $2h$ parameters (compared to the full model's billions).

**Dropout in SP regions:** Random masks must be **synchronized** across TP ranks to maintain deterministic behavior. In practice, this is achieved by synchronizing the random seed across ranks:

$$
\text{seed}_{\text{dropout}}^{(i)} = \text{seed}_{\text{global}}, \quad \forall \; i \in \{1, \ldots, N\}
$$

---

## 9. Embedding Layer Treatment

The vocabulary embedding $E \in \mathbb{R}^{V \times h}$ (where $V$ is vocabulary size) is typically sharded along the vocabulary dimension (row-linear):

| Configuration | Sharding | Communication |
|---------------|----------|---------------|
| Vanilla TP | $h$: full (all-reduce for correctness), $s$: full | All-reduce |
| TP + SP | $h$: full (reduce-scatter for correctness), $s$: reduce-scatter to sharded | Reduce-scatter |

---

## 10. Limitations of TP + SP

| Limitation | Description |
|------------|-------------|
| **Sequence length scaling** | In the TP region, activations are $(b, s, \frac{h}{N})$; as $s$ grows, activation memory still scales linearly with $s$ in these regions |
| **Inter-node communication** | For $N_{\text{TP}} > 8$ (exceeding a single node), bandwidth drops from NVLink (~900 GB/s) to network interconnect (~100–400 GB/s), causing severe throughput degradation |
| **Critical path communication** | All-gather and reduce-scatter remain on the critical path and cannot be fully overlapped with computation |
| **Head count constraint** | $N_{\text{TP}} \leq n_h$; limits maximum parallelism degree |

**Solutions to these limitations:**
- **Context Parallelism (CP)**: Addresses sequence-length-induced activation memory blowup by sharding the attention computation across the sequence dimension
- **Pipeline Parallelism (PP)**: Addresses model-too-large-for-one-node by partitioning layers across nodes, avoiding the need for $N_{\text{TP}} > 8$

---

## 11. Complete Communication Pattern Summary

For a single Transformer layer with TP+SP:

$$
\boxed{
\begin{aligned}
&\textbf{Forward:} \quad \underbrace{g \;(\text{all-gather})}_{\text{SP} \to \text{TP}} \;\to\; \text{Column-Linear} \;\to\; \sigma \;\to\; \text{Row-Linear} \;\to\; \underbrace{g^* \;(\text{reduce-scatter})}_{\text{TP} \to \text{SP}} \\[6pt]
&\textbf{Backward:} \quad \underbrace{g^* \;(\text{all-gather})}_{\text{SP} \to \text{TP}} \;\to\; \nabla\text{Row-Linear} \;\to\; \nabla\sigma \;\to\; \nabla\text{Column-Linear} \;\to\; \underbrace{g \;(\text{reduce-scatter})}_{\text{TP} \to \text{SP}}
\end{aligned}
}
$$

This pattern repeats for both the **MHA** and **MLP** sub-blocks within each Transformer layer, yielding **4 communication operations per layer per pass** (2 for MHA + 2 for MLP).