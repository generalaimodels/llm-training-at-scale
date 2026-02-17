

# Tensor Parallelism (TP) and Sequence Parallelism (SP)

---

## 1. Motivation: Why Tensor Parallelism?

ZeRO (Zero Redundancy Optimizer) successfully shards **parameters**, **gradients**, and **optimizer states** across GPUs. However, **activation memory** — the intermediate tensors produced during the forward pass — remains replicated on every device. As model size and sequence length grow, activation memory dominates the per-GPU memory budget, creating an insurmountable bottleneck.

**Tensor Parallelism (TP)** resolves this by sharding not only parameters, gradients, and optimizer states but also **activations** — and critically, it does so **without requiring a full gather of all shards before computation**. Instead, TP exploits the algebraic structure of matrix multiplication to distribute computation natively across devices.

---

## 2. Mathematical Foundations of Tensor Parallelism

Tensor parallelism rests on two fundamental decomposition properties of matrix multiplication. Given matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:

### 2.1 Column-wise Decomposition (Splitting $B$ by Columns)

Partition $B$ into $N$ column blocks:

$$
B = \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{k \times (n/N)}
$$

Then:

$$
A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots & AB_N \end{bmatrix}
$$

Each partial product $AB_i \in \mathbb{R}^{m \times (n/N)}$ can be computed **independently** on GPU $i$, and the final result is obtained by **concatenating** (all-gather) the partial outputs along the column dimension.

### 2.2 Row-wise Decomposition (Splitting $B$ by Rows and $A$ by Columns)

Partition $A$ into $N$ column blocks and $B$ into $N$ row blocks:

$$
A = \begin{bmatrix} A_1 & A_2 & \cdots & A_N \end{bmatrix}, \quad A_i \in \mathbb{R}^{m \times (k/N)}
$$

$$
B = \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{(k/N) \times n}
$$

Then:

$$
A \cdot B = \sum_{i=1}^{N} A_i B_i
$$

Each partial product $A_i B_i \in \mathbb{R}^{m \times n}$ is computed independently on GPU $i$, and the final result is obtained by **summing** (all-reduce) the partial outputs.

### 2.3 Neural Network Notation

In a neural network linear layer, the operation is expressed as:

$$
Y = X \cdot W
$$

where:
- $X \in \mathbb{R}^{b \times s \times h}$ — input activations (batch size $b$, sequence length $s$, hidden dimension $h$)
- $W \in \mathbb{R}^{h \times h'}$ — weight matrix of the linear layer
- $Y \in \mathbb{R}^{b \times s \times h'}$ — output activations

---

## 3. Column-Linear Parallelism

### 3.1 Procedure

Given $N$ GPUs and weight matrix $W \in \mathbb{R}^{h \times h'}$:

1. **Broadcast** (or replicate) the full input $X$ to all $N$ GPUs.
2. **Shard** $W$ along its **column** dimension:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{h \times (h'/N)}
$$

3. Each GPU $i$ computes:

$$
Y_i = X \cdot W_i \in \mathbb{R}^{b \times s \times (h'/N)}
$$

4. **All-gather** across GPUs to reconstruct:

$$
Y = \begin{bmatrix} Y_1 & Y_2 & \cdots & Y_N \end{bmatrix} \in \mathbb{R}^{b \times s \times h'}
$$

### 3.2 Communication Primitives

| Step | Operation | Communication |
|------|-----------|---------------|
| Input distribution | **Broadcast** | $X$ replicated to all ranks |
| Computation | Local matmul | No communication |
| Output combination | **All-Gather** | Concatenate partial outputs |

---

## 4. Row-Linear Parallelism

### 4.1 Procedure

Given $N$ GPUs and weight matrix $W \in \mathbb{R}^{h \times h'}$:

1. **Scatter** the input $X$ along the hidden (or appropriate) dimension:

$$
X = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}, \quad X_i \in \mathbb{R}^{b \times s \times (h/N)}
$$

2. **Shard** $W$ along its **row** dimension:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{(h/N) \times h'}
$$

3. Each GPU $i$ computes:

$$
Y_i = X_i \cdot W_i \in \mathbb{R}^{b \times s \times h'}
$$

4. **All-reduce** (sum) across GPUs:

$$
Y = \sum_{i=1}^{N} Y_i \in \mathbb{R}^{b \times s \times h'}
$$

### 4.2 Communication Primitives

| Step | Operation | Communication |
|------|-----------|---------------|
| Input distribution | **Scatter** | $X$ split across ranks |
| Computation | Local matmul | No communication |
| Output combination | **All-Reduce** | Sum partial outputs |

---

## 5. Tensor Parallelism in a Transformer Block

A standard Transformer decoder layer comprises two sub-blocks:

1. **Multi-Head Attention (MHA)** block
2. **Feedforward MLP** block

Each is amenable to tensor parallelism due to the existence of **naturally independent dimensions**.

### 5.1 Feedforward MLP Block

A typical MLP in a Transformer consists of two linear projections with a nonlinearity:

$$
\text{MLP}(X) = \text{GELU}(X W_1) \cdot W_2
$$

where $W_1 \in \mathbb{R}^{h \times 4h}$ (up-projection) and $W_2 \in \mathbb{R}^{4h \times h}$ (down-projection).

**Optimal TP strategy for MLP:**

| Layer | Parallelism Type | Rationale |
|-------|-----------------|-----------|
| $W_1$ (FC1) | **Column-linear** | Split output dimension; GELU applied independently per shard |
| $W_2$ (FC2) | **Row-linear** | Accepts sharded input from FC1; produces full output via all-reduce |

**Why this ordering (column → row) is superior to (row → column):**

- Column-linear first requires only a **broadcast** (or no-op if inputs are already synchronized) to distribute $X$.
- Row-linear second requires an **all-reduce** to combine results.
- Total: **1 broadcast + 1 all-reduce** per MLP block in forward pass.
- The reverse ordering (row → column) would require an **intermediate all-reduce** between the two linear layers plus additional communication, making it strictly less efficient.

The forward computation on GPU $i$ proceeds as:

$$
Z_i = \text{GELU}\left(X \cdot W_1^{(i)}\right), \quad W_1^{(i)} \in \mathbb{R}^{h \times (4h/N)}
$$

$$
Y_i = Z_i \cdot W_2^{(i)}, \quad W_2^{(i)} \in \mathbb{R}^{(4h/N) \times h}
$$

$$
Y = \sum_{i=1}^{N} Y_i \quad \text{(all-reduce)}
$$

The GELU nonlinearity is applied **element-wise** within each shard, requiring no cross-GPU communication — this is precisely why column-linear sharding of $W_1$ is essential (sharding along the output dimension preserves the independence of the nonlinearity).

### 5.2 Multi-Head Attention (MHA) Block

The attention mechanism computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and $d_k = h / n_{\text{heads}}$ is the per-head dimension.

**Natural parallelism:** Each attention head operates **independently**. With $n_{\text{heads}}$ attention heads distributed across $N$ GPUs, each GPU handles $n_{\text{heads}} / N$ heads.

| Projection | Parallelism Type | Rationale |
|------------|-----------------|-----------|
| $W_Q, W_K, W_V$ | **Column-linear** | Each column shard corresponds to a subset of attention heads |
| $W_O$ (output projection) | **Row-linear** | Accepts concatenated head outputs (already sharded); produces full hidden via all-reduce |

The communication pattern is identical to the MLP block:
- **Forward:** 1 broadcast (or no-op) + 1 all-reduce per attention block
- **Backward:** Conjugate operations

### 5.3 Grouped Query Attention (GQA) Considerations

In GQA, the number of key/value heads $n_{\text{kv\_heads}}$ is smaller than the number of query heads $n_{\text{attention\_heads}}$:

$$
n_{\text{attention\_heads}} \geq n_{\text{kv\_heads}}
$$

**Constraint on TP degree:**

$$
\text{TP} \leq n_{\text{attention\_heads}}
$$

When $\text{TP} > n_{\text{kv\_heads}}$, K/V heads must be **replicated** or carefully **synchronized** across TP ranks. For example, Llama-3 8B has:
- $n_{\text{attention\_heads}} = 32$
- $n_{\text{kv\_heads}} = 8$
- Maximum TP = 32, but for $\text{TP} > 8$, K/V heads require cross-rank synchronization

---

## 6. Communication Analysis of Tensor Parallelism

### 6.1 Communication Primitives in the Critical Path

For each Transformer decoder layer (MHA + MLP), the forward pass requires:

$$
\text{Forward: } 2 \times \text{all-reduce} \quad \text{(one for MHA, one for MLP)}
$$

$$
\text{Backward: } 2 \times \text{all-reduce} \quad \text{(conjugate operations)}
$$

These all-reduce operations sit **directly on the critical path** of computation — they cannot be trivially overlapped with compute because subsequent operations (e.g., LayerNorm, residual addition) depend on the synchronized result.

> **Critical path**: The longest chain of sequentially dependent operations determining the minimum wall-clock time for a forward or backward pass.

### 6.2 Communication Volume

For an all-reduce of a tensor of size $M$ across $N$ GPUs, the total communication volume (using ring all-reduce) is:

$$
V_{\text{all-reduce}} = 2 \cdot \frac{N-1}{N} \cdot M
$$

For the MLP block, $M = b \cdot s \cdot h$, giving:

$$
V_{\text{MLP}} = 2 \cdot \frac{N-1}{N} \cdot b \cdot s \cdot h \quad \text{per layer (forward)}
$$

### 6.3 Scaling Behavior and Interconnect Dependence

| TP Degree | Interconnect | Observed Behavior |
|-----------|-------------|-------------------|
| $\text{TP} \leq 8$ | **NVLink** (intra-node, ~900 GB/s bidirectional on A100/H100) | High throughput; communication overhead manageable |
| $\text{TP} = 16$ | **Inter-node** (InfiniBand/EFA, ~100–400 GB/s) | Significant throughput degradation |
| $\text{TP} = 32$ | Inter-node | Steep decline; communication dominates compute |

**Practical guideline:**

$$
\text{TP degree} \leq \text{GPUs per node} \quad \text{(typically 8)}
$$

---

## 7. Memory Benefits of Tensor Parallelism

With TP degree $N$, the per-GPU memory for a linear layer with weight $W \in \mathbb{R}^{h \times h'}$ is:

### 7.1 Parameters

$$
\text{Params per GPU} = \frac{h \times h'}{N}
$$

### 7.2 Gradients

$$
\text{Gradients per GPU} = \frac{h \times h'}{N}
$$

### 7.3 Optimizer States (Adam)

Adam maintains first moment $m$ and second moment $v$, each the same size as the parameters:

$$
\text{Optimizer states per GPU} = \frac{2 \times h \times h'}{N} \quad \text{(in fp32, so } \frac{2 \times 4 \times h \times h'}{N} \text{ bytes)}
$$

### 7.4 Activations (Partial Benefit)

Intermediate activations within TP regions are sharded:

$$
\text{Activation per GPU (TP region)} = \frac{b \cdot s \cdot h'}{N} \quad \text{(for column-linear output)}
$$

However, operations like **LayerNorm** and **dropout** still require the **full** activation tensor $b \times s \times h$, limiting the activation memory savings. This is the precise limitation that **Sequence Parallelism** addresses.

---

## 8. Sequence Parallelism (SP)

### 8.1 Core Idea

Sequence parallelism shards the activations along the **sequence dimension** $s$ for operations that are **outside** the tensor-parallel regions — specifically **LayerNorm** and **dropout**.

These operations require the **full hidden dimension** $h$ (e.g., LayerNorm computes statistics across $h$), so they cannot be sharded along $h$. However, they operate **independently** across sequence positions, making sharding along $s$ natural.

### 8.2 LayerNorm Definition

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

$$
\mu = \frac{1}{h} \sum_{j=1}^{h} x_j, \quad \sigma^2 = \frac{1}{h} \sum_{j=1}^{h} (x_j - \mu)^2
$$

Both $\mu$ and $\sigma^2$ are computed **across the hidden dimension** $h$ for each sequence position independently. Therefore:

- Cannot shard along $h$ (would produce incorrect statistics)
- **Can** shard along $s$ (each position is independent)

### 8.3 Transition Operations: Conjugate Pairs $f/f^*$ and $g/g^*$

The interplay between TP regions and SP regions requires carefully designed communication operators:

#### TP Region Operators ($f$ and $f^*$)

| Pass | $f$ | $f^*$ |
|------|-----|-------|
| **Forward** | No-op (activations already replicated) | All-reduce (synchronize partial results) |
| **Backward** | All-reduce (synchronize gradients) | No-op (gradients already replicated) |

$f$ and $f^*$ are **conjugate pairs**: when one is a no-op, the other is an all-reduce, and vice versa across forward and backward passes.

#### SP ↔ TP Transition Operators ($g$ and $g^*$)

| Pass | $g$ | $g^*$ |
|------|-----|-------|
| **Forward** | All-gather (reconstruct full sequence for TP) | Reduce-scatter (shard sequence for SP) |
| **Backward** | Reduce-scatter (distribute gradients) | All-gather (reconstruct gradients) |

$g$ and $g^*$ are also conjugate pairs.

### 8.4 Data Flow Through a Transformer Layer with TP+SP

Consider a two-GPU setup ($N = 2$) with input $X \in \mathbb{R}^{b \times s \times h}$:

**Step 1: LayerNorm (SP region)**

Each GPU holds $X_i^* \in \mathbb{R}^{b \times (s/N) \times h}$ (sharded along sequence).

Each GPU computes LayerNorm independently on its chunk:

$$
Y_i^* = \text{LayerNorm}(X_i^*) \in \mathbb{R}^{b \times (s/N) \times h}
$$

**Step 2: SP → TP transition ($g$: all-gather)**

Reconstruct the full sequence on each GPU:

$$
Y = \text{AllGather}(Y_1^*, Y_2^*, \ldots, Y_N^*) \in \mathbb{R}^{b \times s \times h}
$$

**Step 3: Column-linear / FC1 (TP region)**

Each GPU computes with its column shard $W_1^{(i)}$:

$$
Z_i^* = \text{GELU}\left(Y \cdot W_1^{(i)}\right) \in \mathbb{R}^{b \times s \times (4h/N)}
$$

**Step 4: Row-linear / FC2 (TP region)**

Each GPU computes with its row shard $W_2^{(i)}$:

$$
O_i = Z_i^* \cdot W_2^{(i)} \in \mathbb{R}^{b \times s \times h}
$$

**Step 5: TP → SP transition ($g^*$: reduce-scatter)**

$$
O_j^* = \text{ReduceScatter}(O_1, O_2, \ldots, O_N) \in \mathbb{R}^{b \times (s/N) \times h}
$$

This simultaneously:
1. **Reduces** (sums) the partial row-linear outputs for correctness
2. **Scatters** the result along the sequence dimension for the subsequent SP region

### 8.5 Activation Memory Comparison

| Configuration | Maximum Activation Size per GPU |
|---------------|---------------------------------|
| No parallelism | $b \cdot s \cdot h$ |
| TP only | $b \cdot s \cdot h$ (LayerNorm/dropout still need full tensor) |
| TP + SP | $\displaystyle\frac{b \cdot s \cdot h}{\text{TP}}$ |

With TP+SP, at **every** point in the computation, activations are sharded along **either** the hidden dimension (in TP regions) **or** the sequence dimension (in SP regions), ensuring:

$$
\text{Max activation per GPU} = \frac{b \cdot s \cdot h}{\text{TP}}
$$

### 8.6 Summary Table: Activation Shape Throughout Forward Pass

| Region | TP Only | TP + SP |
|--------|---------|---------|
| **Enter TP (column-linear)** | $h$: sharded, $s$: full | $h$: sharded, $s$: all-gather → full |
| **TP region (between linears)** | $h$: sharded, $s$: full | $h$: sharded, $s$: full |
| **Exit TP (row-linear)** | $h$: full (all-reduce), $s$: full | $h$: full (reduce-scatter), $s$: reduce-scatter → sharded |
| **SP region (LayerNorm, dropout)** | $h$: full, $s$: full | $h$: full, $s$: sharded |
| **Embedding layer (row-linear)** | $h$: full (all-reduce), $s$: full | $h$: full (reduce-scatter), $s$: reduce-scatter → sharded |

---

## 9. Communication Equivalence: TP vs. TP+SP

### 9.1 Per-Layer Communication Count

| Method | Forward Operations per Layer |
|--------|------------------------------|
| TP only | 2 × all-reduce |
| TP + SP | 2 × all-gather + 2 × reduce-scatter |

### 9.2 Why They Are Equivalent

A single all-reduce can be decomposed into:

$$
\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}
$$

Therefore, 2 all-reduce operations have the same communication volume as 2 all-gather + 2 reduce-scatter operations:

$$
2 \times V_{\text{all-reduce}} = 2 \times (V_{\text{reduce-scatter}} + V_{\text{all-gather}}) = 2 \times V_{\text{all-gather}} + 2 \times V_{\text{reduce-scatter}}
$$

**TP+SP introduces zero additional communication overhead relative to vanilla TP**, while providing strictly superior activation memory savings.

### 9.3 Backward Pass

The backward pass uses the **conjugate** of each forward operation:

$$
\text{no-op} \longleftrightarrow \text{all-reduce}, \quad \text{all-gather} \longleftrightarrow \text{reduce-scatter}
$$

Thus, backward communication cost is also identical between TP and TP+SP.

---

## 10. Gradient Synchronization Notes

### 10.1 LayerNorm Weights

In TP+SP, after an all-gather, each TP rank processes the **same** activations through the TP region. This means:

- LayerNorm weights see the **same forward activations** on every rank after all-gather.
- Therefore, LayerNorm weight gradients are **naturally identical** across TP ranks.
- **No all-reduce is needed** for LayerNorm weight gradients during the backward pass.

However, in the SP region, each rank processes a **different** sequence chunk:

- LayerNorm gradients **differ** across ranks.
- An **all-reduce** of LayerNorm gradients is required, but this is negligible since LayerNorm has only $2h$ parameters ($\gamma$ and $\beta$).

### 10.2 Dropout Synchronization

Dropout in SP regions operates on different sequence chunks per rank. To ensure **deterministic** behavior and reproducibility:

$$
\text{Seed}_{\text{dropout}} \text{ must be synchronized across all TP ranks}
$$

---

## 11. Remaining Limitations of TP + SP

| Limitation | Description | Solution |
|------------|-------------|----------|
| **Sequence length scaling** | Within TP regions, activations still have full sequence length $s$; as $s$ grows, memory explodes in attention computation | **Context Parallelism** (Ring Attention) |
| **Model size exceeding intra-node capacity** | TP > 8 requires inter-node communication with severe throughput penalty | **Pipeline Parallelism** (PP) |
| **Communication on critical path** | All-reduce / reduce-scatter / all-gather cannot be fully overlapped with compute | Active research (e.g., **Domino**, async block-matmul strategies) |

---

## 12. Practical Guidelines Summary

$$
\boxed{
\begin{aligned}
&\text{1. } \text{TP degree} \leq \text{GPUs per node (typically 8)} \\
&\text{2. } \text{TP degree} \leq n_{\text{attention\_heads}} \\
&\text{3. Always use SP with TP (zero extra communication cost, strict memory gain)} \\
&\text{4. For GQA with TP} > n_{\text{kv\_heads}}\text{, implement K/V head replication} \\
&\text{5. For sequence lengths beyond activation budget, add Context Parallelism} \\
&\text{6. For models too large for single-node TP, add Pipeline Parallelism}
\end{aligned}
}
$$