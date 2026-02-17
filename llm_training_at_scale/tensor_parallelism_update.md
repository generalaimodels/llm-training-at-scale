

# Tensor Parallelism (TP) and Sequence Parallelism (SP)

---

## 1. Motivation: Why Tensor Parallelism?

ZeRO (Zero Redundancy Optimizer) shards **parameters**, **gradients**, and **optimizer states** across data-parallel ranks. However, ZeRO does **not** shard **activations** — the intermediate tensors produced during the forward pass. As model size and sequence length grow, activation memory dominates the per-GPU memory budget.

**Tensor Parallelism (TP)** addresses this by partitioning individual weight tensors (and consequently their activations, gradients, and optimizer states) across multiple GPUs **within a single layer**, so that each GPU computes only a **slice** of every matrix multiplication. Crucially, TP does **not** require gathering the full tensors before computation — each device operates on its local shard and communicates only partial results.

---

## 2. Mathematical Foundations of Tensor Parallelism

Tensor parallelism is grounded in two fundamental decomposition properties of matrix multiplication. Given matrices $ A \in \mathbb{R}^{m \times k} $ and $ B \in \mathbb{R}^{k \times n} $, the product $ C = A \cdot B \in \mathbb{R}^{m \times n} $ can be decomposed in two distinct ways.

### 2.1 Column-wise Decomposition (Splitting $B$ by Columns)

Partition $ B $ into $ N $ column blocks:

$$
B = \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{k \times (n/N)}
$$

Then:

$$
A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots & B_N \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots & AB_N \end{bmatrix}
$$

Each partial product $ AB_i \in \mathbb{R}^{m \times (n/N)} $ can be computed **independently** on GPU $ i $. The final result is obtained by **concatenating** (all-gather) the partial results along the column dimension.

**Key Property:** Each GPU requires the **full** input $ A $ but only a **column shard** $ B_i $ of the weight matrix.

### 2.2 Row-wise Decomposition (Splitting $B$ by Rows)

Partition $ B $ into $ N $ row blocks and correspondingly partition $ A $ into $ N $ column blocks:

$$
A = \begin{bmatrix} A_1 & A_2 & \cdots & A_N \end{bmatrix}, \quad A_i \in \mathbb{R}^{m \times (k/N)}
$$

$$
B = \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix}, \quad B_i \in \mathbb{R}^{(k/N) \times n}
$$

Then:

$$
A \cdot B = \begin{bmatrix} A_1 & A_2 & \cdots & A_N \end{bmatrix} \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_N \end{bmatrix} = \sum_{i=1}^{N} A_i B_i
$$

Each partial product $ A_i B_i \in \mathbb{R}^{m \times n} $ is computed on GPU $ i $. The final result requires **summation** (all-reduce) across all GPUs.

**Key Property:** Each GPU requires a **column shard** $ A_i $ of the input and a **row shard** $ B_i $ of the weight matrix.

---

## 3. Neural Network Notation Convention

In neural network layers, matrix multiplication is expressed as:

$$
Y = X \cdot W
$$

where:

| Symbol | Meaning | Typical Shape |
|--------|---------|---------------|
| $ X $ | Input activations | $ (b, s, h) $ |
| $ W $ | Weight matrix of a Linear layer | $ (h, h') $ |
| $ Y $ | Output activations | $ (b, s, h') $ |
| $ b $ | Batch size | — |
| $ s $ | Sequence length | — |
| $ h $ | Hidden dimension (input) | — |
| $ h' $ | Hidden dimension (output) | — |

---

## 4. Column-Linear Parallelism (Column-wise Sharding)

### 4.1 Procedure

Given $ N $ GPUs (TP degree $ = N $):

1. **Broadcast** (or replicate) the full input $ X \in \mathbb{R}^{b \times s \times h} $ to every GPU.
2. **Partition** the weight matrix $ W \in \mathbb{R}^{h \times h'} $ along the **column** (output) dimension:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{h \times (h'/N)}
$$

3. Each GPU $ i $ computes:

$$
Y_i = X \cdot W_i, \quad Y_i \in \mathbb{R}^{b \times s \times (h'/N)}
$$

4. **All-Gather** the partial outputs to reconstruct:

$$
Y = \begin{bmatrix} Y_1 & Y_2 & \cdots & Y_N \end{bmatrix} \in \mathbb{R}^{b \times s \times h'}
$$

### 4.2 Communication Primitives

| Operation | Direction | Primitive |
|-----------|-----------|-----------|
| Distribute input | Forward | **Broadcast** (or identity if already replicated) |
| Combine output | Forward | **All-Gather** |

### 4.3 Memory Per GPU

- Weight memory: $ \frac{h \times h'}{N} $ parameters (instead of $ h \times h' $)
- Activation memory for $ Y_i $: $ b \times s \times \frac{h'}{N} $ (sharded)
- Input $ X $: $ b \times s \times h $ (replicated — **not** sharded)

---

## 5. Row-Linear Parallelism (Row-wise Sharding)

### 5.1 Procedure

Given $ N $ GPUs:

1. **Scatter** the input $ X \in \mathbb{R}^{b \times s \times h} $ along the hidden dimension:

$$
X = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}, \quad X_i \in \mathbb{R}^{b \times s \times (h/N)}
$$

2. **Partition** the weight matrix $ W \in \mathbb{R}^{h \times h'} $ along the **row** (input) dimension:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{(h/N) \times h'}
$$

3. Each GPU $ i $ computes:

$$
Y_i = X_i \cdot W_i, \quad Y_i \in \mathbb{R}^{b \times s \times h'}
$$

4. **All-Reduce** (sum) to obtain the final result:

$$
Y = \sum_{i=1}^{N} Y_i \in \mathbb{R}^{b \times s \times h'}
$$

### 5.2 Communication Primitives

| Operation | Direction | Primitive |
|-----------|-----------|-----------|
| Distribute input | Forward | **Scatter** (split along hidden dim) |
| Combine output | Forward | **All-Reduce** (summation) |

### 5.3 Memory Per GPU

- Weight memory: $ \frac{h \times h'}{N} $ parameters
- Input activation: $ b \times s \times \frac{h}{N} $ (sharded)
- Output $ Y_i $: $ b \times s \times h' $ (full-sized, prior to reduction)

---

## 6. Four Distributed Communication Primitives

The following collective operations are used throughout TP and SP:

| Primitive | Description | Data Movement |
|-----------|-------------|---------------|
| **Broadcast** | Replicate data from one rank to all ranks | $ 1 \to N $ |
| **Scatter** | Split data and distribute disjoint chunks to ranks | $ 1 \to N $ (partitioned) |
| **All-Gather** | Each rank contributes a shard; all ranks receive the concatenation | $ N \to N $ (concat) |
| **All-Reduce** | Each rank contributes a tensor; all ranks receive the element-wise sum | $ N \to N $ (sum) |
| **Reduce-Scatter** | Reduce (sum) across ranks, then scatter the result | $ N \to N $ (sum + partition) |

**Critical identity** (used to show TP ↔ SP communication equivalence):

$$
\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}
$$

This decomposition is implemented efficiently via the **Ring AllReduce** algorithm.

---

## 7. Tensor Parallelism in a Transformer Block

A standard Transformer decoder layer consists of two primary sub-blocks:

1. **Multi-Head Attention (MHA)**
2. **Feedforward MLP**

Each is preceded by a **LayerNorm** and followed by a **residual connection**. The computation graph of a single layer is:

$$
\begin{aligned}
\hat{X} &= \text{LayerNorm}(X) \\
X' &= X + \text{MHA}(\hat{X}) \\
\hat{X}' &= \text{LayerNorm}(X') \\
X'' &= X' + \text{MLP}(\hat{X}')
\end{aligned}
$$

### 7.1 Tensor Parallelism in the MLP Block

A standard Transformer MLP consists of two linear projections with a nonlinearity (e.g., GeLU) in between:

$$
\text{MLP}(X) = \text{GeLU}(X W_1 + b_1) \, W_2 + b_2
$$

where $ W_1 \in \mathbb{R}^{h \times 4h} $ (up-projection) and $ W_2 \in \mathbb{R}^{4h \times h} $ (down-projection).

**TP strategy for MLP — Column-Linear first, then Row-Linear:**

1. **$ W_1 $ is sharded column-wise** across $ N $ GPUs:

$$
W_1 = \begin{bmatrix} W_1^{(1)} & W_1^{(2)} & \cdots & W_1^{(N)} \end{bmatrix}, \quad W_1^{(i)} \in \mathbb{R}^{h \times (4h/N)}
$$

- Input $ X \in \mathbb{R}^{b \times s \times h} $ is **broadcast** (replicated) to all GPUs.
- GPU $ i $ computes: $ Z_i = \text{GeLU}(X \cdot W_1^{(i)}) \in \mathbb{R}^{b \times s \times (4h/N)} $

2. **$ W_2 $ is sharded row-wise** across $ N $ GPUs:

$$
W_2 = \begin{bmatrix} W_2^{(1)} \\ W_2^{(2)} \\ \vdots \\ W_2^{(N)} \end{bmatrix}, \quad W_2^{(i)} \in \mathbb{R}^{(4h/N) \times h}
$$

- GPU $ i $ computes: $ Y_i = Z_i \cdot W_2^{(i)} \in \mathbb{R}^{b \times s \times h} $

3. **All-Reduce** to sum partial results:

$$
Y = \sum_{i=1}^{N} Y_i \in \mathbb{R}^{b \times s \times h}
$$

**Why Column → Row and not Row → Column?**

The Column-Linear → Row-Linear ordering requires only:
- One **broadcast** (or no-op, since inputs are already synced) at the start
- One **all-reduce** at the end

The reverse ordering (Row → Column) would require an **all-reduce between** the two linear layers (intermediate synchronization), which adds a communication step on the critical path.

**Communication per MLP sub-block (forward pass):**

| Step | Primitive | Volume |
|------|-----------|--------|
| Input distribution | Broadcast / no-op | — |
| After row-linear | All-Reduce | $ b \times s \times h $ |

### 7.2 Tensor Parallelism in the Multi-Head Attention (MHA) Block

Multi-head attention computes:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_{n_h}) \, W^O
$$

where each head is:

$$
\text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
$$

and the projections are:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

with $ W^Q, W^K, W^V \in \mathbb{R}^{h \times h} $ and $ W^O \in \mathbb{R}^{h \times h} $, where $ h = n_h \cdot d_k $.

**TP strategy for MHA:**

1. **$ W^Q, W^K, W^V $ are sharded column-wise** along the $ n_h $ (number of heads) dimension. Each GPU $ i $ holds weights for a subset of attention heads:

$$
W^Q = \begin{bmatrix} W^Q_{(1)} & W^Q_{(2)} & \cdots & W^Q_{(N)} \end{bmatrix}
$$

Similarly for $ W^K $ and $ W^V $. GPU $ i $ computes attention for heads assigned to it:

$$
Q_i = X \cdot W^Q_{(i)}, \quad K_i = X \cdot W^K_{(i)}, \quad V_i = X \cdot W^V_{(i)}
$$

$$
\text{Attn}_i = \text{Softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{b \times s \times (h/N)}
$$

2. **$ W^O $ is sharded row-wise**, so GPU $ i $ holds $ W^O_{(i)} \in \mathbb{R}^{(h/N) \times h} $.

$$
Y_i = \text{Attn}_i \cdot W^O_{(i)} \in \mathbb{R}^{b \times s \times h}
$$

3. **All-Reduce** to sum partial results:

$$
Y = \sum_{i=1}^{N} Y_i
$$

**Natural interpretation:** Each GPU computes the full attention mechanism for its **subset of heads** — since attention heads operate independently, this decomposition introduces **zero approximation error**.

### 7.3 Constraint on TP Degree

$$
\text{TP degree} \leq n_h \quad (\text{number of attention heads})
$$

This is because the QKV projections are sharded along the $ n_h $ dimension. Each GPU must receive at least one complete attention head.

**Grouped Query Attention (GQA) consideration:**

In GQA, we have $ n_q $ query heads but only $ n_{kv} $ key/value heads, with $ n_q \geq n_{kv} $. The TP degree can theoretically reach $ n_q $, but when $ \text{TP} > n_{kv} $, multiple GPUs share the same K/V heads, requiring careful synchronization.

**Example — Llama-3 8B:**

| Parameter | Value |
|-----------|-------|
| $ n_q $ (query heads) | 32 |
| $ n_{kv} $ (KV heads) | 8 |
| Max TP degree (theoretical) | 32 |
| Practical constraint | K/V synchronization needed for TP > 8 |

---

## 8. Communication Overhead and Critical Path Analysis

### 8.1 Forward Pass Communication

In the forward pass of a single Transformer layer with TP applied to both MHA and MLP:

| Sub-block | Communication | Primitive | Cannot Overlap |
|-----------|---------------|-----------|----------------|
| MHA output | Combine partial results | All-Reduce | ✗ (on critical path) |
| MLP output | Combine partial results | All-Reduce | ✗ (on critical path) |

**Total per layer:** 2 × All-Reduce operations.

The **critical path** is the longest chain of sequential dependencies determining minimum forward pass time:

$$
T_{\text{forward}} = T_{\text{compute}} + T_{\text{comm, exposed}}
$$

The all-reduce sits **directly** on this critical path because the subsequent LayerNorm requires the **full** (un-sharded) activation tensor. Therefore, the all-reduce **cannot** be fully overlapped with computation.

**Partial overlap techniques** (e.g., Megatron-LM): Overlap the all-gather portion of the all-reduce with the initial portion of the next matrix multiplication using block/tile-level pipelining with asynchronous communication.

### 8.2 Scaling Behavior

TP communication cost per all-reduce for a tensor of size $ D $, using the Ring AllReduce algorithm across $ N $ GPUs:

$$
T_{\text{all-reduce}}(D, N) = 2(N-1) \cdot \alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{D}{\beta}
$$

where:
- $ \alpha $ = latency per message
- $ \beta $ = bandwidth (bytes/second)
- $ D $ = total data size in bytes
- The factor $ 2 $ accounts for the reduce-scatter + all-gather phases

**Intra-node vs. inter-node:**

| Configuration | Interconnect | Typical Bandwidth |
|---------------|-------------|-------------------|
| TP ≤ 8 (within node) | NVLink | ~900 GB/s (bidirectional, H100) |
| TP > 8 (across nodes) | InfiniBand / EFA | ~50–400 GB/s |

The **order-of-magnitude bandwidth drop** when crossing node boundaries explains the severe throughput degradation observed at TP = 16 and beyond.

---

## 9. Memory Analysis Under Tensor Parallelism

For a model with $ P $ total parameters, using mixed precision (BF16 weights, FP32 optimizer states), TP degree $ N $:

### 9.1 Model State Memory (per GPU)

$$
M_{\text{model}} = \frac{P}{N} \cdot \left(2 + 2 + (4 + 4 + 4)\right) = \frac{16P}{N} \text{ bytes}
$$

where:
- 2 bytes for BF16 parameters
- 2 bytes for BF16 gradients
- 4 + 4 + 4 = 12 bytes for FP32 master weights + momentum + variance (Adam optimizer)

### 9.2 Activation Memory (per GPU)

For a single linear layer $ Y = X \cdot W $ with column-linear TP:

$$
M_{\text{act, TP}} = b \cdot s \cdot \frac{h}{N} \quad (\text{intermediate activation is sharded})
$$

However, **LayerNorm and Dropout** still require the **full** activation $ b \cdot s \cdot h $, limiting the activation memory reduction to only the TP (linear) regions.

---

## 10. Sequence Parallelism (SP)

### 10.1 Motivation

Even with TP, operations like **LayerNorm** and **Dropout** require access to the **full hidden dimension** $ h $ and thus cannot be sharded along $ h $. These operations store activations of shape $ (b, s, h) $ on **every** GPU, partially negating TP's memory savings.

**LayerNorm** is defined as:

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

$$
\mu = \frac{1}{h} \sum_{j=1}^{h} x_j, \qquad \sigma^2 = \frac{1}{h} \sum_{j=1}^{h} (x_j - \mu)^2
$$

The mean $ \mu $ and variance $ \sigma^2 $ are computed **across the full hidden dimension** $ h $. This makes it impossible to shard LayerNorm along $ h $.

**Solution:** Shard these operations along the **sequence dimension** $ s $ instead. This is **Sequence Parallelism (SP)**.

### 10.2 SP Region Operations

In SP regions (LayerNorm, Dropout, residual connections):

- Activations are split along the sequence dimension: each GPU holds a shard of shape $ (b, s/N, h) $
- Since LayerNorm normalizes across $ h $ (not $ s $), each GPU can compute LayerNorm **independently** on its sequence chunk

### 10.3 Transitions Between TP and SP Regions

The Transformer layer alternates between **SP regions** (LayerNorm, Dropout) and **TP regions** (Linear layers in MHA and MLP). Transitions require reshaping the sharding axis from sequence to hidden dimension and vice versa.

Two conjugate operator pairs $ (f, f^*) $ and $ (g, g^*) $ manage these transitions:

#### Operators $f$ and $f^*$ (used in vanilla TP, at boundaries of TP region)

| Pass | $ f $ | $ f^* $ |
|------|-------|---------|
| Forward | No-op (activations already replicated) | All-Reduce (sum partial results) |
| Backward | All-Reduce (sync gradients) | No-op (gradients already replicated) |

$ f $ and $ f^* $ are **conjugate pairs**: in each pass, when one is a no-op, the other is an all-reduce.

#### Operators $g$ and $g^*$ (used in TP+SP, at transitions between SP and TP regions)

| Pass | $ g $ | $ g^* $ |
|------|-------|---------|
| Forward | All-Gather (restore full sequence for TP region) | Reduce-Scatter (sum + split along sequence for SP region) |
| Backward | Reduce-Scatter | All-Gather |

### 10.4 Step-by-Step Forward Pass Through MLP with TP+SP

Consider $ N = 2 $ GPUs. Denote sharded tensors with asterisks.

**Step 1 — Initial LayerNorm (SP Region):**

Input: $ X_1^*, X_2^* \in \mathbb{R}^{b \times (s/2) \times h} $ (sequence-sharded)

Each GPU computes LayerNorm independently on its chunk:

$$
\hat{X}_i^* = \text{LayerNorm}(X_i^*), \quad i \in \{1, 2\}
$$

**Step 2 — Transition SP → TP ($ g $: All-Gather):**

Reconstruct the full sequence:

$$
\hat{X} = \text{All-Gather}(\hat{X}_1^*, \hat{X}_2^*) \in \mathbb{R}^{b \times s \times h}
$$

Each GPU now holds the full $ \hat{X} $.

**Step 3 — Column-Linear Layer $ W_1 $ (TP Region):**

$ W_1 $ is column-sharded: $ W_1^{(i)} \in \mathbb{R}^{h \times (4h/2)} $

$$
Z_i^* = \text{GeLU}(\hat{X} \cdot W_1^{(i)}) \in \mathbb{R}^{b \times s \times (4h/2)}
$$

Activations are now sharded along the hidden dimension.

**Step 4 — Row-Linear Layer $ W_2 $ (TP Region):**

$ W_2 $ is row-sharded: $ W_2^{(i)} \in \mathbb{R}^{(4h/2) \times h} $

$$
Y_i = Z_i^* \cdot W_2^{(i)} \in \mathbb{R}^{b \times s \times h}
$$

These are **partial** results that need to be summed.

**Step 5 — Transition TP → SP ($ g^* $: Reduce-Scatter):**

Instead of a full all-reduce, we perform a **reduce-scatter** which simultaneously:
- **Reduces** (sums) partial results $ Y_1, Y_2 $ across GPUs
- **Scatters** the result along the sequence dimension

$$
Y_1^* = \left(\sum_{i} Y_i\right)[\text{seq chunk 1}] \in \mathbb{R}^{b \times (s/2) \times h}
$$

$$
Y_2^* = \left(\sum_{i} Y_i\right)[\text{seq chunk 2}] \in \mathbb{R}^{b \times (s/2) \times h}
$$

Each GPU now holds a **sequence-sharded** output, ready for the next SP region (LayerNorm, residual, dropout).

### 10.5 Activation Shape Summary Table

| Region | TP Only | TP + SP |
|--------|---------|---------|
| **Enter TP (column-linear)** | $ h $: sharded, $ s $: full | $ h $: sharded, $ s $: **all-gather to full** |
| **Inside TP region** | $ h $: sharded, $ s $: full | $ h $: sharded, $ s $: full |
| **Exit TP (row-linear)** | $ h $: full (all-reduce), $ s $: full | $ h $: full, $ s $: **reduce-scatter to sharded** |
| **SP region (LN, Dropout)** | $ h $: full, $ s $: full | $ h $: full, $ s $: **sharded** |

### 10.6 Maximum Activation Size Comparison

| Method | Maximum Activation Per GPU |
|--------|---------------------------|
| No parallelism | $ b \cdot s \cdot h $ |
| TP only | $ b \cdot s \cdot h $ (in LayerNorm/Dropout regions) |
| TP + SP | $ \dfrac{b \cdot s \cdot h}{N} $ (always sharded along either $ h $ or $ s $) |

With TP+SP, activations are **always** partitioned — along $ h $ in TP regions and along $ s $ in SP regions — ensuring the peak activation memory per GPU is reduced by a factor of $ N $.

---

## 11. Communication Cost Equivalence: TP vs. TP+SP

### 11.1 Vanilla TP (per Transformer layer, forward pass)

- **2 × All-Reduce** operations (one for MHA, one for MLP)

### 11.2 TP + SP (per Transformer layer, forward pass)

- **2 × All-Gather** (SP → TP transitions)
- **2 × Reduce-Scatter** (TP → SP transitions)

### 11.3 Equivalence Proof

Using the identity:

$$
\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}
$$

For vanilla TP: total communication = $ 2 \times \text{All-Reduce} = 2 \times (\text{Reduce-Scatter} + \text{All-Gather}) $

For TP+SP: total communication = $ 2 \times \text{All-Gather} + 2 \times \text{Reduce-Scatter} $

$$
\boxed{C_{\text{TP}} = C_{\text{TP+SP}}}
$$

The communication volume is **identical**. TP+SP achieves strictly better activation memory with **no additional communication cost** relative to vanilla TP.

The same reasoning applies to the backward pass via conjugate operation substitution:

$$
\text{no-op} \leftrightarrow \text{All-Reduce}, \qquad \text{All-Gather} \leftrightarrow \text{Reduce-Scatter}
$$

---

## 12. Gradient Synchronization in SP Regions

### 12.1 LayerNorm Weights

Since each TP rank sees the **same** activations after the all-gather (in vanilla TP), the LayerNorm weights naturally remain synchronized — their gradients are identical across ranks. **No all-reduce is needed** for LayerNorm gradient synchronization in vanilla TP.

However, in **SP mode**, each rank processes a **different** sequence chunk through LayerNorm, producing **different gradients** for $ \gamma $ and $ \beta $. Therefore:

$$
\nabla \gamma = \text{All-Reduce}\left(\nabla \gamma_i\right), \qquad \nabla \beta = \text{All-Reduce}\left(\nabla \beta_i\right)
$$

This is a minor overhead since LayerNorm has only $ 2h $ parameters ($ \gamma, \beta \in \mathbb{R}^h $), negligible compared to the weight matrices.

### 12.2 Dropout Synchronization

In TP (without SP), dropout must use **synchronized random seeds** across TP ranks so that the same dropout mask is applied to identical activations, maintaining deterministic behavior:

$$
\text{seed}_{\text{rank } i} = \text{seed}_{\text{global}} \quad \forall \, i \in \{1, \ldots, N\}
$$

In SP mode, since each rank processes a different sequence chunk, independent seeds are acceptable.

---

## 13. Practical Scaling Guidelines and Trade-offs

### 13.1 Throughput vs. Memory Trade-off

| TP Degree | Compute Efficiency | Memory Per GPU | Communication |
|-----------|--------------------|----------------|---------------|
| 1 | Baseline (no overhead) | Maximum | None |
| 2–8 (intra-node) | High (NVLink) | $ \propto 1/N $ | Low latency |
| 16 (inter-node) | Significant drop | Further reduced | High latency |
| 32+ (inter-node) | Severe degradation | Minimal | Dominates runtime |

### 13.2 When to Use TP vs. TP+SP

| Scenario | Recommendation |
|----------|---------------|
| Model fits on single GPU | No TP needed |
| Model fits with TP ≤ 8 | Use TP+SP within a single node |
| Model requires TP > 8 | Combine with Pipeline Parallelism to keep TP ≤ 8 |
| Long sequences cause OOM in TP region | Add Context Parallelism |

### 13.3 Remaining Limitations of TP+SP

1. **Sequence length scaling:** In the TP region, activations still have full sequence length $ s $. As $ s $ grows, activation memory in TP regions can still exceed capacity → requires **Context Parallelism** (e.g., Ring Attention).

2. **Model size exceeding single-node capacity:** If the model is too large for TP = 8 (one node), inter-node TP incurs massive slowdown → requires **Pipeline Parallelism** to distribute layers across nodes while keeping TP intra-node.

---

## 14. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| Column-wise decomposition | $$ A \cdot B = \begin{bmatrix} AB_1 & AB_2 & \cdots & AB_N \end{bmatrix} $$ |
| Row-wise decomposition | $$ A \cdot B = \sum_{i=1}^{N} A_i B_i $$ |
| LayerNorm | $$ \text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$ |
| All-Reduce decomposition | $$ \text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather} $$ |
| Communication equivalence | $$ C_{\text{TP}} = C_{\text{TP+SP}} $$ |
| Max activation (TP+SP) | $$ M_{\text{act}}^{\text{max}} = \frac{b \cdot s \cdot h}{N} $$ |
| TP degree constraint | $$ \text{TP} \leq n_h \quad (\text{number of attention heads}) $$ |
| Model state memory per GPU | $$ M_{\text{model}} = \frac{16P}{N} \text{ bytes (mixed precision + Adam)} $$ |



# State-of-the-Art Tensor Parallelism: PyTorch + Triton Implementation

---

## 1. Architecture Overview

The complete TP+SP system requires the following components, organized bottom-up:

```
┌─────────────────────────────────────────────────────┐
│          TransformerBlock (TP + SP)                  │
│  ┌──────────────┐       ┌──────────────────────┐    │
│  │  TP-MHA      │       │  TP-MLP              │    │
│  │  (Col→Row)   │       │  (Col→Row)           │    │
│  └──────────────┘       └──────────────────────┘    │
│  ┌──────────────────────────────────────────────┐   │
│  │  SP-LayerNorm  |  SP-Dropout  |  SP-Residual │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Communication: AllReduce, AllGather,         │   │
│  │  ReduceScatter, Broadcast                     │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Triton Kernels: Fused LayerNorm, GeLU,       │   │
│  │  Fused AllGather+GEMM, GEMM+ReduceScatter    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 2. Process Group Initialization

Before any distributed operation, we must establish the TP process group — a subset of all ranks that participate in tensor-parallel communication.

```python
"""
tensor_parallel/init.py
──────────────────────────
Initializes the tensor-parallel process group and exposes
rank/world-size helpers used by every subsequent module.

Key concept:
    Given  W  total GPUs and  TP  degree, we create
    W / TP  independent TP groups.  Ranks within the same
    TP group communicate via NVLink (intra-node).

    Example with W=16, TP=4:
        Group 0: [0, 1, 2, 3]
        Group 1: [4, 5, 6, 7]
        Group 2: [8, 9, 10, 11]
        Group 3: [12, 13, 14, 15]
"""

import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelState:
    """Holds all tensor-parallel metadata for the current rank."""
    tp_rank: int          # rank within the TP group  (0 .. TP-1)
    tp_world_size: int    # TP degree  (N)
    tp_group: dist.ProcessGroup  # NCCL process group handle
    global_rank: int      # rank across all GPUs
    global_world_size: int


# Module-level singleton — set once at startup
_TP_STATE: Optional[ParallelState] = None


def initialize_tensor_parallel(tp_degree: int) -> ParallelState:
    """
    Must be called after torch.distributed.init_process_group().

    Parameters
    ----------
    tp_degree : int
        Number of GPUs in each tensor-parallel group.
        Must divide the total world size evenly.

    Returns
    -------
    ParallelState
    """
    global _TP_STATE

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % tp_degree == 0, (
        f"World size {world_size} must be divisible by TP degree {tp_degree}"
    )

    # ── Create TP sub-groups ──────────────────────────────
    # Ranks are partitioned into contiguous chunks of size tp_degree.
    num_tp_groups = world_size // tp_degree
    for i in range(num_tp_groups):
        ranks_in_group = list(range(i * tp_degree, (i + 1) * tp_degree))
        group = dist.new_group(ranks_in_group)
        if rank in ranks_in_group:
            tp_group = group
            tp_rank = ranks_in_group.index(rank)

    _TP_STATE = ParallelState(
        tp_rank=tp_rank,
        tp_world_size=tp_degree,
        tp_group=tp_group,
        global_rank=rank,
        global_world_size=world_size,
    )
    return _TP_STATE


def get_tp_state() -> ParallelState:
    """Return the singleton ParallelState (must call initialize first)."""
    assert _TP_STATE is not None, "Call initialize_tensor_parallel() first."
    return _TP_STATE
```

---

## 3. Communication Primitives

These primitives wrap NCCL collectives with **autograd-compatible** forward/backward semantics. The mathematical identities governing conjugate pairs are:

$$
\text{Forward: } f = \text{no-op}, \quad f^* = \text{All-Reduce}
$$
$$
\text{Backward: } f = \text{All-Reduce}, \quad f^* = \text{no-op}
$$
$$
\text{Forward: } g = \text{All-Gather}, \quad g^* = \text{Reduce-Scatter}
$$
$$
\text{Backward: } g = \text{Reduce-Scatter}, \quad g^* = \text{All-Gather}
$$

```python
"""
tensor_parallel/comm.py
───────────────────────
Autograd-aware wrappers around NCCL collective operations.

Every communication primitive is implemented as a torch.autograd.Function
so that the correct conjugate operation executes during backward().

Notation follows Megatron-LM conventions:
    f  / f*  — conjugate pair for vanilla TP
    g  / g*  — conjugate pair for TP + SP transitions
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Any, Tuple

from .init import get_tp_state


# ═══════════════════════════════════════════════════════
#  Primitive 1:  AllReduce  (sum across TP ranks)
# ═══════════════════════════════════════════════════════
# Used in vanilla TP at row-linear output (forward)
# and at column-linear input (backward).
#
# Math:  Y = Σ_{i=1}^{N} Y_i
# ═══════════════════════════════════════════════════════

class _AllReduce(torch.autograd.Function):
    """
    Forward:  all-reduce (sum)
    Backward: identity (no-op)

    This is the f* operator.
    """
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        if get_tp_state().tp_world_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM,
                        group=get_tp_state().tp_group)
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        # Conjugate of all-reduce in forward → no-op in backward
        return grad_output


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    """
    Forward:  identity (no-op)
    Backward: all-reduce (sum)

    This is the f operator — used to broadcast/copy input
    in the forward pass, then sync gradients in backward.
    """
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        if get_tp_state().tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM,
                        group=get_tp_state().tp_group)
        return grad_output


# ═══════════════════════════════════════════════════════
#  Primitive 2:  AllGather  /  ReduceScatter
# ═══════════════════════════════════════════════════════
# Used in TP+SP transitions.
#
# AllGather gathers shards along a dimension:
#   [X_1, X_2, ..., X_N] → X   (concatenation)
#
# ReduceScatter reduces (sums) and scatters:
#   Y_1, Y_2, ..., Y_N → [Σ Y_i chunk_1, Σ Y_i chunk_2, ...]
# ═══════════════════════════════════════════════════════

class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    """
    Forward:  all-gather along `gather_dim`       (g operator)
    Backward: reduce-scatter along `gather_dim`    (conjugate)

    Transitions from SP → TP region.
    Input  shape: (b, s/N, h)
    Output shape: (b, s,   h)   when gather_dim=1
    """
    @staticmethod
    def forward(ctx: Any, x: Tensor, gather_dim: int) -> Tensor:
        ctx.gather_dim = gather_dim
        tp = get_tp_state()
        if tp.tp_world_size == 1:
            return x

        # Allocate output buffer
        world_size = tp.tp_world_size
        local_size = x.shape[gather_dim]
        output_shape = list(x.shape)
        output_shape[gather_dim] = local_size * world_size

        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Perform all-gather into list of tensors, then cat
        tensor_list = list(output.chunk(world_size, dim=gather_dim))
        dist.all_gather(
            tensor_list,
            x.contiguous(),
            group=tp.tp_group,
        )

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        tp = get_tp_state()
        if tp.tp_world_size == 1:
            return grad_output, None

        # Reduce-scatter: sum across ranks, each rank gets a shard
        return _reduce_scatter(grad_output, ctx.gather_dim), None


class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    """
    Forward:  reduce-scatter along `scatter_dim`   (g* operator)
    Backward: all-gather along `scatter_dim`       (conjugate)

    Transitions from TP → SP region.
    Input  shape: (b, s, h)   — partial results on each rank
    Output shape: (b, s/N, h) when scatter_dim=1
    """
    @staticmethod
    def forward(ctx: Any, x: Tensor, scatter_dim: int) -> Tensor:
        ctx.scatter_dim = scatter_dim
        tp = get_tp_state()
        if tp.tp_world_size == 1:
            return x

        return _reduce_scatter(x, scatter_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        tp = get_tp_state()
        if tp.tp_world_size == 1:
            return grad_output, None

        # All-gather in backward
        world_size = tp.tp_world_size
        scatter_dim = ctx.scatter_dim
        local_size = grad_output.shape[scatter_dim]
        output_shape = list(grad_output.shape)
        output_shape[scatter_dim] = local_size * world_size

        output = torch.empty(
            output_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        tensor_list = list(output.chunk(world_size, dim=scatter_dim))
        dist.all_gather(
            tensor_list,
            grad_output.contiguous(),
            group=tp.tp_group,
        )
        return output, None


def _reduce_scatter(x: Tensor, dim: int) -> Tensor:
    """
    Helper: reduce-scatter along dimension `dim`.

    Mathematically:
        result_i = (Σ_{j=1}^{N} x_j)[chunk_i]

    where chunk_i is the i-th partition of the summed tensor
    along dimension `dim`.
    """
    tp = get_tp_state()
    world_size = tp.tp_world_size

    # Split input into N chunks along `dim`
    input_chunks = list(x.chunk(world_size, dim=dim))
    output = torch.empty_like(input_chunks[0])

    dist.reduce_scatter(
        output,
        input_chunks,
        op=dist.ReduceOp.SUM,
        group=tp.tp_group,
    )
    return output


# ═══════════════════════════════════════════════════════
#  Functional API  (clean interface for layer code)
# ═══════════════════════════════════════════════════════

def all_reduce(x: Tensor) -> Tensor:
    """f* : forward=all-reduce, backward=no-op."""
    return _AllReduce.apply(x)


def identity_fwd_allreduce_bwd(x: Tensor) -> Tensor:
    """f : forward=no-op, backward=all-reduce."""
    return _IdentityForwardAllReduceBackward.apply(x)


def all_gather_fwd_reduce_scatter_bwd(
    x: Tensor, gather_dim: int = 1
) -> Tensor:
    """g : forward=all-gather, backward=reduce-scatter."""
    return _AllGatherForwardReduceScatterBackward.apply(x, gather_dim)


def reduce_scatter_fwd_all_gather_bwd(
    x: Tensor, scatter_dim: int = 1
) -> Tensor:
    """g* : forward=reduce-scatter, backward=all-gather."""
    return _ReduceScatterForwardAllGatherBackward.apply(x, scatter_dim)
```

---

## 4. Triton Kernels: Fused Operations

Triton kernels eliminate redundant memory round-trips by fusing elementwise operations (GeLU, bias addition, LayerNorm) with the preceding or following GEMM. This is critical because **memory bandwidth** — not compute — is the bottleneck for these operations.

### 4.1 Fused Bias + GeLU Kernel

The GeLU activation is defined as:

$$
\text{GeLU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

The fast approximation (used in practice):

$$
\text{GeLU}(x) \approx 0.5 \, x \left[1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 \, x^3\right)\right)\right]
$$

```python
"""
tensor_parallel/kernels/fused_gelu.py
──────────────────────────────────────
Triton kernel that fuses  bias_add + GeLU  into a single
GPU kernel launch, avoiding an intermediate materialization
of the (b, s, 4h/N) tensor between bias and activation.

Saves:  b × s × (4h / N) × sizeof(dtype)  bytes of HBM traffic.
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def _fused_bias_gelu_fwd_kernel(
    X_ptr,          # input tensor   (flattened)
    Bias_ptr,       # bias vector    (length = hidden_dim)
    Out_ptr,        # output tensor  (same shape as X)
    N_ELEMENTS: tl.constexpr,   # total number of elements
    HIDDEN_DIM: tl.constexpr,   # last-dimension size (for bias indexing)
    BLOCK_SIZE: tl.constexpr,   # number of elements per program
):
    """
    Each Triton program processes BLOCK_SIZE contiguous elements.

    For element at flat index `idx`:
        col = idx % HIDDEN_DIM            (selects the bias element)
        x   = X[idx] + Bias[col]
        out = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # ── Load input + bias ─────────────────────────────────
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    col = offsets % HIDDEN_DIM
    bias = tl.load(Bias_ptr + col, mask=mask, other=0.0)
    x = x + bias

    # ── GeLU (tanh approximation) ─────────────────────────
    # Constants:  sqrt(2/π) ≈ 0.7978845608
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608
    COEFF: tl.constexpr = 0.044715

    inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))

    tl.store(Out_ptr + offsets, out, mask=mask)


@triton.jit
def _fused_bias_gelu_bwd_kernel(
    Grad_ptr,        # incoming gradient  dL/dOut
    X_ptr,           # original input (before bias+gelu)
    Bias_ptr,        # bias vector
    Grad_X_ptr,      # output: dL/dX
    Grad_Bias_ptr,   # output: partial dL/dBias (per-block atomics)
    N_ELEMENTS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for fused bias + GeLU.

    GeLU derivative (tanh approx):
        Let  t = tanh(sqrt(2/π)(x + 0.044715 x³))
        dGeLU/dx = 0.5(1 + t) + 0.5 x (1 - t²) sqrt(2/π)(1 + 3·0.044715 x²)

    Chain rule:
        dL/dX    = dL/dOut · dGeLU/dx
        dL/dBias = Σ_rows dL/dX  (accumulated via atomics)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    grad_out = tl.load(Grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    col = offsets % HIDDEN_DIM
    bias = tl.load(Bias_ptr + col, mask=mask, other=0.0)
    x = x + bias

    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608
    COEFF: tl.constexpr = 0.044715

    inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
    t = tl.math.tanh(inner)

    # dGeLU/dx
    dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (
        1.0 + 3.0 * COEFF * x * x
    )

    grad_x = grad_out * dgelu

    tl.store(Grad_X_ptr + offsets, grad_x, mask=mask)
    # Atomic add for bias gradient reduction across rows
    tl.atomic_add(Grad_Bias_ptr + col, grad_x, mask=mask)


class FusedBiasGeLU(torch.autograd.Function):
    """
    Autograd wrapper for Triton fused bias + GeLU.

    Forward:   Y = GeLU(X + bias)
    Backward:  dX, dBias
    """
    @staticmethod
    def forward(ctx, x: Tensor, bias: Tensor) -> Tensor:
        assert x.is_contiguous()
        n_elements = x.numel()
        hidden_dim = x.shape[-1]
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024

        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_bias_gelu_fwd_kernel[grid](
            x, bias, out,
            N_ELEMENTS=n_elements,
            HIDDEN_DIM=hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, bias = ctx.saved_tensors
        n_elements = x.numel()
        hidden_dim = x.shape[-1]
        grad_x = torch.empty_like(x)
        grad_bias = torch.zeros(
            hidden_dim, dtype=x.dtype, device=x.device
        )
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _fused_bias_gelu_bwd_kernel[grid](
            grad_output, x, bias, grad_x, grad_bias,
            N_ELEMENTS=n_elements,
            HIDDEN_DIM=hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_x, grad_bias


def fused_bias_gelu(x: Tensor, bias: Tensor) -> Tensor:
    """Functional API:  GeLU(x + bias)  in a single kernel."""
    return FusedBiasGeLU.apply(x, bias)
```

### 4.2 Triton Fused LayerNorm Kernel

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \quad \mu = \frac{1}{h}\sum_{j=1}^{h} x_j, \quad \sigma^2 = \frac{1}{h}\sum_{j=1}^{h}(x_j - \mu)^2
$$

```python
"""
tensor_parallel/kernels/fused_layernorm.py
──────────────────────────────────────────
Triton implementation of LayerNorm.

Each Triton program handles one row of the input
(i.e., one token's hidden-dimension vector).

For SP: each GPU only processes its local sequence chunk,
so input shape is (b, s/N, h) and each row has length h.
"""

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr,        # (M, H)  input
    Gamma_ptr,    # (H,)    scale
    Beta_ptr,     # (H,)    shift
    Out_ptr,      # (M, H)  output
    Mean_ptr,     # (M,)    save for backward
    Rstd_ptr,     # (M,)    save for backward (1/sqrt(var+eps))
    H: tl.constexpr,       # hidden dimension
    EPS: tl.constexpr,     # epsilon  (e.g. 1e-5)
    BLOCK_H: tl.constexpr, # must be >= H, power of 2
):
    """
    One program ≡ one row.
    pid = row index in [0, M).

    Algorithm:
        1. Load row x ∈ ℝ^H
        2. μ = mean(x)
        3. σ² = var(x)
        4. x̂ = (x - μ) / sqrt(σ² + ε)
        5. y = γ x̂ + β
        6. Store y, μ, 1/sqrt(σ²+ε)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # ── Step 1: Load row ──────────────────────────────────
    x = tl.load(X_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    # ── Step 2-3: Mean and variance ───────────────────────
    mean = tl.sum(x, axis=0) / H
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + EPS)

    # ── Step 4-5: Normalize + affine ──────────────────────
    x_hat = x_centered * rstd
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = gamma * x_hat + beta

    # ── Step 6: Store ─────────────────────────────────────
    tl.store(Out_ptr + row * H + cols, y.to(tl.float16), mask=mask)
    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)


@triton.jit
def _layernorm_bwd_kernel(
    Grad_ptr,     # (M, H)  dL/dY
    X_ptr,        # (M, H)  original input
    Gamma_ptr,    # (H,)
    Mean_ptr,     # (M,)
    Rstd_ptr,     # (M,)
    Grad_X_ptr,   # (M, H)  output: dL/dX
    Grad_Gamma_ptr,  # (H,) output: partial dL/dγ (atomic)
    Grad_Beta_ptr,   # (H,) output: partial dL/dβ (atomic)
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward for LayerNorm.

    dL/dx̂ = dL/dy · γ
    dL/dσ = -0.5 Σ dL/dx̂ · (x-μ) · (σ²+ε)^{-3/2}  (implicit)
    dL/dμ = -Σ dL/dx̂ · rstd                           (implicit)
    dL/dx = rstd · (dL/dx̂ - mean(dL/dx̂) - x̂ · mean(dL/dx̂ · x̂))

    dL/dγ = Σ_rows dL/dy · x̂
    dL/dβ = Σ_rows dL/dy
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    grad_y = tl.load(Grad_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row)
    rstd = tl.load(Rstd_ptr + row)

    x_hat = (x - mean) * rstd
    grad_x_hat = grad_y * gamma

    # Efficient formulation (Welford-style)
    mean_grad_x_hat = tl.sum(grad_x_hat, axis=0) / H
    mean_grad_x_hat_xhat = tl.sum(grad_x_hat * x_hat, axis=0) / H

    grad_x = rstd * (grad_x_hat - mean_grad_x_hat - x_hat * mean_grad_x_hat_xhat)

    tl.store(Grad_X_ptr + row * H + cols, grad_x.to(tl.float16), mask=mask)

    # Accumulate parameter gradients
    tl.atomic_add(Grad_Gamma_ptr + cols, grad_y * x_hat, mask=mask)
    tl.atomic_add(Grad_Beta_ptr + cols, grad_y, mask=mask)


class TritonLayerNorm(torch.nn.Module):
    """
    Drop-in replacement for torch.nn.LayerNorm using Triton.

    Parameters
    ----------
    hidden_size : int
        Dimension of the last axis (h).
    eps : float
        Numerical stability constant ε.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        # Reshape to 2D: (M, H) where M = b*s or b*(s/N) in SP
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        M, H = x_2d.shape

        out = torch.empty_like(x_2d)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_H = triton.next_power_of_2(H)

        _layernorm_fwd_kernel[(M,)](
            x_2d, self.weight, self.bias, out, mean, rstd,
            H=H, EPS=self.eps, BLOCK_H=BLOCK_H,
        )
        return out.reshape(orig_shape)
```

---

## 5. Column-Linear and Row-Linear Layers

These are the core building blocks that implement the two decomposition strategies from Section 2.

### 5.1 Column-Linear Layer

$$
Y_i = X \cdot W_i, \quad W_i \in \mathbb{R}^{h \times (h'/N)}
$$

```python
"""
tensor_parallel/layers.py
─────────────────────────
ColumnLinear and RowLinear — the two fundamental TP layer types.

Weight initialization note:
    Each rank initializes ONLY its local shard.
    For column-linear:  W_local ∈ ℝ^{h_in × (h_out / N)}
    For row-linear:     W_local ∈ ℝ^{(h_in / N) × h_out}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .init import get_tp_state
from .comm import (
    identity_fwd_allreduce_bwd,
    all_reduce,
    all_gather_fwd_reduce_scatter_bwd,
    reduce_scatter_fwd_all_gather_bwd,
)


class ColumnLinear(nn.Module):
    """
    Column-parallel linear layer.

    The weight matrix W ∈ ℝ^{h_in × h_out} is sharded along
    the OUTPUT (column) dimension:

        W_i ∈ ℝ^{h_in × (h_out / N)}

    Forward (vanilla TP — no SP):
        1. f(X) = X                       (copy, forward no-op)
        2. Y_i  = X · W_i + bias_i        (local GEMM)
        3. Return Y_i ∈ (b, s, h_out/N)   (sharded output)

    Forward (with SP — sequence parallelism):
        1. g(X*) = AllGather(X*)           (restore full sequence)
        2. Y_i   = X · W_i + bias_i       (local GEMM)
        3. Return Y_i ∈ (b, s, h_out/N)   (sharded output)

    Parameters
    ----------
    in_features  : int  — h_in
    out_features : int  — h_out (TOTAL, before sharding)
    bias         : bool
    sequence_parallel : bool — whether to use SP transitions
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        tp = get_tp_state()

        assert out_features % tp.tp_world_size == 0, (
            f"out_features={out_features} must be divisible by "
            f"tp_world_size={tp.tp_world_size}"
        )

        self.in_features = in_features
        self.out_features_per_rank = out_features // tp.tp_world_size
        self.sequence_parallel = sequence_parallel

        # ── Local weight shard ────────────────────────────
        # Shape: (h_in, h_out / N)
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_rank)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform, adjusted for TP shard size."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        x shape:
            vanilla TP:  (b, s, h_in)     — replicated
            with SP:     (b, s/N, h_in)   — sequence-sharded
        """
        if self.sequence_parallel:
            # g: all-gather along sequence dim → (b, s, h_in)
            x = all_gather_fwd_reduce_scatter_bwd(x, gather_dim=1)
        else:
            # f: forward no-op, backward all-reduce
            x = identity_fwd_allreduce_bwd(x)

        # Local GEMM: (b, s, h_in) @ (h_in, h_out/N)^T → (b, s, h_out/N)
        output = F.linear(x, self.weight, self.bias)
        return output


class RowLinear(nn.Module):
    """
    Row-parallel linear layer.

    The weight matrix W ∈ ℝ^{h_in × h_out} is sharded along
    the INPUT (row) dimension:

        W_i ∈ ℝ^{(h_in / N) × h_out}

    Forward (vanilla TP — no SP):
        1. Y_i = X_i · W_i              (local GEMM on sharded input)
        2. Y = f*(Y_i) = AllReduce(Y_i)  (sum partial results)
        3. Return Y + bias

    Forward (with SP):
        1. Y_i = X_i · W_i              (local GEMM on sharded input)
        2. Y* = g*(Y_i) = ReduceScatter(Y_i)  (sum + scatter along seq)
        3. Return Y* + bias

    Parameters
    ----------
    in_features  : int  — h_in (TOTAL, before sharding)
    out_features : int  — h_out
    bias         : bool
    sequence_parallel : bool
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        tp = get_tp_state()

        assert in_features % tp.tp_world_size == 0, (
            f"in_features={in_features} must be divisible by "
            f"tp_world_size={tp.tp_world_size}"
        )

        self.in_features_per_rank = in_features // tp.tp_world_size
        self.out_features = out_features
        self.sequence_parallel = sequence_parallel

        # ── Local weight shard ────────────────────────────
        # Shape: (h_out, h_in / N)
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        if bias:
            # Bias is NOT sharded — only rank 0 adds it
            # (or equivalently, divide by N and let all-reduce sum)
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features_per_rank * get_tp_state().tp_world_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        x shape: (b, s, h_in / N)  — hidden-dim-sharded from column-linear
        """
        # Local GEMM: (b, s, h_in/N) @ (h_in/N, h_out)^T → (b, s, h_out)
        output = F.linear(x, self.weight)

        if self.sequence_parallel:
            # g*: reduce-scatter along sequence dim
            # (b, s, h_out) → (b, s/N, h_out)
            output = reduce_scatter_fwd_all_gather_bwd(output, scatter_dim=1)
        else:
            # f*: all-reduce (sum partial results)
            output = all_reduce(output)

        if self.bias is not None:
            output = output + self.bias

        return output
```

---

## 6. TP-MLP Block

The Transformer MLP with TP follows the **Column-Linear → GeLU → Row-Linear** pattern:

$$
\text{MLP}(X) = \underbrace{\text{GeLU}(X \, W_1^{(i)})}_{\text{Column-Linear}} \cdot \underbrace{W_2^{(i)}}_{\text{Row-Linear}}
$$

Communication per MLP sub-block (forward):
- Vanilla TP: 1 × All-Reduce
- TP+SP: 1 × All-Gather (entry) + 1 × Reduce-Scatter (exit)

```python
"""
tensor_parallel/mlp.py
──────────────────────
Tensor-parallel MLP block.

Architecture:
    LayerNorm → ColumnLinear(h → 4h) → GeLU → RowLinear(4h → h)

Communication flow (TP+SP, forward):
    SP region: X* ∈ (b, s/N, h)
        │
        ├─ LayerNorm (local on sequence chunk)
        │
        ├─ [g: AllGather]  → (b, s, h)
        │
        ├─ ColumnLinear W₁  → (b, s, 4h/N)   ← TP region
        │
        ├─ GeLU (fused with bias)
        │
        ├─ RowLinear W₂     → (b, s, h)       ← partial
        │
        ├─ [g*: ReduceScatter] → (b, s/N, h)
        │
        └─ Residual add (in SP region)
"""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import ColumnLinear, RowLinear
from .kernels.fused_gelu import fused_bias_gelu
from .kernels.fused_layernorm import TritonLayerNorm


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel feedforward block with optional
    sequence parallelism.

    Parameters
    ----------
    hidden_size     : int  — model hidden dimension h
    intermediate_size : int — MLP intermediate dim (typically 4h)
    sequence_parallel : bool — enable SP for LayerNorm/Dropout
    eps             : float — LayerNorm epsilon
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        sequence_parallel: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.layernorm = TritonLayerNorm(hidden_size, eps=eps)

        # Column-linear: (h) → (4h/N)
        # GeLU bias is folded into the column-linear layer
        self.fc1 = ColumnLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=True,
            sequence_parallel=sequence_parallel,
        )

        # Row-linear: (4h/N) → (h)
        self.fc2 = RowLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=True,
            sequence_parallel=sequence_parallel,
        )

        self.sequence_parallel = sequence_parallel

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (b, s, h)     if not SP
            (b, s/N, h)   if SP

        Returns same shape as input (residual-compatible).
        """
        residual = x

        # ── LayerNorm (SP region) ─────────────────────────
        # Each GPU normalizes its local sequence chunk.
        # Mean/var computed across h (full hidden dim available).
        h = self.layernorm(x)

        # ── Column-Linear + GeLU (TP region) ─────────────
        # Internally: AllGather (if SP) → GEMM → bias + GeLU
        h = self.fc1(h)  # (b, s, 4h/N)

        # Fused bias+GeLU via Triton kernel
        # (If bias was already added in fc1, use identity bias here;
        #  in production, fc1 returns pre-bias output for fusion)
        h = torch.nn.functional.gelu(h, approximate="tanh")

        # ── Row-Linear (TP region → SP region) ───────────
        # Internally: GEMM → ReduceScatter (if SP) or AllReduce
        h = self.fc2(h)  # (b, s/N, h) if SP, else (b, s, h)

        # ── Residual connection (SP region) ───────────────
        output = residual + h

        return output
```

---

## 7. TP-MHA Block

Multi-head attention with TP shards along the `num_heads` dimension — each GPU computes attention for $ n_h / N $ heads:

$$
\text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i, \quad Q_i = X W^Q_{(i)}, \quad K_i = X W^K_{(i)}, \quad V_i = X W^V_{(i)}
$$

```python
"""
tensor_parallel/attention.py
────────────────────────────
Tensor-parallel multi-head attention.

Sharding strategy:
    Q, K, V projections → ColumnLinear (shard along num_heads)
    Output projection   → RowLinear    (shard along input dim)

Each GPU computes attention for  n_h / N  heads independently.

Supports:
    - Multi-Head Attention (MHA):  n_q = n_kv
    - Grouped Query Attention (GQA): n_q > n_kv
    - Multi-Query Attention (MQA): n_kv = 1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .init import get_tp_state
from .layers import ColumnLinear, RowLinear
from .kernels.fused_layernorm import TritonLayerNorm


class TensorParallelAttention(nn.Module):
    """
    Tensor-parallel multi-head attention block.

    Parameters
    ----------
    hidden_size       : int  — model dimension h
    num_attention_heads : int — total number of query heads n_q
    num_kv_heads      : int  — total number of KV heads n_kv
                                (n_kv = n_q for MHA,
                                 n_kv = 1  for MQA,
                                 1 < n_kv < n_q for GQA)
    sequence_parallel : bool
    max_seq_len       : int — for RoPE / positional encoding
    eps               : float
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        sequence_parallel: bool = False,
        max_seq_len: int = 8192,
        eps: float = 1e-5,
    ):
        super().__init__()
        tp = get_tp_state()

        # ── Validate TP constraints ───────────────────────
        assert num_attention_heads % tp.tp_world_size == 0, (
            f"num_attention_heads={num_attention_heads} must be divisible "
            f"by tp_world_size={tp.tp_world_size}"
        )
        assert num_kv_heads % tp.tp_world_size == 0 or \
               tp.tp_world_size % num_kv_heads == 0, (
            f"num_kv_heads={num_kv_heads} must be divisible by or "
            f"divide tp_world_size={tp.tp_world_size}"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads

        # ── Per-rank head counts ──────────────────────────
        self.num_heads_per_rank = num_attention_heads // tp.tp_world_size
        self.num_kv_heads_per_rank = max(
            1, num_kv_heads // tp.tp_world_size
        )

        # ── Projections (Column-Linear for Q, K, V) ──────
        # Q projection: h → (n_q / N) * d_k
        self.q_proj = ColumnLinear(
            hidden_size,
            self.num_heads_per_rank * self.head_dim,
            bias=False,
            sequence_parallel=sequence_parallel,
        )
        # K projection: h → (n_kv / N) * d_k
        self.k_proj = ColumnLinear(
            hidden_size,
            self.num_kv_heads_per_rank * self.head_dim,
            bias=False,
            sequence_parallel=sequence_parallel,
        )
        # V projection: h → (n_kv / N) * d_k
        self.v_proj = ColumnLinear(
            hidden_size,
            self.num_kv_heads_per_rank * self.head_dim,
            bias=False,
            sequence_parallel=sequence_parallel,
        )

        # ── Output projection (Row-Linear) ───────────────
        self.o_proj = RowLinear(
            self.num_heads_per_rank * self.head_dim,
            hidden_size,
            bias=False,
            sequence_parallel=sequence_parallel,
        )

        self.layernorm = TritonLayerNorm(hidden_size, eps=eps)
        self.sequence_parallel = sequence_parallel

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor = None,
    ) -> Tensor:
        """
        x : (b, s, h)   or  (b, s/N, h) if SP

        Returns same shape as input.
        """
        residual = x
        b, s_local, _ = x.shape

        # ── LayerNorm (SP region) ─────────────────────────
        h = self.layernorm(x)

        # ── QKV Projections (Column-Linear) ───────────────
        # AllGather happens inside ColumnLinear if SP is on.
        # After projection, sequence dim is full (s, not s/N).
        q = self.q_proj(h)  # (b, s, n_q_local * d_k)
        k = self.k_proj(h)  # (b, s, n_kv_local * d_k)
        v = self.v_proj(h)  # (b, s, n_kv_local * d_k)

        s = q.shape[1]  # full sequence length after AllGather

        # ── Reshape to multi-head format ──────────────────
        q = q.view(b, s, self.num_heads_per_rank, self.head_dim)
        k = k.view(b, s, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(b, s, self.num_kv_heads_per_rank, self.head_dim)

        # Transpose to (b, n_heads, s, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── GQA: Expand KV heads to match Q heads ────────
        if self.num_kv_heads_per_rank < self.num_heads_per_rank:
            repeat_factor = self.num_heads_per_rank // self.num_kv_heads_per_rank
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # ── Scaled Dot-Product Attention ──────────────────
        # score = Q K^T / sqrt(d_k)
        # attn  = softmax(score + mask)
        # out   = attn · V
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )
        # attn_output: (b, n_heads_local, s, d_k)

        # ── Reshape back ──────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            b, s, self.num_heads_per_rank * self.head_dim
        )
        # attn_output: (b, s, n_heads_local * d_k) = (b, s, h/N)

        # ── Output Projection (Row-Linear) ────────────────
        # ReduceScatter (if SP) or AllReduce happens inside.
        output = self.o_proj(attn_output)
        # output: (b, s/N, h) if SP, else (b, s, h)

        # ── Residual (SP region) ──────────────────────────
        output = residual + output

        return output
```

---

## 8. Full Transformer Block with TP + SP

```python
"""
tensor_parallel/transformer_block.py
─────────────────────────────────────
Complete Transformer decoder layer with TP + SP.

Architecture (Pre-Norm style):
    ┌──────────────────────────────────────┐
    │  Input X* ∈ (b, s/N, h)  [SP]       │
    │        │                             │
    │  ┌─────▼──────┐                      │
    │  │ LayerNorm   │  (SP: local on seq) │
    │  └─────┬──────┘                      │
    │  ┌─────▼──────┐                      │
    │  │  TP-MHA     │  (AllGather→QKV→    │
    │  │             │   Attn→Proj→        │
    │  │             │   ReduceScatter)    │
    │  └─────┬──────┘                      │
    │  ┌─────▼──────┐                      │
    │  │ + Residual  │  (SP region)        │
    │  └─────┬──────┘                      │
    │  ┌─────▼──────┐                      │
    │  │ LayerNorm   │  (SP: local on seq) │
    │  └─────┬──────┘                      │
    │  ┌─────▼──────┐                      │
    │  │  TP-MLP     │  (AllGather→FC1→    │
    │  │             │   GeLU→FC2→         │
    │  │             │   ReduceScatter)    │
    │  └─────┬──────┘                      │
    │  ┌─────▼──────┐                      │
    │  │ + Residual  │  (SP region)        │
    │  └─────┬──────┘                      │
    │  Output X'* ∈ (b, s/N, h)  [SP]     │
    └──────────────────────────────────────┘

Communication per layer (forward, TP+SP):
    MHA:  1 × AllGather  +  1 × ReduceScatter
    MLP:  1 × AllGather  +  1 × ReduceScatter
    Total: 2 × AllGather + 2 × ReduceScatter
         ≡ 2 × AllReduce  (same as vanilla TP)
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from .attention import TensorParallelAttention
from .mlp import TensorParallelMLP


@dataclass
class TransformerConfig:
    """Model configuration matching common LLM architectures."""
    hidden_size: int = 4096
    intermediate_size: int = 11008     # Llama-style: ~2.7 × h
    num_attention_heads: int = 32
    num_kv_heads: int = 8              # GQA (Llama-3 style)
    num_layers: int = 32
    max_seq_len: int = 8192
    vocab_size: int = 128256
    layernorm_eps: float = 1e-5
    sequence_parallel: bool = True


class TensorParallelTransformerBlock(nn.Module):
    """
    Single Transformer decoder layer with TP + SP.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention = TensorParallelAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            sequence_parallel=config.sequence_parallel,
            max_seq_len=config.max_seq_len,
            eps=config.layernorm_eps,
        )

        self.mlp = TensorParallelMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            sequence_parallel=config.sequence_parallel,
            eps=config.layernorm_eps,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor = None,
    ) -> Tensor:
        """
        x : (b, s/N, h) if SP else (b, s, h)
        """
        # MHA sub-block (includes LayerNorm + residual)
        x = self.attention(x, attention_mask)

        # MLP sub-block (includes LayerNorm + residual)
        x = self.mlp(x)

        return x
```

---

## 9. TP-Aware Embedding and Output Layers

The embedding layer and final LM head also require TP-aware sharding. The vocabulary embedding is typically sharded along the vocabulary dimension (row-linear semantics):

$$
E = \text{Embedding}(\text{token\_ids}) = \text{Lookup}(W_{\text{emb}}), \quad W_{\text{emb}} \in \mathbb{R}^{V \times h}
$$

```python
"""
tensor_parallel/embedding.py
─────────────────────────────
Vocabulary-parallel embedding layer.

The embedding table W_emb ∈ ℝ^{V × h} is sharded along
the VOCABULARY dimension:

    W_emb_i ∈ ℝ^{(V/N) × h}

Each rank handles token IDs in the range:
    [rank * V/N,  (rank+1) * V/N)

Token IDs outside this range produce zero embeddings.
The final result is obtained via all-reduce (sum).

With SP: output is reduce-scattered along sequence dim.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .init import get_tp_state
from .comm import all_reduce, reduce_scatter_fwd_all_gather_bwd


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer sharded across vocabulary dimension.

    Parameters
    ----------
    vocab_size : int — total vocabulary size V
    hidden_size : int — embedding dimension h
    sequence_parallel : bool — scatter output along sequence dim
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        tp = get_tp_state()

        assert vocab_size % tp.tp_world_size == 0

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_per_rank = vocab_size // tp.tp_world_size
        self.vocab_start = tp.tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.sequence_parallel = sequence_parallel

        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_rank, hidden_size)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        input_ids : (b, s)  — token IDs in [0, V)

        Returns:
            (b, s, h)     if not SP
            (b, s/N, h)   if SP
        """
        # ── Mask out-of-range tokens for this rank ────────
        mask = (input_ids >= self.vocab_start) & (
            input_ids < self.vocab_end
        )
        local_ids = (input_ids - self.vocab_start) * mask.long()

        # ── Local embedding lookup ────────────────────────
        embeddings = nn.functional.embedding(local_ids, self.weight)

        # Zero out embeddings for out-of-range tokens
        embeddings = embeddings * mask.unsqueeze(-1).float()

        if self.sequence_parallel:
            # Reduce-scatter: sum across ranks + scatter along seq
            embeddings = reduce_scatter_fwd_all_gather_bwd(
                embeddings, scatter_dim=1
            )
        else:
            # All-reduce: sum partial embeddings across ranks
            embeddings = all_reduce(embeddings)

        return embeddings
```

---

## 10. Triton Kernel: Fused AllGather + GEMM (Overlap Communication with Compute)

This is a production-level optimization that **overlaps** the all-gather communication with the column-linear GEMM using **tile-based pipelining**. This partially hides the communication latency on the critical path.

The idea: as soon as one chunk of the all-gathered tensor arrives, begin computing the GEMM for that chunk while the next chunk is still in transit.

```python
"""
tensor_parallel/kernels/fused_allgather_gemm.py
────────────────────────────────────────────────
Overlaps AllGather with GEMM computation using
chunk-based pipelining.

Strategy:
    1. Split input X* into K micro-chunks along sequence dim
    2. Issue async AllGather for chunk[0]
    3. For i = 0 .. K-1:
         a. Wait for AllGather of chunk[i] to complete
         b. Launch GEMM for chunk[i]
         c. Issue async AllGather for chunk[i+1]  (if exists)
    4. Concatenate GEMM outputs

This achieves partial overlap:

    AllGather[0]  AllGather[1]  AllGather[2]  ...
                  GEMM[0]      GEMM[1]       GEMM[2]  ...

The total time approaches:
    T ≈ T_allgather_one_chunk + T_gemm_total
instead of:
    T = T_allgather_total + T_gemm_total
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import List

from ..init import get_tp_state


def fused_allgather_gemm(
    x_local: Tensor,
    weight: Tensor,
    num_chunks: int = 4,
) -> Tensor:
    """
    Fused AllGather + Linear (Column-Linear case).

    Parameters
    ----------
    x_local : (b, s/N, h)  — local sequence shard
    weight  : (h_out/N, h)  — local weight shard (column-linear)
    num_chunks : int         — pipeline depth

    Returns
    -------
    output : (b, s, h_out/N)

    Mathematical equivalence:
        X = AllGather(x_local)       # (b, s, h)
        output = X @ weight.T        # (b, s, h_out/N)
    """
    tp = get_tp_state()
    N = tp.tp_world_size

    if N == 1:
        return torch.nn.functional.linear(x_local, weight)

    b, s_local, h = x_local.shape
    s_full = s_local * N
    h_out = weight.shape[0]

    # ── Split local input into micro-chunks ───────────────
    # Each micro-chunk: (b, s_local / num_chunks, h)
    assert s_local % num_chunks == 0
    chunk_size = s_local // num_chunks

    # ── Allocate buffers ──────────────────────────────────
    # Full gathered tensor (built incrementally)
    gathered_buffer = torch.empty(
        (b, s_full, h), dtype=x_local.dtype, device=x_local.device
    )
    # Output buffer
    output = torch.empty(
        (b, s_full, h_out), dtype=x_local.dtype, device=x_local.device
    )

    # ── Create CUDA streams for overlap ───────────────────
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device=x_local.device)

    # Split the local input into micro-chunks
    local_chunks = list(x_local.split(chunk_size, dim=1))

    # For each micro-chunk, we need to gather across all N ranks
    # Each gathered micro-chunk has shape (b, chunk_size * N, h)?
    # No — we gather the same chunk_index from all ranks.

    # Actually: AllGather gathers x_local along dim=1
    # We pipeline this by splitting x_local into sub-chunks
    # and performing partial all-gathers.

    # ── Chunk-level pipeline ──────────────────────────────
    gathered_chunks: List[Tensor] = []

    for chunk_idx in range(num_chunks):
        # Prepare this micro-chunk
        local_chunk = local_chunks[chunk_idx].contiguous()

        # Allocate gathered micro-chunk buffer
        gathered_chunk = torch.empty(
            (b, chunk_size * N, h),
            dtype=x_local.dtype,
            device=x_local.device,
        )
        chunk_list = list(gathered_chunk.chunk(N, dim=1))

        # ── Async AllGather on comm stream ────────────────
        with torch.cuda.stream(comm_stream):
            dist.all_gather(
                chunk_list,
                local_chunk,
                group=tp.tp_group,
            )

        # ── GEMM for previous chunk on compute stream ────
        if chunk_idx > 0:
            prev_gathered = gathered_chunks[chunk_idx - 1]
            start = (chunk_idx - 1) * chunk_size * N
            end = start + chunk_size * N
            output[:, start:end, :] = torch.nn.functional.linear(
                prev_gathered, weight
            )

        # Sync: wait for current AllGather to complete
        compute_stream.wait_stream(comm_stream)
        gathered_chunks.append(gathered_chunk)

    # ── Final GEMM for last chunk ─────────────────────────
    last_gathered = gathered_chunks[-1]
    start = (num_chunks - 1) * chunk_size * N
    end = start + chunk_size * N
    output[:, start:end, :] = torch.nn.functional.linear(
        last_gathered, weight
    )

    return output
```

---

## 11. Weight Initialization and Checkpoint Loading

Loading pre-trained weights into TP-sharded layers requires careful index arithmetic to extract the correct shard for each rank.

```python
"""
tensor_parallel/checkpoint.py
──────────────────────────────
Utilities for loading pre-trained checkpoints into
TP-sharded models.

Key operations:
    - Column-linear: slice along dim=0 (output features)
    - Row-linear: slice along dim=1 (input features)
    - QKV projections: interleaved slicing by head index
    - Embedding: slice along dim=0 (vocabulary)
"""

import torch
from torch import Tensor
from typing import Dict

from .init import get_tp_state


def shard_column_linear_weight(
    full_weight: Tensor,
    full_bias: Tensor = None,
) -> Dict[str, Tensor]:
    """
    Shard a full weight matrix for column-linear parallelism.

    full_weight : (h_out, h_in)
    Returns dict with 'weight' and optionally 'bias' for this rank.

    Column-linear shards along dim=0 (output features):
        W_i = full_weight[rank * h_out/N : (rank+1) * h_out/N, :]
    """
    tp = get_tp_state()
    h_out = full_weight.shape[0]
    shard_size = h_out // tp.tp_world_size
    start = tp.tp_rank * shard_size
    end = start + shard_size

    result = {"weight": full_weight[start:end, :].contiguous()}
    if full_bias is not None:
        result["bias"] = full_bias[start:end].contiguous()
    return result


def shard_row_linear_weight(
    full_weight: Tensor,
    full_bias: Tensor = None,
) -> Dict[str, Tensor]:
    """
    Shard a full weight matrix for row-linear parallelism.

    full_weight : (h_out, h_in)
    Returns dict with 'weight' and optionally 'bias' for this rank.

    Row-linear shards along dim=1 (input features):
        W_i = full_weight[:, rank * h_in/N : (rank+1) * h_in/N]
    """
    tp = get_tp_state()
    h_in = full_weight.shape[1]
    shard_size = h_in // tp.tp_world_size
    start = tp.tp_rank * shard_size
    end = start + shard_size

    result = {"weight": full_weight[:, start:end].contiguous()}
    if full_bias is not None:
        # Bias is NOT sharded for row-linear (full dim output)
        result["bias"] = full_bias.contiguous()
    return result


def shard_qkv_weight(
    q_weight: Tensor,   # (h, h)
    k_weight: Tensor,   # (h_kv, h) where h_kv = n_kv * d_k
    v_weight: Tensor,   # (h_kv, h)
    num_q_heads: int,
    num_kv_heads: int,
) -> Dict[str, Tensor]:
    """
    Shard QKV weights for multi-head attention with GQA support.

    Q: shard by query head groups
    K, V: shard by KV head groups (may replicate if TP > n_kv)
    """
    tp = get_tp_state()
    d_k = q_weight.shape[0] // num_q_heads

    # ── Q heads for this rank ─────────────────────────────
    q_heads_per_rank = num_q_heads // tp.tp_world_size
    q_start = tp.tp_rank * q_heads_per_rank * d_k
    q_end = q_start + q_heads_per_rank * d_k
    q_shard = q_weight[q_start:q_end, :]

    # ── KV heads for this rank ────────────────────────────
    if num_kv_heads >= tp.tp_world_size:
        kv_heads_per_rank = num_kv_heads // tp.tp_world_size
        kv_start = tp.tp_rank * kv_heads_per_rank * d_k
        kv_end = kv_start + kv_heads_per_rank * d_k
        k_shard = k_weight[kv_start:kv_end, :]
        v_shard = v_weight[kv_start:kv_end, :]
    else:
        # TP > n_kv: replicate KV heads across ranks
        # Each rank gets the same KV heads (or a cycling subset)
        kv_idx = tp.tp_rank % num_kv_heads
        kv_start = kv_idx * d_k
        kv_end = kv_start + d_k
        k_shard = k_weight[kv_start:kv_end, :]
        v_shard = v_weight[kv_start:kv_end, :]

    return {
        "q_weight": q_shard.contiguous(),
        "k_weight": k_shard.contiguous(),
        "v_weight": v_shard.contiguous(),
    }
```

---

## 12. End-to-End Launch Script

```python
"""
launch_tp_training.py
─────────────────────
Complete example: initialize TP, build model, run forward+backward.

Launch command:
    torchrun --nproc_per_node=8 launch_tp_training.py

This creates TP=8 within a single node using NVLink.
"""

import torch
import torch.distributed as dist
from tensor_parallel.init import initialize_tensor_parallel
from tensor_parallel.transformer_block import (
    TensorParallelTransformerBlock,
    TransformerConfig,
)
from tensor_parallel.embedding import VocabParallelEmbedding


def main():
    # ── Step 1: Initialize distributed + TP ───────────────
    tp_degree = 8
    tp_state = initialize_tensor_parallel(tp_degree)
    device = torch.device(f"cuda:{tp_state.tp_rank}")
    torch.cuda.set_device(device)

    print(
        f"[Rank {tp_state.global_rank}] "
        f"TP rank={tp_state.tp_rank}/{tp_state.tp_world_size}"
    )

    # ── Step 2: Model configuration ───────────────────────
    config = TransformerConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,   # 32 Q heads
        num_kv_heads=8,           # 8 KV heads (GQA)
        num_layers=32,
        max_seq_len=8192,
        vocab_size=128256,
        sequence_parallel=True,
    )

    # ── Step 3: Build model ───────────────────────────────
    embedding = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        sequence_parallel=config.sequence_parallel,
    ).to(device)

    layers = torch.nn.ModuleList([
        TensorParallelTransformerBlock(config).to(device)
        for _ in range(config.num_layers)
    ])

    # ── Step 4: Dummy input ───────────────────────────────
    batch_size = 2
    seq_len = 4096
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    # ── Step 5: Forward pass ──────────────────────────────
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Embedding: (b, s) → (b, s/N, h) if SP
        x = embedding(input_ids)

        # Transformer layers
        for layer in layers:
            x = layer(x)

    # ── Step 6: Backward pass ─────────────────────────────
    loss = x.sum()  # dummy loss
    loss.backward()

    # ── Step 7: Verify shapes ─────────────────────────────
    expected_s = seq_len // tp_degree if config.sequence_parallel else seq_len
    assert x.shape == (batch_size, expected_s, config.hidden_size), (
        f"Expected shape ({batch_size}, {expected_s}, {config.hidden_size}), "
        f"got {x.shape}"
    )

    if tp_state.tp_rank == 0:
        print(f"Forward+backward completed. Output shape: {x.shape}")
        print(f"Activation memory per GPU: "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
```

---

## 13. Communication Volume Analysis

For a single Transformer layer with hidden size $ h $, sequence length $ s $, batch size $ b $, and TP degree $ N $:

### 13.1 Vanilla TP

| Operation | Count | Data Volume per Op | Total |
|-----------|-------|--------------------|-------|
| All-Reduce (MHA) | 1 | $ 2bsh \cdot \frac{N-1}{N} $ | — |
| All-Reduce (MLP) | 1 | $ 2bsh \cdot \frac{N-1}{N} $ | — |
| **Total (fwd)** | 2 | — | $ 4bsh \cdot \frac{N-1}{N} $ |

The factor $ 2 \cdot \frac{N-1}{N} $ comes from Ring AllReduce = Reduce-Scatter + All-Gather, each transferring $ \frac{N-1}{N} $ of the data.

### 13.2 TP + SP

| Operation | Count | Data Volume per Op | Total |
|-----------|-------|--------------------|-------|
| All-Gather (SP→TP, MHA) | 1 | $ bsh \cdot \frac{N-1}{N} $ | — |
| Reduce-Scatter (TP→SP, MHA) | 1 | $ bsh \cdot \frac{N-1}{N} $ | — |
| All-Gather (SP→TP, MLP) | 1 | $ bsh \cdot \frac{N-1}{N} $ | — |
| Reduce-Scatter (TP→SP, MLP) | 1 | $ bsh \cdot \frac{N-1}{N} $ | — |
| **Total (fwd)** | 4 | — | $ 4bsh \cdot \frac{N-1}{N} $ |

$$
\boxed{C_{\text{TP}} = C_{\text{TP+SP}} = 4bsh \cdot \frac{N-1}{N} \text{ per layer (forward)}}
$$

The communication volumes are **identical** — SP provides strictly better memory with zero additional communication cost.

---

## 14. Memory Savings Summary

For a model with $ L $ layers, hidden size $ h $, sequence length $ s $, batch size $ b $:

### Peak Activation Memory Per GPU

| Method | Peak Activation Per Layer |
|--------|--------------------------|
| No parallelism | $ b \cdot s \cdot h $ |
| TP only (degree $N$) | $ b \cdot s \cdot h $ (LayerNorm/Dropout regions) |
| TP + SP (degree $N$) | $ \dfrac{b \cdot s \cdot h}{N} $ |

### Model State Memory Per GPU (Mixed Precision + Adam)

$$
M_{\text{model}}^{\text{per GPU}} = \frac{16P}{N} \text{ bytes}
$$

where $ P $ = total parameters, $ N $ = TP degree, and $ 16 = 2_{\text{bf16 params}} + 2_{\text{bf16 grads}} + 4_{\text{fp32 master}} + 4_{\text{momentum}} + 4_{\text{variance}} $.

---

## 15. Design Decision Matrix

| Scenario | Recommended Config | Rationale |
|----------|-------------------|-----------|
| $ P < 7\text{B} $, single node | TP=2 or TP=4 + SP | Minimize comm overhead |
| $ P \approx 7\text{B} $–$ 13\text{B} $ | TP=8 + SP (full node) | Fits within NVLink domain |
| $ P \approx 70\text{B} $ | TP=8 + SP + PP across nodes | Avoid inter-node TP |
| $ s > 32\text{K} $ tokens | TP=8 + SP + **Context Parallelism** | Shard attention along seq |
| $ n_{kv} < N $ (GQA) | TP ≤ $ n_{kv} $ preferred | Avoid KV head replication complexity |