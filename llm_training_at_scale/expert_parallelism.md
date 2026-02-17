

# Expert Parallelism and 5D Parallelism: A Comprehensive Technical Treatment

---

## 1. Expert Parallelism (EP)

### 1.1 Prerequisite: Mixture of Experts (MoE) Architecture

In a standard Transformer layer, every token passes through a single feedforward network (FFN). The Mixture of Experts paradigm replaces this monolithic FFN with $N_E$ parallel expert sub-networks $\{E_1, E_2, \ldots, E_{N_E}\}$, each structurally identical but with independent learned parameters. A **gating (router) network** $G$ determines which experts process each token.

For a given input hidden state $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$, the router computes gating scores:

$$g(\mathbf{h}) = \text{Softmax}\!\bigl(W_g \cdot \mathbf{h}\bigr) \in \mathbb{R}^{N_E}$$

where $W_g \in \mathbb{R}^{N_E \times d_{\text{model}}}$ is the learnable gating weight matrix.

Under a **Top-$k$** routing strategy (introduced in Shazeer et al., 2017, and refined in Switch Transformers), only $k$ experts with the highest gating scores are selected. Let $\mathcal{T}_k(\mathbf{h})$ denote the index set of the top-$k$ experts. The MoE layer output is:

$$\text{MoE}(\mathbf{h}) = \sum_{i \in \mathcal{T}_k(\mathbf{h})} g_i(\mathbf{h}) \cdot E_i(\mathbf{h})$$

where $g_i(\mathbf{h})$ is the (renormalized) gating weight for expert $i$, and $E_i(\mathbf{h})$ is the output of expert $i$ applied to $\mathbf{h}$.

**Key property:** Each token activates only $k \ll N_E$ experts, meaning total FLOPs per token remain comparable to a dense model, while the total parameter count scales as $\mathcal{O}(N_E \cdot d_{\text{ffn}} \cdot d_{\text{model}})$.

---

### 1.2 Definition of Expert Parallelism

**Expert Parallelism (EP)** is a distributed training and inference strategy that partitions the $N_E$ expert sub-networks across $W_{\text{EP}}$ workers (GPUs), such that each worker holds a disjoint subset of experts.

Formally, if we have $N_E$ experts and $W_{\text{EP}}$ workers in the expert-parallel group, worker $w$ hosts experts:

$$\mathcal{E}_w = \left\{ E_i \;\middle|\; i \in \left[\frac{w \cdot N_E}{W_{\text{EP}}},\; \frac{(w+1) \cdot N_E}{W_{\text{EP}}} - 1\right] \right\}$$

Each worker stores $\displaystyle\frac{N_E}{W_{\text{EP}}}$ experts' parameters in its local memory.

---

### 1.3 Communication Pattern: All-to-All Dispatch and Combine

Since tokens on any given worker may be routed to experts residing on any other worker, EP requires **All-to-All** collective communication operations:

**Step 1 — Dispatch (All-to-All scatter):** Each worker determines, for every local token, which remote worker hosts the assigned expert(s). The token hidden states $\mathbf{h}$ are sent to the appropriate workers.

If worker $w$ has $B_w$ tokens and token $j$ is routed to expert $E_i$ on worker $w'$, then $\mathbf{h}_j \in \mathbb{R}^{d_{\text{model}}}$ is transmitted from worker $w$ to worker $w'$.

**Step 2 — Expert Computation:** Each worker processes all tokens routed to its local experts. Worker $w$ computes:

$$\mathbf{o}_j = E_i(\mathbf{h}_j), \quad \forall\; \mathbf{h}_j \text{ routed to } E_i \in \mathcal{E}_w$$

**Step 3 — Combine (All-to-All gather):** The expert outputs $\mathbf{o}_j$ are sent back to the originating workers. The originating worker aggregates:

$$\text{MoE}(\mathbf{h}_j) = \sum_{i \in \mathcal{T}_k(\mathbf{h}_j)} g_i(\mathbf{h}_j) \cdot \mathbf{o}_j^{(i)}$$

The total communication volume per MoE layer for EP is:

$$V_{\text{EP}} = 2 \times B_{\text{local}} \times k \times d_{\text{model}} \times \left(1 - \frac{1}{W_{\text{EP}}}\right)$$

The factor of 2 accounts for both dispatch and combine phases. The term $(1 - 1/W_{\text{EP}})$ reflects that tokens routed to local experts require no communication.

---

### 1.4 Contrast with Tensor Parallelism

A critical distinction: **EP does not shard individual matrix multiplications.** In Tensor Parallelism (TP), a single linear layer's weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is partitioned across workers, requiring AllReduce or ReduceScatter/AllGather operations to reconstruct the full output. In EP, each expert's weight matrices remain **intact** on a single worker — the parallelism arises from placing *different complete experts* on different workers.

| Aspect | Tensor Parallelism | Expert Parallelism |
|---|---|---|
| What is sharded | A single weight matrix $W$ | Distinct expert networks $E_i$ |
| Communication primitive | AllReduce / ReduceScatter + AllGather | All-to-All |
| Communication content | Partial matrix products (activations) | Full token hidden states |
| Requires model-specific logic | Yes (column/row split patterns) | Minimal (only routing logic) |
| Weight integrity per worker | Partial weights | Complete expert weights |

---

### 1.5 Why EP Alone Is Insufficient: Combination with Data Parallelism

EP only partitions the MoE (FFN) layers. All **non-MoE** components — embedding layers, attention layers, LayerNorm, output heads — remain **fully replicated** across EP workers. Without additional parallelism, every EP worker processes the **same** input batch through these shared components, resulting in **redundant computation**.

This is resolved by combining EP with **Data Parallelism (DP)**. With $W_{\text{DP}}$ data-parallel replicas and $W_{\text{EP}}$ expert-parallel workers, the total worker count for these two dimensions is:

$$W_{\text{total}} = W_{\text{DP}} \times W_{\text{EP}}$$

Under this hybrid scheme:

- **Non-MoE layers:** Each worker processes a distinct micro-batch shard (standard DP behavior). Gradients are synchronized via AllReduce across $W_{\text{DP}}$ replicas.
- **MoE layers:** Tokens are routed across $W_{\text{EP}}$ workers hosting different experts via All-to-All communication.

This eliminates redundancy: each GPU processes a unique data shard through the shared layers, and expert computation is distributed without replication.

---

### 1.6 Practical Engineering: Communication-Aware Routing Constraints

Naive expert routing can create prohibitive All-to-All traffic across nodes. **DeepSeek-V3** (with $N_E = 256$ experts, Top-$k=8$ routing) introduces a **node-bounded routing constraint**: each token is restricted to be sent to at most $M$ nodes (in their case, $M = 4$).

Formally, let $\mathcal{N}(i)$ denote the node hosting expert $E_i$. The constrained routing enforces:

$$\left|\left\{\mathcal{N}(i) \;\middle|\; i \in \mathcal{T}_k(\mathbf{h})\right\}\right| \leq M$$

This bound reduces cross-node communication volume by a factor of approximately $\frac{M}{\text{total nodes}}$, keeping most All-to-All traffic within the high-bandwidth intra-node interconnect (e.g., NVLink at 900 GB/s) rather than the lower-bandwidth inter-node fabric (e.g., InfiniBand at 400 Gb/s).

---

### 1.7 Memory Impact of Expert Parallelism

Per-worker parameter memory for the MoE layers reduces linearly:

$$\text{Mem}_{\text{EP}}^{\text{experts}} = \frac{N_E \cdot \text{Params}(E_i)}{W_{\text{EP}}}$$

where $\text{Params}(E_i) = 2 \times d_{\text{model}} \times d_{\text{ffn}}$ for a standard two-layer FFN expert (ignoring biases). For DeepSeek-V3 with 256 experts and $W_{\text{EP}} = 64$, each worker holds only 4 experts.

Activation memory per worker depends on the number of tokens routed to local experts, which is governed by the load balancing properties of the router.

---

## 2. 5D Parallelism: Unified Framework

### 2.1 The Five Parallelism Dimensions

Modern large-scale training decomposes the computation along **five orthogonal dimensions**, each addressing a distinct axis of the training tensor:

| Strategy | Abbreviation | Parallel/Sharding Dimension | What Is Partitioned |
|---|---|---|---|
| Data Parallelism | DP | Batch dimension $B$ | Input samples |
| Tensor Parallelism | TP | Hidden dimension $d_{\text{model}}$ | Weight matrices and activations |
| Sequence/Context Parallelism | SP/CP | Sequence dimension $s$ | Token sequences |
| Pipeline Parallelism | PP | Model depth (layers) $L$ | Transformer layers |
| Expert Parallelism | EP | Expert dimension $N_E$ | Expert sub-networks |

The total number of GPUs $W$ satisfies:

$$W = W_{\text{DP}} \times W_{\text{TP}} \times W_{\text{PP}} \times W_{\text{CP}} \times W_{\text{EP}}$$

### 2.2 ZeRO Strategies (Orthogonal Memory Optimizations on DP)

ZeRO (Zero Redundancy Optimizer) is not a separate parallelism dimension but a set of memory optimization stages applied **within the DP group** of size $W_{\text{DP}}$:

| Stage | What Is Sharded Across DP Replicas | Memory Reduction Factor (Approx.) |
|---|---|---|
| ZeRO-1 | Optimizer states $O$ | Optimizer memory $\div W_{\text{DP}}$ |
| ZeRO-2 | Optimizer states $O$ + Gradients $G$ | $(O + G) \div W_{\text{DP}}$ |
| ZeRO-3 | Optimizer states $O$ + Gradients $G$ + Parameters $\Theta$ | $(O + G + \Theta) \div W_{\text{DP}}$ |

For a model with $\Phi$ parameters in mixed precision (fp16 params + fp32 optimizer), the per-GPU memory without ZeRO is:

$$M_{\text{base}} = 2\Phi + 2\Phi + (4\Phi + 4\Phi + 4\Phi) = 16\Phi \text{ bytes}$$

where $2\Phi$ for fp16 parameters, $2\Phi$ for fp16 gradients, and $12\Phi$ for Adam optimizer states (fp32 copy, first moment, second moment).

With ZeRO-3 across $W_{\text{DP}}$ workers:

$$M_{\text{ZeRO-3}} = \frac{16\Phi}{W_{\text{DP}}} + M_{\text{activations}}$$

---

## 3. Comparative Analysis: Pipeline Parallelism vs. ZeRO-3

Both PP and ZeRO-3 distribute model parameters across GPUs along the **model depth** axis, but they differ fundamentally in mechanism:

### 3.1 Side-by-Side Comparison

| Property | ZeRO-3 | Pipeline Parallelism |
|---|---|---|
| Per-worker storage | A fraction of each layer's parameters | Complete layers (one or more full layers) |
| Communication transfers | **Weights** (AllGather before forward, ReduceScatter after backward) | **Activations** (point-to-point between pipeline stages) |
| Orchestration complexity | Model-agnostic (automatic parameter gathering) | Model-agnostic but requires schedule design (1F1B, interleaved, etc.) |
| Implementation challenge | Managing parameter partitioning, prefetching, and communication overlap | Managing micro-batch scheduling to minimize pipeline bubble |
| Scaling preference | Large $\text{mbs}$ and $\text{seq\_len}$ to amortize weight communication | Large $\text{grad\_acc}$ (many micro-batches) to minimize bubble ratio |

### 3.2 Why Combining PP + ZeRO-3 Is Rare

When combining PP and ZeRO-3, both weight communication (ZeRO-3) and activation communication (PP) occur simultaneously. The total communication cost becomes:

$$C_{\text{combined}} = C_{\text{ZeRO-3}}^{\text{weights}} + C_{\text{PP}}^{\text{activations}}$$

To amortize both costs, the global batch size $B_{\text{global}}$ must be increased significantly:

$$B_{\text{global}} = \text{mbs} \times W_{\text{DP}} \times \text{grad\_acc}$$

This creates a multi-dimensional trade-off between global batch size, model size, network bandwidth, and training convergence (since excessively large batch sizes can degrade final model quality).

**Practical guidance:** If combining them, ZeRO-3 should be configured to **retain parameters in memory** during the sequence of PP micro-batches, avoiding repeated AllGather operations for the same parameters across micro-batches.

### 3.3 PP + ZeRO-1/ZeRO-2: Natural Combination

ZeRO-1 and ZeRO-2 shard only optimizer states (and gradients), which are only needed during the **optimizer step** — not during forward/backward computation. This means they introduce **no additional communication** during the PP micro-batch processing loop, making them naturally complementary.

**Real-world example:** DeepSeek-V3 training uses PP + ZeRO-1.

---

## 4. Tensor Parallelism + Sequence Parallelism: Interaction with Other Strategies

### 4.1 Natural Complementarity with PP and ZeRO-3

TP exploits the **distributive property of matrix multiplication**. For a linear layer $Y = XW$, with column-parallel sharding of $W$ into $[W_1, W_2]$ across 2 workers:

$$Y = X[W_1, W_2] = [XW_1, XW_2]$$

Each partial computation $XW_i$ is independent, and results are combined via AllGather or ReduceScatter. This operates on a **sub-layer** granularity, making it orthogonal to PP (which operates at layer granularity) and ZeRO-3 (which also operates at layer granularity for parameter gathering).

### 4.2 Two Fundamental Limitations of TP

**Limitation 1 — Communication on Critical Path:**

TP communication (AllReduce or equivalent) lies on the **critical path** of computation. For each Transformer layer, the communication cost scales as:

$$T_{\text{comm}}^{\text{TP}} = 4 \times \frac{2 \cdot (W_{\text{TP}} - 1)}{W_{\text{TP}}} \times \frac{b \times s \times d_{\text{model}}}{\text{BW}_{\text{intra}}}$$

where the factor 4 accounts for two linear layers × (forward + backward), $b$ is micro-batch size, $s$ is sequence length, and $\text{BW}_{\text{intra}}$ is the intra-node bandwidth. As $W_{\text{TP}}$ grows, the compute per worker shrinks as $\mathcal{O}(1/W_{\text{TP}})$ while communication remains $\mathcal{O}(1)$, leading to diminishing returns.

**Limitation 2 — Model-Specific Implementation:**

TP requires explicit knowledge of **where to shard along the hidden dimension** (TP regions) vs. **where to shard along the sequence dimension** (SP regions). Attention projections, FFN layers, LayerNorm, and dropout each require different sharding strategies, making TP non-trivially model-specific.

### 4.3 Consequence: TP Confined to Intra-Node

Given these limitations, TP is kept within high-bandwidth intra-node interconnects (e.g., 8 GPUs connected via NVLink at 900 GB/s per GPU), while PP or ZeRO-3 handles inter-node distribution over lower-bandwidth fabrics (e.g., InfiniBand at 400 Gb/s per link).

---

## 5. Context Parallelism (CP): Complementary to TP

### 5.1 What CP Targets

CP shards activations along the **sequence length dimension** $s$ across $W_{\text{CP}}$ workers. Each worker processes a contiguous subsequence of length $s / W_{\text{CP}}$.

- **MLP, LayerNorm:** These are point-wise or token-independent operations and process sharded sequences **without any communication**.
- **Attention layers:** Each token's query must attend to keys/values from the **full sequence**, requiring communication.

### 5.2 Ring Attention for CP

Ring Attention organizes $W_{\text{CP}}$ workers in a logical ring. Each worker holds a local KV shard and iteratively passes KV blocks to the next worker while computing partial attention on the current KV block. After $W_{\text{CP}} - 1$ communication steps, every worker has attended to all KV blocks.

The communication is overlapped with computation, so the effective overhead is:

$$T_{\text{CP}} \approx \max\left(T_{\text{compute}}^{\text{attn}},\; T_{\text{comm}}^{\text{KV}}\right) \quad \text{(per ring step)}$$

where:

$$T_{\text{comm}}^{\text{KV}} = \frac{2 \times b \times (s / W_{\text{CP}}) \times d_{\text{model}}}{\text{BW}}$$

CP is specifically valuable for extreme sequence lengths ($s \geq 128\text{k}$ tokens), where even with full activation recomputation, attention activation memory $\mathcal{O}(b \times n_h \times s^2)$ (where $n_h$ is the number of attention heads) would exceed single-GPU memory.

---

## 6. Expert Parallelism (EP): Complementary to TP

EP targets the MoE FFN layers exclusively. **Attention layers, LayerNorm, embeddings, and output heads are completely unaffected** by EP. This makes EP orthogonal to:

- **TP/SP** (which shards attention and FFN weight matrices along hidden/sequence dims)
- **CP** (which shards attention KV along sequence dim)
- **PP** (which shards entire layers along depth)

### 6.1 EP vs. DP: Structural Similarity

There is a notable structural similarity between EP and DP regarding **input handling**:

- In **DP**, each worker processes different data through **identical** model copies.
- In **EP** (without additional DP), each worker processes the **same** data through different experts.

This duality is why some frameworks treat EP as a specialized variant of DP, where the "replication" is replaced by "expert routing." The critical difference is that EP workers hold **non-identical** model components (different experts), while DP workers hold **identical** model copies.

---

## 7. Scope of Each Parallelism Strategy Within a Transformer Layer

| Strategy | Attention Layers | FFN / MoE Layers | LayerNorm | Embeddings |
|---|---|---|---|---|
| **TP + SP** | ✅ Shards $W_Q, W_K, W_V, W_O$ along heads/hidden dim | ✅ Shards FFN weights along hidden dim | ✅ (SP: sharded along seq dim) | ✅ Shards embedding matrix |
| **CP** | ✅ **Primary impact** — requires KV communication | ⚪ Independent processing | ⚪ Independent | ⚪ Independent |
| **EP** | ⚪ Unchanged | ✅ **Primary impact** — experts distributed | ⚪ Unchanged | ⚪ Unchanged |
| **PP** | Entire layers assigned to stages | Entire layers assigned to stages | Part of the assigned stage | Often first/last stage (special handling) |
| **ZeRO** | Parameters sharded across DP replicas | Parameters sharded across DP replicas | Parameters sharded | Parameters sharded |

Legend: ✅ = directly affected, ⚪ = unaffected/independent operation.

---

## 8. Comprehensive Comparison Table

| Method | Memory Savings Target | Parallel Dimension | Primary Disadvantage |
|---|---|---|---|
| **DP** | Activations (reduced local batch) | Batch $B$ | Limited by maximum effective batch size |
| **PP** | Model parameters | Model layers $L$ | Pipeline bubble and complex scheduling |
| **TP + SP** | Parameters and activations | Hidden $d$ / sequence $s$ | Requires high-bandwidth intra-node interconnect |
| **CP** | Activations | Sequence length $s$ | Communication overhead in attention |
| **EP** | Expert parameters | Expert dimension $N_E$ | Requires MoE architecture; routing communication |
| **ZeRO-1** | Optimizer states | Sharded across DP replicas | Parameter communication overhead |
| **ZeRO-2** | Optimizer states + gradients | Sharded across DP replicas | Parameter communication overhead |
| **ZeRO-3** | Optimizer states + gradients + parameters | Sharded across DP replicas | Parameter communication overhead |

---

## 9. Interaction and Combination Rules: Practical Summary

### 9.1 Naturally Complementary Combinations

| Combination | Why It Works |
|---|---|
| **TP + PP** | TP shards within layers (intra-node); PP shards across layers (inter-node). Orthogonal axes. |
| **TP + DP** | TP within node, DP across nodes. Standard combination. |
| **PP + ZeRO-1/2** | ZeRO-1/2 shard optimizer/gradients (used only at optimizer step), not interfering with PP micro-batch processing. |
| **TP + CP** | TP shards hidden dim; CP shards sequence dim. Orthogonal. |
| **EP + DP** | EP distributes experts; DP distributes input batches. Eliminates redundant computation on shared layers. |
| **EP + CP** | EP targets MoE layers; CP targets attention. No interference. |

### 9.2 Combinations Requiring Caution

| Combination | Issue |
|---|---|
| **PP + ZeRO-3** | Both introduce communication on the depth axis. Requires very large batch sizes to amortize dual communication costs. Rarely used in practice. |
| **TP at large scale** ($W_{\text{TP}} > 8$) | Communication dominates compute. Typically restricted to $\leq 8$ GPUs within a single node. |

### 9.3 Typical Hierarchy in Practice

For training a model on a large GPU cluster with $H$ nodes of $G$ GPUs each:

$$\underbrace{W_{\text{TP}} = G}_{\text{intra-node}} \times \underbrace{W_{\text{PP}} \times W_{\text{DP}} \times W_{\text{CP}} \times W_{\text{EP}}}_{\text{inter-node}} = H \times G$$

- **Innermost (fastest interconnect):** TP (NVLink, ~900 GB/s)
- **Middle tier:** PP, CP, EP (InfiniBand, ~50–100 GB/s effective)
- **Outermost (most tolerant of latency):** DP with ZeRO (communication only at gradient sync / optimizer step)

---

## 10. Unified 5D Parallelism Diagram: Mathematical Formulation

For a single MoE Transformer layer, the computation on worker $w$ identified by its 5D coordinate $(w_{\text{DP}},\, w_{\text{TP}},\, w_{\text{PP}},\, w_{\text{CP}},\, w_{\text{EP}})$ proceeds as:

**Input:** A micro-batch shard $\mathbf{X}^{(w_{\text{DP}})} \in \mathbb{R}^{(\text{mbs}) \times (s/W_{\text{CP}}) \times d_{\text{model}}}$

**Attention block (TP + CP active):**

$$\mathbf{Q}_w = \mathbf{X}^{(w_{\text{DP}})} \cdot W_Q^{(w_{\text{TP}})}, \quad \mathbf{K}_w = \mathbf{X}^{(w_{\text{DP}})} \cdot W_K^{(w_{\text{TP}})}, \quad \mathbf{V}_w = \mathbf{X}^{(w_{\text{DP}})} \cdot W_V^{(w_{\text{TP}})}$$

CP Ring Attention gathers full $\mathbf{K}, \mathbf{V}$ across $W_{\text{CP}}$ workers for attention computation. TP AllReduce (or ReduceScatter) combines partial attention outputs across $W_{\text{TP}}$ workers.

**MoE FFN block (EP + TP active):**

$$\mathbf{h}_{\text{post-attn}} \xrightarrow{\text{Router}} \text{All-to-All dispatch to } W_{\text{EP}} \text{ workers} \xrightarrow{E_{i}^{(w_{\text{EP}})}} \text{All-to-All combine} \rightarrow \mathbf{h}_{\text{out}}$$

**Pipeline dimension:** The above describes a single stage $w_{\text{PP}}$ in the pipeline. Activations flow from stage $w_{\text{PP}}$ to stage $w_{\text{PP}} + 1$ via point-to-point communication.

**Gradient synchronization (DP + ZeRO):** After backward pass through all micro-batches, gradients are synchronized across $W_{\text{DP}}$ replicas via AllReduce (DP) or ReduceScatter (ZeRO-2/3).

---

## 11. Key Takeaway

No single parallelism strategy is a universal solution. Each addresses a specific dimension of the training tensor and introduces its own communication overhead. The art of large-scale training lies in **composing these strategies** such that:

1. Communication-intensive strategies (TP) use the fastest interconnects.
2. Computation-tolerant strategies (DP, ZeRO) span slower interconnects.
3. The global batch size remains within the convergence-optimal range.
4. Memory is balanced across all workers to avoid stragglers.

The optimal configuration is determined by the interplay of model architecture ($d_{\text{model}}, L, N_E, s$), hardware topology (intra-node vs. inter-node bandwidth), and training hyperparameters ($B_{\text{global}}, \text{mbs}, \text{grad\_acc}$).