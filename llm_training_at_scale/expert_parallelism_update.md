

# Expert Parallelism and 5D Parallelism: A Comprehensive Technical Treatment

---

## 1. Expert Parallelism (EP)

### 1.1 Prerequisite: Mixture of Experts (MoE) Architecture

In a standard Transformer layer, every token passes through a **single** feedforward network (FFN). In a **Mixture of Experts (MoE)** layer, the single FFN is replaced by $N_E$ parallel expert FFNs $\{E_1, E_2, \dots, E_{N_E}\}$ and a **gating (router) network** $G(\cdot)$ that determines which expert(s) each token should be sent to.

Given an input hidden state $\mathbf{h} \in \mathbb{R}^{d}$ for a single token, the MoE layer output is:

$$
\text{MoE}(\mathbf{h}) = \sum_{i=1}^{N_E} g_i(\mathbf{h}) \cdot E_i(\mathbf{h})
$$

where $g_i(\mathbf{h})$ is the gating weight for expert $i$, computed by the router:

$$
g_i(\mathbf{h}) = \begin{cases} \text{softmax}\bigl(\text{TopK}(\mathbf{W}_g \mathbf{h})\bigr)_i & \text{if } i \in \text{TopK indices} \\ 0 & \text{otherwise} \end{cases}
$$

Here $\mathbf{W}_g \in \mathbb{R}^{N_E \times d}$ is the learnable router weight matrix, and $\text{TopK}$ selects the $K$ experts with highest router logits. Typically $K \ll N_E$ (e.g., $K = 2$ out of $N_E = 256$ in DeepSeek-V3), meaning each token activates only a **sparse subset** of experts.

**Key property:** Each expert $E_i$ is a standard FFN:

$$
E_i(\mathbf{h}) = W_i^{(2)} \cdot \sigma\!\left(W_i^{(1)} \mathbf{h} + \mathbf{b}_i^{(1)}\right) + \mathbf{b}_i^{(2)}
$$

where $W_i^{(1)} \in \mathbb{R}^{d_{ff} \times d}$, $W_i^{(2)} \in \mathbb{R}^{d \times d_{ff}}$, and $\sigma(\cdot)$ is a nonlinearity (e.g., SiLU, GeLU). Since each expert is **fully independent** of every other expert, this creates a natural axis for parallelism.

---

### 1.2 Definition of Expert Parallelism

**Expert Parallelism (EP)** distributes the $N_E$ experts across $W_{EP}$ workers (GPUs), where each worker holds a disjoint subset of experts:

$$
\text{Worker } w \text{ holds experts: } \left\{ E_i \;\middle|\; i \in \left[\frac{(w-1) \cdot N_E}{W_{EP}} + 1, \;\; \frac{w \cdot N_E}{W_{EP}}\right] \right\}
$$

For $N_E$ experts distributed across $W_{EP}$ workers, each worker stores:

$$
\frac{N_E}{W_{EP}} \text{ experts}
$$

**Contrast with Tensor Parallelism (TP):** In TP, a single weight matrix $W \in \mathbb{R}^{m \times n}$ is **split** (column-wise or row-wise) across workers, requiring synchronized partial matrix multiplications followed by collective operations (AllReduce or AllGather). In EP, each expert's weight matrices are kept **intact** on a single worker — no matrix splitting is needed. The only communication required is **routing tokens** to the correct worker.

---

### 1.3 Communication Pattern: All-to-All

The fundamental communication primitive in EP is the **All-to-All** operation, which occurs twice per MoE layer:

#### Forward Pass

**Step 1 — Dispatch (All-to-All):** After the router computes gating decisions, each worker sends tokens destined for remote experts to the appropriate worker. Formally, if worker $w$ has a local batch of tokens $\{\mathbf{h}_1, \dots, \mathbf{h}_B\}$ and the router assigns token $\mathbf{h}_j$ to expert $E_i$ residing on worker $w'$, then $\mathbf{h}_j$ must be communicated from worker $w$ to worker $w'$.

**Step 2 — Compute:** Each worker processes received tokens through its local experts.

**Step 3 — Combine (All-to-All):** The expert outputs are sent back to the originating workers, weighted by gating scores, and summed.

The communication volume per MoE layer for a single token routed to $K$ experts is:

$$
V_{\text{comm}} = 2 \cdot K \cdot d \quad \text{(dispatch + combine, per token)}
$$

For a local batch of $B$ tokens across $W_{EP}$ workers:

$$
V_{\text{total}} = 2 \cdot B \cdot K \cdot d \cdot \left(1 - \frac{1}{W_{EP}}\right)
$$

The factor $\left(1 - \frac{1}{W_{EP}}\right)$ accounts for the fact that tokens assigned to local experts do not require inter-worker communication.

---

### 1.4 EP Combined with Data Parallelism (DP)

EP alone only parallelizes the MoE layers. All **non-MoE components** — self-attention, layer normalization, embeddings — remain **replicated** across all EP workers. This means:

- Without DP, every worker processes the **same** input batch through non-MoE layers → **redundant computation**.
- With DP, the input batch is **sharded** across workers, and each worker processes a different micro-batch through non-MoE layers → **no redundancy**.

Given $W$ total GPUs, $W_{EP}$ GPUs for expert parallelism, and $W_{DP}$ GPUs for data parallelism:

$$
W = W_{EP} \times W_{DP}
$$

The effective batch size per GPU for non-MoE layers becomes:

$$
B_{\text{local}} = \frac{B_{\text{global}}}{W_{DP}}
$$

while each GPU holds $\frac{N_E}{W_{EP}}$ experts for MoE layers.

**Gradient synchronization:** After the backward pass, gradients for non-MoE parameters are synchronized via AllReduce across the $W_{DP}$ data-parallel group, while expert gradients are **not** synchronized across the EP group (since each expert exists on exactly one worker).

---

### 1.5 Router Constraints for Communication Efficiency

**DeepSeek-V3 node-bounded routing constraint:** To minimize inter-node communication, the router enforces that each token is sent to at most $M$ nodes (DeepSeek-V3 uses $M = 4$). Formally, define the set of nodes as $\{\mathcal{N}_1, \dots, \mathcal{N}_{N_{\text{nodes}}}\}$, where each node contains a subset of experts. The router first selects the top-$M$ nodes by aggregate affinity:

$$
s_m = \sum_{i \in \mathcal{N}_m} (\mathbf{W}_g \mathbf{h})_i, \quad m = 1, \dots, N_{\text{nodes}}
$$

$$
\mathcal{M}_{\text{selected}} = \text{TopM}(\{s_1, \dots, s_{N_{\text{nodes}}}\})
$$

Then the top-$K$ experts are selected **only** from experts within $\mathcal{M}_{\text{selected}}$. This ensures that the All-to-All communication is bounded to at most $M$ nodes, dramatically reducing cross-node bandwidth requirements:

$$
V_{\text{inter-node}} \leq 2 \cdot B \cdot K \cdot d \cdot \frac{M - 1}{N_{\text{nodes}}} \quad \ll \quad V_{\text{unbounded}}
$$

---

## 2. 5D Parallelism: Complete Taxonomy

### 2.1 The Five Axes of Parallelism

Modern large-scale training combines **five** orthogonal parallelism strategies, each sharding along a different dimension of the computation:

| **Strategy** | **Symbol** | **Sharding Dimension** | **What is Distributed** |
|---|---|---|---|
| Data Parallelism | $\text{DP}$ | Batch | Input micro-batches |
| Tensor Parallelism | $\text{TP}$ | Hidden dimension $d$ | Weight matrices & activations |
| Sequence/Context Parallelism | $\text{SP/CP}$ | Sequence length $L$ | Activations along sequence |
| Pipeline Parallelism | $\text{PP}$ | Model depth (layers) | Consecutive layer groups |
| Expert Parallelism | $\text{EP}$ | Expert index | MoE expert FFNs |

The total number of GPUs satisfies:

$$
W_{\text{total}} = W_{DP} \times W_{TP} \times W_{PP} \times W_{CP} \times W_{EP}
$$

### 2.2 The Three ZeRO Stages (Complementary to DP)

ZeRO (Zero Redundancy Optimizer) progressively eliminates memory redundancy **within** the DP group:

| **Stage** | **Sharded Among DP Replicas** | **Memory per GPU** |
|---|---|---|
| ZeRO-1 | Optimizer states $\mathcal{O}$ | $\frac{\|\mathcal{O}\|}{W_{DP}} + \|\Theta\| + \|G\|$ |
| ZeRO-2 | Optimizer states $\mathcal{O}$ + Gradients $G$ | $\frac{\|\mathcal{O}\| + \|G\|}{W_{DP}} + \|\Theta\|$ |
| ZeRO-3 | Optimizer states $\mathcal{O}$ + Gradients $G$ + Parameters $\Theta$ | $\frac{\|\mathcal{O}\| + \|G\| + \|\Theta\|}{W_{DP}}$ |

For a model with $\Phi$ parameters in mixed-precision training (fp16 params + fp32 optimizer states with Adam):

| Component | Bytes per parameter |
|---|---|
| Parameters (fp16) | 2 |
| Gradients (fp16) | 2 |
| Optimizer: fp32 params copy | 4 |
| Optimizer: first moment | 4 |
| Optimizer: second moment | 4 |
| **Total** | **16** |

Memory per GPU under each stage:

$$
M_{\text{ZeRO-0}} = 16\Phi
$$

$$
M_{\text{ZeRO-1}} = 4\Phi + \frac{12\Phi}{W_{DP}}
$$

$$
M_{\text{ZeRO-2}} = 2\Phi + \frac{14\Phi}{W_{DP}}
$$

$$
M_{\text{ZeRO-3}} = \frac{16\Phi}{W_{DP}}
$$

---

## 3. Pipeline Parallelism vs. ZeRO-3: Detailed Comparison

Both PP and ZeRO-3 distribute model parameters across GPUs along the **depth axis**, but they differ fundamentally in **what** is communicated and **how** computation is organized.

### 3.1 Side-by-Side Analysis

| **Aspect** | **ZeRO-3** | **Pipeline Parallelism** |
|---|---|---|
| Storage per device | A **fraction** of every layer's parameters: $\frac{\|\Theta_\ell\|}{W_{DP}}$ per layer $\ell$ | **Full** parameters of assigned layers: $\|\Theta_\ell\|$ for $\ell \in \mathcal{L}_w$ |
| Communication transfers | **Weights** (AllGather before forward, ReduceScatter after backward) | **Activations** (point-to-point between pipeline stages) |
| Orchestration | Model-agnostic (wraps any model) | Model-agnostic (assigns layer groups to stages) |
| Implementation complexity | Complex parameter partitioning and prefetching | Complex scheduling (1F1B, interleaved, zero-bubble) |
| Scaling preference | Prefers large $\text{mbs} \times L$ to hide weight communication | Prefers large $\text{grad\_acc}$ steps to amortize pipeline bubble |

### 3.2 Communication Volume Comparison

**ZeRO-3** per layer (forward + backward): Each layer's full parameters must be AllGathered before computation and optionally ReduceScattered during backward:

$$
V_{\text{ZeRO-3}}^{(\ell)} = 2 \cdot |\Theta_\ell| \cdot \frac{W_{DP} - 1}{W_{DP}} \quad \text{(AllGather + ReduceScatter)}
$$

Summed over all $L$ layers:

$$
V_{\text{ZeRO-3}}^{\text{total}} = 2 \sum_{\ell=1}^{L} |\Theta_\ell| \cdot \frac{W_{DP} - 1}{W_{DP}} \approx 2|\Theta| \quad \text{for large } W_{DP}
$$

**Pipeline Parallelism** per micro-batch boundary: Only the activation tensor at stage boundaries is communicated (point-to-point):

$$
V_{\text{PP}}^{(\text{boundary})} = B_{\mu} \cdot L_{\text{seq}} \cdot d \quad \text{(per boundary, per micro-batch)}
$$

Total PP communication across $S - 1$ stage boundaries and $n_{\mu}$ micro-batches:

$$
V_{\text{PP}}^{\text{total}} = n_{\mu} \cdot (S - 1) \cdot B_{\mu} \cdot L_{\text{seq}} \cdot d
$$

where $S = W_{PP}$ is the number of pipeline stages, $B_{\mu}$ is the micro-batch size, and $d$ is the hidden dimension.

### 3.3 Combinability

- **ZeRO-3 + PP:** Possible but rarely practical. Combining them requires inflating the global batch size to amortize **both** weight-transfer overhead (ZeRO-3) and bubble overhead (PP). If combined, ZeRO-3 should **cache** (keep in memory) the gathered parameters across all PP micro-batches to avoid re-gathering per micro-batch:

$$
\text{Redundant gathers avoided} = (n_{\mu} - 1) \times V_{\text{ZeRO-3}}^{\text{total}}
$$

- **ZeRO-1/ZeRO-2 + PP:** Naturally complementary. ZeRO-1/2 shard optimizer states and gradients (which are only needed at the update step), while PP shards model layers. No conflicting communication patterns. **DeepSeek-V3 uses this combination (PP + ZeRO-1).**

---

## 4. Interactions Between Parallelism Strategies

### 4.1 Tensor Parallelism + Sequence Parallelism (TP+SP)

TP shards weight matrices along the hidden dimension $d$:

$$
W = [W^{(1)} | W^{(2)} | \cdots | W^{(W_{TP})}], \quad W^{(k)} \in \mathbb{R}^{m \times (n / W_{TP})}
$$

The matrix multiplication $\mathbf{y} = W\mathbf{x}$ is computed as:

$$
\mathbf{y} = \sum_{k=1}^{W_{TP}} W^{(k)} \mathbf{x}^{(k)} \quad \text{(row-parallel)}
$$

or

$$
\mathbf{y}^{(k)} = W^{(k)} \mathbf{x} \quad \text{(column-parallel)}
$$

requiring AllReduce or AllGather/ReduceScatter operations.

**SP** complements TP by sharding activations along the **sequence dimension** $L$ in regions where TP is not active (e.g., LayerNorm, Dropout):

$$
\mathbf{X} \in \mathbb{R}^{L \times d} \rightarrow \mathbf{X}^{(k)} \in \mathbb{R}^{(L/W_{TP}) \times d}, \quad k = 1, \dots, W_{TP}
$$

**Why TP should be intra-node:** TP operations (AllReduce, AllGather) lie on the **critical computation path** — the forward pass cannot proceed until the collective is complete. Thus TP demands the highest bandwidth interconnect (NVLink/NVSwitch at 900 GB/s), making it suitable **only** for intra-node communication:

$$
T_{\text{TP-comm}} = \frac{2 \cdot (W_{TP} - 1)}{W_{TP}} \cdot \frac{|\mathbf{a}|}{\text{BW}_{\text{intra-node}}}
$$

where $|\mathbf{a}|$ is the activation size per collective operation.

### 4.2 Context Parallelism (CP)

CP shards the full sequence across $W_{CP}$ workers:

$$
\mathbf{X} \in \mathbb{R}^{L \times d} \rightarrow \mathbf{X}^{(k)} \in \mathbb{R}^{(L/W_{CP}) \times d}, \quad k = 1, \dots, W_{CP}
$$

- **MLP, LayerNorm:** Process sharded chunks independently — no communication needed.
- **Self-Attention:** Each token's query must attend to **all** keys/values across the full sequence. This requires the **Ring Attention** pattern:

At each step $t$ of the ring ($t = 0, 1, \dots, W_{CP} - 1$), worker $k$:
1. Computes partial attention with the locally available $\mathbf{K}^{(j)}, \mathbf{V}^{(j)}$ block.
2. Sends its KV block to neighbor $(k+1) \mod W_{CP}$ and receives from $(k-1) \mod W_{CP}$.
3. Accumulates partial attention using the **online softmax** correction.

The online softmax update at step $t$ for query $\mathbf{q}$ on worker $k$ is:

$$
m^{(t)} = \max\!\left(m^{(t-1)},\; \max_j \frac{\mathbf{q}^\top \mathbf{k}_j^{(t)}}{\sqrt{d_k}}\right)
$$

$$
\ell^{(t)} = \ell^{(t-1)} e^{m^{(t-1)} - m^{(t)}} + \sum_j e^{\mathbf{q}^\top \mathbf{k}_j^{(t)} / \sqrt{d_k} - m^{(t)}}
$$

$$
\mathbf{o}^{(t)} = \frac{\ell^{(t-1)} e^{m^{(t-1)} - m^{(t)}} \mathbf{o}^{(t-1)} + \sum_j e^{\mathbf{q}^\top \mathbf{k}_j^{(t)} / \sqrt{d_k} - m^{(t)}} \mathbf{v}_j^{(t)}}{\ell^{(t)}}
$$

After $W_{CP}$ steps, $\mathbf{o}^{(W_{CP}-1)}$ is the exact attention output.

CP is especially valuable when:

$$
L \geq 128{,}000 \quad \text{tokens}
$$

since even with full activation recomputation, the memory for attention scales as $\mathcal{O}\!\left(\frac{L^2}{W_{CP}}\right)$ per worker.

### 4.3 Expert Parallelism (EP) — Complementary to TP

EP and TP both operate at the **sub-layer** level but on **different** sub-layers:

- **TP:** Shards the weight matrices of attention projections and (non-expert) FFNs.
- **EP:** Shards expert FFNs across workers.

These are naturally **non-overlapping** and can be combined without conflict. In an MoE Transformer layer:

$$
\underbrace{\text{Self-Attention}}_{\text{TP sharded}} \;\rightarrow\; \underbrace{\text{MoE FFN}}_{\text{EP sharded}} \;\rightarrow\; \text{LayerNorm}
$$

### 4.4 Similarity Between EP and DP

Both EP and DP involve multiple workers processing **different** data through **different** (or identical) parameters:

| | **DP** | **EP** |
|---|---|---|
| Each worker processes | Different micro-batch | Different tokens (routed by expert assignment) |
| Model weights per worker | Identical full copy | Unique subset of experts |
| Gradient sync | AllReduce across DP group | None across EP group (each expert is unique) |

Some frameworks treat EP as a **specialized form of DP** where, instead of identical model replicas processing different batches, each replica holds different experts and receives dynamically routed tokens.

---

## 5. Scope and Focus of Each Strategy

### 5.1 Per-Component Impact

| **Strategy** | **Attention Layers** | **FFN / MoE Layers** | **LayerNorm / Embeddings** |
|---|---|---|---|
| TP + SP | Shards $W_Q, W_K, W_V, W_O$ along $d$ | Shards $W_1, W_2$ along $d_{ff}$ or $d$ | SP shards along $L$ |
| CP | Requires Ring Attention communication | Independent on sharded sequences | Independent on sharded sequences |
| EP | Unchanged | Shards experts across workers | Unchanged |
| PP | Assigned to a pipeline stage | Assigned to a pipeline stage | Assigned (often treated specially due to embedding tying) |
| ZeRO | Shards params/grads/optim states uniformly | Shards params/grads/optim states uniformly | Shards params/grads/optim states uniformly |

### 5.2 Detailed Comparison Table

| | **TP + SP** | **CP** | **EP** |
|---|---|---|---|
| **What is sharded** | Weights & activations along $d$ / $L$ | Activations along $L$ | Expert weights & activations |
| **Communication ops** | AllReduce / AllGather / ReduceScatter for matmul | P2P ring for attention KV | All-to-All for token dispatch/combine |
| **Implementation** | Model-specific (sharding patterns vary per layer type) | Model-agnostic except attention | Model-agnostic except MoE layers |
| **Bandwidth preference** | High-bandwidth intra-node | Moderate (overlapped with compute) | Moderate (bounded by routing constraints) |
| **Prerequisite** | Any model with large matrices | Long sequences | MoE architecture |

---

## 6. Unified 5D Parallelism Diagram — Formal Description

For a single MoE Transformer layer, the computation flow with all five parallelism strategies active is:

$$
\boxed{
\begin{aligned}
&\textbf{Input: } \mathbf{X} \in \mathbb{R}^{B_\mu \times L \times d} \\[6pt]
&\text{1. } \underbrace{\text{DP}}_{\text{shard } B} \text{: each DP group gets } \mathbf{X}_{\text{dp}} \in \mathbb{R}^{(B_\mu/W_{DP}) \times L \times d} \\[4pt]
&\text{2. } \underbrace{\text{PP}}_{\text{shard layers}} \text{: layer } \ell \text{ executed only on stage } s(\ell) \\[4pt]
&\text{3. } \underbrace{\text{CP}}_{\text{shard } L} \text{: } \mathbf{X}_{\text{cp}} \in \mathbb{R}^{B_{\text{local}} \times (L/W_{CP}) \times d} \\[4pt]
&\text{4. } \underbrace{\text{TP+SP}}_{\text{shard } d} \text{: Self-Attn weights split along } d, \text{ activations along } L \text{ in SP regions} \\[4pt]
&\text{5. } \underbrace{\text{EP}}_{\text{shard experts}} \text{: MoE FFN experts distributed, All-to-All dispatch/combine}
\end{aligned}
}
$$

---

## 7. Memory Model Under 5D Parallelism

### 7.1 Parameter Memory

For a model with $\Phi_{\text{dense}}$ non-expert parameters and $\Phi_{\text{expert}}$ total expert parameters:

$$
M_{\text{params}} = \frac{\Phi_{\text{dense}}}{W_{TP} \cdot W_{PP}} + \frac{\Phi_{\text{expert}}}{W_{EP} \cdot W_{PP}}
$$

(In bytes, multiply by 2 for fp16 or 4 for fp32.)

### 7.2 Activation Memory

Per-layer activation memory for micro-batch size $B_\mu$, sequence length $L$, hidden dimension $d$, with selective recomputation:

$$
M_{\text{act}} \approx \frac{B_\mu}{W_{DP}} \cdot \frac{L}{W_{CP}} \cdot \frac{d}{W_{TP}} \cdot C_{\text{act}}
$$

where $C_{\text{act}}$ is a constant depending on the recomputation strategy:
- **No recomputation:** $C_{\text{act}} \approx 34$ (for standard Transformer with attention scores stored)
- **Selective recomputation:** $C_{\text{act}} \approx 10\text{–}16$
- **Full recomputation:** $C_{\text{act}} \approx 2$ (only store layer inputs)

For attention specifically under CP, the per-worker attention memory scales as:

$$
M_{\text{attn}} = \mathcal{O}\!\left(\frac{B_\mu \cdot n_h \cdot L^2}{W_{DP} \cdot W_{CP}^2}\right)
$$

where $n_h$ is the number of attention heads, and the $W_{CP}^2$ factor arises because both query and key sequence dimensions are sharded.

### 7.3 Optimizer State Memory (with ZeRO)

$$
M_{\text{optim}} = \frac{12 \cdot \Phi_{\text{local}}}{W_{DP}^{\text{ZeRO}}}
$$

where:
- $\Phi_{\text{local}}$ is the number of parameters on this GPU (after PP, TP, EP sharding)
- $W_{DP}^{\text{ZeRO}} = W_{DP}$ for ZeRO-1/2/3 (sharding among DP replicas)
- The factor 12 comes from Adam's fp32 copy (4 bytes) + first moment (4 bytes) + second moment (4 bytes)

### 7.4 Total Memory per GPU

$$
\boxed{
M_{\text{total}} = \underbrace{M_{\text{params}}}_{\text{PP, TP, EP}} + \underbrace{M_{\text{act}}}_{\text{DP, CP, TP}} + \underbrace{M_{\text{grad}}}_{\text{ZeRO-2/3}} + \underbrace{M_{\text{optim}}}_{\text{ZeRO-1/2/3}}
}
$$

---

## 8. Comprehensive Summary: All Strategies at a Glance

| **Method** | **Memory Savings Target** | **Parallel Dimension** | **Primary Disadvantage** |
|---|---|---|---|
| DP | Activations (smaller local batch) | Batch $B$ | Limited by maximum useful global batch size $B_{\max}$ |
| PP | Model parameters | Layer depth $\ell$ | Pipeline bubble: $\frac{S - 1}{n_\mu + S - 1}$ idle fraction |
| TP + SP | Parameters + Activations | Hidden $d$ / Sequence $L$ | Requires high-BW intra-node links; model-specific |
| CP | Activations | Sequence $L$ | Communication overhead in attention (Ring Attention) |
| EP | Expert parameters | Expert index $i$ | Requires MoE architecture; All-to-All routing overhead |
| ZeRO-1 | Optimizer states | Sharded across DP | AllGather overhead for parameters during step |
| ZeRO-2 | Optimizer states + Gradients | Sharded across DP | AllGather overhead for parameters during step |
| ZeRO-3 | Optimizer states + Gradients + Parameters | Sharded across DP | AllGather per layer per micro-batch (heaviest communication) |

### 8.1 Pipeline Bubble Fraction (Recap)

For 1F1B schedule with $S$ stages and $n_\mu$ micro-batches:

$$
\eta_{\text{bubble}} = \frac{S - 1}{n_\mu + S - 1}
$$

To keep bubble below fraction $\epsilon$:

$$
n_\mu \geq \frac{S - 1}{\epsilon} - (S - 1) = (S-1)\left(\frac{1}{\epsilon} - 1\right)
$$

### 8.2 Combinability Rules (Decision Heuristics)

1. **TP is always intra-node.** Assign $W_{TP} \leq$ GPUs per node (typically 4 or 8).

2. **PP spans inter-node** with low-bandwidth requirements (only activation tensors at stage boundaries).

3. **ZeRO-1/2 + PP** is straightforward and commonly used.

4. **ZeRO-3 + PP** is rare; if used, cache gathered weights across micro-batches.

5. **CP** is deployed when $L$ is very large; it is orthogonal to PP and TP.

6. **EP** is deployed when the model has MoE layers; it is orthogonal to TP (different sub-layers).

7. **DP** fills the remaining GPU dimensions after all other strategies are assigned:

$$
W_{DP} = \frac{W_{\text{total}}}{W_{TP} \times W_{PP} \times W_{CP} \times W_{EP}}
$$

The global batch size then becomes:

$$
B_{\text{global}} = W_{DP} \times B_\mu \times n_\mu
$$

which must remain within the range that preserves training convergence (typically validated empirically).

---

## 9. Real-World Configuration: DeepSeek-V3

DeepSeek-V3 (671B parameters, 256 experts, top-8 routing) uses the following parallelism configuration:

| Strategy | Value |
|---|---|
| $W_{TP}$ | 1 (no tensor parallelism — they use a novel communication-minimizing attention design) |
| $W_{PP}$ | 16 |
| $W_{EP}$ | 64 |
| $W_{DP}$ | Variable (with ZeRO-1) |
| Node-bounded routing | $M = 4$ nodes per token |
| Total experts $N_E$ | 256 routed + 1 shared |
| Active experts per token $K$ | 8 |

The total parameter count decomposes as:

$$
\Phi_{\text{total}} = \Phi_{\text{attention}} + \Phi_{\text{shared\_expert}} + N_E \cdot \Phi_{\text{per\_expert}} + \Phi_{\text{embed}}
$$

But active parameters per token (what determines FLOPs) are only:

$$
\Phi_{\text{active}} = \Phi_{\text{attention}} + \Phi_{\text{shared\_expert}} + K \cdot \Phi_{\text{per\_expert}} + \Phi_{\text{embed}} \approx 37\text{B}
$$

This exemplifies the core advantage of MoE + EP: **model capacity** scales with $N_E \cdot \Phi_{\text{per\_expert}}$ (memory distributed via EP), while **compute cost** scales only with $K \cdot \Phi_{\text{per\_expert}}$ (sparse activation).

---

This completes the technical treatment of Expert Parallelism and 5D Parallelism, covering the mathematical foundations, communication patterns, memory models, inter-strategy interactions, and real-world deployment configurations.