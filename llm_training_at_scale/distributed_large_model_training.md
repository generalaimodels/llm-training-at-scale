

# Finding the Best Training Configuration for Distributed Large Model Training

---

## 1. Problem Statement and Decision Framework

Given a model with $P$ total parameters, a target global batch size $B_{\text{global}}$, a cluster of $N_{\text{nodes}}$ nodes each containing $G$ GPUs (total GPU count $N_{\text{GPU}} = N_{\text{nodes}} \times G$), and per-GPU memory capacity $M_{\text{GPU}}$, the objective is to determine the optimal **5D parallelism configuration**:

$$\mathcal{C}^{*} = \arg\max_{\mathcal{C} \in \mathcal{S}} \; \text{MFU}(\mathcal{C}) \quad \text{subject to} \quad \text{Mem}(\mathcal{C}) \leq M_{\text{GPU}}, \quad B_{\text{eff}}(\mathcal{C}) = B_{\text{global}}$$

where the configuration vector is:

$$\mathcal{C} = \big(\underbrace{D_{\text{DP}}}_{\text{data parallelism}},\; \underbrace{D_{\text{TP}}}_{\text{tensor parallelism}},\; \underbrace{D_{\text{PP}}}_{\text{pipeline parallelism}},\; \underbrace{D_{\text{CP}}}_{\text{context parallelism}},\; \underbrace{D_{\text{EP}}}_{\text{expert parallelism}},\; \underbrace{N_{\text{GAS}}}_{\text{grad. accum. steps}},\; \underbrace{b_{\text{mbs}}}_{\text{micro-batch size}},\; \underbrace{Z}_{\text{ZeRO stage}}\big)$$

The search space $\mathcal{S}$ is constrained by the fundamental **parallelism decomposition constraint**:

$$\boxed{N_{\text{GPU}} = D_{\text{DP}} \times D_{\text{TP}} \times D_{\text{PP}} \times D_{\text{CP}} \times D_{\text{EP}}}$$

and the **global batch size equation**:

$$\boxed{B_{\text{global}} = D_{\text{DP}} \times D_{\text{CP}} \times N_{\text{GAS}} \times b_{\text{mbs}} \times s}$$

where $s$ is the sequence length in tokens.

**Model FLOPs Utilization (MFU)** is the primary throughput metric:

$$\text{MFU} = \frac{\text{Achieved FLOPs/s}}{\text{Peak Hardware FLOPs/s}} = \frac{F_{\text{model}} \cdot B_{\text{global}}}{T_{\text{step}} \cdot N_{\text{GPU}} \cdot \Phi_{\text{peak}}}$$

where $F_{\text{model}}$ is the forward-pass FLOPs per token (approximately $2P$ for dense transformers), $T_{\text{step}}$ is the wall-clock time per training step, and $\Phi_{\text{peak}}$ is the peak FLOPs/s per GPU (e.g., $\approx 989\;\text{TFLOP/s}$ for H100 SXM in BF16).

---

## 2. Step 1 — Fitting a Training Step in Memory

### 2.1 Memory Accounting

The total per-GPU memory requirement decomposes as:

$$\boxed{M_{\text{total}} = M_{\text{params}} + M_{\text{grads}} + M_{\text{opt}} + M_{\text{act}} + M_{\text{temp}} + M_{\text{frag}}}$$

| Component | Symbol | Formula (Mixed Precision, AdamW) |
|---|---|---|
| Parameters | $M_{\text{params}}$ | $\frac{2P}{D_{\text{TP}} \cdot D_{\text{PP}}}$ bytes (BF16 master copy on each shard) |
| Gradients | $M_{\text{grads}}$ | $\frac{2P}{D_{\text{TP}} \cdot D_{\text{PP}}}$ bytes |
| Optimizer states | $M_{\text{opt}}$ | $\frac{12P}{D_{\text{TP}} \cdot D_{\text{PP}}}$ bytes (FP32 copy + first & second moments) |
| Activations | $M_{\text{act}}$ | $\propto b_{\text{mbs}} \cdot s \cdot h \cdot L_{\text{local}}$ (depends on recomputation strategy) |
| Temporary buffers | $M_{\text{temp}}$ | Communication buffers, workspace allocations |
| Fragmentation overhead | $M_{\text{frag}}$ | CUDA memory allocator overhead |

Here $h$ is the hidden dimension and $L_{\text{local}} = L / D_{\text{PP}}$ is the number of transformer layers assigned to each pipeline stage, with $L$ total layers.

### 2.2 ZeRO Optimization Stages and Their Memory Impact

ZeRO (Zero Redundancy Optimizer) partitions optimizer state, gradients, and optionally parameters across $D_{\text{DP}}$ data-parallel ranks:

| ZeRO Stage | What is Sharded | Per-GPU Memory for Params+Grads+Optimizer |
|---|---|---|
| Stage 0 (baseline) | Nothing | $\frac{(2 + 2 + 12)P}{D_{\text{TP}} \cdot D_{\text{PP}}} = \frac{16P}{D_{\text{TP}} \cdot D_{\text{PP}}}$ |
| Stage 1 | Optimizer states | $\frac{(2 + 2)P}{D_{\text{TP}} \cdot D_{\text{PP}}} + \frac{12P}{D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}}}$ |
| Stage 2 | Optimizer states + Gradients | $\frac{2P}{D_{\text{TP}} \cdot D_{\text{PP}}} + \frac{(2 + 12)P}{D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}}}$ |
| Stage 3 | Optimizer states + Gradients + Parameters | $\frac{(2 + 2 + 12)P}{D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}}} = \frac{16P}{D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}}}$ |

The **memory feasibility constraint** is therefore:

$$\boxed{M_{\text{total}}\!\left(P,\; D_{\text{DP}},\; D_{\text{TP}},\; D_{\text{PP}},\; Z,\; b_{\text{mbs}},\; s,\; \text{recomp}\right) \;\leq\; M_{\text{GPU}}}$$

### 2.3 GPU-Rich Case — Decision Heuristics

The decision tree is determined by parameter count $P$ and available GPU count $N_{\text{GPU}}$:

#### Case A: $P < 10\text{B}$ (Small-to-Medium Models)

A single parallelism dimension typically suffices:

- **Option 1**: Tensor Parallelism with $D_{\text{TP}} \leq 8$ (intra-node).

  Per-GPU parameter memory:
  $$M_{\text{params}}^{\text{TP}} = \frac{2P}{D_{\text{TP}}} \;\text{bytes}$$

- **Option 2**: ZeRO-3 with Data Parallelism across 8 GPUs plus full activation recomputation.

  Per-GPU total model state memory:
  $$M_{\text{model}}^{\text{Z3}} = \frac{16P}{D_{\text{DP}}} \;\text{bytes}$$

For a 7B parameter model on 8 GPUs with ZeRO-3:
$$M_{\text{model}}^{\text{Z3}} = \frac{16 \times 7 \times 10^9}{8} = 14 \;\text{GB per GPU}$$

This easily fits within the 80 GB of an H100, leaving ample room for activations.

#### Case B: $10\text{B} \leq P \leq 100\text{B}$ (Large Models, Multi-Node)

Multiple parallelism dimensions must be composed. The number of GPUs required exceeds one node ($> 8$ GPUs), introducing inter-node communication as a critical consideration. Viable configurations include:

| Configuration | Parallelism Degrees | Communication Pattern |
|---|---|---|
| TP + PP | $D_{\text{TP}} = 8$, $D_{\text{PP}} = N_{\text{GPU}}/8$ | TP intra-node (NVLink), PP inter-node (InfiniBand) |
| TP + ZeRO-3/DP | $D_{\text{TP}} = 8$, $D_{\text{DP}} = N_{\text{GPU}}/8$ | TP intra-node, ZeRO all-gather inter-node |
| Pure ZeRO-3 | $D_{\text{DP}} = N_{\text{GPU}}$ | All-gather + reduce-scatter across all GPUs |

**Communication volume analysis** dictates the preferred choice. For ZeRO-3, the per-step communication volume per GPU scales as:

$$V_{\text{ZeRO-3}} = 2 \times \frac{2P \cdot (D_{\text{DP}} - 1)}{D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}}} \approx \frac{4P}{D_{\text{TP}} \cdot D_{\text{PP}}} \quad \text{for large } D_{\text{DP}}$$

The factor of 2 accounts for one all-gather (forward) and one reduce-scatter (backward) per training step.

#### Case C: $N_{\text{GPU}} \geq 512$ (Large-Scale Clusters)

At this scale, pure ZeRO-3 becomes communication-bound because the all-gather and reduce-scatter collectives span hundreds of GPUs across many nodes. The **communication time** for a ring all-gather over $D_{\text{DP}}$ ranks is:

$$T_{\text{all-gather}} = \frac{(D_{\text{DP}} - 1)}{D_{\text{DP}}} \cdot \frac{M_{\text{shard}}}{\beta_{\text{inter}}} + (D_{\text{DP}} - 1) \cdot \alpha$$

where $\beta_{\text{inter}}$ is the inter-node bandwidth, $\alpha$ is per-message latency, and $M_{\text{shard}} = 2P / (D_{\text{DP}} \cdot D_{\text{TP}} \cdot D_{\text{PP}})$.

As $D_{\text{DP}} \to \infty$, the latency term $(D_{\text{DP}} - 1)\alpha$ dominates, making pure DP inefficient. The solution is to **limit** $D_{\text{DP}}$ by introducing TP and PP:

$$D_{\text{DP}} = \frac{N_{\text{GPU}}}{D_{\text{TP}} \times D_{\text{PP}}} \quad \Rightarrow \quad \text{Reduce } D_{\text{DP}} \text{ by increasing } D_{\text{TP}} \text{ and/or } D_{\text{PP}}$$

#### Case D: $N_{\text{GPU}} \geq 1024$ (Frontier Scale)

The recommended configuration is a **3D parallelism** combination:

$$\boxed{\mathcal{C}_{\text{recommended}} = \left(D_{\text{TP}} = 8,\;\; D_{\text{PP}} = \frac{L}{L_{\text{local}}},\;\; D_{\text{DP}} = \frac{N_{\text{GPU}}}{D_{\text{TP}} \cdot D_{\text{PP}}},\;\; Z = 2\right)}$$

with ZeRO Stage 2 (sharding optimizer states and gradients but not parameters, thus avoiding the additional all-gather for parameters during forward pass).

#### Special Considerations

**Context Parallelism (CP)** for very long sequences ($s \gg 4096$): Activation memory scales linearly with sequence length. When $s$ is large, activations dominate:

$$M_{\text{act}} \propto b_{\text{mbs}} \cdot s \cdot h \cdot L_{\text{local}}$$

Context parallelism partitions the sequence dimension across $D_{\text{CP}}$ GPUs:

$$M_{\text{act}}^{\text{CP}} = \frac{M_{\text{act}}}{D_{\text{CP}}}$$

CP is placed **across nodes** since its communication (ring attention all-to-all for KV exchange) is less bandwidth-intensive than TP.

**Expert Parallelism (EP)** for Mixture-of-Experts (MoE): If the model has $E$ experts per MoE layer, each expert has parameters $P_{\text{expert}}$. Expert parallelism distributes experts across $D_{\text{EP}}$ GPUs:

$$\text{Experts per GPU} = \frac{E}{D_{\text{EP}}}$$

$$M_{\text{expert}}^{\text{per-GPU}} = \frac{E \cdot P_{\text{expert}}}{D_{\text{EP}}} \times \text{bytes per param}$$

EP is placed across nodes because the all-to-all communication pattern (dispatching tokens to experts) can tolerate higher latency compared to the tight synchronization required by TP.

### 2.4 GPU-Poor Case — Memory Reduction Techniques

When $M_{\text{total}} > M_{\text{GPU}}$ and additional GPUs are unavailable, two primary strategies reduce memory:

**Full Activation Recomputation**: Instead of storing all intermediate activations during the forward pass, discard them and recompute during the backward pass. This eliminates $M_{\text{act}}$ at the cost of one additional forward pass:

$$\text{Compute overhead} = \frac{F_{\text{fwd}} + F_{\text{fwd}} + F_{\text{bwd}}}{F_{\text{fwd}} + F_{\text{bwd}}} = \frac{2F_{\text{fwd}} + F_{\text{bwd}}}{F_{\text{fwd}} + F_{\text{bwd}}}$$

Since $F_{\text{bwd}} \approx 2 F_{\text{fwd}}$:

$$\text{Compute overhead} = \frac{2F_{\text{fwd}} + 2F_{\text{fwd}}}{F_{\text{fwd}} + 2F_{\text{fwd}}} = \frac{4}{3} \approx 1.33\times$$

This is approximately a **33% increase in compute time** for a near-complete elimination of activation memory.

**Gradient Accumulation**: Process the global batch as $N_{\text{GAS}}$ sequential micro-batches, accumulating gradients before the optimizer step:

$$B_{\text{global}} = D_{\text{DP}} \times N_{\text{GAS}} \times b_{\text{mbs}} \times s$$

Increasing $N_{\text{GAS}}$ allows using a smaller $b_{\text{mbs}}$, reducing peak activation memory:

$$M_{\text{act}} \propto b_{\text{mbs}} \cdot s \cdot h \cdot L_{\text{local}} \quad \Rightarrow \quad \text{Reduce } b_{\text{mbs}} \text{ to reduce } M_{\text{act}}$$

---

## 3. Step 2 — Achieving the Target Global Batch Size

After Step 1 establishes a memory-feasible configuration with some initial values of $D_{\text{DP}}$, $N_{\text{GAS}}$, and $b_{\text{mbs}}$, the current effective global batch size is:

$$B_{\text{current}} = D_{\text{DP}} \times D_{\text{CP}} \times N_{\text{GAS}} \times b_{\text{mbs}} \times s$$

The target is $B_{\text{global}} = B_{\text{target}}$ (e.g., $1\text{M tokens}$).

### 3.1 Increasing $B_{\text{current}}$ to $B_{\text{target}}$

If $B_{\text{current}} < B_{\text{target}}$, increase the batch size via:

| Mechanism | Action | Trade-off |
|---|---|---|
| Scale $D_{\text{DP}}$ | Add more GPUs to data parallelism | More hardware, more communication |
| Scale $N_{\text{GAS}}$ | Increase gradient accumulation steps | Longer step time (sequential micro-batches), no memory increase |
| Scale $D_{\text{CP}}$ | Increase context parallelism (long sequences) | Sequence-dimension splitting, KV exchange overhead |

The **scaling factor** required is:

$$k = \frac{B_{\text{target}}}{B_{\text{current}}} = \frac{D_{\text{DP}}^{\text{new}} \times N_{\text{GAS}}^{\text{new}}}{D_{\text{DP}}^{\text{old}} \times N_{\text{GAS}}^{\text{old}}}$$

### 3.2 Decreasing $B_{\text{current}}$ to $B_{\text{target}}$

If $B_{\text{current}} > B_{\text{target}}$ (too many data-parallel replicas), reallocate GPUs from DP to other parallelism dimensions:

$$D_{\text{DP}}^{\text{new}} = \frac{D_{\text{DP}}^{\text{old}}}{r}, \quad D_{\text{TP}}^{\text{new}} = D_{\text{TP}}^{\text{old}} \times r \;\;\text{or}\;\; D_{\text{PP}}^{\text{new}} = D_{\text{PP}}^{\text{old}} \times r$$

where $r$ is the reallocation factor, preserving $N_{\text{GPU}} = D_{\text{DP}} \times D_{\text{TP}} \times D_{\text{PP}} \times D_{\text{CP}} \times D_{\text{EP}}$.

---

## 4. Step 3 — Optimizing Training Throughput

With memory feasibility and correct batch size established, the final objective is **maximizing MFU**. The per-step training time decomposes as:

$$\boxed{T_{\text{step}} = T_{\text{compute}} + T_{\text{comm}} - T_{\text{overlap}} + T_{\text{idle}}}$$

where:
- $T_{\text{compute}}$: time for forward + backward + optimizer computation
- $T_{\text{comm}}$: total communication time across all parallelism dimensions
- $T_{\text{overlap}}$: time where communication is hidden behind computation
- $T_{\text{idle}}$: idle time due to pipeline bubbles, synchronization barriers, load imbalance

### 4.1 Communication Time Analysis per Parallelism Dimension

**Tensor Parallelism** communication per layer (2 all-reduce operations in forward, 2 in backward):

$$T_{\text{comm}}^{\text{TP}} = 4L_{\text{local}} \times \frac{2(D_{\text{TP}} - 1)}{D_{\text{TP}}} \times \frac{M_{\text{tensor}}}{\beta_{\text{intra}}}$$

where $M_{\text{tensor}} \propto b_{\text{mbs}} \times s \times h$ is the activation tensor size, and $\beta_{\text{intra}}$ is the intra-node bandwidth (NVLink: $\sim 900\;\text{GB/s}$ bidirectional on H100).

**Data Parallelism (ZeRO-2)** communication per step (one all-reduce of gradients, decomposed as reduce-scatter + all-gather of optimizer states):

$$T_{\text{comm}}^{\text{DP}} = 2 \times \frac{2(D_{\text{DP}} - 1)}{D_{\text{DP}}} \times \frac{M_{\text{grad}}}{\beta_{\text{eff}}}$$

where $M_{\text{grad}} = 2P / (D_{\text{TP}} \cdot D_{\text{PP}})$ and $\beta_{\text{eff}}$ is the effective cross-node bandwidth.

**Pipeline Parallelism** introduces bubble overhead rather than bandwidth-limited communication. For a 1F1B schedule with $D_{\text{PP}}$ stages and $m$ micro-batches:

$$T_{\text{idle}}^{\text{PP}} = \frac{(D_{\text{PP}} - 1)}{m} \times T_{\text{step}}^{\text{ideal}}$$

The **pipeline bubble fraction** is:

$$\boxed{f_{\text{bubble}} = \frac{D_{\text{PP}} - 1}{m + D_{\text{PP}} - 1} \approx \frac{D_{\text{PP}} - 1}{m} \quad \text{when } m \gg D_{\text{PP}}}$$

where $m = D_{\text{DP}} \times N_{\text{GAS}}$ is the total number of micro-batches in the pipeline.

### 4.2 Throughput Optimization Heuristics (Ordered by Priority)

**Priority 1 — Maximize TP within a node:**

$$D_{\text{TP}} \to G \quad (\text{e.g., } D_{\text{TP}} = 8 \text{ on 8-GPU nodes})$$

Since TP communication occurs over NVLink ($\beta_{\text{intra}} \gg \beta_{\text{inter}}$), this minimizes communication latency. However, TP introduces synchronization points per layer, so there exists a diminishing-returns threshold where computation per GPU becomes too small relative to communication.

The **compute-to-communication ratio** for TP must satisfy:

$$\rho_{\text{TP}} = \frac{T_{\text{compute}}^{\text{per-layer}} / D_{\text{TP}}}{T_{\text{comm}}^{\text{TP-per-layer}}} = \frac{F_{\text{layer}} / (D_{\text{TP}} \cdot \Phi_{\text{peak}})}{4 \cdot \frac{2(D_{\text{TP}}-1)}{D_{\text{TP}}} \cdot \frac{b_{\text{mbs}} \cdot s \cdot h}{\beta_{\text{intra}}}} \gg 1$$

When $\rho_{\text{TP}}$ drops below $\sim 1$, TP becomes communication-bound.

**Priority 2 — Scale DP with ZeRO-3 while maintaining $B_{\text{target}}$:**

If $D_{\text{DP}} \times N_{\text{GAS}} \times b_{\text{mbs}} \times s = B_{\text{target}}$ can be maintained, increasing $D_{\text{DP}}$ distributes computation while keeping the batch size constant (by reducing $N_{\text{GAS}}$). This is beneficial because:

$$N_{\text{GAS}}^{\text{new}} = \frac{B_{\text{target}}}{D_{\text{DP}}^{\text{new}} \times b_{\text{mbs}} \times s} < N_{\text{GAS}}^{\text{old}}$$

Fewer gradient accumulation steps means fewer sequential micro-batch forward-backward passes, reducing $T_{\text{step}}$.

**Priority 3 — Transition to PP when DP communication saturates:**

When DP communication time $T_{\text{comm}}^{\text{DP}}$ can no longer be overlapped with backward computation (i.e., $T_{\text{comm}}^{\text{DP}} > T_{\text{compute}}^{\text{bwd}}$), introduce pipeline parallelism to reduce $D_{\text{DP}}$:

$$D_{\text{DP}}^{\text{new}} = \frac{D_{\text{DP}}^{\text{old}}}{D_{\text{PP}}^{\text{new}}}$$

This trades DP communication overhead for pipeline bubble overhead. The transition is beneficial when:

$$\frac{D_{\text{PP}} - 1}{m} \cdot T_{\text{compute}} < T_{\text{comm}}^{\text{DP-excess}}$$

**Priority 4 — Tune micro-batch size $b_{\text{mbs}}$:**

The micro-batch size affects multiple performance dimensions simultaneously:

$$b_{\text{mbs}} \uparrow \;\Rightarrow\; \begin{cases} M_{\text{act}} \uparrow & \text{(higher memory)} \\ \text{GPU utilization} \uparrow & \text{(larger matrix multiplications, better SM occupancy)} \\ m = \frac{B_{\text{target}}}{D_{\text{DP}} \cdot b_{\text{mbs}} \cdot s} \downarrow & \text{(fewer micro-batches, larger pipeline bubble)} \\ T_{\text{comm}}^{\text{TP}} \uparrow & \text{(larger activation tensors to communicate)} \end{cases}$$

The optimal $b_{\text{mbs}}^{*}$ balances these competing effects and must be found empirically.

---

## 5. Benchmarking Thousands of Configurations

### 5.1 Search Space Enumeration

For a given model size $P$ and cluster size $N_{\text{GPU}}$, the total number of valid configurations is:

$$|\mathcal{S}| = \sum_{\substack{D_{\text{TP}}, D_{\text{PP}}, D_{\text{DP}} \\ D_{\text{TP}} \cdot D_{\text{PP}} \cdot D_{\text{DP}} = N_{\text{GPU}}}} \;\; \sum_{Z \in \{0,1,2,3\}} \;\; \sum_{b_{\text{mbs}} \in \mathcal{B}} \;\; \sum_{N_{\text{GAS}} \in \mathcal{G}} \mathbb{1}\!\left[\text{feasible}(\mathcal{C})\right]$$

where $\mathcal{B}$ is the set of candidate micro-batch sizes, $\mathcal{G}$ is the set of valid gradient accumulation steps, and the indicator function $\mathbb{1}[\cdot]$ filters configurations that satisfy memory constraints and batch size targets.

Even after pruning infeasible configurations, $|\mathcal{S}|$ remains in the **thousands** across all model sizes and cluster sizes.

### 5.2 Benchmark Setup and Experimental Conditions

The benchmarks referenced in the content were conducted with:

| Parameter | Value |
|---|---|
| Sequence length $s$ | $4096$ tokens |
| Global batch size $B_{\text{global}}$ | $1\text{M tokens}$ ($= 1{,}048{,}576$) |
| GPUs per node $G$ | $8 \times \text{H100 SXM}$ |
| Nodes $N_{\text{nodes}}$ | $1$ to $64$ |
| Total GPUs $N_{\text{GPU}}$ | $8$ to $512$ |
| Interconnect (intra-node) | NVLink ($900\;\text{GB/s}$ bidirectional) |
| Interconnect (inter-node) | InfiniBand ($400\;\text{Gb/s}$) |
| Precision | BF16 mixed precision |

For each $(P, N_{\text{GPU}})$ pair, every valid configuration $\mathcal{C} \in \mathcal{S}$ was benchmarked, and the **MFU** was recorded.

### 5.3 Heatmap Analysis and Key Insights

The heatmap visualization plots the optimal configuration $\mathcal{C}^{*}$ and its corresponding MFU for each combination of model size $P$ and node count $N_{\text{nodes}}$.

#### Insight 1: Efficiency Decreases with Increasing Node Count (Especially for Small Models)

For a fixed model size $P$, increasing $N_{\text{nodes}}$ (and hence $N_{\text{GPU}}$) reduces MFU. The root cause is the **arithmetic intensity** drop:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs per GPU}}{\text{Communication bytes per GPU}} = \frac{F_{\text{model}} \cdot B_{\text{global}} / N_{\text{GPU}}}{V_{\text{comm}}}$$

When $P$ is small, $F_{\text{model}} = 2P$ is small, so the numerator shrinks as $N_{\text{GPU}}$ increases while $V_{\text{comm}}$ does not decrease proportionally. For small models, even increasing $B_{\text{global}}$ to compensate is impossible when constrained to $1\text{M tokens}$.

The **scaling efficiency** can be quantified as:

$$\eta_{\text{scale}}(N_{\text{GPU}}) = \frac{\text{MFU}(N_{\text{GPU}})}{\text{MFU}(G)} = \frac{T_{\text{step}}(G) \cdot G}{T_{\text{step}}(N_{\text{GPU}}) \cdot N_{\text{GPU}}}$$

For small models, $\eta_{\text{scale}}$ decays sharply because the compute-to-communication ratio falls below the threshold for efficient overlap.

#### Insight 2: Large Models Face Memory Walls on Small Clusters

For large $P$ (e.g., $80\text{B}$) on few nodes (e.g., 4 nodes = 32 GPUs):

$$M_{\text{model}}^{\text{min}} = \frac{16P}{N_{\text{GPU}}} = \frac{16 \times 80 \times 10^9}{32} = 40\;\text{GB per GPU}$$

This is the absolute minimum (ZeRO-3 with no activations). With activations, the memory exceeds 80 GB, forcing aggressive recomputation and small $b_{\text{mbs}}$, which in turn leads to:

- Poor GPU utilization (small matrix multiplications)
- Large pipeline bubble fractions (many stages, few micro-batches)
- Net result: low MFU despite the model fitting

#### Insight 3: Implementation Quality Dominates Configuration Choice

The relative performance ranking between TP and PP is **not fixed** — it depends critically on implementation quality:

$$\text{MFU}_{\text{TP}} \gtrless \text{MFU}_{\text{PP}} \quad \text{depends on overlap quality, kernel efficiency, scheduling}$$

Specifically:
- **TP with naive synchronous all-reduce**: Exposes full communication latency, making TP slower than PP.
- **TP with asynchronous communication-computation overlap**: Hides communication behind computation via CUDA streams, potentially making TP faster.
- **PP with optimized interleaved scheduling** (e.g., interleaved 1F1B, zero-bubble schedules): Reduces $f_{\text{bubble}}$ from $\frac{D_{\text{PP}}-1}{m}$ toward zero.

The lesson is that $\text{MFU}(\mathcal{C})$ is not a pure function of the parallelism configuration — it is also a function of the **implementation**:

$$\text{MFU} = f(\mathcal{C}, \;\mathcal{I}_{\text{implementation}}, \;\mathcal{H}_{\text{hardware}})$$

---

## 6. Practical Engineering Challenges in Large-Scale Benchmarking

### 6.1 Infrastructure Failure Modes

Running thousands of distributed configurations exposes failure modes invisible at small scale:

| Failure Mode | Root Cause | Impact |
|---|---|---|
| **Zombie processes** | PyTorch distributed processes fail to clean up NCCL communicators | GPU memory leaks, subsequent jobs fail |
| **Slurm forced termination** | Job manager kills jobs exceeding time or memory limits | Partial results, node marked unhealthy |
| **Hanging jobs** | NCCL deadlocks from mismatched collective operations or network partitions | Infinite wait, GPU idle time |
| **CUDA OOM at runtime** | Memory fragmentation from CUDA caching allocator | Crash despite theoretical memory feasibility |

### 6.2 CUDA Memory Allocator Behavior

The CUDA caching allocator (used by PyTorch) maintains pools of allocated memory blocks. **Fragmentation** occurs when freed blocks cannot be coalesced:

$$M_{\text{allocated}} > M_{\text{used}} + M_{\text{fragmented}}$$

This means the theoretical memory computation $M_{\text{total}}$ underestimates actual GPU memory consumption. In practice:

$$M_{\text{actual}} \approx M_{\text{total}} \times (1 + \epsilon_{\text{frag}})$$

where $\epsilon_{\text{frag}} \in [0.05, 0.20]$ depending on allocation patterns.

### 6.3 NCCL Communication Overhead

NCCL (NVIDIA Collective Communications Library) implements collectives using GPU **Streaming Multiprocessors (SMs)**. This creates a critical hidden cost:

$$\Phi_{\text{available}}^{\text{compute}} = \Phi_{\text{total}} - \Phi_{\text{NCCL}} < \Phi_{\text{peak}}$$

When communication is overlapped with computation, NCCL kernels consume SMs that would otherwise perform matrix multiplications, resulting in:

$$\text{True MFU} = \frac{F_{\text{model}} \cdot B_{\text{global}}}{T_{\text{step}} \cdot N_{\text{GPU}} \cdot \Phi_{\text{peak}}} < \text{Expected MFU (assuming perfect overlap)}$$

This SM contention is the fundamental reason why the assumption *"computation and communication can be efficiently overlapped without throughput impact"* is violated in practice. The actual throughput during overlap is:

$$\Phi_{\text{compute}}^{\text{during\_overlap}} = \Phi_{\text{peak}} \cdot \left(1 - \frac{N_{\text{SM}}^{\text{NCCL}}}{N_{\text{SM}}^{\text{total}}}\right)$$

where $N_{\text{SM}}^{\text{NCCL}}$ is the number of SMs used by NCCL and $N_{\text{SM}}^{\text{total}}$ is the total SM count (e.g., 132 on H100).

---

## 7. Summary — Complete Decision Algorithm

The complete configuration search algorithm can be formalized as:

$$\boxed{
\begin{aligned}
&\textbf{Step 1: Memory Feasibility} \\
&\quad \text{Find } (D_{\text{TP}}, D_{\text{PP}}, D_{\text{DP}}, Z, \text{recomp}) \;\text{s.t.}\; M_{\text{total}} \leq M_{\text{GPU}} \\[6pt]
&\textbf{Step 2: Batch Size Matching} \\
&\quad \text{Adjust } (D_{\text{DP}}, N_{\text{GAS}}, b_{\text{mbs}}, D_{\text{CP}}) \;\text{s.t.}\; B_{\text{eff}} = B_{\text{target}} \\[6pt]
&\textbf{Step 3: Throughput Maximization} \\
&\quad \mathcal{C}^{*} = \arg\max_{\mathcal{C} \in \mathcal{S}_{\text{feasible}}} \text{MFU}(\mathcal{C}) \\
&\quad \text{via: maximize TP intra-node} \to \text{scale DP} \to \text{add PP when DP saturates} \to \text{tune } b_{\text{mbs}}
\end{aligned}
}$$

The entire process is **iterative and empirical**: theoretical analysis narrows the search space $\mathcal{S}$, but the final optimal $\mathcal{C}^{*}$ must be discovered through systematic benchmarking on the target hardware, as implementation quality, network topology, and GPU-level resource contention introduce performance variations that analytical models cannot fully capture.