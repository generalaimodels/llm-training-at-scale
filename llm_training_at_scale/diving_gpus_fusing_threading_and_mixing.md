
# Diving into the GPUs — Fusing, Threading, and Mixing

---

## 1. GPU Architecture Primer

### 1.1 Compute Hierarchy

An NVIDIA GPU is organized as a **hierarchical array of compute units**. The fundamental building block is the **Streaming Multiprocessor (SM)**, each of which contains multiple **streaming processors (cores)**.

For the NVIDIA H100 SXM:

$$N_{\text{SM}} = 132, \quad N_{\text{cores/SM}} = 128, \quad N_{\text{cores}}^{\text{total}} = N_{\text{SM}} \times N_{\text{cores/SM}} = 132 \times 128 = 16{,}896$$

Each core can execute multiple **threads** concurrently. The compute hierarchy from bottom to top is:

$$\text{Thread} \;\subset\; \text{Warp (32 threads)} \;\subset\; \text{Block} \;\subset\; \text{Grid (all blocks)} \;\to\; \text{Mapped onto SMs}$$

| Level | Description | Mapping |
|---|---|---|
| **Thread** | Smallest unit of execution; executes one instance of the kernel | Runs on a single core |
| **Warp** | Group of exactly 32 threads executing in **lockstep** (SIMD) | Scheduled as an atomic unit on an SM |
| **Block** | Programmer-defined grouping of threads (e.g., 256 or 1024 threads) | Assigned to exactly one SM; an SM may host multiple blocks |
| **Grid** | Collection of all blocks for a kernel launch | Distributed across all available SMs |

### 1.2 Memory Hierarchy

The memory system is equally hierarchical, with a fundamental trade-off: **smaller memories are faster but private; larger memories are slower but shared**.

| Memory Level | Scope | Capacity (H100) | Latency | Bandwidth |
|---|---|---|---|---|
| **Registers** | Private to each thread | 256 KB per SM | ~1 cycle | Highest |
| **Shared Memory / L1 Cache** | Shared across threads in a block (one SM) | 228 KB per SM (configurable) | ~20–30 cycles | ~19 TB/s aggregate |
| **L2 Cache** | Shared across all SMs | 50 MB | ~200 cycles | ~12 TB/s |
| **Global Memory (HBM3)** | Shared across entire GPU | 80 GB | ~400–600 cycles | 3.35 TB/s |

The **performance optimization objective** is:

$$\boxed{\text{Maximize data reuse in fast memories (registers, shared memory) to minimize accesses to slow global memory (HBM)}}$$

### 1.3 Kernel Execution Model

A **kernel** is a function that runs on the GPU. It is written in CUDA (C/C++ extension) or Triton (Python-based) and compiled to **PTX (Parallel Thread Execution)** — NVIDIA's low-level virtual ISA.

The execution requires two components:

**Host Code (CPU side):** Allocates device memory, transfers data, launches kernels.

```c
// Allocate on GPU
cudaMalloc(&d_A, size);
// Copy host → device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// Launch kernel with grid/block dimensions
VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
// Copy device → host
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

**Device Code (GPU side):** The kernel function itself, executed by each thread.

```c
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

The **global thread index** for a 1D grid is computed as:

$$i = \texttt{blockDim.x} \times \texttt{blockIdx.x} + \texttt{threadIdx.x}$$

The **grid dimensions** are chosen to cover all $N$ elements:

$$\texttt{blocksPerGrid} = \left\lceil \frac{N}{\texttt{threadsPerBlock}} \right\rceil$$

### 1.4 Scheduling Constraints

Key hardware constraints govern efficient scheduling:

| Constraint | H100 Value | Impact |
|---|---|---|
| Threads per warp | 32 (fixed) | All 32 threads execute same instruction |
| Max threads per block | 1024 | Upper limit on block size |
| Max blocks per SM | 32 | Limits occupancy if blocks are small |
| Max warps per SM | 64 | $= 64 \times 32 = 2048$ threads max per SM |
| Registers per SM | 65,536 (32-bit) | Shared among all active threads on SM |
| Shared memory per SM | 228 KB | Partitioned among active blocks |

**Occupancy** — the ratio of active warps to the maximum possible — determines how well the SM hides memory latency:

$$\text{Occupancy} = \frac{N_{\text{active warps per SM}}}{N_{\text{max warps per SM}}} = \frac{N_{\text{active warps}}}{64}$$

Higher occupancy provides more warps to switch between during memory stalls, hiding latency through **latency hiding via warp scheduling**.

---

## 2. Improving Performance with Kernels

### 2.1 Toolchain Spectrum

There exists a spectrum of tools for writing GPU kernels, ordered by ease-of-use vs. control:

| Tool | Ease | Performance | Flexibility | Use Case |
|---|---|---|---|---|
| **PyTorch (eager mode)** | Easiest | Slowest | Full Python | Prototyping |
| **`@torch.compile`** | Easy (decorator) | Fast | Limited | Production without custom kernels |
| **Triton** | Moderate | Faster | Block-level control | Custom fused kernels |
| **CUDA** | Hardest | Fastest (if done right) | Full SM/warp/thread control | Maximum performance |

### 2.2 Example: Exponential Linear Unit (ELU)

The **ELU activation function** is defined as:

$$\boxed{\text{ELU}(x) = \begin{cases} \alpha \left(e^x - 1\right) & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases}}$$

where $\alpha$ is typically set to $1.0$.

**PyTorch with `@torch.compile`:**

```python
@torch.compile
def elu(x, alpha=1.0):
    return torch.where(x < 0, alpha * (torch.exp(x) - 1), x)
```

The `@torch.compile` decorator invokes **TorchInductor**, which:

1. **Traces** the PyTorch operations into an intermediate representation (FX graph)
2. **Lowers** the graph to Triton kernels
3. **Compiles** and caches the optimized kernels

The generated Triton kernel can be inspected by setting `TORCH_LOGS="output_code"`:

```python
@triton.jit
def elu_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    input_values = tl.load(input_ptr + block_indices, valid_mask)
    zero_value = 0.0
    negative_mask = input_values < zero_value
    exp_values = tl.math.exp(input_values)
    one_value = 1.0
    shifted_exp_values = exp_values - one_value
    output_values = tl.where(negative_mask, shifted_exp_values, input_values)
    tl.store(output_ptr + block_indices, output_values, valid_mask)
```

**Key Triton concepts:**

- `tl.program_id(0)`: Returns the unique **block ID** in dimension 0 (analogous to `blockIdx.x` in CUDA)
- `BLOCK_SIZE`: Compile-time constant defining how many elements each block processes
- `tl.arange(0, BLOCK_SIZE)`: Creates a vector of consecutive indices within a block
- `valid_mask`: Bounds-checks to prevent out-of-range memory access
- `tl.load` / `tl.store`: Masked memory operations

**Triton vs. CUDA control granularity:**

| Aspect | Triton | CUDA |
|---|---|---|
| Programming unit | Block (program) | Thread |
| Shared memory management | Automatic | Manual (`__shared__`) |
| Warp-level primitives | Not directly exposed | Full access (`__shfl_sync`, etc.) |
| Scheduling within SM | Automatic | Manual (warp/thread indexing) |
| Memory coalescing | Automatic (vectorized loads) | Manual (access pattern design) |

---

## 3. CUDA Optimization Techniques

### 3.1 Memory Coalescing

**Definition:** Memory coalescing is the hardware mechanism by which a GPU combines multiple memory access requests from threads within a **single warp** into a minimal number of memory transactions, exploiting the **burst transfer** behavior of DRAM.

**DRAM burst mechanism:** When any single address $M$ in global memory (HBM/DRAM) is accessed, the DRAM chip reads a contiguous segment of $B_{\text{burst}}$ bytes (typically 32 or 128 bytes) around $M$ in a single operation. Coalescing ensures threads access addresses within the same burst segment.

**Coalescing condition:** For a warp of 32 threads accessing addresses $a_0, a_1, \ldots, a_{31}$, memory is **perfectly coalesced** when:

$$a_i = a_0 + i \cdot \text{sizeof(element)} \quad \forall \; i \in \{0, 1, \ldots, 31\}$$

That is, consecutive threads access consecutive memory locations.

#### Naive Matrix Multiplication (Uncoalesced)

Consider computing $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{M \times K}$, $\mathbf{B} \in \mathbb{R}^{K \times N}$, $\mathbf{C} \in \mathbb{R}^{M \times N}$.

**Row-major storage convention:** An element at row $r$, column $c$ of a matrix with $N_{\text{cols}}$ columns is stored at linear address:

$$\text{addr}(r, c) = \text{base} + (r \times N_{\text{cols}} + c) \times \text{sizeof(element)}$$

**Naive 2D kernel:**

```c
__global__ void matmul_naive(int M, int N, int K, 
                              const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // column
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}
```

**Problem analysis:** Two threads in the same warp with consecutive `threadIdx.x` values (e.g., thread $(0,0)$ and thread $(1,0)$) have the same $y$ but different $x$. At iteration $i = 0$:

- Thread $(0, 0)$ reads $A[0 \cdot K + 0] = A[0]$
- Thread $(1, 0)$ reads $A[1 \cdot K + 0] = A[K]$

These addresses are **$K$ elements apart** in memory — not consecutive. The accesses to $\mathbf{A}$ are **uncoalesced**, resulting in $\sim 32$ separate memory transactions instead of 1.

**Fix — 1D block with recomputed indices:**

```c
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // row
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // column
```

Now consecutive `threadIdx.x` values produce the **same $x$ (row)** but **different $y$ (column)**. At iteration $i = 0$:

- Thread 0 reads $A[x \cdot K + 0]$ and $B[0 \cdot N + y_0]$
- Thread 1 reads $A[x \cdot K + 0]$ (same!) and $B[0 \cdot N + y_0 + 1]$

The $\mathbf{B}$ accesses are now consecutive in memory (coalesced), and the $\mathbf{A}$ accesses are identical (broadcast). The result is a **$\sim 10\times$ improvement** in both memory throughput and execution time.

### 3.2 Tiling (Shared Memory Optimization)

**Motivation:** Even with coalesced access, global memory bandwidth is limited. For matrix multiplication, each element of $\mathbf{A}$ and $\mathbf{B}$ is loaded **multiple times** by different threads. Without optimization, the total number of global memory loads is:

$$\text{Global loads (naive)} = M \times N \times K \times 2$$

(Each of the $M \times N$ output elements requires $K$ loads from $\mathbf{A}$ and $K$ loads from $\mathbf{B}$.)

**Tiling principle:** Partition the computation into **tiles** of size $T_M \times T_K$ (from $\mathbf{A}$) and $T_K \times T_N$ (from $\mathbf{B}$). A block of threads **cooperatively loads** one tile of each matrix into **shared memory** (SRAM), then all threads in the block compute using the fast shared memory.

The output tile $\mathbf{C}_{\text{tile}} \in \mathbb{R}^{T_M \times T_N}$ is accumulated over $\lceil K / T_K \rceil$ iterations:

$$\mathbf{C}_{ij} = \sum_{t=0}^{\lceil K/T_K \rceil - 1} \sum_{k=0}^{T_K - 1} \mathbf{A}_{i,\, t \cdot T_K + k} \cdot \mathbf{B}_{t \cdot T_K + k,\, j}$$

For a square tile of size $T$:

```c
for (int tileIdx = 0; tileIdx < K; tileIdx += TILE_SIZE) {
    // Cooperative load: each thread loads one element of A and B
    sharedA[localRow * TILE_SIZE + localCol] = A[localRow * K + localCol];
    sharedB[localRow * TILE_SIZE + localCol] = B[localRow * N + localCol];
    __syncthreads();  // Barrier: ensure all loads complete
    
    // Compute partial dot product from shared memory
    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += sharedA[localRow * TILE_SIZE + i] 
             * sharedB[i * TILE_SIZE + localCol];
    }
    __syncthreads();  // Barrier before next tile load
    
    A += TILE_SIZE;       // Advance tile in A (across columns)
    B += TILE_SIZE * N;   // Advance tile in B (down rows)
}
```

**Memory access reduction:** With tiling, each element of $\mathbf{A}$ and $\mathbf{B}$ is loaded from global memory only once per tile, then reused $T$ times from shared memory:

$$\text{Global loads (tiled)} = \frac{M \times N \times K \times 2}{T}$$

The **reuse factor** is $T$, giving a proportional reduction in global memory traffic.

**`__syncthreads()`** is a **block-level barrier**: all threads in the block must reach this point before any can proceed. Two barriers are required per tile iteration:

1. After loading data into shared memory (ensure all data is available before computation)
2. After computation (ensure all threads finish before overwriting shared memory with the next tile)

**Shared memory requirement:**

$$M_{\text{shared}} = (T_M \times T_K + T_K \times T_N) \times \text{sizeof(float)}$$

For $T_M = T_K = T_N = T = 32$ with FP32:

$$M_{\text{shared}} = (32 \times 32 + 32 \times 32) \times 4 = 8{,}192 \;\text{bytes} = 8 \;\text{KB}$$

This easily fits within the 228 KB per SM on H100.

### 3.3 Thread Coarsening

**Problem identified via profiling:** After tiling, warp stall analysis reveals that warps spend significant cycles in the `stalled_mio_throttle` state — stalled waiting for the **Memory Input/Output (MIO) pipeline** to service shared memory requests.

The root cause is that each thread computes a single output element, requiring many shared memory loads per thread. The shared memory access pipeline becomes a bottleneck.

**Thread coarsening** merges $C_f$ threads into a single **coarsened thread**, where each coarsened thread computes $C_f$ output elements instead of 1.

**Before coarsening:** Each thread computes 1 element, requiring $2T_K$ shared memory loads per tile iteration:

$$\text{Shared mem loads per thread per tile} = 2 \times T_K$$

(One row of $\mathbf{A}_{\text{shared}}$, one column of $\mathbf{B}_{\text{shared}}$.)

**After coarsening by factor $C_f$:** Each thread computes $C_f$ elements **in the same row**, sharing the same row of $\mathbf{A}_{\text{shared}}$:

$$\text{Shared mem loads per thread per tile} = T_K + C_f \times T_K = (1 + C_f) \times T_K$$

Without coarsening, $C_f$ separate threads would have loaded:

$$\text{Total without coarsening} = C_f \times 2 \times T_K = 2 C_f \times T_K$$

The **savings ratio** is:

$$\text{Reduction factor} = \frac{2 C_f \times T_K}{(1 + C_f) \times T_K} = \frac{2C_f}{1 + C_f}$$

For $C_f = 8$: reduction factor $= 16/9 \approx 1.78\times$ fewer shared memory accesses.

### 3.4 Minimizing Control Divergence

GPUs execute in the **SIMD (Single Instruction, Multiple Data)** model at the warp level: all 32 threads in a warp execute the **same instruction** simultaneously on different data.

**Control divergence** occurs when threads within a warp encounter a conditional branch and take **different execution paths**:

```c
if (condition) {
    // Path A — executed by some threads
} else {
    // Path B — executed by remaining threads
}
```

When divergence occurs, the hardware **serializes** execution:

1. Threads taking Path A execute while Path B threads are **masked (idle)**
2. Then Path B threads execute while Path A threads are masked

The effective warp throughput drops proportionally:

$$\text{Throughput}_{\text{divergent}} = \frac{\text{Throughput}_{\text{peak}}}{N_{\text{paths}}}$$

For a simple if-else, $N_{\text{paths}} = 2$, yielding **50% throughput**.

**Mitigation strategies:**

| Strategy | Description |
|---|---|
| **Predication** | Compiler replaces branches with conditional assignments; both paths execute but results are selectively written |
| **Data reorganization** | Arrange data so threads in the same warp follow the same path (e.g., sort by condition) |
| **Warp-aligned branching** | Ensure the branch condition is uniform across all 32 threads in a warp |

---

## 4. Fused Kernels

### 4.1 The Kernel Launch Overhead Problem

In standard (unfused) execution, each operation is a separate kernel launch. For a sequence of $n$ point-wise operations $f_1, f_2, \ldots, f_n$ on a tensor $\mathbf{x}$:

$$\mathbf{x}_1 = f_1(\mathbf{x}), \quad \mathbf{x}_2 = f_2(\mathbf{x}_1), \quad \ldots, \quad \mathbf{x}_n = f_n(\mathbf{x}_{n-1})$$

Each kernel launch requires:

1. **Write** $\mathbf{x}_i$ from SM registers/shared memory → global memory (HBM)
2. **Kernel launch overhead** (CPU→GPU scheduling, ~5–10 μs)
3. **Read** $\mathbf{x}_i$ from global memory → SM for the next kernel

The total HBM traffic for $n$ unfused kernels:

$$V_{\text{HBM}}^{\text{unfused}} = 2(n-1) \times |\mathbf{x}| \times \text{bytes\_per\_element}$$

(Each intermediate result is written once and read once from HBM.)

### 4.2 Kernel Fusion

**Definition:** Kernel fusion combines multiple operations into a **single kernel** that executes all computations without materializing intermediate results in global memory.

$$\mathbf{x}_n = f_n \circ f_{n-1} \circ \cdots \circ f_1(\mathbf{x}) \quad \text{(computed entirely in registers/shared memory)}$$

The fused HBM traffic:

$$V_{\text{HBM}}^{\text{fused}} = |\mathbf{x}| \times \text{bytes\_per\_element} \;\;(\text{read input}) + |\mathbf{x}_n| \times \text{bytes\_per\_element} \;\;(\text{write output})$$

The **bandwidth savings** are:

$$\text{Savings} = \frac{V_{\text{HBM}}^{\text{unfused}}}{V_{\text{HBM}}^{\text{fused}}} = \frac{2(n-1) \times |\mathbf{x}|}{2 \times |\mathbf{x}|} = n - 1$$

For LayerNorm, which involves ~5 point-wise operations (subtract mean, square, average, reciprocal square root, scale+shift), the savings factor is $\sim 4\times$.

### 4.3 Applicability in Transformers

Fusion is most beneficial for **memory-bound** operations — those where HBM bandwidth, not compute throughput, is the bottleneck. The **arithmetic intensity** (AI) determines this:

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes accessed from memory}}$$

| Operation | FLOPs | Bytes | AI | Bound |
|---|---|---|---|---|
| MatMul ($M \times K \times N$) | $2MKN$ | $\sim 2(MK + KN + MN) \times$ bytes | High | **Compute-bound** |
| LayerNorm | $\sim 5 \times d$ per token | $\sim 2d \times$ bytes per token | Low ($\sim 2.5$) | **Memory-bound** |
| Activation (GELU, ELU) | $\sim 1$ per element | $2 \times$ bytes per element | $\sim 0.5$ | **Memory-bound** |
| Softmax | $\sim 5N$ per row | $\sim 2N \times$ bytes per row | Low | **Memory-bound** |

Fusion provides the greatest speedup for memory-bound operations since reducing HBM traffic directly reduces the bottleneck.

---

## 5. FlashAttention

### 5.1 Standard Attention and Its Memory Bottleneck

The standard scaled dot-product attention for a single head is:

$$\boxed{\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}}$$

where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}$, $N$ is the sequence length, and $d_k$ is the head dimension.

The **naive computation** proceeds in three steps:

1. **Compute score matrix:** $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$
2. **Compute attention weights:** $\mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}$
3. **Compute output:** $\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d_k}$

**Memory problem:** Both $\mathbf{S}$ and $\mathbf{P}$ must be **materialized in HBM**. Their size is:

$$M_{\mathbf{S}} = M_{\mathbf{P}} = N^2 \times \text{bytes\_per\_element}$$

For $N = 4096$, $d_k = 128$, in BF16:

$$M_{\mathbf{S}} = 4096^2 \times 2 = 33.6 \;\text{MB per head}$$

With $n_h = 96$ heads (as in a large model):

$$M_{\text{attn}}^{\text{total}} = 96 \times 33.6 \;\text{MB} = 3.2 \;\text{GB}$$

This is a **significant fraction** of the 80 GB HBM on an H100, and the HBM read/write traffic for these matrices becomes the dominant bottleneck.

**HBM traffic for naive attention:** Each of the three steps requires reading inputs from HBM and writing outputs back:

$$V_{\text{HBM}}^{\text{naive}} = \underbrace{2 \cdot (2Nd_k + N^2)}_{\text{Step 1: read Q,K; write S}} + \underbrace{2 \cdot (N^2 + N^2)}_{\text{Step 2: read S; write P}} + \underbrace{2 \cdot (N^2 + Nd_k + Nd_k)}_{\text{Step 3: read P,V; write O}}$$

The dominant terms are all $O(N^2)$, making naive attention **memory-bandwidth-bound** for typical $N \gg d_k$.

### 5.2 FlashAttention Algorithm

**Core idea:** Compute the output $\mathbf{O}$ by processing $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ in **tiles** that fit in SRAM, **never materializing** the full $N \times N$ matrices $\mathbf{S}$ or $\mathbf{P}$ in HBM.

The key mathematical challenge is that **softmax requires global normalization** across the entire row:

$$\text{softmax}(\mathbf{s}_i)_j = \frac{e^{s_{ij}}}{\sum_{k=1}^{N} e^{s_{ik}}}$$

FlashAttention uses the **online softmax** algorithm (Milakov & Gimelshein, 2018) to compute softmax **incrementally** over tiles. The algorithm maintains running statistics $m_i$ (row-wise maximum) and $\ell_i$ (row-wise sum of exponentials) that are updated as each new tile is processed.

**Tiled computation:** Partition $\mathbf{K}$ and $\mathbf{V}$ into blocks of $B_c$ rows (columns of the attention matrix), and $\mathbf{Q}$ into blocks of $B_r$ rows:

$$\mathbf{K} = [\mathbf{K}_1; \mathbf{K}_2; \ldots; \mathbf{K}_{T_c}], \quad \mathbf{V} = [\mathbf{V}_1; \mathbf{V}_2; \ldots; \mathbf{V}_{T_c}]$$

where $T_c = \lceil N / B_c \rceil$.

For each query block $\mathbf{Q}_i$ ($B_r$ rows), iterate over KV blocks $j = 1, \ldots, T_c$:

**Step 1:** Compute the local score tile in SRAM:

$$\mathbf{S}_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d_k}} \in \mathbb{R}^{B_r \times B_c}$$

**Step 2:** Compute local row-wise maximum and update running maximum:

$$\tilde{m}_{ij} = \text{rowmax}(\mathbf{S}_{ij}) \in \mathbb{R}^{B_r}$$

$$m_i^{\text{new}} = \max(m_i^{\text{old}}, \tilde{m}_{ij})$$

**Step 3:** Compute local exponentials and update running sum:

$$\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - m_i^{\text{new}}) \in \mathbb{R}^{B_r \times B_c}$$

$$\ell_i^{\text{new}} = e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot \ell_i^{\text{old}} + \text{rowsum}(\tilde{\mathbf{P}}_{ij})$$

**Step 4:** Update the output accumulator:

$$\mathbf{O}_i^{\text{new}} = \frac{e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot \ell_i^{\text{old}} \cdot \mathbf{O}_i^{\text{old}} + \tilde{\mathbf{P}}_{ij} \mathbf{V}_j}{\ell_i^{\text{new}}}$$

After all $T_c$ iterations, $\mathbf{O}_i^{\text{final}}$ equals the exact attention output — no approximation is involved.

**Tile sizes** are chosen to maximize SRAM utilization:

$$B_c = \left\lfloor \frac{M_{\text{SRAM}}}{4 \cdot d_k \cdot \text{sizeof(element)}} \right\rfloor, \quad B_r = \min\!\left(B_c, \;\left\lfloor \frac{M_{\text{SRAM}}}{4 \cdot d_k \cdot \text{sizeof(element)}} \right\rfloor\right)$$

### 5.3 Complexity Analysis

| Metric | Naive Attention | FlashAttention |
|---|---|---|
| **FLOPs** | $O(N^2 d_k)$ | $O(N^2 d_k)$ (identical — exact computation) |
| **HBM reads/writes** | $O(N^2 + Nd_k)$ | $O\!\left(\frac{N^2 d_k}{M_{\text{SRAM}}}\right)$ |
| **Extra memory (beyond Q,K,V,O)** | $O(N^2)$ (store $\mathbf{S}$, $\mathbf{P}$) | $O(N)$ (store $m_i$, $\ell_i$ per row) |

Since $M_{\text{SRAM}} \gg d_k$ typically, the HBM access is reduced by a factor of approximately $M_{\text{SRAM}} / d_k$. For H100 with $M_{\text{SRAM}} \approx 228$ KB and $d_k = 128$ in BF16 ($= 256$ bytes per row):

$$\text{HBM reduction factor} \approx \frac{228 \times 1024}{256} \approx 912\times$$

### 5.4 Impact and Significance

FlashAttention resolves multiple bottlenecks simultaneously:

1. **Memory reduction:** The $O(N^2)$ memory for the attention matrix is eliminated, reducing to $O(N)$ auxiliary storage.

2. **Wall-clock speedup:** By reducing HBM traffic — the true bottleneck for attention — FlashAttention achieves **2–4× speedup** over standard attention implementations despite performing the same number of FLOPs.

3. **Enabling longer sequences:** Without the $N^2$ memory overhead, much longer sequences become feasible. This is why earlier linear/subquadratic attention approximations have been largely abandoned in favor of FlashAttention — exact attention is now fast enough.

### 5.5 FlashAttention-2 and FlashAttention-3

| Version | Key Improvements |
|---|---|
| **FlashAttention-2** | (1) Reduced non-matmul FLOPs (rewrote online softmax to minimize non-GEMM operations); (2) Better work partitioning among warps within a thread block (parallelism along sequence length instead of head dimension); (3) Added parallelism across the sequence length dimension |
| **FlashAttention-3** | (1) Optimized for Hopper (H100) architecture — exploits **asynchronous WGMMA** (Warpgroup Matrix Multiply-Accumulate) instructions; (2) FP8 attention support with per-tile quantization; (3) Exploits **TMA** (Tensor Memory Accelerator) for efficient data movement between HBM and shared memory |

---

## 6. Mixed Precision Training

### 6.1 Floating-Point Number Representation

A floating-point number $x$ in IEEE 754 format is represented as:

$$\boxed{x = (-1)^{s} \times 2^{E - \text{bias}} \times \left(1 + \sum_{i=1}^{p} b_i \cdot 2^{-i}\right)}$$

where:
- $s \in \{0, 1\}$: sign bit
- $E$: stored exponent value (unsigned integer)
- $\text{bias} = 2^{e-1} - 1$: exponent bias ($e$ = number of exponent bits)
- $p$: number of mantissa bits
- $b_i$: individual mantissa bits

The three components control distinct properties:

| Component | Controls | More bits → |
|---|---|---|
| **Sign** ($s$) | Positive/negative | Always 1 bit |
| **Exponent** ($E$, $e$ bits) | **Dynamic range** (magnitude span) | Wider range of representable magnitudes |
| **Mantissa** ($b_i$, $p$ bits) | **Precision** (significant figures) | Finer resolution between consecutive numbers |

### 6.2 Format Comparison

| Format | Total Bits | Sign | Exponent ($e$) | Mantissa ($p$) | Bias | $\epsilon$ (machine epsilon) | Dynamic Range |
|---|---|---|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 | 127 | $2^{-23} \approx 1.19 \times 10^{-7}$ | $\sim 10^{\pm 38}$ |
| FP16 | 16 | 1 | 5 | 10 | 15 | $2^{-10} \approx 9.77 \times 10^{-4}$ | $\sim 10^{\pm 4.8}$ |
| BF16 | 16 | 1 | 8 | 7 | 127 | $2^{-7} \approx 7.81 \times 10^{-3}$ | $\sim 10^{\pm 38}$ |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | 7 | $2^{-3} = 0.125$ | $\sim 10^{\pm 2.4}$ |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | 15 | $2^{-2} = 0.25$ | $\sim 10^{\pm 4.8}$ |

**Machine epsilon** $\epsilon$ is defined as the smallest $\epsilon > 0$ such that $\text{fl}(1 + \epsilon) \neq \text{fl}(1)$:

$$\boxed{\epsilon = 2^{-p}}$$

where $p$ is the number of mantissa bits.

**Dynamic range** is determined by the exponent bits:

$$x_{\max} = (2 - 2^{-p}) \times 2^{2^{e-1} - 1}, \quad x_{\min}^{\text{normal}} = 2^{-(2^{e-1} - 2)}$$

**Key trade-off:**

- **BF16** sacrifices precision (only 7 mantissa bits vs. FP16's 10) but **preserves the full FP32 dynamic range** (8 exponent bits). This is critical for training stability because gradient magnitudes can span many orders of magnitude.
- **FP16** has better precision but a much narrower dynamic range ($\sim 10^{\pm 4.8}$), making overflow/underflow more likely.

The number of representable values between consecutive powers of 2 (e.g., in $[1, 2]$) is $2^p$:

| Format | Values in $[1, 2]$ |
|---|---|
| FP32 | $2^{23} = 8{,}388{,}608$ |
| FP16 | $2^{10} = 1{,}024$ |
| BF16 | $2^7 = 128$ |
| FP8 (E4M3) | $2^3 = 8$ |
| FP8 (E5M2) | $2^2 = 4$ |

### 6.3 FP16/BF16 Mixed Precision Training

Naively replacing all FP32 tensors with FP16/BF16 causes training divergence due to three failure modes, each addressed by a specific technique:

#### Trick 1: FP32 Master Copy of Weights

**Problem — Weight update underflow:** If a weight $w$ has magnitude $\sim 1$ and the gradient-based update $\Delta w$ has magnitude $\sim 10^{-5}$, then in FP16:

$$\text{fl}_{16}(w + \Delta w) = \text{fl}_{16}(1.0 + 0.00001) = \text{fl}_{16}(1.0) = 1.0$$

because $\Delta w < \epsilon_{\text{FP16}} \times |w| \approx 10^{-3}$. The update is **lost entirely**. Once weights reach zero through underflow, they remain zero permanently (no gradient signal).

**Solution:** Maintain a **FP32 master copy** of weights $\mathbf{w}^{(32)}$. The training loop becomes:

$$\mathbf{w}^{(16)} = \text{cast}_{16}(\mathbf{w}^{(32)}) \quad \text{(for forward/backward)}$$
$$\mathbf{g}^{(16)} = \nabla_{\mathbf{w}^{(16)}} \mathcal{L} \quad \text{(computed in 16-bit)}$$
$$\mathbf{w}^{(32)} \leftarrow \mathbf{w}^{(32)} - \eta \cdot \text{cast}_{32}(\mathbf{g}^{(16)}) \quad \text{(update in FP32)}$$

#### Trick 2: Loss Scaling

**Problem — Gradient underflow:** Gradients are often much smaller than 1 (e.g., $|\mathbf{g}| \sim 10^{-6}$), falling below the minimum representable value in FP16 ($\sim 5.96 \times 10^{-8}$) or being too imprecise in BF16.

**Solution:** Scale the loss before the backward pass and unscale gradients afterward:

$$\hat{\mathcal{L}} = \alpha \cdot \mathcal{L} \quad (\text{scaled loss, } \alpha \gg 1)$$

$$\hat{\mathbf{g}}^{(16)} = \nabla_{\mathbf{w}} \hat{\mathcal{L}} = \alpha \cdot \nabla_{\mathbf{w}} \mathcal{L} = \alpha \cdot \mathbf{g}$$

$$\mathbf{g}^{(16)}_{\text{true}} = \frac{\hat{\mathbf{g}}^{(16)}}{\alpha} \quad (\text{unscale before optimizer step})$$

By linearity of differentiation, scaling the loss by $\alpha$ scales all gradients by $\alpha$, shifting them into a representable range. The unscaling restores the correct values before any further processing (clipping, optimizer step).

**Dynamic loss scaling** starts with a large $\alpha$ and halves it whenever overflow (Inf/NaN) is detected in gradients, doubling it after a fixed number of successful steps.

#### Trick 3: FP32 Accumulation

**Problem — Accumulation error:** When summing many small values (e.g., computing means, batch norms, reductions), the running sum grows large while individual addends remain small, causing catastrophic cancellation:

$$\text{fl}_{16}\!\left(\sum_{i=1}^{N} x_i\right) \neq \sum_{i=1}^{N} x_i \quad \text{when } N \text{ is large and } x_i \text{ are small}$$

The relative error of naive summation in precision $\epsilon$ is bounded by:

$$\left|\frac{\text{fl}(\sum x_i) - \sum x_i}{\sum x_i}\right| \leq (N-1) \cdot \epsilon + O(\epsilon^2)$$

For $N = 4096$ and $\epsilon_{\text{BF16}} \approx 7.8 \times 10^{-3}$: relative error $\leq 32$, which means the result can be completely wrong.

**Solution:** Accumulate intermediate sums in FP32 even when inputs/outputs are in 16-bit:

$$\text{acc}^{(32)} = \sum_{i=1}^{N} \text{cast}_{32}(x_i^{(16)})$$
$$\text{result}^{(16)} = \text{cast}_{16}(\text{acc}^{(32)})$$

This is typically implemented at the hardware level in Tensor Cores, which accept FP16/BF16 inputs but accumulate in FP32 internally:

$$\mathbf{D}^{(32)} = \mathbf{A}^{(16)} \cdot \mathbf{B}^{(16)} + \mathbf{C}^{(32)}$$

### 6.4 Per-Element Memory Cost Summary (Mixed Precision)

The full mixed precision training memory per parameter is:

| Component | FP32 Baseline | BF16 Mixed | Description |
|---|---|---|---|
| Master weights | 4 B | 4 B (FP32) | Always FP32 for accurate updates |
| Working weights | — | 2 B (BF16) | Used in forward/backward |
| Gradients | 4 B | 2 B (BF16) | Computed in BF16 |
| Optimizer state 1 (momentum) | 4 B | 4 B (FP32) | Adam first moment |
| Optimizer state 2 (variance) | 4 B | 4 B (FP32) | Adam second moment |
| Grad accumulation buffer | — | 4 B (FP32) | Optional FP32 accumulation |
| **Total** | **16 B** | **20 B** (with accum) or **16 B** (without) | |

### 6.5 FP8 Pretraining

FP8 matrix multiplications on H100 achieve **twice the peak throughput** of BF16:

$$\Phi_{\text{FP8}}^{\text{H100}} = 1{,}979 \;\text{TFLOP/s} \approx 2 \times \Phi_{\text{BF16}}^{\text{H100}} = 989 \;\text{TFLOP/s}$$

However, FP8 introduces severe **stability challenges** due to the extremely limited dynamic range and precision.

#### Quantization for FP8

Converting a tensor $\mathbf{x}$ from high precision to FP8 requires **scaling** to fit within FP8's representable range:

$$\mathbf{x}_{\text{FP8}} = \text{cast}_{\text{FP8}}\!\left(\frac{\mathbf{x}}{s_x}\right), \quad s_x = \frac{\max(|\mathbf{x}|)}{x_{\max}^{\text{FP8}}}$$

where $x_{\max}^{\text{FP8}} = 448$ for E4M3 or $57344$ for E5M2.

**DeepSeek-V3's tile-wise quantization** computes separate scaling factors per tile to reduce the impact of outlier values:

- **Activations/inputs:** tiles of size $1 \times 128$ (per-token granularity)
- **Weights:** tiles of size $128 \times 128$

For a tile $\mathcal{T}$:

$$s_{\mathcal{T}} = \frac{\max_{x \in \mathcal{T}}(|x|)}{x_{\max}^{\text{FP8}}}$$

This is much more robust than per-tensor scaling because a single outlier value in one tile doesn't compress the dynamic range of all other tiles.

#### FP8 Mixed Precision Configurations

| Configuration | GEMM Precision | Master Weights | Gradients | Optimizer States | Total Memory per Param |
|---|---|---|---|---|---|
| BF16 baseline (with FP32 accum) | BF16 | FP32 (4B) | BF16 (2B) | FP32+FP32 (8B) | **$\sim$20 B** |
| Transformer Engine | FP8 | FP32 (4B) | FP32 (4B) | FP32+FP32 (8B) | **16 B** (20% reduction) |
| FP8-LM O3 | FP8 | FP16 (2B) | FP8 (1B) | FP8+FP16 (3B) | **9 B** (55% reduction) |
| DeepSeek-V3 | FP8 | FP32 (4B) | BF16 (2B) | BF16+BF16 (4B) | **15 B** (25% reduction) |
| Nanotron FP8 | FP8 | FP32 (4B) | FP8 (1B) | FP8+FP8 (2B) | **10 B** (50% reduction) |

The primary stability risk of FP8 pretraining is that the E4M3 format has only $2^3 = 8$ representable values per binade and a dynamic range of only $\sim 10^{\pm 2.4}$. Loss divergence typically manifests as:

1. Gradient underflow → zero gradients → stalled learning
2. Activation overflow → NaN propagation → loss explosion
3. Accumulation error → incorrect weight updates → slow divergence

The instability worsens with higher learning rates because the gradient magnitudes increase, exceeding FP8's representable range more frequently:

$$\text{P}(\text{overflow}) = \text{P}\!\left(|\eta \cdot g| > x_{\max}^{\text{FP8}}\right) \uparrow \text{ as } \eta \uparrow$$

### 6.6 Computational Throughput Benefit

The throughput gain from lower precision comes from two sources:

**1. Higher peak FLOPS:** Tensor Cores execute lower-precision operations in fewer cycles:

$$\text{Speedup}_{\text{peak}} = \frac{\Phi_{\text{FP8}}}{\Phi_{\text{BF16}}} = \frac{1979}{989} \approx 2.0\times$$

**2. Reduced memory traffic:** Smaller data types reduce HBM bandwidth requirements:

$$\text{Bandwidth reduction} = \frac{\text{bytes}_{\text{BF16}}}{\text{bytes}_{\text{FP8}}} = \frac{2}{1} = 2.0\times$$

For memory-bound operations, this directly translates to a $2\times$ speedup. For compute-bound operations (GEMMs), the peak FLOPS doubling provides the speedup.

---

## 7. Connecting All Concepts: The GPU Performance Model

The overall throughput of a training step on a single GPU is determined by the **roofline model**:

$$\boxed{\text{Attainable FLOP/s} = \min\!\left(\Phi_{\text{peak}}, \;\; \text{AI} \times \beta_{\text{HBM}}\right)}$$

where:
- $\Phi_{\text{peak}}$: peak compute throughput (FLOP/s) at the chosen precision
- $\text{AI} = \frac{\text{FLOPs}}{\text{Bytes transferred}}$: arithmetic intensity of the operation
- $\beta_{\text{HBM}}$: HBM bandwidth (bytes/s)

The **ridge point** — the AI where the transition from memory-bound to compute-bound occurs — is:

$$\text{AI}_{\text{ridge}} = \frac{\Phi_{\text{peak}}}{\beta_{\text{HBM}}}$$

For H100 in BF16:

$$\text{AI}_{\text{ridge}}^{\text{BF16}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \;\text{FLOPs/byte}$$

For H100 in FP8:

$$\text{AI}_{\text{ridge}}^{\text{FP8}} = \frac{1979 \times 10^{12}}{3.35 \times 10^{12}} \approx 591 \;\text{FLOPs/byte}$$

Operations with $\text{AI} < \text{AI}_{\text{ridge}}$ (attention, LayerNorm, activations) benefit most from **kernel fusion** and **FlashAttention** (which reduce bytes transferred). Operations with $\text{AI} > \text{AI}_{\text{ridge}}$ (large GEMMs) benefit most from **lower precision** (which increases $\Phi_{\text{peak}}$).

This unified view explains why **all three techniques — kernel fusion, FlashAttention, and mixed precision — are complementary and simultaneously necessary** for maximizing GPU utilization in modern large-scale training.