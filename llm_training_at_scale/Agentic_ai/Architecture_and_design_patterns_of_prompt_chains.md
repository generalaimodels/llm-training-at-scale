

# Chapter 1: Prompt Chaining

## 1.2 Architecture and Design Patterns of Prompt Chains

---

### 1.2.1 Chain Topologies

A prompt chain's **topology** is the graph structure that governs how prompt nodes are connected, how data flows between them, and which execution orderings are valid. The choice of topology is not arbitrary — it must be derived from the **dependency structure** of the underlying task. An incorrectly chosen topology either introduces unnecessary sequential bottlenecks (increasing latency without accuracy gain) or removes necessary dependencies (causing information loss and correctness failures).

We formalize all topologies within the DAG framework $G = (V, E)$ introduced in Section 1.1, extending it where cyclic structures (iterative chains) require generalization to directed graphs with controlled cycles.

---

#### Sequential / Linear Chains

**Definition.** A sequential chain is a **path graph** $G = (V, E)$ where $V = \{v_1, v_2, \dots, v_n\}$ and $E = \{(v_i, v_{i+1}) : 1 \leq i < n\}$. Each node has exactly one predecessor (except $v_1$) and one successor (except $v_n$).

The computation is strict function composition:

$$
y = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1(x)
$$

Equivalently, the recurrence relation:

$$
y_0 = x, \quad y_i = f_i(y_{i-1}, \theta_i) \quad \text{for } i = 1, 2, \dots, n
$$

**Execution semantics.** Strictly serial — step $i$ cannot begin until step $i-1$ has completed and its output has been parsed. The total latency is the sum of individual step latencies:

$$
T_{\text{seq}} = \sum_{i=1}^{n} T_i
$$

where $T_i$ includes model inference time, network latency, and parsing time for step $i$.

**Structural properties:**

| Property | Value |
|----------|-------|
| Graph width (max antichain) | $1$ |
| Critical path length | $n$ |
| Maximum parallelism | $1$ |
| In-degree of each node | $\leq 1$ |
| Out-degree of each node | $\leq 1$ |
| Topological orderings | Exactly $1$ |

**Information flow analysis.** In a strictly sequential chain without skip connections, the Data Processing Inequality (DPI) applies at every step:

$$
I(x; y_n) \leq I(x; y_{n-1}) \leq \cdots \leq I(x; y_1) \leq H(x)
$$

Each step is a potential information bottleneck. The severity of information loss at step $i$ depends on the fidelity of the output parser $\phi_i$ and the completeness of the prompt template $\tau_i$ in requesting all downstream-relevant information.

**When to use sequential chains:**

1. The task has an inherent **linear dependency structure** — each subtask genuinely requires the output of the preceding subtask.
2. Subtasks are **not parallelizable** — e.g., outline generation must precede section drafting, which must precede editing.
3. The number of steps $n$ is small (typically $n \leq 7$), keeping error accumulation and latency manageable.

**Concrete example — Document summarization pipeline:**

```
Raw Document → f₁: Extract Key Claims → f₂: Group by Theme →
f₃: Synthesize per Theme → f₄: Generate Executive Summary → Output
```

Each step receives a progressively distilled representation. The raw document ($\sim$10,000 tokens) is reduced to key claims ($\sim$1,000 tokens), then grouped ($\sim$800 tokens), then synthesized ($\sim$500 tokens), then summarized ($\sim$200 tokens).

**Error propagation model.** If each step has independent error probability $\epsilon_i = 1 - p_i$, the probability of a correct final output is:

$$
P(\text{correct}) = \prod_{i=1}^n p_i = \prod_{i=1}^n (1 - \epsilon_i)
$$

For homogeneous error rates $\epsilon_i = \epsilon$:

$$
P(\text{correct}) = (1 - \epsilon)^n \approx e^{-n\epsilon} \quad \text{for small } \epsilon
$$

This exponential decay mandates that each step maintain very high accuracy. For $n = 5$ steps to achieve 90% overall success, each step needs $p_i \geq 0.9^{1/5} \approx 0.979$ — approximately 98% per-step accuracy.

---

#### Branching Chains (Fan-Out)

**Definition.** A fan-out node $v_i$ has out-degree $> 1$: its output is dispatched to multiple independent downstream nodes $\{v_{j_1}, v_{j_2}, \dots, v_{j_k}\}$.

$$
E_{\text{fan-out}} = \{(v_i, v_{j_1}), (v_i, v_{j_2}), \dots, (v_i, v_{j_k})\}
$$

The computation from the fan-out point:

$$
y_{j_m} = f_{j_m}(y_i, \theta_{j_m}) \quad \text{for } m = 1, 2, \dots, k
$$

All $k$ downstream computations receive the **same input** $y_i$ but apply **different prompt functions** $f_{j_m}$. Since there are no edges among $\{v_{j_1}, \dots, v_{j_k}\}$, they form an **antichain** and can execute in parallel.

```
                ┌─── f_{j₁}(y_i) ──→ y_{j₁}
                │
y_i ────────────┼─── f_{j₂}(y_i) ──→ y_{j₂}
                │
                └─── f_{j₃}(y_i) ──→ y_{j₃}
```

**Execution semantics.** All branches execute concurrently. The latency of the fan-out stage is determined by the **slowest branch**:

$$
T_{\text{fan-out}} = \max_{m \in \{1, \dots, k\}} T_{j_m}
$$

**Use cases:**

| Pattern | Description |
|---------|-------------|
| **Multi-perspective analysis** | Same document analyzed for sentiment, entities, and key facts simultaneously |
| **Multi-model ensemble** | Same prompt sent to different models; outputs compared or aggregated |
| **Multi-format generation** | Same content formatted as email, report, and presentation simultaneously |
| **Redundancy for reliability** | Same prompt executed $k$ times with different temperatures; majority vote on outputs |

**Formal parallel composition operator:**

$$
(f_{j_1} \| f_{j_2} \| \cdots \| f_{j_k})(y_i) = \bigl(f_{j_1}(y_i),\; f_{j_2}(y_i),\; \dots,\; f_{j_k}(y_i)\bigr)
$$

The output is a **tuple** of individual outputs, requiring a downstream aggregation or routing mechanism to combine them.

**Information-theoretic perspective.** Fan-out does not create information — it creates **multiple views** of the same information. Each branch extracts different aspects:

$$
I(y_i; y_{j_m}) \leq H(y_i) \quad \text{for each } m
$$

However, the joint information across all branches can exceed any single branch's information about downstream targets:

$$
I\bigl((y_{j_1}, y_{j_2}, \dots, y_{j_k}); Y_{\text{final}}\bigr) \geq \max_m I(y_{j_m}; Y_{\text{final}})
$$

This **information amplification through diverse views** is the theoretical justification for fan-out architectures.

---

#### Merging Chains (Fan-In)

**Definition.** A fan-in node $v_j$ has in-degree $> 1$: it receives and aggregates outputs from multiple upstream nodes $\{v_{i_1}, v_{i_2}, \dots, v_{i_k}\}$.

$$
\text{Pa}(v_j) = \{v_{i_1}, v_{i_2}, \dots, v_{i_k}\}
$$

The computation at the fan-in node:

$$
y_j = f_j\bigl(y_{i_1}, y_{i_2}, \dots, y_{i_k},\; \theta_j\bigr) = f_j\bigl(\{y_m : v_m \in \text{Pa}(v_j)\},\; \theta_j\bigr)
$$

The aggregation prompt template $\tau_j$ must include placeholders for all upstream outputs:

```
Given the following analyses:

**Sentiment Analysis:** {y_sentiment}
**Entity Extraction:** {y_entities}
**Key Facts:** {y_facts}

Synthesize these into a unified assessment.
```

**Execution semantics.** The fan-in node cannot execute until **all** parent nodes have completed. Its wait time is:

$$
T_{\text{wait}} = \max_{m \in \text{Pa}(v_j)} T_{i_m}^{\text{completion}}
$$

where $T_{i_m}^{\text{completion}}$ is the wall-clock time at which parent $v_{i_m}$ finishes.

**Aggregation strategies:**

| Strategy | Implementation | When to Use |
|----------|---------------|-------------|
| **Concatenation** | Simply join all upstream outputs into the prompt | Upstream outputs are complementary, non-overlapping |
| **Structured merge** | Insert each output into a labeled section of the template | Outputs need to be distinguished by source |
| **Summarization-then-merge** | Summarize each upstream output before merging | Upstream outputs are verbose; context window pressure |
| **Voting / majority rule** | Count occurrences of each answer across branches | Ensemble pattern for reliability |
| **Weighted combination** | Assign confidence scores to each upstream output | Different branches have different reliability |
| **Conflict resolution** | Explicitly prompt the LLM to resolve contradictions | Upstream outputs may be contradictory |

**Context window constraint at fan-in nodes.** The fan-in node faces the most severe context window pressure in the chain, as it must accommodate all upstream outputs simultaneously:

$$
|\text{context}_j| = |\tau_j| + \sum_{m \in \text{Pa}(v_j)} |y_{i_m}|
$$

If this exceeds the context window $C$, mitigation strategies include:

1. **Upstream summarization**: compress each $y_{i_m}$ before forwarding.
2. **Hierarchical fan-in**: aggregate pairwise in a binary tree structure, reducing per-step input to two upstream outputs.
3. **Selective forwarding**: forward only the most relevant portions of each upstream output.

**Hierarchical fan-in as binary tree:**

$$
y_{\text{agg}}^{(\ell)} = f_{\text{merge}}\bigl(y_{\text{agg},\text{left}}^{(\ell-1)},\; y_{\text{agg},\text{right}}^{(\ell-1)}\bigr)
$$

This reduces the maximum per-node input from $k$ upstream outputs to $2$, at the cost of $\lceil \log_2 k \rceil$ aggregation levels.

---

#### Conditional / Routing Chains

**Definition.** A routing node $v_r$ evaluates a condition on its input and dispatches execution to one of multiple downstream branches. Only the selected branch executes.

$$
y = \begin{cases}
f_A(y_r) & \text{if } c(y_r) = A \\
f_B(y_r) & \text{if } c(y_r) = B \\
f_C(y_r) & \text{if } c(y_r) = C \\
\vdots
\end{cases}
$$

where $c: \mathcal{Y}_r \to \{A, B, C, \dots\}$ is the **routing function** that determines which branch to follow.

```
                     ┌─── [Route A] ──→ f_A ──→
y_r ──→ c(y_r) ─────┼─── [Route B] ──→ f_B ──→
                     └─── [Route C] ──→ f_C ──→
```

**Routing function implementations:**

| Implementation | Description | Determinism |
|---------------|-------------|-------------|
| **Rule-based** | Programmatic `if/else` on parsed output fields (e.g., `if sentiment == "negative"`) | Deterministic |
| **LLM-as-classifier** | A dedicated LLM call that classifies the input into one of $k$ categories | Stochastic |
| **Embedding similarity** | Compute embedding of $y_r$ and route to the nearest cluster centroid | Deterministic (given fixed centroids) |
| **Threshold-based** | Route based on a numeric score (e.g., confidence > 0.8 → fast path, else → detailed analysis) | Deterministic |
| **LLM-as-planner** | The LLM itself decides which step to execute next (approaches agentic behavior) | Stochastic |

**Formal conditional composition operator:**

$$
\text{Cond}(c, f_A, f_B)(x) = \begin{cases} f_A(x) & \text{if } c(x) = \text{true} \\ f_B(x) & \text{if } c(x) = \text{false} \end{cases}
$$

Generalized to $k$ branches:

$$
\text{Route}(c, \{f_j\}_{j=1}^k)(x) = f_{c(x)}(x)
$$

**Design considerations:**

1. **Exhaustiveness**: the routing function must cover all possible cases. An unhandled case leads to chain failure. Always include a **default / fallback route**.
2. **Mutual exclusivity**: ideally, exactly one route should match. Overlapping conditions create ambiguity.
3. **Routing accuracy**: if an LLM performs routing, its classification accuracy directly impacts chain correctness. The routing step should be the simplest possible classification task — minimize the LLM's cognitive load at the routing node.

**Latency analysis.** Routing reduces expected latency compared to executing all branches:

$$
\mathbb{E}[T_{\text{routing}}] = T_{\text{route}} + \sum_{j=1}^k P(c(x) = j) \cdot T_{f_j}
$$

versus the fan-out-then-aggregate pattern:

$$
T_{\text{fan-out}} = \max_{j} T_{f_j}
$$

Routing is faster when the expected branch latency is lower than the maximum branch latency, which is typical when branches have heterogeneous complexity.

**Cost optimization via routing.** Routing enables **cost-aware chain design**: simple inputs are routed to cheap, fast chains (fewer steps, smaller models), while complex inputs are routed to expensive, thorough chains (more steps, larger models):

$$
\text{Cost}_{\text{routing}} = \sum_{j=1}^k P(c(x) = j) \cdot \text{Cost}_{f_j}
$$

This weighted average is typically much lower than the cost of always executing the most expensive branch.

---

#### Looping / Iterative Chains

**Definition.** An iterative chain introduces a **controlled cycle**: the output of a step is fed back as input to the same (or an earlier) step, repeating until a convergence criterion or maximum iteration count is reached.

$$
y^{(0)} = x, \quad y^{(t+1)} = f(y^{(t)}, \theta) \quad \text{for } t = 0, 1, \dots
$$

Termination:

$$
\text{stop at } t^* = \min\Bigl\{t : \text{done}(y^{(t)}) = \text{true}\Bigr\} \quad \text{or} \quad t^* = t_{\max}
$$

where $\text{done}: \mathcal{Y} \to \{\text{true}, \text{false}\}$ is the **stopping predicate**.

```
        ┌──────────────────────┐
        │                      │
x ──→ f(·) ──→ done? ──[no]───┘
                  │
                [yes]
                  │
                  ▼
                output
```

**Strictly speaking, iterative chains violate the DAG constraint** — the execution graph contains a cycle. However, if we **unroll** the loop, treating each iteration as a separate node ($v^{(0)}, v^{(1)}, \dots, v^{(t^*)}$), the unrolled graph is a DAG (a path graph of variable length). The cycle exists at the **design level** but not at the **execution level** (since each iteration produces a distinct node in the execution trace).

**Convergence analysis.** For the iterative chain to be well-behaved, we need guarantees on convergence. Define the **quality metric** $q: \mathcal{Y} \to \mathbb{R}$ measuring output quality:

$$
q^{(t)} = q(y^{(t)})
$$

The chain converges if $\{q^{(t)}\}_{t=0}^{\infty}$ is a monotonically non-decreasing sequence bounded above:

$$
q^{(0)} \leq q^{(1)} \leq \cdots \leq q^{(t^*)} \leq q_{\max}
$$

In practice, LLMs do **not** guarantee monotonic improvement. An iteration may degrade quality (introduce hallucinations, over-edit, drift from the original intent). Mitigation strategies:

| Strategy | Mechanism |
|----------|-----------|
| **Quality gate** | Only accept iteration output if $q^{(t+1)} > q^{(t)}$; otherwise revert to $y^{(t)}$ |
| **Maximum iterations** | Hard cap $t_{\max}$ preventing infinite loops |
| **Delta threshold** | Stop if $|q^{(t+1)} - q^{(t)}| < \delta$ (diminishing returns) |
| **Separate evaluator** | A distinct LLM call (or programmatic check) evaluates whether the iteration improved quality |
| **Accumulating context** | Include all previous iterations in the prompt to prevent cyclic repetition |

**Formal iterative composition operator:**

$$
\text{Iterate}(f, \text{done}, t_{\max})(x) = y^{(t^*)}, \quad t^* = \min\bigl(\min\{t : \text{done}(y^{(t)})\},\; t_{\max}\bigr)
$$

**Relationship to fixed-point iteration.** If $f$ is a **contraction mapping** on metric space $(\mathcal{Y}, d)$:

$$
d(f(y), f(y')) \leq \lambda \cdot d(y, y') \quad \text{for some } \lambda \in [0, 1)
$$

then by the **Banach fixed-point theorem**, the iteration converges to a unique fixed point $y^* = f(y^*)$, and the convergence rate is geometric:

$$
d(y^{(t)}, y^*) \leq \frac{\lambda^t}{1 - \lambda} \cdot d(y^{(1)}, y^{(0)})
$$

In practice, LLM-based iterations rarely satisfy the contraction condition rigorously, but empirically exhibit convergence-like behavior for well-designed refinement prompts (e.g., "identify and fix one error in the following text").

**Use cases:**

- **Iterative refinement**: draft → critique → revise → critique → revise → …
- **Self-correction**: generate → validate → if errors found, regenerate with error feedback
- **Progressive elaboration**: outline → expand section 1 → expand section 2 → …
- **Negotiation / debate**: perspective A → counterargument B → rebuttal A → …

**Cost and latency.** Both grow linearly with the number of iterations:

$$
T_{\text{iter}} = \sum_{t=0}^{t^*} T_f^{(t)}, \quad C_{\text{iter}} = \sum_{t=0}^{t^*} C_f^{(t)}
$$

If iteration context grows (accumulating previous outputs), per-iteration cost also grows:

$$
C_f^{(t)} = O(|x| + t \cdot |\bar{y}|)
$$

where $|\bar{y}|$ is the average output length per iteration.

---

#### Hierarchical Chains

**Definition.** A hierarchical chain is a **nested structure** where individual nodes in a parent chain are themselves sub-chains. This is the prompt-chain analog of **subroutines** in programming or **hierarchical task networks (HTN)** in AI planning.

Formally, let $G_{\text{parent}} = (V_p, E_p)$ be the parent chain. For some node $v_i \in V_p$, the function $f_i$ is not a single LLM invocation but is itself a chain:

$$
f_i = g_{i,m_i} \circ g_{i,m_i-1} \circ \cdots \circ g_{i,1}
$$

where $g_{i,1}, \dots, g_{i,m_i}$ are the sub-chain steps within node $v_i$.

```
Parent chain:  x → [f₁] → [f₂ = sub-chain] → [f₃] → y
                              │
                    ┌─────────┴──────────┐
Sub-chain f₂:      g₂₁ → g₂₂ → g₂₃ → g₂₄
```

**Recursive decomposition.** Hierarchical chains can be nested to arbitrary depth:

$$
f_i \to g_{i,j} \to h_{i,j,k} \to \cdots
$$

Each level of nesting applies the same decomposition principle: breaking a complex step into simpler sub-steps. The total depth of nesting is bounded by the intrinsic complexity hierarchy of the task.

**Encapsulation.** Each sub-chain is **encapsulated**: the parent chain interacts with it only through a well-defined input/output interface. Internal sub-chain details are hidden from the parent chain. This enables:

1. **Reusability**: a sub-chain for "extract entities from text" can be reused across multiple parent chains.
2. **Independent optimization**: sub-chains can be optimized, tested, and versioned independently.
3. **Abstraction management**: the parent chain operates at a higher level of abstraction.

**Interface contract between parent and sub-chain:**

$$
\text{SubChain}_i: \mathcal{Y}_{i-1} \xrightarrow{\text{input schema}} \mathcal{Y}_i \xrightarrow{\text{output schema}}
$$

The parent chain specifies what it provides (input schema) and what it expects (output schema). The sub-chain's internal structure is free to vary as long as it honors this contract — the **Liskov Substitution Principle** applied to prompt chains.

---

#### Parallel Chains

**Definition.** Parallel chains are **independent sub-chains** with no data dependencies between them, enabling concurrent execution. In graph terms, they form disconnected subgraphs (connected only through shared ancestors or descendants).

$$
y_A = f_A(x_A), \quad y_B = f_B(x_B) \quad \text{(no edge between } f_A \text{ and } f_B\text{)}
$$

**Execution semantics.** All parallel chains execute simultaneously. The wall-clock time is:

$$
T_{\text{parallel}} = \max_j T_{f_j}
$$

versus serial execution:

$$
T_{\text{serial}} = \sum_j T_{f_j}
$$

The speedup factor:

$$
S = \frac{T_{\text{serial}}}{T_{\text{parallel}}} = \frac{\sum_j T_{f_j}}{\max_j T_{f_j}}
$$

For $k$ chains with equal latency $T$, $S = k$ — linear speedup.

**Implementation requirements:**

- **Async / concurrent execution**: use `asyncio` (Python), thread pools, or distributed task queues.
- **Independent state**: each parallel chain must operate on independent state to avoid race conditions.
- **Synchronization barrier**: a downstream fan-in node waits for all parallel chains to complete before proceeding.

```python
# Conceptual implementation
import asyncio

async def parallel_chains(x, chains):
    tasks = [chain.arun(x) for chain in chains]
    results = await asyncio.gather(*tasks)
    return results
```

**Amdahl's Law applied to prompt chains.** If a fraction $s$ of the chain is inherently sequential and the remaining fraction $1 - s$ is parallelizable across $k$ workers:

$$
S(k) = \frac{1}{s + \frac{1 - s}{k}}
$$

As $k \to \infty$, $S(k) \to 1/s$, meaning the sequential portion is the bottleneck. For prompt chains, the sequential portion includes routing decisions, aggregation steps, and any step with data dependencies on all predecessors.

---

#### Diamond Chains

**Definition.** A diamond chain is the composition of **fan-out followed by fan-in**: a single node distributes to multiple parallel branches, which are then aggregated by a single downstream node.

```
              ┌─── f_A ───┐
x ──→ f₀ ────┤            ├──→ f_agg ──→ y
              ├─── f_B ───┤
              └─── f_C ───┘
```

The computation:

$$
y_0 = f_0(x), \quad y_A = f_A(y_0), \quad y_B = f_B(y_0), \quad y_C = f_C(y_0)
$$
$$
y = f_{\text{agg}}(y_A, y_B, y_C)
$$

**The diamond pattern is the most common non-trivial topology** in production prompt chains, as it naturally arises whenever a task decomposes into independent subtasks that must be recombined.

**Formal composition:**

$$
\text{Diamond}(f_0, \{f_j\}_{j=1}^k, f_{\text{agg}})(x) = f_{\text{agg}}\bigl((f_1 \| f_2 \| \cdots \| f_k)(f_0(x))\bigr)
$$

**Latency:**

$$
T_{\text{diamond}} = T_{f_0} + \max_j T_{f_j} + T_{f_{\text{agg}}}
$$

**Example — Multi-aspect document analysis:**

```
Document → f₀: Preprocess & Clean
          → f₁: Legal Risk Analysis      ──┐
          → f₂: Financial Impact Analysis ──┼──→ f_agg: Integrated Report
          → f₃: Regulatory Compliance    ──┘
```

---

#### Hybrid Topologies

Real-world prompt chains typically combine multiple topological patterns into **hybrid structures**. A production-grade chain might include sequential preprocessing, a diamond pattern for multi-aspect analysis, conditional routing based on analysis results, and iterative refinement of the final output.

**Formal representation.** Any hybrid topology is a directed graph $G = (V, E)$ that can be decomposed into **structural motifs**:

$$
G = \text{Seq}(M_1, M_2, \dots, M_p)
$$

where each motif $M_j$ is one of: sequential segment, fan-out, fan-in, diamond, conditional branch, or iterative loop.

**Motif composition rules:**

| Motif A → Motif B | Valid? | Constraint |
|-------------------|--------|------------|
| Sequential → Fan-out | ✓ | Last sequential node feeds fan-out |
| Fan-out → Fan-in | ✓ | All fan-out branches must terminate at fan-in |
| Fan-in → Sequential | ✓ | Fan-in output feeds sequential chain |
| Conditional → Any | ✓ | Each branch can contain any motif |
| Loop → Sequential | ✓ | Loop termination feeds sequential continuation |
| Fan-out → Loop | ✓ | Each branch can iterate independently |

**Example hybrid topology — Automated research assistant:**

```
Query → f₁: Query Decomposition (Sequential)
      → [Conditional: Simple vs. Complex]
         → [Simple]: f₂: Direct Answer
         → [Complex]: 
            → f₃: Literature Search (Fan-out across databases)
               → f₃ₐ: arXiv search
               → f₃ᵦ: Semantic Scholar search  
               → f₃ᵧ: Google Scholar search
            → f₄: Merge results (Fan-in)
            → f₅: Synthesize findings
            → f₆: Quality check (Loop: refine if quality < threshold)
      → f₇: Format final output
```

---

### 1.2.2 Core Components of a Chain

Each prompt chain, regardless of topology, is constructed from a set of **reusable architectural components**. These components form the **runtime infrastructure** that manages data flow, error handling, and observability. Each component has a well-defined interface, a specific responsibility, and known failure modes.

---

#### Input Preprocessor

The **input preprocessor** is the entry point of the chain, responsible for transforming raw user input into a validated, normalized representation suitable for the first prompt step.

**Responsibilities:**

| Function | Description | Example |
|----------|-------------|---------|
| **Input validation** | Verify input conforms to expected type, format, and constraints | Reject empty strings, inputs exceeding length limits |
| **Sanitization** | Remove or escape characters that could cause prompt injection or parsing failures | Strip control characters, escape curly braces in template engines |
| **Normalization** | Standardize input format for consistent downstream processing | Lowercase, Unicode normalization (NFC/NFD), whitespace collapsing |
| **Enrichment** | Augment input with additional context (metadata, timestamps, user profile) | Attach user's language preference, session ID |
| **Chunking** | Split oversized inputs into processable segments | Split 50,000-token document into 4,000-token chunks with overlap |
| **Type coercion** | Convert input to expected data types | Parse string "42" to integer, parse ISO date strings |

**Formal specification:**

$$
\text{Preprocess}: \mathcal{X}_{\text{raw}} \to \mathcal{X}_{\text{valid}} \cup \{\bot\}
$$

where $\bot$ denotes rejection (invalid input). The preprocessor is a **total function** — it must handle every possible raw input, either transforming it to a valid representation or explicitly rejecting it.

**Chunking with overlap.** For inputs exceeding context limits, the chunking function splits text into overlapping segments:

$$
\text{chunks}(x, w, s) = \{x[i \cdot s : i \cdot s + w] : i = 0, 1, \dots, \lceil(|x| - w) / s\rceil\}
$$

where $w$ is the chunk window size and $s$ is the stride ($s < w$ ensures overlap of $w - s$ tokens). The overlap ensures no information is lost at chunk boundaries.

---

#### Prompt Template Engine

The **prompt template engine** dynamically constructs prompt strings by injecting variable content into parameterized templates. This is the mechanism that connects chain steps — upstream outputs are injected into downstream templates.

**Template structure:**

```
SYSTEM: {system_instructions}

CONTEXT:
{injected_upstream_output}

TASK:
{task_specific_instructions}

OUTPUT FORMAT:
{format_specification}

INPUT:
{current_step_input}
```

**Formal definition.** A prompt template is a function:

$$
\tau: \mathcal{S}_1 \times \mathcal{S}_2 \times \cdots \times \mathcal{S}_m \to \Sigma^*
$$

mapping $m$ input slots (each from its own domain $\mathcal{S}_j$) to a string. The template is a **higher-order function** — it produces a prompt string that, when fed to an LLM, produces an output.

**Variable injection mechanisms:**

| Mechanism | Syntax Example | Capability |
|-----------|---------------|------------|
| **String interpolation** | `f"Analyze: {text}"` | Simple variable substitution |
| **Template language** (Jinja2) | `{% for item in list %}...{% endfor %}` | Loops, conditionals, filters |
| **Schema-driven** | Define input variables with types and validation | Type-safe, self-documenting |
| **Few-shot injection** | Dynamically select and insert relevant examples | Adaptive in-context learning |

**Template composition.** Templates can be composed to build complex prompts from reusable parts:

$$
\tau_{\text{composed}} = \tau_{\text{system}} \oplus \tau_{\text{context}} \oplus \tau_{\text{task}} \oplus \tau_{\text{format}}
$$

where $\oplus$ denotes string concatenation with appropriate delimiters.

**Critical design principle — separation of concerns within templates:**

1. **System instructions**: model persona, global constraints (should be stable across inputs).
2. **Context injection**: upstream outputs and retrieved information (varies per execution).
3. **Task specification**: what the model should do in this step (should be specific and unambiguous).
4. **Format specification**: how the output should be structured (JSON schema, markdown, etc.).
5. **Input data**: the current step's specific input (varies per execution).

This separation enables independent optimization of each template section.

---

#### LLM Invocation Layer

The **LLM invocation layer** abstracts the details of calling language model APIs, providing a unified interface regardless of the underlying model provider.

**Abstraction interface:**

$$
\text{invoke}: (\text{prompt}: \Sigma^*,\; \theta: \Theta) \to \Sigma^*
$$

where $\theta$ encapsulates all decoding parameters.

**Key decoding parameters ($\theta$):**

| Parameter | Symbol | Range | Effect |
|-----------|--------|-------|--------|
| Temperature | $\tau$ | $[0, 2]$ | Controls sampling randomness: $P(x_i) \propto \exp(z_i / \tau)$ |
| Top-$p$ (nucleus sampling) | $p$ | $(0, 1]$ | Sample from smallest token set with cumulative probability $\geq p$ |
| Top-$k$ | $k$ | $\mathbb{Z}^+$ | Sample from top-$k$ highest-probability tokens |
| Max tokens | $L_{\max}$ | $\mathbb{Z}^+$ | Maximum output length |
| Stop sequences | $\{s_j\}$ | Set of strings | Terminate generation upon producing any $s_j$ |
| Frequency penalty | $\alpha_f$ | $[-2, 2]$ | Penalize repeated tokens: $z_i \leftarrow z_i - \alpha_f \cdot \text{count}(i)$ |
| Presence penalty | $\alpha_p$ | $[-2, 2]$ | Penalize any repeated token: $z_i \leftarrow z_i - \alpha_p \cdot \mathbb{1}[\text{count}(i) > 0]$ |

**Per-step model selection.** Different chain steps may benefit from different models:

| Step Type | Recommended Model Characteristics |
|-----------|----------------------------------|
| Factual extraction | High accuracy, low temperature ($\tau \approx 0$) |
| Creative generation | Moderate temperature ($\tau \approx 0.7$–$1.0$), strong generation model |
| Classification / routing | Fast, cheap model; structured output mode |
| Complex reasoning | Most capable model (e.g., GPT-4, Claude 3.5 Sonnet) |
| Code generation | Code-specialized model (e.g., Codex, DeepSeek-Coder) |
| Formatting / templating | Cheapest available model; simple instruction following |

**Retry and timeout logic:**

$$
\text{invoke\_robust}(\text{prompt}, \theta, k_{\max}, t_{\text{timeout}}) = \begin{cases}
\text{response} & \text{if successful within } t_{\text{timeout}} \\
\text{retry up to } k_{\max} \text{ times with exponential backoff} \\
\text{fallback}(\text{prompt}) & \text{if all retries fail}
\end{cases}
$$

Exponential backoff with jitter:

$$
t_{\text{wait}}^{(k)} = \min\bigl(b^k + \text{Uniform}(0, j),\; t_{\max}\bigr)
$$

where $b$ is the base (typically 2), $j$ is the jitter range, and $t_{\max}$ is the maximum wait time.

---

#### Output Parser

The **output parser** is arguably the most failure-prone component in a prompt chain. It transforms raw LLM output (unstructured text) into structured data that downstream steps can reliably consume.

**Parsing hierarchy (from most to least robust):**

| Method | Description | Robustness | Flexibility |
|--------|-------------|------------|-------------|
| **Constrained decoding** | Grammar-constrained generation (guaranteed valid JSON/XML) | Very high | Low (requires model support) |
| **Function calling / tool use** | Model outputs structured function call parameters | High | Moderate |
| **JSON mode** | Model generates JSON; API enforces JSON syntax | High | Moderate |
| **Schema-guided parsing** | Parse output against a Pydantic/JSON Schema; retry on failure | Moderate | High |
| **Regex extraction** | Extract specific patterns from free-form text | Moderate | Moderate |
| **Delimiter-based splitting** | Split output on known delimiters (`---`, `###`, etc.) | Low | High |
| **LLM-based parsing** | Use another LLM call to extract structured data from free-form output | Low–Moderate | Very High |

**Formal specification:**

$$
\phi: \Sigma^* \to \mathcal{Y} \cup \{\text{ParseError}\}
$$

The parser is a **partial function** on raw strings — some strings cannot be parsed (malformed JSON, missing fields, type mismatches). The chain must handle $\text{ParseError}$ gracefully.

**Parser with retry pattern:**

```python
def parse_with_retry(raw_output, parser, llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return parser.parse(raw_output)
        except ParseError as e:
            fix_prompt = f"Fix this output to match the schema:\n{raw_output}\nError: {e}"
            raw_output = llm.invoke(fix_prompt)
    raise ChainFailure("Parser failed after max retries")
```

**Schema validation.** Define the output schema as a type $\mathcal{Y}_i$ for step $i$:

```python
class StepOutput(BaseModel):
    summary: str
    key_points: List[str]
    confidence: float  # 0.0 to 1.0
    category: Literal["positive", "negative", "neutral"]
```

The parser validates both **syntactic correctness** (valid JSON) and **semantic correctness** (field types, value ranges, required fields).

---

#### State Manager

The **state manager** maintains the accumulated context across chain steps, providing each step with access to relevant upstream outputs and global chain metadata.

**State representation.** The chain state at step $i$ is a structured object:

$$
\mathcal{S}_i = \bigl(x,\; y_1, y_2, \dots, y_{i-1},\; \text{metadata}_i\bigr)
$$

where $\text{metadata}_i$ includes execution timestamps, model versions, retry counts, and any user-provided context.

**State propagation strategies:**

| Strategy | Description | Context Efficiency | Information Preservation |
|----------|-------------|-------------------|------------------------|
| **Full history** | Forward all previous outputs to every step | Low (context bloat) | Maximum |
| **Last-$k$ window** | Forward only the last $k$ outputs | Moderate | Moderate |
| **Selective extraction** | Forward only specific fields from specific steps | High | Depends on selection |
| **Summarized history** | Periodically summarize accumulated state | High | Lossy |
| **Key-value store** | Store outputs in a dictionary; steps query by key | High | Full (addressable) |

**Key-value state store** (most flexible pattern):

```python
class ChainState:
    store: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any):
        self.store[key] = value
    
    def get(self, key: str) -> Any:
        return self.store[key]
    
    def get_subset(self, keys: List[str]) -> Dict[str, Any]:
        return {k: self.store[k] for k in keys}
```

Each step declares its **read dependencies** (which state keys it needs) and **write effects** (which state keys it produces). This enables:

1. **Dependency analysis**: determine which steps can execute in parallel.
2. **Minimal context injection**: each step receives only the state keys it needs.
3. **State versioning**: maintain history of state changes for debugging.

---

#### Router / Dispatcher

The **router** implements conditional branching logic, directing execution flow based on the evaluation of intermediate outputs.

**Router architecture:**

$$
\text{Router}: \mathcal{Y}_r \to \{1, 2, \dots, k\}
$$

mapping the output of the preceding step to a branch index.

**Implementation patterns:**

**1. Rule-based router** (deterministic, fast, limited):

```python
def route(output: StepOutput) -> int:
    if output.confidence > 0.9:
        return 1  # High-confidence fast path
    elif output.category == "negative":
        return 2  # Negative sentiment handling
    else:
        return 3  # Default detailed analysis
```

**2. LLM-based router** (flexible, stochastic, more expensive):

```
Given the following analysis result:
{output}

Which of the following actions is most appropriate?
A) Generate a brief summary (simple case)
B) Perform detailed multi-source analysis (complex case)  
C) Escalate to human review (ambiguous case)

Respond with only the letter.
```

**3. Semantic router** (embedding-based, deterministic, requires pre-defined routes):

$$
\text{route}(y) = \arg\max_{j \in \{1, \dots, k\}} \; \text{sim}\bigl(\text{embed}(y),\; \mathbf{c}_j\bigr)
$$

where $\mathbf{c}_j$ is the embedding centroid for route $j$ and $\text{sim}$ is cosine similarity.

**Routing with confidence calibration.** If the router produces a confidence distribution $P(\text{route} = j \mid y)$ over branches, we can implement **uncertain routing**:

$$
\text{route}(y) = \begin{cases}
\arg\max_j P(j \mid y) & \text{if } \max_j P(j \mid y) > \tau_{\text{conf}} \\
\text{fallback route} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{conf}}$ is a confidence threshold. Inputs with uncertain classification are routed to a safe default (e.g., human review, more thorough analysis).

---

#### Aggregator

The **aggregator** combines outputs from multiple upstream branches into a single unified representation for downstream consumption. It is the computational core of fan-in nodes.

**Aggregation functions ($f_{\text{agg}}$):**

| Type | Formal Definition | Use Case |
|------|-------------------|----------|
| **Concatenation** | $f_{\text{agg}}(y_1, \dots, y_k) = y_1 \oplus \cdots \oplus y_k$ | Outputs are complementary sections |
| **Union** | $f_{\text{agg}} = \bigcup_j \text{set}(y_j)$ | Outputs are sets of items (entities, facts) |
| **Majority vote** | $f_{\text{agg}} = \text{mode}(y_1, \dots, y_k)$ | Ensemble classification |
| **Weighted average** | $f_{\text{agg}} = \sum_j w_j y_j$ (for numeric outputs) | Calibrated ensemble |
| **LLM synthesis** | $f_{\text{agg}} = \text{LLM}(\text{``Synthesize:''} \oplus y_1 \oplus \cdots \oplus y_k)$ | Outputs require intelligent merging |
| **Conflict resolution** | $f_{\text{agg}} = \text{LLM}(\text{``Resolve contradictions:''} \oplus \cdots)$ | Outputs may contradict each other |

**Formal properties of aggregation:**

For well-designed aggregators, the following properties are desirable:

1. **Completeness**: no upstream information is lost unless explicitly intended.

$$
\forall j,\; I(y_j; f_{\text{agg}}(y_1, \dots, y_k)) > 0
$$

2. **Commutativity** (order-independence): the aggregation result should not depend on the arbitrary ordering of upstream outputs:

$$
f_{\text{agg}}(y_1, y_2, \dots, y_k) = f_{\text{agg}}(y_{\sigma(1)}, y_{\sigma(2)}, \dots, y_{\sigma(k)})
$$

for any permutation $\sigma$. This is naturally satisfied by set-based operations but requires care with concatenation (where order affects the LLM's attention patterns).

3. **Idempotency** (for duplicate-tolerant aggregation):

$$
f_{\text{agg}}(y_1, y_1) = f_{\text{agg}}(y_1)
$$

---

#### Error Handler

The **error handler** manages failures at any point in the chain, implementing retry, fallback, and graceful degradation logic.

**Error taxonomy for prompt chains:**

| Error Category | Examples | Typical Handling |
|---------------|----------|-----------------|
| **API errors** | Rate limiting (429), timeout, server error (500) | Exponential backoff retry |
| **Parse errors** | Malformed JSON, missing fields, type errors | Retry with fix instructions |
| **Content errors** | Hallucinated facts, off-topic response, refusal | Re-prompt with clarification |
| **Quality errors** | Output below quality threshold | Iterative refinement or model upgrade |
| **Safety errors** | Content policy violation, toxic output | Fallback to safe default |
| **Budget errors** | Cost or token limit exceeded | Truncate, summarize, or abort |

**Error handling strategy hierarchy:**

$$
\text{Handle}(e) = \begin{cases}
\text{Retry}(f_i, k_{\max}) & \text{if } e \in \text{TransientErrors} \\
\text{Fix-and-Retry}(f_i, e) & \text{if } e \in \text{ParseErrors} \\
\text{Fallback}(f_i^{\text{alt}}) & \text{if } e \in \text{PersistentErrors} \\
\text{GracefulDegrade}() & \text{if all retries exhausted} \\
\text{Abort}(e) & \text{if } e \in \text{CriticalErrors}
\end{cases}
$$

**Circuit breaker pattern.** If a step fails repeatedly, the circuit breaker trips, preventing further attempts and routing to a fallback:

$$
\text{CircuitBreaker}(f, k_{\text{trip}}, t_{\text{reset}}) = \begin{cases}
f(x) & \text{if failures} < k_{\text{trip}} \\
\text{fallback}(x) & \text{if failures} \geq k_{\text{trip}} \text{ and } t < t_{\text{reset}} \\
f(x) \text{ (probe)} & \text{if } t \geq t_{\text{reset}}
\end{cases}
$$

---

#### Logger / Tracer

The **logger/tracer** provides observability into chain execution, enabling debugging, performance monitoring, and auditing.

**Trace record per step:**

```json
{
    "step_id": "step_3",
    "step_name": "sentiment_analysis",
    "timestamp_start": "2024-01-15T10:23:45.123Z",
    "timestamp_end": "2024-01-15T10:23:47.456Z",
    "duration_ms": 2333,
    "model": "gpt-4o",
    "prompt_tokens": 1250,
    "completion_tokens": 340,
    "cost_usd": 0.0193,
    "temperature": 0.0,
    "input_hash": "a3f2c1...",
    "output_hash": "b7d4e9...",
    "input_preview": "Analyze the sentiment...",
    "output_preview": "{\"sentiment\": \"positive\"...}",
    "parse_success": true,
    "retry_count": 0,
    "parent_step_ids": ["step_2"],
    "error": null
}
```

**Tracing enables:**

| Capability | Implementation |
|-----------|---------------|
| **Latency profiling** | Identify slowest steps via duration fields |
| **Cost attribution** | Sum costs per step, per chain, per user |
| **Error diagnosis** | Inspect input/output at the exact failure point |
| **Regression detection** | Compare traces across code versions |
| **Caching** | Cache outputs keyed by input hash for identical future inputs |
| **Replay** | Re-execute the chain from any step using logged inputs |

**Distributed tracing.** For chains spanning multiple services or models, use distributed tracing standards (OpenTelemetry) with **trace IDs** propagated across steps. Each step is a **span** within the overall trace:

$$
\text{Trace} = \text{Span}_1 \to \text{Span}_2 \to \cdots \to \text{Span}_n
$$

---

### 1.2.3 Chain Composition Operators

Chain composition operators are the **algebraic primitives** from which arbitrarily complex chain topologies are constructed. They provide a formal, composable language for chain design, analogous to how combinators in functional programming build complex programs from simple building blocks.

---

#### Sequential Composition: $g \circ f$

**Definition.** The output of $f$ is fed as input to $g$:

$$
(g \circ f)(x) = g(f(x))
$$

**Type constraint:**

$$
f: \mathcal{X} \to \mathcal{Y}, \quad g: \mathcal{Y} \to \mathcal{Z} \quad \Rightarrow \quad g \circ f: \mathcal{X} \to \mathcal{Z}
$$

The output type of $f$ must match the input type of $g$. A **type mismatch** (e.g., $f$ outputs free-form text but $g$ expects JSON) causes a runtime error. The output parser $\phi_f$ and the transformation function $g_g$ together enforce this type bridge:

$$
g \circ f = g \circ \phi_f \circ \mathcal{M}_f \circ \tau_f
$$

**Properties:**

- **Associative**: $(h \circ g) \circ f = h \circ (g \circ f)$
- **Not commutative**: $g \circ f \neq f \circ g$ in general
- **Identity element**: $\text{id}(x) = x$ satisfies $f \circ \text{id} = \text{id} \circ f = f$
- These properties make the set of chain steps a **monoid** under sequential composition (a category in category-theoretic terms)

**Latency**: $T_{g \circ f} = T_f + T_g$ (strictly additive).

**Error propagation**: $P(\text{success}) = P(f \text{ succeeds}) \cdot P(g \text{ succeeds} \mid f \text{ succeeded})$.

---

#### Parallel Composition: $(f \| g)(x) = (f(x), g(x))$

**Definition.** Both $f$ and $g$ receive the same input $x$ and produce outputs independently:

$$
(f \| g)(x) = \bigl(f(x),\; g(x)\bigr)
$$

**Generalized to $k$ parallel steps:**

$$
(f_1 \| f_2 \| \cdots \| f_k)(x) = \bigl(f_1(x), f_2(x), \dots, f_k(x)\bigr)
$$

**Type signature:**

$$
f: \mathcal{X} \to \mathcal{Y}_1, \quad g: \mathcal{X} \to \mathcal{Y}_2 \quad \Rightarrow \quad (f \| g): \mathcal{X} \to \mathcal{Y}_1 \times \mathcal{Y}_2
$$

**Properties:**

- **Commutative** (up to output ordering): $(f \| g)(x) \cong (g \| f)(x)$ when the output tuple is treated as a set.
- **Associative**: $(f \| g) \| h \cong f \| (g \| h) \cong f \| g \| h$ (flat tuple).
- **Latency**: $T_{f \| g} = \max(T_f, T_g)$ (wall-clock time determined by slowest branch).
- **Cost**: $C_{f \| g} = C_f + C_g$ (total token cost is additive regardless of parallelism).

**Interaction with sequential composition (distribution law):**

$$
h \circ (f \| g) \neq (h \circ f) \| (h \circ g)
$$

The left side applies $h$ to the **tuple** $(f(x), g(x))$; the right side applies $h$ independently to each output. These are fundamentally different operations. The left side is the standard **diamond pattern** (fan-out then fan-in); the right side is **independent parallel chains** with no aggregation.

---

#### Conditional Composition

**Definition.** A conditional composition selects between alternative steps based on a predicate:

$$
\text{Cond}(c, f, g)(x) = \begin{cases} f(x) & \text{if } c(x) = \text{true} \\ g(x) & \text{otherwise} \end{cases}
$$

**Generalized multi-way conditional:**

$$
\text{Switch}(c, \{f_j\}_{j=1}^k, f_{\text{default}})(x) = \begin{cases}
f_j(x) & \text{if } c(x) = j \text{ for some } j \in \{1, \dots, k\} \\
f_{\text{default}}(x) & \text{otherwise}
\end{cases}
$$

**Type constraint.** All branches must produce outputs of compatible types (or a union type) so that downstream steps can process the result uniformly:

$$
f_j: \mathcal{X} \to \mathcal{Y}_j \quad \Rightarrow \quad \text{Switch}(\cdots): \mathcal{X} \to \bigcup_{j=1}^k \mathcal{Y}_j \cup \mathcal{Y}_{\text{default}}
$$

If $\mathcal{Y}_j$ differ across branches, the downstream step must handle this polymorphism explicitly — either through a union type parser or through branch-specific downstream chains.

**Expected latency:**

$$
\mathbb{E}[T_{\text{cond}}] = T_c + \sum_{j=1}^k P(c(x) = j) \cdot T_{f_j}
$$

where $T_c$ is the evaluation cost of the condition $c$ (negligible for rule-based, non-trivial for LLM-based routing).

---

#### Iterative Composition: $f^{(k)}(x) = f(f^{(k-1)}(x))$

**Definition.** Repeated application of the same function:

$$
f^{(0)}(x) = x, \quad f^{(k)}(x) = f(f^{(k-1)}(x)) \quad \text{for } k \geq 1
$$

**With stopping criterion:**

$$
\text{Iterate}(f, \text{done})(x) = f^{(t^*)}(x), \quad t^* = \min\{k : \text{done}(f^{(k)}(x))\}
$$

**Variant: iteration with memory (non-Markovian).** Each iteration receives not just the previous output but the full iteration history:

$$
y^{(k)} = f\bigl(y^{(k-1)}, y^{(k-2)}, \dots, y^{(0)}, x\bigr)
$$

This prevents the model from repeating the same edits across iterations. Implementation requires accumulating history in the prompt:

```
Original: {x}
Draft 1: {y⁰}
Feedback 1: {critique of y⁰}
Draft 2: {y¹}
Feedback 2: {critique of y¹}
...
Task: Write the next improved draft.
```

**Context growth concern.** Each iteration adds $\sim |y^{(k)}|$ tokens to the prompt. After $t$ iterations:

$$
|\text{prompt}^{(t)}| \approx |\tau| + |x| + \sum_{k=0}^{t-1} |y^{(k)}|
$$

This grows linearly and can exceed the context window. Mitigation: summarize iteration history, or include only the most recent $w$ iterations (sliding window).

---

#### Map-Reduce Composition

**Definition.** The **map** phase applies a function to each element of a collection independently. The **reduce** phase aggregates the results.

$$
\text{MapReduce}(f_{\text{map}}, f_{\text{reduce}})(x_1, x_2, \dots, x_n) = f_{\text{reduce}}\bigl(f_{\text{map}}(x_1), f_{\text{map}}(x_2), \dots, f_{\text{map}}(x_n)\bigr)
$$

**Type signatures:**

$$
f_{\text{map}}: \mathcal{X} \to \mathcal{Y}, \quad f_{\text{reduce}}: \mathcal{Y}^n \to \mathcal{Z}
$$

**Parallelism.** The map phase is **embarrassingly parallel** — all $n$ map invocations are independent and can execute concurrently:

$$
T_{\text{map}} = \max_i T_{f_{\text{map}}}(x_i), \quad T_{\text{total}} = T_{\text{map}} + T_{f_{\text{reduce}}}
$$

**Hierarchical reduce.** When $n$ is large and the reduce function's context window cannot accommodate all map outputs simultaneously, use a **tree-structured reduce**:

$$
\text{TreeReduce}(f_{\text{reduce}}, [y_1, \dots, y_n], b) = \begin{cases}
y_1 & \text{if } n = 1 \\
f_{\text{reduce}}\bigl(\text{TreeReduce}(f_r, [y_1, \dots, y_{\lfloor n/b \rfloor}], b),\; \dots\bigr) & \text{otherwise}
\end{cases}
$$

where $b$ is the branching factor (number of outputs aggregated per reduce step). A binary tree ($b = 2$) requires $\lceil \log_2 n \rceil$ reduce levels.

**Example — Large document summarization:**

```
Document (50,000 tokens)
    → Split into 10 chunks of 5,000 tokens
    → Map: Summarize each chunk (parallel, 10 LLM calls)
    → Reduce: Synthesize 10 summaries into one (1 LLM call)
```

If 10 chunk summaries are too long for one reduce call, use hierarchical reduce:

```
Level 0: 10 chunk summaries
Level 1: Reduce [1-3], [4-6], [7-10] → 3 partial summaries
Level 2: Reduce [partial_1, partial_2, partial_3] → final summary
```

**Map-reduce with custom combiner.** Insert a **combiner** step between map and reduce to perform local aggregation:

$$
\text{MapCombineReduce}(f_m, f_c, f_r)(\mathbf{x}) = f_r\Bigl(f_c\bigl(\{f_m(x_i)\}_{i \in G_1}\bigr),\; f_c\bigl(\{f_m(x_i)\}_{i \in G_2}\bigr),\; \dots\Bigr)
$$

where $\{G_j\}$ is a partition of $\{1, \dots, n\}$. This reduces the load on the final reduce step.

---

#### Passthrough Composition

**Definition.** The original input (or any upstream output) is forwarded alongside the current step's output to downstream steps, bypassing the information bottleneck.

$$
\text{Passthrough}(f)(x) = \bigl(x,\; f(x)\bigr)
$$

More generally, a passthrough **skip connection** forwards the output of step $i$ directly to step $j > i + 1$, bypassing intermediate steps:

$$
y_j = f_j(y_{j-1}, y_i) \quad \text{where } i < j - 1
$$

**Purpose.** Passthrough mitigates the **information loss** from the Data Processing Inequality. Without passthrough, information can only degrade as it flows through the chain. With passthrough, later steps can access the original input directly:

$$
I(x; y_j^{\text{with passthrough}}) \geq I(x; y_j^{\text{without passthrough}})
$$

**Common passthrough patterns:**

| Pattern | Description |
|---------|-------------|
| **Input passthrough** | Original user input forwarded to every step |
| **Metadata passthrough** | Chain metadata (user ID, session info) available at every step |
| **Intermediate passthrough** | Specific upstream outputs forwarded to non-adjacent downstream steps |
| **Residual connection** | Step output is appended to (not replaces) the accumulated context — analogous to residual connections in neural networks |

**Formal residual composition:**

$$
\text{Residual}(f)(x) = x \oplus f(x)
$$

where $\oplus$ denotes concatenation (for text) or merge (for structured data). The downstream step receives both the original and the transformation:

$$
y_{i+1} = f_{i+1}\bigl(y_{i-1} \oplus f_i(y_{i-1})\bigr)
$$

This directly mirrors the **skip connection** in ResNets:

$$
h_{l+1} = h_l + F(h_l, W_l)
$$

The analogy is deep: both prevent information degradation in deep compositional structures.

---

### 1.2.4 Design Principles

The following design principles govern the construction of robust, maintainable, and high-performance prompt chains. Each principle is derived from software engineering best practices, adapted to the unique characteristics of LLM-based computation (stochasticity, natural language interfaces, high per-step cost).

---

#### Single Responsibility Principle per Chain Step

**Principle.** Each chain step should perform **exactly one** well-defined cognitive operation. A step that performs multiple operations conflates error surfaces and reduces debuggability.

**Formal statement.** For step $f_i$ with input $y_{i-1}$ and output $y_i$, the mapping should be **unidimensional**: it should transform the input along a **single semantic axis**.

| Violation | Correction |
|-----------|------------|
| "Analyze the data and format as a table" | Step A: Analyze → Step B: Format |
| "Extract entities and classify sentiment" | Step A: Extract entities → Step B: Classify sentiment |
| "Translate, summarize, and evaluate quality" | Step A: Translate → Step B: Summarize → Step C: Evaluate |

**Why this matters for LLMs specifically:**

1. **Instruction interference.** When a prompt contains multiple instructions, the LLM may prioritize one at the expense of others. Empirical studies show that compliance with later instructions decreases as the number of instructions increases.

2. **Error attribution.** If a step performs both extraction and formatting, and the output is malformatted, it is ambiguous whether the extraction logic or the formatting logic failed.

3. **Optimization independence.** A single-responsibility step can be independently optimized (prompt refined, model swapped, temperature tuned) without affecting other operations.

**Granularity calibration.** The single responsibility principle can be applied too aggressively, creating an excessive number of trivially simple steps. The optimal granularity balances:

$$
\text{Granularity}^* = \arg\min_g \bigl[\text{PerStepComplexity}(g) + \text{IntegrationOverhead}(g) + \text{LatencyCost}(g)\bigr]
$$

A practical heuristic: a step should be **atomic** — it should not be possible to meaningfully decompose it further without the sub-steps being trivial or requiring more context than they produce.

---

#### Minimal Information Transfer Between Steps

**Principle.** Each step should receive and forward only the **minimum information** required for its computation and for downstream steps. Unnecessary information wastes context window capacity and introduces noise.

**Formal statement.** For each edge $(v_i, v_j)$ in the chain DAG, the information transferred should satisfy:

$$
I(y_i^{\text{forwarded}}; Y_{\text{final}}) \approx I(y_i^{\text{full}}; Y_{\text{final}})
$$

while minimizing:

$$
|y_i^{\text{forwarded}}| \ll |y_i^{\text{full}}|
$$

This is the information bottleneck principle operationalized as a design guideline.

**Implementation strategies:**

1. **Schema-constrained output.** Define output schemas that include only downstream-relevant fields:

```python
# Step 1 full output has many fields
# Only forward what Step 2 needs
class Step1OutputForwarded(BaseModel):
    key_entities: List[str]      # needed by Step 2
    primary_topic: str           # needed by Step 2
    # word_count, processing_time, raw_text — NOT forwarded
```

2. **Explicit extraction.** Rather than forwarding the full raw response, extract and forward specific fields:

```python
step1_result = llm(prompt1)
forwarded = {
    "entities": step1_result.entities,
    "topic": step1_result.topic
}
step2_input = template2.format(**forwarded)
```

3. **Summarization as compression.** If a step produces verbose output but downstream steps need only the gist, insert a summarization step as a lossy compressor.

---

#### Explicit Input/Output Contracts (Schemas)

**Principle.** Every chain step should have a formally specified **input schema** and **output schema**. These schemas serve as contracts between steps, enabling type checking, validation, and documentation.

**Contract specification:**

$$
\text{Contract}(f_i) = \bigl(\mathcal{Y}_{i-1}^{\text{schema}},\; \mathcal{Y}_i^{\text{schema}}\bigr)
$$

where $\mathcal{Y}_i^{\text{schema}}$ defines the structure, types, and constraints of step $i$'s output.

**Schema elements:**

| Element | Description | Example |
|---------|-------------|---------|
| **Field name** | Semantic identifier | `summary`, `entities`, `score` |
| **Field type** | Data type | `str`, `List[str]`, `float`, `bool` |
| **Constraints** | Value restrictions | `score >= 0.0 and score <= 1.0` |
| **Required vs. optional** | Whether the field must be present | `summary: required`, `metadata: optional` |
| **Enum values** | Allowed categorical values | `category: Literal["A", "B", "C"]` |
| **Description** | Human-readable field explanation | `"A 2-3 sentence summary of the key finding"` |

**Pydantic-based contract example:**

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class AnalysisInput(BaseModel):
    """Input contract for the analysis step."""
    document_text: str = Field(..., min_length=10, max_length=50000)
    analysis_type: Literal["sentiment", "topic", "entity"]

class AnalysisOutput(BaseModel):
    """Output contract for the analysis step."""
    result: str = Field(..., description="Analysis result")
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(..., min_items=1)
```

**Contract enforcement.** At each step boundary, the orchestrator validates:

$$
\text{Valid}(y_i) = \begin{cases}
\text{true} & \text{if } y_i \in \mathcal{Y}_i^{\text{schema}} \\
\text{ParseError} & \text{otherwise}
\end{cases}
$$

Invalid outputs trigger the error handler (retry, fix, or abort).

**Benefits of explicit contracts:**

1. **Early failure detection**: schema violations are caught immediately, not at downstream steps.
2. **Documentation**: schemas serve as self-documenting interfaces.
3. **Testing**: schemas enable property-based testing of individual steps.
4. **Refactoring safety**: changes to a step's internal implementation are safe as long as the contract is preserved.

---

#### Idempotency of Individual Chain Steps

**Principle.** Each chain step should be **idempotent**: executing it multiple times with the same input should produce the same output (or at least outputs of equivalent quality and structure).

$$
f_i(x) \equiv f_i(x) \quad \forall x \in \mathcal{Y}_{i-1}
$$

**Why idempotency matters:**

1. **Safe retries.** If a step fails due to a transient error (network timeout, rate limit), the orchestrator can safely retry without worrying about side effects.
2. **Caching.** Idempotent steps can be cached: if the same input is seen again, the cached output is returned without re-executing the step.
3. **Debugging.** Idempotent steps can be re-executed during debugging to reproduce behavior.

**Challenges to idempotency in LLM-based steps:**

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| **Stochastic sampling** | Temperature > 0 introduces randomness | Set temperature = 0 for deterministic steps; use seed parameter if available |
| **API non-determinism** | Even with temperature = 0, some APIs have minor non-determinism due to batching | Accept approximate idempotency; compare outputs by semantic similarity rather than exact match |
| **State-dependent prompts** | If the prompt includes timestamps, random IDs, etc. | Remove non-deterministic elements from prompts |
| **External data dependencies** | If the prompt retrieves from a live database or API | Snapshot external data; use versioned retrieval |

**Formal definition (approximate idempotency):**

$$
d\bigl(f_i(x),\; f_i'(x)\bigr) < \epsilon_{\text{idem}} \quad \text{for independent invocations } f_i, f_i'
$$

where $d$ is an appropriate distance metric (exact match, edit distance, or semantic similarity) and $\epsilon_{\text{idem}}$ is a tolerance threshold.

---

#### Determinism vs. Stochasticity Management per Step

**Principle.** Each step should have an explicit, justified choice of determinism level, controlled through decoding parameters.

**Determinism spectrum:**

$$
\underbrace{\tau = 0}_{\text{greedy (deterministic)}} \longleftrightarrow \underbrace{\tau = 1.0}_{\text{standard sampling}} \longleftrightarrow \underbrace{\tau = 2.0}_{\text{high randomness}}
$$

**Guidelines for parameter selection:**

| Step Type | Temperature | Top-$p$ | Rationale |
|-----------|------------|---------|-----------|
| Factual extraction | $0$ | $1.0$ | Correctness is paramount; no room for creativity |
| Classification / routing | $0$ | $1.0$ | Deterministic decisions for reproducibility |
| Summarization | $0$–$0.3$ | $0.9$ | Slight variation acceptable; faithfulness critical |
| Creative writing | $0.7$–$1.0$ | $0.95$ | Diversity and originality desired |
| Brainstorming | $1.0$–$1.5$ | $0.98$ | Maximum diversity; quantity over quality |
| Code generation | $0$–$0.2$ | $0.95$ | Functional correctness required |

**Per-step stochasticity as a design parameter.** In the chain parameter tuple $\theta_i$, the temperature $\tau_i$ is a **first-class design decision**, not a default to be left unchanged:

$$
\theta_i = (\mathcal{M}_i, \tau_i, p_i, k_i, L_{\max,i}, \{s_j\}_i)
$$

**Variance propagation through chains.** The variance of the final output is a function of per-step variances:

$$
\text{Var}[y_n] = h\bigl(\text{Var}[f_1],\; \text{Var}[f_2],\; \dots,\; \text{Var}[f_n]\bigr)
$$

In the worst case (no variance reduction between steps), the variance accumulates:

$$
\text{Var}[y_n] \geq \max_i \text{Var}[f_i]
$$

A single high-variance step can dominate the overall output variance. Best practice: **use the minimum temperature needed at each step** to achieve the desired output characteristics.

---

#### Graceful Degradation Design

**Principle.** When a chain step fails or produces suboptimal output, the chain should **degrade gracefully** rather than fail catastrophically — producing a partial, lower-quality result rather than no result.

**Degradation hierarchy:**

$$
\text{Full Quality} \;\xrightarrow{\text{step failure}}\; \text{Reduced Quality} \;\xrightarrow{\text{more failures}}\; \text{Partial Result} \;\xrightarrow{\text{critical failure}}\; \text{Informative Error}
$$

**Implementation patterns:**

| Pattern | Description | Example |
|---------|-------------|---------|
| **Fallback model** | If primary model fails, use backup model | GPT-4 fails → fall back to GPT-3.5 |
| **Fallback prompt** | If complex prompt fails, use simpler prompt | Detailed analysis fails → simple summary |
| **Skip optional steps** | Non-critical steps are skipped on failure | Formatting step fails → return unformatted output |
| **Cached result** | Return previously cached result for similar input | Exact match → return cache; similar → return with disclaimer |
| **Human escalation** | Route to human reviewer when automated chain fails | All retries exhausted → create human review ticket |
| **Partial output assembly** | Assemble available partial results from completed steps | Steps 1-3 completed, step 4 failed → return partial analysis |

**Formal degradation function:**

$$
y = \begin{cases}
f_n \circ \cdots \circ f_1(x) & \text{if all steps succeed (full quality)} \\
f_n^{\text{fallback}} \circ \cdots \circ f_1(x) & \text{if step } n \text{ primary fails} \\
\text{partial}(y_1, \dots, y_{k}) & \text{if steps } k+1 \text{ through } n \text{ fail} \\
\text{error\_msg}(e) & \text{if step 1 fails (no partial result possible)}
\end{cases}
$$

---

#### Separation of Reasoning Steps from Action Steps

**Principle.** Steps that **reason** (analyze, plan, evaluate, decide) should be architecturally separated from steps that **act** (generate final output, call external APIs, modify databases, send emails).

**Motivation:**

1. **Safety.** Reasoning steps can be freely retried, inspected, and overridden without side effects. Action steps have real-world consequences and must be executed carefully.
2. **Reversibility.** Reasoning steps are inherently reversible (just discard the output). Action steps may be irreversible (sent email cannot be unsent).
3. **Human oversight.** A human-in-the-loop gate is naturally placed between the reasoning phase and the action phase.

**Two-phase chain architecture:**

```
Phase 1 (Reasoning): x → Analyze → Plan → Validate → Decision
                                                          │
                                              [Human approval gate]
                                                          │
Phase 2 (Action):                               Execute → Format → Deliver → y
```

**Formal separation:**

$$
f_{\text{chain}} = f_{\text{act}} \circ \text{Gate} \circ f_{\text{reason}}
$$

where $\text{Gate}$ is a synchronization/approval mechanism:

$$
\text{Gate}(y_{\text{reason}}) = \begin{cases}
y_{\text{reason}} & \text{if approved (automatically or by human)} \\
\text{modify}(y_{\text{reason}}) & \text{if human edits the plan} \\
\bot & \text{if rejected (chain aborts)}
\end{cases}
$$

**Side-effect classification:**

| Step Type | Side Effects | Safety Level |
|-----------|-------------|-------------|
| **Pure reasoning** | None (read-only) | Safe to retry, cache, discard |
| **Retrieval** | None (read from external sources) | Safe to retry; may incur cost |
| **Generation** | None (produces text) | Safe to retry; may incur cost |
| **API call** | External state change | Requires idempotency or confirmation |
| **Database write** | Persistent state change | Requires transaction safety |
| **User communication** | Irreversible (email, notification) | Requires explicit approval |

Steps with side effects should be:

1. **Positioned at the end** of the chain (after all reasoning is complete).
2. **Guarded by approval gates** (human or automated validation).
3. **Wrapped in transaction logic** (rollback on failure).
4. **Logged with full audit trail** (who approved, what was executed, when).

---

## Summary Table: Section 1.2 Key Constructs

| Construct | Mathematical Form | Key Property |
|-----------|------------------|-------------|
| Sequential composition | $g \circ f$ | Latency additive: $T_{g \circ f} = T_f + T_g$ |
| Parallel composition | $(f \| g)(x) = (f(x), g(x))$ | Latency: $\max(T_f, T_g)$ |
| Conditional composition | $\text{Cond}(c, f, g)$ | Only one branch executes |
| Iterative composition | $f^{(k)}(x)$ until done | Requires termination guarantee |
| Map-reduce | $f_r(f_m(x_1), \dots, f_m(x_n))$ | Map phase embarrassingly parallel |
| Passthrough | $(x, f(x))$ | Mitigates DPI information loss |
| Diamond topology | $f_{\text{agg}}((f_1 \| \cdots \| f_k)(f_0(x)))$ | Most common non-trivial topology |
| Hierarchical chain | $f_i = g_{i,m} \circ \cdots \circ g_{i,1}$ | Encapsulation and reusability |
| Error propagation | $P_{\text{success}} = \prod_i p_i$ | Per-step accuracy must be high |
| Amdahl's law for chains | $S(k) = 1/(s + (1-s)/k)$ | Sequential fraction limits speedup |