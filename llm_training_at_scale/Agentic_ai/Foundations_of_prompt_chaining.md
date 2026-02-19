

# Chapter 1: Prompt Chaining — Comprehensive Treatment

---

## 1.1 Foundations of Prompt Chaining

---

### 1.1.1 Definition and Formal Framework

---

#### What is Prompt Chaining

Prompt Chaining is a **structured paradigm for decomposing a complex reasoning or generation task into an ordered sequence of subtasks**, where each subtask is handled by a distinct prompt submitted to a language model (or possibly different models), and the output of each preceding prompt is programmatically transformed and injected into the input of the subsequent prompt. The chain terminates when the final prompt produces the desired end-to-end output.

At its core, prompt chaining operationalizes a principle borrowed from software engineering and mathematical analysis: **decompose an intractable monolithic computation into a composition of simpler, individually tractable computations**. Rather than relying on a single prompt to carry the full burden of reasoning, retrieval, formatting, and generation simultaneously, prompt chaining distributes these responsibilities across discrete, auditable stages.

A concrete instantiation: consider generating a research literature review. A single prompt asking an LLM to "write a comprehensive literature review on diffusion models" conflates at least five distinct cognitive operations:

1. **Topic scoping** — determining which sub-areas to cover.
2. **Source identification** — recalling or retrieving relevant papers.
3. **Claim extraction** — identifying key findings per paper.
4. **Synthesis** — identifying thematic connections and contradictions.
5. **Prose generation** — producing coherent academic prose.

Prompt chaining separates these into five discrete prompts, each receiving the output of the previous stage as structured input, thereby reducing per-step cognitive load on the model, enabling intermediate human or programmatic inspection, and dramatically improving output quality.

---

#### Formal Definition: Chain as a Directed Acyclic Graph (DAG) of Prompts

A prompt chain is formally defined as a **Directed Acyclic Graph (DAG)** $G = (V, E)$ where:

- $V = \{v_1, v_2, \dots, v_n\}$ is a finite set of **prompt nodes**, each representing a distinct prompt-response computation.
- $E \subseteq V \times V$ is a set of **directed edges**, where $(v_i, v_j) \in E$ indicates that the output of node $v_i$ is (possibly after transformation) consumed as part of the input to node $v_j$.
- The graph is **acyclic**: there exists no sequence $v_{i_1}, v_{i_2}, \dots, v_{i_k}$ such that $(v_{i_j}, v_{i_{j+1}}) \in E$ for all $j$ and $v_{i_1} = v_{i_k}$.

Each node $v_i$ encapsulates:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Prompt template | $\tau_i$ | A parameterized string with placeholders for upstream outputs |
| Model invocation | $\mathcal{M}_i$ | The specific LLM (or model variant) executing this step |
| Decoding parameters | $\theta_i$ | Temperature, top-$p$, max tokens, stop sequences, etc. |
| Output parser | $\phi_i$ | A deterministic function extracting structured data from raw model output |
| Transformation | $g_i$ | A deterministic function mapping parsed outputs from parent nodes into the prompt template |

The **linear chain** is the simplest DAG topology: a path graph $v_1 \to v_2 \to \cdots \to v_n$. However, the DAG formulation generalizes to:

- **Fan-out** (parallel branching): one node feeds multiple downstream nodes.
- **Fan-in** (aggregation): multiple upstream nodes feed one downstream node.
- **Diamond patterns**: fan-out followed by fan-in, enabling parallel processing of subtasks with subsequent aggregation.

The acyclicity constraint is critical: it distinguishes prompt chaining from **agentic loops** (which permit cycles for iterative refinement). If cycles are introduced, the system transitions from a chain to a **state machine** or **agent architecture** with fundamentally different termination and convergence properties.

---

#### Mathematical Formulation

For the **linear chain** case, the end-to-end computation is expressed as a function composition:

$$
P_{\text{chain}}(x) = f_n\bigl(f_{n-1}\bigl(\dots f_2\bigl(f_1(x, \theta_1), \theta_2\bigr)\dots, \theta_{n-1}\bigr), \theta_n\bigr)
$$

where:

- $x \in \mathcal{X}$ is the initial user input.
- $f_i: \mathcal{Y}_{i-1} \times \Theta_i \to \mathcal{Y}_i$ represents the computation at node $v_i$, mapping the output space of the preceding node and decoding parameters to the output space of the current node.
- $\theta_i \in \Theta_i$ denotes the full parameter and configuration tuple for step $i$.
- $\mathcal{Y}_0 \triangleq \mathcal{X}$ (the input space), and $\mathcal{Y}_n$ is the final output space.

More precisely, each $f_i$ decomposes into three sub-operations:

$$
f_i(y_{i-1}, \theta_i) = \phi_i\Bigl(\mathcal{M}_i\bigl(\tau_i\bigl(g_i(y_{i-1})\bigr),\; \theta_i^{\text{decode}}\bigr)\Bigr)
$$

where:

- $g_i: \mathcal{Y}_{i-1} \to \mathcal{S}_i$ is the **transformation function** that converts upstream output into the format expected by the template.
- $\tau_i: \mathcal{S}_i \to \Sigma^*$ is the **template instantiation function** that produces the final prompt string (where $\Sigma^*$ denotes the set of all finite strings over vocabulary $\Sigma$).
- $\mathcal{M}_i: \Sigma^* \times \Theta_i^{\text{decode}} \to \Sigma^*$ is the **model invocation** that takes a prompt string and decoding parameters and returns a response string.
- $\phi_i: \Sigma^* \to \mathcal{Y}_i$ is the **output parser** that extracts structured information from the raw response.

For the **general DAG** case, let $\text{Pa}(v_i)$ denote the parent set of node $v_i$. Then:

$$
y_i = f_i\Bigl(\{y_j : v_j \in \text{Pa}(v_i)\},\; \theta_i\Bigr)
$$

The full chain output is the collection of outputs from all **sink nodes** (nodes with no outgoing edges):

$$
P_{\text{chain}}(x) = \{y_i : v_i \in \text{Sinks}(G)\}
$$

A topological sort of $G$ determines a valid execution order. If $G$ has width $w$ (maximum antichain size), then up to $w$ nodes can execute in parallel, giving a **critical path length** (chain latency) equal to the longest path in $G$.

---

#### Distinction from Single-Shot Prompting

| Dimension | Single-Shot Prompting | Prompt Chaining |
|-----------|----------------------|-----------------|
| **Structure** | One prompt $\to$ one response | $n$ prompts $\to$ $n$ responses, composed |
| **Computation** | $y = f(x, \theta)$ | $y = f_n \circ f_{n-1} \circ \cdots \circ f_1(x)$ |
| **Error surface** | Monolithic — failure at any reasoning step corrupts entire output | Decomposed — failure is isolated to a specific step |
| **Context utilization** | Entire context window consumed by one task | Each step uses context efficiently for a focused subtask |
| **Debuggability** | Black-box: only input and output are observable | Glass-box: every intermediate output $y_i$ is inspectable |
| **Controllability** | Limited to prompt engineering within a single prompt | Each step can have distinct instructions, temperature, model |
| **Latency** | Single API call | $n$ sequential API calls (or fewer with parallelism) |
| **Cost** | Single inference cost | $\sum_{i=1}^n \text{cost}(f_i)$ — cumulative token costs |

The fundamental tradeoff: single-shot prompting minimizes latency and cost but maximizes per-prompt cognitive complexity. Prompt chaining inverts this tradeoff, accepting higher latency and cost in exchange for dramatically improved reliability, controllability, and output quality on complex tasks.

A critical theoretical distinction: single-shot prompting assumes the model can perform the **entire latent computation** within a single forward pass (or a single autoregressive generation sequence). For tasks requiring intermediate reasoning states that exceed the model's implicit computational depth, this assumption fails. Prompt chaining provides **explicit external memory** (intermediate outputs) and **additional computational steps** (multiple inference calls), effectively augmenting the model's computational capacity.

---

#### Distinction from Multi-Turn Conversation

Multi-turn conversation and prompt chaining both involve multiple LLM invocations, but they differ fundamentally:

| Dimension | Multi-Turn Conversation | Prompt Chaining |
|-----------|------------------------|-----------------|
| **Control locus** | User-driven: a human decides what to say next based on the model's response | Program-driven: the orchestration logic determines the next prompt algorithmically |
| **Context management** | Typically appended: full conversation history is sent with each turn | Selective injection: only relevant parsed outputs are forwarded |
| **Goal structure** | Often exploratory, open-ended | Directed toward a predetermined output structure |
| **Intermediate processing** | Minimal — raw conversational text carries forward | Substantial — parsing, validation, transformation between steps |
| **Reproducibility** | Low — depends on human decisions within the loop | High — deterministic orchestration given fixed model outputs |
| **Prompt engineering** | Each turn's prompt is ad hoc, shaped by prior conversation | Each step's prompt template is pre-designed and optimized |

Formally, in multi-turn conversation, the input to turn $t$ is the **concatenation** of the full dialogue history:

$$
x_t = [u_1, r_1, u_2, r_2, \dots, u_{t-1}, r_{t-1}, u_t]
$$

where $u_i$ and $r_i$ are user and response messages respectively. This linear growth in context length leads to context window saturation.

In prompt chaining, the input to step $i$ is a **curated extraction** from predecessor outputs:

$$
x_i = \tau_i\bigl(g_i(\{y_j : v_j \in \text{Pa}(v_i)\})\bigr)
$$

This selective forwarding acts as an **information bottleneck**, discarding irrelevant details and preserving only the structured information needed by the current step. This distinction is not merely implementational — it has profound implications for information flow efficiency and error propagation.

---

#### Distinction from Agentic Loops

Agentic loops (as in ReAct, AutoGPT, or tool-augmented agents) introduce **cycles** into the execution graph, transforming the DAG into a **directed cyclic graph** or a **state machine**:

| Dimension | Prompt Chaining (DAG) | Agentic Loops (Cyclic Graph / State Machine) |
|-----------|----------------------|----------------------------------------------|
| **Graph topology** | Directed Acyclic Graph | Directed graph with cycles |
| **Termination** | Guaranteed: finite DAG has finite topological sort | Not guaranteed: requires explicit termination conditions |
| **Number of steps** | Fixed at design time (or bounded by DAG depth) | Variable: determined at runtime by loop exit conditions |
| **Decision authority** | Orchestrator follows a predetermined plan | The LLM itself often decides whether to continue, which tool to call, or when to terminate |
| **State management** | Implicit: intermediate outputs form the state | Explicit: requires a maintained state/scratchpad |
| **Complexity class** | Bounded computation | Potentially unbounded (Turing-complete with unrestricted looping) |
| **Predictability** | High: execution path is deterministic modulo model stochasticity | Low: execution path is highly variable |

Formally, an agentic loop introduces a transition function with a **self-loop**:

$$
s_{t+1} = \begin{cases} T(s_t, a_t) & \text{if } \neg\text{done}(s_t) \\ s_t & \text{otherwise} \end{cases}
$$

where $a_t = \pi(s_t)$ is the action selected by the LLM policy $\pi$, and the loop continues until the predicate $\text{done}(s_t)$ evaluates to true. The key risk is **non-termination** or **excessive iteration**, neither of which arises in DAG-structured prompt chains.

Prompt chaining occupies a **middle ground** on the autonomy spectrum:

$$
\text{Single-Shot} \;\xrightarrow{\text{decomposition}}\; \text{Prompt Chaining (DAG)} \;\xrightarrow{\text{+ cycles, autonomy}}\; \text{Agentic Loops} \;\xrightarrow{\text{+ multi-agent}}\; \text{Multi-Agent Systems}
$$

---

#### Prompt Chaining as a Computational Graph

Prompt chaining can be rigorously understood through the lens of **computational graph theory**, analogous to how deep learning frameworks (PyTorch, TensorFlow) represent neural network computations.

In a deep learning computational graph:

- **Nodes** represent tensor operations (matrix multiplications, activations, losses).
- **Edges** represent tensor data flow.
- **Backpropagation** flows gradients in reverse through the graph.

In a prompt chain computational graph:

- **Nodes** represent LLM invocations (prompt $\to$ response).
- **Edges** represent structured data flow (parsed outputs $\to$ template instantiation).
- There is **no backpropagation** — the chain is executed in the forward direction only. However, **error signals** can be propagated backward through manual debugging or automated evaluation at each step.

This analogy extends to:

| Deep Learning Graph | Prompt Chain Graph |
|-|-|
| Tensor shapes must be compatible across edges | Output schemas must be compatible with downstream template expectations |
| Activation functions introduce non-linearity | LLM generation introduces stochastic non-linear transformation |
| Batch normalization stabilizes intermediate representations | Output parsing and validation stabilize intermediate data |
| Skip connections allow information to bypass layers | Direct edges from early nodes to later nodes bypass intermediate steps |
| Graph optimization (operator fusion, memory planning) | Chain optimization (parallelization, caching, prompt compression) |

The computational graph perspective enables formal reasoning about:

- **Data dependencies**: which steps can execute in parallel (independent subgraphs).
- **Critical path**: the longest sequential dependency chain, determining minimum latency.
- **Redundancy**: identifying steps whose outputs are unused or duplicated.
- **Fault tolerance**: inserting validation/retry nodes at critical junctions.

---

#### Prompt Chaining as Function Composition

From the perspective of functional programming and mathematical analysis, prompt chaining is **function composition** over the space of natural language (or structured data representations).

Let $\mathcal{F} = \{f_1, f_2, \dots, f_n\}$ be a set of **prompt functions**, where each $f_i: \mathcal{Y}_{i-1} \to \mathcal{Y}_i$ maps from one representational space to another. The chain is:

$$
P_{\text{chain}} = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1
$$

This composition satisfies several properties from category theory and functional analysis:

**1. Associativity.** Function composition is associative:

$$
(f_n \circ f_{n-1}) \circ f_{n-2} = f_n \circ (f_{n-1} \circ f_{n-2})
$$

This means we can freely group adjacent chain steps into **macro-steps** without altering the overall computation — a property exploited when merging steps for efficiency or splitting steps for debuggability.

**2. Non-commutativity.** In general, $f_i \circ f_j \neq f_j \circ f_i$. The order of prompt execution matters critically. Summarizing before extracting key facts yields different results than extracting facts before summarizing.

**3. Stochasticity.** Unlike deterministic mathematical functions, each $f_i$ is **stochastic** due to LLM sampling:

$$
f_i(y_{i-1}) \sim P_{\mathcal{M}_i}(\cdot \mid \tau_i(g_i(y_{i-1})),\; \theta_i^{\text{decode}})
$$

This means $P_{\text{chain}}$ defines a **distribution over outputs**, not a single deterministic output. The variance of the final output is a function of the variances at each step and their interactions:

$$
\text{Var}[P_{\text{chain}}(x)] = h\bigl(\text{Var}[f_1], \text{Var}[f_2], \dots, \text{Var}[f_n], \text{covariance structure}\bigr)
$$

In practice, setting temperature $= 0$ (greedy decoding) at each step makes $f_i$ approximately deterministic (modulo numerical precision and batching effects), making the chain reproducible.

**4. Type signatures.** Each $f_i$ has a type signature $\mathcal{Y}_{i-1} \to \mathcal{Y}_i$. Type mismatches (e.g., step $i$ outputs free-form text when step $i+1$ expects JSON) cause **chain failures**. Robust chains enforce type contracts through output parsing and validation — analogous to type checking in programming languages.

**5. Partial application and currying.** A prompt template with fixed instructions but a variable input slot is a **partially applied function**: the instructions are "curried in," and the upstream output is the remaining argument. Formally:

$$
f_i = \tau_i(\text{instructions}_i, \cdot)
$$

where $\text{instructions}_i$ are fixed at design time and the input slot $\cdot$ is filled at runtime.

---

### 1.1.2 Motivation and Rationale

---

#### Why Single Prompts Fail on Complex Tasks

Single-shot prompting fails systematically on complex tasks due to multiple interacting factors, each grounded in the computational and statistical properties of autoregressive language models.

**1. Finite computational depth per token.** A transformer with $L$ layers performs $L$ sequential computational steps per generated token. Each layer applies a fixed-width computation (multi-head attention + feed-forward network). For a model with $L$ layers and hidden dimension $d$, the computational capacity per token is bounded by $O(L \cdot d^2)$ operations. Tasks requiring deeper reasoning chains than $L$ steps cannot be faithfully executed within a single forward pass. Chain-of-thought prompting partially addresses this by converting serial depth into sequential token generation, but the per-token computational budget remains fixed.

**2. Attention dilution.** In a single long prompt, the attention mechanism must distribute its capacity across all input tokens. For a prompt of length $N$, each attention head in a standard transformer computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

As $N$ grows, the softmax distribution becomes increasingly diffuse (especially without strong positional signals), reducing the model's ability to focus on the most relevant information. This phenomenon, empirically documented as the "lost in the middle" effect, means that critical information embedded deep within a long prompt receives diminished attention weight.

**3. Instruction interference.** When a single prompt contains multiple complex instructions (e.g., "analyze the data, then critique the methodology, then suggest improvements, then format as a table"), the model must maintain a **multi-objective focus** throughout generation. Empirically, later instructions receive lower compliance rates — the model's autoregressive generation tends to "forget" or deprioritize instructions that appear far from the current generation point.

**4. Error compounding within generation.** In autoregressive generation, each token conditions on all previously generated tokens. An error early in the generation (e.g., a factual mistake in the analysis section) propagates and contaminates all subsequent text. There is **no mechanism for the model to backtrack** within a single generation — it cannot "undo" an early mistake. Prompt chaining provides explicit checkpoints where errors can be detected and corrected before propagating.

**5. Format and content coupling.** A single prompt must simultaneously specify **what** to generate (content requirements) and **how** to format it (structural requirements). These objectives can conflict: producing analytically deep content may be at odds with fitting it into a rigid tabular format. Separating content generation from formatting into distinct chain steps eliminates this interference.

**6. Context window as a hard constraint.** For a model with context window size $C$ tokens, a single-shot prompt must fit the entire task description, all relevant context, and leave room for the full response within $C$ tokens. For complex tasks with substantial input data, this constraint is easily violated. Prompt chaining allows each step to operate within a manageable fraction of the context window.

---

#### Cognitive Load Decomposition Principle

The motivation for prompt chaining is grounded in a principle directly analogous to **cognitive load theory** in human instruction design (Sweller, 1988) and **modular decomposition** in software engineering.

**Cognitive load theory** posits that working memory has limited capacity, and learning/performance degrades when intrinsic load (task complexity), extraneous load (poorly designed instructions), and germane load (schema construction) exceed this capacity. The solution is to decompose complex tasks into manageable chunks.

For LLMs, the analogous "working memory" is the **context window** and the **per-token computational budget**. The analogous decomposition is prompt chaining:

$$
\text{Total Task Complexity} = \sum_{i=1}^{n} \text{Subtask Complexity}_i + \text{Integration Overhead}
$$

The key insight: while total complexity is conserved (or slightly increased due to integration overhead), the **peak per-step complexity** is dramatically reduced:

$$
\max_i \text{Subtask Complexity}_i \ll \text{Total Task Complexity}
$$

This ensures each step operates well within the model's effective capacity, yielding higher per-step accuracy and, by composition, higher end-to-end accuracy.

Formally, if the probability of a single step succeeding is $p_i$ for step $i$, and steps are approximately independent conditioned on correct upstream outputs, then:

$$
P(\text{chain succeeds}) = \prod_{i=1}^n p_i
$$

For this product to exceed the success probability of a single monolithic prompt $p_{\text{mono}}$, we need:

$$
\prod_{i=1}^n p_i > p_{\text{mono}}
$$

Empirically, for complex tasks, individual step success probabilities $p_i$ can be driven very close to 1.0 through focused prompting, while $p_{\text{mono}}$ may be substantially below 1.0. For example, if $n = 5$ and each $p_i = 0.95$, then $\prod p_i = 0.774$. If $p_{\text{mono}} = 0.40$ for the same task, chaining is clearly superior. The design challenge is ensuring each $p_i$ is sufficiently high and managing integration overhead.

---

#### Context Window Limitations and Mitigation

The context window $C$ of a language model imposes a **hard constraint** on the total number of tokens (input + output) processable in a single invocation. Current models range from $C = 4{,}096$ (older models) to $C = 128{,}000$ or beyond (GPT-4, Claude), but even large context windows become insufficient for complex tasks involving:

- Large input documents (legal contracts, codebases, research corpora).
- Multi-step reasoning requiring extensive scratchpad space.
- Complex output formats with many sections.

Prompt chaining mitigates context window limitations through several mechanisms:

**1. Selective forwarding.** Only the **parsed, compressed output** of each step is forwarded, not the entire raw response. If step $i$ generates $L_i$ tokens but the essential information is captured in $k_i \ll L_i$ tokens of structured output, the context savings are:

$$
\text{Savings}_i = L_i - k_i
$$

Accumulated over $n$ steps, the total context consumed at step $j$ is:

$$
\text{Context}_j = |\tau_j| + \sum_{i \in \text{Pa}(v_j)} k_i
$$

rather than the monolithic $|x| + \sum_{i=1}^{j-1} L_i$ that would be required in a multi-turn conversation approach.

**2. Map-reduce patterns.** For large inputs exceeding $C$, the input is split into chunks, each chunk is processed independently (map phase), and results are aggregated (reduce phase). This is naturally expressed as a fan-out/fan-in DAG:

```
        ┌─── f_map(chunk_1) ───┐
x ──────┼─── f_map(chunk_2) ───┼─── f_reduce ──→ y
        └─── f_map(chunk_3) ───┘
```

**3. Hierarchical summarization.** For extremely long documents, recursive chaining builds summaries of summaries:

$$
y^{(0)}_j = \text{chunk}_j, \quad y^{(\ell)}_j = f_{\text{summarize}}\bigl(\{y^{(\ell-1)}_k : k \in \text{children}(j)\}\bigr)
$$

continuing until a single root summary $y^{(L)}_1$ is obtained.

---

#### Error Isolation and Localization

One of the most compelling practical advantages of prompt chaining is **error isolation**: when the final output is incorrect, the intermediate outputs provide a **diagnostic trace** enabling precise localization of the failure point.

In a single-shot prompt, a flawed output provides minimal diagnostic information — the user sees only the final result and must guess where the reasoning went wrong. In a chain with $n$ steps, the intermediate outputs $y_1, y_2, \dots, y_n$ are all observable. The error localization procedure is:

$$
i^* = \min\{i : y_i \text{ is incorrect given correct } y_{i-1}\}
$$

This identifies the first step where the model failed, enabling targeted prompt refinement at step $i^*$ without modifying the rest of the chain.

Error isolation enables:

- **Targeted retry**: re-execute only the failed step (and downstream dependents), saving compute.
- **Step-specific prompt optimization**: refine the prompt template $\tau_{i^*}$ independently.
- **Model swapping**: use a more capable (and expensive) model only for the step that fails most frequently.
- **Automated validation**: insert validation nodes after high-risk steps to catch errors before propagation.

Formally, define the **error propagation function** $\epsilon_i$ as the probability that an error at step $i$ causes an error in the final output:

$$
\epsilon_i = P(\text{final output incorrect} \mid \text{step } i \text{ incorrect}, \text{all } j < i \text{ correct})
$$

In a linear chain without error correction, $\epsilon_i = 1$ for all $i$ (any error propagates to the end). With validation and retry at each step, $\epsilon_i$ is reduced to:

$$
\epsilon_i^{\text{corrected}} = (1 - p_{\text{detect},i})
$$

where $p_{\text{detect},i}$ is the probability of detecting the error at step $i$ through validation.

---

#### Controllability and Debuggability

Prompt chaining transforms LLM-based computation from an opaque, monolithic process into a **transparent, modular pipeline** with fine-grained control points.

**Controllability** manifests across multiple dimensions:

| Control Point | Mechanism |
|--------------|-----------|
| **Per-step model selection** | Use GPT-4 for reasoning steps, GPT-3.5 for formatting steps |
| **Per-step temperature** | Low temperature ($\tau \approx 0$) for factual extraction, higher temperature for creative brainstorming |
| **Per-step output schema** | Enforce JSON schema at step 3, free-form text at step 5 |
| **Conditional branching** | Route to different downstream prompts based on parsed output (e.g., sentiment-dependent branches) |
| **Human-in-the-loop** | Insert human review/approval gates between steps |
| **Retry policies** | Retry a specific step up to $k$ times if output validation fails |

**Debuggability** is achieved through:

1. **Intermediate output logging**: every $y_i$ is stored, creating a full execution trace.
2. **Step-level evaluation**: each step can be independently evaluated against ground truth.
3. **Counterfactual analysis**: modify $y_i$ manually and observe the effect on downstream steps.
4. **Ablation studies**: remove or replace individual steps to assess their contribution.
5. **Latency profiling**: measure per-step latency to identify bottlenecks.

---

#### Latent Space Traversal Across Chain Steps

A deep and underappreciated perspective on prompt chaining involves understanding each step as a **traversal through the model's latent representational space**.

A single LLM invocation maps input text to a trajectory through the model's internal representation space. The initial layers encode low-level linguistic features; middle layers encode semantic relationships; final layers encode task-specific output representations. The generated output text is a **projection** of this internal trajectory back into token space.

When the output of step $i$ is fed as input to step $i+1$, the following occurs:

1. The internal representation from step $i$ is **projected** to text via the output embedding and autoregressive decoding.
2. This text is **re-encoded** by the input embedding and transformer layers of the model in step $i+1$.
3. The re-encoded representation occupies a **different region** of the latent space, conditioned on the new prompt template $\tau_{i+1}$.

This process can be understood as a sequence of **encode-decode-re-encode** operations:

$$
z_i^{\text{internal}} \xrightarrow{\text{decode}} y_i^{\text{text}} \xrightarrow{\text{encode}} z_{i+1}^{\text{internal}}
$$

Each text bottleneck acts as a **regularizer** — it forces the chain to communicate only through human-interpretable text, preventing the accumulation of pathological internal states. This is analogous to the **information bottleneck** in representation learning: the text interface constrains the mutual information between consecutive steps, forcing each step to transmit only the most salient information.

However, this bottleneck also represents an **information loss**: nuances captured in the internal representation $z_i^{\text{internal}}$ that are not expressible in text $y_i^{\text{text}}$ are permanently lost. This motivates research into **latent-space chaining** (passing hidden states directly between model invocations), which preserves richer information at the cost of interpretability.

---

#### Task Complexity vs. Chain Length Tradeoffs

The relationship between task complexity and optimal chain length involves several competing forces:

**1. Accuracy improvement with length.** As chain length $n$ increases (more granular decomposition), per-step task complexity decreases, and per-step accuracy $p_i$ increases. However, the total accuracy is:

$$
P_{\text{success}} = \prod_{i=1}^n p_i(n)
$$

where $p_i(n)$ is the per-step success probability given $n$ total steps. As $n$ increases, $p_i(n) \to 1$, but the number of multiplicative terms also increases.

**2. Diminishing returns.** There exists an optimal chain length $n^*$ that maximizes $P_{\text{success}}$. Beyond $n^*$, the overhead of additional steps (integration errors, parsing failures, unnecessary decomposition of already-simple subtasks) outweighs the accuracy gains from further decomposition.

$$
n^* = \arg\max_n \prod_{i=1}^n p_i(n)
$$

**3. Latency scaling.** For a linear chain, latency scales as:

$$
T_{\text{chain}} = \sum_{i=1}^n T_i
$$

where $T_i$ is the latency of step $i$. For a DAG with critical path length $d$ and maximum parallelism $w$:

$$
T_{\text{chain}} \geq \sum_{j \in \text{critical path}} T_j
$$

**4. Cost scaling.** Total cost (in tokens) scales as:

$$
C_{\text{chain}} = \sum_{i=1}^n \bigl(|\text{input}_i| \cdot c_{\text{in}} + |\text{output}_i| \cdot c_{\text{out}}\bigr)
$$

where $c_{\text{in}}$ and $c_{\text{out}}$ are per-token costs for input and output respectively. Note that forwarding intermediate outputs adds to input costs of downstream steps.

**5. Empirical heuristic.** In practice, 3–7 steps suffice for most complex tasks. Chains beyond 10 steps often indicate over-decomposition or a need for hierarchical sub-chains rather than flat sequential chains.

The optimal chain length is task-dependent and can be determined through:

- **Ablation studies**: systematically varying $n$ and measuring end-to-end accuracy.
- **Error analysis**: examining where the chain fails to determine if more or fewer steps would help.
- **Complexity estimation**: assessing the intrinsic dimensionality of the task space.

---

### 1.1.3 Theoretical Underpinnings

---

#### Compositionality in Language Models

**Compositionality** — the principle that the meaning of a complex expression is determined by the meanings of its parts and the rules combining them — is a foundational property of natural language (Frege, 1892; Montague, 1970) and a critical theoretical underpinning for prompt chaining.

In formal semantics, compositionality is expressed as:

$$
\llbracket \alpha \cdot \beta \rrbracket = F(\llbracket \alpha \rrbracket, \llbracket \beta \rrbracket)
$$

where $\llbracket \cdot \rrbracket$ denotes semantic interpretation and $F$ is a composition function.

For prompt chaining, the relevant question is: **do LLMs exhibit compositional generalization?** That is, if a model can successfully execute subtask $A$ and subtask $B$ individually, can it also successfully execute their composition $B \circ A$?

Empirical evidence is mixed:

- **Positive**: LLMs demonstrate strong compositional abilities in many practical settings — they can follow multi-step instructions, chain logical deductions, and combine skills (e.g., translate, then summarize).
- **Negative**: Systematic failures occur with out-of-distribution compositions (e.g., novel combinations of familiar skills), deep recursive structures, and compositional generalization benchmarks (COGS, SCAN).

Prompt chaining **exploits** compositionality while **mitigating** its failures:

- **Exploitation**: each step assumes the model can perform a single, well-defined subtask (leveraging the model's compositional capabilities at the subtask level).
- **Mitigation**: by decomposing the full task into subtasks, the chain avoids requiring the model to perform **novel, deep compositions** within a single prompt — the composition is handled externally by the orchestration logic.

This is analogous to the distinction between **compiled** and **interpreted** composition: single-shot prompting requires the model to "compile" the entire composition internally, while prompt chaining "interprets" the composition step-by-step with external orchestration.

---

#### Chain as a Markov Decision Process (MDP)

A prompt chain can be formally modeled as a **Markov Decision Process** (MDP), providing a rigorous framework for analyzing sequential decision-making within the chain.

An MDP is defined as the tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$:

| Component | Chain Interpretation |
|-----------|---------------------|
| $\mathcal{S}$ — State space | The set of all possible intermediate outputs: $S_t = y_t$ (the parsed output after step $t$) |
| $\mathcal{A}$ — Action space | The set of possible prompt templates, model choices, and decoding parameters for the next step |
| $T: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ — Transition function | The stochastic mapping from current intermediate output and chosen action to next intermediate output: $S_{t+1} \sim T(\cdot \mid S_t, A_t)$ |
| $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ — Reward function | Quality metric evaluating the output of each step (e.g., relevance, correctness, format compliance) |
| $\gamma \in [0, 1]$ — Discount factor | Weighting of future step quality relative to current step quality |

The transition dynamics are:

$$
S_{t+1} = T(S_t, A_t) = \phi_{t+1}\bigl(\mathcal{M}_{t+1}(\tau_{t+1}(g_{t+1}(S_t)), \theta_{t+1}^{\text{decode}})\bigr)
$$

**Markov property**: the critical assumption is that $S_{t+1}$ depends only on $S_t$ and $A_t$, not on the full history $S_0, S_1, \dots, S_{t-1}$. This holds if and only if the parsed output $y_t$ contains **all information** from previous steps that is needed by subsequent steps.

In practice, the Markov property holds when:

- The output parser $\phi_t$ extracts all relevant information.
- The prompt template $\tau_{t+1}$ does not reference outputs from steps before $t$.
- There are no skip connections in the DAG (i.e., the chain is strictly sequential).

When skip connections exist (edges from step $i$ directly to step $j > i+1$), the process is **non-Markovian** in the sequential sense but remains Markovian with respect to the full parent set (by the DAG structure). Formally, the **DAG-factored** Markov property holds:

$$
P(S_i \mid \text{all predecessors}) = P(S_i \mid \{S_j : v_j \in \text{Pa}(v_i)\})
$$

The MDP formulation enables several analytical tools:

- **Policy optimization**: selecting the best action (prompt template, model, parameters) at each step to maximize cumulative reward.
- **Value function estimation**: estimating the expected quality of the final output given the current intermediate state.
- **Planning**: using the MDP structure to plan the optimal chain design before execution.

The **optimal policy** $\pi^*$ for a chain MDP selects actions that maximize the expected cumulative reward:

$$
\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{n-1} \gamma^t R(S_t, A_t) \mid \pi\right]
$$

In static chain design (fixed templates and parameters), the "policy" is determined at design time. In **adaptive chains** (where downstream prompt templates are selected based on upstream outputs), the policy is a function of the runtime state, and MDP optimization techniques become directly applicable.

---

#### Information Bottleneck Theory Applied to Chaining

The **Information Bottleneck (IB) method** (Tishby et al., 1999) provides a powerful theoretical framework for understanding prompt chaining. The IB principle seeks a compressed representation $T$ of input $X$ that maximally preserves information about target $Y$:

$$
\min_{P(T|X)} \; I(X; T) - \beta \cdot I(T; Y)
$$

where $I(\cdot; \cdot)$ denotes mutual information and $\beta > 0$ controls the tradeoff between compression and prediction.

In prompt chaining, each intermediate output $y_i$ serves as a **bottleneck representation** between the upstream computation and downstream computation:

$$
\text{Upstream context} \xrightarrow{\text{compression}} y_i \xrightarrow{\text{expansion}} \text{Downstream computation}
$$

The mapping from the full upstream context (original input + all intermediate computations) to the text output $y_i$ acts as a lossy compression. The downstream step re-expands this compressed representation in the context of a new prompt template.

For each step $i$, the IB tradeoff is:

$$
\min_{\phi_i} \; I(\text{Context}_{<i}; y_i) - \beta_i \cdot I(y_i; Y_{\text{final}})
$$

where:

- $I(\text{Context}_{<i}; y_i)$ measures how much of the upstream context is preserved (compression).
- $I(y_i; Y_{\text{final}})$ measures how much information $y_i$ carries about the final desired output (relevance).

**Optimal intermediate outputs** achieve maximal relevance with minimal redundancy. Overly verbose intermediate outputs ($I(\text{Context}_{<i}; y_i)$ too high) waste context window in downstream steps. Overly compressed intermediate outputs ($I(y_i; Y_{\text{final}})$ too low) lose critical information.

The practical implication: **prompt and parser design at each step should be tuned to extract exactly the information needed by downstream steps — no more, no less.** This is the engineering manifestation of the information bottleneck principle.

**Connection to the Data Processing Inequality (DPI)**: for any Markov chain $X \to Y \to Z$:

$$
I(X; Z) \leq I(X; Y)
$$

Applied to prompt chaining: information about the original input $x$ can only decrease (or remain constant) as it flows through the chain. Each step is a potential information loss point. This places a fundamental limit on chain performance:

$$
I(x; y_n) \leq I(x; y_{n-1}) \leq \cdots \leq I(x; y_1) \leq H(x)
$$

where $H(x)$ is the entropy of the input. To mitigate information loss, chains should:

1. Use **skip connections** (direct edges from the original input to later steps) to bypass the bottleneck.
2. Ensure intermediate outputs are sufficiently rich to preserve all task-relevant information.
3. Avoid unnecessary chain length (each additional step is an additional DPI application).

---

#### Conditional Probability Decomposition

The most direct probabilistic justification for prompt chaining comes from the **chain rule of probability** (also called the product rule). For the final output $Y = (Y_1, Y_2, \dots, Y_n)$ decomposed into $n$ components:

$$
P(Y \mid X) = \prod_{i=1}^{n} P(Y_i \mid Y_{<i}, X)
$$

where $Y_{<i} = (Y_1, \dots, Y_{i-1})$.

This identity is exact and holds for any joint distribution. Prompt chaining operationalizes this decomposition by assigning each conditional factor $P(Y_i \mid Y_{<i}, X)$ to a dedicated prompt step.

The key advantages of this decomposition:

**1. Reduced conditional complexity.** Each factor $P(Y_i \mid Y_{<i}, X)$ conditions on a **smaller, more focused** set of variables compared to the monolithic $P(Y \mid X)$. If the model approximates each factor well, the product approximation is also accurate:

$$
\hat{P}(Y \mid X) = \prod_{i=1}^n \hat{P}(Y_i \mid Y_{<i}, X) \approx P(Y \mid X)
$$

**2. Natural factorization.** Many tasks have a **natural factorization** that aligns with this decomposition. For example, a report generation task factorizes as:

$$
P(\text{Report} \mid \text{Data}) = P(\text{Outline} \mid \text{Data}) \cdot P(\text{Body} \mid \text{Outline}, \text{Data}) \cdot P(\text{Summary} \mid \text{Body}, \text{Data})
$$

**3. Conditional independence simplification.** If some $Y_i$ are conditionally independent of early $Y_j$ given recent intermediate outputs, the conditioning set can be reduced:

$$
P(Y_i \mid Y_{<i}, X) = P(Y_i \mid Y_{i-1}, X) \quad \text{(Markov assumption)}
$$

This Markov simplification reduces the context required at each step and aligns with the DAG structure of the chain.

**4. Bayesian perspective.** Each step can be viewed as a **Bayesian update**: the prior is the current state of knowledge (accumulated from previous steps), the evidence is the original input and any new information, and the posterior is the output of the current step:

$$
P(Y_i \mid Y_{<i}, X) \propto P(X \mid Y_i, Y_{<i}) \cdot P(Y_i \mid Y_{<i})
$$

---

#### Expressiveness of Chained vs. Monolithic Prompts

A fundamental theoretical question: **are chained prompts strictly more expressive than monolithic prompts?**

**Theorem (informal).** For a fixed autoregressive language model $\mathcal{M}$ with context window $C$ and $L$ transformer layers, there exist tasks $\mathcal{T}$ such that:

1. A chain of $n > 1$ prompts to $\mathcal{M}$ can solve $\mathcal{T}$ with high probability.
2. No single prompt to $\mathcal{M}$ (within context window $C$) can solve $\mathcal{T}$ with probability significantly above chance.

**Proof sketch.** Consider a task requiring $k$ intermediate reasoning steps where $k > L$. A single forward pass through $\mathcal{M}$ computes at most $L$ sequential operations per token position. While autoregressive generation can perform additional computation by generating "scratchpad" tokens, this requires the model to have been trained to use scratchpads effectively for this type of reasoning.

A chain provides $n \cdot L$ sequential operations (each step performs $L$ layers of computation over its input), strictly increasing the total computational depth. Furthermore, each step's output serves as an "external scratchpad" that is then re-encoded, providing a fresh $L$-layer processing opportunity.

More formally, consider the circuit complexity perspective. A single-step transformer computation can be modeled as a circuit of depth $O(L)$ and width $O(N \cdot d)$ (where $N$ is sequence length and $d$ is hidden dimension). A chain of $n$ such computations yields a circuit of depth $O(n \cdot L)$ and width $O(\max_i N_i \cdot d)$. By the time hierarchy theorem in circuit complexity, deeper circuits can compute functions that shallower circuits cannot.

**Practical implications:**

- For tasks within the model's single-pass capability, chaining adds overhead without expressiveness gains.
- For tasks exceeding single-pass capability, chaining is **necessary**, not merely helpful.
- The decision to chain should be based on task complexity assessment, not applied blindly.

---

#### Computational Complexity Analysis of Chains

Let us analyze the computational complexity of prompt chain execution along several dimensions.

**Time complexity.** For a linear chain of $n$ steps, where step $i$ has input length $L_i^{\text{in}}$ and output length $L_i^{\text{out}}$, and the model uses standard transformer architecture with context window $C$:

The cost of a single autoregressive generation step with input length $L^{\text{in}}$ and output length $L^{\text{out}}$ is:

$$
T_{\text{step}} = O\bigl(L^{\text{in}} \cdot d^2 \cdot L_{\text{layers}}\bigr) + O\bigl(L^{\text{out}} \cdot (L^{\text{in}} + L^{\text{out}}) \cdot d \cdot L_{\text{layers}}\bigr)
$$

The first term is the **prefill** cost (encoding the input), and the second term is the **decode** cost (generating tokens autoregressively).

Total chain time complexity:

$$
T_{\text{chain}} = \sum_{i=1}^n T_{\text{step},i}
$$

For the critical path in a DAG:

$$
T_{\text{chain}}^{\text{DAG}} = \max_{\text{path } P} \sum_{i \in P} T_{\text{step},i}
$$

**Space complexity.** Each step requires storing the full KV-cache for its context. The KV-cache size for step $i$ is:

$$
\text{KV-cache}_i = 2 \cdot L_{\text{layers}} \cdot L_{\text{heads}} \cdot d_{\text{head}} \cdot (L_i^{\text{in}} + L_i^{\text{out}})
$$

Since steps are executed sequentially (in the linear case), the maximum concurrent memory is:

$$
\text{Memory}_{\text{peak}} = \max_i \text{KV-cache}_i
$$

For DAG execution with parallelism, the peak memory is:

$$
\text{Memory}_{\text{peak}}^{\text{DAG}} = \sum_{i \in \text{max antichain}} \text{KV-cache}_i
$$

**Token cost complexity.** Total tokens consumed (relevant for API billing):

$$
\text{Tokens}_{\text{total}} = \sum_{i=1}^n \bigl(L_i^{\text{in}} + L_i^{\text{out}}\bigr)
$$

Compared to a single-shot prompt with input $L_0^{\text{in}}$ and output $L_0^{\text{out}}$, the chain typically consumes more total tokens due to:

1. Template overhead (instructions at each step).
2. Repeated forwarding of intermediate outputs.
3. Structural overhead (JSON formatting, delimiters).

However, the single-shot prompt may require a much larger $L_0^{\text{out}}$ (e.g., generating scratchpad reasoning inline), partially offsetting this difference.

---

#### Approximation Theory: Chain Depth vs. Approximation Quality

Drawing from **approximation theory** in deep learning, we can analyze how chain depth affects the quality of task approximation.

**Universal approximation in transformers.** A sufficiently wide single-layer transformer can approximate any continuous sequence-to-sequence function (Yun et al., 2020). However, the required width may be exponential in the function's complexity. Deeper transformers achieve the same approximation with polynomial width.

**Analogous result for chains.** A sufficiently long single-shot prompt (with extensive chain-of-thought) can, in principle, approximate any reasoning chain. However, the required context length may be exponential in the reasoning depth. Chaining achieves the same result with polynomial context length per step.

Formally, define the **approximation error** of a chain with $n$ steps for a target function $f^*$:

$$
\epsilon(n) = \mathbb{E}_{x \sim \mathcal{D}} \left[\mathcal{L}\bigl(P_{\text{chain}}^{(n)}(x), f^*(x)\bigr)\right]
$$

where $\mathcal{L}$ is an appropriate loss function and $P_{\text{chain}}^{(n)}$ is the $n$-step chain.

Under reasonable assumptions about the model's per-step capability and the task's inherent decomposability:

$$
\epsilon(n) = O\left(\frac{1}{n^\alpha}\right) + O\left(\delta^n\right)
$$

The first term represents **approximation improvement** with chain depth (the task is better approximated with more steps, with rate $\alpha > 0$ depending on task smoothness). The second term represents **error accumulation** from per-step noise (with per-step error rate $\delta < 1$).

The optimal chain depth $n^*$ balances these terms:

$$
n^* = \arg\min_n \left[\frac{1}{n^\alpha} + \delta^n\right]
$$

For small $\delta$ (high per-step accuracy), $n^*$ is larger (deeper chains are beneficial). For large $\delta$ (low per-step accuracy), $n^*$ is smaller (error accumulation dominates).

---

### 1.1.4 Historical Evolution

---

#### Early Template-Based NLP Pipelines

Prompt chaining is not a novel concept born with large language models — it is the intellectual descendant of decades of **pipeline architectures** in natural language processing.

**Classical NLP pipelines (1990s–2010s)** were explicitly decomposed into sequential modules:

```
Raw Text → Tokenization → POS Tagging → Named Entity Recognition
         → Dependency Parsing → Coreference Resolution → Semantic Role Labeling
         → Task-Specific Module (e.g., Relation Extraction, Sentiment Analysis)
```

Each module was a separate statistical or rule-based model, with the output of each stage serving as features for the next. This architecture embodied the same decomposition principle as modern prompt chaining, but with critical differences:

| Classical NLP Pipeline | Modern Prompt Chain |
|-|-|
| Each module is a separate trained model (HMM, CRF, SVM) | Each step uses the same LLM (or different LLMs) |
| Fixed, hand-designed feature interfaces between modules | Flexible, natural language interfaces between steps |
| Modules are domain-specific and require separate training data | Steps are defined by prompts — no additional training required |
| Error propagation is well-studied but difficult to mitigate | Error propagation is mitigated by the LLM's ability to "heal" from mildly noisy inputs |
| Pipeline construction requires deep NLP expertise | Chain construction requires prompt engineering skill |

The key limitation of classical pipelines was **error cascading**: a POS tagging error would corrupt dependency parsing, which would corrupt relation extraction. This problem motivated **joint models** (performing all tasks simultaneously) and ultimately the development of end-to-end neural models.

Ironically, the pendulum has swung back: end-to-end neural models (single-shot prompting with LLMs) face their own limitations on complex tasks, motivating a return to pipeline architectures — now implemented as prompt chains rather than separate trained models.

**Template-based generation (2000s–2010s)** used explicit templates with slot filling:

```
"Dear {name}, Thank you for your {action} on {date}. We will {response}."
```

These systems were rigid (no reasoning, fixed structure) but highly reliable. Modern prompt chaining preserves the template concept (each step has a prompt template with placeholders) while replacing static slot filling with LLM-powered generation.

---

#### Emergence with GPT-3 and In-Context Learning

The emergence of **in-context learning (ICL)** with GPT-3 (Brown et al., 2020) was the catalytic event that made prompt chaining practical and powerful.

**Pre-GPT-3**, each NLP task required a separately fine-tuned model. Chaining would require training and maintaining multiple fine-tuned models — an expensive and inflexible approach.

**Post-GPT-3**, a single general-purpose model could perform diverse tasks via prompting alone. This meant that a chain of prompts to the same model could implement a complex pipeline without any model training.

Key enablers from GPT-3:

1. **Few-shot learning**: the ability to define a task through examples in the prompt, eliminating the need for fine-tuning at each chain step.
2. **Instruction following**: the ability to follow natural language instructions, enabling flexible prompt templates.
3. **Format compliance**: the ability to output structured formats (JSON, lists, tables) when instructed, enabling reliable parsing between chain steps.
4. **Knowledge breadth**: sufficient world knowledge to handle diverse subtasks without domain-specific fine-tuning.

Early prompt chaining examples (2020–2021) were **ad hoc**: researchers manually constructed chains of API calls with string manipulation between steps. There was no standardized framework, error handling, or abstraction layer.

The **chain-of-thought (CoT) prompting** paradigm (Wei et al., 2022) can be viewed as an **internalized prompt chain**: the model generates intermediate reasoning steps within a single prompt, mimicking the step-by-step decomposition of chaining but within a single context window. Prompt chaining **externalizes** this process, providing explicit control, auditability, and the ability to exceed single-context computational limits.

---

#### LangChain, LlamaIndex, and Framework Evolution

The transition from ad hoc prompt chaining to systematic frameworks occurred rapidly between 2022 and 2024.

**LangChain** (October 2022): the first widely adopted framework explicitly designed for prompt chaining. Key abstractions:

| Abstraction | Purpose |
|-------------|---------|
| `PromptTemplate` | Parameterized prompt with input variables |
| `LLMChain` | A single prompt template + LLM invocation |
| `SequentialChain` | Linear composition of multiple `LLMChain` instances |
| `RouterChain` | Conditional branching based on intermediate output |
| `OutputParser` | Structured extraction from raw LLM output |
| `Memory` | State persistence across chain steps |

LangChain introduced the **LCEL (LangChain Expression Language)**, enabling declarative chain construction through the pipe operator:

```python
chain = prompt_1 | llm | parser_1 | prompt_2 | llm | parser_2
```

This syntax directly mirrors the mathematical function composition $f_2 \circ \text{parse}_1 \circ \mathcal{M} \circ \tau_1$.

**LlamaIndex** (November 2022): focused on **retrieval-augmented chains**, combining document indexing with prompt chaining. The core innovation was treating retrieval as a chain step:

```
Query → Retrieve Relevant Documents → Synthesize Answer
```

This pattern — retrieval as a chain node — became foundational for retrieval-augmented generation (RAG) pipelines.

**Subsequent framework evolution:**

| Framework | Year | Key Innovation |
|-----------|------|----------------|
| **Semantic Kernel** (Microsoft) | 2023 | Plugin architecture, native .NET/Python support |
| **Haystack** (deepset) | 2023 | Pipeline-first design with typed components |
| **DSPy** (Stanford) | 2023 | **Programmatic prompt optimization** — treating chain design as a machine learning problem with learnable prompt parameters |
| **LangGraph** (LangChain) | 2024 | **Graph-based orchestration** with explicit state machines, enabling cycles (agentic loops) as a generalization of DAG chains |
| **CrewAI** | 2024 | Multi-agent chains with role-based decomposition |
| **AutoGen** (Microsoft) | 2024 | Multi-agent conversation frameworks |

**DSPy** represents a particularly important evolution: rather than manually designing prompt templates at each step, DSPy treats the chain as a **differentiable program** where prompt content is optimized through automated search:

$$
\theta^* = \arg\min_\theta \; \mathcal{L}\bigl(P_{\text{chain}}(x; \theta), y^*\bigr)
$$

where $\theta$ includes prompt instructions, few-shot examples, and retrieval queries at each step, and $\mathcal{L}$ is a task-specific loss function.

---

#### From Static Chains to Dynamic/Adaptive Chains

The evolution from static to dynamic chains represents a fundamental shift in chain architecture:

**Static chains** (2020–2022): the chain topology, prompt templates, and model choices are fixed at design time. Every input traverses the same sequence of steps.

```
Input → Step 1 → Step 2 → Step 3 → Output
```

**Conditional chains** (2022–2023): branching logic is introduced, allowing different paths based on intermediate outputs:

```
Input → Step 1 → [if condition A] → Step 2A → Step 3
                  [if condition B] → Step 2B → Step 3
```

**Adaptive chains** (2023–present): the chain structure itself is determined at runtime by the LLM. The model decides which steps to execute, in what order, and when to terminate:

```
Input → Planner (LLM decides steps) → Execute Step 1 → 
        Evaluator (LLM decides next step) → Execute Step 2 → ... → Output
```

This progression can be formalized as increasing **degrees of freedom** in the chain:

| Chain Type | Fixed at Design Time | Determined at Runtime |
|------------|---------------------|-----------------------|
| Static | Topology, templates, models, parameters | Nothing (only the data flowing through the chain varies) |
| Conditional | Templates, models, parameters | Topology (branch selection) |
| Adaptive | Models, parameter ranges | Topology, templates, number of steps |
| Fully Agentic | Model availability | Everything (the agent plans and executes autonomously) |

The tradeoff along this spectrum: **predictability decreases** (harder to guarantee behavior, cost, latency) while **flexibility increases** (able to handle more diverse and unexpected inputs).

Formally, let $\Omega$ be the space of possible chain configurations (topologies, templates, parameters). A static chain selects a single $\omega_0 \in \Omega$ at design time. An adaptive chain defines a **policy** $\pi: \mathcal{S} \to \Omega$ that selects the configuration dynamically based on the current state. A fully agentic system's policy has access to the full state space and action space, with no predetermined constraints on the chain structure.

---

#### Relationship to Classical AI Planning

Prompt chaining has deep intellectual roots in **classical AI planning** (Fikes & Nilsson, 1971; Sacerdoti, 1975), which addresses the problem of finding a sequence of actions to achieve a goal from an initial state.

**STRIPS planning** defines:

- **State**: a set of logical propositions describing the world.
- **Actions**: operations with preconditions and effects.
- **Goal**: a set of propositions to be achieved.
- **Plan**: a sequence of actions transforming the initial state to a goal state.

The mapping to prompt chaining:

| STRIPS Planning | Prompt Chaining |
|-----------------|-----------------|
| State $s_t$ | Intermediate output $y_t$ |
| Action $a_t$ | Prompt step $f_t$ (template + model + parameters) |
| Precondition of $a_t$ | Schema/format expected by $\tau_t$ |
| Effect of $a_t$ | The output $y_t$ added to the knowledge state |
| Goal | Desired final output specification |
| Plan | The chain topology and step sequence |

**Hierarchical Task Network (HTN) planning** (Erol et al., 1994) is particularly relevant: HTN decomposes high-level tasks into subtasks recursively:

```
Write-Report → [Outline, Draft-Body, Write-Conclusion, Format]
Draft-Body → [Write-Introduction, Write-Methods, Write-Results, Write-Discussion]
```

This hierarchical decomposition maps directly to **nested prompt chains**: a high-level chain where individual steps are themselves sub-chains.

**Partial-order planning** (which allows parallel execution of independent actions) maps to the DAG formulation of prompt chains, where nodes without dependency edges can execute in parallel.

Key differences from classical AI planning:

1. **Stochastic actions**: classical planning assumes deterministic action effects; prompt chain steps are stochastic (LLM outputs vary).
2. **Natural language state representation**: classical planning uses formal logic; prompt chains use natural language (and structured data) as state representations.
3. **No explicit search**: classical planners search through the action space to find a valid plan; prompt chains are typically designed by humans (though adaptive chains automate this).
4. **Richer action effects**: a single prompt step can produce complex, nuanced outputs far beyond the simple predicate additions/deletions of STRIPS.

Recent work on **LLM-based planning** (e.g., SayCan, Inner Monologue, LLM+P) combines classical planning algorithms with LLM-generated plans, creating a synthesis where:

- The LLM proposes candidate plans (chain structures).
- A classical planner verifies feasibility and optimality.
- The LLM executes each step of the verified plan.

This hybrid approach addresses the well-documented weakness of LLMs in **systematic planning** (inability to guarantee plan correctness) while leveraging their strength in **natural language understanding and generation** (flexible, knowledge-rich execution of individual steps).

---

## Summary Table: Section 1.1 Key Concepts

| Concept | Core Insight | Mathematical Expression |
|---------|-------------|------------------------|
| Chain as DAG | Prompt chain is a directed acyclic graph of prompt nodes | $G = (V, E)$, $E \subseteq V \times V$, acyclic |
| Function composition | Chain is composition of stochastic functions | $P_{\text{chain}} = f_n \circ \cdots \circ f_1$ |
| MDP formulation | Chain states evolve via Markov transitions | $S_{t+1} = T(S_t, A_t)$ |
| Information bottleneck | Intermediate outputs compress upstream information | $\min I(X;T) - \beta \cdot I(T;Y)$ |
| Probability decomposition | Joint probability factorizes over chain steps | $P(Y \mid X) = \prod_i P(Y_i \mid Y_{<i}, X)$ |
| Data processing inequality | Information cannot increase through chain steps | $I(x; y_n) \leq I(x; y_1)$ |
| Approximation-accumulation tradeoff | Chain depth balances approximation gain vs. error growth | $\epsilon(n) = O(n^{-\alpha}) + O(\delta^n)$ |
| Cognitive load decomposition | Per-step complexity is reduced by decomposition | $\max_i \text{Complexity}_i \ll \text{Total Complexity}$ |
| Success probability | Chain success is product of step successes | $P_{\text{success}} = \prod_{i=1}^n p_i$ |

---