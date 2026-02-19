

# 1.4 State Management and Context Propagation

> **Scope.** In any multi-step LLM chain or agentic system, *state* is the single structure that gives each step awareness of what has happened, what is happening, and what should happen next. Mismanaging state is the dominant root cause of hallucinated tool calls, lost context, duplicated work, and non-deterministic agent behavior. This section provides a first-principles, mathematically grounded treatment of how state is represented, how finite context windows are managed, how memory mechanisms extend state beyond a single forward pass, and how data flows between chain nodes.

---

## 1.4.1 State Representation

### 1.4.1.1 Formal Definition

At any discrete reasoning step $t$ of a chain of length $T$, the **complete agent state** is the tuple:

$$
S_t = \bigl(\, x,\; y_1, y_2, \dots, y_{t-1},\; m_t,\; \mathcal{E}_t \,\bigr)
$$

| Symbol | Semantics |
|---|---|
| $x$ | Original user query / task specification (immutable across the chain) |
| $y_1 \dots y_{t-1}$ | Ordered sequence of all prior step outputs (the *trajectory*) |
| $m_t$ | **Metadata** — step index, timestamps, tool-call results, token counts, error flags, confidence scores |
| $\mathcal{E}_t$ | **External state snapshot** — pointers or cached values from databases, vector stores, file systems |

The LLM at step $t$ produces its output by sampling from the conditional:

$$
y_t \sim p_\theta\!\bigl(\,\cdot \mid \text{Render}(S_t)\,\bigr)
$$

where $\text{Render}: \mathcal{S} \rightarrow \mathcal{V}^{\leq C_{\max}}$ is a **serialization function** that converts the structured state into a token sequence that fits within the model's context window $C_{\max}$. The design of $\text{Render}$ is itself a critical engineering decision (see §1.4.2).

### 1.4.1.2 Five Canonical State Categories

#### A. Explicit State (Structured, Typed, Programmatic)

Explicit state is *deliberately constructed* data passed between chain steps via code-level data structures, **not** via the LLM's text stream. It provides type safety, inspectability, and deterministic propagation.

**Characteristics:**

- Represented as typed objects: Python `dataclass`, `TypedDict`, Pydantic `BaseModel`, Protocol Buffers, JSON Schema-validated dictionaries.
- The **contract** between steps: step $i$ guarantees it will produce an output conforming to schema $\sigma_i$; step $i{+}1$ expects input conforming to $\sigma_i$.
- Enables **compile-time or runtime validation** — if step $i$'s output violates $\sigma_i$, the orchestrator can retry, fall back, or raise before step $i{+}1$ ever runs.

**Formal model.** Define a schema lattice $(\Sigma, \sqsubseteq)$ where $\sigma_a \sqsubseteq \sigma_b$ denotes that schema $\sigma_a$ is a subtype of $\sigma_b$. A chain is **type-safe** iff for every edge $(i, i{+}1)$:

$$
\sigma_i^{\text{out}} \sqsubseteq \sigma_{i+1}^{\text{in}}
$$

This is the **chain compatibility condition** — it prevents silent data-shape mismatches that corrupt downstream reasoning.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ResearchState(BaseModel):
    """Explicit state object passed between chain steps."""
    query: str                                  # immutable original input x
    search_results: List[dict] = Field(default_factory=list)
    extracted_facts: List[str] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    confidence: float = 0.0
    step_index: int = 0
    token_budget_remaining: int = 120000
    timestamps: List[datetime] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)

    def advance(self) -> "ResearchState":
        """Return a copy with incremented step index and timestamp."""
        return self.model_copy(update={
            "step_index": self.step_index + 1,
            "timestamps": self.timestamps + [datetime.utcnow()]
        })
```

**Why this matters:** Every field is named, typed, versioned, and diffable. If `extracted_facts` is empty when `draft_answer` is being generated, the orchestrator can detect this programmatically — no LLM "judgment" required.

#### B. Implicit State (Conversation History / Message List)

Implicit state is the **raw sequence of messages** (system, user, assistant, tool) that the LLM has seen. It is the *default* state mechanism in chat-completion APIs and the most common pattern in simple chains.

**Representation:**

$$
H_t = \bigl[\, \text{msg}_1^{(\text{sys})},\; \text{msg}_2^{(\text{usr})},\; \text{msg}_3^{(\text{ast})},\; \dots,\; \text{msg}_{k}^{(\text{tool})} \,\bigr]
$$

Each message $\text{msg}_j$ has a **role** $r_j \in \{\texttt{system}, \texttt{user}, \texttt{assistant}, \texttt{tool}\}$ and a **content** string $c_j$.

**Critical properties:**

| Property | Implication |
|---|---|
| **Unstructured** | The LLM must *parse* relevant information from raw text — fragile, non-deterministic |
| **Grows linearly** | $|H_t| = \mathcal{O}(t)$ messages, each potentially thousands of tokens |
| **Position-sensitive** | Transformers exhibit primacy/recency bias — information in the middle of $H_t$ is attended to less (the "lost in the middle" phenomenon, Liu et al., 2023) |
| **No type safety** | A step can produce malformed output undetected until a downstream step fails |

**When appropriate:** Implicit state is sufficient for short chains ($T \leq 3$) where the entire history fits comfortably within $C_{\max}$ and no structured validation is needed.

#### C. Compressed State (Summarized / Distilled Context)

When the full trajectory $y_1, \dots, y_{t-1}$ exceeds the token budget, **compressed state** replaces verbatim history with a lossy but informationally dense summary.

**Formal model.** Define a compression operator $\mathcal{C}$:

$$
\mathcal{C}: \mathcal{V}^{n} \rightarrow \mathcal{V}^{m}, \quad m \ll n
$$

such that the **information loss** is bounded:

$$
\mathcal{L}_{\text{compression}} = D_{\text{KL}}\!\Bigl(\, p_\theta(\cdot \mid H_t) \;\Big\|\; p_\theta\bigl(\cdot \mid \mathcal{C}(H_t)\bigr) \,\Bigr) \leq \epsilon
$$

where $\epsilon$ is an acceptable divergence threshold. In practice, $\epsilon$ is not computed exactly but is approximated by evaluating downstream task accuracy with and without compression.

**Compression strategies (ordered by fidelity):**

| Strategy | Mechanism | Token Ratio | Information Loss |
|---|---|---|---|
| **Verbatim truncation** | Keep first/last $k$ messages | Fixed | High (middle lost) |
| **Extractive summarization** | Select key sentences via salience scoring | ~5–10× | Moderate |
| **Abstractive summarization** | LLM generates a summary of prior steps | ~10–50× | Moderate–High |
| **Key-value extraction** | Pull structured facts into a JSON object | ~20–100× | Low for factual chains |
| **Embedding compression** | Encode history into dense vectors (MemoryTransformer, Gisting) | Extreme | Task-dependent |

```python
async def compress_state(
    history: list[dict],
    max_tokens: int = 500,
    llm: LLMClient = None
) -> str:
    """Abstractive compression: distill conversation history."""
    compression_prompt = (
        "You are a precise summarizer. Given the conversation below, "
        "produce a structured summary containing:\n"
        "1. Original user goal\n"
        "2. Key decisions made and their rationale\n"
        "3. Current intermediate results\n"
        "4. Unresolved questions or pending actions\n\n"
        "Conversation:\n"
        + "\n".join(f"[{m['role']}]: {m['content']}" for m in history)
    )
    summary = await llm.generate(
        prompt=compression_prompt,
        max_tokens=max_tokens,
        temperature=0.0  # deterministic compression
    )
    return summary
```

**Key engineering principle:** Compression should be **task-aware** — a coding agent must preserve variable names and error messages exactly, while a research agent can afford to paraphrase background context.

#### D. External State (Databases, Vector Stores, File Systems)

External state resides **outside** the LLM's context window and the chain orchestrator's in-memory structures. It persists across chain runs, sessions, and even system restarts.

**Components:**

| Store Type | Read Latency | Write Latency | Query Model | Typical Use |
|---|---|---|---|---|
| **Relational DB** (Postgres) | ~1–10 ms | ~1–10 ms | SQL (structured) | User profiles, task logs, transactional records |
| **Vector store** (Pinecone, Weaviate, pgvector) | ~10–100 ms | ~10–100 ms | Approximate nearest neighbor (ANN) over embeddings | Semantic retrieval of past reasoning traces |
| **Document store** (MongoDB, S3) | ~5–50 ms | ~5–50 ms | Key-based / full-text | Chain artifacts (generated code files, reports) |
| **Key-value cache** (Redis) | ~0.1–1 ms | ~0.1–1 ms | Exact key lookup | Session state, rate-limit counters, tool-call caching |
| **File system** | ~1–10 ms | ~1–100 ms | Path-based | Code workspaces, generated assets |

**Interaction pattern.** External state is accessed via **tool calls** that are themselves chain steps:

$$
\mathcal{E}_{t+1} = \text{ToolExec}\!\bigl(\, a_t,\; \mathcal{E}_t \,\bigr)
$$

where $a_t$ is the action (SQL query, vector search, file write) determined by the LLM at step $t$.

**Consistency concerns:** When multiple chain branches or concurrent agents share external state, standard distributed-systems issues arise — **read-after-write consistency**, **lost updates**, **phantom reads**. The state management layer must specify an isolation level (e.g., snapshot isolation for vector stores that support transactions, or optimistic concurrency control via version vectors for key-value caches).

#### E. Composite State in Practice

Production agentic systems **combine all four categories** simultaneously. A well-architected state manager maintains a clear boundary between them:

```
┌──────────────────────────────────────────────────────┐
│                  Agent State Manager                  │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │  Explicit    │  │  Implicit   │  │  Compressed  │ │
│  │  (Pydantic)  │  │  (Messages) │  │  (Summaries) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                │                │          │
│         └────────┬───────┴────────┬───────┘          │
│                  │                │                   │
│           ┌──────▼──────┐  ┌─────▼──────┐            │
│           │  Render()   │  │  External  │            │
│           │  → tokens   │  │  State I/O │            │
│           └─────────────┘  └────────────┘            │
└──────────────────────────────────────────────────────┘
```

The `Render()` function selects which subset of explicit, implicit, and compressed state to serialize into the token budget for the current step, while external state is fetched on demand.

---

## 1.4.2 Context Window Management

### 1.4.2.1 The Fundamental Constraint

Every autoregressive transformer has a **hard context window** $C_{\max}$ (measured in tokens). The context at step $t$ must satisfy:

$$
\sum_{i=1}^{n} |c_i| \leq C_{\max}
$$

where $c_i$ are the individual content blocks included in the prompt (system instruction, history, retrieved documents, tool outputs, current query) and $|c_i|$ denotes the token count of block $c_i$.

Violating this constraint causes either a hard API error (truncation or rejection) or, in models with soft extrapolation (ALiBi, YaRN), a severe degradation in attention quality beyond the trained length.

### 1.4.2.2 Token Budget Allocation

Treat the context window as a **resource allocation problem**. The total budget $C_{\max}$ must be partitioned across $n$ blocks:

$$
C_{\max} = B_{\text{sys}} + B_{\text{history}} + B_{\text{retrieval}} + B_{\text{tools}} + B_{\text{query}} + B_{\text{reserved}}
$$

| Budget Slot | Typical Allocation | Rationale |
|---|---|---|
| $B_{\text{sys}}$ — System prompt | 500–2000 tokens (fixed) | Persona, guardrails, output format — rarely changes |
| $B_{\text{history}}$ — Conversation history | 30–50% of remaining | Prior turns, chain trajectory |
| $B_{\text{retrieval}}$ — Retrieved documents | 20–40% of remaining | RAG chunks, tool documentation |
| $B_{\text{tools}}$ — Tool schemas + outputs | 10–20% of remaining | Function signatures, API responses |
| $B_{\text{query}}$ — Current user turn | Variable (usually small) | The immediate instruction |
| $B_{\text{reserved}}$ — Generation headroom | 1000–4000 tokens | Space for the model's output |

**The allocation policy** $\pi_{\text{budget}}$ is a function:

$$
\pi_{\text{budget}}: (S_t, C_{\max}) \longrightarrow (B_{\text{sys}}, B_{\text{history}}, B_{\text{retrieval}}, B_{\text{tools}}, B_{\text{query}}, B_{\text{reserved}})
$$

subject to the constraint that all allocations sum to at most $C_{\max}$.

```python
from dataclasses import dataclass

@dataclass
class TokenBudget:
    total: int          # C_max
    system: int         # fixed
    reserved: int       # generation headroom
    query: int          # measured from current input
    
    @property
    def available(self) -> int:
        return self.total - self.system - self.reserved - self.query
    
    def allocate(self, history_ratio: float = 0.5,
                 retrieval_ratio: float = 0.3,
                 tools_ratio: float = 0.2) -> dict:
        avail = self.available
        return {
            "history":   int(avail * history_ratio),
            "retrieval": int(avail * retrieval_ratio),
            "tools":     int(avail * tools_ratio),
        }

# Example: GPT-4o with 128k context
budget = TokenBudget(total=128000, system=1200, reserved=4000, query=150)
alloc = budget.allocate()
# {'history': 61325, 'retrieval': 36795, 'tools': 24530}
```

### 1.4.2.3 Sliding Window Context Strategies

When the cumulative history $|H_t|$ exceeds $B_{\text{history}}$, a **sliding window** retains only the most recent $w$ messages:

$$
H_t^{\text{window}} = \bigl[\, \text{msg}_{t-w+1},\; \text{msg}_{t-w+2},\; \dots,\; \text{msg}_{t} \,\bigr]
$$

**Variants:**

| Variant | Description | Trade-off |
|---|---|---|
| **Fixed window** | Always keep last $w$ messages | Simple; loses early context entirely |
| **Anchored window** | Keep first $k$ messages (anchor) + last $w{-}k$ messages | Preserves original instructions; gap in middle |
| **Importance-weighted window** | Score each message by relevance; keep top-$w$ | Better retention; requires scoring mechanism |
| **Hierarchical window** | Keep recent messages verbatim, older messages as summaries | Progressive compression; more complex |

**Anchored sliding window** is the most common production pattern:

$$
H_t^{\text{anchored}} = \underbrace{[\text{msg}_1, \dots, \text{msg}_k]}_{\text{anchor (first } k \text{ msgs)}} \;\oplus\; \underbrace{[\text{msg}_{t-w+k+1}, \dots, \text{msg}_t]}_{\text{recency window}}
$$

This addresses the "lost in the middle" problem by ensuring the original task specification (in the anchor) and the most recent reasoning (in the recency window) both receive strong attention.

### 1.4.2.4 Selective Context Inclusion (Relevance-Based Filtering)

Rather than retaining messages by position, **selective inclusion** scores each candidate block by relevance to the current step's objective and includes only those above a threshold $\tau$:

$$
\text{Include}(c_i) = \mathbb{1}\!\bigl[\, \text{rel}(c_i, q_t) \geq \tau \,\bigr]
$$

**Relevance scoring methods:**

1. **Embedding similarity:**
$$
\text{rel}(c_i, q_t) = \cos\!\bigl(\, \mathbf{e}(c_i),\; \mathbf{e}(q_t) \,\bigr)
$$
where $\mathbf{e}(\cdot)$ is a sentence-embedding model.

2. **LLM-based scoring:**
$$
\text{rel}(c_i, q_t) = p_\theta\!\bigl(\, \texttt{"relevant"} \mid \text{``Is } c_i \text{ relevant to } q_t \text{?''} \,\bigr)
$$

3. **TF-IDF / BM25 lexical overlap** (fast, no neural inference).

4. **Attention-score replay:** If the model has already attended to $c_i$ in a previous step, the attention weight serves as a relevance proxy.

**Algorithm — Greedy Knapsack Context Selection:**

```python
def select_context(
    candidates: list[ContextBlock],
    query: str,
    budget: int,
    embed_fn: callable,
    threshold: float = 0.3
) -> list[ContextBlock]:
    """
    Greedy knapsack: include highest-relevance blocks
    until token budget is exhausted.
    """
    q_emb = embed_fn(query)
    scored = []
    for block in candidates:
        sim = cosine_similarity(embed_fn(block.text), q_emb)
        if sim >= threshold:
            scored.append((sim, block))
    
    # Sort descending by relevance
    scored.sort(key=lambda x: x[0], reverse=True)
    
    selected, used = [], 0
    for sim, block in scored:
        if used + block.token_count <= budget:
            selected.append(block)
            used += block.token_count
        else:
            break  # or try next smaller block (fractional knapsack)
    
    # Re-sort by original position to preserve chronological order
    selected.sort(key=lambda b: b.position)
    return selected
```

### 1.4.2.5 Context Compression Techniques

When even selective inclusion cannot fit the necessary information, **compression** reduces the token footprint of each block.

**Taxonomy of compression methods:**

| Method | Compression Ratio | Fidelity | Latency Cost |
|---|---|---|---|
| **Truncation** (hard cut) | Arbitrary | Very low | Zero |
| **Sentence extraction** (TextRank) | 3–5× | Moderate | Low |
| **Abstractive summarization** (LLM) | 5–20× | High | High (LLM call) |
| **Structured extraction** (→ JSON) | 10–50× | Very high for facts | Moderate |
| **Gisting tokens** (Mu et al., 2023) | 10–25× | Moderate | One forward pass to learn gist |
| **AutoCompressor** (Chevalier et al., 2023) | 6–12× | High | Recursive forward passes |

**Hierarchical progressive compression** — the most robust production pattern:

```
Step 1–3:    Full verbatim history     (recent, fits in budget)
Step 4–7:    Steps 1–3 summarized      (1 summary block), steps 4–7 verbatim
Step 8–12:   Steps 1–7 re-summarized   (1 block), steps 8–12 verbatim
...
Step t:      Recursively compressed prefix + verbatim recent window
```

Formally, at step $t$ the context is:

$$
\text{Context}_t = \mathcal{C}^{(d)}\!\bigl(H_1^{t-w}\bigr) \;\oplus\; H_{t-w+1}^{t}
$$

where $\mathcal{C}^{(d)}$ is $d$ rounds of recursive compression applied to the prefix, and $H_{t-w+1}^{t}$ is the verbatim recent window of size $w$.

### 1.4.2.6 Long-Context Models vs. Chaining as Context Extension

Two fundamentally different strategies exist for handling tasks whose state exceeds any single context window:

| Dimension | Long-Context Model | Chaining with Compression |
|---|---|---|
| **Mechanism** | Extend $C_{\max}$ (128k → 1M+ tokens) via RoPE scaling, ring attention, etc. | Keep $C_{\max}$ moderate; propagate compressed state across calls |
| **Attention complexity** | $\mathcal{O}(C_{\max}^2)$ or $\mathcal{O}(C_{\max} \log C_{\max})$ with efficient variants | $\mathcal{O}(w^2)$ per step (only window attends) |
| **Information access** | Theoretically full; practically degraded in the middle | Controlled by $\text{Render}()$; explicitly managed |
| **Cost per step** | High (large KV cache, slow prefill) | Lower per call; more calls total |
| **Failure mode** | Silent degradation (ignores distant context) | Explicit information loss (compression artifacts) |
| **Debuggability** | Low (which part of 500k tokens did it attend to?) | High (each step's context is inspectable) |

**Decision criterion.** Use long context when the task requires **simultaneous attention** across the entire document (e.g., cross-referencing distant passages in legal review). Use chaining when the task is **sequentially decomposable** and each step depends on a compact state representation.

**Hybrid approach** (recommended for production):

$$
C_{\text{effective}} = C_{\max} + \sum_{t=2}^{T} |\mathcal{C}(H_{1}^{t-1})|
$$

Use the largest practical $C_{\max}$, but still implement compression and selective inclusion so the system degrades gracefully when state grows.

---

## 1.4.3 Memory Mechanisms in Chains

### 1.4.3.1 Memory Taxonomy

Memory in agentic chains mirrors the taxonomy from cognitive science, adapted for computational systems:

$$
\text{Memory} = \begin{cases}
\text{Short-Term Memory (STM)} & \text{within a single chain execution} \\
\text{Working Memory (WM)} & \text{scratchpad / reasoning buffer} \\
\text{Long-Term Memory (LTM)} & \text{persists across chain runs} \\
\text{Episodic Memory (EM)} & \text{retrieval of past execution traces} \\
\text{Semantic Memory (SM)} & \text{factual knowledge base}
\end{cases}
$$

```
                    ┌──────────────────────────────────┐
   Persistence      │         Long-Term Memory         │
   (cross-run)      │  ┌────────────┐ ┌──────────────┐ │
                    │  │  Episodic  │ │  (knowledge) │ │
                    │  └─────┬──────┘ └──────┬───────┘ │
                    └────────┼───────────────┼─────────┘
                             │  retrieve     │  query
                    ┌────────▼───────────────▼─────────┐
   Active use       │        Working Memory            │
   (within step)    │  (scratchpad, intermediate vars) │
                    └────────┬─────────────────────────┘
                             │  feeds
                    ┌────────▼─────────────────────────┐
   Transient        │       Short-Term Memory          │
   (within chain)   │  (current context window)        │
                    └──────────────────────────────────┘
```

### 1.4.3.2 Short-Term Memory (STM)

**Definition.** The state that exists only during a single chain execution and is discarded upon completion.

**Implementation.** Simply the in-memory variables held by the orchestrator:

$$
\text{STM}_t = \{S_t, H_t, \text{local variables}\}
$$

**Lifetime:** Created when the chain starts, garbage-collected when the chain returns its final output.

**Key design principle:** STM should be **immutable-by-default** — each step receives a snapshot of state and produces a *new* state, never mutating the previous one. This enables rollback, branching, and debugging:

```python
from copy import deepcopy

class ChainExecutor:
    def __init__(self):
        self.state_history: list[ResearchState] = []  # full trajectory for debugging
    
    async def run_step(self, state: ResearchState, step_fn: callable) -> ResearchState:
        self.state_history.append(deepcopy(state))  # snapshot before mutation
        new_state = await step_fn(state)
        return new_state.advance()
```

### 1.4.3.3 Working Memory (WM) — Scratchpad Patterns

**Definition.** A dedicated mutable buffer where the LLM (or the orchestrator) writes intermediate computations that are *not* part of the final output but are essential for multi-step reasoning.

**Cognitive analog:** The "mental notepad" humans use when performing multi-digit arithmetic or planning a sequence of actions.

**Implementation patterns:**

**Pattern A — In-prompt scratchpad:**

```
<scratchpad>
- User wants: quarterly revenue comparison
- Step 1 result: Q1=$2.3M, Q2=$2.7M, Q3=$3.1M, Q4=$2.9M
- Computed: Q3 is highest (+14.8% over Q2)
- Still need: year-over-year comparison
</scratchpad>

Based on the data, Q3 had the highest revenue at $3.1M...
```

The scratchpad is included in the context but **excluded from the user-facing output** by the orchestrator.

**Pattern B — External scratchpad (chain-of-thought externalized):**

```python
class WorkingMemory:
    """External scratchpad — not in the LLM's context unless needed."""
    def __init__(self):
        self._pad: dict[str, any] = {}
        self._log: list[tuple[str, str, any]] = []  # (timestamp, key, value)
    
    def write(self, key: str, value: any) -> None:
        self._pad[key] = value
        self._log.append((datetime.utcnow().isoformat(), key, value))
    
    def read(self, key: str) -> any:
        return self._pad.get(key)
    
    def render_for_prompt(self, keys: list[str] = None) -> str:
        """Serialize selected keys into a prompt-injectable string."""
        subset = {k: self._pad[k] for k in (keys or self._pad.keys())}
        return json.dumps(subset, indent=2, default=str)
    
    def clear(self) -> None:
        self._pad.clear()
```

**Pattern C — Reasoning traces as working memory (ReAct-style):**

Each step's **Thought** is working memory, **Action** is the tool call, and **Observation** is the tool result written back to working memory:

$$
\text{WM}_{t} = \text{WM}_{t-1} \cup \{(\text{Thought}_t, \text{Action}_t, \text{Observation}_t)\}
$$

### 1.4.3.4 Long-Term Memory (LTM)

**Definition.** State that persists across multiple independent chain executions, potentially across sessions, users, or time scales of days to months.

**Storage backends:** Relational databases, vector stores, or hybrid systems.

**Write path (Memory Formation):**

$$
\text{LTM} \leftarrow \text{LTM} \cup \text{Extract}(S_T, y_T)
$$

At the end of a chain execution, a **memory extraction** step identifies facts, preferences, decisions, or skills worth retaining and writes them to persistent storage.

**Read path (Memory Retrieval):**

$$
\text{retrieved} = \text{TopK}\!\bigl(\, \text{LTM},\; q_t,\; k \,\bigr)
$$

where $\text{TopK}$ returns the $k$ most relevant long-term memories given the current query $q_t$, ranked by a combination of:

$$
\text{score}(m, q_t) = \alpha \cdot \text{relevance}(m, q_t) + \beta \cdot \text{recency}(m) + \gamma \cdot \text{importance}(m)
$$

This scoring function is adapted from **Generative Agents** (Park et al., 2023), where:

- $\text{relevance}$: cosine similarity between the memory embedding and the query embedding.
- $\text{recency}$: exponential decay $e^{-\lambda \Delta t}$ based on time since last access $\Delta t$.
- $\text{importance}$: a scalar assigned at write time (by the LLM or heuristics) reflecting the memory's general significance.

```python
import numpy as np
from datetime import datetime

class LongTermMemory:
    def __init__(self, embed_fn, decay_rate: float = 0.995):
        self.memories: list[dict] = []
        self.embed_fn = embed_fn
        self.decay_rate = decay_rate
    
    def add(self, content: str, importance: float = 0.5, metadata: dict = None):
        embedding = self.embed_fn(content)
        self.memories.append({
            "content": content,
            "embedding": embedding,
            "importance": importance,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            "metadata": metadata or {}
        })
    
    def retrieve(self, query: str, k: int = 5,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> list[dict]:
        q_emb = self.embed_fn(query)
        now = datetime.utcnow()
        scored = []
        
        for mem in self.memories:
            relevance = np.dot(q_emb, mem["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(mem["embedding"]) + 1e-8
            )
            hours_since = (now - mem["last_accessed"]).total_seconds() / 3600
            recency = self.decay_rate ** hours_since
            importance = mem["importance"]
            
            score = alpha * relevance + beta * recency + gamma * importance
            scored.append((score, mem))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update access metadata for retrieved memories
        for _, mem in scored[:k]:
            mem["last_accessed"] = now
            mem["access_count"] += 1
        
        return [mem for _, mem in scored[:k]]
```

### 1.4.3.5 Episodic Memory (EM)

**Definition.** A specialized form of LTM that stores **complete or summarized execution traces** of past chain runs, indexed by task type, outcome, and context.

**Purpose:** Enables the agent to:
1. **Learn from past successes:** Retrieve a similar successful trace and use it as a few-shot example.
2. **Avoid past failures:** Detect when the current situation resembles a past failure and choose an alternative strategy.
3. **Estimate difficulty:** If similar past tasks took $T$ steps, budget accordingly.

**Schema:**

$$
\text{Episode}_i = \bigl(\, x_i,\; \{(a_t, o_t)\}_{t=1}^{T_i},\; r_i,\; \text{meta}_i \,\bigr)
$$

| Field | Meaning |
|---|---|
| $x_i$ | Task description |
| $(a_t, o_t)$ | Action-observation pairs (the trajectory) |
| $r_i$ | Outcome / reward (success, failure, partial) |
| $\text{meta}_i$ | Duration, token cost, tools used, error types |

**Retrieval at inference time:**

$$
\text{exemplars} = \text{TopK}_{\text{episodes}}\!\bigl(\, x_{\text{new}},\; k,\; \text{filter}=\{r_i = \text{success}\} \,\bigr)
$$

These exemplars are injected into the prompt as **demonstration trajectories**, implementing a form of in-context learning from the agent's own past behavior.

### 1.4.3.6 Semantic Memory (SM)

**Definition.** A structured knowledge base containing **domain facts, rules, ontologies, and procedural knowledge** that the agent can query during chain execution.

**Distinct from episodic memory:** Semantic memory stores *what is true* (atemporal facts), while episodic memory stores *what happened* (temporal traces).

**Implementations:**

| Backend | Content Type | Query Interface |
|---|---|---|
| **Knowledge graph** (Neo4j) | Entity-relation triples $(e_1, r, e_2)$ | Cypher / SPARQL |
| **Vector store** | Embedded text chunks | ANN similarity search |
| **Structured DB** | Tables with domain schemas | SQL |
| **Document index** | Indexed documents with metadata | BM25 + dense retrieval |

**Integration as a chain step:**

```python
async def semantic_memory_step(state: ResearchState, sm: SemanticMemory) -> ResearchState:
    """Query semantic memory and inject relevant knowledge."""
    query = state.query + " " + (state.draft_answer or "")
    facts = await sm.retrieve(query, k=10)
    state.extracted_facts.extend([f["content"] for f in facts])
    state.token_budget_remaining -= sum(f["token_count"] for f in facts)
    return state
```

### 1.4.3.7 Memory Read/Write Operations as Chain Steps

**Critical architectural principle:** Memory operations are **not invisible side effects** — they are explicit, inspectable, auditable chain steps.

**Formalization.** Define two primitive operations:

$$
\text{MemRead}(q, \mathcal{M}) \rightarrow \{m_1, m_2, \dots, m_k\}
$$

$$
\text{MemWrite}(v, \mathcal{M}) \rightarrow \mathcal{M}' = \mathcal{M} \cup \{v\}
$$

These are embedded in the chain DAG as first-class nodes:

```
[User Query] → [MemRead: retrieve relevant memories]
                        ↓
              [Reasoning Step (LLM call)]
                        ↓
              [MemWrite: store new insights]
                        ↓
              [Response Generation]
```

**Read-before-write consistency:** Within a single chain execution, every `MemWrite` in step $i$ must be visible to any `MemRead` in step $j > i$. This is trivially satisfied for in-memory stores but requires explicit flush/commit semantics for external stores.

```python
class MemoryManager:
    """Unified interface for all memory types."""
    def __init__(self, stm: dict, wm: WorkingMemory,
                 ltm: LongTermMemory, em: EpisodicMemory,
                 sm: SemanticMemory):
        self.stm = stm
        self.wm = wm
        self.ltm = ltm
        self.em = em
        self.sm = sm
    
    async def read(self, query: str, sources: list[str] = None) -> dict:
        """Unified read across memory systems."""
        sources = sources or ["wm", "ltm", "em", "sm"]
        results = {}
        if "wm" in sources:
            results["working"] = self.wm.render_for_prompt()
        if "ltm" in sources:
            results["long_term"] = self.ltm.retrieve(query, k=5)
        if "em" in sources:
            results["episodes"] = self.em.retrieve_similar(query, k=3)
        if "sm" in sources:
            results["knowledge"] = await self.sm.retrieve(query, k=10)
        return results
    
    async def write(self, content: str, target: str,
                    importance: float = 0.5, **kwargs) -> None:
        """Route writes to appropriate memory system."""
        if target == "wm":
            self.wm.write(kwargs.get("key", "default"), content)
        elif target == "ltm":
            self.ltm.add(content, importance=importance)
        elif target == "em":
            self.em.store_episode(**kwargs)
        elif target == "sm":
            await self.sm.add_fact(content, **kwargs)
```

---

## 1.4.4 Data Flow Patterns

### 1.4.4.1 Overview

Data flow patterns govern **how information moves between chain nodes**. The choice of pattern determines latency, coupling, debuggability, and scalability of the agentic system.

$$
\text{DataFlow}: \text{Node}_i \xrightarrow{\text{pattern}} \text{Node}_j
$$

### 1.4.4.2 Push-Based Data Flow

**Mechanism.** When node $i$ completes, it **actively sends** its output to all registered downstream nodes $\{j_1, j_2, \dots\}$.

$$
\text{Node}_i \xrightarrow{\text{push}(y_i)} \text{Node}_{j_k} \quad \forall\, j_k \in \text{downstream}(i)
$$

```python
class PushNode:
    def __init__(self, name: str, process_fn: callable):
        self.name = name
        self.process_fn = process_fn
        self.downstream: list["PushNode"] = []
    
    def register_downstream(self, node: "PushNode"):
        self.downstream.append(node)
    
    async def execute(self, input_data: dict) -> None:
        output = await self.process_fn(input_data)
        # Push to all downstream nodes
        for node in self.downstream:
            await node.execute(output)
```

**Characteristics:**

| Property | Value |
|---|---|
| **Coupling** | Tight — upstream must know about downstream |
| **Latency** | Minimal (immediate propagation) |
| **Flow control** | Upstream controls pace — downstream may be overwhelmed |
| **Error handling** | Upstream must handle downstream failures (retry, circuit-break) |
| **Best for** | Linear chains, simple fan-out DAGs |

### 1.4.4.3 Pull-Based Data Flow

**Mechanism.** Node $j$ **requests** data from upstream node $i$ only when it is ready to process. Upstream produces lazily.

$$
\text{Node}_j \xrightarrow{\text{request}} \text{Node}_i \xrightarrow{\text{respond}(y_i)} \text{Node}_j
$$

```python
class PullNode:
    def __init__(self, name: str, process_fn: callable):
        self.name = name
        self.process_fn = process_fn
        self._result_cache: dict = None
        self._computed: bool = False
    
    async def get_result(self) -> dict:
        """Lazy evaluation: compute only when pulled."""
        if not self._computed:
            # Pull from upstream dependencies first
            upstream_data = {}
            for dep_name, dep_node in self.dependencies.items():
                upstream_data[dep_name] = await dep_node.get_result()
            self._result_cache = await self.process_fn(upstream_data)
            self._computed = True
        return self._result_cache
```

**Characteristics:**

| Property | Value |
|---|---|
| **Coupling** | Loose — downstream knows what it needs, upstream is passive |
| **Latency** | Higher (request-response round trip) |
| **Flow control** | Downstream controls pace — natural backpressure |
| **Computation** | Lazy — unnecessary branches are never computed |
| **Best for** | Conditional chains where not all branches are needed |

### 1.4.4.4 Publish-Subscribe (Pub-Sub) Within Chains

**Mechanism.** Nodes publish events to named **topics** (or channels). Other nodes subscribe to topics of interest. A **message broker** mediates, decoupling publishers from subscribers entirely.

$$
\text{Node}_i \xrightarrow{\text{publish}(y_i, \text{topic})} \text{Broker} \xrightarrow{\text{deliver}} \{\text{Node}_j \mid j \in \text{subscribers}(\text{topic})\}
$$

```python
import asyncio
from collections import defaultdict

class ChainEventBus:
    """In-process pub-sub for chain data flow."""
    def __init__(self):
        self._subscribers: dict[str, list[callable]] = defaultdict(list)
        self._event_log: list[tuple] = []  # audit trail
    
    def subscribe(self, topic: str, handler: callable):
        self._subscribers[topic].append(handler)
    
    async def publish(self, topic: str, data: dict, source: str = ""):
        event = {"topic": topic, "data": data, "source": source,
                 "timestamp": datetime.utcnow().isoformat()}
        self._event_log.append(event)
        
        handlers = self._subscribers.get(topic, [])
        # Fan-out: deliver to all subscribers concurrently
        await asyncio.gather(*[h(data) for h in handlers])

# Usage
bus = ChainEventBus()
bus.subscribe("search_complete", reasoning_node.on_search_results)
bus.subscribe("search_complete", logging_node.on_search_results)
await bus.publish("search_complete", {"results": [...]}, source="search_node")
```

**Characteristics:**

| Property | Value |
|---|---|
| **Coupling** | Very loose — publisher and subscriber know only the topic schema |
| **Extensibility** | New subscribers can be added without modifying publishers |
| **Ordering** | Not guaranteed unless explicitly managed (sequence numbers) |
| **Debugging** | Event log provides full audit trail |
| **Best for** | Complex multi-agent systems, observation/logging side-channels |

### 1.4.4.5 Shared Blackboard Pattern

**Mechanism.** All nodes read from and write to a **shared data structure** (the "blackboard"). Each node examines the blackboard, determines if it can contribute, and writes its results back. A **controller** decides which node runs next based on blackboard state.

$$
\mathcal{B}_{t+1} = \mathcal{B}_t \oplus \text{Node}_{k_t}\!\bigl(\mathcal{B}_t\bigr)
$$

where $k_t$ is the node selected by the controller at step $t$.

```python
class Blackboard:
    """Shared mutable state for blackboard architecture."""
    def __init__(self):
        self._state: dict[str, any] = {}
        self._history: list[dict] = []  # versioned snapshots
        self._version: int = 0
    
    def read(self, key: str) -> any:
        return self._state.get(key)
    
    def write(self, key: str, value: any, author: str = ""):
        self._version += 1
        self._state[key] = value
        self._history.append({
            "version": self._version,
            "key": key, "value": value,
            "author": author,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def snapshot(self) -> dict:
        return deepcopy(self._state)
    
    def rollback(self, version: int):
        """Restore blackboard to a prior version."""
        self._state.clear()
        for entry in self._history:
            if entry["version"] <= version:
                self._state[entry["key"]] = entry["value"]
        self._version = version


class BlackboardController:
    """Selects the next knowledge source (node) to activate."""
    def __init__(self, knowledge_sources: list, blackboard: Blackboard):
        self.sources = knowledge_sources
        self.bb = blackboard
    
    async def run(self, max_iterations: int = 20) -> dict:
        for _ in range(max_iterations):
            # Each source inspects the blackboard and bids
            bids = []
            for source in self.sources:
                if source.can_contribute(self.bb):
                    bids.append((source.priority(self.bb), source))
            
            if not bids:
                break  # no source can contribute — done
            
            # Highest-priority source executes
            bids.sort(key=lambda x: x[0], reverse=True)
            _, best_source = bids[0]
            await best_source.execute(self.bb)
        
        return self.bb.snapshot()
```

**Characteristics:**

| Property | Value |
|---|---|
| **Coupling** | Medium — nodes must agree on blackboard key schema |
| **Coordination** | Centralized controller, opportunistic node activation |
| **Concurrency** | Requires locking / MVCC if nodes run in parallel |
| **Transparency** | Full version history of all writes |
| **Best for** | Expert-system-style agents, collaborative multi-specialist architectures |

### 1.4.4.6 Event-Driven Data Flow

**Mechanism.** State changes are modeled as **events** (immutable facts about what happened). Nodes react to events, potentially emitting new events. The system's state at any point is the **fold** (reduction) over the event stream.

$$
S_t = \text{fold}\!\bigl(\, S_0,\; [e_1, e_2, \dots, e_t] \,\bigr) = S_0 \oplus e_1 \oplus e_2 \oplus \cdots \oplus e_t
$$

This is formally an **event-sourcing** architecture.

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class EventType(Enum):
    CHAIN_STARTED = "chain_started"
    STEP_COMPLETED = "step_completed"
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    MEMORY_WRITTEN = "memory_written"
    ERROR_OCCURRED = "error_occurred"
    CHAIN_COMPLETED = "chain_completed"

@dataclass(frozen=True)  # immutable
class ChainEvent:
    event_type: EventType
    payload: dict
    source_node: str
    timestamp: str
    event_id: str = field(default_factory=lambda: str(uuid4()))

class EventDrivenChain:
    def __init__(self):
        self._event_log: list[ChainEvent] = []
        self._handlers: dict[EventType, list[callable]] = defaultdict(list)
    
    def on(self, event_type: EventType, handler: callable):
        self._handlers[event_type].append(handler)
    
    async def emit(self, event: ChainEvent):
        self._event_log.append(event)  # append-only log
        for handler in self._handlers.get(event.event_type, []):
            # Handler may emit further events (cascading)
            await handler(event, self)
    
    def reconstruct_state(self, up_to_event_id: str = None) -> dict:
        """Replay events to reconstruct state at any point."""
        state = {}
        for event in self._event_log:
            state = self._apply_event(state, event)
            if event.event_id == up_to_event_id:
                break
        return state
    
    @staticmethod
    def _apply_event(state: dict, event: ChainEvent) -> dict:
        """Pure function: state transition given an event."""
        new_state = {**state}
        if event.event_type == EventType.STEP_COMPLETED:
            new_state["last_step"] = event.payload
            new_state.setdefault("steps", []).append(event.payload)
        elif event.event_type == EventType.ERROR_OCCURRED:
            new_state.setdefault("errors", []).append(event.payload)
        # ... handle other event types
        return new_state
```

**Characteristics:**

| Property | Value |
|---|---|
| **Coupling** | Very loose — nodes know only event schemas |
| **Auditability** | Complete — the event log is the single source of truth |
| **Replayability** | Any past state can be reconstructed by replaying events |
| **Debugging** | Excellent — "time-travel" debugging is trivial |
| **Complexity** | Higher — requires careful event schema design and ordering guarantees |
| **Best for** | Production agentic systems requiring full observability and reproducibility |

### 1.4.4.7 Comparative Analysis

| Pattern | Coupling | Complexity | Scalability | Debuggability | Latency |
|---|---|---|---|---|---|
| **Push** | High | Low | Moderate | Low | Lowest |
| **Pull** | Moderate | Low–Med | Moderate | Moderate | Moderate |
| **Pub-Sub** | Low | Medium | High | High | Moderate |
| **Blackboard** | Medium | Medium | Low–Med | High | Variable |
| **Event-Driven** | Very Low | High | Very High | Very High | Moderate |

### 1.4.4.8 Recommended Composite Architecture

Production agentic systems **blend multiple patterns**:

```
┌──────────────────────────────────────────────────────────────┐
│                   Agentic Chain Runtime                       │
│                                                              │
│   Primary Flow:   Push-based (linear step → step)            │
│   Branching:      Pull-based (conditional, lazy evaluation)  │
│   Observation:    Pub-Sub (logging, monitoring, metrics)      │
│   Shared State:   Blackboard (multi-agent collaboration)     │
│   Audit Layer:    Event-sourced (full replay capability)      │
│                                                              │
│   ┌──────────┐ push ┌──────────┐ push ┌──────────┐          │
│   │  Step 1   │─────▶│  Step 2   │─────▶│  Step 3   │        │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘          │
│        │ emit             │ emit             │ emit           │
│        ▼                  ▼                  ▼                │
│   ┌─────────────── Event Bus (Pub-Sub) ──────────────────┐   │
│   │  topic: step_completed │ topic: tool_called │ ...     │   │
│   └──┬──────────────┬──────────────────┬─────────────────┘   │
│      │              │                  │                      │
│      ▼              ▼                  ▼                      │
│   [Logger]    [Monitor]         [Event Store]                │
│                                 (append-only log)            │
│                                                              │
│   ┌──────────────── Blackboard ──────────────────┐           │
│   │  shared_state: { ... }                        │           │
│   │  accessed by: Step 1, Step 2, Step 3          │           │
│   └───────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

**Design rationale:**

1. **Push** for the happy path — minimal latency for sequential reasoning.
2. **Pull** for optional branches — avoid computing expensive tool calls unless downstream actually needs them.
3. **Pub-Sub** for cross-cutting concerns — logging, monitoring, and guardrail checks observe every step without coupling to it.
4. **Blackboard** for multi-agent coordination — when multiple specialist agents (coder, reviewer, tester) collaborate on a shared artifact.
5. **Event sourcing** for the audit layer — enables full reproducibility, time-travel debugging, and post-hoc analysis of chain behavior.

---

## Summary: State Management Decision Matrix

| Decision | Key Question | Recommended Approach |
|---|---|---|
| **State representation** | Is the chain short ($T \leq 3$) and simple? | Implicit (message list) |
| | Does the chain require validation and branching? | Explicit (typed schema) |
| | Does state exceed context window? | Compressed + External |
| **Context management** | Is total state $< 50\%$ of $C_{\max}$? | Include everything verbatim |
| | Is state $50{-}100\%$ of $C_{\max}$? | Selective inclusion + anchored window |
| | Is state $> C_{\max}$? | Hierarchical compression + external memory |
| **Memory type** | Within a single chain run? | STM + Working Memory |
| | Across runs, same user? | LTM with relevance-recency-importance scoring |
| | Learning from past executions? | Episodic Memory |
| | Domain knowledge? | Semantic Memory (knowledge base / vector store) |
| **Data flow** | Simple sequential chain? | Push |
| | Complex DAG with optional branches? | Pull + Push hybrid |
| | Multi-agent or extensible system? | Pub-Sub + Blackboard |
| | Production system needing auditability? | Event-driven (event sourcing) |