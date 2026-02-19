

# 1.10 Advanced Prompt Chaining Techniques

> **Scope.** The foundational chain patterns (linear, DAG, conditional) covered in prior sections handle the majority of structured workflows. This section extends the repertoire with **advanced compositional patterns** — recursive structures, parallel map-reduce topologies, iterative refinement loops, ensemble architectures, retrieval-augmented pipelines, tool-augmented execution, multi-modal cross-model chains, and self-adaptive meta-chains. Each pattern is formalized mathematically, analyzed for convergence/correctness properties, and accompanied by production-grade implementations. Mastery of these patterns is what separates simple prompt-template systems from true agentic architectures.

---

## 1.10.1 Chain-of-Thought as a Micro-Chain

### 1.10.1.1 CoT as Implicit Single-Step Chaining

Chain-of-Thought (CoT) prompting (Wei et al., 2022) instructs the LLM to produce **intermediate reasoning steps** before its final answer — all within a **single LLM call**. Formally, CoT transforms the generation from:

$$
y \sim p_\theta(y \mid x)
$$

to:

$$
(r_1, r_2, \dots, r_k, y) \sim p_\theta(r_1, \dots, r_k, y \mid x, \text{prompt}_{\text{CoT}})
$$

where $r_1, \dots, r_k$ are reasoning tokens that the model generates **autoregressively before** the final answer $y$. Each $r_i$ conditions the generation of $r_{i+1}$ and ultimately $y$, creating what is functionally a **sequential chain executed entirely within the model's forward pass**.

**Why CoT works — the representational argument.** Transformers are bounded in the computation they can perform per forward pass. Without CoT, the model must map $x \to y$ in a fixed number of layers $L$. With CoT, the model gets $k$ additional "computation steps" via autoregressive token generation:

$$
\text{Effective compute} \propto L + k \cdot L
$$

Each generated reasoning token provides an additional $L$ layers of computation, effectively turning a bounded-depth circuit into an arbitrarily deep one — analogous to a Turing machine's tape.

### 1.10.1.2 Explicit Decomposition of CoT into Multi-Step Chains

CoT can be **externalized** from a single prompt into an explicit multi-step chain where each reasoning step is a separate LLM call with its own prompt, validation, and state management.

**Single-call CoT (implicit):**

```
Prompt: "Think step by step. What is 23 × 47?"
Output: "Step 1: 23 × 40 = 920. Step 2: 23 × 7 = 161. Step 3: 920 + 161 = 1081. Answer: 1081"
```

**Multi-call explicit chain (externalized CoT):**

```
Step 1 prompt: "Decompose 23 × 47 into simpler sub-problems."
Step 1 output: "Sub-problems: 23 × 40 and 23 × 7"

Step 2 prompt: "Compute 23 × 40."
Step 2 output: "920"

Step 3 prompt: "Compute 23 × 7."  
Step 3 output: "161"

Step 4 prompt: "Sum: 920 + 161 = ?"
Step 4 output: "1081"
```

**Formal equivalence.** Let $f_{\text{CoT}}(x)$ denote the single-call CoT output and $f_{\text{chain}}(x)$ denote the explicit chain output. Under ideal conditions:

$$
f_{\text{CoT}}(x) = f_{\text{chain}}(x) = y^*
$$

In practice, they diverge due to different error profiles.

### 1.10.1.3 Comparative Analysis: CoT vs. Explicit Prompt Chains

| Dimension | CoT (Single Call) | Explicit Chain (Multi-Call) |
|---|---|---|
| **Latency** | Single round-trip ($\sim l_1$) | Multiple round-trips ($\sum l_i$) |
| **Cost** | Lower (one call, reasoning tokens are output tokens) | Higher (each step has full prompt overhead) |
| **Error isolation** | Impossible — if step 3 is wrong, can't retry just step 3 | Each step independently verifiable and retryable |
| **Validation** | Post-hoc only (parse reasoning after generation) | Per-step validation gates between calls |
| **Context utilization** | All reasoning shares one context window | Each step can have a tailored context |
| **Controllability** | Low — model decides its own reasoning path | High — orchestrator controls step sequence |
| **Tool integration** | Not possible mid-reasoning (unless native tool-use) | Natural — any step can invoke tools |
| **Debugging** | Parse the monolithic output | Inspect each step independently |
| **Parallelism** | None (sequential token generation) | Independent sub-problems can parallelize |

### 1.10.1.4 Decision Framework: CoT Within a Step vs. Across Steps

$$
\text{Use CoT within a single step when:}
\begin{cases}
\text{Reasoning is short} & (k \leq 10 \text{ reasoning steps}) \\
\text{No tool calls needed mid-reasoning} & \\
\text{Error tolerance is moderate} & \\
\text{Latency is critical} & \\
\text{The sub-problem is self-contained} &
\end{cases}
$$

$$
\text{Use explicit chain across steps when:}
\begin{cases}
\text{Reasoning is long} & (k > 10 \text{ steps}) \\
\text{Intermediate results need validation} & \\
\text{Tools/retrieval needed mid-reasoning} & \\
\text{Errors in sub-steps are costly} & \\
\text{Sub-problems can parallelize} & \\
\text{Different models optimal for different sub-steps} &
\end{cases}
$$

**Hybrid pattern — the most practical approach:** Use CoT within each step of a multi-step chain. Each chain step gets "think step by step" in its prompt, but the chain orchestrator controls the macro-level flow:

```python
class HybridCoTChainStep:
    """A chain step that uses CoT internally but is externally orchestrated."""
    
    COT_WRAPPER = (
        "Think through this step by step before providing your final answer.\n\n"
        "Task: {task}\n"
        "Context from previous steps: {context}\n\n"
        "Show your reasoning, then provide the final answer in the specified format.\n"
        "Reasoning:\n"
    )
    
    def __init__(self, llm: "LLMClient", step_name: str, 
                 output_schema: type = None):
        self.llm = llm
        self.step_name = step_name
        self.output_schema = output_schema
    
    async def execute(self, task: str, context: str = "") -> dict:
        prompt = self.COT_WRAPPER.format(task=task, context=context)
        
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=2000
        )
        
        # Separate reasoning from answer
        reasoning, answer = self._extract_reasoning_and_answer(response)
        
        return {
            "step": self.step_name,
            "reasoning": reasoning,  # for debugging, not passed downstream
            "answer": answer,        # this is what downstream steps receive
        }
    
    @staticmethod
    def _extract_reasoning_and_answer(response: str) -> tuple[str, str]:
        # Look for explicit answer markers
        markers = ["Final answer:", "Answer:", "Therefore:", "Result:"]
        for marker in markers:
            if marker.lower() in response.lower():
                idx = response.lower().index(marker.lower())
                reasoning = response[:idx].strip()
                answer = response[idx + len(marker):].strip()
                return reasoning, answer
        # Fallback: last paragraph is the answer
        paragraphs = response.strip().split("\n\n")
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:-1]), paragraphs[-1]
        return "", response
```

---

## 1.10.2 Recursive Chains

### 1.10.2.1 Self-Referential Chain Definition

A recursive chain is a chain that **invokes itself** as a sub-step, processing progressively smaller or simpler versions of the problem until a base case is reached.

**Formal definition.** A recursive chain $f$ is:

$$
f(x) = \begin{cases}
g(x) & \text{if } \text{base}(x) = \texttt{True} \quad \text{(base case)} \\
h\!\bigl(x,\; f(\text{reduce}(x))\bigr) & \text{otherwise} \quad \text{(recursive case)}
\end{cases}
$$

where:
- $\text{base}: \mathcal{X} \to \{0,1\}$ is the base case predicate
- $g: \mathcal{X} \to \mathcal{Y}$ is the direct solution for base cases
- $\text{reduce}: \mathcal{X} \to \mathcal{X}$ reduces the problem to a smaller instance
- $h: \mathcal{X} \times \mathcal{Y} \to \mathcal{Y}$ combines the current level with recursive results

More generally, for multi-branch recursion:

$$
f(x) = h\!\bigl(x,\; f(\text{reduce}_1(x)),\; f(\text{reduce}_2(x)),\; \dots,\; f(\text{reduce}_m(x))\bigr)
$$

### 1.10.2.2 Recursion Depth Control and Base Cases

**Termination guarantees.** A recursive chain terminates iff there exists a well-founded ordering $\prec$ on inputs such that:

$$
\text{reduce}(x) \prec x \quad \forall\, x \text{ where } \text{base}(x) = \texttt{False}
$$

and there are no infinite descending chains under $\prec$.

In practice, since LLM outputs are stochastic and $\text{reduce}$ may not strictly reduce problem size, we enforce termination via **hard depth limits**:

$$
f_d(x) = \begin{cases}
g(x) & \text{if } \text{base}(x) = \texttt{True} \\
g_{\text{approx}}(x) & \text{if } d \geq d_{\max} \quad \text{(forced base case)} \\
h\!\bigl(x,\; f_{d+1}(\text{reduce}(x))\bigr) & \text{otherwise}
\end{cases}
$$

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class RecursionConfig:
    max_depth: int = 5
    min_chunk_size: int = 100      # tokens
    base_case_detector: Optional[Callable[[str], bool]] = None
    depth_exceeded_strategy: str = "force_base"  # force_base | error | approximate

class RecursiveChain:
    """Generic recursive chain with depth control and base case detection."""
    
    def __init__(self, 
                 llm: "LLMClient",
                 base_solver: Callable[[str], str],
                 reducer: Callable[[str], list[str]],
                 combiner: Callable[[str, list[str]], str],
                 config: RecursionConfig = None):
        self.llm = llm
        self.base_solver = base_solver    # g(x): solve base case directly
        self.reducer = reducer            # reduce(x): split into sub-problems
        self.combiner = combiner          # h(x, results): combine sub-results
        self.config = config or RecursionConfig()
        self._trace: list[dict] = []
    
    async def execute(self, input_text: str, depth: int = 0) -> str:
        trace_entry = {
            "depth": depth,
            "input_length": len(input_text),
            "is_base_case": False
        }
        
        # Check base case
        is_base = self._is_base_case(input_text, depth)
        trace_entry["is_base_case"] = is_base
        
        if is_base:
            result = await self.base_solver(input_text)
            trace_entry["action"] = "base_solve"
            self._trace.append(trace_entry)
            return result
        
        # Check depth limit
        if depth >= self.config.max_depth:
            if self.config.depth_exceeded_strategy == "force_base":
                result = await self.base_solver(input_text)
                trace_entry["action"] = "forced_base_at_max_depth"
                self._trace.append(trace_entry)
                return result
            elif self.config.depth_exceeded_strategy == "error":
                raise RecursionError(f"Max depth {self.config.max_depth} exceeded")
            else:
                result = await self.base_solver(input_text)
                self._trace.append(trace_entry)
                return result
        
        # Recursive case: reduce → recurse → combine
        sub_problems = await self.reducer(input_text)
        trace_entry["num_sub_problems"] = len(sub_problems)
        trace_entry["action"] = "recurse"
        self._trace.append(trace_entry)
        
        # Recurse on each sub-problem (can parallelize independent branches)
        sub_results = await asyncio.gather(*[
            self.execute(sub, depth + 1) for sub in sub_problems
        ])
        
        # Combine
        combined = await self.combiner(input_text, list(sub_results))
        return combined
    
    def _is_base_case(self, text: str, depth: int) -> bool:
        if self.config.base_case_detector:
            return self.config.base_case_detector(text)
        # Default: base case when input is small enough
        return len(text.split()) <= self.config.min_chunk_size
```

### 1.10.2.3 Recursive Summarization Chains

The canonical application of recursive chains: summarize a document too long for a single context window by recursively summarizing sub-sections.

**Algorithm.** Given document $D$ with $|D|$ tokens and target summary length $L$:

$$
\text{RecSummarize}(D) = \begin{cases}
\text{Summarize}(D) & \text{if } |D| \leq C_{\max} - L \\
\text{Summarize}\!\Bigl(\bigoplus_{i=1}^{k} \text{RecSummarize}(D_i)\Bigr) & \text{otherwise}
\end{cases}
$$

where $D = D_1 \oplus D_2 \oplus \cdots \oplus D_k$ is a partition of $D$ into $k$ chunks that each fit in the context window, and $\oplus$ denotes concatenation.

**Compression analysis.** If each level achieves a compression ratio $\rho$ (summary is $\rho$ fraction of input), then after $d$ levels of recursion:

$$
|S_d| = |D| \cdot \rho^d
$$

The required depth to achieve target length $L$:

$$
d^* = \left\lceil \frac{\log(L / |D|)}{\log \rho} \right\rceil
$$

For $|D| = 100{,}000$ tokens, $L = 500$ tokens, $\rho = 0.1$: $d^* = \lceil \log(0.005) / \log(0.1) \rceil = \lceil 2.3 \rceil = 3$ levels.

```python
class RecursiveSummarizer:
    """Recursively summarize documents of arbitrary length."""
    
    SUMMARIZE_PROMPT = """Summarize the following text concisely, preserving all 
key facts, arguments, and conclusions. Target length: {target_words} words.

Text:
{text}

Summary:"""
    
    def __init__(self, llm: "LLMClient", 
                 context_window: int = 120000,
                 target_summary_tokens: int = 500,
                 chunk_overlap: int = 200,
                 compression_ratio: float = 0.1):
        self.llm = llm
        self.context_window = context_window
        self.target_tokens = target_summary_tokens
        self.overlap = chunk_overlap
        self.compression_ratio = compression_ratio
    
    async def summarize(self, text: str, depth: int = 0, 
                        max_depth: int = 5) -> str:
        tokens = text.split()  # approximate tokenization
        
        # Base case: fits in context window
        usable_window = self.context_window - self.target_tokens - 500  # prompt overhead
        if len(tokens) <= usable_window or depth >= max_depth:
            return await self._single_summarize(text, self.target_tokens)
        
        # Recursive case: chunk → summarize each → combine summaries → summarize
        chunks = self._chunk_text(tokens, usable_window)
        
        # Summarize each chunk (in parallel)
        chunk_target = max(
            int(self.target_tokens * len(tokens) / usable_window * self.compression_ratio),
            100
        )
        
        chunk_summaries = await asyncio.gather(*[
            self.summarize(" ".join(chunk), depth + 1, max_depth)
            for chunk in chunks
        ])
        
        # Combine all chunk summaries
        combined = "\n\n---\n\n".join(chunk_summaries)
        
        # If combined summaries fit in window, do final summarization
        if len(combined.split()) <= usable_window:
            return await self._single_summarize(combined, self.target_tokens)
        else:
            # Need another level of recursion
            return await self.summarize(combined, depth + 1, max_depth)
    
    async def _single_summarize(self, text: str, target_tokens: int) -> str:
        prompt = self.SUMMARIZE_PROMPT.format(
            text=text,
            target_words=int(target_tokens * 0.75)  # tokens → words approx
        )
        return await self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=target_tokens)
    
    def _chunk_text(self, tokens: list[str], max_chunk: int) -> list[list[str]]:
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_chunk, len(tokens))
            chunks.append(tokens[start:end])
            start = end - self.overlap  # overlap for context continuity
            if start >= len(tokens) - self.overlap:
                break
        return chunks
```

### 1.10.2.4 Recursive Refinement Chains

Apply the recursive pattern to **iterative improvement**: each recursion level refines the output of the previous level rather than processing a sub-problem.

$$
y^{(k+1)} = \text{Refine}\!\bigl(x, y^{(k)}, \text{Critique}(y^{(k)})\bigr)
$$

$$
f(x, d) = \begin{cases}
y^{(0)} = \text{Draft}(x) & \text{if } d = 0 \\
\text{Refine}\!\bigl(x, f(x, d-1), \text{Critique}(f(x, d-1))\bigr) & \text{if } d > 0
\end{cases}
$$

This recursion unfolds to:

$$
y^{(d)} = \underbrace{\text{Refine} \circ \text{Critique} \circ \cdots \circ \text{Refine} \circ \text{Critique}}_{d \text{ rounds}} \circ \text{Draft}(x)
$$

---

## 1.10.3 Map-Reduce Chains

### 1.10.3.1 Formal Definition

Map-Reduce is a computational paradigm that decomposes a problem into **independent sub-problems** (Map), solves each in parallel, and **aggregates** the results (Reduce).

**Formal model.** Given input $X = \{x_1, x_2, \dots, x_n\}$:

$$
\text{MapReduce}(X) = \text{Reduce}\!\Bigl(\, \bigl\{\text{Map}(x_i)\bigr\}_{i=1}^{n} \,\Bigr)
$$

where:
- $\text{Map}: \mathcal{X} \to \mathcal{Y}$ is applied **independently** to each element
- $\text{Reduce}: \mathcal{Y}^n \to \mathcal{Z}$ aggregates all intermediate results into a final output

**Key constraint.** Map operations must be **independent** — no map operation depends on the result of another. This independence is what enables parallelism:

$$
\forall\, i \neq j: \quad \text{Map}(x_i) \perp \text{Map}(x_j)
$$

### 1.10.3.2 Three-Phase Architecture

```
Phase 1: SPLIT            Phase 2: MAP              Phase 3: REDUCE
┌──────────┐          ┌─────────────┐
│          │──chunk₁─▶│  Map Step 1  │──result₁─┐
│          │          └─────────────┘           │    ┌──────────────┐
│  Input   │          ┌─────────────┐           ├───▶│              │
│  Splitter│──chunk₂─▶│  Map Step 2  │──result₂─┤    │  Reduce Step │──▶ Final
│          │          └─────────────┘           ├───▶│              │    Output
│          │          ┌─────────────┐           │    └──────────────┘
│          │──chunk₃─▶│  Map Step 3  │──result₃─┘
└──────────┘          └─────────────┘
```

```python
from typing import TypeVar, Generic
import asyncio

T_Input = TypeVar("T_Input")
T_Mapped = TypeVar("T_Mapped")
T_Output = TypeVar("T_Output")

class MapReduceChain(Generic[T_Input, T_Mapped, T_Output]):
    """Generic Map-Reduce chain with configurable split, map, and reduce."""
    
    def __init__(self,
                 splitter: Callable[[T_Input], list[T_Input]],
                 mapper: Callable[[T_Input, int], T_Mapped],
                 reducer: Callable[[list[T_Mapped]], T_Output],
                 max_concurrency: int = 10):
        self.splitter = splitter
        self.mapper = mapper
        self.reducer = reducer
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _bounded_map(self, chunk: T_Input, index: int) -> T_Mapped:
        async with self.semaphore:
            return await self.mapper(chunk, index)
    
    async def execute(self, input_data: T_Input) -> dict:
        # Phase 1: Split
        chunks = self.splitter(input_data)
        
        # Phase 2: Map (parallel)
        map_results = await asyncio.gather(*[
            self._bounded_map(chunk, i) for i, chunk in enumerate(chunks)
        ], return_exceptions=True)
        
        # Handle map failures
        successful_results = []
        failed_indices = []
        for i, result in enumerate(map_results):
            if isinstance(result, Exception):
                failed_indices.append(i)
            else:
                successful_results.append(result)
        
        # Phase 3: Reduce
        final_output = await self.reducer(successful_results)
        
        return {
            "output": final_output,
            "total_chunks": len(chunks),
            "successful_maps": len(successful_results),
            "failed_maps": failed_indices,
            "parallelism": min(len(chunks), self.semaphore._value)
        }
```

### 1.10.3.3 Hierarchical Reduce

When the number of map results $n$ is too large for a single reduce step's context window, apply **hierarchical (tree) reduction**:

$$
\text{HierReduce}(\{r_1, \dots, r_n\}) = \begin{cases}
\text{Reduce}(\{r_1, \dots, r_n\}) & \text{if } \sum_i |r_i| \leq C_{\max} \\
\text{Reduce}\!\Bigl(\bigl\{\text{HierReduce}(R_1), \dots, \text{HierReduce}(R_k)\bigr\}\Bigr) & \text{otherwise}
\end{cases}
$$

where $\{R_1, \dots, R_k\}$ is a partition of $\{r_1, \dots, r_n\}$ into groups that each fit in the context window.

**Depth of hierarchical reduce.** If each reduce combines $b$ results (branching factor) and there are $n$ total results:

$$
d_{\text{reduce}} = \lceil \log_b n \rceil
$$

**Total LLM calls:**

$$
\text{Calls}_{\text{total}} = \underbrace{n}_{\text{map}} + \underbrace{\frac{n - 1}{b - 1}}_{\text{reduce tree nodes}} \approx n + \frac{n}{b}
$$

```python
class HierarchicalReducer:
    """Tree-structured reduce for large result sets."""
    
    def __init__(self, reduce_fn: Callable[[list[str]], str],
                 max_items_per_reduce: int = 5,
                 context_budget: int = 80000):
        self.reduce_fn = reduce_fn
        self.max_items = max_items_per_reduce
        self.context_budget = context_budget
    
    async def reduce(self, results: list[str]) -> str:
        # Base case: few enough results for a single reduce
        total_tokens = sum(len(r.split()) for r in results)
        if len(results) <= self.max_items and total_tokens <= self.context_budget:
            return await self.reduce_fn(results)
        
        # Recursive case: group and reduce
        groups = self._partition(results)
        
        # Reduce each group in parallel
        group_summaries = await asyncio.gather(*[
            self.reduce_fn(group) for group in groups
        ])
        
        # Recursively reduce the group summaries
        return await self.reduce(list(group_summaries))
    
    def _partition(self, results: list[str]) -> list[list[str]]:
        groups = []
        current_group = []
        current_tokens = 0
        
        for r in results:
            r_tokens = len(r.split())
            if (len(current_group) >= self.max_items or 
                current_tokens + r_tokens > self.context_budget):
                if current_group:
                    groups.append(current_group)
                current_group = [r]
                current_tokens = r_tokens
            else:
                current_group.append(r)
                current_tokens += r_tokens
        
        if current_group:
            groups.append(current_group)
        
        return groups
```

### 1.10.3.4 Concrete Application — Multi-Document Summarization

```python
class MultiDocumentSummarizer:
    """Map-Reduce chain for summarizing a corpus of documents."""
    
    MAP_PROMPT = """Extract the key findings, arguments, and data from this document.
Focus on facts and insights that would be relevant to a comprehensive summary.

Document ({doc_index} of {total_docs}):
{document}

Key Findings (structured list):"""
    
    REDUCE_PROMPT = """Synthesize the following extracted findings from {n_sources} documents
into a coherent, comprehensive summary. Identify common themes, contradictions, 
and unique insights across sources.

Extracted Findings:
{findings}

Comprehensive Synthesis:"""
    
    def __init__(self, llm: "LLMClient", context_window: int = 128000):
        self.llm = llm
        self.context_window = context_window
        
        self.chain = MapReduceChain(
            splitter=lambda docs: docs,  # documents are already split
            mapper=self._map_document,
            reducer=self._reduce_findings,
            max_concurrency=5
        )
        self.hier_reducer = HierarchicalReducer(
            reduce_fn=self._reduce_findings,
            max_items_per_reduce=5
        )
    
    async def _map_document(self, document: str, index: int) -> str:
        prompt = self.MAP_PROMPT.format(
            document=document[:50000],  # truncate if needed
            doc_index=index + 1,
            total_docs="N"
        )
        return await self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=1000)
    
    async def _reduce_findings(self, findings: list[str]) -> str:
        combined = "\n\n---\n\n".join(
            f"Source {i+1}:\n{f}" for i, f in enumerate(findings)
        )
        prompt = self.REDUCE_PROMPT.format(
            n_sources=len(findings),
            findings=combined
        )
        return await self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=2000)
    
    async def summarize(self, documents: list[str]) -> dict:
        result = await self.chain.execute(documents)
        
        # If too many map results, use hierarchical reduce
        if result.get("total_chunks", 0) > 5:
            map_results = result["output"]
            if isinstance(map_results, list):
                final = await self.hier_reducer.reduce(map_results)
                result["output"] = final
        
        return result
```

---

## 1.10.4 Iterative Refinement Chains

### 1.10.4.1 Generate → Critique → Revise Loop

Iterative refinement chains repeatedly improve an output through a three-phase cycle:

$$
\text{Draft} \xrightarrow{\text{Critique}} \text{Feedback} \xrightarrow{\text{Revise}} \text{Improved Draft} \xrightarrow{\text{Critique}} \cdots
$$

**Formal model.** Define the refinement operator $\mathcal{R}$:

$$
y^{(0)} = \text{Generate}(x)
$$

$$
c^{(t)} = \text{Critique}(x, y^{(t)})
$$

$$
y^{(t+1)} = \text{Revise}(x, y^{(t)}, c^{(t)})
$$

The chain iterates until a **convergence criterion** is met.

### 1.10.4.2 Convergence Criteria

**Definition.** The refinement loop converges when the quality improvement between iterations falls below a threshold $\epsilon$:

$$
\|Q(y^{(t+1)}) - Q(y^{(t)})\| < \epsilon
$$

where $Q: \mathcal{Y} \to [0,1]$ is a quality function.

**Practical convergence criteria:**

| Criterion | Formula | Robustness |
|---|---|---|
| **Quality plateau** | $Q(y^{(t+1)}) - Q(y^{(t)}) < \epsilon$ | Good, requires quality function |
| **Output stability** | $\text{sim}(y^{(t+1)}, y^{(t)}) > 1 - \epsilon$ | Good, catches when changes stop |
| **Critic satisfaction** | Critic says "no issues found" | Moderate, may be premature |
| **Max iterations** | $t \geq t_{\max}$ | Guaranteed termination, may under/over-iterate |
| **Budget exhaustion** | $\sum_t \text{cost}(t) \geq B$ | Practical, but quality-agnostic |

**Combined criterion (recommended):**

$$
\text{Stop} \iff (Q(y^{(t)}) \geq Q_{\min}) \;\vee\; (t \geq t_{\max}) \;\vee\; \bigl(|Q(y^{(t)}) - Q(y^{(t-1)})| < \epsilon \;\wedge\; t \geq 2\bigr)
$$

### 1.10.4.3 Quality-Monotonic Refinement Guarantees

A refinement chain is **quality-monotonic** iff:

$$
Q(y^{(t+1)}) \geq Q(y^{(t)}) \quad \forall\, t
$$

**This is NOT guaranteed by default.** LLMs can degrade output quality during revision (introducing new errors while fixing old ones, over-editing, losing nuance).

**Strategies to enforce monotonicity:**

1. **Accept-if-better gate:**

$$
y^{(t+1)}_{\text{final}} = \begin{cases}
y^{(t+1)} & \text{if } Q(y^{(t+1)}) > Q(y^{(t)}) \\
y^{(t)} & \text{otherwise (reject revision)}
\end{cases}
$$

2. **Diff-based revision:** Instead of regenerating the entire output, apply only targeted edits.

3. **Independent evaluator:** Use a different model or method for $Q$ than for generation/revision.

```python
class IterativeRefinementChain:
    """Generate-Critique-Revise loop with convergence control."""
    
    CRITIQUE_PROMPT = """Critically evaluate the following output for the given task.
Identify specific issues and suggest concrete improvements.

Task: {task}
Current Output: {output}

Provide your critique as JSON:
{{"issues": [{{"description": "...", "severity": "low|medium|high", 
  "location": "...", "suggestion": "..."}}], 
  "overall_quality": 0.0-1.0,
  "needs_revision": true/false}}"""
    
    REVISE_PROMPT = """Revise the output to address the identified issues.
Make targeted improvements while preserving what is already good.
Do NOT introduce new content beyond what is needed to fix the issues.

Task: {task}
Current Output: {output}

Issues to address:
{critique}

Revised Output:"""
    
    def __init__(self, llm: "LLMClient", 
                 critic_llm: "LLMClient" = None,
                 max_iterations: int = 3,
                 quality_threshold: float = 0.85,
                 convergence_epsilon: float = 0.02,
                 enforce_monotonicity: bool = True):
        self.llm = llm
        self.critic_llm = critic_llm or llm
        self.max_iter = max_iterations
        self.q_threshold = quality_threshold
        self.epsilon = convergence_epsilon
        self.monotonic = enforce_monotonicity
    
    async def execute(self, task: str, initial_draft: str = None) -> dict:
        # Phase 0: Generate initial draft
        if initial_draft is None:
            current = await self.llm.generate(
                prompt=f"Complete the following task:\n{task}",
                temperature=0.3, max_tokens=3000
            )
        else:
            current = initial_draft
        
        history = [{"iteration": 0, "output_preview": current[:200], "quality": None}]
        prev_quality = 0.0
        best_output = current
        best_quality = 0.0
        
        for iteration in range(1, self.max_iter + 1):
            # Phase 1: Critique
            critique_response = await self.critic_llm.generate(
                prompt=self.CRITIQUE_PROMPT.format(task=task, output=current),
                temperature=0.0, max_tokens=1000
            )
            critique = json.loads(critique_response)
            current_quality = critique.get("overall_quality", 0.5)
            
            history.append({
                "iteration": iteration,
                "quality": current_quality,
                "issues_found": len(critique.get("issues", [])),
                "needs_revision": critique.get("needs_revision", False)
            })
            
            # Track best
            if current_quality > best_quality:
                best_quality = current_quality
                best_output = current
            
            # Check convergence
            if current_quality >= self.q_threshold:
                break
            if not critique.get("needs_revision", True):
                break
            if abs(current_quality - prev_quality) < self.epsilon and iteration >= 2:
                break
            
            # Phase 2: Revise
            issues_text = "\n".join(
                f"- [{iss['severity']}] {iss['description']}: {iss['suggestion']}"
                for iss in critique.get("issues", [])
            )
            
            revised = await self.llm.generate(
                prompt=self.REVISE_PROMPT.format(
                    task=task, output=current, critique=issues_text
                ),
                temperature=0.1, max_tokens=3000
            )
            
            # Enforce monotonicity
            if self.monotonic:
                revised_critique = await self.critic_llm.generate(
                    prompt=self.CRITIQUE_PROMPT.format(task=task, output=revised),
                    temperature=0.0, max_tokens=500
                )
                revised_quality = json.loads(revised_critique).get("overall_quality", 0.0)
                
                if revised_quality >= current_quality:
                    current = revised
                    prev_quality = current_quality
                else:
                    # Reject revision — keep current
                    history[-1]["revision_rejected"] = True
                    prev_quality = current_quality
                    continue
            else:
                current = revised
                prev_quality = current_quality
        
        return {
            "output": best_output if self.monotonic else current,
            "final_quality": best_quality if self.monotonic else current_quality,
            "iterations": len(history) - 1,
            "history": history,
            "converged": (current_quality >= self.q_threshold) or 
                        (abs(current_quality - prev_quality) < self.epsilon)
        }
```

---

## 1.10.5 Ensemble Chains

### 1.10.5.1 Formal Definition

An ensemble chain generates $k$ **independent candidate outputs** and selects or aggregates the best:

$$
y^* = \text{Aggregate}\!\bigl(\, y^{(1)}, y^{(2)}, \dots, y^{(k)} \,\bigr)
$$

where each $y^{(i)}$ may be produced by a different chain variant, different model, or different random seed.

### 1.10.5.2 Aggregation Strategies

#### A. Majority Voting

For discrete outputs (classification, multiple-choice):

$$
y^* = \arg\max_y \sum_{i=1}^{k} \mathbb{1}[y^{(i)} = y]
$$

#### B. Confidence-Weighted Voting

When each candidate has an associated confidence $w_i$:

$$
y^* = \arg\max_y \sum_{i=1}^{k} w_i \cdot \mathbb{1}[y^{(i)} = y]
$$

#### C. Best-of-N Selection

Generate $N$ candidates and select the one with the highest quality score:

$$
y^* = \arg\max_{y^{(i)}} \; Q(y^{(i)})
$$

**Quality improvement.** If individual output quality follows distribution $F(q)$, then the expected quality of the best of $N$:

$$
\mathbb{E}[Q_{\text{best-of-N}}] = \int_0^1 q \cdot N \cdot F(q)^{N-1} \cdot f(q) \, dq
$$

For a uniform distribution on $[0,1]$: $\mathbb{E}[Q_{\text{best-of-N}}] = \frac{N}{N+1}$.

For $N=1$: $0.50$. For $N=3$: $0.75$. For $N=5$: $0.83$. For $N=10$: $0.91$.

#### D. LLM-as-Judge Selection

Use a separate LLM to evaluate all candidates and select the best:

$$
y^* = y^{(j^*)} \quad \text{where} \quad j^* = \arg\max_j \; \text{LLM}_{\text{judge}}(y^{(j)}, x)
$$

#### E. Synthesis Aggregation

Instead of selecting one candidate, **synthesize** a new output that combines the best aspects of all candidates:

$$
y^* = \text{LLM}_{\text{synthesize}}\!\bigl(\, x,\; y^{(1)}, \dots, y^{(k)} \,\bigr)
$$

```python
class EnsembleChain:
    """Execute multiple chain variants and aggregate results."""
    
    def __init__(self, 
                 chain_variants: list[Callable],
                 aggregation: str = "best_of_n",
                 quality_fn: Callable[[str, str], float] = None,
                 judge_llm: "LLMClient" = None):
        self.variants = chain_variants
        self.aggregation = aggregation
        self.quality_fn = quality_fn
        self.judge_llm = judge_llm
    
    async def execute(self, input_data: dict) -> dict:
        # Generate all candidates in parallel
        candidates = await asyncio.gather(*[
            variant(input_data) for variant in self.variants
        ], return_exceptions=True)
        
        # Filter out failures
        valid = [(i, c) for i, c in enumerate(candidates) 
                 if not isinstance(c, Exception)]
        
        if not valid:
            raise RuntimeError("All ensemble variants failed")
        
        task = input_data.get("task", input_data.get("query", ""))
        
        if self.aggregation == "best_of_n":
            return await self._best_of_n(valid, task)
        elif self.aggregation == "majority_vote":
            return self._majority_vote(valid)
        elif self.aggregation == "synthesis":
            return await self._synthesis(valid, task)
        elif self.aggregation == "judge":
            return await self._judge_selection(valid, task)
    
    async def _best_of_n(self, candidates: list, task: str) -> dict:
        scored = []
        for idx, output in candidates:
            score = self.quality_fn(str(output), task) if self.quality_fn else 0.5
            scored.append((score, idx, output))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_output = scored[0]
        
        return {
            "output": best_output,
            "selected_variant": best_idx,
            "quality_score": best_score,
            "all_scores": [(idx, score) for score, idx, _ in scored],
            "n_candidates": len(candidates)
        }
    
    def _majority_vote(self, candidates: list) -> dict:
        from collections import Counter
        outputs = [str(c) for _, c in candidates]
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        
        return {
            "output": most_common,
            "vote_count": count,
            "total_votes": len(candidates),
            "agreement_ratio": count / len(candidates)
        }
    
    async def _synthesis(self, candidates: list, task: str) -> dict:
        candidate_texts = "\n\n---\n\n".join(
            f"Candidate {i+1}:\n{str(output)}" for i, (_, output) in enumerate(candidates)
        )
        
        prompt = (
            f"Given the task and multiple candidate solutions below, "
            f"synthesize the best possible output by combining the strengths "
            f"of each candidate and avoiding their weaknesses.\n\n"
            f"Task: {task}\n\n"
            f"Candidates:\n{candidate_texts}\n\n"
            f"Synthesized Best Output:"
        )
        
        synthesized = await self.judge_llm.generate(
            prompt=prompt, temperature=0.0, max_tokens=3000
        )
        
        return {
            "output": synthesized,
            "n_candidates": len(candidates),
            "aggregation": "synthesis"
        }
    
    async def _judge_selection(self, candidates: list, task: str) -> dict:
        candidate_texts = "\n\n---\n\n".join(
            f"Option {i+1}:\n{str(output)}" for i, (_, output) in enumerate(candidates)
        )
        
        prompt = (
            f"You are evaluating multiple candidate responses to a task.\n\n"
            f"Task: {task}\n\n{candidate_texts}\n\n"
            f"Which option is the BEST? Respond as JSON:\n"
            f'{{"best_option": N, "reasoning": "..."}}'
        )
        
        response = await self.judge_llm.generate(prompt=prompt, temperature=0.0)
        result = json.loads(response)
        best_idx = result["best_option"] - 1
        
        return {
            "output": candidates[min(best_idx, len(candidates)-1)][1],
            "judge_reasoning": result["reasoning"],
            "selected_option": result["best_option"],
            "n_candidates": len(candidates)
        }
```

### 1.10.5.3 Mixture-of-Chains Architecture

A more structured ensemble where different chains are **specialized** for different aspects of the task:

```
                    Input
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
     ┌──────────┐ ┌────────┐ ┌──────────┐
     │ Chain A:  │ │Chain B:│ │ Chain C:  │
     │ Factual   │ │Creative│ │ Analytical│
     │ Accuracy  │ │Writing │ │ Depth     │
     └─────┬────┘ └───┬────┘ └─────┬────┘
           │          │            │
           └──────────┼────────────┘
                      ▼
              ┌───────────────┐
              │   Synthesizer  │
              │  (merge best   │
              │   aspects)     │
              └───────┬───────┘
                      ▼
                 Final Output
```

---

## 1.10.6 Retrieval-Augmented Chains

### 1.10.6.1 RAG as a Chain Step

Retrieval-Augmented Generation (RAG) naturally decomposes into a **three-step chain**:

$$
\text{RAG}(q) = \text{Generate}\!\bigl(\, q,\; \text{Augment}(q, \text{Retrieve}(q)) \,\bigr)
$$

**Step 1 — Retrieve:** Query a knowledge base $\mathcal{K}$ to find relevant documents:

$$
\mathcal{D}_q = \text{TopK}\!\bigl(\, \mathcal{K},\; q,\; k \,\bigr) = \arg\text{top-}k_{d \in \mathcal{K}} \; \text{sim}(e(q), e(d))
$$

**Step 2 — Augment:** Construct a context-augmented prompt:

$$
p_{\text{aug}} = \text{Template}(q, \mathcal{D}_q) = \text{``Context: } \bigoplus_{d \in \mathcal{D}_q} d \text{. Question: } q\text{''}
$$

**Step 3 — Generate:** Produce the answer conditioned on augmented context:

$$
y \sim p_\theta(y \mid p_{\text{aug}})
$$

### 1.10.6.2 Multi-Hop Retrieval Chains

For questions requiring information synthesis from multiple sources, a single retrieval is insufficient. **Multi-hop retrieval** chains iteratively retrieve and reason:

$$
\text{for } t = 1, 2, \dots, T:
$$

$$
q_t = \text{QueryRewrite}(q, y_{<t}, \mathcal{D}_{<t})
$$

$$
\mathcal{D}_t = \text{Retrieve}(q_t)
$$

$$
y_t = \text{Reason}(q, \mathcal{D}_{\leq t}, y_{<t})
$$

Each hop refines the query based on what was learned from previous retrievals.

```python
class MultiHopRAGChain:
    """Iterative retrieval-reasoning chain for complex queries."""
    
    QUERY_REWRITE_PROMPT = """Based on the original question and what we've learned so far,
formulate a follow-up search query to find missing information.

Original question: {question}
Information gathered so far: {gathered}
What we still need to know: {gaps}

Follow-up search query:"""
    
    REASONING_PROMPT = """Answer the question using the provided evidence.
If the evidence is insufficient, indicate what additional information is needed.

Question: {question}
Evidence:
{evidence}

Respond as JSON:
{{"answer": "...", "confidence": 0.0-1.0, "sufficient": true/false,
  "missing_information": ["..."] or null}}"""
    
    def __init__(self, llm: "LLMClient", retriever: "Retriever",
                 max_hops: int = 3, confidence_threshold: float = 0.8):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops
        self.conf_threshold = confidence_threshold
    
    async def execute(self, question: str) -> dict:
        all_evidence = []
        hop_log = []
        current_query = question
        
        for hop in range(self.max_hops):
            # Retrieve
            docs = await self.retriever.search(current_query, k=5)
            all_evidence.extend(docs)
            
            # Reason
            evidence_text = "\n---\n".join(
                f"[Source {i+1}]: {d['content']}" for i, d in enumerate(all_evidence)
            )
            
            reasoning_response = await self.llm.generate(
                prompt=self.REASONING_PROMPT.format(
                    question=question, evidence=evidence_text
                ),
                temperature=0.0
            )
            result = json.loads(reasoning_response)
            
            hop_log.append({
                "hop": hop + 1,
                "query": current_query,
                "docs_retrieved": len(docs),
                "confidence": result["confidence"],
                "sufficient": result["sufficient"]
            })
            
            # Check if we have enough information
            if result["sufficient"] or result["confidence"] >= self.conf_threshold:
                return {
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "hops": hop + 1,
                    "total_sources": len(all_evidence),
                    "hop_log": hop_log
                }
            
            # Rewrite query for next hop
            if result.get("missing_information"):
                gaps = ", ".join(result["missing_information"])
                gathered = result["answer"]
                current_query = await self.llm.generate(
                    prompt=self.QUERY_REWRITE_PROMPT.format(
                        question=question,
                        gathered=gathered,
                        gaps=gaps
                    ),
                    temperature=0.0, max_tokens=100
                )
        
        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "hops": self.max_hops,
            "total_sources": len(all_evidence),
            "exhausted_hops": True,
            "hop_log": hop_log
        }
```

### 1.10.6.3 Adaptive Retrieval (Retrieve Only When Needed)

Not every query requires retrieval. An **adaptive retrieval** chain decides at runtime whether retrieval is necessary:

$$
\text{AdaptiveRAG}(q) = \begin{cases}
\text{DirectAnswer}(q) & \text{if } \text{NeedsRetrieval}(q) = \texttt{False} \\
\text{RAG}(q) & \text{if } \text{NeedsRetrieval}(q) = \texttt{True}
\end{cases}
$$

**Decision criteria for retrieval necessity:**

| Signal | Indicates Retrieval Needed | Method |
|---|---|---|
| Query mentions specific facts/dates/numbers | Yes | NER + pattern matching |
| Query asks about recent events | Yes | Temporal keyword detection |
| Query is opinion/creative/hypothetical | No | Intent classification |
| Model uncertainty is high | Yes | Confidence-based routing |
| Query references specific documents | Yes | Reference detection |

```python
class AdaptiveRAGChain:
    """Retrieve only when the model lacks sufficient knowledge."""
    
    def __init__(self, llm: "LLMClient", retriever: "Retriever",
                 confidence_router: "ConfidenceRouter"):
        self.llm = llm
        self.retriever = retriever
        self.router = confidence_router
    
    async def execute(self, question: str) -> dict:
        # Step 1: Attempt direct answer
        direct = await self.llm.generate(
            prompt=f"Answer if you are confident. If unsure, say 'UNCERTAIN'.\n\nQ: {question}",
            temperature=0.0, logprobs=True
        )
        
        # Step 2: Check confidence
        confidence = await self.router.estimate_confidence_logprob(
            f"Q: {question}"
        )
        
        if confidence > 0.85 and "UNCERTAIN" not in direct.upper():
            return {
                "answer": direct,
                "source": "direct_knowledge",
                "retrieval_used": False,
                "confidence": confidence
            }
        
        # Step 3: Retrieval-augmented path
        docs = await self.retriever.search(question, k=5)
        context = "\n\n".join(d["content"] for d in docs)
        
        augmented_answer = await self.llm.generate(
            prompt=(
                f"Answer based on the following context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            ),
            temperature=0.0
        )
        
        return {
            "answer": augmented_answer,
            "source": "retrieval_augmented",
            "retrieval_used": True,
            "num_sources": len(docs),
            "direct_confidence_was": confidence
        }
```

### 1.10.6.4 Query Rewriting Chains

Raw user queries are often suboptimal for retrieval (vague, conversational, multi-faceted). A **query rewriting chain** transforms the user query into one or more optimized search queries:

$$
q_{\text{user}} \xrightarrow{\text{Rewrite}} \{q_1^{\text{search}}, q_2^{\text{search}}, \dots, q_m^{\text{search}}\}
$$

```python
class QueryRewriter:
    """Transform user queries into optimized search queries."""
    
    REWRITE_PROMPT = """Given the user's question, generate {n} optimized search queries
that would retrieve the most relevant information. Consider:
1. Different phrasings of the same concept
2. Specific technical terms vs. general language
3. Breaking compound questions into atomic searches
4. Adding relevant context terms

User question: {question}

Generate {n} search queries as a JSON list: ["query1", "query2", ...]"""
    
    def __init__(self, llm: "LLMClient"):
        self.llm = llm
    
    async def rewrite(self, question: str, n: int = 3) -> list[str]:
        response = await self.llm.generate(
            prompt=self.REWRITE_PROMPT.format(question=question, n=n),
            temperature=0.3
        )
        queries = json.loads(response)
        return queries if isinstance(queries, list) else [question]
```

---

## 1.10.7 Tool-Augmented Chains

### 1.10.7.1 Tool Invocation as a Chain Step

Tools extend the chain's capabilities beyond what the LLM can do alone. A tool call is a chain step where the LLM **generates a structured action** and a tool executor **performs the action deterministically**.

**Formal model:**

$$
\text{ToolStep}(S_t) = \begin{cases}
(a_t, \text{args}_t) = \text{LLM}_{\text{plan}}(S_t) & \text{(plan the tool call)} \\
o_t = \text{Tool}_{a_t}(\text{args}_t) & \text{(execute the tool)} \\
S_{t+1} = S_t \cup \{(a_t, o_t)\} & \text{(update state with observation)}
\end{cases}
$$

### 1.10.7.2 Tool Registry and Type-Safe Invocation

```python
from typing import Any, get_type_hints
from inspect import signature
import json

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict           # JSON Schema for parameters
    function: Callable
    return_type: str
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0

class ToolRegistry:
    """Registry of available tools with schema generation."""
    
    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}
    
    def register(self, func: Callable, description: str = "",
                 requires_confirmation: bool = False) -> ToolDefinition:
        """Register a function as a tool, auto-generating schema."""
        hints = get_type_hints(func)
        sig = signature(func)
        
        params = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            param_type = hints.get(name, str)
            json_type = self._python_to_json_type(param_type)
            params["properties"][name] = {"type": json_type}
            if param.default is param.empty:
                params["required"].append(name)
        
        tool_def = ToolDefinition(
            name=func.__name__,
            description=description or func.__doc__ or "",
            parameters=params,
            function=func,
            return_type=self._python_to_json_type(hints.get("return", str)),
            requires_confirmation=requires_confirmation
        )
        
        self.tools[func.__name__] = tool_def
        return tool_def
    
    def get_schemas_for_llm(self) -> list[dict]:
        """Generate tool schemas in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    async def execute(self, tool_name: str, arguments: dict) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        try:
            result = await asyncio.wait_for(
                self._call(tool.function, arguments),
                timeout=tool.timeout_seconds
            )
            return {"status": "success", "result": result}
        except asyncio.TimeoutError:
            return {"status": "error", "error": f"Tool timed out after {tool.timeout_seconds}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _call(self, func: Callable, args: dict) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        return func(**args)
    
    @staticmethod
    def _python_to_json_type(py_type) -> str:
        mapping = {str: "string", int: "integer", float: "number",
                   bool: "boolean", list: "array", dict: "object"}
        return mapping.get(py_type, "string")


# --- Register tools ---
registry = ToolRegistry()

@registry.register(description="Search the web for information")
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    pass  # implementation

@registry.register(description="Execute Python code in a sandboxed environment")
def execute_python(code: str) -> dict:
    """Run Python code and return output."""
    pass  # implementation

@registry.register(description="Query a SQL database", requires_confirmation=True)
def sql_query(query: str, database: str = "default") -> list[dict]:
    """Execute a SQL query and return results."""
    pass  # implementation
```

### 1.10.7.3 Tool Result Parsing and Integration

After a tool executes, its result must be **parsed, validated, and integrated** back into the chain's context:

```python
class ToolIntegrationStep:
    """Parse tool results and integrate into the chain context."""
    
    INTEGRATION_PROMPT = """The following tool was called and produced the result below.
Interpret this result in the context of the original task.

Original Task: {task}
Tool Called: {tool_name}({arguments})
Tool Result:
{result}

Based on this result:
1. What key information was obtained?
2. Does this answer the original question, or do we need additional steps?
3. Are there any issues with the result (errors, empty, unexpected)?

Analysis:"""
    
    async def integrate(self, task: str, tool_name: str,
                       arguments: dict, result: Any,
                       llm: "LLMClient") -> dict:
        result_str = json.dumps(result, indent=2, default=str)[:5000]
        
        analysis = await llm.generate(
            prompt=self.INTEGRATION_PROMPT.format(
                task=task,
                tool_name=tool_name,
                arguments=json.dumps(arguments),
                result=result_str
            ),
            temperature=0.0
        )
        
        return {
            "tool_name": tool_name,
            "raw_result": result,
            "interpretation": analysis,
            "result_token_count": len(result_str.split())
        }
```

---

## 1.10.8 Multi-Model Chains

### 1.10.8.1 Different LLMs for Different Steps

Multi-model chains assign **specialized models** to each step based on the step's requirements:

$$
y_i = p_{\theta_i}(y \mid \text{Render}(S_i)) \quad \text{where } \theta_i \text{ may differ per step}
$$

**Model assignment criteria:**

| Step Characteristic | Optimal Model Profile |
|---|---|
| Simple classification/routing | Small, fast ($\leq$ 8B parameters) |
| Complex reasoning | Large, capable (GPT-4o, Claude Sonnet) |
| Code generation | Code-specialized (Codex, DeepSeek-Coder) |
| Creative writing | Creative-tuned (Claude, GPT-4o at high temperature) |
| Structured extraction | Instruction-tuned with structured output support |
| Safety evaluation | Safety-specialized or separate classifier |
| Multilingual | Multilingual-optimized model |

### 1.10.8.2 Vision-Language Chains

Cross-modal chains where visual inputs are processed by vision models and results feed into language-model reasoning:

```
Image → [Vision Model] → Caption/Description → [Language Model] → Analysis
  │                                                                  │
  └──── [OCR Tool] → Extracted Text ─────────────────────────────────┘
```

```python
class VisionLanguageChain:
    """Cross-modal chain: image → visual understanding → text reasoning."""
    
    def __init__(self, vision_llm: "MultimodalLLM", 
                 text_llm: "LLMClient",
                 ocr_tool: Callable = None):
        self.vision = vision_llm
        self.text = text_llm
        self.ocr = ocr_tool
    
    async def analyze_image(self, image_data: bytes, 
                           question: str) -> dict:
        # Phase 1: Visual understanding (multimodal model)
        visual_analysis = await self.vision.generate(
            prompt=(
                "Describe this image in detail. Include:\n"
                "1. What objects/entities are present\n"
                "2. Any text visible in the image\n"
                "3. Spatial relationships between elements\n"
                "4. Any data, charts, or structured information"
            ),
            images=[image_data],
            temperature=0.0
        )
        
        # Phase 1b: OCR (parallel with visual analysis)
        extracted_text = ""
        if self.ocr:
            extracted_text = await self.ocr(image_data)
        
        # Phase 2: Text-based reasoning (language model)
        reasoning = await self.text.generate(
            prompt=(
                f"Based on the following image analysis, answer the question.\n\n"
                f"Image Description:\n{visual_analysis}\n\n"
                f"Extracted Text (OCR):\n{extracted_text}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            ),
            temperature=0.0
        )
        
        return {
            "visual_analysis": visual_analysis,
            "extracted_text": extracted_text,
            "answer": reasoning
        }
```

### 1.10.8.3 Speech-Text-Speech Chains

End-to-end chains spanning speech and text modalities:

```
Audio → [ASR] → Transcript → [NLP Chain] → Response Text → [TTS] → Audio
```

```python
class SpeechToSpeechChain:
    """End-to-end speech processing chain: ASR → NLP → TTS."""
    
    def __init__(self, asr_model, nlp_chain: Callable, tts_model):
        self.asr = asr_model       # Speech-to-Text
        self.nlp = nlp_chain       # Text processing chain
        self.tts = tts_model       # Text-to-Speech
    
    async def process(self, audio_input: bytes) -> dict:
        # Phase 1: Automatic Speech Recognition
        transcript = await self.asr.transcribe(audio_input)
        
        # Phase 2: NLP Chain (can be any complex chain)
        nlp_result = await self.nlp({"query": transcript["text"]})
        
        # Phase 3: Text-to-Speech
        response_text = nlp_result.get("output", "")
        audio_output = await self.tts.synthesize(response_text)
        
        return {
            "transcript": transcript,
            "nlp_result": nlp_result,
            "response_text": response_text,
            "audio_output": audio_output
        }
```

---

## 1.10.9 Self-Adaptive Chains

### 1.10.9.1 Chains That Modify Their Own Structure

The most advanced pattern: chains that **dynamically alter their own topology** at runtime based on intermediate results. This is the bridge between static prompt chains and fully autonomous agents.

**Formal model.** A self-adaptive chain is a function:

$$
\mathcal{A}(x) = \text{Execute}\!\bigl(\, \text{MetaChain}(x) \,\bigr)
$$

where $\text{MetaChain}$ is itself a chain that **produces a chain specification** as its output:

$$
\text{MetaChain}(x) \to G = (V, E, \Sigma) \quad \text{(the execution plan)}
$$

$$
\text{Execute}(G, x) \to y \quad \text{(run the generated plan)}
$$

### 1.10.9.2 Dynamic Step Insertion/Deletion

Based on intermediate results, the chain can:
- **Insert** additional steps (e.g., "this needs more research, add a retrieval step")
- **Delete** planned steps (e.g., "we already have enough information, skip analysis")
- **Replace** steps with alternatives (e.g., "the API is down, switch to cached data")

```python
class AdaptiveChainExecutor:
    """Chain that modifies its own structure at runtime."""
    
    ADAPTATION_PROMPT = """Given the current chain state, decide if the planned 
next steps should be modified.

Original task: {task}
Completed steps: {completed}
Current result quality: {quality}
Remaining planned steps: {remaining}

Should we:
A) PROCEED with the next planned step
B) INSERT an additional step before the next one
C) SKIP the next planned step
D) REPLACE the next step with a different approach
E) TERMINATE early (output is sufficient)

Respond as JSON:
{{"decision": "A|B|C|D|E", "reasoning": "...",
  "inserted_step": {{...}} or null,
  "replacement_step": {{...}} or null}}"""
    
    def __init__(self, llm: "LLMClient", 
                 step_library: dict[str, Callable],
                 quality_fn: Callable):
        self.llm = llm
        self.step_library = step_library
        self.quality_fn = quality_fn
    
    async def execute(self, task: str, 
                      initial_plan: list[dict]) -> dict:
        plan = list(initial_plan)  # mutable copy
        completed_steps = []
        state = {"task": task}
        current_output = None
        adaptations = []
        
        step_idx = 0
        while step_idx < len(plan):
            current_step = plan[step_idx]
            
            # Adaptive decision point
            quality = self.quality_fn(str(current_output)) if current_output else 0.0
            adaptation = await self._decide_adaptation(
                task=task,
                completed=completed_steps,
                quality=quality,
                remaining=plan[step_idx:]
            )
            
            adaptations.append({
                "before_step": step_idx,
                "decision": adaptation["decision"]
            })
            
            if adaptation["decision"] == "E":  # TERMINATE
                break
            elif adaptation["decision"] == "C":  # SKIP
                step_idx += 1
                continue
            elif adaptation["decision"] == "B":  # INSERT
                inserted = adaptation.get("inserted_step", {})
                plan.insert(step_idx, inserted)
                # Don't increment — execute the inserted step
                continue
            elif adaptation["decision"] == "D":  # REPLACE
                plan[step_idx] = adaptation.get("replacement_step", current_step)
                # Don't increment — execute the replacement
                continue
            
            # A) PROCEED — execute the step
            step_fn = self.step_library.get(current_step.get("type", "llm"))
            if step_fn:
                current_output = await step_fn({**state, **current_step})
                state["latest_output"] = current_output
            
            completed_steps.append({
                "step": current_step,
                "output_preview": str(current_output)[:200]
            })
            
            step_idx += 1
        
        return {
            "output": current_output,
            "steps_executed": len(completed_steps),
            "original_plan_length": len(initial_plan),
            "final_plan_length": len(plan),
            "adaptations": adaptations
        }
    
    async def _decide_adaptation(self, task, completed, 
                                   quality, remaining) -> dict:
        response = await self.llm.generate(
            prompt=self.ADAPTATION_PROMPT.format(
                task=task,
                completed=json.dumps(completed[-3:], default=str),  # last 3
                quality=quality,
                remaining=json.dumps(remaining[:3], default=str)  # next 3
            ),
            temperature=0.0
        )
        return json.loads(response)
```

### 1.10.9.3 Meta-Chain: A Chain That Constructs and Executes Sub-Chains

The ultimate generalization: a **meta-chain** where the first step generates a complete chain specification, and subsequent steps execute that specification.

```
Input → [Planner: Generate Chain Spec] → [Validator: Check Spec] 
    → [Executor: Run Generated Chain] → [Evaluator: Assess Output] → Output
```

**Formal model:**

$$
\text{MetaChain}(x) = \text{Evaluate}\!\Bigl(\, \text{Execute}\!\bigl(\, \text{Validate}(\text{Plan}(x))\bigr) \,\Bigr)
$$

This is a **two-level architecture**:

- **Level 1 (Meta):** Plans, validates, and orchestrates
- **Level 0 (Object):** The dynamically generated chain that actually solves the task

```python
class MetaChainOrchestrator:
    """A chain that generates, validates, and executes sub-chains."""
    
    PLAN_PROMPT = """Design a step-by-step chain to accomplish this task.
For each step, specify:
- step_id: unique identifier
- type: one of [llm_call, tool_call, code_execute, retrieval, conditional]
- description: what this step does
- prompt_template: the prompt to use (with {variable} placeholders)
- dependencies: list of step_ids this step depends on
- output_key: key name for this step's output

Available tools: {tools}

Task: {task}

Respond with a JSON chain specification:
{{"steps": [...], "final_output_key": "..."}}"""
    
    def __init__(self, planner_llm: "LLMClient",
                 executor_llm: "LLMClient",
                 tool_registry: ToolRegistry):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tool_registry
    
    async def run(self, task: str) -> dict:
        # Level 1, Step 1: PLAN
        plan_response = await self.planner.generate(
            prompt=self.PLAN_PROMPT.format(
                task=task,
                tools=json.dumps(self.tools.get_schemas_for_llm(), indent=2)
            ),
            temperature=0.0
        )
        chain_spec = json.loads(plan_response)
        
        # Level 1, Step 2: VALIDATE
        validation = self._validate_chain_spec(chain_spec)
        if not validation["valid"]:
            # Re-plan with feedback
            chain_spec = await self._replan(task, chain_spec, validation)
        
        # Level 1, Step 3: EXECUTE (Level 0 chain)
        execution_result = await self._execute_chain(chain_spec, task)
        
        # Level 1, Step 4: EVALUATE
        evaluation = await self._evaluate_result(task, execution_result)
        
        return {
            "output": execution_result.get(chain_spec.get("final_output_key", "result")),
            "chain_spec": chain_spec,
            "validation": validation,
            "execution_log": execution_result,
            "evaluation": evaluation
        }
    
    def _validate_chain_spec(self, spec: dict) -> dict:
        """Validate chain specification for structural correctness."""
        issues = []
        step_ids = {s["step_id"] for s in spec.get("steps", [])}
        
        for step in spec.get("steps", []):
            # Check dependencies reference valid step IDs
            for dep in step.get("dependencies", []):
                if dep not in step_ids:
                    issues.append(f"Step {step['step_id']} depends on unknown step {dep}")
            
            # Check for required fields
            for field in ["step_id", "type", "description"]:
                if field not in step:
                    issues.append(f"Step missing required field: {field}")
            
            # Check tool references
            if step.get("type") == "tool_call":
                tool_name = step.get("tool_name", "")
                if tool_name not in self.tools.tools:
                    issues.append(f"Unknown tool: {tool_name}")
        
        # Check for cycles (topological sort)
        try:
            self._topological_sort(spec.get("steps", []))
        except ValueError as e:
            issues.append(f"Cyclic dependency: {e}")
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    async def _execute_chain(self, spec: dict, task: str) -> dict:
        """Execute the generated chain specification."""
        steps = spec.get("steps", [])
        ordered = self._topological_sort(steps)
        results = {"task": task}
        
        for step in ordered:
            step_type = step.get("type", "llm_call")
            
            if step_type == "llm_call":
                # Fill in template variables from prior results
                prompt = step.get("prompt_template", step["description"])
                for key, value in results.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value)[:2000])
                
                output = await self.executor.generate(
                    prompt=prompt, temperature=0.1, max_tokens=2000
                )
                results[step["output_key"]] = output
            
            elif step_type == "tool_call":
                args = step.get("arguments", {})
                # Resolve variable references in arguments
                resolved_args = {}
                for k, v in args.items():
                    if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                        var_name = v[1:-1]
                        resolved_args[k] = results.get(var_name, v)
                    else:
                        resolved_args[k] = v
                
                tool_result = await self.tools.execute(
                    step.get("tool_name", ""), resolved_args
                )
                results[step["output_key"]] = tool_result
            
            elif step_type == "code_execute":
                # Sandboxed code execution
                code = step.get("code", "")
                for key, value in results.items():
                    code = code.replace(f"{{{key}}}", str(value))
                # Execute in sandbox...
                results[step["output_key"]] = f"[code execution result]"
        
        return results
    
    @staticmethod
    def _topological_sort(steps: list[dict]) -> list[dict]:
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = {s["step_id"]: 0 for s in steps}
        id_to_step = {s["step_id"]: s for s in steps}
        
        for step in steps:
            for dep in step.get("dependencies", []):
                graph[dep].append(step["step_id"])
                in_degree[step["step_id"]] += 1
        
        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        ordered = []
        
        while queue:
            current = queue.popleft()
            ordered.append(id_to_step[current])
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(ordered) != len(steps):
            raise ValueError("Cyclic dependency detected")
        return ordered
    
    async def _evaluate_result(self, task: str, result: dict) -> dict:
        """Evaluate whether the chain's output satisfies the task."""
        output_keys = [k for k in result if k != "task"]
        output_summary = json.dumps(
            {k: str(result[k])[:500] for k in output_keys[-3:]},
            indent=2
        )
        
        eval_response = await self.planner.generate(
            prompt=(
                f"Evaluate whether this result satisfies the original task.\n\n"
                f"Task: {task}\n\n"
                f"Result:\n{output_summary}\n\n"
                f"Score (0-1) and brief justification as JSON:"
            ),
            temperature=0.0
        )
        return json.loads(eval_response)
```

### 1.10.9.4 Learned Chain Topologies

The frontier of self-adaptive chains: **learning optimal chain structures from data** rather than hand-designing them.

**Approach 1 — Reinforcement Learning over Chain Structures.** Treat chain topology as an action space and task success as reward:

$$
\pi^*(a_t \mid S_t) = \arg\max_\pi \; \mathbb{E}\!\left[\sum_{t=0}^{T} \gamma^t r_t \right]
$$

where $a_t$ is the choice of which step to add/execute next, and $r_t$ reflects quality, cost, and latency.

**Approach 2 — Bayesian Optimization over Chain Hyperparameters.** Optimize the chain's structural parameters (number of steps, which steps to include, model assignments) using Bayesian optimization with a Gaussian Process surrogate:

$$
\theta^* = \arg\max_\theta \; \alpha(\theta) = \mu(\theta) + \kappa \cdot \sigma(\theta)
$$

where $\alpha$ is the acquisition function (Upper Confidence Bound), $\mu$ is the predicted quality, and $\sigma$ is the uncertainty.

**Approach 3 — Evolutionary Search.** Maintain a population of chain topologies, evaluate each on a benchmark, and evolve through mutation (add/remove/swap steps) and crossover:

```python
class ChainTopologyEvolver:
    """Evolve chain structures through mutation and selection."""
    
    @dataclass
    class ChainGenome:
        steps: list[dict]
        fitness: float = 0.0
    
    def __init__(self, step_library: list[dict],
                 fitness_fn: Callable[[list[dict]], float],
                 population_size: int = 20):
        self.step_library = step_library
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
    
    def mutate(self, genome: "ChainTopologyEvolver.ChainGenome") -> "ChainTopologyEvolver.ChainGenome":
        import random
        steps = list(genome.steps)
        mutation = random.choice(["add", "remove", "swap", "modify"])
        
        if mutation == "add" and len(steps) < 15:
            new_step = random.choice(self.step_library)
            pos = random.randint(0, len(steps))
            steps.insert(pos, {**new_step})
        elif mutation == "remove" and len(steps) > 2:
            pos = random.randint(0, len(steps) - 1)
            steps.pop(pos)
        elif mutation == "swap" and len(steps) >= 2:
            i, j = random.sample(range(len(steps)), 2)
            steps[i], steps[j] = steps[j], steps[i]
        elif mutation == "modify" and steps:
            pos = random.randint(0, len(steps) - 1)
            steps[pos] = random.choice(self.step_library)
        
        return self.ChainGenome(steps=steps)
    
    async def evolve(self, generations: int = 10) -> "ChainTopologyEvolver.ChainGenome":
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            n_steps = random.randint(2, 8)
            steps = random.choices(self.step_library, k=n_steps)
            population.append(self.ChainGenome(steps=steps))
        
        for gen in range(generations):
            # Evaluate fitness
            for genome in population:
                genome.fitness = await self.fitness_fn(genome.steps)
            
            # Selection (top 50%)
            population.sort(key=lambda g: g.fitness, reverse=True)
            survivors = population[:self.pop_size // 2]
            
            # Reproduction
            offspring = []
            for _ in range(self.pop_size - len(survivors)):
                parent = random.choice(survivors)
                child = self.mutate(parent)
                offspring.append(child)
            
            population = survivors + offspring
        
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population[0]
```

---

## Summary: Advanced Technique Selection Matrix

| Technique | Best For | Complexity | Cost Multiplier | Key Risk |
|---|---|---|---|---|
| **CoT as Micro-Chain** | Short reasoning tasks | Very Low | 1× | Unverifiable intermediate steps |
| **Recursive Chains** | Hierarchical problems (summarization, decomposition) | Medium | $\mathcal{O}(\log n)$× | Stack overflow, non-termination |
| **Map-Reduce** | Large inputs decomposable into independent parts | Medium | $n$× (parallelizable) | Information loss at reduce boundaries |
| **Iterative Refinement** | Quality-critical outputs needing polish | Medium | $t_{\max}$× | Non-monotonic quality, divergence |
| **Ensemble Chains** | High-reliability tasks | Low–Med | $k$× | Expensive; diminishing returns past $k \approx 5$ |
| **RAG Chains** | Knowledge-intensive tasks | Medium | 1.5–3× | Retrieval quality bottleneck |
| **Tool-Augmented** | Tasks requiring computation, search, or external actions | Medium–High | Variable | Tool errors, injection attacks |
| **Multi-Model** | Tasks spanning modalities or requiring specialized models | High | Variable | Integration complexity, format mismatches |
| **Self-Adaptive** | Novel tasks with unpredictable structure | Very High | Unpredictable | Instability, runaway cost, debugging difficulty |