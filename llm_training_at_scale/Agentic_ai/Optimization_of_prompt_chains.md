# 1.9 Optimization of Prompt Chains

> **Scope.** Once a chain is functionally correct and robust, optimization reduces its latency, cost, and error rate — potentially by orders of magnitude. The three optimization axes (latency, cost, quality) are typically in tension, demanding multi-objective optimization. This section provides a rigorous treatment of each axis with production-ready patterns and formal cost models.

---

## 1.9.1 Latency Optimization

### 1.9.1.1 Latency Model

The total latency of a chain is determined by its **critical path** — the longest sequence of dependent steps:

$$
L_{\text{total}} = \max_{\pi \in \text{Paths}(G)} \sum_{i \in \pi} l_i
$$

where $l_i$ is the latency of step $i$. Steps not on the critical path can execute in parallel without affecting total latency.

**Per-step latency decomposition:**

$$
l_i = l_i^{\text{network}} + l_i^{\text{TTFT}} + l_i^{\text{generation}} + l_i^{\text{processing}}
$$

| Component | Description | Typical Range |
|---|---|---|
| $l_i^{\text{network}}$ | Network round-trip time to API | 10–100 ms |
| $l_i^{\text{TTFT}}$ | Time to first token (prompt processing) | 100–5000 ms (scales with prompt length) |
| $l_i^{\text{generation}}$ | Token generation time: $|y_i| / \text{TPS}$ | 200–10000 ms |
| $l_i^{\text{processing}}$ | Post-processing (parsing, validation, tool execution) | 1–1000 ms |

### 1.9.1.2 Parallelization of Independent Steps

Any two steps $s_i, s_j$ with no dependency between them can execute concurrently:

$$
(s_i, s_j) \text{ parallelizable} \iff s_i \notin \text{ancestors}(s_j) \wedge s_j \notin \text{ancestors}(s_i)
$$

**Parallel execution groups** are computed via topological-level assignment:

```python
import asyncio
from collections import defaultdict

class ParallelChainExecutor:
    """Execute chain steps with maximum parallelism."""
    
    def __init__(self, steps: dict[str, Callable],
                 dependencies: dict[str, list[str]]):
        self.steps = steps
        self.deps = dependencies
    
    def compute_levels(self) -> list[list[str]]:
        """Assign steps to parallel execution levels."""
        in_degree = {s: 0 for s in self.steps}
        adj = defaultdict(list)
        
        for step, deps in self.deps.items():
            for dep in deps:
                adj[dep].append(step)
                in_degree[step] += 1
        
        levels = []
        queue = [s for s, d in in_degree.items() if d == 0]
        
        while queue:
            levels.append(queue)
            next_queue = []
            for step in queue:
                for neighbor in adj[step]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
        
        return levels
    
    async def execute(self, initial_state: dict) -> dict:
        levels = self.compute_levels()
        results = {}
        state = {**initial_state}
        
        for level_idx, level in enumerate(levels):
            # Execute all steps in this level concurrently
            tasks = {}
            for step_name in level:
                step_fn = self.steps[step_name]
                dep_results = {d: results[d] for d in self.deps.get(step_name, [])}
                tasks[step_name] = step_fn({**state, **dep_results})
            
            level_results = await asyncio.gather(
                *[tasks[name] for name in tasks],
                return_exceptions=True
            )
            
            for name, result in zip(tasks.keys(), level_results):
                if isinstance(result, Exception):
                    results[name] = {"error": str(result)}
                else:
                    results[name] = result
        
        return results
```

**Speedup analysis.** If a chain has $n$ steps and the critical path has $k$ steps, the theoretical maximum speedup from parallelization:

$$
\text{Speedup}_{\max} = \frac{\sum_{i=1}^{n} l_i}{\sum_{i \in \text{critical path}} l_i} = \frac{n}{k} \quad \text{(for equal-duration steps)}
$$

### 1.9.1.3 Streaming Between Steps

Instead of waiting for step $i$ to complete before starting step $i{+}1$, **stream** partial output from step $i$ to step $i{+}1$ as it is generated.

**Applicable when:** Step $i{+}1$ can begin processing with a prefix of step $i$'s output (e.g., step $i$ generates a list of items and step $i{+}1$ processes each independently).

```python
async def streaming_pipeline(source_llm, processor_fn, prompt):
    """Stream tokens from source LLM into a processor as they arrive."""
    buffer = ""
    results = []
    
    async for token in source_llm.stream(prompt=prompt):
        buffer += token
        
        # Check if we have a complete processable unit (e.g., a line)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                result = await processor_fn(line)
                results.append(result)
    
    # Process any remaining content
    if buffer.strip():
        results.append(await processor_fn(buffer.strip()))
    
    return results
```

### 1.9.1.4 Speculative Execution

When the most likely routing branch is predictable, **speculatively execute** that branch while the routing decision is still being computed. If the speculation is correct, latency is saved; if wrong, discard the speculative result.

$$
L_{\text{speculative}} = \begin{cases}
\max(l_{\text{route}}, l_{\text{branch}}) & \text{if speculation correct (saves } l_{\text{branch}} \text{)} \\
l_{\text{route}} + l_{\text{correct branch}} & \text{if speculation wrong (wastes } l_{\text{branch}} \text{)}
\end{cases}
$$

**Expected latency:**

$$
\mathbb{E}[L_{\text{spec}}] = p_{\text{correct}} \cdot \max(l_r, l_b) + (1 - p_{\text{correct}}) \cdot (l_r + l_b)
$$

Speculation is beneficial when $p_{\text{correct}}$ is high and $l_b$ is large.

### 1.9.1.5 Caching of Deterministic Steps

If a step is deterministic ($\text{temperature} = 0$, no external randomness), cache its output keyed by its input hash:

$$
\text{cache}[h(x_i)] = y_i \implies \text{next call with } h(x_i) \text{ returns } y_i \text{ in } \mathcal{O}(1)
$$

```python
import hashlib
from functools import lru_cache

class StepCache:
    """Cache deterministic step outputs to avoid redundant LLM calls."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self._cache: dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def _hash_input(self, prompt: str, params: dict) -> str:
        content = json.dumps({"prompt": prompt, **params}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, params: dict) -> Any | None:
        key = self._hash_input(prompt, params)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, prompt: str, params: dict, value: Any):
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        key = self._hash_input(prompt, params)
        self._cache[key] = (value, time.time())
```

---

## 1.9.2 Cost Optimization

### 1.9.2.1 Cost Model

The total cost of a chain execution:

$$
C_{\text{total}} = \sum_{i=1}^{n} c_{\text{model}_i} \cdot \bigl(\, r_{\text{in}} \cdot |p_i| + r_{\text{out}} \cdot |y_i| \,\bigr)
$$

where $c_{\text{model}_i}$ is the per-token cost of the model used at step $i$, $r_{\text{in}}$ and $r_{\text{out}}$ are the input/output pricing ratios, $|p_i|$ is the prompt token count, and $|y_i|$ is the output token count.

**Cost breakdown for a typical chain:**

| Component | Typical % of Total Cost |
|---|---|
| Input tokens (prompts + context) | 30–60% |
| Output tokens (generations) | 20–50% |
| Retries (wasted calls) | 5–20% |
| Verification/critic steps | 10–25% |

### 1.9.2.2 Model Selection Per Step

The most impactful cost optimization: use the **cheapest model that achieves acceptable quality** for each step.

$$
m_i^* = \arg\min_{m \in \mathcal{M}} \; \text{cost}(m) \quad \text{s.t.} \quad \mathbb{E}[Q_i(m)] \geq q_i^{\min}
$$

| Step Type | Recommended Model Tier | Rationale |
|---|---|---|
| Routing/classification | Small/fast (GPT-4o-mini, Haiku) | Low complexity, structured output |
| Data extraction | Small/fast | Pattern matching, not creative |
| Complex reasoning | Large (GPT-4o, Sonnet) | Requires deep reasoning |
| Code generation | Large | Correctness is critical |
| Summarization | Medium | Moderate reasoning required |
| Formatting/translation | Small/fast | Mechanical transformation |
| Evaluation/critique | Large (different from generator) | Must catch generator's errors |

### 1.9.2.3 Token Usage Minimization

**Prompt compression** reduces $|p_i|$ by removing redundant or low-information tokens:

| Technique | Token Savings | Quality Impact |
|---|---|---|
| Remove verbose instructions | 10–30% | Low if concise is still clear |
| Remove few-shot examples after fine-tuning | 30–60% | Requires fine-tuned model |
| Compress history (§1.4.2.5) | 50–90% | Moderate, task-dependent |
| Shorten field names in JSON | 5–15% | Very low |
| Use abbreviations in system prompt | 5–20% | Low if model understands |

### 1.9.2.4 Caching and Memoization

For sub-chains or steps that are frequently invoked with identical or similar inputs, caching eliminates redundant computation entirely:

$$
C_{\text{saved}} = \text{hit\_rate} \cdot C_{\text{per\_call}} \cdot N_{\text{calls}}
$$

### 1.9.2.5 Early Termination

If a chain's intermediate output already satisfies the quality threshold, **skip remaining steps**:

$$
\text{if } Q(y_t) \geq Q_{\text{sufficient}} \text{ and } t < n: \quad \text{return } y_t, \text{ skip } s_{t+1}, \dots, s_n
$$

This converts the fixed cost $C = \sum_{i=1}^{n} c_i$ into a variable cost $C = \sum_{i=1}^{t} c_i$ where $t \leq n$.

```python
class EarlyTerminationExecutor:
    """Execute chain with early exit when quality is sufficient."""
    
    def __init__(self, steps: list[Callable],
                 quality_fn: Callable[[str], float],
                 sufficient_quality: float = 0.85):
        self.steps = steps
        self.quality_fn = quality_fn
        self.threshold = sufficient_quality
    
    async def execute(self, state: dict) -> dict:
        output = None
        steps_executed = 0
        
        for i, step in enumerate(self.steps):
            output = await step(state)
            steps_executed = i + 1
            state["latest_output"] = output
            
            quality = self.quality_fn(str(output))
            
            if quality >= self.threshold:
                return {
                    "output": output,
                    "steps_executed": steps_executed,
                    "total_steps": len(self.steps),
                    "early_terminated": steps_executed < len(self.steps),
                    "final_quality": quality,
                    "cost_saved_pct": (len(self.steps) - steps_executed) / len(self.steps)
                }
        
        return {
            "output": output,
            "steps_executed": steps_executed,
            "early_terminated": False,
            "final_quality": self.quality_fn(str(output))
        }
```

---

## 1.9.3 Quality Optimization

### 1.9.3.1 Prompt Tuning Per Chain Step

Each step's prompt can be independently optimized to maximize that step's output quality:

$$
p_i^* = \arg\max_{p_i \in \mathcal{P}} \; \mathbb{E}_{x \sim \mathcal{D}}\!\bigl[\, Q_i\!\bigl(p_\theta(y \mid p_i, x)\bigr) \,\bigr]
$$

**Methods:**

| Method | Description | Automation Level |
|---|---|---|
| **Manual iteration** | Expert rewrites prompt based on error analysis | Fully manual |
| **A/B testing** | Run two prompt variants, compare metrics | Semi-automated |
| **DSPy** (Khattab et al., 2023) | Compile chains with optimized prompts from examples | Automated |
| **OPRO** (Yang et al., 2023) | LLM proposes prompt variations, scores them, iterates | Automated |
| **TextGrad** (Yuksekgonul et al., 2024) | Backpropagate text-based gradients through chain | Automated |

### 1.9.3.2 Temperature and Sampling Parameter Optimization

Different steps benefit from different sampling strategies:

| Step Type | Optimal Temperature | Sampling Strategy | Rationale |
|---|---|---|---|
| Classification/routing | $\tau = 0.0$ | Greedy (argmax) | Determinism needed |
| Data extraction | $\tau = 0.0$ | Greedy | Precision over creativity |
| Reasoning | $\tau = 0.0 - 0.3$ | Low temperature | Logical consistency |
| Creative writing | $\tau = 0.7 - 1.0$ | Top-$p$ ($p = 0.9$) | Diversity desired |
| Brainstorming | $\tau = 1.0 - 1.2$ | Top-$p$ + top-$k$ | Maximum variety |
| Code generation | $\tau = 0.0 - 0.2$ | Greedy or low temp | Correctness critical |

### 1.9.3.3 Automated Prompt Optimization (DSPy)

DSPy compiles a declarative chain specification into optimized prompts through systematic search:

```python
# DSPy-style declarative chain definition (conceptual)
import dspy

class ResearchChain(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.ChainOfThought("query -> search_results")
        self.analyze = dspy.ChainOfThought("search_results -> analysis")
        self.synthesize = dspy.ChainOfThought("analysis, query -> report")
    
    def forward(self, query: str):
        search_results = self.search(query=query)
        analysis = self.analyze(search_results=search_results.search_results)
        report = self.synthesize(
            analysis=analysis.analysis, query=query
        )
        return report

# Compile with optimization
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=quality_metric, max_bootstrapped_demos=4)
optimized_chain = optimizer.compile(ResearchChain(), trainset=training_examples)
```

---

## 1.9.4 Multi-Objective Optimization

### 1.9.4.1 The Three-Way Trade-off

Latency, cost, and quality form a **Pareto frontier** — improving one typically degrades another:

$$
\min_{\theta} \; \bigl(\, L(\theta),\; C(\theta),\; -Q(\theta) \,\bigr)
$$

where $\theta$ represents all chain design parameters (model selection, parallelism, retries, prompt design, temperature, etc.).

**Pareto dominance.** Configuration $\theta_1$ **dominates** $\theta_2$ iff:

$$
L(\theta_1) \leq L(\theta_2) \;\wedge\; C(\theta_1) \leq C(\theta_2) \;\wedge\; Q(\theta_1) \geq Q(\theta_2)
$$

with at least one strict inequality. The **Pareto frontier** is the set of non-dominated configurations.

### 1.9.4.2 Constraint-Based Optimization

In practice, one or two objectives are treated as **constraints** and the third is optimized:

$$
\min_\theta \; C(\theta) \quad \text{s.t.} \quad Q(\theta) \geq Q_{\min}, \;\; L(\theta) \leq L_{\max}
$$

This is the most common production formulation: minimize cost subject to minimum quality and maximum latency.

```python
from dataclasses import dataclass

@dataclass
class ChainConfiguration:
    """A specific configuration of a prompt chain."""
    models: list[str]                    # model per step
    temperatures: list[float]            # temperature per step
    max_retries: list[int]               # retries per step
    parallel_groups: list[list[int]]     # parallelization structure
    cache_enabled: list[bool]            # caching per step
    verification_level: str              # none, basic, full
    
    # Measured performance
    latency_ms: float = 0
    cost_usd: float = 0
    quality_score: float = 0

class ChainOptimizer:
    """Multi-objective optimizer for chain configurations."""
    
    def __init__(self, 
                 evaluate_fn: Callable[[ChainConfiguration], dict],
                 quality_min: float = 0.8,
                 latency_max_ms: float = 10000,
                 cost_max_usd: float = 0.10):
        self.evaluate = evaluate_fn
        self.q_min = quality_min
        self.l_max = latency_max_ms
        self.c_max = cost_max_usd
    
    def is_feasible(self, config: ChainConfiguration) -> bool:
        return (config.quality_score >= self.q_min and
                config.latency_ms <= self.l_max and
                config.cost_usd <= self.c_max)
    
    def pareto_frontier(self, configs: list[ChainConfiguration]) -> list[ChainConfiguration]:
        """Extract Pareto-optimal configurations."""
        feasible = [c for c in configs if self.is_feasible(c)]
        frontier = []
        
        for c in feasible:
            dominated = False
            for other in feasible:
                if (other.cost_usd <= c.cost_usd and
                    other.latency_ms <= c.latency_ms and
                    other.quality_score >= c.quality_score and
                    (other.cost_usd < c.cost_usd or
                     other.latency_ms < c.latency_ms or
                     other.quality_score > c.quality_score)):
                    dominated = True
                    break
            if not dominated:
                frontier.append(c)
        
        return frontier
    
    def grid_search(self, search_space: dict) -> list[ChainConfiguration]:
        """Exhaustive search over discrete configuration space."""
        import itertools
        
        configs = []
        keys = list(search_space.keys())
        for values in itertools.product(*search_space.values()):
            params = dict(zip(keys, values))
            config = ChainConfiguration(**params)
            metrics = self.evaluate(config)
            config.latency_ms = metrics["latency_ms"]
            config.cost_usd = metrics["cost_usd"]
            config.quality_score = metrics["quality_score"]
            configs.append(config)
        
        return self.pareto_frontier(configs)
```

### 1.9.4.3 Dynamic Runtime Optimization

Adjust chain parameters **at runtime** based on current conditions:

$$
\theta_t^* = \arg\min_\theta \; C(\theta) \quad \text{s.t.} \quad Q(\theta) \geq Q_{\min}(\text{priority}_t), \;\; L(\theta) \leq L_{\max}(\text{load}_t)
$$

| Runtime Signal | Adaptation |
|---|---|
| **High API latency** | Switch to faster model, reduce retries |
| **Budget nearly exhausted** | Switch to cheaper model, enable caching |
| **Low traffic** | Use best model, full verification |
| **High traffic** | Use fastest configuration, skip optional steps |
| **Previous step failed** | Increase retries and verification for next step |

```python
class DynamicOptimizer:
    """Adjust chain parameters dynamically based on runtime conditions."""
    
    def __init__(self, configurations: dict[str, ChainConfiguration]):
        self.configs = configurations  # preset configurations
    
    def select_config(self, runtime_context: dict) -> ChainConfiguration:
        budget_remaining = runtime_context.get("budget_remaining_usd", float("inf"))
        current_latency = runtime_context.get("avg_api_latency_ms", 500)
        traffic_level = runtime_context.get("traffic_level", "normal")
        priority = runtime_context.get("priority", "balanced")
        
        if budget_remaining < 0.01:
            return self.configs["minimum_cost"]
        
        if current_latency > 2000:
            return self.configs["low_latency"]
        
        if traffic_level == "high":
            return self.configs["high_throughput"]
        
        if priority == "quality":
            return self.configs["maximum_quality"]
        
        return self.configs["balanced"]
```

---

## Summary: Optimization Decision Matrix

| Objective | Primary Technique | Expected Improvement | Trade-Off |
|---|---|---|---|
| **Reduce latency** | Parallelize independent steps | 2–5× speedup | Higher concurrent API usage |
| | Speculative execution | 20–50% on conditional branches | Wasted compute on misprediction |
| | Streaming between steps | 10–30% | Increased code complexity |
| | Caching | 50–99% on cache hits | Stale results risk |
| **Reduce cost** | Model selection per step | 3–10× cost reduction | Potential quality loss on small models |
| | Token compression | 20–50% token savings | Information loss risk |
| | Early termination | 10–50% fewer steps | May miss needed refinement |
| | Caching + memoization | Proportional to hit rate | Memory cost |
| **Increase quality** | Prompt optimization (DSPy/OPRO) | 5–25% quality gain | Requires labeled eval set |
| | Self-consistency (multi-sample) | 5–15% accuracy gain | $k\times$ cost multiplier |
| | Verification chains | Catches 60–90% of errors | Adds latency and cost |
| | Self-healing/reflection | 5–20% error reduction | Adds 1–2 extra LLM calls |