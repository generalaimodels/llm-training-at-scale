

# 1.6 Routing and Conditional Logic in Chains

> **Scope.** In any non-trivial agentic system, the execution path is not a fixed linear sequence — it is a **conditional computation graph** where the next step depends on the content, quality, or classification of the current step's output. Routing is the mechanism that implements this conditional branching: given an intermediate result, which downstream path should execute? Gate mechanisms are the dual concept: given an intermediate result, should execution proceed at all, or should it loop, retry, escalate, or halt? Together, routing and gating transform a static chain into a dynamic, adaptive system capable of handling the full distribution of real-world inputs. This section provides a first-principles treatment of every routing and gating pattern, with formal decision-theoretic foundations, production-grade implementations, and rigorous analysis of failure modes.

---

## 1.6.1 Static Routing

### 1.6.1.1 Foundational Definition

Static routing uses **predetermined, code-defined rules** to select the next chain step. The routing function is a pure deterministic mapping from observable features of the state to a branch identifier:

$$
R_{\text{static}}: \mathcal{F}(S_t) \longrightarrow \{b_1, b_2, \dots, b_k\}
$$

where $\mathcal{F}(S_t)$ extracts discrete features from the current state $S_t$ (keywords, numeric thresholds, type tags, regex matches), and $\{b_1, \dots, b_k\}$ is the finite set of possible downstream branches.

**Key properties:**

| Property | Value |
|---|---|
| **Determinism** | Given identical input, always selects identical branch |
| **Latency** | Near-zero (no LLM call, no embedding computation) |
| **Interpretability** | Fully transparent — every routing decision is inspectable |
| **Flexibility** | Low — cannot handle novel inputs outside predefined rules |
| **Maintenance cost** | Grows linearly with rule count; rules may conflict |

### 1.6.1.2 Predefined Conditional Branches

The simplest routing pattern: explicit `if/elif/else` logic that examines structured fields of the state.

**Formal model.** Define a set of predicates $\{P_1, P_2, \dots, P_k\}$ where each $P_j: \mathcal{S} \rightarrow \{0, 1\}$, and a **priority ordering** among predicates. The routing decision is:

$$
R(S_t) = b_j \quad \text{where } j = \min\{i : P_i(S_t) = 1\}
$$

If no predicate fires, a **default branch** $b_{\text{default}}$ handles the input (critical for robustness — never leave an input unrouted).

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Route:
    name: str
    predicate: Callable[[dict], bool]
    handler: Callable[[dict], Any]
    priority: int = 0  # lower = higher priority

class StaticRouter:
    """Deterministic, rule-based router with priority ordering."""
    
    def __init__(self, default_handler: Callable[[dict], Any]):
        self.routes: list[Route] = []
        self.default_handler = default_handler
        self._routing_log: list[dict] = []
    
    def add_route(self, route: Route) -> None:
        self.routes.append(route)
        self.routes.sort(key=lambda r: r.priority)  # maintain priority order
    
    def route(self, state: dict) -> tuple[str, Callable]:
        """Select the first matching route by priority."""
        for route in self.routes:
            if route.predicate(state):
                self._routing_log.append({
                    "state_summary": str(state.get("query", ""))[:100],
                    "matched_route": route.name,
                    "priority": route.priority
                })
                return route.name, route.handler
        
        self._routing_log.append({
            "state_summary": str(state.get("query", ""))[:100],
            "matched_route": "default",
            "priority": -1
        })
        return "default", self.default_handler


# --- Instantiation Example ---

router = StaticRouter(default_handler=general_handler)

router.add_route(Route(
    name="code_generation",
    predicate=lambda s: s.get("task_type") == "code",
    handler=code_generation_chain,
    priority=0
))

router.add_route(Route(
    name="data_analysis",
    predicate=lambda s: s.get("task_type") == "analysis",
    handler=data_analysis_chain,
    priority=1
))

router.add_route(Route(
    name="simple_qa",
    predicate=lambda s: s.get("estimated_complexity", 1.0) < 0.3,
    handler=simple_qa_handler,
    priority=2
))
```

### 1.6.1.3 Rule-Based Routing (Regex, Keyword, Threshold)

When the routing decision depends on the **content** of unstructured text (the user query or a previous step's output), static routing employs pattern-matching heuristics.

#### A. Keyword-Based Routing

$$
R_{\text{keyword}}(x) = b_j \quad \text{where } j = \arg\max_j \; |\mathcal{K}_j \cap \text{tokens}(x)|
$$

Here $\mathcal{K}_j$ is the keyword set associated with branch $b_j$, and $\text{tokens}(x)$ is the tokenized input.

```python
class KeywordRouter:
    """Route based on keyword overlap with predefined vocabularies."""
    
    def __init__(self):
        self.keyword_sets: dict[str, set[str]] = {}
        self.handlers: dict[str, Callable] = {}
    
    def add_route(self, name: str, keywords: list[str], handler: Callable):
        self.keyword_sets[name] = {kw.lower() for kw in keywords}
        self.handlers[name] = handler
    
    def route(self, text: str) -> tuple[str, float]:
        """Return (route_name, match_score) for the best matching route."""
        tokens = set(text.lower().split())
        scores = {}
        
        for name, keywords in self.keyword_sets.items():
            overlap = tokens & keywords
            # Normalized score: fraction of route keywords found
            scores[name] = len(overlap) / max(len(keywords), 1)
        
        if not scores or max(scores.values()) == 0:
            return "default", 0.0
        
        best_route = max(scores, key=scores.get)
        return best_route, scores[best_route]


# Example
kw_router = KeywordRouter()
kw_router.add_route("code", ["write", "code", "function", "implement", "debug", "python", "script"], code_handler)
kw_router.add_route("math", ["calculate", "solve", "equation", "integral", "derivative", "proof"], math_handler)
kw_router.add_route("search", ["find", "search", "look up", "what is", "who is", "when did"], search_handler)
```

**Limitations:** Keyword matching is brittle — "I want to *debug* my *relationship*" would incorrectly route to `code`. This motivates semantic routing (§1.6.2).

#### B. Regex-Based Routing

More expressive than keyword matching, regex routing captures structural patterns:

$$
R_{\text{regex}}(x) = b_j \quad \text{where } j = \min\{i : \text{regex}_i \text{ matches } x\}
$$

```python
import re

class RegexRouter:
    """Route based on regular expression pattern matching."""
    
    def __init__(self):
        self.patterns: list[tuple[str, re.Pattern, Callable]] = []
    
    def add_route(self, name: str, pattern: str, handler: Callable, 
                  flags: int = re.IGNORECASE):
        self.patterns.append((name, re.compile(pattern, flags), handler))
    
    def route(self, text: str) -> tuple[str, dict]:
        for name, pattern, handler in self.patterns:
            match = pattern.search(text)
            if match:
                return name, {
                    "handler": handler,
                    "matched_text": match.group(),
                    "groups": match.groups(),
                    "named_groups": match.groupdict()
                }
        return "default", {"handler": None}


regex_router = RegexRouter()
regex_router.add_route(
    "sql_query", 
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
    sql_handler
)
regex_router.add_route(
    "url_analysis",
    r"https?://[^\s]+",
    url_analysis_handler
)
regex_router.add_route(
    "email_task",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    email_handler
)
```

#### C. Threshold-Based Routing

When the state contains numeric signals (confidence scores, token counts, complexity estimates), routing decisions are based on threshold comparisons:

$$
R_{\text{threshold}}(S_t) = \begin{cases}
b_{\text{simple}} & \text{if } \text{score}(S_t) < \tau_1 \\
b_{\text{standard}} & \text{if } \tau_1 \leq \text{score}(S_t) < \tau_2 \\
b_{\text{complex}} & \text{if } \text{score}(S_t) \geq \tau_2
\end{cases}
$$

```python
class ThresholdRouter:
    """Multi-threshold routing based on numeric state features."""
    
    def __init__(self, feature_extractor: Callable[[dict], float]):
        self.feature_extractor = feature_extractor
        self.thresholds: list[tuple[float, str, Callable]] = []
        # Stored as (upper_bound, name, handler), sorted ascending
    
    def add_band(self, upper_bound: float, name: str, handler: Callable):
        self.thresholds.append((upper_bound, name, handler))
        self.thresholds.sort(key=lambda x: x[0])
    
    def route(self, state: dict) -> tuple[str, float]:
        score = self.feature_extractor(state)
        for upper_bound, name, handler in self.thresholds:
            if score < upper_bound:
                return name, score
        # Above all thresholds: use the last band
        _, name, handler = self.thresholds[-1]
        return name, score


# Example: Route by estimated query complexity
complexity_router = ThresholdRouter(
    feature_extractor=lambda s: s.get("estimated_complexity", 0.5)
)
complexity_router.add_band(0.3, "simple_path", simple_handler)    # < 0.3
complexity_router.add_band(0.7, "standard_path", standard_handler) # 0.3 - 0.7
complexity_router.add_band(1.1, "complex_path", complex_handler)   # >= 0.7
```

### 1.6.1.4 Deterministic Path Selection — Formal Properties

A static router $R_{\text{static}}$ satisfies three formal guarantees that dynamic routers cannot:

**Property 1 — Idempotency:**

$$
R_{\text{static}}(S_t) = R_{\text{static}}(S_t) \quad \forall S_t
$$

The same state always produces the same routing decision (no stochastic sampling, no model temperature effects).

**Property 2 — Composability:**

$$
R_{\text{static}}^{(1)} \circ R_{\text{static}}^{(2)} \text{ is also a static router}
$$

Chaining two static routers produces a static router — the combined behavior remains deterministic and inspectable.

**Property 3 — Decidability:**

$$
\forall S_t:\; \exists\, b_j \in \{b_1, \dots, b_k, b_{\text{default}}\} \text{ s.t. } R_{\text{static}}(S_t) = b_j
$$

Every input is routed somewhere (given a default branch), with guaranteed $\mathcal{O}(k)$ time complexity.

**When static routing fails.** Define the **coverage** of a static router:

$$
\text{Coverage}(R_{\text{static}}) = \Pr_{x \sim \mathcal{D}_{\text{input}}}\!\bigl[\, R_{\text{static}}(x) \neq b_{\text{default}} \,\bigr]
$$

If coverage falls below an acceptable threshold (many inputs hitting the default branch), the router's rule set is insufficient and should be augmented — or replaced with a dynamic router.

---

## 1.6.2 Dynamic / Semantic Routing

### 1.6.2.1 Foundational Motivation

Static routing fails when:
1. The input space is too large or varied for enumerable rules.
2. The routing decision requires **semantic understanding** (meaning, intent, topic).
3. The classification boundary is **non-linear** in feature space.
4. New categories emerge over time without code changes.

Dynamic routing uses **learned representations** — either from an LLM or from embedding models — to classify inputs and select branches.

**General formulation:**

$$
R_{\text{dynamic}}: x \longrightarrow \arg\max_{b_j \in \mathcal{B}} \; \text{score}(x, b_j)
$$

where $\text{score}$ is computed by a model (LLM, embedding similarity, classifier).

### 1.6.2.2 LLM-as-a-Router: Classification Prompt

The most flexible dynamic routing approach: ask the LLM to classify the input into one of $k$ predefined categories.

**Formal model.** The LLM acts as a $k$-class classifier:

$$
R_{\text{LLM}}(x) = \arg\max_{b_j \in \mathcal{B}} \; p_\theta\!\bigl(\, b_j \mid x,\; \text{prompt}_{\text{routing}} \,\bigr)
$$

**Implementation — Structured Output Router:**

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class RouteDecision(BaseModel):
    """Structured output from the LLM router."""
    reasoning: str = Field(description="Brief explanation of classification logic")
    selected_route: str = Field(description="The chosen route name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this classification")
    alternative_route: str | None = Field(
        default=None, 
        description="Second-best route if confidence is low"
    )


class LLMRouter:
    """Use an LLM to semantically classify inputs and select chain branches."""
    
    ROUTING_PROMPT_TEMPLATE = """You are a precise query classifier. Given the user's 
input, classify it into exactly ONE of the following categories:

{route_descriptions}

Rules:
1. Choose the MOST specific matching category
2. If uncertain between two categories, choose the one that better serves the user
3. Provide your confidence (0.0 to 1.0) — be calibrated, not overconfident
4. If confidence < 0.6, also specify an alternative route

User input: {input}

Respond as JSON matching this schema:
{{"reasoning": "...", "selected_route": "...", "confidence": 0.X, "alternative_route": "..." or null}}"""
    
    def __init__(self, llm: "LLMClient", routes: dict[str, dict]):
        """
        Args:
            llm: LLM client for classification
            routes: dict mapping route_name -> {
                "description": str,   # what this route handles
                "examples": list[str], # few-shot examples
                "handler": callable
            }
        """
        self.llm = llm
        self.routes = routes
    
    def _build_route_descriptions(self) -> str:
        lines = []
        for name, config in self.routes.items():
            desc = f"- **{name}**: {config['description']}"
            if config.get("examples"):
                examples = "; ".join(f'"{ex}"' for ex in config["examples"][:3])
                desc += f"\n  Examples: {examples}"
            lines.append(desc)
        return "\n".join(lines)
    
    async def route(self, user_input: str) -> RouteDecision:
        prompt = self.ROUTING_PROMPT_TEMPLATE.format(
            route_descriptions=self._build_route_descriptions(),
            input=user_input
        )
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,       # deterministic classification
            max_tokens=200
        )
        decision = RouteDecision.model_validate_json(response)
        
        # Validate that selected_route exists
        if decision.selected_route not in self.routes:
            # Fallback: find closest route name (fuzzy matching)
            decision.selected_route = self._fuzzy_match(decision.selected_route)
        
        return decision
    
    def _fuzzy_match(self, candidate: str) -> str:
        """Find the closest valid route name."""
        from difflib import get_close_matches
        matches = get_close_matches(candidate, self.routes.keys(), n=1, cutoff=0.4)
        return matches[0] if matches else list(self.routes.keys())[0]


# --- Instantiation ---

llm_router = LLMRouter(
    llm=llm_client,
    routes={
        "code_assistance": {
            "description": "Writing, debugging, explaining, or reviewing code in any programming language",
            "examples": ["Write a Python function to sort a list", "Why does this SQL query fail?"],
            "handler": code_chain
        },
        "research_analysis": {
            "description": "In-depth research requiring multiple sources, data analysis, or literature review",
            "examples": ["Compare transformer architectures published in 2024", "Analyze market trends in EV sector"],
            "handler": research_chain
        },
        "creative_writing": {
            "description": "Generating creative content: stories, poems, marketing copy, brainstorming",
            "examples": ["Write a haiku about machine learning", "Generate taglines for a coffee brand"],
            "handler": creative_chain
        },
        "simple_qa": {
            "description": "Direct factual questions answerable in a few sentences without research",
            "examples": ["What is the capital of France?", "When was Python first released?"],
            "handler": simple_qa_handler
        }
    }
)
```

**Latency-accuracy trade-off:** LLM routing adds a full LLM call (~200–1000ms) before the actual chain begins. This is acceptable only when routing savings (avoiding expensive wrong-path execution) exceed the routing cost:

$$
\text{Net benefit} = \underbrace{\mathbb{E}[\text{cost}(\text{wrong path})] \cdot \Pr[\text{misroute without LLM}]}_{\text{avoided cost}} - \underbrace{\text{cost}(\text{LLM routing call})}_{\text{routing overhead}}
$$

### 1.6.2.3 Embedding-Based Routing

Instead of a full LLM call, compute dense embeddings of the input and compare against pre-computed **route prototype embeddings** using cosine similarity. This reduces routing latency from ~500ms (LLM call) to ~5ms (embedding lookup + similarity computation).

**Formal model.** Let $\mathbf{e}: \mathcal{V}^* \rightarrow \mathbb{R}^d$ be a sentence embedding function. For each route $b_j$, precompute a prototype embedding $\mathbf{p}_j$ from its description and examples:

$$
\mathbf{p}_j = \frac{1}{|\mathcal{E}_j|} \sum_{e \in \mathcal{E}_j} \mathbf{e}(e)
$$

where $\mathcal{E}_j$ is the set of example texts for route $b_j$. At inference time:

$$
R_{\text{embed}}(x) = \arg\max_{b_j} \; \text{sim}\!\bigl(\, \mathbf{e}(x),\; \mathbf{p}_j \,\bigr) = \arg\max_{b_j} \; \frac{\mathbf{e}(x) \cdot \mathbf{p}_j}{\|\mathbf{e}(x)\| \cdot \|\mathbf{p}_j\|}
$$

```python
import numpy as np
from typing import Optional

class EmbeddingRouter:
    """Fast semantic routing via embedding cosine similarity."""
    
    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        self.embed_fn = embed_fn
        self.routes: dict[str, dict] = {}
        self.prototypes: dict[str, np.ndarray] = {}
        self._all_example_embeddings: dict[str, list[np.ndarray]] = {}
    
    def add_route(self, name: str, description: str, 
                  examples: list[str], handler: Callable,
                  prototype_method: str = "centroid"):
        """Register a route with examples; compute prototype embedding."""
        example_embeds = [self.embed_fn(ex) for ex in examples]
        desc_embed = self.embed_fn(description)
        
        if prototype_method == "centroid":
            # Average of description + all examples
            all_embeds = [desc_embed] + example_embeds
            prototype = np.mean(all_embeds, axis=0)
        elif prototype_method == "description_only":
            prototype = desc_embed
        elif prototype_method == "medoid":
            # The example closest to the centroid (more robust to outliers)
            centroid = np.mean(example_embeds, axis=0)
            idx = np.argmax([
                np.dot(e, centroid) / (np.linalg.norm(e) * np.linalg.norm(centroid))
                for e in example_embeds
            ])
            prototype = example_embeds[idx]
        
        # L2-normalize for efficient cosine similarity via dot product
        prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
        
        self.routes[name] = {"description": description, "handler": handler}
        self.prototypes[name] = prototype
        self._all_example_embeddings[name] = example_embeds
    
    def route(self, text: str, threshold: float = 0.0) -> tuple[str, float, dict]:
        """Route input to the most similar route prototype.
        
        Returns:
            (route_name, similarity_score, full_scores_dict)
        """
        query_embed = self.embed_fn(text)
        query_embed = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        
        scores = {}
        for name, prototype in self.prototypes.items():
            scores[name] = float(np.dot(query_embed, prototype))
        
        best_route = max(scores, key=scores.get)
        best_score = scores[best_route]
        
        if best_score < threshold:
            return "default", best_score, scores
        
        return best_route, best_score, scores
    
    def route_knn(self, text: str, k: int = 5) -> tuple[str, float]:
        """K-nearest-neighbor routing: vote among k closest examples."""
        query_embed = self.embed_fn(text)
        query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        
        all_similarities = []
        for name, examples in self._all_example_embeddings.items():
            for ex_embed in examples:
                ex_norm = ex_embed / (np.linalg.norm(ex_embed) + 1e-8)
                sim = float(np.dot(query_norm, ex_norm))
                all_similarities.append((sim, name))
        
        # Sort by similarity, take top-k
        all_similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = all_similarities[:k]
        
        # Majority vote (weighted by similarity)
        votes: dict[str, float] = {}
        for sim, name in top_k:
            votes[name] = votes.get(name, 0.0) + sim
        
        best_route = max(votes, key=votes.get)
        confidence = votes[best_route] / sum(votes.values())
        return best_route, confidence
```

**Prototype refinement.** The initial prototype computed from a handful of examples may be suboptimal. Improve it by:

1. **Online update:** As inputs are routed and confirmed correct, update the prototype with exponential moving average:

$$
\mathbf{p}_j^{(t+1)} = \alpha \cdot \mathbf{e}(x_t) + (1 - \alpha) \cdot \mathbf{p}_j^{(t)}
$$

2. **Contrastive learning:** Fine-tune the embedding model so that inputs belonging to route $j$ are closer to $\mathbf{p}_j$ and farther from other prototypes:

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp\!\bigl(\text{sim}(\mathbf{e}(x), \mathbf{p}_+) / \tau\bigr)}{\sum_{j=1}^{k} \exp\!\bigl(\text{sim}(\mathbf{e}(x), \mathbf{p}_j) / \tau\bigr)}
$$

where $\mathbf{p}_+$ is the correct route prototype and $\tau$ is a temperature parameter.

### 1.6.2.4 Confidence-Based Routing

Confidence-based routing uses the **model's own uncertainty** about its output to decide the execution path. If the model is confident, take the fast path; if uncertain, escalate to a more capable (or more cautious) path.

**Formal model:**

$$
R_{\text{confidence}}(x) = \begin{cases}
b_{\text{fast}} & \text{if } \max_y \; p_\theta(y \mid x) > \tau_{\text{high}} \\
b_{\text{standard}} & \text{if } \tau_{\text{low}} \leq \max_y \; p_\theta(y \mid x) \leq \tau_{\text{high}} \\
b_{\text{escalate}} & \text{if } \max_y \; p_\theta(y \mid x) < \tau_{\text{low}}
\end{cases}
$$

**Confidence estimation methods:**

| Method | Mechanism | Reliability |
|---|---|---|
| **Token log-probabilities** | $\text{conf}(y) = \exp\!\bigl(\frac{1}{T}\sum_{t=1}^{T} \log p(y_t \mid y_{<t}, x)\bigr)$ | Moderate — well-calibrated in some models |
| **Self-verbalized confidence** | Ask the LLM: "Rate your confidence 0–1" | Low — models tend toward overconfidence |
| **Consistency sampling** | Generate $N$ responses; measure agreement | High — directly estimates epistemic uncertainty |
| **Entropy of output distribution** | $H(Y \mid x) = -\sum_y p(y \mid x) \log p(y \mid x)$ | Theoretically sound; hard to compute over full sequence space |

**Consistency-based confidence (most robust):**

$$
\text{Confidence}(x) = \frac{1}{\binom{N}{2}} \sum_{i < j} \text{sim}(y_i, y_j)
$$

where $y_1, \dots, y_N \sim p_\theta(\cdot \mid x)$ are $N$ independent samples and $\text{sim}$ is semantic similarity (e.g., ROUGE-L or embedding cosine).

```python
class ConfidenceRouter:
    """Route based on model's confidence in its own output."""
    
    def __init__(self, llm: "LLMClient", 
                 high_threshold: float = 0.85,
                 low_threshold: float = 0.5,
                 n_samples: int = 5):
        self.llm = llm
        self.tau_high = high_threshold
        self.tau_low = low_threshold
        self.n_samples = n_samples
    
    async def estimate_confidence_logprob(self, prompt: str) -> float:
        """Confidence via mean token log-probability."""
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            logprobs=True,
            max_tokens=500
        )
        token_logprobs = response.logprobs  # list of log-probabilities
        if not token_logprobs:
            return 0.5  # no logprobs available
        
        # Geometric mean of token probabilities
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
        confidence = float(np.exp(mean_logprob))
        return confidence
    
    async def estimate_confidence_consistency(self, prompt: str) -> float:
        """Confidence via consistency across multiple samples."""
        responses = await asyncio.gather(*[
            self.llm.generate(prompt=prompt, temperature=0.7, max_tokens=500)
            for _ in range(self.n_samples)
        ])
        
        texts = [r.text if hasattr(r, 'text') else str(r) for r in responses]
        
        # Pairwise similarity (using simple token overlap as proxy)
        from itertools import combinations
        similarities = []
        for a, b in combinations(texts, 2):
            tokens_a = set(a.lower().split())
            tokens_b = set(b.lower().split())
            if tokens_a | tokens_b:
                jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                similarities.append(jaccard)
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    async def route(self, prompt: str, 
                    method: str = "logprob") -> tuple[str, float]:
        if method == "logprob":
            confidence = await self.estimate_confidence_logprob(prompt)
        elif method == "consistency":
            confidence = await self.estimate_confidence_consistency(prompt)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if confidence > self.tau_high:
            return "fast_path", confidence
        elif confidence >= self.tau_low:
            return "standard_path", confidence
        else:
            return "escalation_path", confidence
```

### 1.6.2.5 Multi-Class Routing with Softmax over Chain Paths

When routing among $k$ paths, the most principled approach is to model the routing decision as a categorical distribution over paths, computed via **softmax over route scores**:

$$
P(b_j \mid x) = \frac{\exp\!\bigl(\text{score}(x, b_j) / \tau\bigr)}{\sum_{i=1}^{k} \exp\!\bigl(\text{score}(x, b_i) / \tau\bigr)}
$$

where $\tau$ is a temperature parameter controlling the sharpness of the distribution:
- $\tau \to 0$: deterministic (argmax) selection
- $\tau = 1$: standard softmax
- $\tau \to \infty$: uniform random selection

**Advantages of probabilistic routing:**

1. **Calibrated uncertainty:** $P(b_j \mid x)$ directly measures how well the input fits each route.
2. **Rejection detection:** If $\max_j P(b_j \mid x) < \tau_{\text{reject}}$, the input doesn't fit any route well.
3. **Multi-path execution:** When $P(b_1 \mid x) \approx P(b_2 \mid x)$, both paths can be executed and results merged.

```python
class SoftmaxRouter:
    """Probabilistic multi-class router with softmax scoring."""
    
    def __init__(self, embed_fn: Callable, temperature: float = 0.1):
        self.embed_fn = embed_fn
        self.temperature = temperature
        self.routes: dict[str, dict] = {}
        self.prototypes: dict[str, np.ndarray] = {}
    
    def add_route(self, name: str, prototype_texts: list[str], handler: Callable):
        embeds = [self.embed_fn(t) for t in prototype_texts]
        prototype = np.mean(embeds, axis=0)
        prototype /= (np.linalg.norm(prototype) + 1e-8)
        self.routes[name] = {"handler": handler}
        self.prototypes[name] = prototype
    
    def compute_distribution(self, text: str) -> dict[str, float]:
        """Compute softmax probability distribution over routes."""
        query = self.embed_fn(text)
        query /= (np.linalg.norm(query) + 1e-8)
        
        scores = {name: float(np.dot(query, proto)) 
                  for name, proto in self.prototypes.items()}
        
        # Softmax
        names = list(scores.keys())
        raw_scores = np.array([scores[n] for n in names])
        exp_scores = np.exp((raw_scores - raw_scores.max()) / self.temperature)
        probs = exp_scores / exp_scores.sum()
        
        return {name: float(prob) for name, prob in zip(names, probs)}
    
    def route_deterministic(self, text: str) -> tuple[str, float]:
        """Select the highest-probability route."""
        dist = self.compute_distribution(text)
        best = max(dist, key=dist.get)
        return best, dist[best]
    
    def route_with_rejection(self, text: str, 
                              reject_threshold: float = 0.4) -> tuple[str | None, dict]:
        """Route with rejection: return None if no route is confident enough."""
        dist = self.compute_distribution(text)
        best = max(dist, key=dist.get)
        
        if dist[best] < reject_threshold:
            return None, dist  # no route matches well
        return best, dist
    
    def route_multi_path(self, text: str, 
                          threshold: float = 0.2) -> list[tuple[str, float]]:
        """Return all routes above threshold, for parallel execution."""
        dist = self.compute_distribution(text)
        return [(name, prob) for name, prob in dist.items() if prob >= threshold]
```

### 1.6.2.6 Ensemble Routing (Multiple Classifiers Voting)

When no single routing method is sufficiently reliable, **ensemble routing** combines multiple routing signals through a voting or aggregation mechanism.

**Formal model.** Given $M$ routers $R_1, R_2, \dots, R_M$, each producing a probability distribution over routes:

$$
P_{\text{ensemble}}(b_j \mid x) = \sum_{m=1}^{M} w_m \cdot P_m(b_j \mid x)
$$

where $w_m$ are router weights satisfying $\sum_m w_m = 1$, calibrated based on each router's historical accuracy.

**Voting strategies:**

| Strategy | Formula | When to Use |
|---|---|---|
| **Majority vote** | $R_{\text{ens}}(x) = \text{mode}\{R_m(x)\}_{m=1}^{M}$ | All routers are equally reliable |
| **Weighted vote** | $R_{\text{ens}}(x) = \arg\max_j \sum_m w_m \cdot \mathbb{1}[R_m(x) = b_j]$ | Routers have known accuracy differences |
| **Probability averaging** | $P_{\text{ens}}(b_j) = \frac{1}{M}\sum_m P_m(b_j)$ | Routers produce calibrated probabilities |
| **Cascading** | Use $R_1$; if confidence $< \tau$, try $R_2$; etc. | Routers have different cost/accuracy profiles |

```python
class EnsembleRouter:
    """Ensemble of multiple routing strategies with weighted voting."""
    
    def __init__(self):
        self.routers: list[tuple[str, Callable, float]] = []
        # (name, route_fn, weight)
    
    def add_router(self, name: str, 
                   route_fn: Callable[[str], tuple[str, float]], 
                   weight: float = 1.0):
        self.routers.append((name, route_fn, weight))
        # Renormalize weights
        total = sum(w for _, _, w in self.routers)
        self.routers = [(n, fn, w/total) for n, fn, w in self.routers]
    
    async def route(self, text: str, 
                    strategy: str = "weighted_vote") -> tuple[str, dict]:
        # Collect all router decisions
        decisions = {}
        for name, route_fn, weight in self.routers:
            route_name, confidence = await route_fn(text) if asyncio.iscoroutinefunction(route_fn) \
                                     else (route_fn(text))
            decisions[name] = {
                "route": route_name,
                "confidence": confidence,
                "weight": weight
            }
        
        if strategy == "weighted_vote":
            # Aggregate weighted votes
            vote_scores: dict[str, float] = {}
            for dec in decisions.values():
                route = dec["route"]
                vote_scores[route] = vote_scores.get(route, 0) + dec["weight"]
            
            best_route = max(vote_scores, key=vote_scores.get)
            return best_route, {
                "individual_decisions": decisions,
                "aggregated_scores": vote_scores,
                "strategy": strategy
            }
        
        elif strategy == "cascade":
            # Use first router with confidence above threshold
            for name, route_fn, weight in self.routers:
                dec = decisions[name]
                if dec["confidence"] >= 0.7:
                    return dec["route"], {
                        "decided_by": name,
                        "confidence": dec["confidence"],
                        "all_decisions": decisions
                    }
            # Fallback: use highest-weight router
            primary = max(decisions.items(), key=lambda x: x[1]["weight"])
            return primary[1]["route"], {"decided_by": "fallback", "all": decisions}
```

---

## 1.6.3 Routing Architectures

### 1.6.3.1 Architecture Taxonomy

Each routing architecture addresses a specific axis of variation in the input distribution:

```
                    ┌──────────────────────────────┐
                    │     Input Classification      │
                    │         Dimensions             │
                    └──────────┬───────────────────┘
          ┌───────────┬───────┼────────┬────────────┐
          ▼           ▼       ▼        ▼            ▼
     ┌─────────┐ ┌────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
     │ Intent  │ │ Topic  │ │Compl.│ │  Model   │ │ Fallback │
     │ Router  │ │ Router │ │Router│ │ Selector │ │  Router  │
     └─────────┘ └────────┘ └──────┘ └──────────┘ └──────────┘
     "What does  "What is   "How     "Which      "What if
      the user    it about?" hard?"   model?"     nothing
      want to                                     works?"
      achieve?"
```

### 1.6.3.2 Intent Classification Routers

An intent router determines the user's **communicative goal** — what action they want the system to perform.

**Formal model.** The intent space $\mathcal{I} = \{i_1, \dots, i_k\}$ partitions all possible user inputs by desired action:

$$
R_{\text{intent}}(x) = \arg\max_{i \in \mathcal{I}} \; p(i \mid x)
$$

**Standard intent taxonomy for agentic systems:**

| Intent Category | Sub-Intents | Chain Mapping |
|---|---|---|
| **Informational** | Factual Q&A, explanation, comparison | RAG chain / direct answer |
| **Instructional** | Write code, generate text, create plan | Generation chain |
| **Analytical** | Analyze data, evaluate options, summarize | Analysis chain |
| **Transactional** | Book appointment, send email, execute trade | Tool-action chain |
| **Navigational** | Find a document, locate a setting | Search/retrieval chain |
| **Conversational** | Chitchat, clarification, follow-up | Simple response / clarification loop |

```python
class IntentRouter:
    """Hierarchical intent classification router."""
    
    INTENT_PROMPT = """Classify the user's intent into one of these categories.
    
Primary intents:
- INFORMATIONAL: User wants to learn or understand something
- INSTRUCTIONAL: User wants something created or generated
- ANALYTICAL: User wants data analyzed or options evaluated
- TRANSACTIONAL: User wants an action performed (email, API call, etc.)
- NAVIGATIONAL: User wants to find a specific resource
- CONVERSATIONAL: User is engaging in dialogue, not requesting a task
- CLARIFICATION_NEEDED: User's intent is ambiguous; need to ask follow-up

User input: {input}

Respond as JSON:
{{"primary_intent": "...", "sub_intent": "...", "confidence": 0.X, 
  "requires_tools": true/false, "clarification_question": "..." or null}}"""
    
    def __init__(self, llm: "LLMClient", intent_to_chain: dict[str, Callable]):
        self.llm = llm
        self.intent_to_chain = intent_to_chain
    
    async def classify_and_route(self, user_input: str) -> dict:
        response = await self.llm.generate(
            prompt=self.INTENT_PROMPT.format(input=user_input),
            temperature=0.0,
            max_tokens=150
        )
        classification = json.loads(response)
        
        primary = classification["primary_intent"]
        
        if primary == "CLARIFICATION_NEEDED":
            return {
                "action": "clarify",
                "question": classification.get("clarification_question",
                    "Could you please provide more details about what you need?")
            }
        
        handler = self.intent_to_chain.get(primary)
        if handler is None:
            handler = self.intent_to_chain.get("DEFAULT")
        
        return {
            "action": "execute",
            "intent": classification,
            "handler": handler
        }
```

### 1.6.3.3 Topic-Based Routers

Topic routers direct inputs to **domain-specific chains** optimized for particular subject areas. Unlike intent routers (which classify *what the user wants to do*), topic routers classify *what the input is about*.

**Formal model:**

$$
R_{\text{topic}}(x) = \arg\max_{t \in \mathcal{T}_{\text{topics}}} \; p(t \mid x)
$$

**Implementation via embedding clusters:**

```python
class TopicRouter:
    """Route to domain-specific chains based on topic classification."""
    
    def __init__(self, embed_fn: Callable):
        self.embed_fn = embed_fn
        self.topic_centroids: dict[str, np.ndarray] = {}
        self.topic_handlers: dict[str, Callable] = {}
        self.topic_examples: dict[str, list[np.ndarray]] = {}
    
    def register_topic(self, topic: str, 
                       seed_documents: list[str], 
                       handler: Callable):
        """Register a topic with seed documents to compute centroid."""
        embeddings = [self.embed_fn(doc) for doc in seed_documents]
        centroid = np.mean(embeddings, axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        
        self.topic_centroids[topic] = centroid
        self.topic_handlers[topic] = handler
        self.topic_examples[topic] = embeddings
    
    def route(self, text: str) -> tuple[str, float, dict[str, float]]:
        query = self.embed_fn(text)
        query /= (np.linalg.norm(query) + 1e-8)
        
        similarities = {
            topic: float(np.dot(query, centroid))
            for topic, centroid in self.topic_centroids.items()
        }
        
        best_topic = max(similarities, key=similarities.get)
        return best_topic, similarities[best_topic], similarities


# Example: Multi-domain support agent
topic_router = TopicRouter(embed_fn=sentence_embed)
topic_router.register_topic(
    "finance",
    seed_documents=[
        "quarterly earnings report analysis",
        "stock portfolio rebalancing strategy",
        "compound interest calculation",
        "credit score improvement"
    ],
    handler=finance_chain
)
topic_router.register_topic(
    "healthcare",
    seed_documents=[
        "symptoms of seasonal allergies",
        "medication interaction checker",
        "blood test results interpretation",
        "vaccination schedule"
    ],
    handler=healthcare_chain
)
```

### 1.6.3.4 Complexity-Based Routers

Complexity routers estimate the **difficulty** of the input and select an execution path calibrated to that difficulty — simple inputs take a fast/cheap path; complex inputs take a thorough/expensive path.

**Formal model.** Define a complexity function $\kappa: \mathcal{X} \rightarrow [0, 1]$ and a set of paths ordered by capability:

$$
R_{\text{complexity}}(x) = \begin{cases}
b_{\text{direct}} & \text{if } \kappa(x) < \tau_1 \quad \text{(single LLM call)} \\
b_{\text{chain}} & \text{if } \tau_1 \leq \kappa(x) < \tau_2 \quad \text{(multi-step chain)} \\
b_{\text{agent}} & \text{if } \kappa(x) \geq \tau_2 \quad \text{(full agent loop with tools)}
\end{cases}
$$

**Complexity estimation heuristics:**

| Signal | Measurement | Complexity Contribution |
|---|---|---|
| **Query length** | Token count | Longer queries correlate with complexity |
| **Entity count** | NER or keyword extraction | More entities → more relationships to reason about |
| **Constraint count** | Clause parsing | "but not...", "only if...", "unless..." add complexity |
| **Required tools** | Keyword/intent detection | Tool use implies non-trivial execution |
| **Temporal reasoning** | Tense/date detection | "Compare last year vs. this year" requires multi-step |
| **Compositional depth** | Parse tree depth | Nested questions ("What is the GDP of the country where...") |

```python
class ComplexityRouter:
    """Estimate input complexity and route to appropriate execution tier."""
    
    COMPLEXITY_TIERS = [
        {"name": "direct",   "max_complexity": 0.3, "description": "Single LLM call"},
        {"name": "chain",    "max_complexity": 0.7, "description": "Multi-step chain"},
        {"name": "agent",    "max_complexity": 1.0, "description": "Full agent with tools"},
    ]
    
    def __init__(self, llm: "LLMClient" = None):
        self.llm = llm
        self.tier_handlers: dict[str, Callable] = {}
    
    def register_tier(self, name: str, handler: Callable):
        self.tier_handlers[name] = handler
    
    def estimate_complexity_heuristic(self, text: str) -> float:
        """Fast heuristic complexity estimation (no LLM call)."""
        tokens = text.split()
        
        # Feature extraction
        length_score = min(len(tokens) / 100, 1.0)  # normalize by 100 tokens
        
        question_words = sum(1 for t in tokens if t.lower() in 
            {"what", "why", "how", "compare", "analyze", "evaluate", "explain"})
        question_score = min(question_words / 5, 1.0)
        
        constraint_words = sum(1 for t in tokens if t.lower() in
            {"but", "unless", "except", "only", "must", "should", "between",
             "versus", "vs", "compared"})
        constraint_score = min(constraint_words / 3, 1.0)
        
        tool_indicators = sum(1 for t in tokens if t.lower() in
            {"calculate", "search", "find", "look up", "code", "run",
             "execute", "fetch", "download", "scrape"})
        tool_score = min(tool_indicators / 2, 1.0)
        
        # Weighted combination
        complexity = (
            0.2 * length_score +
            0.25 * question_score +
            0.25 * constraint_score +
            0.3 * tool_score
        )
        return min(complexity, 1.0)
    
    async def estimate_complexity_llm(self, text: str) -> float:
        """LLM-based complexity estimation (more accurate, higher latency)."""
        prompt = (
            "Estimate the complexity of answering this query on a 0.0-1.0 scale.\n"
            "0.0 = trivial factual question\n"
            "0.5 = requires some reasoning or multiple pieces of info\n"
            "1.0 = requires research, tools, multi-step analysis\n\n"
            f"Query: {text}\n\n"
            "Respond with ONLY a number between 0.0 and 1.0."
        )
        response = await self.llm.generate(prompt=prompt, max_tokens=10, temperature=0.0)
        return float(response.strip())
    
    def route(self, text: str, use_llm: bool = False) -> tuple[str, float]:
        if use_llm and self.llm:
            complexity = asyncio.run(self.estimate_complexity_llm(text))
        else:
            complexity = self.estimate_complexity_heuristic(text)
        
        for tier in self.COMPLEXITY_TIERS:
            if complexity <= tier["max_complexity"]:
                return tier["name"], complexity
        
        return "agent", complexity  # fallback to most capable tier
```

### 1.6.3.5 Model-Selection Routers

Model-selection routers choose **which LLM** to use for a given step — routing easy tasks to small, fast, cheap models and hard tasks to large, capable, expensive models.

**Decision-theoretic formulation.** For input $x$, model $m \in \mathcal{M}$, the optimal model selection minimizes expected cost subject to a quality constraint:

$$
m^*(x) = \arg\min_{m \in \mathcal{M}} \; \text{cost}(m) \quad \text{s.t.} \quad \mathbb{E}[\text{quality}(m, x)] \geq q_{\min}
$$

Equivalently, maximize quality-adjusted utility:

$$
m^*(x) = \arg\max_{m \in \mathcal{M}} \; \bigl[\, \text{quality}(m, x) - \lambda \cdot \text{cost}(m) \,\bigr]
$$

where $\lambda$ is the cost-quality trade-off parameter.

```python
@dataclass
class ModelConfig:
    name: str
    cost_per_input_token: float    # in USD per 1M tokens
    cost_per_output_token: float
    latency_ms: float              # average time-to-first-token
    capability_score: float        # 0-1, estimated general capability
    context_window: int
    supports_tools: bool
    supports_structured_output: bool

class ModelRouter:
    """Select the optimal model for each task based on requirements."""
    
    def __init__(self, models: list[ModelConfig]):
        self.models = models
    
    def select(self, requirements: dict) -> ModelConfig:
        """
        Requirements dict:
        - min_capability: float (0-1)
        - max_cost_per_call: float (USD)
        - max_latency_ms: float
        - needs_tools: bool
        - needs_structured_output: bool
        - estimated_input_tokens: int
        - estimated_output_tokens: int
        - priority: "cost" | "quality" | "latency"
        """
        candidates = self.models.copy()
        
        # Hard filters
        min_cap = requirements.get("min_capability", 0.0)
        candidates = [m for m in candidates if m.capability_score >= min_cap]
        
        if requirements.get("needs_tools"):
            candidates = [m for m in candidates if m.supports_tools]
        
        if requirements.get("needs_structured_output"):
            candidates = [m for m in candidates if m.supports_structured_output]
        
        max_latency = requirements.get("max_latency_ms", float("inf"))
        candidates = [m for m in candidates if m.latency_ms <= max_latency]
        
        # Cost filter
        in_tok = requirements.get("estimated_input_tokens", 1000)
        out_tok = requirements.get("estimated_output_tokens", 500)
        max_cost = requirements.get("max_cost_per_call", float("inf"))
        
        def call_cost(m: ModelConfig) -> float:
            return (m.cost_per_input_token * in_tok + 
                    m.cost_per_output_token * out_tok) / 1e6
        
        candidates = [m for m in candidates if call_cost(m) <= max_cost]
        
        if not candidates:
            raise ValueError("No model satisfies all requirements")
        
        # Soft ranking
        priority = requirements.get("priority", "cost")
        if priority == "cost":
            return min(candidates, key=call_cost)
        elif priority == "quality":
            return max(candidates, key=lambda m: m.capability_score)
        elif priority == "latency":
            return min(candidates, key=lambda m: m.latency_ms)
        else:
            return candidates[0]


# --- Model Inventory ---
models = [
    ModelConfig("gpt-4o", 2.50, 10.00, 800, 0.95, 128000, True, True),
    ModelConfig("gpt-4o-mini", 0.15, 0.60, 300, 0.82, 128000, True, True),
    ModelConfig("claude-3.5-sonnet", 3.00, 15.00, 900, 0.96, 200000, True, True),
    ModelConfig("claude-3.5-haiku", 0.25, 1.25, 400, 0.80, 200000, True, True),
    ModelConfig("gemini-2.0-flash", 0.10, 0.40, 250, 0.78, 1000000, True, True),
]

model_router = ModelRouter(models)

# Route a simple task to the cheapest model
simple_model = model_router.select({
    "min_capability": 0.7,
    "needs_tools": False,
    "priority": "cost",
    "estimated_input_tokens": 500,
    "estimated_output_tokens": 200
})
# Returns: gemini-2.0-flash

# Route a complex task to the best model
complex_model = model_router.select({
    "min_capability": 0.9,
    "needs_tools": True,
    "needs_structured_output": True,
    "priority": "quality"
})
# Returns: claude-3.5-sonnet
```

### 1.6.3.6 Fallback Routing and Escalation Paths

Fallback routing ensures that **every input receives a response**, even when the primary routing path fails. Escalation paths progressively invoke more capable (and expensive) resources.

**Formal model — Escalation Ladder:**

$$
\text{Escalation}: b_1 \xrightarrow{\text{fail}} b_2 \xrightarrow{\text{fail}} b_3 \xrightarrow{\text{fail}} \cdots \xrightarrow{\text{fail}} b_{\text{human}}
$$

At each rung, the system checks whether the output satisfies quality criteria. If not, it escalates:

$$
\text{output} = \begin{cases}
y_1 & \text{if } Q(y_1) \geq q_{\min} \\
y_2 & \text{if } Q(y_1) < q_{\min} \text{ and } Q(y_2) \geq q_{\min} \\
\vdots \\
\text{human response} & \text{if all automated paths fail}
\end{cases}
$$

```python
class EscalationRouter:
    """Progressive escalation through increasingly capable paths."""
    
    @dataclass
    class EscalationLevel:
        name: str
        handler: Callable
        quality_checker: Callable[[str], float]  # output → quality score
        cost: float  # estimated cost in USD
        max_latency_ms: float
    
    def __init__(self, quality_threshold: float = 0.7):
        self.levels: list["EscalationRouter.EscalationLevel"] = []
        self.quality_threshold = quality_threshold
    
    def add_level(self, level: "EscalationRouter.EscalationLevel"):
        self.levels.append(level)
    
    async def execute(self, input_data: dict) -> dict:
        """Try each escalation level until quality threshold is met."""
        attempts = []
        
        for level in self.levels:
            try:
                output = await asyncio.wait_for(
                    level.handler(input_data),
                    timeout=level.max_latency_ms / 1000
                )
                quality = level.quality_checker(output)
                
                attempts.append({
                    "level": level.name,
                    "output": output,
                    "quality": quality,
                    "cost": level.cost
                })
                
                if quality >= self.quality_threshold:
                    return {
                        "final_output": output,
                        "served_by": level.name,
                        "quality": quality,
                        "total_cost": sum(a["cost"] for a in attempts),
                        "escalation_depth": len(attempts),
                        "attempts": attempts
                    }
            except (asyncio.TimeoutError, Exception) as e:
                attempts.append({
                    "level": level.name,
                    "error": str(e),
                    "cost": level.cost
                })
                continue
        
        # All levels failed: return best attempt or escalate to human
        valid_attempts = [a for a in attempts if "output" in a]
        if valid_attempts:
            best = max(valid_attempts, key=lambda a: a["quality"])
            return {
                "final_output": best["output"],
                "served_by": best["level"],
                "quality": best["quality"],
                "warning": "Below quality threshold — best effort",
                "total_cost": sum(a.get("cost", 0) for a in attempts),
                "attempts": attempts
            }
        
        return {
            "final_output": None,
            "escalate_to": "human",
            "all_attempts_failed": True,
            "attempts": attempts
        }


# --- Escalation ladder ---
escalation = EscalationRouter(quality_threshold=0.75)

escalation.add_level(EscalationRouter.EscalationLevel(
    name="cache_lookup",
    handler=cache_handler,
    quality_checker=lambda o: 1.0 if o else 0.0,
    cost=0.0,
    max_latency_ms=50
))

escalation.add_level(EscalationRouter.EscalationLevel(
    name="small_model_direct",
    handler=small_model_handler,
    quality_checker=quality_scorer,
    cost=0.001,
    max_latency_ms=2000
))

escalation.add_level(EscalationRouter.EscalationLevel(
    name="large_model_chain",
    handler=large_model_chain_handler,
    quality_checker=quality_scorer,
    cost=0.05,
    max_latency_ms=15000
))

escalation.add_level(EscalationRouter.EscalationLevel(
    name="agent_with_tools",
    handler=agent_handler,
    quality_checker=quality_scorer,
    cost=0.50,
    max_latency_ms=60000
))
```

### 1.6.3.7 Composite Routing Architecture

Production systems combine multiple routing dimensions into a **multi-stage routing pipeline**:

```
User Input
    │
    ▼
┌───────────────────────────────┐
│ Stage 1: Safety/Guardrail     │  ← Static (keyword + regex)
│ Block harmful inputs          │
└──────────┬────────────────────┘
           │ pass
           ▼
┌───────────────────────────────┐
│ Stage 2: Intent Classification│  ← LLM or classifier
│ What does the user want?      │
└──────────┬────────────────────┘
           │ intent
           ▼
┌───────────────────────────────┐
│ Stage 3: Topic Routing        │  ← Embedding similarity
│ What domain is this about?    │
└──────────┬────────────────────┘
           │ topic + intent
           ▼
┌───────────────────────────────┐
│ Stage 4: Complexity Estimation│  ← Heuristic + optional LLM
│ How hard is this task?        │
└──────────┬────────────────────┘
           │ complexity level
           ▼
┌───────────────────────────────┐
│ Stage 5: Model Selection      │  ← Cost-quality optimization
│ Which model for this step?    │
└──────────┬────────────────────┘
           │ model + chain
           ▼
┌───────────────────────────────┐
│ Stage 6: Execution            │  ← With escalation fallback
│ Run chain; escalate if needed │
└───────────────────────────────┘
```

```python
class MultiStageRouter:
    """Production-grade multi-stage routing pipeline."""
    
    def __init__(self,
                 safety_router: StaticRouter,
                 intent_router: IntentRouter,
                 topic_router: TopicRouter,
                 complexity_router: ComplexityRouter,
                 model_router: ModelRouter,
                 escalation_router: EscalationRouter):
        self.safety = safety_router
        self.intent = intent_router
        self.topic = topic_router
        self.complexity = complexity_router
        self.model = model_router
        self.escalation = escalation_router
    
    async def route(self, user_input: str) -> dict:
        routing_trace = {"input": user_input, "stages": []}
        
        # Stage 1: Safety check (static, ~0ms)
        safety_result, _ = self.safety.route({"query": user_input})
        routing_trace["stages"].append({"safety": safety_result})
        if safety_result == "blocked":
            return {"action": "block", "reason": "Safety filter", "trace": routing_trace}
        
        # Stage 2: Intent classification (~200-500ms if LLM)
        intent_result = await self.intent.classify_and_route(user_input)
        routing_trace["stages"].append({"intent": intent_result})
        if intent_result["action"] == "clarify":
            return {**intent_result, "trace": routing_trace}
        
        # Stage 3: Topic routing (~5ms with embeddings)
        topic, topic_score, all_topics = self.topic.route(user_input)
        routing_trace["stages"].append({
            "topic": topic, "score": topic_score, "all_scores": all_topics
        })
        
        # Stage 4: Complexity estimation (~0ms heuristic)
        tier, complexity = self.complexity.route(user_input)
        routing_trace["stages"].append({"tier": tier, "complexity": complexity})
        
        # Stage 5: Model selection (~0ms)
        model = self.model.select({
            "min_capability": 0.7 if tier == "direct" else 0.85 if tier == "chain" else 0.93,
            "needs_tools": tier == "agent",
            "priority": "cost" if tier == "direct" else "quality"
        })
        routing_trace["stages"].append({"model": model.name})
        
        return {
            "action": "execute",
            "intent": intent_result["intent"],
            "topic": topic,
            "complexity_tier": tier,
            "model": model,
            "trace": routing_trace
        }
```

---

## 1.6.4 Gate Mechanisms

### 1.6.4.1 Foundational Definition

A **gate** is a binary or graded decision function inserted between chain steps that determines whether execution should **proceed**, **retry**, **branch**, or **halt**:

$$
g: \mathcal{Y} \times \mathcal{S} \longrightarrow \{\texttt{PASS}, \texttt{FAIL}\} \times \mathcal{A}_{\text{action}}
$$

where $\mathcal{Y}$ is the space of step outputs, $\mathcal{S}$ is the state space, and $\mathcal{A}_{\text{action}} = \{\texttt{proceed}, \texttt{retry}, \texttt{retry\_modified}, \texttt{escalate}, \texttt{abort}, \texttt{human\_review}\}$ is the set of possible gate actions.

**Formally, a gate acts as a binary classifier with an action policy:**

$$
g(y_t) = \begin{cases}
(\texttt{PASS}, \texttt{proceed}) & \text{if } Q(y_t) \geq \tau \\
(\texttt{FAIL}, \pi_{\text{fail}}(y_t, S_t)) & \text{if } Q(y_t) < \tau
\end{cases}
$$

where $Q: \mathcal{Y} \rightarrow [0, 1]$ is a quality function and $\pi_{\text{fail}}$ is the failure-handling policy.

**Chain topology with gates:**

```
[Step 1] → [Gate 1] → [Step 2] → [Gate 2] → [Step 3] → [Gate 3] → [Output]
              │                      │                      │
              ▼                      ▼                      ▼
          ┌───────┐              ┌───────┐              ┌───────┐
          │ Retry │              │ Retry │              │ Human │
          │ Loop  │              │ Loop  │              │Review │
          └───────┘              └───────┘              └───────┘
```

### 1.6.4.2 Quality Gates Between Chain Steps

Quality gates evaluate whether a step's output meets a minimum quality standard before allowing execution to continue.

**Multi-dimensional quality function:**

$$
Q(y_t) = \sum_{d=1}^{D} w_d \cdot q_d(y_t) \quad \text{where } \sum_{d=1}^{D} w_d = 1
$$

| Quality Dimension $q_d$ | Measurement Method | Typical Weight |
|---|---|---|
| **Relevance** | Semantic similarity to task specification | 0.30 |
| **Completeness** | Required fields present / all sub-questions answered | 0.25 |
| **Coherence** | Logical consistency, no contradictions | 0.20 |
| **Format compliance** | Matches expected schema / structure | 0.15 |
| **Length appropriateness** | Within expected token range | 0.10 |

```python
class QualityGate:
    """Multi-dimensional quality gate between chain steps."""
    
    @dataclass
    class QualityCheck:
        name: str
        checker: Callable[[str, dict], float]  # (output, state) → score [0,1]
        weight: float
        min_score: float = 0.0  # hard minimum for this dimension
    
    def __init__(self, threshold: float = 0.7, max_retries: int = 2):
        self.threshold = threshold
        self.max_retries = max_retries
        self.checks: list[QualityGate.QualityCheck] = []
        self._gate_log: list[dict] = []
    
    def add_check(self, check: "QualityGate.QualityCheck"):
        self.checks.append(check)
        # Renormalize weights
        total = sum(c.weight for c in self.checks)
        for c in self.checks:
            c.weight /= total
    
    async def evaluate(self, output: str, state: dict) -> dict:
        """Evaluate output quality across all dimensions."""
        scores = {}
        failed_hard = []
        
        for check in self.checks:
            score = check.checker(output, state)
            scores[check.name] = {
                "score": score,
                "weight": check.weight,
                "weighted_score": score * check.weight,
                "passed_minimum": score >= check.min_score
            }
            if score < check.min_score:
                failed_hard.append(check.name)
        
        aggregate = sum(s["weighted_score"] for s in scores.values())
        
        passed = aggregate >= self.threshold and len(failed_hard) == 0
        
        result = {
            "passed": passed,
            "aggregate_score": aggregate,
            "dimension_scores": scores,
            "failed_hard_checks": failed_hard,
            "threshold": self.threshold
        }
        self._gate_log.append(result)
        return result
    
    async def gate(self, output: str, state: dict,
                   retry_fn: Callable = None) -> tuple[str, dict]:
        """Apply the gate with retry logic."""
        for attempt in range(self.max_retries + 1):
            eval_result = await self.evaluate(output, state)
            
            if eval_result["passed"]:
                return output, {**eval_result, "attempts": attempt + 1}
            
            if attempt < self.max_retries and retry_fn:
                # Construct feedback for retry
                feedback = self._build_feedback(eval_result)
                output = await retry_fn(output, feedback, state)
            else:
                return output, {
                    **eval_result, 
                    "attempts": attempt + 1,
                    "action": "escalate" if not eval_result["passed"] else "proceed"
                }
        
        return output, eval_result
    
    def _build_feedback(self, eval_result: dict) -> str:
        """Generate specific feedback for retry based on failed checks."""
        feedback_parts = ["The previous output did not meet quality standards:"]
        
        for name, scores in eval_result["dimension_scores"].items():
            if scores["score"] < 0.6:
                feedback_parts.append(
                    f"- {name}: scored {scores['score']:.2f} "
                    f"(minimum: {self.threshold:.2f}). Please improve this aspect."
                )
        
        if eval_result["failed_hard_checks"]:
            feedback_parts.append(
                f"Critical failures in: {', '.join(eval_result['failed_hard_checks'])}"
            )
        
        return "\n".join(feedback_parts)
```

### 1.6.4.3 Validation Gates (Schema, Factuality, Safety)

Validation gates enforce **hard constraints** — outputs that violate these constraints are categorically unacceptable, regardless of other quality dimensions.

#### A. Schema Validation Gate

Ensures the output conforms to a predefined structure:

$$
g_{\text{schema}}(y_t) = \begin{cases}
\texttt{PASS} & \text{if } y_t \in \mathcal{L}(\sigma) \\
\texttt{FAIL} & \text{otherwise}
\end{cases}
$$

where $\mathcal{L}(\sigma)$ is the language (set of valid instances) defined by schema $\sigma$.

```python
from pydantic import ValidationError

class SchemaValidationGate:
    """Ensure output conforms to a Pydantic schema."""
    
    def __init__(self, schema: type[BaseModel], max_retries: int = 3):
        self.schema = schema
        self.max_retries = max_retries
    
    async def validate(self, raw_output: str, 
                       retry_fn: Callable = None) -> tuple[BaseModel | None, dict]:
        errors = []
        
        for attempt in range(self.max_retries + 1):
            try:
                # Attempt to parse
                parsed = self.schema.model_validate_json(raw_output)
                return parsed, {
                    "valid": True,
                    "attempts": attempt + 1,
                    "errors": errors
                }
            except ValidationError as e:
                error_detail = e.errors()
                errors.append({
                    "attempt": attempt + 1,
                    "errors": error_detail
                })
                
                if attempt < self.max_retries and retry_fn:
                    feedback = (
                        f"Your output failed schema validation:\n"
                        f"{json.dumps(error_detail, indent=2)}\n\n"
                        f"Expected schema:\n"
                        f"{self.schema.model_json_schema()}\n\n"
                        f"Please fix the output to match the schema exactly."
                    )
                    raw_output = await retry_fn(raw_output, feedback)
            except json.JSONDecodeError as e:
                errors.append({"attempt": attempt + 1, "error": f"Invalid JSON: {e}"})
                if attempt < self.max_retries and retry_fn:
                    raw_output = await retry_fn(
                        raw_output, 
                        f"Output is not valid JSON: {e}. Please output valid JSON."
                    )
        
        return None, {"valid": False, "attempts": self.max_retries + 1, "errors": errors}
```

#### B. Factuality Validation Gate

Checks whether claims in the output are supported by provided evidence:

$$
g_{\text{factual}}(y_t, \mathcal{E}) = \begin{cases}
\texttt{PASS} & \text{if } \forall\, c \in \text{claims}(y_t):\; \exists\, e \in \mathcal{E} \text{ s.t. } \text{supports}(e, c) \\
\texttt{FAIL} & \text{otherwise}
\end{cases}
$$

```python
class FactualityGate:
    """Verify that output claims are supported by provided evidence."""
    
    def __init__(self, llm: "LLMClient", strictness: float = 0.8):
        self.llm = llm
        self.strictness = strictness
    
    async def validate(self, output: str, evidence: list[str]) -> dict:
        # Step 1: Extract claims from output
        claims = await self._extract_claims(output)
        
        # Step 2: Verify each claim against evidence
        verification_results = []
        for claim in claims:
            result = await self._verify_claim(claim, evidence)
            verification_results.append(result)
        
        # Step 3: Aggregate
        if not verification_results:
            return {"passed": True, "reason": "No verifiable claims found"}
        
        supported_ratio = sum(
            1 for r in verification_results if r["supported"]
        ) / len(verification_results)
        
        return {
            "passed": supported_ratio >= self.strictness,
            "supported_ratio": supported_ratio,
            "total_claims": len(claims),
            "unsupported_claims": [
                r["claim"] for r in verification_results if not r["supported"]
            ],
            "details": verification_results
        }
    
    async def _extract_claims(self, text: str) -> list[str]:
        prompt = (
            "Extract all factual claims from the following text. "
            "List each claim as a separate, self-contained statement.\n\n"
            f"Text: {text}\n\n"
            "Claims (one per line):"
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return [c.strip("- ").strip() for c in response.strip().split("\n") if c.strip()]
    
    async def _verify_claim(self, claim: str, evidence: list[str]) -> dict:
        evidence_text = "\n---\n".join(evidence)
        prompt = (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Is this claim supported by the evidence? "
            f"Respond as JSON: {{\"supported\": true/false, \"reasoning\": \"...\", "
            f"\"supporting_evidence\": \"...\" or null}}"
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        result = json.loads(response)
        result["claim"] = claim
        return result
```

#### C. Safety Validation Gate

Blocks outputs containing harmful, biased, or policy-violating content:

$$
g_{\text{safety}}(y_t) = \begin{cases}
\texttt{PASS} & \text{if } \forall\, d \in \mathcal{D}_{\text{safety}}:\; h_d(y_t) < \tau_d \\
\texttt{FAIL} & \text{if } \exists\, d \in \mathcal{D}_{\text{safety}}:\; h_d(y_t) \geq \tau_d
\end{cases}
$$

where $\mathcal{D}_{\text{safety}}$ is the set of safety dimensions (toxicity, bias, PII, etc.) and $h_d$ is a hazard scorer for dimension $d$.

```python
class SafetyGate:
    """Multi-dimensional safety validation gate."""
    
    SAFETY_DIMENSIONS = [
        "toxicity",
        "personal_identifiable_information",
        "medical_advice",
        "financial_advice",
        "legal_advice",
        "bias_discrimination",
        "misinformation"
    ]
    
    def __init__(self, llm: "LLMClient", 
                 thresholds: dict[str, float] = None):
        self.llm = llm
        self.thresholds = thresholds or {d: 0.5 for d in self.SAFETY_DIMENSIONS}
    
    async def validate(self, output: str) -> dict:
        prompt = (
            "Evaluate the following text for safety concerns. "
            "For each dimension, provide a risk score from 0.0 (safe) to 1.0 (unsafe).\n\n"
            f"Text: {output}\n\n"
            "Dimensions to evaluate:\n"
            + "\n".join(f"- {d}" for d in self.SAFETY_DIMENSIONS) +
            "\n\nRespond as JSON: {\"dimension_name\": score, ...}"
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        scores = json.loads(response)
        
        violations = []
        for dim, threshold in self.thresholds.items():
            score = scores.get(dim, 0.0)
            if score >= threshold:
                violations.append({
                    "dimension": dim,
                    "score": score,
                    "threshold": threshold
                })
        
        return {
            "passed": len(violations) == 0,
            "scores": scores,
            "violations": violations,
            "action": "proceed" if not violations else "block"
        }
```

### 1.6.4.4 Human-in-the-Loop Gates

Human gates insert a **human decision point** into the chain, pausing automated execution until a human approves, rejects, or modifies the intermediate result.

**Formal model:**

$$
g_{\text{human}}(y_t) = \text{await}\; h_{\text{human}}(y_t) \in \{\texttt{approve}, \texttt{reject}, \texttt{modify}(y_t')\}
$$

**When to insert human gates:**

| Criterion | Trigger Condition |
|---|---|
| **High-stakes output** | Financial transactions, legal documents, medical advice |
| **Low model confidence** | $\text{confidence}(y_t) < \tau_{\text{human}}$ |
| **Novel situation** | Input is far from training distribution (OOD detection) |
| **Policy requirement** | Regulatory or compliance mandate |
| **Quality gate failure** | Automated retries exhausted without meeting threshold |

```python
import asyncio
from enum import Enum

class HumanDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"

@dataclass
class HumanReview:
    decision: HumanDecision
    modified_output: str | None = None
    reviewer_notes: str = ""
    reviewer_id: str = ""
    timestamp: str = ""

class HumanInTheLoopGate:
    """Pause chain execution for human review and approval."""
    
    def __init__(self, review_queue: "AsyncQueue",
                 timeout_seconds: float = 3600,
                 auto_approve_after_timeout: bool = False):
        self.review_queue = review_queue
        self.timeout = timeout_seconds
        self.auto_approve = auto_approve_after_timeout
    
    async def request_review(self, output: str, context: dict,
                              urgency: str = "normal") -> HumanReview:
        """Submit output for human review and wait for decision."""
        review_request = {
            "id": str(uuid4()),
            "output": output,
            "context": context,
            "urgency": urgency,
            "submitted_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        # Submit to review queue (webhook, Slack, email, dashboard)
        await self.review_queue.put(review_request)
        
        try:
            # Wait for human response
            review = await asyncio.wait_for(
                self._wait_for_decision(review_request["id"]),
                timeout=self.timeout
            )
            return review
        except asyncio.TimeoutError:
            if self.auto_approve:
                return HumanReview(
                    decision=HumanDecision.APPROVE,
                    reviewer_notes="Auto-approved after timeout"
                )
            else:
                return HumanReview(
                    decision=HumanDecision.REJECT,
                    reviewer_notes=f"Rejected: no review within {self.timeout}s"
                )
    
    async def gate(self, output: str, state: dict) -> tuple[str, dict]:
        """Apply human-in-the-loop gate."""
        review = await self.request_review(
            output=output,
            context={
                "step": state.get("step_index"),
                "task": state.get("query"),
                "confidence": state.get("confidence", "unknown")
            }
        )
        
        if review.decision == HumanDecision.APPROVE:
            return output, {"action": "proceed", "reviewer": review.reviewer_id}
        elif review.decision == HumanDecision.MODIFY:
            return review.modified_output, {
                "action": "proceed_modified",
                "reviewer": review.reviewer_id,
                "notes": review.reviewer_notes
            }
        else:  # REJECT
            return output, {
                "action": "abort",
                "reviewer": review.reviewer_id,
                "reason": review.reviewer_notes
            }
    
    async def _wait_for_decision(self, request_id: str) -> HumanReview:
        """Poll or listen for a human decision on the given request."""
        # Implementation depends on infrastructure:
        # - Webhook callback
        # - Database polling
        # - WebSocket subscription
        raise NotImplementedError("Implement per deployment infrastructure")
```

### 1.6.4.5 Approval Workflows Within Chains

For complex enterprise applications, gates are organized into **multi-stage approval workflows** where different reviewers or validators must approve at different stages.

```python
@dataclass
class ApprovalStage:
    name: str
    gate: Callable  # any gate type: quality, schema, safety, human
    required: bool = True  # if False, failure is logged but doesn't block
    on_fail: str = "retry"  # retry | escalate | abort | skip

class ApprovalWorkflow:
    """Multi-stage approval workflow embedded within a chain."""
    
    def __init__(self, stages: list[ApprovalStage]):
        self.stages = stages
    
    async def run(self, output: str, state: dict) -> tuple[str, dict]:
        """Execute all approval stages sequentially."""
        workflow_log = []
        current_output = output
        
        for stage in self.stages:
            result_output, gate_result = await stage.gate(current_output, state)
            
            stage_log = {
                "stage": stage.name,
                "passed": gate_result.get("passed", gate_result.get("action") == "proceed"),
                "details": gate_result
            }
            workflow_log.append(stage_log)
            
            if stage_log["passed"]:
                current_output = result_output
                continue
            
            # Handle failure
            if not stage.required:
                # Non-blocking gate: log and continue
                stage_log["skipped"] = True
                continue
            
            if stage.on_fail == "abort":
                return current_output, {
                    "workflow_passed": False,
                    "failed_at": stage.name,
                    "log": workflow_log
                }
            elif stage.on_fail == "escalate":
                return current_output, {
                    "workflow_passed": False,
                    "escalate": True,
                    "failed_at": stage.name,
                    "log": workflow_log
                }
            # "retry" is handled within the gate itself
        
        return current_output, {
            "workflow_passed": True,
            "log": workflow_log
        }


# --- Example: Document Generation Approval Workflow ---

workflow = ApprovalWorkflow(stages=[
    ApprovalStage(
        name="schema_check",
        gate=schema_gate.gate,
        required=True,
        on_fail="retry"
    ),
    ApprovalStage(
        name="factuality_check",
        gate=factuality_gate.validate_as_gate,
        required=True,
        on_fail="retry"
    ),
    ApprovalStage(
        name="safety_check",
        gate=safety_gate.validate_as_gate,
        required=True,
        on_fail="abort"  # safety violations are non-negotiable
    ),
    ApprovalStage(
        name="quality_review",
        gate=quality_gate.gate,
        required=True,
        on_fail="retry"
    ),
    ApprovalStage(
        name="human_approval",
        gate=human_gate.gate,
        required=True,
        on_fail="escalate"
    )
])
```

### 1.6.4.6 Gate as a Binary Classifier — Formal Analysis

Treating the gate $g$ as a binary classifier enables rigorous analysis using standard classification metrics.

**Definition.** Let $g: \mathcal{Y} \rightarrow \{0, 1\}$ where $g(y) = 1$ means PASS and $g(y) = 0$ means FAIL. Let $y^*$ be the ground-truth quality label (1 = genuinely good output, 0 = genuinely bad output).

**Confusion matrix:**

$$
\begin{array}{c|cc}
 & g(y) = 1 \; (\text{PASS}) & g(y) = 0 \; (\text{FAIL}) \\
\hline
y^* = 1 \; (\text{good}) & \text{True Positive (TP)} & \text{False Negative (FN)} \\
y^* = 0 \; (\text{bad}) & \text{False Positive (FP)} & \text{True Negative (TN)}
\end{array}
$$

**Critical metrics for gates:**

| Metric | Formula | Interpretation for Gates |
|---|---|---|
| **Precision** | $\frac{TP}{TP + FP}$ | Of outputs that pass the gate, what fraction are truly good? |
| **Recall** | $\frac{TP}{TP + FN}$ | Of truly good outputs, what fraction does the gate let through? |
| **False Pass Rate** (FPR) | $\frac{FP}{FP + TN}$ | Fraction of bad outputs that incorrectly pass (most dangerous) |
| **False Block Rate** (FNR) | $\frac{FN}{FN + TP}$ | Fraction of good outputs incorrectly blocked (wastes compute) |

**The fundamental gate trade-off:**

$$
\tau \uparrow \;\Rightarrow\; \text{FPR} \downarrow,\; \text{FNR} \uparrow \quad (\text{stricter gate: fewer bad pass, more good blocked})
$$

$$
\tau \downarrow \;\Rightarrow\; \text{FPR} \uparrow,\; \text{FNR} \downarrow \quad (\text{lenient gate: more bad pass, fewer good blocked})
$$

**Optimal threshold selection.** The optimal $\tau^*$ minimizes expected cost:

$$
\tau^* = \arg\min_\tau \; \bigl[\, c_{\text{FP}} \cdot \text{FPR}(\tau) + c_{\text{FN}} \cdot \text{FNR}(\tau) \,\bigr]
$$

where $c_{\text{FP}}$ is the cost of passing a bad output (user-facing error, reputational damage) and $c_{\text{FN}}$ is the cost of blocking a good output (retry cost, latency).

In high-stakes applications (medical, legal, financial): $c_{\text{FP}} \gg c_{\text{FN}}$, so set $\tau$ high.
In low-stakes applications (creative writing, brainstorming): $c_{\text{FP}} \approx c_{\text{FN}}$, so set $\tau$ moderate.

```python
class GateAnalyzer:
    """Analyze gate performance using classification metrics."""
    
    def __init__(self):
        self.predictions: list[bool] = []  # gate decisions
        self.ground_truth: list[bool] = []  # actual quality (from human review)
    
    def record(self, gate_passed: bool, actually_good: bool):
        self.predictions.append(gate_passed)
        self.ground_truth.append(actually_good)
    
    def compute_metrics(self) -> dict:
        tp = sum(p and g for p, g in zip(self.predictions, self.ground_truth))
        fp = sum(p and not g for p, g in zip(self.predictions, self.ground_truth))
        fn = sum(not p and g for p, g in zip(self.predictions, self.ground_truth))
        tn = sum(not p and not g for p, g in zip(self.predictions, self.ground_truth))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        
        return {
            "precision": precision,
            "recall": recall,
            "false_pass_rate": fpr,
            "false_block_rate": fnr,
            "f1": 2 * precision * recall / max(precision + recall, 1e-8),
            "total_samples": len(self.predictions),
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        }
    
    def suggest_threshold(self, cost_false_pass: float, 
                           cost_false_block: float) -> float:
        """Suggest optimal threshold based on asymmetric costs."""
        # Theoretical: threshold where marginal cost of FP = marginal cost of FN
        # Approximation: bias toward strictness when FP cost is high
        ratio = cost_false_pass / (cost_false_pass + cost_false_block)
        return ratio  # higher ratio → higher (stricter) threshold
```

---

## Summary: Routing and Gating Decision Framework

| Decision | Criteria | Recommended Approach |
|---|---|---|
| **Routing type** | Input categories are well-defined and stable? | **Static** (regex, keyword, threshold) |
| | Categories are fuzzy or evolving? | **Dynamic** (embedding or LLM-based) |
| | Multiple routing signals available? | **Ensemble** (weighted voting) |
| **Routing dimension** | What does the user want? | **Intent** router |
| | What is the topic? | **Topic** router (embedding-based) |
| | How hard is the task? | **Complexity** router |
| | Which model should handle it? | **Model selector** router |
| | What if everything fails? | **Escalation** router with fallback |
| **Gate type** | Output must match a specific structure? | **Schema** validation gate |
| | Output must be factually grounded? | **Factuality** gate |
| | Output must be safe/compliant? | **Safety** gate (hard block on violation) |
| | High-stakes or uncertain output? | **Human-in-the-loop** gate |
| | Multiple gate dimensions? | **Approval workflow** (sequential multi-gate) |
| **Threshold calibration** | High cost of bad output reaching user? | High $\tau$ (strict) — prioritize precision |
| | High cost of unnecessary retries? | Low $\tau$ (lenient) — prioritize recall |
| | Both costs balanced? | Optimize $\tau^*$ via cost-weighted objective |