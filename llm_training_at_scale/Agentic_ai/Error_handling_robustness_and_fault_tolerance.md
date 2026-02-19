

# 1.7 Error Handling, Robustness, and Fault Tolerance

> **Scope.** Every production prompt chain operates in a stochastic, failure-prone environment: LLM outputs are non-deterministic, APIs timeout, schemas are violated, and errors in early steps silently corrupt downstream reasoning. Robust chain engineering requires a systematic taxonomy of failure modes, formal analysis of error propagation dynamics, principled mitigation strategies, and self-healing mechanisms that detect and correct errors without human intervention. This section treats error handling not as an afterthought but as a **first-class architectural concern** on par with the happy-path design.

---

## 1.7.1 Error Taxonomy in Prompt Chains

### 1.7.1.1 Formal Error Model

Define the error state of a chain at step $t$ as a random variable $\mathcal{E}_t$ drawn from a structured error space:

$$
\mathcal{E}_t \in \{\texttt{NONE}\} \cup \mathcal{E}_{\text{input}} \cup \mathcal{E}_{\text{gen}} \cup \mathcal{E}_{\text{parse}} \cup \mathcal{E}_{\text{prop}} \cup \mathcal{E}_{\text{infra}} \cup \mathcal{E}_{\text{state}}
$$

Each error category has distinct **detection methods**, **recovery strategies**, and **downstream impact profiles**.

### 1.7.1.2 Category A — Input Errors ($\mathcal{E}_{\text{input}}$)

Errors in the data arriving at a chain step, either from the user or from a predecessor step.

| Sub-Type | Description | Detection | Example |
|---|---|---|---|
| **Malformed input** | Data violates expected format or schema | Schema validation (Pydantic, JSON Schema) | JSON with missing required fields |
| **Adversarial input** | Deliberately crafted to exploit chain behavior | Input sanitization, prompt injection detection | "Ignore previous instructions and..." |
| **Out-of-distribution (OOD)** | Input is semantically valid but outside the domain the chain was designed for | Embedding distance from training distribution centroid | Medical question routed to a finance chain |
| **Empty/null input** | Predecessor step produced no output | Null/empty check | Tool call returned empty response |
| **Encoding errors** | Character encoding issues, Unicode artifacts | Encoding validation | Mojibake from API response |

**Formal OOD detection.** Given a reference distribution of valid inputs $\{x_1, \dots, x_N\}$ with centroid $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ in embedding space:

$$
\text{OOD}(x) = \mathbb{1}\!\bigl[\, (e(x) - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (e(x) - \boldsymbol{\mu}) > \chi^2_{d, \alpha} \,\bigr]
$$

where $\chi^2_{d,\alpha}$ is the chi-squared critical value at confidence level $\alpha$ with $d$ degrees of freedom (the Mahalanobis distance test).

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

class ErrorCategory(Enum):
    INPUT_MALFORMED = auto()
    INPUT_ADVERSARIAL = auto()
    INPUT_OOD = auto()
    INPUT_EMPTY = auto()
    GEN_HALLUCINATION = auto()
    GEN_FORMAT_VIOLATION = auto()
    GEN_REFUSAL = auto()
    GEN_TRUNCATION = auto()
    PARSE_SCHEMA = auto()
    PARSE_TYPE = auto()
    PROPAGATION = auto()
    INFRA_TIMEOUT = auto()
    INFRA_RATE_LIMIT = auto()
    INFRA_API_ERROR = auto()
    STATE_CORRUPTION = auto()

@dataclass
class ChainError:
    category: ErrorCategory
    step_index: int
    step_name: str
    message: str
    raw_output: Optional[str] = None
    recoverable: bool = True
    severity: str = "medium"  # low, medium, high, critical
    context: dict = field(default_factory=dict)
    
    @property
    def is_transient(self) -> bool:
        """Transient errors may resolve on retry."""
        return self.category in {
            ErrorCategory.INFRA_TIMEOUT,
            ErrorCategory.INFRA_RATE_LIMIT,
            ErrorCategory.INFRA_API_ERROR,
            ErrorCategory.GEN_TRUNCATION
        }


class InputValidator:
    """Comprehensive input validation for chain steps."""
    
    def __init__(self, embed_fn=None, reference_embeddings=None,
                 ood_threshold: float = 3.0):
        self.embed_fn = embed_fn
        self.reference_embeddings = reference_embeddings
        self.ood_threshold = ood_threshold
        
        # Precompute reference distribution parameters
        if reference_embeddings is not None:
            import numpy as np
            self.mu = np.mean(reference_embeddings, axis=0)
            self.cov_inv = np.linalg.inv(
                np.cov(reference_embeddings.T) + 1e-6 * np.eye(reference_embeddings.shape[1])
            )
    
    def validate(self, input_data: Any, schema: type = None,
                 step_name: str = "") -> list[ChainError]:
        errors = []
        
        # Check empty/null
        if input_data is None or (isinstance(input_data, str) and not input_data.strip()):
            errors.append(ChainError(
                category=ErrorCategory.INPUT_EMPTY,
                step_index=-1, step_name=step_name,
                message="Input is empty or null",
                recoverable=False, severity="high"
            ))
            return errors
        
        # Schema validation
        if schema is not None and isinstance(input_data, str):
            try:
                schema.model_validate_json(input_data)
            except Exception as e:
                errors.append(ChainError(
                    category=ErrorCategory.INPUT_MALFORMED,
                    step_index=-1, step_name=step_name,
                    message=f"Schema validation failed: {e}",
                    raw_output=str(input_data)[:500],
                    recoverable=True, severity="medium"
                ))
        
        # Adversarial input detection
        if isinstance(input_data, str):
            adversarial_patterns = [
                r"ignore\s+(all\s+)?previous\s+instructions",
                r"you\s+are\s+now\s+(?:a|an)\s+",
                r"system\s*:\s*",
                r"<\|im_start\|>",
                r"\[INST\]",
            ]
            import re
            for pattern in adversarial_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    errors.append(ChainError(
                        category=ErrorCategory.INPUT_ADVERSARIAL,
                        step_index=-1, step_name=step_name,
                        message=f"Potential prompt injection detected: {pattern}",
                        recoverable=False, severity="critical"
                    ))
        
        # OOD detection
        if self.embed_fn and self.reference_embeddings is not None and isinstance(input_data, str):
            import numpy as np
            emb = self.embed_fn(input_data)
            diff = emb - self.mu
            mahal_dist = float(np.sqrt(diff @ self.cov_inv @ diff))
            if mahal_dist > self.ood_threshold:
                errors.append(ChainError(
                    category=ErrorCategory.INPUT_OOD,
                    step_index=-1, step_name=step_name,
                    message=f"Input is out-of-distribution (Mahalanobis={mahal_dist:.2f})",
                    recoverable=True, severity="medium",
                    context={"mahalanobis_distance": mahal_dist}
                ))
        
        return errors
```

### 1.7.1.3 Category B — LLM Generation Errors ($\mathcal{E}_{\text{gen}}$)

Errors in the LLM's output itself, independent of how it is parsed.

| Sub-Type | Description | Detection Method | Severity |
|---|---|---|---|
| **Hallucination** | Fabricated facts, non-existent references, invented data | RAG verification, fact-checking gate, citation validation | High |
| **Format violation** | Output doesn't follow requested structure (e.g., asked for JSON, got prose) | Regex/schema check, structured output mode | Medium |
| **Refusal** | Model declines to answer ("I can't help with that") | Refusal pattern detection | Medium |
| **Truncation** | Output cut off due to max-token limit | Check for `finish_reason == "length"` | Low–Medium |
| **Repetition/degeneration** | Model enters repetitive loop | N-gram repetition detector, perplexity spike | Medium |
| **Instruction drift** | Model ignores or reinterprets the prompt instruction | Semantic similarity between output and instruction intent | Medium |

```python
class GenerationErrorDetector:
    """Detect errors in LLM-generated outputs."""
    
    REFUSAL_PATTERNS = [
        r"i (?:can't|cannot|am unable to|won't|will not)",
        r"i'm (?:sorry|afraid|not able)",
        r"as an ai",
        r"i don't (?:have|think) (?:that's|it's) appropriate",
        r"it (?:is|would be) (?:unethical|inappropriate|against)",
    ]
    
    def detect_all(self, output: str, expected_format: str = None,
                   finish_reason: str = None,
                   max_repetition_ratio: float = 0.3) -> list[ChainError]:
        errors = []
        
        # Refusal detection
        import re
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, output.lower()):
                errors.append(ChainError(
                    category=ErrorCategory.GEN_REFUSAL,
                    step_index=-1, step_name="",
                    message=f"Model refused to respond: matched '{pattern}'",
                    raw_output=output[:200],
                    recoverable=True, severity="medium"
                ))
                break
        
        # Truncation detection
        if finish_reason == "length":
            errors.append(ChainError(
                category=ErrorCategory.GEN_TRUNCATION,
                step_index=-1, step_name="",
                message="Output truncated due to max_tokens limit",
                raw_output=output[-100:],
                recoverable=True, severity="low"
            ))
        
        # Format violation detection
        if expected_format == "json":
            try:
                import json
                json.loads(output)
            except json.JSONDecodeError as e:
                errors.append(ChainError(
                    category=ErrorCategory.GEN_FORMAT_VIOLATION,
                    step_index=-1, step_name="",
                    message=f"Expected JSON, got invalid: {e}",
                    raw_output=output[:300],
                    recoverable=True, severity="medium"
                ))
        
        # Repetition detection
        words = output.split()
        if len(words) > 20:
            # Check for repeated n-grams (n=5)
            ngrams = [tuple(words[i:i+5]) for i in range(len(words)-4)]
            unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)
            if unique_ratio < (1 - max_repetition_ratio):
                errors.append(ChainError(
                    category=ErrorCategory.GEN_HALLUCINATION,
                    step_index=-1, step_name="",
                    message=f"Repetition detected: unique 5-gram ratio = {unique_ratio:.2f}",
                    raw_output=output[:200],
                    recoverable=True, severity="medium"
                ))
        
        return errors
```

### 1.7.1.4 Category C — Parsing Errors ($\mathcal{E}_{\text{parse}}$)

Errors that occur when attempting to extract structured data from the LLM's text output.

$$
\text{Parse Error} = \text{output} \notin \mathcal{L}(\sigma_{\text{expected}})
$$

where $\mathcal{L}(\sigma)$ is the language of valid strings under schema $\sigma$.

**Common parse error sub-types:**

| Sub-Type | Cause | Example |
|---|---|---|
| **Schema mismatch** | Output has wrong field names, missing required fields | `{"answer": "..."}` when `{"response": "..."}` expected |
| **Type mismatch** | Field value has wrong type | `"confidence": "high"` when `float` expected |
| **Extra text** | LLM includes preamble/postamble around the structured output | `"Sure! Here's the JSON: {...}"` |
| **Partial output** | Valid prefix but incomplete (truncation) | `{"field1": "value", "fie` |
| **Nested error** | Outer structure valid, inner content invalid | Valid JSON but inner `code` field contains syntax errors |

```python
import json
import re

class OutputParser:
    """Robust output parser with multiple fallback strategies."""
    
    def parse_json(self, raw_output: str, schema: type = None) -> tuple[dict | None, list[ChainError]]:
        errors = []
        
        # Strategy 1: Direct parse
        try:
            parsed = json.loads(raw_output)
            if schema:
                validated = schema.model_validate(parsed)
                return validated.model_dump(), []
            return parsed, []
        except (json.JSONDecodeError, Exception):
            pass
        
        # Strategy 2: Extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if schema:
                    validated = schema.model_validate(parsed)
                    return validated.model_dump(), []
                return parsed, []
            except Exception:
                pass
        
        # Strategy 3: Find first { ... } or [ ... ] block
        brace_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', raw_output, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(1))
                if schema:
                    validated = schema.model_validate(parsed)
                    return validated.model_dump(), []
                return parsed, []
            except Exception:
                pass
        
        # Strategy 4: Attempt repair (common issues)
        repaired = self._attempt_json_repair(raw_output)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if schema:
                    validated = schema.model_validate(parsed)
                    return validated.model_dump(), []
                return parsed, []
            except Exception:
                pass
        
        # All strategies failed
        errors.append(ChainError(
            category=ErrorCategory.PARSE_SCHEMA,
            step_index=-1, step_name="",
            message="All JSON parsing strategies failed",
            raw_output=raw_output[:500],
            recoverable=True, severity="medium"
        ))
        return None, errors
    
    @staticmethod
    def _attempt_json_repair(text: str) -> str | None:
        """Attempt common JSON repairs."""
        # Remove trailing commas
        repaired = re.sub(r',\s*([}\]])', r'\1', text)
        # Add missing closing braces
        open_braces = repaired.count('{') - repaired.count('}')
        if open_braces > 0:
            repaired += '}' * open_braces
        # Replace single quotes with double quotes
        repaired = repaired.replace("'", '"')
        
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None
```

### 1.7.1.5 Category D — Propagation Errors ($\mathcal{E}_{\text{prop}}$)

An error in step $i$ that is not detected and corrupts the outputs of steps $i{+}1, \dots, n$. This is the **most dangerous** error category because the corruption is silent and may not surface until the final output.

### 1.7.1.6 Category E — Infrastructure Errors ($\mathcal{E}_{\text{infra}}$)

| Sub-Type | Cause | Typical Duration | Recovery |
|---|---|---|---|
| **Timeout** | LLM call exceeds deadline | Seconds–minutes | Retry with smaller input or fallback model |
| **Rate limit** | API quota exceeded | Seconds–hours | Exponential backoff, queue, or switch provider |
| **API error** (5xx) | Server-side failure | Seconds–minutes | Retry with backoff |
| **Network error** | Connectivity loss | Variable | Retry, fallback to cached result |
| **Token limit** | Prompt exceeds $C_{\max}$ | Immediate | Compress context, truncate, split |

### 1.7.1.7 Category F — State Corruption Errors ($\mathcal{E}_{\text{state}}$)

Errors where the chain's internal state becomes inconsistent, stale, or invalid.

| Sub-Type | Cause | Detection |
|---|---|---|
| **Stale state** | Step reads outdated value from a shared store | Version checking, timestamps |
| **Race condition** | Concurrent writes to shared state | Locking, compare-and-swap |
| **Schema drift** | State schema changes mid-execution (code deployment during run) | Schema version tags |
| **Memory corruption** | Working memory or scratchpad contains contradictory entries | Consistency invariant checks |

---

## 1.7.2 Error Propagation Analysis

### 1.7.2.1 Formal Error Propagation Model

Consider a chain of $n$ steps where step $i$ introduces a relative error $\delta_i \geq 0$ to its input. If the initial input has error $\epsilon_0$, the accumulated error after $n$ steps is:

$$
\epsilon_n = \epsilon_0 \cdot \prod_{i=1}^{n}(1 + \delta_i)
$$

**Proof by induction.** $\epsilon_1 = \epsilon_0(1 + \delta_1)$. Assume $\epsilon_k = \epsilon_0 \prod_{i=1}^{k}(1+\delta_i)$. Then $\epsilon_{k+1} = \epsilon_k(1+\delta_{k+1}) = \epsilon_0 \prod_{i=1}^{k+1}(1+\delta_i)$. $\square$

**Special case: homogeneous error.** If all steps introduce the same relative error $\delta$:

$$
\epsilon_n = \epsilon_0 (1 + \delta)^n
$$

This is **exponential growth.** For $\delta = 0.05$ (5% error per step) and $n = 10$:

$$
\epsilon_{10} = \epsilon_0 \cdot 1.05^{10} \approx 1.63 \cdot \epsilon_0
$$

For $n = 20$: $\epsilon_{20} \approx 2.65 \cdot \epsilon_0$. The initial error nearly triples.

### 1.7.2.2 Probabilistic Error Compounding

In practice, step errors are stochastic. Model each step's success as a Bernoulli trial with success probability $p_i$. The probability that all $n$ steps succeed:

$$
P(\text{chain success}) = \prod_{i=1}^{n} p_i
$$

For homogeneous $p_i = p$:

$$
P(\text{chain success}) = p^n
$$

**Reliability table:**

| Per-Step Accuracy $p$ | $n=3$ | $n=5$ | $n=10$ | $n=20$ |
|---|---|---|---|---|
| 0.99 | 0.970 | 0.951 | 0.904 | 0.818 |
| 0.95 | 0.857 | 0.774 | 0.599 | 0.358 |
| 0.90 | 0.729 | 0.590 | 0.349 | 0.122 |
| 0.85 | 0.614 | 0.444 | 0.197 | 0.039 |

**Key insight:** Even 95% per-step accuracy yields only 36% end-to-end success for a 20-step chain. This is the **fundamental reliability challenge** of prompt chaining.

### 1.7.2.3 Sensitivity Analysis of Chain Steps

Not all steps contribute equally to error. **Sensitivity analysis** identifies which steps are **critical** (high impact on final output quality) vs. **tolerant** (errors are absorbed or corrected downstream).

**Formal definition.** The sensitivity of the final output quality $Q_n$ to step $i$'s error $\delta_i$:

$$
S_i = \frac{\partial Q_n}{\partial \delta_i}
$$

In the multiplicative error model:

$$
S_i = \frac{\partial}{\partial \delta_i}\!\left[\epsilon_0 \prod_{j=1}^{n}(1+\delta_j)\right] = \epsilon_0 \prod_{j \neq i}(1+\delta_j)
$$

All steps have equal sensitivity in the homogeneous case, but in practice steps vary due to:
1. **Information bottlenecks** — a step that summarizes or compresses information has outsized impact.
2. **Early steps** — errors in step 1 propagate through all subsequent steps.
3. **Decision points** — routing steps that select the wrong branch corrupt the entire downstream path.

### 1.7.2.4 Critical Path Identification

**Definition.** The **critical path** is the longest dependency path through the chain DAG. Steps on the critical path determine both **latency** and **error exposure**.

**Error-weighted critical path.** Define the error-risk of a path $\pi = (v_1, v_2, \dots, v_k)$:

$$
\text{Risk}(\pi) = 1 - \prod_{i=1}^{k} p_{v_i}
$$

The **most error-prone path** is:

$$
\pi^* = \arg\max_{\pi \in \text{Paths}(G)} \; \text{Risk}(\pi)
$$

Steps on $\pi^*$ should receive the highest investment in error handling (more retries, better prompts, stronger validation gates).

```python
def identify_critical_steps(
    chain_dag: dict[str, list[str]],   # adjacency list
    step_reliability: dict[str, float]  # step_name → P(success)
) -> dict:
    """Identify the most error-prone path through the chain DAG."""
    
    # Find all source-to-sink paths
    sources = [n for n in chain_dag if not any(n in deps for deps in chain_dag.values())]
    sinks = [n for n in chain_dag if not chain_dag.get(n)]
    
    def find_all_paths(start, end, visited=None):
        visited = visited or set()
        if start == end:
            return [[start]]
        visited.add(start)
        paths = []
        for neighbor in chain_dag.get(start, []):
            if neighbor not in visited:
                for path in find_all_paths(neighbor, end, visited.copy()):
                    paths.append([start] + path)
        return paths
    
    all_paths = []
    for src in sources:
        for sink in sinks:
            all_paths.extend(find_all_paths(src, sink))
    
    # Compute risk for each path
    path_risks = []
    for path in all_paths:
        success_prob = 1.0
        for step in path:
            success_prob *= step_reliability.get(step, 0.95)
        risk = 1.0 - success_prob
        path_risks.append({"path": path, "risk": risk, "success_prob": success_prob})
    
    path_risks.sort(key=lambda x: x["risk"], reverse=True)
    
    # Steps on the riskiest path are critical
    critical_path = path_risks[0] if path_risks else None
    
    # Per-step sensitivity: how much does improving this step reduce path risk?
    sensitivities = {}
    if critical_path:
        for step in critical_path["path"]:
            p = step_reliability.get(step, 0.95)
            # If we improve this step to perfect (p=1), how much does risk drop?
            improved_success = critical_path["success_prob"] / p  # remove step's contribution
            risk_reduction = (1 - critical_path["success_prob"]) - (1 - improved_success)
            sensitivities[step] = {
                "current_reliability": p,
                "risk_if_improved": 1 - improved_success,
                "risk_reduction": risk_reduction
            }
    
    return {
        "critical_path": critical_path,
        "all_paths_ranked": path_risks,
        "step_sensitivities": sensitivities
    }
```

---

## 1.7.3 Mitigation Strategies

### 1.7.3.1 Retry with Exponential Backoff

For **transient** infrastructure errors (timeouts, rate limits, 5xx errors), retry with exponentially increasing delays.

**Formal schedule.** The wait time before attempt $k$ (0-indexed):

$$
t_k = \min\!\bigl(\, t_{\text{base}} \cdot 2^k + \text{jitter}(k),\; t_{\max} \,\bigr)
$$

where $\text{jitter}(k) \sim \text{Uniform}(0, t_{\text{base}} \cdot 2^{k-1})$ prevents thundering herd effects when multiple chains retry simultaneously.

```python
import asyncio
import random
from typing import TypeVar, Callable

T = TypeVar("T")

async def retry_with_backoff(
    fn: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError),
    on_retry: Callable[[int, Exception], None] = None,
    **kwargs
) -> T:
    """Execute fn with exponential backoff on transient failures."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter
            
            if on_retry:
                on_retry(attempt + 1, e)
            
            await asyncio.sleep(wait)
    
    raise last_exception
```

### 1.7.3.2 Retry with Rephrasing

When the LLM produces a generation error (format violation, refusal, low quality), retrying with the **exact same prompt** often produces the same error. **Rephrased retry** modifies the prompt to give the model a different path through its distribution.

**Rephrasing strategies:**

| Strategy | Description | When to Use |
|---|---|---|
| **Append error feedback** | Add "The previous attempt failed because: [error]. Please fix." | Format violations, incomplete output |
| **Simplify instruction** | Reduce prompt complexity, break into smaller ask | Instruction drift, refusal |
| **Change format request** | Switch from JSON to YAML, or from freeform to structured | Format violations |
| **Add/modify examples** | Provide a concrete example of correct output | Schema mismatches |
| **Temperature adjustment** | Increase $\tau$ for diversity or decrease for determinism | Repetition or inconsistency |
| **Role reframing** | Change the system prompt's persona | Refusal |

```python
class RephrasingRetryStrategy:
    """Retry LLM calls with modified prompts on generation failure."""
    
    def __init__(self, llm: "LLMClient", max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
    
    async def execute_with_rephrase(
        self, 
        prompt: str,
        expected_format: str = "json",
        schema: type = None,
        temperature: float = 0.0
    ) -> tuple[str, dict]:
        
        attempts = []
        current_prompt = prompt
        current_temp = temperature
        
        for attempt in range(self.max_retries + 1):
            response = await self.llm.generate(
                prompt=current_prompt,
                temperature=current_temp,
                max_tokens=2000
            )
            
            # Validate
            errors = self._validate(response, expected_format, schema)
            
            attempts.append({
                "attempt": attempt + 1,
                "prompt_hash": hash(current_prompt),
                "temperature": current_temp,
                "errors": [e.message for e in errors],
                "output_preview": str(response)[:200]
            })
            
            if not errors:
                return response, {"attempts": attempts, "success_on": attempt + 1}
            
            # Rephrase for next attempt
            if attempt < self.max_retries:
                current_prompt, current_temp = self._rephrase(
                    original_prompt=prompt,
                    failed_output=response,
                    errors=errors,
                    attempt=attempt + 1,
                    base_temperature=temperature
                )
        
        return response, {"attempts": attempts, "all_failed": True}
    
    def _rephrase(self, original_prompt: str, failed_output: str,
                  errors: list[ChainError], attempt: int,
                  base_temperature: float) -> tuple[str, float]:
        
        error_categories = {e.category for e in errors}
        
        if ErrorCategory.GEN_FORMAT_VIOLATION in error_categories:
            rephrased = (
                f"{original_prompt}\n\n"
                f"IMPORTANT: Your previous response was not valid JSON. "
                f"You MUST respond with ONLY a valid JSON object. "
                f"Do not include any text before or after the JSON. "
                f"Do not wrap it in markdown code blocks."
            )
            return rephrased, base_temperature
        
        elif ErrorCategory.GEN_REFUSAL in error_categories:
            rephrased = (
                f"You are a helpful assistant performing a structured data processing task. "
                f"This is a legitimate analytical request.\n\n{original_prompt}"
            )
            return rephrased, base_temperature + 0.1
        
        elif ErrorCategory.GEN_TRUNCATION in error_categories:
            rephrased = (
                f"{original_prompt}\n\n"
                f"IMPORTANT: Be concise. Provide only the essential information."
            )
            return rephrased, base_temperature
        
        else:
            # Generic rephrase: add temperature jitter
            return original_prompt, min(base_temperature + 0.2 * attempt, 1.0)
    
    def _validate(self, output: str, expected_format: str, schema: type) -> list[ChainError]:
        detector = GenerationErrorDetector()
        return detector.detect_all(output, expected_format=expected_format)
```

### 1.7.3.3 Fallback Chains

When the primary chain path fails beyond retry capacity, a **fallback chain** provides an alternative execution path — typically simpler, less accurate, but more reliable.

```
Primary Chain: [Search] → [Analyze] → [Synthesize] → [Format]
                   ↓ fail
Fallback Chain: [LLM Direct Answer with lower quality guarantee]
```

**Formal model.** The system's overall reliability with one fallback:

$$
P(\text{system success}) = P(\text{primary}) + (1 - P(\text{primary})) \cdot P(\text{fallback})
$$

For independent chains with $P(\text{primary}) = 0.85$ and $P(\text{fallback}) = 0.90$:

$$
P(\text{system}) = 0.85 + 0.15 \times 0.90 = 0.985
$$

With $k$ fallback levels:

$$
P(\text{system}) = 1 - \prod_{j=0}^{k}(1 - P_j)
$$

```python
class FallbackChainExecutor:
    """Execute primary chain with ordered fallback alternatives."""
    
    @dataclass
    class ChainOption:
        name: str
        executor: Callable
        expected_quality: float  # 0-1
        expected_latency_ms: float
        cost: float
    
    def __init__(self, options: list["FallbackChainExecutor.ChainOption"]):
        # Ordered by preference (primary first)
        self.options = options
    
    async def execute(self, input_data: dict) -> dict:
        results = []
        
        for option in self.options:
            try:
                output = await asyncio.wait_for(
                    option.executor(input_data),
                    timeout=option.expected_latency_ms / 1000 * 2  # 2x timeout
                )
                
                results.append({
                    "chain": option.name,
                    "output": output,
                    "status": "success"
                })
                
                return {
                    "output": output,
                    "served_by": option.name,
                    "fallback_depth": len(results),
                    "expected_quality": option.expected_quality,
                    "attempts": results
                }
            
            except Exception as e:
                results.append({
                    "chain": option.name,
                    "error": str(e),
                    "status": "failed"
                })
                continue
        
        return {
            "output": None,
            "served_by": None,
            "all_failed": True,
            "attempts": results
        }
```

### 1.7.3.4 Output Validation and Self-Correction Loops

The LLM itself can detect and correct its own errors through a **validate-then-fix** loop:

$$
y_t^{(k+1)} = \text{LLM}\!\bigl(\, y_t^{(k)},\; \text{Feedback}(y_t^{(k)}) \,\bigr) \quad \text{until } Q(y_t^{(k)}) \geq \tau \text{ or } k = k_{\max}
$$

### 1.7.3.5 Checkpointing

Save intermediate state after each successful step so that on failure, the chain can **resume from the last checkpoint** rather than restart from scratch.

$$
\text{Resume}(\mathcal{T}, \text{failure at step } j) = \text{Execute}(s_{j}, s_{j+1}, \dots, s_n \mid S_{j-1}^{\text{checkpoint}})
$$

**Cost savings:**

$$
\text{Saved Cost} = \sum_{i=1}^{j-1} \text{cost}(s_i)
$$

```python
import pickle
from pathlib import Path

class CheckpointManager:
    """Save and restore chain state at each step boundary."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, chain_id: str, step_index: int, state: dict) -> Path:
        path = self.dir / f"{chain_id}_step_{step_index}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "chain_id": chain_id,
                "step_index": step_index,
                "state": state,
                "timestamp": datetime.utcnow().isoformat()
            }, f)
        return path
    
    def load_latest(self, chain_id: str) -> tuple[int, dict] | None:
        checkpoints = sorted(
            self.dir.glob(f"{chain_id}_step_*.pkl"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True
        )
        if not checkpoints:
            return None
        with open(checkpoints[0], "rb") as f:
            data = pickle.load(f)
        return data["step_index"], data["state"]
    
    def cleanup(self, chain_id: str):
        for path in self.dir.glob(f"{chain_id}_step_*.pkl"):
            path.unlink()
```

### 1.7.3.6 Graceful Degradation

Return **partial results** when the full chain cannot complete, rather than returning nothing.

$$
\text{Output} = \begin{cases}
y_n & \text{if chain completes successfully} \\
\text{Partial}(y_1, \dots, y_{j-1}) & \text{if step } j \text{ fails and partial output is useful} \\
\text{ErrorReport} & \text{if no useful partial output exists}
\end{cases}
$$

### 1.7.3.7 Circuit Breaker Pattern

Inspired by electrical circuit breakers: after $k$ consecutive failures, **stop attempting** and immediately return a failure response. This prevents cascading failures and resource waste.

**State machine:**

$$
\text{CircuitBreaker}: \texttt{CLOSED} \xrightarrow{k \text{ failures}} \texttt{OPEN} \xrightarrow{t_{\text{cooldown}}} \texttt{HALF\_OPEN} \xrightarrow{\text{success}} \texttt{CLOSED}
$$

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing, reject all calls
    HALF_OPEN = "half_open"  # testing if service recovered

class CircuitBreaker:
    """Prevent cascading failures by halting after repeated errors."""
    
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
    
    async def call(self, fn: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit is OPEN. Retry after "
                    f"{self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s"
                )
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError("HALF_OPEN call limit reached")
            self.half_open_calls += 1
        
        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class CircuitBreakerOpenError(Exception):
    pass
```

### 1.7.3.8 Redundant Execution (Majority Voting)

Run critical steps $N$ times independently and select the result by majority vote or best quality score:

$$
y_t^* = \text{MajorityVote}\!\bigl(\, y_t^{(1)}, y_t^{(2)}, \dots, y_t^{(N)} \,\bigr)
$$

**Reliability improvement.** If each execution succeeds with probability $p$ and we require a majority of $N$ (odd) to be correct:

$$
P(\text{majority correct}) = \sum_{k=\lceil N/2\rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k}
$$

For $p = 0.80$, $N = 3$: $P = 3(0.8)^2(0.2) + (0.8)^3 = 0.384 + 0.512 = 0.896$.

---

## 1.7.4 Self-Healing Chains

### 1.7.4.1 LLM-Based Error Detection and Correction

Self-healing chains embed **reflection steps** that evaluate the previous step's output and either approve it or generate a corrected version.

**Formal model.** A reflection operator $\mathcal{R}$:

$$
\mathcal{R}(y_t, S_t) = \begin{cases}
y_t & \text{if } \text{LLM}_{\text{critic}}(y_t, S_t) = \texttt{CORRECT} \\
y_t' = \text{LLM}_{\text{fix}}(y_t, \text{feedback}, S_t) & \text{if } \text{LLM}_{\text{critic}}(y_t, S_t) = \texttt{INCORRECT}
\end{cases}
$$

```python
class SelfHealingStep:
    """Chain step with built-in reflection and self-correction."""
    
    REFLECTION_PROMPT = """Review the following output for errors:

Task: {task}
Output: {output}

Check for:
1. Factual accuracy — are there any incorrect claims?
2. Completeness — does it address all aspects of the task?
3. Format compliance — does it match the expected format?
4. Logical consistency — are there contradictions?
5. Relevance — is the output actually addressing the task?

Respond as JSON:
{{"has_errors": true/false, "error_descriptions": ["..."], 
  "severity": "none|low|medium|high", "correction_guidance": "..."}}"""
    
    def __init__(self, llm: "LLMClient", max_corrections: int = 2):
        self.llm = llm
        self.max_corrections = max_corrections
    
    async def execute_with_healing(self, task: str, 
                                     initial_output: str) -> tuple[str, dict]:
        current_output = initial_output
        healing_log = []
        
        for correction in range(self.max_corrections):
            # Reflect
            reflection = await self.llm.generate(
                prompt=self.REFLECTION_PROMPT.format(
                    task=task, output=current_output
                ),
                temperature=0.0
            )
            reflection_data = json.loads(reflection)
            healing_log.append({
                "correction_round": correction + 1,
                "reflection": reflection_data
            })
            
            if not reflection_data.get("has_errors", False):
                break
            
            if reflection_data.get("severity") == "none":
                break
            
            # Correct
            correction_prompt = (
                f"The following output has errors:\n{current_output}\n\n"
                f"Errors found:\n"
                + "\n".join(f"- {e}" for e in reflection_data.get("error_descriptions", []))
                + f"\n\nGuidance: {reflection_data.get('correction_guidance', '')}\n\n"
                f"Original task: {task}\n\n"
                f"Please produce a corrected version."
            )
            
            current_output = await self.llm.generate(
                prompt=correction_prompt,
                temperature=0.0
            )
        
        return current_output, {"healing_log": healing_log, 
                                "corrections_applied": len(healing_log)}
```



