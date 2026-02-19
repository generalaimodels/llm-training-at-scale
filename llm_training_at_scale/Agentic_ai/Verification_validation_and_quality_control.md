# 1.8 Verification, Validation, and Quality Control

> **Scope.** Verification and validation (V&V) ensure that each step and the overall chain produce outputs that are correct, consistent, and aligned with the original intent. Verification asks "Did we build the thing right?" (does the output conform to specification?). Validation asks "Did we build the right thing?" (does the output satisfy the user's actual need?). This section provides a taxonomy of V&V patterns at per-step, chain-level, and cross-execution granularity.

---

## 1.8.1 Per-Step Verification

### 1.8.1.1 Output Schema Validation

The most deterministic form of verification: check that the output conforms to a predefined structural schema $\sigma$.

$$
V_{\text{schema}}(y_t) = \mathbb{1}\!\bigl[\, y_t \in \mathcal{L}(\sigma_t) \,\bigr]
$$

This was implemented in §1.7.1.4. Key addition: **semantic schema validation** goes beyond structural compliance to check that field values are semantically reasonable.

```python
class SemanticSchemaValidator:
    """Validate both structure and semantic plausibility of output fields."""
    
    def __init__(self, llm: "LLMClient" = None):
        self.llm = llm
        self.field_validators: dict[str, Callable] = {}
    
    def register_field_validator(self, field_name: str, 
                                  validator: Callable[[Any], bool]):
        self.field_validators[field_name] = validator
    
    async def validate(self, data: dict, schema: type = None) -> dict:
        results = {"structural": True, "semantic": True, "field_results": {}}
        
        # Structural validation
        if schema:
            try:
                schema.model_validate(data)
            except Exception as e:
                results["structural"] = False
                results["structural_error"] = str(e)
                return results
        
        # Per-field semantic validation
        for field_name, validator in self.field_validators.items():
            if field_name in data:
                is_valid = validator(data[field_name])
                results["field_results"][field_name] = is_valid
                if not is_valid:
                    results["semantic"] = False
        
        return results

# Example field validators
validator = SemanticSchemaValidator()
validator.register_field_validator(
    "confidence", lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0
)
validator.register_field_validator(
    "url", lambda v: isinstance(v, str) and v.startswith("http")
)
validator.register_field_validator(
    "year", lambda v: isinstance(v, int) and 1900 <= v <= 2030
)
```

### 1.8.1.2 Factuality Checking (Retrieval-Augmented Verification)

Verify that claims in the output are supported by retrieved evidence:

$$
V_{\text{factual}}(y_t) = \frac{|\{c \in \text{claims}(y_t) : \exists\, e \in \mathcal{E},\; \text{entails}(e, c)\}|}{|\text{claims}(y_t)|}
$$

### 1.8.1.3 Consistency Checking with Prior Steps

Verify that step $t$'s output does not contradict information established in previous steps:

$$
V_{\text{consist}}(y_t, y_1, \dots, y_{t-1}) = \mathbb{1}\!\bigl[\, \neg\exists\, i < t : \text{contradicts}(y_t, y_i) \,\bigr]
$$

```python
class ConsistencyChecker:
    """Check that a step's output is consistent with prior outputs."""
    
    def __init__(self, llm: "LLMClient"):
        self.llm = llm
    
    async def check(self, current_output: str, 
                    prior_outputs: list[str]) -> dict:
        if not prior_outputs:
            return {"consistent": True, "contradictions": []}
        
        prior_summary = "\n---\n".join(
            f"Step {i+1}: {out[:500]}" for i, out in enumerate(prior_outputs)
        )
        
        prompt = (
            f"Check if the Current Output contradicts any Prior Output.\n\n"
            f"Prior Outputs:\n{prior_summary}\n\n"
            f"Current Output:\n{current_output}\n\n"
            f"List any contradictions. Respond as JSON:\n"
            f'{{"consistent": true/false, "contradictions": '
            f'[{{"claim_current": "...", "claim_prior": "...", '
            f'"step": N, "explanation": "..."}}]}}'
        )
        
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return json.loads(response)
```

### 1.8.1.4 Constraint Satisfaction Verification

For steps that must satisfy explicit constraints (numeric bounds, logical conditions, format rules):

$$
V_{\text{constraint}}(y_t) = \bigwedge_{c \in \mathcal{C}_t} c(y_t)
$$

where $\mathcal{C}_t$ is the set of constraints for step $t$.

---

## 1.8.2 Chain-Level Verification

### 1.8.2.1 End-to-End Output Quality Assessment

The final output is evaluated against the **original task specification**, not against any intermediate step's specification:

$$
Q_{\text{e2e}} = \text{sim}_{\text{semantic}}\!\bigl(\, y_n,\; \text{ideal}(x) \,\bigr)
$$

When no ground truth exists, use a **judge LLM** to evaluate quality on multiple dimensions:

```python
class EndToEndEvaluator:
    """Evaluate final chain output against original task intent."""
    
    EVALUATION_PROMPT = """Evaluate how well the Output addresses the original Task.

Task: {task}
Output: {output}

Rate on each dimension (1-5):
1. Relevance: Does it address what was asked?
2. Completeness: Does it cover all aspects?
3. Accuracy: Are the facts and reasoning correct?
4. Clarity: Is it well-written and easy to understand?
5. Actionability: Can the user act on this output?

Respond as JSON:
{{"relevance": N, "completeness": N, "accuracy": N, 
  "clarity": N, "actionability": N, 
  "overall": N, "justification": "..."}}"""
    
    def __init__(self, evaluator_llm: "LLMClient"):
        self.llm = evaluator_llm
    
    async def evaluate(self, task: str, output: str) -> dict:
        response = await self.llm.generate(
            prompt=self.EVALUATION_PROMPT.format(task=task, output=output),
            temperature=0.0
        )
        scores = json.loads(response)
        
        # Normalized aggregate
        dims = ["relevance", "completeness", "accuracy", "clarity", "actionability"]
        scores["aggregate"] = sum(scores.get(d, 3) for d in dims) / (5 * len(dims))
        return scores
```

### 1.8.2.2 Invariant Checking Across Chain Execution

Define **chain invariants** — properties that must hold at every step boundary:

$$
\forall\, t \in \{1, \dots, n\}: \quad I(S_t) = \texttt{True}
$$

| Invariant Type | Example | Check Method |
|---|---|---|
| **Monotonic progress** | Task completion percentage never decreases | Compare `state.progress` across steps |
| **Budget adherence** | Total token usage stays within allocation | Cumulative sum check |
| **Schema consistency** | State always conforms to the declared schema | Pydantic validation at every boundary |
| **No information loss** | Key facts from step 1 remain accessible in step $n$ | Keyword/entity presence check |
| **Safety** | No step output contains unsafe content | Safety classifier at every boundary |

```python
class ChainInvariantChecker:
    """Verify that chain-wide invariants hold after every step."""
    
    def __init__(self):
        self.invariants: list[tuple[str, Callable[[dict], bool]]] = []
    
    def add_invariant(self, name: str, check: Callable[[dict], bool]):
        self.invariants.append((name, check))
    
    def check_all(self, state: dict) -> dict:
        results = {}
        all_passed = True
        
        for name, check in self.invariants:
            try:
                passed = check(state)
                results[name] = {"passed": passed}
                if not passed:
                    all_passed = False
            except Exception as e:
                results[name] = {"passed": False, "error": str(e)}
                all_passed = False
        
        return {"all_passed": all_passed, "invariants": results}


# Example invariants
invariant_checker = ChainInvariantChecker()

invariant_checker.add_invariant(
    "token_budget",
    lambda s: s.get("tokens_used", 0) <= s.get("token_budget", float("inf"))
)
invariant_checker.add_invariant(
    "step_progress",
    lambda s: s.get("step_index", 0) >= s.get("prev_step_index", 0)
)
invariant_checker.add_invariant(
    "no_empty_results",
    lambda s: bool(s.get("latest_output", "").strip())
)
```

---

## 1.8.3 Verification Patterns

### 1.8.3.1 Verifier Chain Pattern

A **separate, independent chain** whose sole purpose is to validate the main chain's output. The verifier chain has different prompts, potentially a different model, and evaluates along orthogonal criteria.

```
Main Chain:     [Step 1] → [Step 2] → [Step 3] → Output
                                                     │
                                                     ▼
Verifier Chain: [Extract Claims] → [Check Facts] → [Score] → Verdict
```

### 1.8.3.2 Self-Consistency (Multi-Sample Agreement)

Run the chain $k$ times with different random seeds (via temperature $> 0$) and measure agreement:

$$
\text{Consistency}(x) = \frac{2}{k(k-1)} \sum_{i<j} \text{sim}(y^{(i)}, y^{(j)})
$$

**Decision rule:**

$$
\text{Accept} \iff \text{Consistency}(x) > \tau_{\text{agree}}
$$

If consistency is low, the chain's output on this input is unreliable and should be flagged for human review or re-executed with a different strategy.

```python
class SelfConsistencyVerifier:
    """Verify output reliability via multi-sample agreement."""
    
    def __init__(self, chain_executor: Callable, 
                 similarity_fn: Callable[[str, str], float],
                 k: int = 5, agreement_threshold: float = 0.7):
        self.executor = chain_executor
        self.sim_fn = similarity_fn
        self.k = k
        self.threshold = agreement_threshold
    
    async def verify(self, input_data: dict) -> dict:
        # Generate k independent outputs
        outputs = await asyncio.gather(*[
            self.executor({**input_data, "seed": i})
            for i in range(self.k)
        ])
        
        # Compute pairwise similarity
        from itertools import combinations
        similarities = []
        for a, b in combinations(range(self.k), 2):
            sim = self.sim_fn(str(outputs[a]), str(outputs[b]))
            similarities.append(sim)
        
        avg_agreement = sum(similarities) / len(similarities) if similarities else 0
        
        # Select the output most consistent with others
        output_scores = []
        for i in range(self.k):
            avg_sim = sum(
                self.sim_fn(str(outputs[i]), str(outputs[j]))
                for j in range(self.k) if j != i
            ) / (self.k - 1)
            output_scores.append((avg_sim, i))
        
        best_idx = max(output_scores, key=lambda x: x[0])[1]
        
        return {
            "reliable": avg_agreement >= self.threshold,
            "agreement_score": avg_agreement,
            "selected_output": outputs[best_idx],
            "selected_index": best_idx,
            "all_outputs_preview": [str(o)[:200] for o in outputs],
            "pairwise_similarities": similarities
        }
```

### 1.8.3.3 Critic Step Pattern

A dedicated LLM step evaluates and scores the output using a detailed rubric:

```python
class CriticStep:
    """Dedicated LLM step to critique and score chain output."""
    
    CRITIC_PROMPT = """You are a strict quality evaluator. Analyze the following output 
produced for the given task. Be critical and identify ALL issues.

Task: {task}
Output to evaluate: {output}

Evaluation rubric:
1. FACTUAL ACCURACY (0-10): Are all claims verifiable and correct?
2. LOGICAL COHERENCE (0-10): Is the reasoning sound, with no contradictions?
3. COMPLETENESS (0-10): Are all parts of the task addressed?
4. FORMAT COMPLIANCE (0-10): Does the output match the expected format?
5. CONCISENESS (0-10): Is the output appropriately detailed without unnecessary content?

For each dimension:
- Provide the score
- List specific issues found
- Suggest concrete improvements

Final verdict: ACCEPT (avg >= 7), REVISE (avg 4-7), or REJECT (avg < 4)

Respond as JSON."""
    
    def __init__(self, critic_llm: "LLMClient"):
        self.llm = critic_llm
    
    async def critique(self, task: str, output: str) -> dict:
        response = await self.llm.generate(
            prompt=self.CRITIC_PROMPT.format(task=task, output=output),
            temperature=0.0
        )
        return json.loads(response)
```

### 1.8.3.4 Formal Verification for Code-Generating Chains

When the chain produces executable code, apply **deterministic verification** via test suites and static analysis:

$$
V_{\text{formal}}(\text{code}) = \bigwedge_{t \in \mathcal{T}_{\text{tests}}} \text{passes}(\text{code}, t) \;\wedge\; \text{lint}(\text{code}) = \texttt{CLEAN}
$$

```python
import subprocess
import tempfile

class CodeVerifier:
    """Verify generated code via execution, testing, and static analysis."""
    
    async def verify(self, code: str, test_code: str = None,
                     language: str = "python") -> dict:
        results = {
            "syntax_valid": False,
            "tests_passed": None,
            "lint_clean": None,
            "security_issues": []
        }
        
        # 1. Syntax check
        try:
            compile(code, "<generated>", "exec")
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["syntax_error"] = str(e)
            return results
        
        # 2. Execute tests if provided
        if test_code:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code + "\n\n" + test_code)
                f.flush()
                try:
                    proc = subprocess.run(
                        ["python", f.name],
                        capture_output=True, text=True, timeout=30
                    )
                    results["tests_passed"] = proc.returncode == 0
                    results["test_output"] = proc.stdout
                    results["test_errors"] = proc.stderr
                except subprocess.TimeoutExpired:
                    results["tests_passed"] = False
                    results["test_errors"] = "Execution timed out"
        
        # 3. Static analysis (basic)
        dangerous_patterns = ["eval(", "exec(", "os.system(", "__import__(",
                              "subprocess.call(", "open(", "pickle.loads("]
        for pattern in dangerous_patterns:
            if pattern in code:
                results["security_issues"].append(
                    f"Potentially dangerous: {pattern}"
                )
        
        results["lint_clean"] = len(results["security_issues"]) == 0
        return results
```

---

## 1.8.4 Human-in-the-Loop Verification

**Checkpoint-based human review** inserts human evaluation at predefined chain boundaries. **Active learning** uses human feedback on uncertain outputs to improve automated verification models. **Escalation policies** define when and how to route to humans (covered in §1.6.4.4).

