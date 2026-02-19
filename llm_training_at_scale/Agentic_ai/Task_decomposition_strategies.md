

# 1.5 Task Decomposition Strategies

> **Scope.** Task decomposition is the foundational cognitive and computational operation that transforms an intractable monolithic problem into a sequence or graph of tractable sub-problems, each solvable within a single LLM call or tool invocation. The quality of decomposition directly determines chain accuracy, latency, cost, and debuggability. A poorly decomposed task produces either under-specified steps that hallucinate or over-specified steps that waste compute. This section provides a rigorous, first-principles treatment of manual, automatic, and hybrid decomposition strategies; analyzes granularity trade-offs mathematically; and maps classical AI planning frameworks onto modern LLM chain construction.

---

## 1.5.1 Manual Decomposition

### 1.5.1.1 Foundational Principle

Manual decomposition is the process whereby a **human expert** analyzes a complex task $\mathcal{T}$ and produces an ordered set of sub-tasks $\{s_1, s_2, \dots, s_n\}$ such that:

$$
\mathcal{T} = s_1 \circ s_2 \circ \cdots \circ s_n
$$

where $\circ$ denotes functional composition — the output of $s_i$ feeds the input of $s_{i+1}$. More generally, the decomposition produces a **directed acyclic graph** (DAG) $G = (V, E)$ where each vertex $v \in V$ is a sub-task and each edge $(v_i, v_j) \in E$ encodes a data dependency.

**The decomposition function** performed by the expert is:

$$
\text{Decompose}_{\text{human}}: \mathcal{T} \times \mathcal{K}_{\text{domain}} \rightarrow G = (V, E, \Sigma)
$$

where $\mathcal{K}_{\text{domain}}$ is the expert's domain knowledge and $\Sigma = \{\sigma_v\}_{v \in V}$ assigns an input/output schema to each sub-task.

### 1.5.1.2 Expert-Driven Task Breakdown

The expert performs decomposition by reasoning about the **cognitive and computational requirements** of each sub-problem. The key questions that guide this process:

| Decomposition Question | Operational Impact |
|---|---|
| Can a single LLM call solve this sub-task reliably? | Determines whether further decomposition is needed |
| Does this sub-task require external tools? | Determines node type (LLM call vs. tool call vs. code execution) |
| What information does this sub-task need? | Defines incoming edges in the DAG |
| What information does this sub-task produce? | Defines outgoing edges and the output schema $\sigma_v^{\text{out}}$ |
| Can this sub-task fail? How? | Determines error-handling and retry strategies |
| Can this sub-task run in parallel with others? | Determines DAG structure (sequential vs. parallel branches) |

**Example: Decomposing "Write a market analysis report for Company X"**

```
Task: Write a market analysis report for Company X

Expert Decomposition:
├── s₁: Extract Company X's financial data (tool: SEC API)
├── s₂: Identify Company X's competitors (LLM + search)
├── s₃: Retrieve industry trends (tool: market data API)
├── s₄: Analyze competitive positioning (LLM, depends on s₁, s₂, s₃)
├── s₅: Generate financial projections (LLM + code execution, depends on s₁, s₃)
├── s₆: Write executive summary (LLM, depends on s₄, s₅)
├── s₇: Compile full report (LLM, depends on s₄, s₅, s₆)
└── s₈: Quality review and fact-check (LLM, depends on s₇)
```

The corresponding DAG:

```
    s₁ ──────┐
             ├──▶ s₄ ──┐
    s₂ ──────┘         │
                       ├──▶ s₆ ──▶ s₇ ──▶ s₈
    s₃ ──┬──▶ s₅ ──────┘
         │
         └──▶ s₄ (also depends on s₃)
```

Sub-tasks $s_1$, $s_2$, $s_3$ are **independent** and can execute in parallel. Sub-task $s_4$ is a **join node** requiring all three predecessors to complete.

### 1.5.1.3 Domain-Specific Decomposition Heuristics

Each domain has evolved characteristic decomposition patterns through accumulated engineering practice. These heuristics encode **domain-specific knowledge** about what constitutes an effective sub-task boundary.

#### A. Software Engineering Domain

| Heuristic | Rationale |
|---|---|
| **Separate specification from implementation** | LLMs produce better code when given a clear spec; mixing spec generation with coding causes drift |
| **Separate code generation from testing** | The same model that writes code is biased toward confirming its own correctness |
| **One function per step** | Keeps each sub-task's output verifiable and testable in isolation |
| **Decompose by abstraction layer** | UI → API → Business Logic → Data Access — each has distinct prompt patterns |
| **Isolate dependency resolution** | Package/library selection is a distinct reasoning task from implementation |

```python
# Software engineering decomposition template
SOFTWARE_CHAIN = [
    {"step": "requirements_analysis",  "type": "llm",  "input": "user_request",
     "output": "structured_requirements"},
    {"step": "architecture_design",    "type": "llm",  "input": "structured_requirements",
     "output": "component_specs"},
    {"step": "code_generation",        "type": "llm",  "input": "component_specs[i]",
     "output": "source_code[i]",       "parallel": True},
    {"step": "unit_test_generation",   "type": "llm",  "input": "source_code[i]",
     "output": "test_code[i]",         "parallel": True},
    {"step": "test_execution",         "type": "tool", "input": "source_code[i] + test_code[i]",
     "output": "test_results[i]",      "parallel": True},
    {"step": "integration",           "type": "llm",  "input": "all source_code + test_results",
     "output": "integrated_system"},
    {"step": "review",                "type": "llm",  "input": "integrated_system",
     "output": "review_feedback"},
]
```

#### B. Research & Analysis Domain

| Heuristic | Rationale |
|---|---|
| **Separate retrieval from synthesis** | Mixing search and reasoning causes the LLM to fabricate sources |
| **Separate claim extraction from evaluation** | Evaluation requires comparing claims against evidence — a distinct cognitive task |
| **Decompose by evidence type** | Quantitative data, qualitative reports, expert opinions require different processing |
| **Separate drafting from revision** | First-draft generation and critical editing are cognitively distinct |

#### C. Data Processing Domain

| Heuristic | Rationale |
|---|---|
| **Separate schema inference from transformation** | Understanding the data shape before transforming prevents silent errors |
| **One transformation per step** | Composable, testable transformations; rollback is trivial |
| **Separate validation from processing** | Validation should be a distinct gate that blocks downstream if data is malformed |

### 1.5.1.4 Cognitive Task Analysis (CTA) for Chain Design

Cognitive Task Analysis is a systematic methodology borrowed from human factors engineering that elicits the **mental processes** an expert uses to solve a task, then translates those into explicit chain steps.

**CTA Protocol for Chain Design:**

**Phase 1 — Task Identification.** Define the top-level task $\mathcal{T}$ and its acceptance criteria.

**Phase 2 — Knowledge Elicitation.** Interview domain experts or analyze expert behavior traces:
- What information do you gather first?
- What decisions do you make at each stage?
- What cues tell you something is going wrong?
- What mental models do you use?

**Phase 3 — Cognitive Demands Analysis.** For each identified sub-task, characterize:

$$
\text{CogDemand}(s_i) = \bigl(\, \text{knowledge\_type},\; \text{reasoning\_type},\; \text{uncertainty\_level},\; \text{error\_consequence} \,\bigr)
$$

| Knowledge Type | LLM Suitability | Alternative |
|---|---|---|
| **Factual recall** | Moderate (risk of hallucination) | Retrieval (RAG) |
| **Procedural** | High (follows instructions well) | Direct LLM |
| **Analytical** | High (CoT reasoning) | LLM with scratchpad |
| **Evaluative/Judgmental** | Moderate | LLM + rubric |
| **Perceptual** | Low for text LLMs | Vision model / specialized tool |
| **Computational** | Very low | Code execution tool |

**Phase 4 — Chain Mapping.** Map each cognitive demand to a chain step type:

```python
from enum import Enum
from dataclasses import dataclass

class StepType(Enum):
    LLM_GENERATION = "llm_generation"
    LLM_ANALYSIS = "llm_analysis"
    LLM_EVALUATION = "llm_evaluation"
    TOOL_RETRIEVAL = "tool_retrieval"
    TOOL_COMPUTATION = "tool_computation"
    TOOL_API_CALL = "tool_api_call"
    CODE_EXECUTION = "code_execution"
    HUMAN_REVIEW = "human_review"

@dataclass
class CognitiveStep:
    name: str
    knowledge_type: str        # factual, procedural, analytical, evaluative
    reasoning_type: str        # deductive, inductive, abductive, analogical
    uncertainty: float         # 0.0 = certain, 1.0 = highly uncertain
    error_consequence: str     # low, medium, high, critical
    recommended_step_type: StepType
    
    @property
    def needs_verification(self) -> bool:
        return self.uncertainty > 0.5 or self.error_consequence in ("high", "critical")

# Example CTA output
cta_results = [
    CognitiveStep(
        name="identify_relevant_regulations",
        knowledge_type="factual",
        reasoning_type="deductive",
        uncertainty=0.7,
        error_consequence="critical",
        recommended_step_type=StepType.TOOL_RETRIEVAL  # Don't trust LLM for legal facts
    ),
    CognitiveStep(
        name="assess_compliance_gaps",
        knowledge_type="analytical",
        reasoning_type="deductive",
        uncertainty=0.4,
        error_consequence="high",
        recommended_step_type=StepType.LLM_ANALYSIS
    ),
]
```

### 1.5.1.5 Workflow Mapping to Chain Steps

Workflow mapping translates an existing **business process** or **standard operating procedure** (SOP) into a chain of automated steps. The key insight is that many expert workflows already embody optimal decomposition — they were refined through years of practice.

**Mapping procedure:**

1. **Document the existing workflow** as a flowchart or BPMN diagram.
2. **Classify each step** as automatable (LLM/tool) or requiring human judgment.
3. **Identify decision points** → these become conditional branching nodes in the chain.
4. **Identify parallel paths** → these become concurrent branches.
5. **Define handoff schemas** → these become the explicit state schemas between steps.

```python
@dataclass
class WorkflowStep:
    id: str
    name: str
    description: str
    inputs: list[str]       # IDs of upstream steps
    outputs: list[str]      # keys this step produces
    automatable: bool
    step_type: StepType
    decision_point: bool = False
    branches: dict = None   # if decision_point: condition → next_step_id

def workflow_to_chain(steps: list[WorkflowStep]) -> dict:
    """Convert a documented workflow into a chain specification."""
    chain_spec = {"nodes": [], "edges": []}
    
    for step in steps:
        node = {
            "id": step.id,
            "name": step.name,
            "type": step.step_type.value,
            "config": {
                "description": step.description,
                "output_keys": step.outputs,
            }
        }
        if step.decision_point and step.branches:
            node["type"] = "conditional"
            node["config"]["branches"] = step.branches
        
        chain_spec["nodes"].append(node)
        
        for upstream_id in step.inputs:
            chain_spec["edges"].append({
                "from": upstream_id,
                "to": step.id
            })
    
    return chain_spec
```

**Mapping completeness criterion.** A workflow-to-chain mapping is **complete** iff:

$$
\forall\, v \in V_{\text{workflow}}:\; \exists\, v' \in V_{\text{chain}} \;\text{s.t.}\; \text{semantics}(v) \subseteq \text{semantics}(v')
$$

and **sound** iff:

$$
\forall\, (v_i, v_j) \in E_{\text{workflow}}:\; \exists\, \text{path } v_i' \leadsto v_j' \text{ in } G_{\text{chain}}
$$

That is, every workflow step is covered, and every dependency is preserved.

---

## 1.5.2 Automatic Decomposition

### 1.5.2.1 LLM-Driven Task Decomposition (Plan-Then-Execute)

In automatic decomposition, the LLM itself generates the decomposition. The **plan-then-execute** paradigm separates planning (decomposition) from execution (solving each sub-task):

$$
\underbrace{\mathcal{T} \xrightarrow{\text{Planner LLM}} \{s_1, s_2, \dots, s_n\}}_{\text{Phase 1: Plan}} \xrightarrow{\text{Executor LLM}} \underbrace{y_1, y_2, \dots, y_n}_{\text{Phase 2: Execute}}
$$

**Formal model.** The planner is a conditional distribution:

$$
P_{\text{plan}} = p_\theta\!\bigl(\, \{s_1, \dots, s_n\} \mid \mathcal{T},\; \mathcal{I}_{\text{tools}},\; \mathcal{C}_{\text{constraints}} \,\bigr)
$$

where $\mathcal{I}_{\text{tools}}$ is the inventory of available tools and $\mathcal{C}_{\text{constraints}}$ encodes budget limits, quality requirements, and deadline constraints.

**Critical design decision: Planner-Executor separation.**

| Architecture | Description | Trade-off |
|---|---|---|
| **Same model, same call** | One LLM plans and executes in a single CoT | Simple; plan quality degrades as execution progresses |
| **Same model, separate calls** | Plan in call 1, execute steps in calls 2..n+1 | Plan is frozen; can be inspected/edited before execution |
| **Different models** | Strong model plans, weaker/faster model executes | Cost-efficient; plan quality matches strong model, execution is fast |
| **Planner + Orchestrator** | LLM plans, deterministic code orchestrates execution | Most robust; orchestrator enforces plan structure |

**Implementation — Plan-Then-Execute with structured output:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class SubTask(BaseModel):
    id: str
    description: str
    step_type: Literal["llm", "tool", "code", "retrieval"]
    dependencies: list[str] = Field(default_factory=list)
    expected_output: str
    estimated_tokens: int = 500
    tools_needed: list[str] = Field(default_factory=list)

class TaskPlan(BaseModel):
    original_task: str
    reasoning: str           # planner's CoT explaining decomposition rationale
    subtasks: list[SubTask]
    estimated_total_cost: float = 0.0
    
    def topological_order(self) -> list[SubTask]:
        """Return subtasks in valid execution order (Kahn's algorithm)."""
        in_degree = {s.id: 0 for s in self.subtasks}
        adj = {s.id: [] for s in self.subtasks}
        id_to_task = {s.id: s for s in self.subtasks}
        
        for s in self.subtasks:
            for dep in s.dependencies:
                adj[dep].append(s.id)
                in_degree[s.id] += 1
        
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            current = queue.pop(0)
            order.append(id_to_task[current])
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(order) != len(self.subtasks):
            raise ValueError("Cyclic dependency detected in task plan")
        return order
    
    def parallel_groups(self) -> list[list[SubTask]]:
        """Return groups of subtasks that can execute concurrently."""
        completed = set()
        remaining = {s.id: s for s in self.subtasks}
        groups = []
        
        while remaining:
            # Find all tasks whose dependencies are fully satisfied
            ready = [s for s in remaining.values()
                     if all(d in completed for d in s.dependencies)]
            if not ready:
                raise ValueError("Deadlock: no task can proceed")
            groups.append(ready)
            for s in ready:
                completed.add(s.id)
                del remaining[s.id]
        
        return groups


PLANNING_PROMPT = """You are a task decomposition expert. Given a complex task, 
break it into atomic sub-tasks that can each be solved in a single step.

Rules:
1. Each sub-task must have a clear, verifiable output
2. Specify dependencies between sub-tasks (which must complete before this one starts)
3. Classify each sub-task type: "llm" (reasoning/writing), "tool" (API/search), 
   "code" (computation), "retrieval" (database/vector store lookup)
4. Minimize the total number of sub-tasks while ensuring each is solvable atomically
5. Identify opportunities for parallelism (independent sub-tasks)

Available tools: {tools}
Constraints: {constraints}

Task: {task}

Respond with a structured plan in JSON format matching the TaskPlan schema."""


async def plan_task(
    task: str,
    tools: list[str],
    constraints: dict,
    planner_llm: LLMClient
) -> TaskPlan:
    prompt = PLANNING_PROMPT.format(
        task=task,
        tools=json.dumps(tools),
        constraints=json.dumps(constraints)
    )
    response = await planner_llm.generate(
        prompt=prompt,
        response_format=TaskPlan,  # structured output
        temperature=0.0            # deterministic planning
    )
    plan = TaskPlan.model_validate_json(response)
    
    # Validate: check for cycles, verify dependencies reference valid IDs
    _ = plan.topological_order()  # raises if cyclic
    
    return plan
```

### 1.5.2.2 Recursive Decomposition

When a sub-task is still too complex for a single LLM call, **recursive decomposition** applies the planning step again to that sub-task, producing a tree of ever-finer sub-tasks.

**Formal definition.** Define a recursive decomposition operator $\mathcal{D}$:

$$
\mathcal{D}(\mathcal{T}) = \begin{cases}
\{\mathcal{T}\} & \text{if } \text{complexity}(\mathcal{T}) \leq \tau \\
\bigcup_{i=1}^{k} \mathcal{D}(s_i) & \text{where } \{s_1, \dots, s_k\} = \text{Split}(\mathcal{T})
\end{cases}
$$

where $\tau$ is the **atomicity threshold** — the maximum complexity solvable by a single LLM call — and $\text{Split}$ is the LLM-driven decomposition function.

**Termination condition.** Recursion terminates when **any** of the following hold:

1. **Complexity below threshold:** $\text{complexity}(s_i) \leq \tau$
2. **Maximum depth reached:** $\text{depth}(s_i) \geq d_{\max}$
3. **No further decomposition possible:** $\text{Split}(s_i) = \{s_i\}$ (the LLM cannot break it further)
4. **Token budget exhausted:** cumulative planning cost exceeds allocation

**Complexity estimation.** Since true task complexity is not directly measurable, we use proxy heuristics:

$$
\text{complexity}(s) \approx \hat{c}(s) = w_1 \cdot |\text{entities}(s)| + w_2 \cdot |\text{constraints}(s)| + w_3 \cdot |\text{reasoning\_steps}(s)| + w_4 \cdot \mathbb{1}[\text{tool\_required}(s)]
$$

where the weights $w_i$ are calibrated empirically or estimated by the LLM itself.

```python
async def recursive_decompose(
    task: str,
    planner_llm: LLMClient,
    max_depth: int = 3,
    complexity_threshold: float = 0.3,
    current_depth: int = 0
) -> dict:
    """Recursively decompose a task into a tree of atomic sub-tasks."""
    
    # Estimate complexity
    complexity = await estimate_complexity(task, planner_llm)
    
    # Base cases
    if complexity <= complexity_threshold or current_depth >= max_depth:
        return {
            "task": task,
            "atomic": True,
            "complexity": complexity,
            "depth": current_depth,
            "children": []
        }
    
    # Recursive case: decompose
    plan = await plan_task(task, tools=[], constraints={}, planner_llm=planner_llm)
    
    children = []
    for subtask in plan.subtasks:
        child_tree = await recursive_decompose(
            task=subtask.description,
            planner_llm=planner_llm,
            max_depth=max_depth,
            complexity_threshold=complexity_threshold,
            current_depth=current_depth + 1
        )
        child_tree["id"] = subtask.id
        child_tree["dependencies"] = subtask.dependencies
        children.append(child_tree)
    
    return {
        "task": task,
        "atomic": False,
        "complexity": complexity,
        "depth": current_depth,
        "children": children,
        "reasoning": plan.reasoning
    }


async def estimate_complexity(task: str, llm: LLMClient) -> float:
    """Ask the LLM to estimate task complexity on [0, 1] scale."""
    prompt = (
        "Rate the complexity of the following task on a scale from 0.0 to 1.0, "
        "where 0.0 = trivially simple (answerable in one sentence) and "
        "1.0 = extremely complex (requires multiple tools, data sources, "
        "and reasoning steps). Respond with ONLY a number.\n\n"
        f"Task: {task}"
    )
    response = await llm.generate(prompt=prompt, max_tokens=10, temperature=0.0)
    return float(response.strip())
```

### 1.5.2.3 Decomposition Prompt Patterns

The quality of automatic decomposition depends critically on the **prompt engineering** used to elicit good plans from the LLM. Below are rigorously validated patterns:

#### Pattern A — Numbered Step List (Linear Chain)

```
Break the following task into a numbered list of steps.
Each step should:
- Be completable in a single action
- Have a clearly defined input and output  
- Be described in one concise sentence

Task: {task}

Steps:
1. ...
```

**When to use:** Simple sequential tasks with no parallelism opportunities.

#### Pattern B — Dependency-Annotated Plan (DAG)

```
Decompose this task into sub-tasks. For each sub-task, specify:
- ID (e.g., s1, s2, ...)
- Description
- Type: one of [llm_reasoning, tool_call, code_execution, retrieval]
- Dependencies: list of sub-task IDs that must complete first
- Expected output format

Task: {task}
Available tools: {tools}

Plan (as JSON):
```

**When to use:** Complex tasks with parallel opportunities and tool interactions.

#### Pattern C — Question-Decomposition (QD) Pattern

Inspired by **Least-to-Most Prompting** (Zhou et al., 2022):

```
To answer the following question, what sub-questions do I need 
to answer first? List them in order from simplest to most complex.

Question: {task}

Sub-questions:
1. (simplest) ...
2. ...
...
n. (most complex, builds on answers to all previous) ...
```

**When to use:** Knowledge-intensive tasks where the decomposition structure is a chain of increasingly complex questions.

#### Pattern D — Role-Based Decomposition

```
Imagine you are managing a team of specialists:
- Researcher (can search the web and databases)
- Analyst (can perform calculations and data analysis)
- Writer (can draft and edit text)
- Reviewer (can evaluate quality and correctness)

Assign tasks to each specialist to accomplish:
Task: {task}

For each assignment, specify:
- Specialist role
- Specific instruction for that specialist
- What information they need from other specialists
- What they should produce
```

**When to use:** Multi-faceted tasks requiring diverse capabilities; naturally maps to multi-agent architectures.

#### Pattern E — Failure-Aware Decomposition

```
Decompose this task into steps. For each step, also identify:
- What could go wrong (failure modes)
- How to detect failure (verification criteria)
- What to do if this step fails (fallback strategy)

Task: {task}
```

**When to use:** High-stakes tasks where error detection and recovery are critical.

### 1.5.2.4 Evaluation of Decomposition Quality

A decomposition can be evaluated along multiple orthogonal axes:

**Definition.** Given a task $\mathcal{T}$ and a decomposition $D = \{s_1, \dots, s_n\}$, the **decomposition quality** is:

$$
Q(D \mid \mathcal{T}) = \alpha \cdot \text{Coverage}(D) + \beta \cdot \text{Atomicity}(D) + \gamma \cdot \text{Coherence}(D) + \delta \cdot \text{Efficiency}(D) - \lambda \cdot \text{Redundancy}(D)
$$

**Metric definitions:**

| Metric | Definition | Measurement |
|---|---|---|
| **Coverage** | Fraction of original task requirements addressed by at least one sub-task | $\text{Coverage}(D) = \frac{|\text{reqs addressed by } D|}{|\text{reqs of } \mathcal{T}|}$ |
| **Atomicity** | Fraction of sub-tasks that are truly solvable in a single LLM call | $\text{Atomicity}(D) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\text{complexity}(s_i) \leq \tau]$ |
| **Coherence** | Logical consistency of the dependency structure; no circular dependencies, no missing inputs | Binary or graded (0 = cyclic/broken, 1 = valid DAG with complete data flow) |
| **Efficiency** | Inverse of total steps, weighted by parallelism | $\text{Efficiency}(D) = \frac{1}{\text{critical\_path\_length}(D)}$ |
| **Redundancy** | Degree of overlapping work across sub-tasks | $\text{Redundancy}(D) = \frac{\text{duplicated content across sub-task outputs}}{\text{total content}}$ |

**Automated evaluation via LLM-as-Judge:**

```python
DECOMPOSITION_EVAL_PROMPT = """Evaluate the quality of this task decomposition.

Original Task: {task}

Proposed Decomposition:
{decomposition}

Rate each dimension from 1-5:
1. Coverage: Does the decomposition address ALL aspects of the original task?
2. Atomicity: Is each sub-task simple enough to solve in one step?
3. Coherence: Are dependencies logical? Is the data flow complete?
4. Efficiency: Is the decomposition minimal (no unnecessary steps)?
5. Redundancy: Is there duplicated work across sub-tasks?

For each dimension, provide:
- Score (1-5)
- Brief justification
- Specific issues found (if any)

Then provide an overall assessment and suggested improvements."""


async def evaluate_decomposition(
    task: str,
    decomposition: TaskPlan,
    evaluator_llm: LLMClient
) -> dict:
    """Evaluate decomposition quality using LLM-as-judge."""
    # Structural checks (deterministic, not LLM-dependent)
    structural_score = 1.0
    try:
        decomposition.topological_order()
    except ValueError:
        structural_score = 0.0  # cyclic dependency
    
    # Check all dependency references are valid
    valid_ids = {s.id for s in decomposition.subtasks}
    for s in decomposition.subtasks:
        for dep in s.dependencies:
            if dep not in valid_ids:
                structural_score *= 0.5  # broken reference
    
    # LLM-based semantic evaluation
    semantic_eval = await evaluator_llm.generate(
        prompt=DECOMPOSITION_EVAL_PROMPT.format(
            task=task,
            decomposition=decomposition.model_dump_json(indent=2)
        ),
        temperature=0.0
    )
    
    return {
        "structural_score": structural_score,
        "semantic_evaluation": semantic_eval,
        "num_subtasks": len(decomposition.subtasks),
        "critical_path_length": compute_critical_path(decomposition),
        "parallelism_factor": len(decomposition.subtasks) / max(compute_critical_path(decomposition), 1)
    }


def compute_critical_path(plan: TaskPlan) -> int:
    """Longest path in the DAG = minimum sequential steps."""
    id_to_task = {s.id: s for s in plan.subtasks}
    memo = {}
    
    def dfs(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        task = id_to_task[task_id]
        if not task.dependencies:
            memo[task_id] = 1
        else:
            memo[task_id] = 1 + max(dfs(dep) for dep in task.dependencies)
        return memo[task_id]
    
    return max(dfs(s.id) for s in plan.subtasks) if plan.subtasks else 0
```

### 1.5.2.5 Decomposition as a Chain Step Itself

A fundamental architectural insight: **decomposition is not a preprocessing step — it is the first step of the chain itself.** This means it is subject to the same quality controls, error handling, and evaluation as any other chain step.

**Meta-chain architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                       Meta-Chain                             │
│                                                             │
│  Step 0: Decompose(T) → Plan                                │
│          ↓                                                  │
│  Step 0.5: Evaluate(Plan) → Quality Score                   │
│          ↓                                                  │
│  [If quality < threshold: re-decompose with feedback]       │
│          ↓                                                  │
│  Step 0.9: Approve(Plan) → Frozen Plan                      │
│          ↓                                                  │
│  Steps 1..n: Execute(Frozen Plan) → Results                 │
│          ↓                                                  │
│  Step n+1: Synthesize(Results) → Final Output               │
└─────────────────────────────────────────────────────────────┘
```

**Self-improving decomposition loop:**

$$
D^{(k+1)} = \text{Replan}\!\bigl(\, \mathcal{T},\; D^{(k)},\; \text{Feedback}(D^{(k)}) \,\bigr) \quad \text{until } Q(D^{(k)}) \geq Q_{\min}
$$

```python
async def iterative_decomposition(
    task: str,
    planner_llm: LLMClient,
    evaluator_llm: LLMClient,
    quality_threshold: float = 0.8,
    max_iterations: int = 3
) -> TaskPlan:
    """Iteratively refine decomposition until quality threshold is met."""
    plan = await plan_task(task, tools=[], constraints={}, planner_llm=planner_llm)
    
    for iteration in range(max_iterations):
        eval_result = await evaluate_decomposition(task, plan, evaluator_llm)
        
        quality_score = eval_result["structural_score"]  # combine with semantic
        if quality_score >= quality_threshold:
            break
        
        # Re-plan with feedback
        replan_prompt = (
            f"Your previous decomposition of the task had issues:\n"
            f"{eval_result['semantic_evaluation']}\n\n"
            f"Original task: {task}\n"
            f"Previous plan: {plan.model_dump_json(indent=2)}\n\n"
            f"Please produce an improved decomposition that addresses "
            f"the identified issues."
        )
        plan = await plan_task(
            replan_prompt, tools=[], constraints={},
            planner_llm=planner_llm
        )
    
    return plan
```

---

## 1.5.3 Decomposition Granularity

### 1.5.3.1 The Granularity Spectrum

Granularity refers to the **size and specificity** of each sub-task in a decomposition. It exists on a continuous spectrum:

$$
\text{Granularity} \in [\text{coarse},\; \text{fine}]
$$

| Granularity | Steps for "Write a research report" | Step Example |
|---|---|---|
| **Very Coarse** ($n = 1$) | 1 step | "Write a complete research report on X" |
| **Coarse** ($n = 3$) | 3 steps | "Research → Draft → Review" |
| **Medium** ($n = 7$) | 7 steps | Individual sections + verification |
| **Fine** ($n = 15$) | 15 steps | Per-paragraph generation + fact-check |
| **Very Fine** ($n = 50$) | 50 steps | Per-sentence generation + citation verification |

### 1.5.3.2 Formal Analysis: Optimal Chain Length

Define the **end-to-end accuracy** of a chain of $n$ steps, where each step has independent success probability $p_i$:

$$
A(n) = \prod_{i=1}^{n} p_i
$$

If all steps have equal reliability $p$:

$$
A(n) = p^n
$$

This decays exponentially with $n$. For $p = 0.95$ and $n = 10$: $A(10) = 0.95^{10} \approx 0.60$. For $n = 20$: $A(20) \approx 0.36$.

**However**, finer decomposition typically **increases per-step accuracy** because each step is simpler:

$$
p_i(n) = p_{\text{base}} + (1 - p_{\text{base}}) \cdot f\!\left(\frac{1}{n}\right)
$$

where $f$ is a monotonically increasing function reflecting that simpler tasks are easier to solve. A reasonable model:

$$
p_i(n) = 1 - (1 - p_{\text{base}}) \cdot \left(\frac{n_0}{n}\right)^{-\alpha}
$$

where $n_0$ is a reference number of steps and $\alpha > 0$ controls how quickly per-step accuracy improves with finer granularity.

The **net accuracy** balances these two opposing forces:

$$
A(n) = \prod_{i=1}^{n} p_i(n) = \bigl[p(n)\bigr]^n
$$

Taking the log:

$$
\log A(n) = n \cdot \log p(n)
$$

The optimal chain length $n^*$ maximizes $A(n)$:

$$
n^* = \arg\max_n \; n \cdot \log p(n)
$$

Taking the derivative and setting to zero:

$$
\frac{d}{dn}\bigl[n \cdot \log p(n)\bigr] = \log p(n) + n \cdot \frac{p'(n)}{p(n)} = 0
$$

$$
\Rightarrow \quad n^* \cdot \frac{p'(n^*)}{p(n^*)} = -\log p(n^*)
$$

This equation shows that the optimal $n^*$ is where the **marginal improvement in per-step accuracy** (left side) exactly offsets the **marginal cost of adding one more step** (right side, which grows as $p$ decreases).

**Numerical illustration:**

```python
import numpy as np
from scipy.optimize import minimize_scalar

def per_step_accuracy(n, p_base=0.7, alpha=0.5):
    """Per-step accuracy improves as steps get simpler (larger n)."""
    return 1.0 - (1.0 - p_base) * (n ** (-alpha))

def chain_accuracy(n, p_base=0.7, alpha=0.5):
    """End-to-end accuracy: product of per-step accuracies."""
    n = max(int(round(n)), 1)
    p = per_step_accuracy(n, p_base, alpha)
    return p ** n

# Find optimal n
result = minimize_scalar(lambda n: -chain_accuracy(n), bounds=(1, 50), method='bounded')
n_optimal = int(round(result.x))
# Typical result: n* ≈ 5-8 for p_base=0.7, alpha=0.5
```

### 1.5.3.3 Diminishing Returns with Excessive Decomposition

Beyond the optimal $n^*$, additional decomposition **degrades** performance through multiple mechanisms:

**Mechanism 1 — Accumulating error propagation.** Each additional step is an opportunity for errors to compound. Even with high per-step accuracy, the exponential decay $p^n$ eventually dominates.

**Mechanism 2 — Information loss at boundaries.** When a complex thought is split across steps, the inter-step serialization (via `Render()` function from §1.4) inevitably loses nuance. Each serialization-deserialization cycle introduces:

$$
\mathcal{L}_{\text{boundary}}(s_i \to s_{i+1}) = D_{\text{KL}}\!\Bigl(\, p(\cdot \mid \text{full context}) \;\Big\|\; p\bigl(\cdot \mid \text{Render}(\text{output of } s_i)\bigr) \,\Bigr)
$$

The cumulative boundary loss across $n$ steps:

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{n-1} \mathcal{L}_{\text{boundary}}(s_i \to s_{i+1})
$$

**Mechanism 3 — Context dilution.** In very fine-grained chains, each step's context is dominated by orchestration metadata (step IDs, schemas, instructions) rather than the actual task content. The **signal-to-noise ratio** in the prompt degrades:

$$
\text{SNR}(n) = \frac{|\text{task-relevant tokens in context}|}{|\text{total tokens in context}|} \propto \frac{1}{n}
$$

**Mechanism 4 — Latency amplification.** Each additional step adds at least one LLM call (TTFT + generation time):

$$
\text{Latency}(n) = \sum_{i=1}^{n} \bigl(\, \text{TTFT}_i + \text{TPS}_i^{-1} \cdot |y_i| \,\bigr) + \sum_{i=1}^{n-1} t_{\text{overhead}_i}
$$

where $t_{\text{overhead}_i}$ includes serialization, validation, tool-call latency, and network round trips.

**Mechanism 5 — Cost amplification.** Cost scales with total tokens processed:

$$
\text{Cost}(n) = \sum_{i=1}^{n} \bigl(\, c_{\text{input}} \cdot |\text{prompt}_i| + c_{\text{output}} \cdot |y_i| \,\bigr)
$$

Finer granularity means more prompts, each carrying substantial overhead (system prompt, instructions, context), so total cost grows **super-linearly** in $n$.

### 1.5.3.4 Granularity Impact Summary

| Dimension | Coarse (small $n$) | Fine (large $n$) | Optimal |
|---|---|---|---|
| **Per-step accuracy** | Low (steps too complex) | High (steps trivial) | Balanced |
| **End-to-end accuracy** | Low (single step fails on complex task) | Low (error accumulation) | Maximum at $n^*$ |
| **Latency** | Low (few calls) | High (many calls) | Minimize critical path |
| **Cost** | Low (less total tokens) | High (overhead duplication) | Budget-constrained $n$ |
| **Debuggability** | Low (monolithic failure) | High (pinpoint failure step) | Fine enough to isolate failures |
| **Flexibility** | Low (rigid single step) | High (composable steps) | Match to reuse needs |

### 1.5.3.5 Information Loss at Decomposition Boundaries

This critical phenomenon deserves dedicated analysis. When the expert or LLM draws a boundary between steps $s_i$ and $s_{i+1}$, it implicitly makes a judgment about **what information crosses the boundary** and **what is discarded**.

**Formal model.** Let $I_i$ be the total information available at the end of step $i$ (including internal reasoning, discarded alternatives, confidence estimates). The boundary transmits only $I_i^{\text{transmitted}} \subseteq I_i$:

$$
\text{Boundary Loss}_i = H(I_i) - H(I_i^{\text{transmitted}})
$$

where $H(\cdot)$ is Shannon entropy. The **total information retained** across the full chain is:

$$
I_{\text{final}} = I_0 - \sum_{i=1}^{n-1} \text{Boundary Loss}_i
$$

**Sources of boundary loss:**

| Source | Example | Mitigation |
|---|---|---|
| **Truncation of reasoning** | Step $i$ considers 5 alternatives but only passes the chosen one | Pass top-$k$ candidates with confidence scores |
| **Schema rigidity** | Output schema doesn't have a field for an unexpected insight | Use flexible schemas with `additional_notes` field |
| **Summarization loss** | Compressed state drops nuanced details | Keep verbatim window + compressed prefix (§1.4.2) |
| **Type coercion** | Rich object → JSON string → parsed object loses method semantics | Use shared typed objects (Pydantic models) |
| **Implicit knowledge loss** | The LLM's "understanding" of context cannot be fully serialized | Include key reasoning traces in the output |

**Practical mitigation — rich output schemas:**

```python
class StepOutput(BaseModel):
    """Schema that minimizes boundary information loss."""
    primary_result: str                     # the main answer
    confidence: float                       # how certain the step is
    alternatives_considered: list[str]      # what else was considered
    reasoning_trace: str                    # why this result was chosen
    unresolved_questions: list[str]         # what the step couldn't determine
    metadata: dict = Field(default_factory=dict)  # extensible
    warnings: list[str] = Field(default_factory=list)
```

---

## 1.5.4 Decomposition Frameworks

### 1.5.4.1 Hierarchical Task Network (HTN)-Inspired Decomposition

HTN planning (Erol, Hendler & Nau, 1994) is a classical AI planning formalism where complex tasks are decomposed by applying **decomposition methods** that replace abstract tasks with networks of simpler tasks.

**Formal definition.** An HTN planning problem is a tuple:

$$
\Pi_{\text{HTN}} = \bigl(\, \mathcal{O},\; \mathcal{M},\; s_0,\; \mathcal{T}_0 \,\bigr)
$$

| Component | Definition |
|---|---|
| $\mathcal{O}$ | Set of **primitive operators** (directly executable actions) |
| $\mathcal{M}$ | Set of **decomposition methods** (rules for breaking compound tasks into sub-networks) |
| $s_0$ | Initial state |
| $\mathcal{T}_0$ | Initial task network (the top-level goal) |

A **method** $m \in \mathcal{M}$ is:

$$
m = \bigl(\, \text{task}(m),\; \text{preconditions}(m),\; \text{subtasks}(m),\; \text{constraints}(m) \,\bigr)
$$

- $\text{task}(m)$: the compound task this method decomposes
- $\text{preconditions}(m)$: conditions on the state that must hold for this method to apply
- $\text{subtasks}(m)$: the sub-task network that replaces the compound task
- $\text{constraints}(m)$: ordering and binding constraints among subtasks

**Application to LLM chains.** Each "decomposition method" becomes a **template** that the orchestrator (or planner LLM) can select:

```python
from dataclasses import dataclass, field

@dataclass
class HTNMethod:
    """A decomposition method in the HTN framework."""
    name: str
    target_task_type: str                    # what compound task this decomposes
    preconditions: callable                  # state → bool
    subtask_template: list[dict]             # ordered list of sub-task specs
    ordering_constraints: list[tuple] = field(default_factory=list)
    
    def applies(self, task: dict, state: dict) -> bool:
        return (task["type"] == self.target_task_type and 
                self.preconditions(state))
    
    def decompose(self, task: dict, state: dict) -> list[dict]:
        """Instantiate subtask template with task-specific bindings."""
        subtasks = []
        for template in self.subtask_template:
            subtask = {**template}
            # Bind variables from parent task
            for key, value in subtask.items():
                if isinstance(value, str) and value.startswith("$"):
                    var_name = value[1:]
                    subtask[key] = task.get(var_name, state.get(var_name))
            subtasks.append(subtask)
        return subtasks


class HTNPlanner:
    """HTN-inspired planner for LLM chain construction."""
    def __init__(self):
        self.methods: list[HTNMethod] = []
        self.primitive_types: set[str] = set()
    
    def register_method(self, method: HTNMethod):
        self.methods.append(method)
    
    def register_primitive(self, task_type: str):
        self.primitive_types.add(task_type)
    
    def plan(self, task: dict, state: dict) -> list[dict]:
        """Recursively decompose until all tasks are primitive."""
        if task["type"] in self.primitive_types:
            return [task]  # base case: already executable
        
        # Find applicable methods
        applicable = [m for m in self.methods if m.applies(task, state)]
        if not applicable:
            raise ValueError(f"No method decomposes task type: {task['type']}")
        
        # Select best method (here: first applicable; could use heuristics)
        method = applicable[0]
        subtasks = method.decompose(task, state)
        
        # Recursively decompose each subtask
        plan = []
        for subtask in subtasks:
            plan.extend(self.plan(subtask, state))
        
        return plan


# --- Register methods for a research agent ---

planner = HTNPlanner()

# Primitive task types (directly executable)
planner.register_primitive("web_search")
planner.register_primitive("llm_generate")
planner.register_primitive("llm_evaluate")
planner.register_primitive("code_execute")

# Compound → decomposition methods
planner.register_method(HTNMethod(
    name="research_report_method",
    target_task_type="write_research_report",
    preconditions=lambda state: True,
    subtask_template=[
        {"type": "gather_evidence", "topic": "$topic"},
        {"type": "llm_generate", "instruction": "Draft report on $topic using gathered evidence"},
        {"type": "llm_evaluate", "instruction": "Review report for accuracy and completeness"},
    ]
))

planner.register_method(HTNMethod(
    name="gather_evidence_method",
    target_task_type="gather_evidence",
    preconditions=lambda state: True,
    subtask_template=[
        {"type": "web_search", "query": "$topic recent developments"},
        {"type": "web_search", "query": "$topic statistics data"},
        {"type": "llm_generate", "instruction": "Extract key facts from search results"},
    ]
))

# Usage
task = {"type": "write_research_report", "topic": "quantum computing market"}
primitive_plan = planner.plan(task, state={})
# Returns flat list of 6 primitive tasks
```

**Advantages of HTN-inspired decomposition:**
1. **Reusable methods** — once defined, a method can decompose any task of its type.
2. **Hierarchical abstraction** — reasoning occurs at the appropriate level.
3. **Domain knowledge encoding** — experts embed their decomposition heuristics directly.
4. **Predictable structure** — the decomposition follows known patterns, enabling optimization.

### 1.5.4.2 Goal-Subgoal Decomposition

Goal-subgoal decomposition models the task as achieving a **top-level goal** $G$ by identifying and achieving a set of **subgoals** $\{g_1, \dots, g_k\}$ whose conjunction entails $G$.

**Formal model:**

$$
G = g_1 \wedge g_2 \wedge \cdots \wedge g_k
$$

Each subgoal may itself be decomposed:

$$
g_i = g_{i,1} \wedge g_{i,2} \wedge \cdots \wedge g_{i, m_i}
$$

forming a **goal tree** (AND-tree) or, when alternative decompositions exist, an **AND-OR tree.**

**AND-OR Tree:**

$$
G = \underbrace{(g_1 \wedge g_2)}_{\text{Method A}} \;\vee\; \underbrace{(g_3 \wedge g_4 \wedge g_5)}_{\text{Method B}}
$$

- **AND nodes:** All children must be achieved.
- **OR nodes:** At least one child must be achieved (alternative strategies).

```
                      G (OR)
                     / \
                   /     \
            AND-1          AND-2
           /    \         / | \
         g₁    g₂      g₃  g₄  g₅
```

```python
from enum import Enum

class NodeType(Enum):
    AND = "and"    # all children must succeed
    OR = "or"      # at least one child must succeed
    LEAF = "leaf"  # directly achievable

@dataclass
class GoalNode:
    id: str
    description: str
    node_type: NodeType
    children: list["GoalNode"] = field(default_factory=list)
    achieved: bool = False
    result: any = None
    
    def is_achievable(self) -> bool:
        if self.node_type == NodeType.LEAF:
            return True
        elif self.node_type == NodeType.AND:
            return all(child.is_achievable() for child in self.children)
        elif self.node_type == NodeType.OR:
            return any(child.is_achievable() for child in self.children)

    def evaluate(self) -> bool:
        """Check if the goal is achieved based on children's status."""
        if self.node_type == NodeType.LEAF:
            return self.achieved
        elif self.node_type == NodeType.AND:
            return all(child.evaluate() for child in self.children)
        elif self.node_type == NodeType.OR:
            return any(child.evaluate() for child in self.children)


class GoalDecomposer:
    """LLM-driven goal-subgoal decomposition."""
    
    DECOMPOSE_PROMPT = """Given the goal below, identify the subgoals 
needed to achieve it. Classify the relationship:
- AND: ALL subgoals must be achieved
- OR: achieving ANY ONE subgoal is sufficient

Goal: {goal}

For each subgoal, indicate if it is directly achievable (LEAF) 
or needs further decomposition (COMPOUND).

Respond as JSON:
{{
  "relationship": "AND" or "OR",
  "subgoals": [
    {{"description": "...", "type": "LEAF" or "COMPOUND"}}
  ]
}}"""
    
    async def decompose(self, goal: str, llm: LLMClient, 
                        max_depth: int = 3, depth: int = 0) -> GoalNode:
        if depth >= max_depth:
            return GoalNode(id=f"g_{depth}", description=goal, 
                          node_type=NodeType.LEAF)
        
        response = await llm.generate(
            prompt=self.DECOMPOSE_PROMPT.format(goal=goal),
            temperature=0.0
        )
        parsed = json.loads(response)
        
        node_type = NodeType.AND if parsed["relationship"] == "AND" else NodeType.OR
        children = []
        
        for i, sg in enumerate(parsed["subgoals"]):
            if sg["type"] == "LEAF":
                child = GoalNode(
                    id=f"g_{depth}_{i}", 
                    description=sg["description"],
                    node_type=NodeType.LEAF
                )
            else:
                child = await self.decompose(
                    sg["description"], llm, max_depth, depth + 1
                )
            children.append(child)
        
        return GoalNode(
            id=f"g_{depth}", 
            description=goal,
            node_type=node_type, 
            children=children
        )
```

**Execution strategy for AND-OR trees:**

- **AND nodes:** Execute all children (potentially in parallel). If any fails, the AND node fails — but if the parent is an OR node, try the alternative branch.
- **OR nodes:** Execute children in order of estimated cost/likelihood of success. Stop at the first success.

$$
\text{Expected Cost}(\text{OR}[g_1, g_2]) = c_1 + (1 - p_1) \cdot c_2
$$

$$
\text{Expected Cost}(\text{AND}[g_1, g_2]) = c_1 + c_2
$$

This gives a clear ordering heuristic for OR nodes: **try cheapest-most-likely first.**

### 1.5.4.3 Means-Ends Analysis (MEA)

Means-Ends Analysis, introduced by Newell & Simon (1963) in the General Problem Solver, is a recursive strategy that:

1. Identifies the **difference** $\Delta$ between the current state $s$ and the goal state $g$.
2. Selects an **operator** $o$ that reduces $\Delta$.
3. If $o$'s preconditions are not met, recursively achieves them first.

**Formal algorithm:**

$$
\text{MEA}(s, g) = \begin{cases}
\emptyset & \text{if } s = g \\
[o] \oplus \text{MEA}(\text{apply}(o, s),\; g) & \text{if } \text{precond}(o) \subseteq s \\
\text{MEA}(s, \text{precond}(o)) \oplus [o] \oplus \text{MEA}(\text{apply}(o, s),\; g) & \text{otherwise}
\end{cases}
$$

where $o = \arg\max_{o'} \text{reduction}(o', \Delta(s, g))$.

**Mapping to LLM chains:**

| MEA Concept | LLM Chain Analog |
|---|---|
| Current state $s$ | Current chain state $S_t$ |
| Goal state $g$ | Desired output specification |
| Difference $\Delta$ | Gap between current output and acceptance criteria |
| Operator $o$ | An LLM call, tool invocation, or code execution |
| Preconditions | Required inputs / state fields for a step |
| Operator selection | LLM-driven reasoning about which step to take next |

```python
class MeansEndsPlanner:
    """Means-Ends Analysis planner for chain construction."""
    
    def __init__(self, operators: list[dict], llm: LLMClient):
        self.operators = operators  # available actions
        self.llm = llm
    
    async def identify_difference(self, current_state: dict, 
                                   goal: dict) -> dict:
        """Use LLM to identify the gap between current and goal state."""
        prompt = (
            f"Current state:\n{json.dumps(current_state, indent=2)}\n\n"
            f"Goal state:\n{json.dumps(goal, indent=2)}\n\n"
            f"What is the primary difference/gap between the current state "
            f"and the goal? What needs to change? Respond as JSON:\n"
            f'{{"difference": "...", "magnitude": 0.0-1.0, '
            f'"category": "missing_data|wrong_format|incomplete_analysis|..."}}'
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return json.loads(response)
    
    async def select_operator(self, difference: dict, 
                               current_state: dict) -> dict:
        """Select the best operator to reduce the identified difference."""
        prompt = (
            f"Difference to resolve: {json.dumps(difference)}\n\n"
            f"Available operators:\n"
            + "\n".join(f"- {op['name']}: {op['description']} "
                       f"(preconditions: {op['preconditions']})"
                       for op in self.operators)
            + f"\n\nSelect the operator that best reduces this difference. "
            f"Check if its preconditions are met given the current state.\n"
            f"Current state keys: {list(current_state.keys())}\n\n"
            f"Respond as JSON: "
            f'{{"operator": "name", "preconditions_met": true/false, '
            f'"unmet_preconditions": [...]}}'
        )
        response = await self.llm.generate(prompt=prompt, temperature=0.0)
        return json.loads(response)
    
    async def plan(self, initial_state: dict, goal: dict, 
                   max_steps: int = 15) -> list[dict]:
        """Generate a plan using Means-Ends Analysis."""
        plan = []
        state = {**initial_state}
        
        for step in range(max_steps):
            diff = await self.identify_difference(state, goal)
            
            if diff["magnitude"] < 0.05:  # close enough to goal
                break
            
            selection = await self.select_operator(diff, state)
            
            if not selection["preconditions_met"]:
                # Recursively achieve preconditions
                for precond in selection["unmet_preconditions"]:
                    sub_plan = await self.plan(
                        state, 
                        {"requirement": precond},
                        max_steps=max_steps - step
                    )
                    plan.extend(sub_plan)
                    # Simulate state update
                    state[precond] = "achieved"
            
            plan.append({
                "operator": selection["operator"],
                "target_difference": diff["difference"],
                "step_number": len(plan)
            })
            
            # Simulate: assume operator achieves its effect
            state[f"result_{selection['operator']}"] = "completed"
        
        return plan
```

### 1.5.4.4 PDDL-Inspired Planning for Chain Construction

PDDL (Planning Domain Definition Language) is the standard formal language for classical AI planning. Its key contribution to LLM chain design is its **rigorous separation of domain definition from problem specification**.

**PDDL structure mapped to LLM chains:**

| PDDL Component | LLM Chain Analog |
|---|---|
| **Domain** | The set of available tools, LLM capabilities, and state schemas |
| **Actions** (with preconditions and effects) | Chain steps with input requirements and output guarantees |
| **Predicates** | Boolean conditions on state (e.g., `has_data`, `is_verified`) |
| **Initial state** | User query + initial context |
| **Goal state** | Acceptance criteria for the chain output |

**Formal PDDL-style action definition for LLM chains:**

$$
\text{Action}(a) = \bigl(\, \text{name}(a),\; \text{params}(a),\; \text{pre}(a),\; \text{eff}^+(a),\; \text{eff}^-(a) \,\bigr)
$$

- $\text{pre}(a)$: preconditions — predicates that must be true before $a$ executes
- $\text{eff}^+(a)$: add effects — predicates that become true after $a$
- $\text{eff}^-(a)$: delete effects — predicates that become false after $a$

```python
@dataclass
class PDDLAction:
    """PDDL-inspired action definition for chain planning."""
    name: str
    parameters: list[str]
    preconditions: set[str]     # predicates that must be True
    add_effects: set[str]       # predicates set to True
    delete_effects: set[str]    # predicates set to False
    cost: float = 1.0           # for cost-optimal planning
    executor: callable = None   # the actual function to run

@dataclass
class PDDLProblem:
    """A planning problem in PDDL terms."""
    domain_actions: list[PDDLAction]
    initial_state: set[str]     # set of true predicates
    goal_state: set[str]        # predicates that must be true at end


class PDDLPlanner:
    """Forward-search planner using PDDL-style actions."""
    
    def __init__(self, problem: PDDLProblem):
        self.problem = problem
    
    def get_applicable(self, state: set[str]) -> list[PDDLAction]:
        """Return all actions whose preconditions are satisfied."""
        return [a for a in self.problem.domain_actions 
                if a.preconditions.issubset(state)]
    
    def apply_action(self, state: set[str], action: PDDLAction) -> set[str]:
        """Return new state after applying action."""
        new_state = (state - action.delete_effects) | action.add_effects
        return new_state
    
    def goal_reached(self, state: set[str]) -> bool:
        return self.problem.goal_state.issubset(state)
    
    def plan_bfs(self) -> list[PDDLAction]:
        """Breadth-first search for a plan (optimal for unit costs)."""
        from collections import deque
        
        queue = deque()
        queue.append((frozenset(self.problem.initial_state), []))
        visited = {frozenset(self.problem.initial_state)}
        
        while queue:
            state, actions = queue.popleft()
            
            if self.goal_reached(state):
                return actions
            
            for action in self.get_applicable(state):
                new_state = frozenset(self.apply_action(state, action))
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, actions + [action]))
        
        raise ValueError("No plan found")
    
    def plan_astar(self, heuristic: callable = None) -> list[PDDLAction]:
        """A* search with a heuristic for cost-optimal planning."""
        import heapq
        
        if heuristic is None:
            heuristic = lambda state: len(self.problem.goal_state - state)
        
        start = frozenset(self.problem.initial_state)
        heap = [(heuristic(start), 0, start, [])]  # (f, g, state, actions)
        visited = {}
        
        while heap:
            f, g, state, actions = heapq.heappop(heap)
            
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g
            
            if self.goal_reached(state):
                return actions
            
            for action in self.get_applicable(state):
                new_state = frozenset(self.apply_action(state, action))
                new_g = g + action.cost
                new_f = new_g + heuristic(new_state)
                heapq.heappush(heap, (new_f, new_g, new_state, 
                                      actions + [action]))
        
        raise ValueError("No plan found")


# --- Define a research-agent planning domain ---

domain_actions = [
    PDDLAction(
        name="search_web",
        parameters=["query"],
        preconditions={"has_query"},
        add_effects={"has_search_results"},
        delete_effects=set(),
        cost=1.0
    ),
    PDDLAction(
        name="extract_facts",
        parameters=["search_results"],
        preconditions={"has_search_results"},
        add_effects={"has_extracted_facts"},
        delete_effects=set(),
        cost=1.0
    ),
    PDDLAction(
        name="analyze_data",
        parameters=["facts"],
        preconditions={"has_extracted_facts"},
        add_effects={"has_analysis"},
        delete_effects=set(),
        cost=2.0
    ),
    PDDLAction(
        name="draft_report",
        parameters=["analysis"],
        preconditions={"has_analysis", "has_extracted_facts"},
        add_effects={"has_draft"},
        delete_effects=set(),
        cost=2.0
    ),
    PDDLAction(
        name="review_report",
        parameters=["draft"],
        preconditions={"has_draft"},
        add_effects={"has_reviewed_report"},
        delete_effects=set(),
        cost=1.5
    ),
]

problem = PDDLProblem(
    domain_actions=domain_actions,
    initial_state={"has_query"},
    goal_state={"has_reviewed_report"}
)

planner = PDDLPlanner(problem)
plan = planner.plan_astar()
# Returns: [search_web, extract_facts, analyze_data, draft_report, review_report]
```

### 1.5.4.5 Decomposition Trees and Dependency Graphs

The final unifying representation for all decomposition frameworks is the **decomposition tree** (hierarchical view) and its flattened form, the **dependency graph** (execution view).

**Decomposition Tree (Hierarchical View):**

$$
\text{Tree}(\mathcal{T}) = (\mathcal{T}, \; \{(\mathcal{T}, s_i)\}_{i=1}^{k}, \; \bigcup_{i=1}^{k} \text{Tree}(s_i))
$$

Leaf nodes are **primitive tasks** (directly executable). Internal nodes are **compound tasks** (requiring further decomposition).

**Dependency Graph (Execution View):**

$$
G_{\text{exec}} = \bigl(\, V_{\text{leaves}},\; E_{\text{depends}} \,\bigr)
$$

where $V_{\text{leaves}}$ is the set of leaf (primitive) tasks from the decomposition tree, and $E_{\text{depends}}$ captures data-flow dependencies.

**Conversion algorithm:**

```python
def tree_to_dependency_graph(tree: dict) -> tuple[list[dict], list[tuple]]:
    """Flatten a decomposition tree into a dependency graph."""
    nodes = []
    edges = []
    
    def collect_leaves(node: dict, parent_deps: list[str] = None) -> list[str]:
        """Recursively collect leaf nodes and infer dependencies."""
        parent_deps = parent_deps or []
        
        if not node.get("children"):
            # Leaf node
            leaf_id = node["id"]
            nodes.append({
                "id": leaf_id,
                "description": node["task"],
                "depth": node.get("depth", 0)
            })
            # Depends on all leaves from prior sibling subtrees
            for dep in parent_deps:
                edges.append((dep, leaf_id))
            return [leaf_id]
        
        all_leaf_ids = []
        prior_sibling_leaves = []
        
        for child in node["children"]:
            child_deps = parent_deps + prior_sibling_leaves
            child_leaves = collect_leaves(child, child_deps)
            
            # Sequential assumption: each child depends on prior siblings
            # (for parallel, omit this line and track only data deps)
            prior_sibling_leaves.extend(child_leaves)
            all_leaf_ids.extend(child_leaves)
        
        return all_leaf_ids
    
    collect_leaves(tree)
    return nodes, edges


def visualize_dependency_graph(nodes: list[dict], edges: list[tuple]) -> str:
    """Generate a Mermaid diagram of the dependency graph."""
    lines = ["graph LR"]
    for node in nodes:
        lines.append(f'    {node["id"]}["{node["description"][:40]}"]')
    for src, dst in edges:
        lines.append(f'    {src} --> {dst}')
    return "\n".join(lines)
```

**Critical path analysis on the dependency graph** determines the theoretical minimum execution time:

$$
T_{\text{min}} = \max_{\text{path } \pi \in G_{\text{exec}}} \sum_{v \in \pi} \text{duration}(v)
$$

The critical path length divided by total steps gives the **parallelism factor**:

$$
\text{Parallelism Factor} = \frac{|V_{\text{leaves}}|}{|\text{Critical Path}|}
$$

A parallelism factor of 1.0 means purely sequential; higher values indicate greater parallelization opportunities.

---

## Summary: Decomposition Strategy Selection

| Scenario | Recommended Strategy | Rationale |
|---|---|---|
| Well-understood, repeatable workflow | **Manual + HTN** methods | Domain expertise encodes optimal decomposition; reusable templates |
| Novel, one-off complex task | **Automatic** (LLM planner) with iterative refinement | No prior template exists; LLM adapts to task structure |
| High-stakes, error-sensitive | **CTA-driven manual** + failure-aware patterns | Expert identifies critical failure points; explicit verification steps |
| Multi-step problem-solving | **Goal-Subgoal** (AND-OR trees) | Natural recursive structure; alternative strategies via OR nodes |
| Tasks with clear precondition/effect semantics | **PDDL-inspired** formal planning | Guarantees completeness; enables cost-optimal plan search |
| Progressive refinement needed | **MEA** | Iteratively reduces difference between current state and goal |
| Unknown optimal granularity | Start **coarse** ($n \approx 3{-}5$), measure accuracy, then refine | Avoids premature over-decomposition; empirically finds $n^*$ |

**Universal principle:** The decomposition is not a static artifact — it is a **living hypothesis** about task structure. Monitor execution, measure step-level success rates, and iteratively refine the decomposition based on empirical evidence.