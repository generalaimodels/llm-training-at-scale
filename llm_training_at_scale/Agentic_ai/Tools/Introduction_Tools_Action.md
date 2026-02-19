

# Chapter 2: Tools — Comprehensive SOTA Reference

---

## 2.1 Introduction: Models, Tools, and Agents

### 2.1.1 Formal Definitions and Taxonomy

---

#### Definition of a Foundation Model in the Tool-Use Context

A foundation model, in the context of tool-augmented systems, is not merely a next-token predictor. It functions as a **reasoning kernel** — a probabilistic controller that conditions its output distribution on the current environmental state, the set of available tools, and the full history of prior interactions.

**Formal Statement.** Let $\mathcal{M}_\theta$ denote a language model parameterized by $\theta$. At each decision step $t$, the model computes a distribution over the next action $a_t$:

$$
P_\theta(a_t \mid s_t, \mathcal{T}, \mathcal{H}_{<t})
$$

where:

- $s_t \in \mathcal{S}$ is the **current state** of the interaction (includes the latest user message, the most recent tool result, or any environmental observation),
- $\mathcal{T} = \{\tau_1, \tau_2, \ldots, \tau_N\}$ is the **available tool set**, each tool described by its schema and natural-language documentation injected into the prompt,
- $\mathcal{H}_{<t} = \{(a_0, o_0), (a_1, o_1), \ldots, (a_{t-1}, o_{t-1})\}$ is the **interaction history**, the full sequence of prior actions and observations,
- $a_t$ is the **action**, which belongs to the union space $a_t \in \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{tool}}$, meaning the model either generates a natural-language response or emits a structured tool invocation.

The action space is formally:

$$
\mathcal{A} = \underbrace{\Sigma^*}_{\text{text generation}} \;\cup\; \underbrace{\{(\tau_i, \text{args}_i) \mid \tau_i \in \mathcal{T}, \; \text{args}_i \in \sigma_{in}^{(\tau_i)}\}}_{\text{tool invocations}}
$$

where $\Sigma^*$ is the set of all finite token sequences, and $\sigma_{in}^{(\tau_i)}$ is the valid input schema for tool $\tau_i$.

**Parametric vs. Non-Parametric Knowledge.** A critical distinction in tool-augmented systems is between two knowledge sources:

| Knowledge Type | Source | Properties | Examples |
|---|---|---|---|
| **Parametric** $K_\theta$ | Model weights $\theta$ | Fixed post-training, subject to knowledge cutoff, susceptible to hallucination, no real-time access | Facts memorized during pretraining, learned reasoning patterns |
| **Non-parametric** $K_{\text{ext}}$ | Tools and external systems | Dynamic, real-time, grounded in world state, verifiable, requires tool invocation to access | Database records, web search results, file contents, computation outputs |

The total knowledge available to a tool-augmented model is:

$$
K_{\text{total}} = K_\theta \cup K_{\text{ext}}(\mathcal{T})
$$

This distinction is fundamental: the model must learn to recognize when its parametric knowledge is insufficient or unreliable and must delegate to non-parametric sources via tool invocation. This is a meta-cognitive capability — the model must possess calibrated uncertainty over its own knowledge to decide *when* to call a tool.

**The Model as Controller vs. Executor.** In a tool-augmented architecture, the model occupies one of two roles at any given step:

1. **Controller role**: The model decides *what* action to take — which tool to call, with what arguments, or whether to respond directly. It performs planning, reasoning, and decision-making. The model does not directly execute side effects; it delegates execution to the tool runtime.

2. **Executor role** (rare, but relevant): In scenarios like code generation without a sandbox, the model produces outputs that *are* the final artifact — the model directly "executes" by generating text. However, this conflates reasoning and execution, and modern best practices favor separation.

The clean separation:

$$
\underbrace{\mathcal{M}_\theta}_{\text{Controller: decides actions}} \;\longrightarrow\; \underbrace{\text{Runtime}}_{\text{Executor: performs actions}} \;\longrightarrow\; \underbrace{\text{Environment}}_{\text{World: provides observations}}
$$

This architectural separation is not merely stylistic — it is a **safety boundary**. The model's outputs are inspectable, validatable, and rejectable before execution. This enables human-in-the-loop oversight, policy enforcement, and sandboxing.

---

#### Definition of a Tool

A tool is a discrete, externally executable capability that the language model can invoke through a structured interface. It is the fundamental unit of extensibility in tool-augmented systems.

**Formal Specification.** A tool is defined as a 5-tuple:

$$
\tau = (\text{name}, \; \text{desc}, \; \sigma_{in}, \; \sigma_{out}, \; f)
$$

where:

| Component | Type | Description |
|---|---|---|
| $\text{name}$ | $\text{String}$ | A unique identifier (e.g., `get_weather`, `execute_sql`). Used by the model to reference the tool in its output. |
| $\text{desc}$ | $\text{String}$ | A natural-language description explaining *what* the tool does, *when* to use it, and *when not to*. This is the primary signal the model uses for tool selection. |
| $\sigma_{in}$ | $\text{JSON Schema}$ | A formal schema defining the tool's input parameters — their types, constraints, required/optional status, and documentation. |
| $\sigma_{out}$ | $\text{JSON Schema}$ | A formal schema defining the structure of the tool's output (not always explicitly specified in current APIs, but critical for robust systems). |
| $f$ | $\sigma_{in} \rightarrow \sigma_{out}$ | The executable function — the actual implementation that performs the computation or side effect. This is opaque to the model. |

**Tools as External Function Interfaces.** The core insight is that tools extend the model's capability space beyond the closed world of token generation. Without tools, the model is confined to:

$$
\mathcal{C}_{\text{base}} = \{\text{text generation, reasoning over context, pattern completion}\}
$$

With tools, the capability space expands to:

$$
\mathcal{C}_{\text{augmented}} = \mathcal{C}_{\text{base}} \cup \bigcup_{\tau \in \mathcal{T}} \mathcal{C}_\tau
$$

where $\mathcal{C}_\tau$ is the set of capabilities provided by tool $\tau$ (e.g., arithmetic computation, database querying, web search, code execution).

**Tools as the Boundary Between Latent Reasoning and the External World.** The model operates in a latent representational space — it manipulates token sequences that represent but do not directly interact with the world. Tools are the **transduction boundary** — they convert the model's symbolic intentions (structured JSON arguments) into real-world effects (API calls, computations, file operations) and convert real-world observations (results, errors, data) back into symbolic form (text injected into context).

$$
\underbrace{\text{Latent Space (Model)}}_{\text{tokens, attention, reasoning}} \;\xrightleftharpoons[\text{result injection}]{\text{tool call emission}}\; \underbrace{\text{External World}}_{\text{APIs, databases, code runtimes, physical actuators}}
$$

---

#### Definition of an Agent

An agent is a composite system that integrates a language model with tools, planning capabilities, and state management to autonomously pursue goals through multi-step interaction with an environment.

**Formal Specification.** An agent is defined as a 4-tuple:

$$
\mathcal{A} = (\mathcal{M}, \; \mathcal{T}, \; \mathcal{P}, \; \mathcal{S})
$$

| Component | Role | Description |
|---|---|---|
| $\mathcal{M}$ | **Model** | The foundation model serving as the reasoning engine and controller. |
| $\mathcal{T}$ | **Tool Set** | The collection of available tools $\{\tau_1, \ldots, \tau_N\}$ the agent can invoke. |
| $\mathcal{P}$ | **Planning Module** | The strategy for decomposing goals into sub-tasks and selecting tool sequences. May be implicit (in-context reasoning) or explicit (external planner). |
| $\mathcal{S}$ | **State Manager** | The mechanism for maintaining, updating, and querying the agent's internal state across interaction steps — conversation history, working memory, task progress, intermediate results. |

**Agent Loop Formalization: Observe → Think → Act → Observe (OTAO Cycle).** An agent operates in a loop:

$$
\text{For } t = 1, 2, \ldots, T:
$$

$$
o_t = \text{Observe}(\text{env}_t, \; \mathcal{S}_{t-1}) \quad \text{(Perceive current state)}
$$

$$
p_t = \text{Think}(\mathcal{M}, \; o_t, \; \mathcal{H}_{<t}, \; \mathcal{T}) \quad \text{(Reason, plan, select action)}
$$

$$
a_t = \text{Act}(p_t) \quad \text{(Execute: generate text or invoke tool)}
$$

$$
o_{t+1} = \text{Observe}(\text{result}(a_t)) \quad \text{(Receive feedback)}
$$

$$
\mathcal{S}_t = \text{Update}(\mathcal{S}_{t-1}, \; a_t, \; o_{t+1}) \quad \text{(Update state)}
$$

The loop terminates when a stopping condition is met: the task is complete, a maximum step limit is reached, or the model emits a final response without a tool call.

**Critical Distinction: Agents vs. Pipelines vs. Chains vs. Single-Turn Tool Calling.**

| System Type | Decision-Making | Loop Structure | Tool Use | State |
|---|---|---|---|---|
| **Single-turn tool call** | One-shot | No loop | 0 or 1 tool call per turn | Stateless |
| **Chain** | Predetermined | Fixed sequence | Each step is a fixed tool/model call | Passed linearly |
| **Pipeline** | Predetermined with branching | DAG | Fixed tool assignment per node | DAG state |
| **Agent** | Dynamic, model-decided | Open-ended loop | Model chooses tools at runtime | Maintained across steps |

The defining property of an agent is **dynamic control flow**: the model itself determines the next action at each step, rather than following a pre-programmed sequence. This introduces both power (adaptability) and risk (unbounded execution, error accumulation).

Mathematically, a chain has a fixed computation graph $G_{\text{chain}}$, while an agent's computation graph $G_{\text{agent}}$ is generated at runtime:

$$
G_{\text{chain}} = \text{const} \quad \text{(determined at design time)}
$$

$$
G_{\text{agent}} = \mathcal{M}(G_{\text{partial}}, \; o_t) \quad \text{(extended at each step by the model)}
$$

---

### 2.1.2 Historical Evolution of Tool-Augmented Language Models

The trajectory from purely parametric language models to modern tool-augmented agents represents a fundamental shift in the role of language models — from knowledge stores to reasoning controllers.

**Phase 1: Purely Parametric Completion (Pre-2020).**
Early language models (GPT-2, BERT) operated as closed systems. All knowledge was encoded in parameters $\theta$ during pretraining. At inference, the model could only recombine information already stored in its weights. Limitations were severe: knowledge cutoffs, inability to perform reliable arithmetic, hallucination of facts, and zero capacity for real-world interaction. The model's output space was:

$$
\mathcal{Y} = \Sigma^* \quad \text{(only token sequences)}
$$

**Phase 2: Early Retrieval Augmentation (2020–2022).**
REALM (Guu et al., 2020) and RAG (Lewis et al., 2020) introduced the idea of conditioning generation on retrieved documents. The model's knowledge was augmented at inference time:

$$
P(y \mid x) = \sum_{d \in \mathcal{D}} P(y \mid x, d) \cdot P(d \mid x)
$$

where $\mathcal{D}$ is a document corpus and $P(d \mid x)$ is a retrieval distribution (typically based on dense embeddings). This was the first non-parametric knowledge augmentation but was limited to a single modality (text retrieval) and a single tool type (retriever).

**Phase 3: Toolformer — Self-Supervised Tool Learning (2023).**
Toolformer (Schick et al., 2023) demonstrated that language models could learn to use tools *without explicit tool-use supervision*. The method:

1. Annotated a pretraining corpus with candidate tool calls (API calls for calculator, search, etc.).
2. Retained only those annotations where the tool call reduced perplexity on future tokens:

$$
\text{Keep annotation } c \text{ if } \; L_{\text{with\_tool}}(c) < L_{\text{without\_tool}}
$$

3. Fine-tuned the model on the filtered corpus.

This was a milestone because it showed tool use could be learned as a *self-supervised* objective rather than requiring curated tool-use datasets.

**Phase 4: WebGPT, ChatGPT Plugins, and Function Calling APIs (2023).**
OpenAI's WebGPT introduced browsing as a tool action. ChatGPT Plugins (March 2023) opened a marketplace of third-party tools. The function calling API (June 2023) provided a structured interface for tool invocation — the model generates JSON-formatted function calls rather than free-text tool descriptions. This standardized the tool-calling interface:

```json
{
  "name": "get_weather",
  "arguments": "{\"city\": \"San Francisco\", \"unit\": \"celsius\"}"
}
```

**Phase 5: Native Tool Calling with Structured Outputs (2024–Present).**
Current-generation models (GPT-4o, Claude 3.5/4, Gemini 2.x) support native tool calling as a first-class capability:

- Tool schemas are part of the API request, not manually injected into prompts.
- Models generate structured tool calls with schema-validated outputs.
- Parallel tool calling enables multiple independent invocations per turn.
- Structured output modes guarantee JSON conformance via constrained decoding.
- Tool-use behavior is fine-tuned via RLHF/RLAIF with tool-use reward signals.

**Phase 6: Multi-Tool Orchestration and Agentic Systems (2024–Present).**
The frontier has moved from single-tool invocation to orchestrating complex workflows across multiple tools, with planning, error recovery, and multi-agent coordination. This is the agentic paradigm.

| Era | Capability | Tool Relationship | Control Flow |
|---|---|---|---|
| Phase 1 | Closed generation | None | Fixed |
| Phase 2 | Retrieval-augmented | Single implicit tool | Fixed |
| Phase 3 | Self-learned tool use | Multiple tools, learned | Model-decided, single-step |
| Phase 4 | API-based tool calling | Multiple tools, API-defined | Model-decided, multi-step |
| Phase 5 | Native structured tool calling | Multiple tools, schema-validated | Model-decided, parallel |
| Phase 6 | Agentic orchestration | Dynamic tool sets, multi-agent | Model-decided, recursive, planning-based |

---

### 2.1.3 Why Tools Are Necessary

Tools are not an optional enhancement — they are **architecturally necessary** to overcome fundamental limitations of purely parametric language models. Each limitation is structural, not merely a matter of scale.

**Limitation 1: Parametric Knowledge is Static, Incomplete, and Unreliable.**
Model weights $\theta$ are fixed after training. Any fact that changed after the knowledge cutoff date $t_{\text{cut}}$ is inaccessible:

$$
K_\theta(t) = K_\theta(t_{\text{cut}}) \quad \forall \; t > t_{\text{cut}}
$$

Moreover, even within the training distribution, parametric recall is lossy — the model compresses the entire training corpus into a fixed-dimensional parameter space, inevitably losing or distorting information. This compression leads to *hallucination*: the model generates plausible but factually incorrect outputs because it has learned the distributional statistics of correct answers without retaining ground-truth records.

Tools provide **grounded, verifiable, real-time access** to information: database queries return actual records, web searches return current pages, and file reads return true contents.

**Limitation 2: Models Cannot Perform Reliable Precision Computation.**
Transformer-based language models perform computation through learned approximate functions encoded in their weight matrices. For precision tasks — multi-digit arithmetic, symbolic algebra, formal logic — this approximation is unreliable:

$$
\mathcal{M}_\theta(\text{"What is } 7823 \times 4519\text{?"}) \neq 35,354,137 \quad \text{(with non-trivial probability)}
$$

A calculator tool returns the exact answer with probability 1. The model's role shifts from *computing* to *deciding when to compute externally*.

**Limitation 3: Models Cannot Act on the World.**
Token generation is a purely informational operation — it produces sequences of symbols. To effect changes in the world (send an email, update a database record, deploy code, control a robot), the model must interface with external systems through tools. Without tools:

$$
\mathcal{Y}_{\text{model}} \subseteq \Sigma^* \quad \text{(model can only produce text)}
$$

With tools:

$$
\mathcal{Y}_{\text{model+tools}} \subseteq \Sigma^* \cup \{\text{side effects via } \mathcal{T}\}
$$

**Limitation 4: Grounding in Real-World State.**
Effective assistance requires awareness of current state — the user's file system, database contents, external API responses, real-time sensor data. This state is not in the model's parameters and can only be accessed through tool invocation.

**Limitation 5: Real-Time Information Access.**
Stock prices, weather, news, social media, live sensor feeds — all require real-time retrieval that parametric models fundamentally cannot provide.

---

### 2.1.4 The Model–Tool–Agent Triad: Interaction Dynamics

**Information Flow.**
The canonical information flow in a tool-augmented system follows a well-defined cycle:

$$
\text{User Query} \xrightarrow{\text{input}} \mathcal{M} \xrightarrow{\text{tool\_call}(\tau_i, \text{args}_i)} \tau_i \xrightarrow{\text{result } r_i} \mathcal{M} \xrightarrow{\text{response}} \text{User}
$$

In multi-step interactions, this becomes an iterative loop:

$$
\text{User} \rightarrow \mathcal{M} \rightarrow \tau_{i_1} \rightarrow \mathcal{M} \rightarrow \tau_{i_2} \rightarrow \mathcal{M} \rightarrow \cdots \rightarrow \tau_{i_k} \rightarrow \mathcal{M} \rightarrow \text{User}
$$

At each step, the model incorporates the tool result into its context and decides the next action — another tool call, a final response, or further reasoning.

**Multi-Step Interaction Loops and Recursive Tool Invocation.**
Complex tasks require multiple tool invocations with intermediate reasoning:

1. **Linear chains**: $\tau_1 \rightarrow \tau_2 \rightarrow \tau_3$ (output of each feeds the next)
2. **Branching**: Based on $\tau_1$'s result, choose between $\tau_2$ or $\tau_3$
3. **Iterative refinement**: Call $\tau_i$ repeatedly until a quality criterion is met
4. **Recursive composition**: A tool's implementation itself invokes the model, which may call further tools (agent-within-agent patterns)

**The Role of the Context Window as Shared Memory.**
The context window serves as the **shared memory** between the model and its tools. Every tool invocation and result is appended to the conversation history, which functions as a working memory:

$$
W_{\text{total}} = W_{\text{system}} + W_{\text{tool\_defs}} + \sum_{t=1}^{T} \left( W_{\text{call}_t} + W_{\text{result}_t} \right) + W_{\text{generation}}
$$

where $W_x$ denotes the token count of component $x$, and the total must not exceed the model's context limit $W_{\max}$. This imposes a hard constraint on the number of tool interactions per session and the verbosity of tool outputs.

**Feedback Loops and Iterative Refinement.**
Tool results provide feedback that the model uses to:

- **Verify hypotheses**: "The search returned no results for X, so X may not exist."
- **Correct errors**: "The code execution returned an error; let me fix the syntax."
- **Refine queries**: "The initial search was too broad; let me add more specific terms."
- **Accumulate state**: "I now have data from tools A and B; let me combine them."

This creates a closed-loop control system:

$$
\text{Error}_t = \text{Goal} - \text{Current\_State}(o_t)
$$

$$
a_{t+1} = \mathcal{M}(\text{Error}_t, \; \mathcal{H}_{\leq t}, \; \mathcal{T})
$$

The model acts as a feedback controller, adjusting its actions based on the discrepancy between the desired outcome and the observed tool results.

---

## 2.2 Tools and Tool Calling

### 2.2.1 What Do We Mean by a Tool?

#### 2.2.1.1 Formal Definition

A tool, in the context of LLM-augmented systems, is a **callable interface** exposed to the language model through a structured description, enabling the model to invoke external computations or side effects that lie outside its intrinsic generative capabilities.

**Mathematical Representation.** A tool is a typed function with a formal schema:

$$
\tau : \mathcal{X} \rightarrow \mathcal{Y}, \quad \text{where } \mathcal{X} \subseteq \text{JSON Schema}, \; \mathcal{Y} \subseteq \text{Structured Output}
$$

More precisely, the input domain $\mathcal{X}$ is the set of all JSON objects that validate against the tool's input schema $\sigma_{in}$, and the output codomain $\mathcal{Y}$ is the set of all possible return values conforming to $\sigma_{out}$.

**Tool Signature Specification.** Every tool exposed to a model must provide a complete signature:

```
Tool Signature := {
    name:        String          // unique identifier, e.g., "get_stock_price"
    description: String          // natural language: what, when, when-not
    parameters:  JSON Schema     // input schema (Draft 2020-12)
    returns:     JSON Schema     // output schema (optional in some APIs)
    strict:      Boolean         // whether to enforce exact schema adherence
}
```

The JSON Schema for parameters follows the Draft 2020-12 specification, supporting:

- Primitive types: `string`, `number`, `integer`, `boolean`, `null`
- Composite types: `object`, `array`
- Constraints: `required`, `enum`, `minimum`, `maximum`, `pattern`, `format`, `minItems`, `maxItems`, `default`
- Nested schemas: `properties`, `items`, `additionalProperties`, `$ref`

**Example — Full Tool Specification:**

```json
{
  "name": "query_database",
  "description": "Executes a read-only SQL query against the application database and returns the result set. Use this when the user asks about data stored in the database (users, orders, products). Do NOT use for write operations (INSERT, UPDATE, DELETE). Maximum 1000 rows returned.",
  "parameters": {
    "type": "object",
    "properties": {
      "sql_query": {
        "type": "string",
        "description": "A valid SELECT SQL query. Must not contain DML statements."
      },
      "database": {
        "type": "string",
        "enum": ["production_readonly", "analytics"],
        "description": "Which database to query."
      },
      "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000,
        "default": 100,
        "description": "Maximum number of rows to return."
      }
    },
    "required": ["sql_query", "database"],
    "additionalProperties": false
  }
}
```

**Tools as Typed Functions with Contracts.** Borrowing from design-by-contract (Meyer, 1988), each tool implicitly or explicitly carries:

- **Preconditions**: constraints that must hold on the input for the tool to execute correctly (encoded in $\sigma_{in}$, plus runtime checks).
- **Postconditions**: guarantees about the output structure and properties (encoded in $\sigma_{out}$).
- **Invariants**: properties that hold across all invocations (e.g., a read-only tool never modifies state).

$$
\text{Contract}(\tau) = \left\{
\begin{aligned}
&\text{Pre}: \; \forall x \in \mathcal{X}, \; \text{validate}(x, \sigma_{in}) = \text{true} \\
&\text{Post}: \; \forall x \in \mathcal{X}, \; \text{validate}(f(x), \sigma_{out}) = \text{true} \\
&\text{Inv}: \; \text{side\_effect\_class}(\tau) \in \{\text{pure}, \text{read}, \text{write}\}
\end{aligned}
\right\}
$$

---

#### 2.2.1.2 Tool vs. Function vs. API vs. Skill

These terms are often used interchangeably but denote distinct abstraction levels:

**Function.** A function is a code-level callable in a specific programming language. It is the lowest-level unit of computation:

```python
def celsius_to_fahrenheit(temp_c: float) -> float:
    return temp_c * 9/5 + 32
```

Functions are language-specific, have no natural-language description, and are not directly accessible to the model.

**Tool.** A tool is a **model-facing abstraction** over one or more functions. It wraps a function (or sequence of functions) with:

- A natural-language description
- A formal input/output schema
- A unique name

The model interacts with tools, not functions. A single tool may internally invoke multiple functions, handle error cases, perform input normalization, and format outputs.

**API (Application Programming Interface).** An API is a network-accessible service endpoint, typically accessed via HTTP. A tool *may* wrap an API, but the concepts are not equivalent:

- A tool might call multiple APIs in sequence.
- A tool might not call any API (e.g., a local computation tool).
- An API may expose dozens of endpoints, each potentially mapped to a different tool.

**Skill.** A skill is a higher-order composition of multiple tools with an associated strategy. Skills are emergent behaviors that arise from the model's ability to plan and orchestrate tool sequences:

- Skill: "Research a topic and write a summary" = search tool + read tool + summarization (model reasoning)
- Skill: "Debug a code error" = code execution + error analysis (reasoning) + code modification + re-execution

The layered abstraction:

$$
\text{Skill} \supset \text{Tool} \supset \text{Function} \supset \text{API Call}
$$

| Abstraction | Visible to Model? | Has Schema? | Has NL Description? | Composable? |
|---|---|---|---|---|
| API Call | No | Yes (OpenAPI) | Partially | At API level |
| Function | No | Yes (type hints) | No | At code level |
| Tool | **Yes** | **Yes (JSON Schema)** | **Yes** | By model reasoning |
| Skill | Implicitly | No | Implicitly (in prompt) | By model planning |

---

#### 2.2.1.3 Tool as a Contract

**Input Validation: Type Checking, Range Constraints, Required vs. Optional Parameters.**
Before executing a tool, the runtime validates the model-generated arguments against $\sigma_{in}$:

1. **Type checking**: Is `"temperature"` a number? Is `"city"` a string?
2. **Range constraints**: Is $0 \leq \text{limit} \leq 1000$?
3. **Required fields**: Are all `required` parameters present?
4. **Enum enforcement**: Is `"unit"` one of `["celsius", "fahrenheit"]`?
5. **Pattern matching**: Does the email string match `^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$`?

Failed validation produces an error message that is fed back to the model, enabling self-correction.

**Output Guarantees: Deterministic vs. Stochastic Tools.**

- A **deterministic** tool guarantees: $\forall x, \; f(x) = f(x)$ across all invocations.
  - Examples: calculator, hash function, unit converter.
  - Properties: cacheable, reproducible, idempotent.
  
- A **stochastic** tool may return different results for the same input:
  - Examples: web search (results change over time), model-backed tools (sampling temperature > 0), random number generator.
  - Properties: not cacheable, not reproducible in general, requires the model to handle variability.

**Side-Effect Classification.**

| Category | Description | Examples | Safety Implication |
|---|---|---|---|
| **Pure (read-only)** | No observable state change | Calculator, search, weather lookup | Low risk, freely retryable |
| **Impure (write/mutate)** | Modifies external state | Send email, write file, update DB | Higher risk, may need confirmation |
| **Irreversible** | State change cannot be undone | Delete record, send payment | Highest risk, requires approval gate |

**Idempotency Properties and Retry Semantics.**
An idempotent tool satisfies:

$$
f(x) = f(f(x)) \quad \text{or more precisely} \quad \text{effect}(f(x)) = \text{effect}(f(x); f(x))
$$

Idempotent tools are safe to retry on failure (e.g., `PUT` operations, status checks). Non-idempotent tools (e.g., `POST` operations, payment processing) require at-most-once semantics and deduplication mechanisms.

---

#### 2.2.1.4 The Anatomy of a Tool Call

The complete lifecycle of a tool call consists of three phases:

**Phase 1: Request Phase (Model → Runtime).**
The model generates a structured tool call as part of its output. This involves the model autoregressively producing tokens that conform to the tool-call format:

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"London\", \"unit\": \"celsius\"}"
  }
}
```

Key technical aspects:
- The model generates this JSON token-by-token, with each token conditioned on all previous tokens and the full context.
- **Constrained decoding** may be employed to guarantee valid JSON and schema compliance (see §2.2.5.1).
- The model may generate **multiple** tool calls in a single turn (parallel invocation).

**Phase 2: Execution Phase (Runtime → Tool).**
The runtime:
1. Parses the model's output to extract tool call(s).
2. Validates arguments against $\sigma_{in}$.
3. Dispatches the call to the tool implementation $f$.
4. Handles execution (local function call, HTTP request, subprocess, etc.).
5. Captures the result (and any errors).

**Phase 3: Response Phase (Tool → Model).**
The tool result is marshalled back into the model's context as a new message:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 18, \"condition\": \"Partly cloudy\", \"humidity\": 72}"
}
```

The model then continues generation, conditioning on the original context plus the tool result.

**Formal Call-Response Cycle.**

At step $t$, the model generates zero or more tool calls:

$$
c_t = \mathcal{M}(\text{prompt}, \; \mathcal{H}_{<t}) \in \{(\tau_i, \; \text{args}_i)\}_{i=1}^{k}
$$

Each tool call is executed:

$$
r_t = \bigcup_{i=1}^{k} \tau_i(\text{args}_i)
$$

The model then generates the next turn, conditioning on the accumulated context:

$$
\text{next\_turn} = \mathcal{M}(\text{prompt}, \; \mathcal{H}_{<t}, \; c_t, \; r_t)
$$

This next turn may itself contain further tool calls (multi-step reasoning), or it may be a final text response to the user. The recursion continues until the model decides no further tool calls are needed.

---

### 2.2.2 Types of Tools

#### 2.2.2.1 Classification by Capability Domain

| Category | Examples | Properties | Input Characteristics | Output Characteristics |
|---|---|---|---|---|
| **Information Retrieval** | Web search, database query, document retrieval, knowledge graph lookup | Read-only, latency-sensitive, stochastic (results may vary) | Query string, filters, pagination params | Ranked list of results, snippets, metadata |
| **Computation** | Calculator, symbolic math (SymPy), unit converter, statistical functions | Deterministic, side-effect-free, exact | Mathematical expression or parameters | Exact numerical/symbolic result |
| **Code Execution** | Sandboxed Python, JavaScript, shell commands, Jupyter kernels | Potentially stateful, security-critical, may have side effects within sandbox | Code string, language specification | stdout, stderr, generated files, rendered outputs |
| **Data Manipulation** | File read/write, CSV/Excel processing, spreadsheet operations, data transforms | Stateful, idempotency concerns, may modify persistent storage | File path, operation type, data payload | Modified data, confirmation, new file references |
| **Communication** | Email sending, Slack/Teams messaging, webhook triggers, SMS | Side-effecting, often irreversible, requires authentication | Recipient, message content, attachments | Delivery status, message ID |
| **Perception** | Image analysis (captioning, OCR), audio transcription (Whisper), video analysis | Model-backed, probabilistic output, compute-intensive | Binary data (image, audio, video) or URL | Structured perception results (text, bounding boxes, transcripts) |
| **Actuation** | Robotic control commands, IoT device management, CI/CD triggers, deployment | Physical side effects, safety-critical, latency-sensitive | Action specification, target device/system | Execution status, sensor feedback |
| **Memory & State** | Vector store read/write, knowledge graph update, session state management | Persistence concerns, consistency guarantees, concurrent access | Key-value pairs, embeddings, graph triples | Stored/retrieved state, confirmation |

Each category imposes different requirements on the agent's reasoning:

- **Retrieval** tools require the model to formulate effective queries and synthesize multiple results.
- **Computation** tools require correct expression formulation but provide guaranteed-correct results.
- **Execution** tools require the model to generate syntactically valid code and handle runtime errors.
- **Communication** tools require the model to exercise caution about irreversible actions.
- **Perception** tools require the model to interpret probabilistic outputs with appropriate uncertainty.

---

#### 2.2.2.2 Classification by Invocation Pattern

**Synchronous Tools.** The model emits a tool call, and the runtime blocks until the tool returns a result. The model cannot proceed until it receives the response. This is the default and most common pattern:

$$
\text{Model} \xrightarrow{\text{call}} \tau_i \xrightarrow[\text{blocks}]{\text{wait}} \text{result} \xrightarrow{} \text{Model continues}
$$

**Asynchronous Tools.** The tool call is dispatched, and the result arrives later via callback, polling, or event:

$$
\text{Model} \xrightarrow{\text{call}} \tau_i \xrightarrow{\text{returns immediately with task\_id}} \text{Model can proceed}
$$
$$
\text{Later}: \; \text{poll}(\text{task\_id}) \rightarrow \text{result (when ready)}
$$

Use cases: long-running computations (ML training jobs, large data processing), external approval workflows (human review).

**Streaming Tools.** Results are delivered progressively as they become available:

$$
\tau_i(x) \rightarrow \{r_1, r_2, \ldots, r_n\} \quad \text{(partial results over time)}
$$

Use cases: code execution with incremental output, real-time data feeds, long-running searches.

**Parallel Tool Calls.** When multiple tool calls are independent — no data dependencies between them — they can be executed simultaneously:

$$
\text{Parallel}: \; \{(\tau_1, \text{args}_1), (\tau_2, \text{args}_2), (\tau_3, \text{args}_3)\} \xrightarrow{\text{concurrent}} \{r_1, r_2, r_3\}
$$

**Dependency Graph Analysis.** To determine which tool calls can be parallelized, construct a dependency graph $G = (V, E)$ where:

- $V = \{c_1, c_2, \ldots, c_k\}$ is the set of tool calls in the current plan.
- $E = \{(c_i, c_j) \mid c_j \text{ depends on the output of } c_i\}$.

The parallelism condition:

$$
\tau_i \parallel \tau_j \iff \nexists \; \text{path } \tau_i \rightsquigarrow \tau_j \text{ in } G \; \land \; \nexists \; \text{path } \tau_j \rightsquigarrow \tau_i \text{ in } G
$$

In other words, two tool calls can be parallelized if and only if neither depends (directly or transitively) on the other's output.

**Sequential (Chained) Tool Calls.** When the output of one tool is required as input to the next, execution must be sequential:

$$
\text{Chain}: \; y = \tau_n(\tau_{n-1}(\cdots \tau_2(\tau_1(x)) \cdots))
$$

More generally, with intermediate model reasoning between each step:

$$
y_1 = \tau_1(x) \;\rightarrow\; z_1 = \mathcal{M}(y_1) \;\rightarrow\; y_2 = \tau_2(z_1) \;\rightarrow\; z_2 = \mathcal{M}(y_2) \;\rightarrow\; \cdots
$$

The model serves as an **interstitial reasoner** between tool calls, interpreting results, formulating next queries, and handling errors.

---

#### 2.2.2.3 Classification by Trust Level

**Trusted / Verified Tools.**
- Provided by the platform or system developer.
- Source code is audited, tested, and controlled.
- Execution environment is managed.
- Examples: built-in code interpreter, platform-provided search.

**Semi-Trusted Tools.**
- Third-party tools that have undergone some vetting.
- Executed in sandboxed environments with limited permissions.
- May have rate limits, usage monitoring.
- Examples: tools from a curated marketplace with review process.

**Untrusted Tools.**
- User-provided or dynamically discovered tools.
- No guarantees about correctness, safety, or honesty.
- Must be executed in strict isolation with minimal permissions.
- Potential attack vectors: tools that return adversarial content to manipulate the model (indirect prompt injection), tools that exfiltrate data.
- Examples: user-uploaded tool definitions, tools discovered from unvetted registries.

**Trust Propagation in Tool Composition.**
When tools are composed (the output of one feeds another), the trust level of the composed pipeline is bounded by the least trusted component:

$$
\text{trust}(\tau_1 \circ \tau_2 \circ \cdots \circ \tau_n) = \min_{i \in \{1, \ldots, n\}} \text{trust}(\tau_i)
$$

This follows the weakest-link principle: a trusted tool that processes output from an untrusted tool inherits the untrusted tool's risk level, because the untrusted tool's output may contain adversarial content.

---

#### 2.2.2.4 Classification by Determinism

| Property | Deterministic | Stochastic | Stateful |
|---|---|---|---|
| **Definition** | $f(x) = f(x)$ always | $f(x)$ may vary across calls | $f(x)$ depends on hidden state $s$: $f(x, s)$ |
| **Examples** | Calculator, hash, unit conversion | Web search, LLM-backed tools, random sampling | Database query (DB contents change), file read (file may change) |
| **Caching** | Safe to cache indefinitely | Cache with TTL or invalidation | Cache with state-awareness |
| **Reproducibility** | Fully reproducible | Not reproducible without seed/snapshot | Reproducible only with state snapshot |
| **Retry safety** | Safe, idempotent | May get different (not wrong) results | May get different results if state changed |

The determinism classification has direct implications for:

- **Agent reasoning**: The model should treat stochastic tool results as evidence with uncertainty, not ground truth.
- **Caching strategies**: Deterministic tool results can be cached aggressively to save compute and latency. Stochastic results require cache invalidation policies.
- **Debugging and reproducibility**: Deterministic tools enable reproducible agent traces. Stochastic tools require logging of actual results for post-hoc analysis.

---

### 2.2.3 Built-in Tools

#### 2.2.3.1 Definition and Scope

Built-in tools are tools that are **natively provided by the model provider or platform**, pre-integrated with the model's inference infrastructure. They differ from user-defined tools in several important ways:

| Property | Built-in Tools | User-Defined Tools |
|---|---|---|
| Provider | Platform (OpenAI, Anthropic, Google, etc.) | Developer |
| Integration | Native, optimized, pre-tested | Via API schema registration |
| Trust Level | Trusted (audited by provider) | Variable |
| Configuration | Enable/disable via API parameter | Full definition provided by developer |
| Schema | Often implicit / internal | Explicit JSON Schema |
| Execution | Server-side (provider infrastructure) | Client-side or developer-managed server |

Common built-in tools across major platforms:

- **Code Interpreter** (OpenAI Assistants, Anthropic Claude with tool use)
- **File Search / Retrieval** (OpenAI Assistants file_search, Google Vertex Search)
- **Web Browsing / Search** (ChatGPT browsing, Gemini Google Search)
- **Image Generation** (DALL·E integration)

---

#### 2.2.3.2 Code Interpreter / Code Execution

The code interpreter is arguably the most powerful built-in tool, as it transforms the language model from a text generator into a **computational agent** capable of executing arbitrary programs.

**Architecture.** The code interpreter consists of:

1. **Sandbox Environment**: An isolated execution runtime (typically a Jupyter kernel or equivalent) with:
   - Restricted system calls (no network access, limited filesystem)
   - Resource limits (CPU time, memory, disk)
   - No persistence across sessions (typically)
   - Pre-installed libraries (numpy, pandas, matplotlib, scipy, sympy, etc.)

2. **State Persistence Within a Session**: Unlike stateless tool calls, the code interpreter maintains state across multiple invocations within the same conversation:

$$
\text{State}_t = \text{State}_{t-1} \cup \Delta(\text{code}_t)
$$

Variables, imported libraries, and data loaded in earlier code blocks remain available in subsequent blocks. This enables iterative, exploratory computation.

3. **Input/Output Interface**:
   - Input: Code string (Python, typically)
   - Output: `stdout`, `stderr`, generated files (images, CSVs), rendered visualizations

**Security Model.** The sandbox enforces:

$$
\text{Permissions} = \{\text{read\_uploaded\_files}, \text{write\_temp\_files}, \text{execute\_code}\} \setminus \{\text{network\_access}, \text{persistent\_storage}, \text{system\_calls}\}
$$

This isolation is critical: user-facing models may generate and execute code based on user requests, and the sandbox prevents code execution from compromising the host system.

**Use Cases:**

| Use Case | Example | Why Code Interpreter > Pure LLM |
|---|---|---|
| Data analysis | Analyzing a CSV, computing statistics, generating plots | Exact computation, handles large datasets |
| Mathematical computation | Solving equations, numerical integration, matrix operations | Guaranteed correctness, arbitrary precision |
| Algorithm prototyping | Implementing and testing a sorting algorithm | Verifiable execution, actual runtime |
| File transformation | Converting data formats, image processing | Real I/O operations |
| Visualization | Creating charts, plots, diagrams | Actual rendered images |

---

#### 2.2.3.3 File Search and Retrieval (RAG-as-a-Tool)

File search integrates **Retrieval-Augmented Generation (RAG)** as a first-class tool, enabling the model to search through user-uploaded documents to find relevant information.

**Architecture Components:**

1. **Document Ingestion Pipeline**:
   - Document upload and parsing (PDF, DOCX, TXT, MD, etc.)
   - Text extraction (including OCR for scanned documents)
   - Chunking: splitting documents into retrievable units

2. **Chunking Strategies:**

| Strategy | Method | Pros | Cons |
|---|---|---|---|
| Fixed-size | Split every $k$ tokens | Simple, predictable | May split mid-sentence/concept |
| Semantic | Split at paragraph/section boundaries | Preserves meaning | Variable chunk sizes |
| Recursive | Split by structure (headers → paragraphs → sentences) | Adapts to document structure | More complex implementation |
| Document-structure-aware | Use document hierarchy (sections, subsections) | Best semantic coherence | Requires parsed structure |

3. **Embedding and Indexing.** Each chunk $d_i$ is embedded into a dense vector:

$$
\mathbf{e}_{d_i} = \text{Embed}(d_i) \in \mathbb{R}^{d}
$$

These vectors are stored in a vector index (e.g., FAISS, HNSW, Pinecone, Weaviate).

4. **Retrieval.** Given a query $q$, the retrieval process:

$$
\text{sim}(q, d_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{d_i}}{\|\mathbf{e}_q\| \|\mathbf{e}_{d_i}\|} = \cos(\mathbf{e}_q, \mathbf{e}_{d_i})
$$

Top-$k$ chunks by similarity are retrieved:

$$
\text{Retrieved} = \text{top-}k_{d_i \in \mathcal{D}} \; \text{sim}(q, d_i)
$$

5. **Hybrid Retrieval.** Combining dense (embedding-based) and sparse (lexical) retrieval:

- **Sparse retrieval (BM25)**:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot |d| / \text{avgdl})}
$$

- **Fusion**: Reciprocal Rank Fusion (RRF):

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
$$

where $k$ is a constant (typically 60).

6. **Post-Retrieval Processing:**
   - **Re-ranking**: A cross-encoder scores (query, document) pairs jointly for more accurate relevance.
   - **Contextual compression**: Extracting only the relevant portions of retrieved chunks.
   - **Citation tracking**: Each piece of retrieved information carries provenance metadata (source document, page number, chunk ID).

---

#### 2.2.3.4 Web Browsing / Web Search

Web browsing tools provide the model with real-time access to internet content, overcoming the knowledge cutoff limitation.

**Capabilities:**

1. **Search Query Formulation**: The model generates search queries optimized for the search engine (keyword-based, not conversational).
2. **Result Processing**: Search results (title, snippet, URL) are parsed and presented to the model.
3. **Page Rendering**: For detailed information, the tool fetches and renders web pages, extracting text content.
4. **Content Extraction**: Stripping navigation, ads, and boilerplate to extract the main content.
5. **Multi-Hop Navigation**: Following links from one page to another to gather comprehensive information.
6. **Pagination**: Browsing through multiple pages of search results.

**Technical Challenges:**

- **Dynamic content**: Many modern web pages require JavaScript execution to render content. The browsing tool must use a headless browser (Puppeteer, Playwright) to render such pages.
- **Rate limiting and CAPTCHAs**: Web services may block automated access.
- **Content quality**: Web content varies enormously in reliability, requiring the model to assess source credibility.
- **Token budget**: Web pages can be very large; content extraction and summarization are essential to fit within context limits.

---

#### 2.2.3.5 Image Generation and Vision Tools

**Text-to-Image Generation.**
The model can invoke image generation tools (DALL·E, Stable Diffusion) to create images from textual descriptions:

$$
\text{generate\_image}: \text{String (prompt)} \rightarrow \text{Image (binary/URL)}
$$

The model's role is to craft an effective prompt for the image generator, which may involve:
- Translating user intent into detailed visual descriptions.
- Specifying style, composition, lighting, and other artistic parameters.
- Handling iterative refinement based on user feedback.

**Vision / Image Analysis Tools.**
Vision-language models can be invoked as tools to analyze images:

- **Image captioning**: generating descriptions of image content.
- **OCR**: extracting text from images of documents, signs, screenshots.
- **Diagram parsing**: interpreting flowcharts, architecture diagrams, UML.
- **Chart data extraction**: reading values from bar charts, line graphs, pie charts.
- **Object detection**: identifying and locating objects within images.

---

#### 2.2.3.6 Built-in Tool Selection and Routing

When multiple built-in tools are enabled, the model must decide which tool(s) to invoke for a given query:

**Automatic Tool Selection.** The model implicitly decides based on the query semantics:
- "What's the weather in Tokyo?" → Web search
- "Analyze this CSV file" → Code interpreter
- "Find information in the uploaded report" → File search
- "Create an image of a sunset" → Image generation

**Configuration Options:**
- Enable/disable specific built-in tools per session or per request.
- Set priority/preference ordering among tools.
- Define fallback mechanisms: if tool A fails, try tool B.

**Routing Logic.** Internally, the model's tool selection can be modeled as:

$$
\tau^* = \arg\max_{\tau \in \mathcal{T}_{\text{enabled}}} P(\tau \mid q, \mathcal{H}, \text{tool\_descriptions})
$$

The model selects the tool with the highest probability of being useful for the current query, given the full context.

---

### 2.2.4 Agent Tools (Custom / User-Defined Tools)

#### 2.2.4.1 Definition and Registration

Developer-defined tools are custom functions exposed to the model through the tool-use API. The developer provides:

1. **Tool definition**: Name, description, and parameter schema (JSON Schema).
2. **Tool implementation**: The actual code that executes when the tool is called.

The model only sees the definition; the implementation is opaque.

**Registration via API.** Tool definitions are passed as part of the API request:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price for a given ticker symbol. Returns the latest price in USD, along with the daily change. Use when the user asks about stock prices or market data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')"
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Whether to include 30-day price history",
                        "default": False
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False
            }
        }
    }
]
```

**Runtime Binding.** The developer maintains a mapping from tool names to executable implementations:

```python
tool_implementations = {
    "get_stock_price": get_stock_price_function,
    "send_email": send_email_function,
    "query_database": query_database_function,
}
```

When the model generates a tool call with `name: "get_stock_price"`, the runtime dispatches to `get_stock_price_function`.

---

#### 2.2.4.2 Tool Definition Schema

**JSON Schema Specification for Input Parameters.**

The parameter schema is the contract between the model and the tool. It must be precise enough for the model to generate valid arguments, and the schema serves as both documentation and validation:

**Supported Types and Constraints:**

| Type | JSON Schema | Example | Model Behavior |
|---|---|---|---|
| `string` | `{"type": "string"}` | `"hello"` | Free text generation |
| `string` (enum) | `{"type": "string", "enum": ["a", "b"]}` | `"a"` | Constrained to enumerated values |
| `string` (pattern) | `{"type": "string", "pattern": "^\\d{3}-\\d{4}$"}` | `"555-1234"` | Must match regex |
| `number` | `{"type": "number", "minimum": 0, "maximum": 100}` | `42.5` | Floating-point within range |
| `integer` | `{"type": "integer"}` | `42` | Integer only |
| `boolean` | `{"type": "boolean"}` | `true` | Binary choice |
| `array` | `{"type": "array", "items": {"type": "string"}}` | `["a", "b"]` | List of typed elements |
| `object` | `{"type": "object", "properties": {...}}` | `{"key": "val"}` | Nested structured data |

**Nested Objects and Complex Schemas.** Tools can accept deeply structured inputs:

```json
{
  "type": "object",
  "properties": {
    "filters": {
      "type": "object",
      "properties": {
        "date_range": {
          "type": "object",
          "properties": {
            "start": {"type": "string", "format": "date"},
            "end": {"type": "string", "format": "date"}
          },
          "required": ["start", "end"]
        },
        "categories": {
          "type": "array",
          "items": {"type": "string", "enum": ["A", "B", "C"]}
        }
      }
    }
  }
}
```

**Tool Description as Natural Language Documentation.** The `description` field is the **most critical** component for model accuracy (see §2.3.1). It should clearly state:
- What the tool does.
- When to use it (positive guidance).
- When NOT to use it (negative guidance).
- Any important constraints or prerequisites.

**Return Type Specification.** While not universally supported in current APIs, specifying the return type helps the model anticipate the structure of results:

```json
{
  "returns": {
    "type": "object",
    "properties": {
      "price": {"type": "number"},
      "currency": {"type": "string"},
      "timestamp": {"type": "string", "format": "date-time"}
    }
  }
}
```

---

#### 2.2.4.3 Tool Implementation Patterns

**Pattern 1: Local Function Execution.**
The tool runs in the same process as the agent:

```python
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    bmi = weight_kg / (height_m ** 2)
    return {"bmi": round(bmi, 1), "category": classify_bmi(bmi)}
```

- **Pros**: Low latency, no network overhead, simple debugging.
- **Cons**: Limited isolation, shared failure domain, potential for tool crashes to bring down the agent.

**Pattern 2: Remote API Proxy.**
The tool forwards the call to an external HTTP endpoint:

```python
def get_weather(city: str, unit: str = "celsius") -> dict:
    response = requests.get(f"https://api.weather.com/v3/forecast",
                           params={"city": city, "unit": unit},
                           headers={"Authorization": f"Bearer {API_KEY}"})
    return response.json()
```

- **Pros**: Access to external services, separation of concerns.
- **Cons**: Network latency, authentication management, external service reliability.

**Pattern 3: Subprocess Execution.**
The tool spawns a child process, typically wrapping CLI tools:

```python
def run_linter(code: str, language: str) -> dict:
    result = subprocess.run(["pylint", "--from-stdin", "-"],
                           input=code, capture_output=True, text=True, timeout=30)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
```

- **Pros**: Leverages existing CLI tools, process isolation.
- **Cons**: Process overhead, security concerns (command injection), platform dependencies.

**Pattern 4: Container-Based Execution.**
For untrusted or complex tools, execution occurs in isolated containers (Docker, Wasm):

```python
def execute_untrusted_code(code: str) -> dict:
    container = docker.run("sandbox:latest", command=f"python -c '{code}'",
                          mem_limit="256m", cpu_period=100000, cpu_quota=50000,
                          network_disabled=True, timeout=30)
    return {"output": container.logs(), "exit_code": container.exit_code}
```

- **Pros**: Strong isolation, resource limits, security.
- **Cons**: Higher latency (container startup), infrastructure complexity.

**Pattern 5: Model-as-a-Tool.**
Another LLM or ML model is invoked as a tool:

```python
def classify_sentiment(text: str) -> dict:
    result = sentiment_model.predict(text)  # a specialized classifier
    return {"sentiment": result.label, "confidence": result.score}
```

This pattern is common in multi-model architectures where a general-purpose LLM orchestrates specialized models (classifiers, embedders, image models).

---

#### 2.2.4.4 Dynamic Tool Registration and Discovery

**Static vs. Dynamic Tool Sets.**

- **Static**: The tool set $\mathcal{T}$ is fixed at system design time. All tools are known in advance and included in every API call. Simple but inflexible.

- **Dynamic**: Tools are added, removed, or modified at runtime based on:
  - The current task context (load math tools for math queries, load email tools for communication tasks).
  - The user's permissions (different users have access to different tools).
  - The task phase (planning phase uses search tools; execution phase uses action tools).

**Dynamic Registration Patterns:**

1. **Context-dependent loading**: Analyze the user query and load only relevant tools:

$$
\mathcal{T}_{\text{active}} = \text{select}(\mathcal{T}_{\text{all}}, \; q, \; \text{context})
$$

2. **Progressive disclosure**: Start with a small core tool set; add specialized tools as the conversation evolves.

3. **Tool discovery protocols**: The agent queries a registry or marketplace to find tools matching a need:

$$
\mathcal{T}_{\text{discovered}} = \text{Registry.search}(\text{capability\_description})
$$

**Implications for Context Window Management.** Each tool definition consumes tokens in the system prompt. With $N$ tools, each requiring $L$ tokens:

$$
W_{\text{tools}} = N \times L
$$

For large tool sets ($N > 50$), this can consume a significant portion of the context window. Dynamic loading mitigates this by keeping $N$ small for any given request.

---

#### 2.2.4.5 Multi-Agent Tool Sharing

In multi-agent systems, multiple agents may need to access the same tools. This introduces coordination challenges:

**Tool Access Control.** Not all agents should have access to all tools. An access control matrix defines permissions:

$$
\text{ACL}: \mathcal{A} \times \mathcal{T} \rightarrow \{\text{allow}, \text{deny}\}
$$

Agent $\mathcal{A}_i$ can invoke tool $\tau_j$ only if $\text{ACL}(\mathcal{A}_i, \tau_j) = \text{allow}$.

**Concurrent Tool Access.** When multiple agents invoke the same stateful tool concurrently:

- **Locking**: Exclusive access to prevent race conditions (pessimistic concurrency).
- **Queuing**: Serializing tool access through a message queue.
- **Rate limiting**: Preventing any single agent from monopolizing a shared tool.
- **Optimistic concurrency**: Allow concurrent access, detect conflicts, and retry.

**Tool State Isolation.** Each agent should have its own view of stateful tools (e.g., separate file system namespaces, separate database schemas) to prevent unintended interference:

$$
\text{State}(\tau, \mathcal{A}_i) \cap \text{State}(\tau, \mathcal{A}_j) = \emptyset \quad \forall i \neq j
$$

---

### 2.2.5 The Mechanics of Tool Calling in LLMs

#### 2.2.5.1 How Models Generate Tool Calls

**Special Token / Structured Output Mode.** Modern LLMs generate tool calls through one of two mechanisms:

1. **Special tokens**: The model has been trained with special tokens (e.g., `<tool_call>`, `</tool_call>`) that delimit tool invocations in the output. When the model generates these tokens, the runtime intercepts them and dispatches the tool call.

2. **Structured output mode**: The API enforces that the model's output conforms to a specific JSON structure for tool calls. The model's output is parsed as structured data rather than free text.

**Internal Representation of Tool Schemas.** Tool definitions are injected into the model's context — typically as part of the system prompt or a dedicated tool-definition section. The model "sees" each tool as text:

```
You have access to the following tools:

Function: get_weather
Description: Retrieves weather forecast for a given city...
Parameters:
  - city (string, required): The city name
  - unit (string, optional): "celsius" or "fahrenheit", default "celsius"
```

The model must parse these descriptions, understand the tool's purpose, determine when to invoke it, and generate correctly-formatted arguments — all through its standard autoregressive generation process.

**Autoregressive Generation of Tool Call JSON.** The tool call is generated token-by-token:

$$
P(\text{tool\_call} \mid \text{context}) = \prod_{i=1}^{n} P(t_i \mid t_1, t_2, \ldots, t_{i-1}, \text{context})
$$

where $t_i$ are the individual tokens of the JSON tool call string. At each step, the model's next-token distribution is conditioned on all previous tokens, including the tool descriptions and conversation history.

**Constrained Decoding.** A critical challenge is ensuring the generated JSON is valid and conforms to the tool's schema. Unconstrained generation can produce malformed JSON, missing required fields, or type errors. **Constrained decoding** restricts the model's output distribution at each step to only tokens that maintain valid JSON/schema conformance:

$$
P_{\text{constrained}}(t_i \mid t_{<i}) = \frac{P_{\text{model}}(t_i \mid t_{<i}) \cdot \mathbb{1}[t_i \in \text{Valid}(t_{<i}, \sigma_{in})]}{Z(t_{<i})}
$$

where:
- $\text{Valid}(t_{<i}, \sigma_{in})$ is the set of tokens that maintain validity of the partially generated JSON with respect to schema $\sigma_{in}$.
- $Z(t_{<i})$ is the normalizing constant.
- $\mathbb{1}[\cdot]$ is the indicator function.

**Implementation Approaches for Constrained Decoding:**

| Method | Description | Examples |
|---|---|---|
| **Grammar-based** | Define a formal grammar (GBNF, PEG) for valid outputs; mask invalid tokens at each step | llama.cpp GBNF, Outlines |
| **JSON Schema compilation** | Compile JSON Schema to a finite-state automaton; use automaton to constrain token generation | Outlines, guidance, SGLang |
| **Regex-based** | Use regular expressions to define valid token sequences | Outlines regex-guided generation |
| **Post-hoc validation** | Generate freely, validate after; re-prompt on failure | Simple but unreliable |

Grammar-based sampling ensures 100% valid output at the cost of slightly reduced generation speed and potential quality degradation (the model's preferred token may be masked).

---

#### 2.2.5.2 Parallel vs. Sequential Tool Calling

**Single Turn, Single Tool Call.** The simplest case: the model generates exactly one tool call per turn.

```
User: "What's the weather in Paris?"
Model: [tool_call: get_weather(city="Paris")]
Tool Result: {"temp": 22, "condition": "sunny"}
Model: "It's currently 22°C and sunny in Paris."
```

**Single Turn, Multiple Parallel Tool Calls.** The model generates multiple tool calls in a single turn, and the runtime executes them concurrently:

$$
\text{Model output}: \{(\tau_1, \text{args}_1), (\tau_2, \text{args}_2), \ldots, (\tau_k, \text{args}_k)\}
$$

```
User: "What's the weather in Paris and London?"
Model: [tool_call: get_weather(city="Paris"), tool_call: get_weather(city="London")]
Tool Results: [{"city": "Paris", "temp": 22}, {"city": "London", "temp": 16}]
Model: "Paris: 22°C sunny. London: 16°C cloudy."
```

This is valid when the calls are independent — no output of one call is needed as input to another.

**Multi-Turn Sequential Tool Calls.** The model makes one or more tool calls, receives results, reasons about them, and makes further tool calls:

```
Turn 1: Model calls search("CEO of Anthropic")
Turn 2: Model receives result → calls get_linkedin_profile("Dario Amodei")  
Turn 3: Model receives result → generates final answer
```

Each turn involves model reasoning between tool calls, enabling adaptive, conditional workflows.

**Strategy Selection.** The choice between parallel and sequential execution depends on:

$$
\text{Strategy} = \begin{cases}
\text{Parallel} & \text{if } \forall i \neq j: \; \text{args}_j \not\ni \text{result}(\tau_i) \\
\text{Sequential} & \text{if } \exists \; \text{data dependency } \tau_i \rightarrow \tau_j \\
\text{Mixed} & \text{if some calls are independent, others dependent}
\end{cases}
$$

---

#### 2.2.5.3 Forced Tool Use vs. Auto-Detection

The `tool_choice` parameter controls the model's tool-calling behavior:

| Value | Behavior | Use Case |
|---|---|---|
| `"auto"` | Model decides whether to call a tool or respond with text | Default; most flexible |
| `"required"` | Model must call at least one tool (no text-only response allowed) | When the task always requires external data |
| `{"type": "function", "function": {"name": "X"}}` | Model must call the specific tool `X` | When you know exactly which tool is needed |
| `"none"` | Model cannot call any tool; text-only response | When you want to suppress tool use |

**Impact on Model Behavior.**

- `"auto"`: The model applies its own judgment. May sometimes call a tool unnecessarily, or fail to call a tool when it should. Most natural for general-purpose use.
- `"required"`: Eliminates the possibility of the model answering from parametric knowledge when a tool should be used. Useful for tasks where freshness/accuracy is critical.
- Forced specific tool: Eliminates the model's decision about *which* tool to call. Only the argument generation is left to the model. Useful when the routing decision is made by external logic.

---

#### 2.2.5.4 Tool Call Parsing and Validation

**Parsing Model Output.** The runtime must extract tool calls from the model's output. In structured-output APIs, this is straightforward (the API returns a parsed object). In text-based protocols, the runtime must parse the output:

1. Detect tool call delimiters (special tokens, JSON blocks).
2. Extract the tool name and arguments string.
3. Parse the arguments string as JSON.

**Schema Validation.** After extraction, arguments are validated against $\sigma_{in}$:

```python
import jsonschema

try:
    jsonschema.validate(instance=parsed_args, schema=tool.parameters)
except jsonschema.ValidationError as e:
    # Feed error back to model for self-correction
    error_msg = f"Invalid arguments for {tool.name}: {e.message}"
```

**Error Recovery: Re-Prompting.** When validation fails, the error message is injected into the conversation and the model is prompted to correct its call:

$$
\text{If validate}(\text{args}, \sigma_{in}) = \text{Error}(msg): \quad \mathcal{M}(\mathcal{H} \cup \{\text{error\_msg}\}) \rightarrow \text{corrected\_call}
$$

This creates a self-correction loop that typically converges within 1–2 retries for well-trained models.

**Handling Ambiguous or Partial Generation.** The model may generate:
- Incomplete JSON (truncated due to output length limit).
- Multiple tool calls when only one was expected.
- A mix of text and tool calls in ambiguous formatting.

Robust parsing requires heuristics and fallback strategies for each failure mode.

---

#### 2.2.5.5 Tool Result Injection

After tool execution, results must be formatted and injected back into the model's context.

**Formatting Strategies:**

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\n  \"temperature\": 22,\n  \"unit\": \"celsius\",\n  \"condition\": \"sunny\"\n}"
}
```

**Key Considerations:**

- **Structured vs. free-text results**: Structured (JSON) results are more parseable by the model. Free-text results are more natural but harder to extract specific values from.

- **Truncation strategies**: Large tool outputs (e.g., a database query returning 10,000 rows) must be truncated to fit within the context budget:
  - Return only the first $k$ results with a total count indicator.
  - Return a summary/aggregation rather than raw data.
  - Use pagination: return the first page and indicate that more pages are available.

- **Multi-modal results**: Some tools return non-text data:
  - **Images**: Embedded as base64 or referenced by URL.
  - **Tables**: Formatted as Markdown tables or JSON arrays.
  - **Binary data**: Referenced by file path or download URL.

- **Error results**: Tool failures should be clearly distinguished from successful results:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"error\": \"Rate limit exceeded. Retry after 30 seconds.\", \"error_code\": 429}"
}
```

The model can then decide whether to retry, use an alternative tool, or inform the user.

---

## 2.3 Best Practices for Tool Design

### 2.3.1 Documentation Is Important

#### 2.3.1.1 Why Documentation Quality Directly Affects Tool-Use Accuracy

The model has no access to tool source code, no ability to execute a tool speculatively to observe its behavior, and no implicit understanding of what a tool does beyond what is written in its description and schema. **The description IS the tool**, from the model's perspective.

Empirically, tool selection accuracy is strongly correlated with description quality. Studies on function-calling benchmarks show that:

- Vague descriptions reduce selection accuracy by 20–40%.
- Missing parameter descriptions increase argument errors by 30–50%.
- Adding examples to descriptions improves accuracy by 10–15%.

**Information-Theoretic Perspective.** The model's ability to correctly use a tool depends on the mutual information between the description and the correct usage pattern:

$$
I(\text{desc}; \text{correct\_usage}) = H(\text{correct\_usage}) - H(\text{correct\_usage} \mid \text{desc})
$$

A maximally informative description minimizes $H(\text{correct\_usage} \mid \text{desc})$ — the uncertainty about correct usage given the description. A poor description leaves high residual uncertainty, leading to errors.

---

#### 2.3.1.2 Components of Effective Tool Documentation

**1. Tool-Level Description.** This is the most important single piece of text. It should contain:

- **What the tool does**: A clear, specific statement of the tool's purpose.
- **When to use it**: Conditions under which this tool is the right choice.
- **When NOT to use it**: Explicit negative guidance to prevent misuse.
- **Important constraints**: Rate limits, permissions, prerequisites.

Example of an excellent tool description:

```
"Retrieves the current stock price and daily change for a given ticker symbol 
from the NYSE or NASDAQ exchanges. Use this when the user asks about current 
stock prices, market performance, or portfolio values. Do NOT use this for 
cryptocurrency prices (use get_crypto_price instead), historical data beyond 
30 days (use get_historical_prices instead), or market predictions. Returns 
prices in USD. Requires a valid ticker symbol; the tool will return an error 
for delisted or invalid tickers."
```

**2. Parameter-Level Descriptions.** Each parameter needs:
- Its purpose (what it controls).
- Valid value ranges and formats.
- Default value (if optional).
- Examples of valid inputs.

```json
"ticker": {
  "type": "string",
  "description": "Stock ticker symbol (e.g., 'AAPL' for Apple, 'GOOGL' for Alphabet). Must be a valid NYSE or NASDAQ listed symbol. Case-insensitive."
}
```

**3. Return Value Description.** What the tool returns, including:
- Structure of the response.
- Units and formats.
- Edge cases (what is returned when no data is found).

**4. Examples.** Input-output pairs demonstrating correct usage:

```
Example: get_stock_price(ticker="AAPL")
Returns: {"price": 178.52, "change": +2.31, "change_pct": +1.31, "currency": "USD", "timestamp": "2024-01-15T16:00:00Z"}
```

**5. Constraints and Preconditions.** State requirements, authentication needs, rate limits:

```
"Rate limited to 60 calls per minute. Requires market to be open for real-time prices; returns last closing price outside market hours."
```

**6. Error Descriptions.** Possible error codes and their meanings:

```
"Returns error 'INVALID_TICKER' if the symbol is not found. Returns error 'MARKET_CLOSED' with last closing price if called outside trading hours."
```

---

#### 2.3.1.3 Documentation Anti-Patterns

| Anti-Pattern | Example | Problem |
|---|---|---|
| **Overly terse** | `"Gets data"` | Model cannot determine what data, from where, or when to use it |
| **Implementation-leaking** | `"Calls the /api/v2/users endpoint with OAuth2 Bearer token"` | Exposes internal architecture; model doesn't need endpoint details; potential security leak |
| **Ambiguous scope** | `"Handles user operations"` | Could mean read, create, update, delete — model cannot distinguish |
| **Missing negative guidance** | No mention of when NOT to use | Model may misapply the tool for similar-sounding but inappropriate tasks |
| **Copy-pasted API docs** | Verbatim OpenAPI spec with HTTP details | Not adapted for model consumption; includes irrelevant technical details |
| **Jargon without context** | `"Executes a CQRS command against the event store"` | Model may not understand domain-specific architecture patterns |

---

#### 2.3.1.4 Techniques for Documentation Optimization

**A/B Testing.** Systematically compare different descriptions:

1. Create variant descriptions for the same tool.
2. Run a benchmark of tool-use scenarios.
3. Measure selection accuracy, argument correctness, and task completion rate.
4. Select the best-performing description.

**Automated Generation and Refinement.** Use the model itself to improve descriptions:

```
Prompt: "Given this tool implementation and its current description, suggest 
an improved description that would help an AI model correctly decide when 
to use this tool and how to provide its arguments."
```

**Version-Controlled Documentation.** Treat tool descriptions like code:
- Track changes in version control.
- Run regression tests when descriptions change.
- Monitor tool-use accuracy metrics in production after description updates.

---

### 2.3.2 Describe Actions, Not Implementations

#### 2.3.2.1 Abstraction Principle

Tool descriptions should communicate **what** the tool accomplishes from the user's perspective, not **how** it accomplishes it from a systems perspective. The model is a semantic reasoner operating at the level of user intent, not a systems engineer concerned with API endpoints, authentication mechanisms, or database schemas.

**The Principle:**

$$
\text{desc}(\tau) \in \text{Semantic Space (actions, outcomes)} \;\;\not\in\;\; \text{Implementation Space (endpoints, protocols, internals)}
$$

**Concrete Examples:**

| ❌ Implementation-Oriented | ✅ Action-Oriented |
|---|---|
| "Sends a POST request to `https://api.weather.com/v3/forecast` with API key in X-Auth header" | "Retrieves the weather forecast for a given city and date range, returning temperature, conditions, and precipitation probability" |
| "Executes `SELECT * FROM orders WHERE user_id = ? AND created_at > ?` against PostgreSQL replica" | "Looks up a user's recent orders, returning order details including items, prices, and delivery status" |
| "Invokes the SendGrid v3 API with template ID tm_abc123 and dynamic template data" | "Sends a formatted email to the specified recipient with the given subject and body content" |

---

#### 2.3.2.2 Benefits of Action-Oriented Descriptions

1. **Backend independence**: If the weather API changes from v3 to v4, or the provider switches from SendGrid to Mailgun, the tool description doesn't need to change.

2. **Reduced prompt injection surface**: Implementation details (endpoints, API keys, internal paths) in descriptions can be extracted by adversarial users and exploited.

3. **Better generalization**: An action-oriented description enables the model to correctly apply the tool to a wider range of phrasings and contexts.

4. **Cleaner multi-tool reasoning**: When the model is reasoning about which tools to use, action-oriented descriptions map directly to the user's intent, making selection more accurate.

---

### 2.3.3 Publish Tasks, Not API Calls

#### 2.3.3.1 Task-Level Abstraction

Rather than exposing every low-level API endpoint as a separate tool, combine related operations into **task-level tools** that correspond to complete user-visible actions.

**Example: Email Sending.**

❌ **Low-level (4 separate tools):**

```
create_draft(subject, body) → draft_id
add_attachment(draft_id, file_id) → void
set_recipients(draft_id, to, cc, bcc) → void
send_draft(draft_id) → message_id
```

The model must correctly sequence these 4 calls, passing IDs between them, handling each possible failure point.

✅ **Task-level (1 tool):**

```
send_email(to, cc, bcc, subject, body, attachments) → message_id
```

The model makes a single call; the tool implementation handles the internal sequencing.

**Quantitative Impact.** With $k$ low-level tools that must be called in sequence:

- Probability of correct execution: $P_{\text{correct}} = \prod_{i=1}^{k} p_i$, where $p_i$ is the probability of correctly calling tool $i$.
- For $k = 4$ and $p_i = 0.95$: $P_{\text{correct}} = 0.95^4 \approx 0.815$.
- With a single task-level tool: $P_{\text{correct}} = 0.95$ (one decision point).

---

#### 2.3.3.2 When to Decompose vs. When to Compose

| Criterion | Compose (Single Task Tool) | Decompose (Multiple Fine-Grained Tools) |
|---|---|---|
| Workflow is deterministic | ✅ Always the same sequence | |
| Steps require conditional logic | | ✅ Model needs to decide between branches |
| Human approval needed at intermediate steps | | ✅ Approval gate between sub-steps |
| Sub-steps are independently useful | | ✅ Other tasks reuse individual tools |
| Error recovery requires granularity | | ✅ Need to retry individual steps |
| Reducing model planning burden | ✅ One decision instead of many | |

**Decision Framework:**

$$
\text{Use composition if: } \text{branching\_factor} = 1 \;\land\; \text{no\_approval\_gates} \;\land\; \text{deterministic\_sequence}
$$

$$
\text{Use decomposition if: } \text{branching\_factor} > 1 \;\lor\; \text{requires\_human\_review} \;\lor\; \text{sub-steps\_independently\_valuable}
$$

---

### 2.3.4 Make Tools as Granular as Possible

#### 2.3.4.1 The Granularity Spectrum

Tool granularity ranges from coarse (a single tool that does everything) to fine (each tool does exactly one thing):

**Coarse-Grained (God Tool Anti-Pattern):**

```
manage_user(action, user_id, data) 
# action ∈ {"create", "read", "update", "delete", "list", "search", "export"}
```

The model must determine the `action` parameter, which effectively makes this 7 tools overloaded into one. The description becomes vague, and the parameter schema becomes overly complex with conditional fields.

**Fine-Grained (Unix Philosophy):**

```
create_user(name, email, role)
get_user(user_id)
update_user(user_id, fields)
delete_user(user_id)
list_users(filters, pagination)
search_users(query)
export_users(format)
```

Each tool has a clear, unambiguous purpose. The model can accurately select the right tool for any user management task.

---

#### 2.3.4.2 Why Granularity Matters

1. **Disambiguation**: Fine-grained tools have distinct names and descriptions, making it easier for the model to select the correct one. Coarse tools require the model to understand complex conditional logic within a single tool.

2. **Composability**: Fine-grained tools can be composed in novel ways by the model. Coarse tools with fixed internal workflows are rigid.

3. **Error Isolation**: When a fine-grained tool fails, the failure is localized. The model knows exactly which operation failed and can retry or adapt. With a coarse tool, the failure may be ambiguous.

4. **Testability**: Each fine-grained tool can be independently tested with clear inputs and expected outputs.

---

#### 2.3.4.3 Balancing Granularity with Context Window Efficiency

There is a tension: finer granularity means more tools, and more tools consume more context tokens. This creates a trade-off:

$$
\text{Total Tool Token Cost} = N \times L
$$

where $N$ is the number of tools and $L$ is the average token count per tool definition.

**Empirical Observation.** Tool selection accuracy degrades as the number of tools increases, approximately following:

$$
\text{Selection Accuracy} \approx f\left(\frac{1}{\log N}\right)
$$

This logarithmic decay means that doubling the number of tools doesn't halve accuracy, but it does degrade it noticeably. For very large tool sets ($N > 100$), accuracy can drop significantly.

**Mitigation Strategies:**

| Strategy | Description | Trade-off |
|---|---|---|
| **Dynamic tool loading** | Only load tools relevant to the current query | Requires a tool selection/routing pre-step |
| **Tool summarization** | Use shorter descriptions with key details only | May reduce model's understanding of edge cases |
| **Hierarchical organization** | Group tools by category; route to category first, then specific tool | Adds an extra routing step |
| **Tool compression** | Combine related tools into a single tool with a mode parameter (controlled decomposition) | Slightly coarser granularity |
| **Two-stage selection** | First model call selects relevant tool subset; second call uses selected tools | Higher latency (two LLM calls) |

---

### 2.3.5 Design for Concise Output

#### 2.3.5.1 Why Output Size Matters

Tool results are injected into the model's context window, consuming tokens that reduce the space available for reasoning and generation. The context window budget is:

$$
W_{\text{max}} = W_{\text{system}} + W_{\text{history}} + W_{\text{tool\_defs}} + W_{\text{tool\_results}} + W_{\text{generation}}
$$

Each component competes for the finite context budget $W_{\text{max}}$. If $W_{\text{tool\_results}}$ is large, it crowds out $W_{\text{generation}}$ (the model's ability to reason and respond) and $W_{\text{history}}$ (the model's memory of prior conversation).

**Information Density.** The goal is to maximize the signal-to-noise ratio in tool outputs:

$$
\text{Info Density} = \frac{\text{task-relevant information (bits)}}{\text{total output tokens}}
$$

A database query that returns all 50 columns when only 3 are needed has low information density and wastes context tokens.

---

#### 2.3.5.2 Techniques for Concise Output Design

**1. Projection (Return Only Relevant Fields).**
Instead of returning entire records, return only the fields relevant to the tool's purpose:

```python
# ❌ Returns everything
def get_user(user_id):
    return db.query("SELECT * FROM users WHERE id = ?", user_id)

# ✅ Returns only what's needed
def get_user(user_id):
    return db.query("SELECT name, email, role FROM users WHERE id = ?", user_id)
```

**2. Pagination for Large Result Sets.**
Never return unbounded result sets:

```json
{
  "results": [...first 10 items...],
  "total_count": 1547,
  "page": 1,
  "has_more": true
}
```

**3. Summarization of Verbose Outputs.**
For tools that naturally produce large outputs (e.g., web page content, log files), summarize before returning:
- Extract key facts, dates, numbers.
- Remove boilerplate, headers, footers.
- Optionally use a specialized model for summarization.

**4. Structured Output Over Free Text.**
Return JSON/tables rather than narrative text:

```json
// ❌ Free text
"The server is currently running with 72% CPU utilization, 8.3 GB of 16 GB RAM used..."

// ✅ Structured
{"cpu_pct": 72, "ram_used_gb": 8.3, "ram_total_gb": 16, "status": "healthy"}
```

**5. Progressive Disclosure.**
Return a summary first; provide details only on request:

```json
{
  "summary": "Found 23 matching orders totaling $4,521.00",
  "sample": [...first 3 orders...],
  "detail_available": true,
  "detail_tool": "get_order_details"
}
```

**6. Truncation with Indicators.**

```json
{
  "log_excerpt": "...first 500 chars...",
  "truncated": true,
  "total_length": 15420,
  "full_log_available_via": "get_full_log(log_id='abc123')"
}
```

---

#### 2.3.5.3 Output Formatting Best Practices

| Practice | Rationale |
|---|---|
| Consistent structure across similar tools | Model learns patterns; reduces parsing errors |
| Use enums/status codes over verbose messages | `"status": "success"` vs. `"The operation completed successfully"` |
| ISO 8601 for dates | `"2024-01-15T16:00:00Z"` — unambiguous, parseable |
| ISO 4217 for currencies | `"currency": "USD"` — not `"dollars"` or `"$"` |
| SI units with explicit unit fields | `{"value": 22, "unit": "celsius"}` — not `"22°C"` |
| Markdown-compatible text | Enables direct rendering in chat interfaces |

---

### 2.3.6 Use Validation Effectively

#### 2.3.6.1 Input Validation

Input validation is the first line of defense against malformed tool calls. It serves two purposes:

1. **Preventing execution errors**: Catching invalid inputs before they cause runtime failures.
2. **Guiding model self-correction**: Informative error messages help the model fix its tool call.

**Validation Layers:**

| Layer | Check | Example |
|---|---|---|
| **Type validation** | Correct JSON types | Is `"temperature"` a number, not a string? |
| **Schema validation** | Conformance to JSON Schema | Are all `required` fields present? |
| **Range validation** | Value within acceptable bounds | Is $0 \leq \text{limit} \leq 1000$? |
| **Pattern validation** | String matches expected format | Does email match `^[^@]+@[^@]+\.[^@]+$`? |
| **Enum validation** | Value is one of allowed options | Is `unit` ∈ `{"celsius", "fahrenheit"}`? |
| **Semantic validation** | Value makes domain sense | Is `date_start` before `date_end`? |
| **Cross-field validation** | Consistency between parameters | If `action` = "update", is `data` non-empty? |

**Type Coercion and Normalization.** Some tools should gracefully handle minor type mismatches:
- `"42"` → `42` (string to number coercion)
- `"TRUE"` → `true` (case-insensitive boolean)
- `"new york"` → `"New York"` (capitalization normalization)

**Fail-Fast with Informative Error Messages.** Error messages should guide the model to correct its call:

```json
// ❌ Unhelpful
{"error": "Invalid input"}

// ✅ Helpful
{"error": "Parameter 'date' has invalid format 'January 15'. Expected ISO 8601 format: 'YYYY-MM-DD' (e.g., '2024-01-15')."}
```

---

#### 2.3.6.2 Output Validation

After tool execution, validate the output before injecting it into the model's context:

**Schema Conformance.** Verify that the tool's output matches $\sigma_{out}$:

```python
if not validate(result, tool.output_schema):
    result = {"error": "Tool returned unexpected output format", "raw": str(result)[:500]}
```

**Sanity Checks:**
- **Null/empty detection**: If a search returns no results, is this expected or an error?
- **Anomaly detection**: If a stock price is negative or a temperature is 500°C, flag as potentially erroneous.
- **Content validation**: Does the result contain the expected type of information?

---

#### 2.3.6.3 Validation-Driven Self-Correction

When validation fails, the error is fed back to the model, creating a self-correction loop:

$$
\text{If } \text{validate}(\tau(\text{args})) = \text{Error}(msg):
$$

$$
\mathcal{M}(\mathcal{H} \cup \{\text{error\_msg}\}) \rightarrow \text{corrected\_call}
$$

This loop is remarkably effective — well-trained models can correct their tool calls based on error messages with high reliability.

**Retry Policies:**

| Policy | Description | Use Case |
|---|---|---|
| **Fixed retry** | Retry up to $k$ times | Simple tool calls with likely transient errors |
| **Exponential backoff** | Wait $2^n$ seconds between retries | Rate-limited tools |
| **Error-dependent** | Different retry strategy based on error type | User error (re-prompt) vs. system error (wait) vs. permanent error (give up) |

**Error Taxonomy:**

| Error Type | Cause | Recovery |
|---|---|---|
| **User error** (bad args) | Model generated invalid arguments | Re-prompt with error message |
| **System error** (tool failure) | Tool implementation crashed | Retry, fallback to alternative tool |
| **Transient error** | Network timeout, rate limit | Retry with backoff |
| **Permanent error** | Resource not found, insufficient permissions | Inform user, do not retry |

The model should be able to distinguish these error types and apply appropriate recovery strategies.

---

#### 2.3.6.4 Human-in-the-Loop Validation

For high-stakes tool invocations, automated validation is insufficient. Human oversight is required:

**Approval Gates.** Before executing irreversible or high-risk actions:

```
Agent: "I'd like to send the following email to 500 customers:
  Subject: Price Change Notification
  Body: [...]
  
  Shall I proceed? [Approve / Reject / Modify]"
```

**Preview/Confirm Patterns:**

1. Model generates the tool call.
2. System presents the call to the human for review.
3. Human approves, rejects, or modifies.
4. Only approved calls are executed.

**Risk-Based Gating.** Define risk levels for tool actions and require human approval proportional to risk:

$$
\text{approval\_required}(\tau, \text{args}) = \begin{cases}
\text{false} & \text{if risk}(\tau, \text{args}) \leq \theta_{\text{auto}} \\
\text{true} & \text{if risk}(\tau, \text{args}) > \theta_{\text{auto}}
\end{cases}
$$

Examples:
- Reading data: auto-approve (low risk).
- Modifying a draft: auto-approve (reversible).
- Sending an email: require approval (irreversible).
- Deleting records: require explicit confirmation with details (irreversible, high impact).
- Financial transactions: multi-level approval (high value, regulatory requirements).

**Audit Logging.** Every tool invocation — whether approved, rejected, or auto-approved — should be logged:

```json
{
  "timestamp": "2024-01-15T16:00:00Z",
  "agent_id": "agent_abc",
  "tool": "send_email",
  "arguments": {"to": "...", "subject": "..."},
  "risk_level": "high",
  "approval": "human_approved",
  "approver": "user_xyz",
  "result": {"status": "sent", "message_id": "msg_123"},
  "execution_time_ms": 1520
}
```

This audit trail is essential for:
- Post-hoc analysis of agent behavior.
- Compliance and regulatory requirements.
- Debugging agent failures.
- Training data collection for improving tool-use models.

---

## Summary: Key Principles

| Principle | Core Idea |
|---|---|
| **Tool = 5-tuple** | $\tau = (\text{name}, \text{desc}, \sigma_{in}, \sigma_{out}, f)$ — complete specification |
| **Model = Controller** | Model decides; runtime executes; environment responds |
| **Agent = $(\mathcal{M}, \mathcal{T}, \mathcal{P}, \mathcal{S})$** | Model + tools + planning + state management |
| **Description is everything** | The model sees only the description; quality directly determines accuracy |
| **Actions, not implementations** | Describe what the tool does, not how |
| **Tasks, not API calls** | Wrap multi-step workflows into single task-level tools |
| **Fine granularity** | Each tool = one clear action; compose rather than overload |
| **Concise outputs** | Minimize token consumption; maximize information density |
| **Validation at every layer** | Input validation, output validation, self-correction loops |
| **Human oversight for high risk** | Approval gates, preview/confirm, audit logging |