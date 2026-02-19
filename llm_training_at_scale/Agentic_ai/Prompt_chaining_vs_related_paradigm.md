
## 1.16 Prompt Chaining vs. Related Paradigms

### 1.16.1 Prompt Chaining vs. Agentic Systems

#### Fundamental Distinction

| Dimension | Prompt Chain | Agentic System |
|---|---|---|
| **Control flow** | Pre-determined at design time | Dynamically determined at runtime |
| **Decision authority** | Developer defines all branches | Agent decides next action |
| **Termination** | Fixed steps complete | Goal-satisfaction condition |
| **Tool selection** | Predetermined per step | Agent selects tools dynamically |
| **Error handling** | Explicit retry/fallback logic | Agent reasons about failures |
| **Predictability** | High (deterministic flow) | Low (emergent behavior) |
| **Debuggability** | Easy (trace each step) | Hard (reasoning opaque) |
| **Cost predictability** | Bounded (fixed steps) | Unbounded (variable iterations) |

#### Formal Characterization

**Chain** as a fixed function composition:

$$
\mathcal{C}(x) = (s_n \circ s_{n-1} \circ \cdots \circ s_1)(x)
$$

where the sequence $(s_1, \ldots, s_n)$ is **determined at design time**.

**Agent** as a dynamic program:

$$
\text{Agent}(x) = \begin{cases}
a_t = \pi(s_t, h_t) & \text{(select action)} \\
o_t = \text{env}(a_t) & \text{(execute action)} \\
s_{t+1} = \text{update}(s_t, o_t) & \text{(update state)} \\
\text{terminate if } \text{goal}(s_t) & \text{(check goal)}
\end{cases}
$$

where $\pi$ is the **policy** (LLM-based), $h_t$ is the action history, and the number of iterations is **not predetermined**.

#### Decision Criteria: When to Use Which

```
                          Task Complexity
                    Low ◄──────────────────► High
                    │                          │
     Predictability │   Single Prompt          │
         High       │   ────────────           │   Prompt Chain
                    │                          │   ─────────────
                    │                          │
                    │   Prompt Chain            │   Hybrid (Chain
                    │   ─────────────          │   + Agentic Steps)
         Low        │                          │
                    │   Simple Agent            │   Full Agent
                    │   ────────────           │   ──────────
                    │                          │
```

**Use chains when**:
- Task decomposition is known a priori
- Consistent latency/cost is required
- Audit trail must be deterministic
- Error modes are well-characterized

**Use agents when**:
- Task requires exploration/planning
- Available actions depend on intermediate results
- Problem solving requires iterative refinement with unknown step count
- Environment is dynamic

#### Hybrid Pattern: Chain with Agentic Steps

```python
class HybridChain:
    """Chain where specific steps use agentic reasoning."""
    
    def execute(self, query: str) -> dict:
        # Step 1: Chain step (deterministic)
        classification = self.classify(query)
        
        # Step 2: Agentic step (dynamic tool use)
        # The agent decides which tools to use and how many iterations
        research_result = self.research_agent.run(
            goal=f"Find comprehensive information about: {classification['topic']}",
            available_tools=["web_search", "arxiv_search", "calculator"],
            max_iterations=10,
            budget_tokens=5000
        )
        
        # Step 3: Chain step (deterministic)
        report = self.generate_report(classification, research_result)
        
        # Step 4: Chain step (deterministic)
        formatted = self.format_output(report)
        
        return formatted
```

#### The Chain-to-Agent Continuum

```
Rigid Chain ◄─────────────────────────────────────► Full Agent

│ Fixed steps   │ Conditional  │ Loops with   │ Dynamic    │ Autonomous │
│ no branching  │ branching    │ termination  │ tool       │ goal       │
│               │              │ conditions   │ selection  │ pursuit    │
│               │              │              │            │            │
│ s1→s2→s3     │ s1→{s2a|s2b} │ s1→(s2→s3)* │ s1→agent→  │ agent(goal)│
│               │ →s3          │ →s4          │ s3         │            │
```

---

### 1.16.2 Prompt Chaining vs. Fine-Tuning

#### Cost-Benefit Analysis Framework

Define the **break-even point** between chaining and fine-tuning:

$$
\text{Fine-tune if: } N_{\text{queries}} \cdot C_{\text{chain}} > C_{\text{finetune}} + N_{\text{queries}} \cdot C_{\text{single-call}}
$$

$$
N_{\text{break-even}} = \frac{C_{\text{finetune}}}{C_{\text{chain}} - C_{\text{single-call}}}
$$

where:
- $C_{\text{chain}}$ = per-query cost of the multi-step chain
- $C_{\text{single-call}}$ = per-query cost of the fine-tuned model (single call)
- $C_{\text{finetune}}$ = one-time cost of fine-tuning (data curation + training compute)

**Example**: If $C_{\text{chain}} = \$0.05$, $C_{\text{single-call}} = \$0.005$, and $C_{\text{finetune}} = \$500$:

$$
N_{\text{break-even}} = \frac{500}{0.05 - 0.005} = 11{,}111 \text{ queries}
$$

#### Decision Matrix

| Factor | Favors Chaining | Favors Fine-Tuning |
|---|---|---|
| Data availability | Few/no examples | Thousands of examples |
| Task stability | Rapidly evolving | Stable and well-defined |
| Latency requirements | Tolerant (>2s) | Strict (<500ms) |
| Query volume | Low (<10K/month) | High (>100K/month) |
| Interpretability | Need step-by-step trace | Black-box acceptable |
| Update frequency | Frequent changes | Rare updates |
| Task complexity | Multi-faceted reasoning | Single well-defined mapping |

#### Chain Distillation

Convert a successful chain into training data for a single fine-tuned model:

```python
class ChainDistiller:
    """
    Distill a multi-step chain into a single fine-tuned model.
    
    Process:
    1. Run chain on large input corpus
    2. Collect (input, final_output) pairs
    3. Filter by quality
    4. Fine-tune a single model on these pairs
    """
    
    def generate_training_data(self, chain, inputs: list[str],
                                quality_threshold: float = 0.8) -> list[dict]:
        training_data = []
        
        for input_text in inputs:
            try:
                result = chain.invoke({"text": input_text})
                quality = self.evaluate_quality(input_text, result)
                
                if quality >= quality_threshold:
                    training_data.append({
                        "messages": [
                            {"role": "user", "content": input_text},
                            {"role": "assistant", "content": result["output"]}
                        ],
                        "quality_score": quality
                    })
            except Exception:
                continue  # Skip failed executions
        
        return training_data
    
    def distill(self, chain, inputs: list[str]) -> str:
        """Full distillation pipeline."""
        # Generate data
        data = self.generate_training_data(chain, inputs)
        
        # Fine-tune
        model_id = self.fine_tune(data)
        
        # Validate distilled model matches chain quality
        validation_score = self.validate(model_id, chain, validation_inputs)
        
        if validation_score < 0.95 * chain_baseline_score:
            raise DistillationError(
                f"Distilled model quality ({validation_score:.3f}) "
                f"below threshold (95% of chain: {0.95 * chain_baseline_score:.3f})"
            )
        
        return model_id
```

---

### 1.16.3 Prompt Chaining vs. Single Complex Prompt

#### Complexity Threshold Analysis

The decision between a single prompt and a chain depends on **task decomposability** and **context window utilization**:

$$
\text{Use chain if: } \frac{T_{\text{task}}}{W_{\text{context}}} > \theta_{\text{window}} \text{ OR } D_{\text{task}} > \theta_{\text{decomp}}
$$

where:
- $T_{\text{task}}$ = total tokens required for task context + instructions + output
- $W_{\text{context}}$ = model context window size
- $\theta_{\text{window}} \approx 0.7$ (filling >70% of context degrades quality)
- $D_{\text{task}}$ = task decomposability score (number of independent sub-tasks)
- $\theta_{\text{decomp}} \approx 3$ (more than 3 independent sub-tasks)

#### Empirical Quality Comparison

For a task with complexity $d$ (number of reasoning steps), the expected quality of single-prompt vs. chain approaches:

$$
Q_{\text{single}}(d) \approx Q_0 \cdot e^{-\lambda d}
$$

$$
Q_{\text{chain}}(d) \approx Q_0 \cdot \prod_{i=1}^{d} (1 - \epsilon_i)
$$

where $\lambda$ is the **complexity decay rate** for single prompts and $\epsilon_i$ is the **per-step error rate** in chains.

For small $\epsilon_i$:

$$
Q_{\text{chain}}(d) \approx Q_0 \cdot e^{-\sum_i \epsilon_i} \approx Q_0 \cdot e^{-d \bar{\epsilon}}
$$

Chains outperform single prompts when:

$$
d \bar{\epsilon} < \lambda d \implies \bar{\epsilon} < \lambda
$$

This holds when **per-step error rate** $\bar{\epsilon}$ is lower than the **single-prompt complexity decay rate** $\lambda$, which is true for well-decomposed tasks since each sub-task is simpler than the aggregate.

#### Context Window Utilization Analysis

```
Single Prompt:
┌─────────────────────────────────────────────────────┐
│ System + Instructions + All Context + All Examples  │
│                                                     │
│  ████████████████████████████████████████████████    │ 85% utilized
│                                                     │
│  ↑ Quality degrades as context fills ↑              │
└─────────────────────────────────────────────────────┘

Chain (3 steps):
Step 1: ┌──────────────────┐
        │ Focused context  │ 30% utilized
        │ ██████████       │
        └──────────────────┘
Step 2: ┌──────────────────┐
        │ Step 1 output +  │ 25% utilized
        │ new instructions │
        │ ████████         │
        └──────────────────┘
Step 3: ┌──────────────────┐
        │ Step 2 output +  │ 20% utilized
        │ final task       │
        │ ██████           │
        └──────────────────┘
```

Each chain step operates with a **focused context window**, avoiding the quality degradation that occurs with heavily-loaded single prompts.

---

### 1.16.4 Prompt Chaining vs. Multi-Agent Systems

#### Structural Comparison

| Aspect | Single-Agent Chain | Multi-Agent System |
|---|---|---|
| **Agents** | One LLM, multiple steps | Multiple specialized LLMs |
| **Communication** | Sequential data passing | Message-passing protocols |
| **Roles** | Defined by step prompt | Defined by agent persona |
| **Conflict resolution** | N/A (single executor) | Debate, voting, arbitration |
| **Parallelism** | Step-level | Agent-level |
| **State management** | Centralized context | Distributed per-agent state |
| **Failure modes** | Step failure cascades | Agent disagreement, deadlock |

#### Chain as Multi-Agent Orchestration Backbone

```python
class MultiAgentOrchestrator:
    """
    Chain topology orchestrating multiple specialized agents.
    """
    
    def __init__(self):
        self.researcher = Agent("researcher", model="gpt-4o", 
                                persona="Expert research analyst")
        self.critic = Agent("critic", model="claude-3.5-sonnet",
                           persona="Rigorous critic and fact-checker")
        self.writer = Agent("writer", model="gpt-4o",
                           persona="Expert technical writer")
    
    def execute(self, topic: str) -> dict:
        # Step 1: Research agent gathers information
        research = self.researcher.run(
            f"Research the topic thoroughly: {topic}"
        )
        
        # Step 2: Critic agent reviews research
        critique = self.critic.run(
            f"""Review this research for accuracy and completeness.
            Identify gaps, errors, and unsupported claims.
            
            Research: {research}"""
        )
        
        # Step 3: Researcher addresses critique
        revised_research = self.researcher.run(
            f"""Address the following critique of your research.
            Fill gaps, correct errors, provide evidence.
            
            Original research: {research}
            Critique: {critique}"""
        )
        
        # Step 4: Writer synthesizes
        article = self.writer.run(
            f"""Write a comprehensive article on {topic} using this research.
            
            Research: {revised_research}
            Key critiques addressed: {critique}"""
        )
        
        # Step 5: Final review by critic
        final_review = self.critic.run(
            f"""Final review of this article. Score 1-10 on:
            accuracy, completeness, clarity, engagement.
            
            Article: {article}"""
        )
        
        return {
            "article": article,
            "review": final_review,
            "research": revised_research
        }
```

#### Communication Protocol Comparison

| Protocol | Description | Use Case |
|---|---|---|
| **Sequential passing** | Output of agent $A$ → input of agent $B$ | Chains, pipelines |
| **Blackboard** | Shared workspace all agents read/write | Collaborative editing |
| **Message queue** | Async pub/sub between agents | Scalable distributed systems |
| **Debate** | Agents argue positions, judge decides | Adversarial verification |
| **Voting** | Multiple agents propose, majority wins | Ensemble decision making |

Formally, the communication overhead in a multi-agent system with $n$ agents using pairwise messaging is:

$$
\text{Messages} = O(n^2 \cdot r)
$$

where $r$ is the number of communication rounds. In contrast, a chain has:

$$
\text{Messages} = O(n)
$$

making chains inherently more token-efficient when the topology is predetermined.

---

### Summary: Paradigm Selection Decision Tree

```
                        Task Definition
                             │
                     ┌───────┴───────┐
                     │ Is the workflow│
                     │ well-defined? │
                     └───────┬───────┘
                        Yes  │  No
                  ┌──────────┴──────────┐
                  ▼                     ▼
           ┌─────────────┐      ┌──────────────┐
           │ Multiple     │      │  Agent or    │
           │ sub-tasks?   │      │  Multi-Agent │
           └──────┬──────┘      └──────────────┘
             Yes  │  No
          ┌───────┴────────┐
          ▼                ▼
   ┌──────────────┐  ┌──────────────┐
   │ >10K queries │  │Single Prompt │
   │ /month?      │  └──────────────┘
   └──────┬──────┘
     Yes  │  No
  ┌───────┴────────┐
  ▼                ▼
┌──────────┐  ┌──────────┐
│Fine-tune │  │ Prompt   │
│(distill  │  │ Chain    │
│ chain)   │  │          │
└──────────┘  └──────────┘
```

This decision tree provides a systematic framework for selecting the appropriate paradigm given task characteristics, volume requirements, and operational constraints. The key insight is that **these paradigms are not mutually exclusive**—production systems frequently combine chains for well-understood workflows, agents for exploratory steps, and fine-tuned models for high-volume bottleneck steps.