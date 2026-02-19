## 1.15 Real-World Applications and Case Studies

### 1.15.1 Document Processing Chains

#### Architecture: End-to-End Document Analysis

```
                  Document Processing Chain
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Ingest  │─▶│  Chunk  │─▶│  Embed  │─▶│Retrieve │─▶│Generate │
│         │  │         │  │         │  │         │  │         │
│• PDF    │  │• Semantic│  │• Dense  │  │• Hybrid │  │• Answer │
│• DOCX   │  │  split  │  │  embed  │  │  search │  │• Summary│
│• HTML   │  │• Overlap│  │• Sparse │  │• Rerank │  │• Report │
│• Images │  │• Metadata│  │  index  │  │• Filter │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

**Step 1: Document Ingestion** — Parse heterogeneous document formats into unified text + metadata representation. Handle OCR for scanned documents, table extraction, image captioning.

**Step 2: Semantic Chunking** — Split documents into semantically coherent segments. Use recursive character splitting with overlap, or embedding-based boundary detection:

$$
\text{Split at position } i \text{ if } \cos(\mathbf{e}_{i-1}, \mathbf{e}_i) < \tau
$$

where $\mathbf{e}_i$ is the embedding of sentence $i$ and $\tau$ is a similarity threshold.

**Step 3: Embedding and Indexing** — Generate dense embeddings and build vector index. Optionally create sparse (BM25) index for hybrid search.

**Step 4: Retrieval** — Given a query, retrieve top-$k$ chunks via hybrid search (dense + sparse fusion), then rerank with a cross-encoder:

$$
\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(q, d)
$$

**Step 5: Generation** — Synthesize answer from retrieved context, with citation tracking.

#### Multi-Document Summarization Chain

```python
class MultiDocSummarizationChain:
    """
    Map-Reduce summarization chain for large document collections.
    
    Stage 1 (Map): Summarize each document independently
    Stage 2 (Reduce): Hierarchically merge summaries
    """
    
    def __init__(self, llm, max_tokens_per_summary: int = 500):
        self.llm = llm
        self.max_tokens = max_tokens_per_summary
    
    def execute(self, documents: list[str]) -> str:
        # Stage 1: Map — parallel summarization
        summaries = []
        for doc in documents:
            summary = self.llm.invoke(
                f"Summarize the following document in {self.max_tokens} tokens:\n\n{doc}"
            )
            summaries.append(summary)
        
        # Stage 2: Hierarchical reduce
        while len(summaries) > 1:
            merged = []
            for i in range(0, len(summaries), 3):
                batch = summaries[i:i+3]
                combined = "\n---\n".join(batch)
                merged_summary = self.llm.invoke(
                    f"Merge these summaries into a single coherent summary:\n\n{combined}"
                )
                merged.append(merged_summary)
            summaries = merged
        
        return summaries[0]
```

**Complexity**: For $N$ documents, the map stage requires $N$ LLM calls. The reduce stage requires $\lceil \log_k N \rceil$ rounds where $k$ is the merge factor, each round with $\lceil N/k^r \rceil$ calls at round $r$. Total calls:

$$
\text{Total LLM calls} = N + \sum_{r=1}^{\lceil \log_k N \rceil} \left\lceil \frac{N}{k^r} \right\rceil \approx N + \frac{N}{k-1}
$$

---

### 1.15.2 Code Generation Chains

#### Full Pipeline: Specification → Deployment

```
Spec → Plan → Code → Test → Debug → Refine → Review → Deploy

┌────────┐   ┌────────┐   ┌─────────┐   ┌──────────┐
│ Parse  │──▶│ Design │──▶│Generate │──▶│ Generate │
│ Spec   │   │  Plan  │   │  Code   │   │  Tests   │
└────────┘   └────────┘   └─────────┘   └────┬─────┘
                                              │
                                              ▼
                          ┌──────────┐   ┌──────────┐
                          │  Debug   │◀──│ Execute  │
                          │  & Fix   │   │  Tests   │
                          └────┬─────┘   └──────────┘
                               │              ▲
                               └──────────────┘ (retry loop)
                               │
                               ▼ (tests pass)
                          ┌──────────┐   ┌──────────┐
                          │  Code    │──▶│  Final   │
                          │  Review  │   │  Output  │
                          └──────────┘   └──────────┘
```

```python
class CodeGenerationChain:
    def __init__(self, planner_llm, coder_llm, reviewer_llm,
                 max_debug_iterations: int = 5):
        self.planner = planner_llm
        self.coder = coder_llm
        self.reviewer = reviewer_llm
        self.max_debug_iterations = max_debug_iterations
    
    def execute(self, specification: str) -> dict:
        # Step 1: Parse and plan
        plan = self.planner.invoke(
            f"""Analyze this specification and create a detailed implementation plan.
            Include: file structure, key functions, data models, error handling strategy.
            
            Specification: {specification}"""
        )
        
        # Step 2: Generate code
        code = self.coder.invoke(
            f"""Implement the following plan. Write production-quality Python code.
            
            Plan: {plan}
            
            Requirements:
            - Include type hints
            - Include docstrings
            - Handle edge cases
            - Follow PEP 8"""
        )
        
        # Step 3: Generate tests
        tests = self.coder.invoke(
            f"""Write comprehensive pytest tests for the following code.
            Cover: happy path, edge cases, error conditions.
            
            Code: {code}"""
        )
        
        # Step 4: Debug loop
        for iteration in range(self.max_debug_iterations):
            test_result = self._run_tests(code, tests)
            
            if test_result["all_passed"]:
                break
            
            # Debug: fix code based on test failures
            code = self.coder.invoke(
                f"""Fix the following code based on test failures.
                
                Current code: {code}
                Test code: {tests}
                Failures: {test_result['failures']}
                
                Return only the corrected code."""
            )
        
        # Step 5: Code review
        review = self.reviewer.invoke(
            f"""Review this code for:
            1. Security vulnerabilities
            2. Performance issues
            3. Code quality
            4. Potential bugs
            
            Code: {code}
            Tests: {tests}"""
        )
        
        return {
            "plan": plan,
            "code": code,
            "tests": tests,
            "test_results": test_result,
            "review": review,
            "debug_iterations": iteration + 1
        }
    
    def _run_tests(self, code: str, tests: str) -> dict:
        """Execute tests in sandboxed environment."""
        # Implementation uses subprocess with timeout and resource limits
        ...
```

---

### 1.15.3 Data Analysis Chains

#### Natural Language → SQL → Insight Chain

```python
class NL2SQLAnalysisChain:
    """
    Chain: Question → SQL → Execute → Interpret → Visualize
    """
    
    def __init__(self, llm, db_connection, schema_info: str):
        self.llm = llm
        self.db = db_connection
        self.schema = schema_info
    
    def execute(self, question: str) -> dict:
        # Step 1: Generate SQL
        sql = self.llm.invoke(
            f"""Given the database schema:
            {self.schema}
            
            Generate a SQL query to answer: {question}
            
            Rules:
            - Use only SELECT statements (no mutations)
            - Limit results to 1000 rows
            - Include appropriate JOINs
            - Add comments explaining the query logic
            
            Return only the SQL query."""
        )
        
        # Step 2: Validate and sanitize SQL
        validated_sql = self._validate_sql(sql)
        
        # Step 3: Execute
        results = self.db.execute(validated_sql, timeout=30)
        
        # Step 4: Interpret
        interpretation = self.llm.invoke(
            f"""Analyze these query results and provide insights.
            
            Original question: {question}
            SQL query: {validated_sql}
            Results (first 20 rows): {results[:20]}
            Total rows: {len(results)}
            
            Provide:
            1. Direct answer to the question
            2. Key patterns or trends
            3. Caveats or limitations of the analysis"""
        )
        
        # Step 5: Visualization recommendation
        viz = self.llm.invoke(
            f"""Based on this data analysis, recommend and describe 
            the best visualization.
            
            Data summary: {interpretation}
            Available columns: {[r.keys() for r in results[:1]]}
            
            Specify: chart type, x-axis, y-axis, color encoding, title."""
        )
        
        return {
            "sql": validated_sql,
            "raw_results": results,
            "interpretation": interpretation,
            "visualization": viz
        }
    
    def _validate_sql(self, sql: str) -> str:
        """Enforce read-only, prevent injection."""
        sql_upper = sql.upper().strip()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", 
                      "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
        for keyword in forbidden:
            if keyword in sql_upper:
                raise SecurityError(f"Forbidden SQL keyword: {keyword}")
        
        if not sql_upper.startswith("SELECT"):
            raise SecurityError("Only SELECT queries allowed")
        
        return sql
```

---

### 1.15.4 Content Creation Chains

```
Research → Outline → Draft → Edit → Polish → SEO → Publish

┌──────────┐   ┌──────────┐   ┌──────────┐
│ Research │──▶│ Generate │──▶│  Write   │
│ Topic    │   │ Outline  │   │  Draft   │
│          │   │          │   │          │
│• Web     │   │• Thesis  │   │• Section │
│  search  │   │• Sections│   │  by      │
│• Papers  │   │• Key     │   │  section │
│• Existing│   │  points  │   │          │
│  content │   │          │   │          │
└──────────┘   └──────────┘   └────┬─────┘
                                    │
         ┌──────────────────────────┤
         ▼                          ▼
    ┌──────────┐             ┌──────────┐
    │ Fact     │             │ Style    │
    │ Check    │             │ Edit     │
    └────┬─────┘             └────┬─────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
              ┌──────────┐   ┌──────────┐
              │ Polish   │──▶│ SEO      │
              │ & Format │   │ Optimize │
              └──────────┘   └──────────┘
```

---

### 1.15.5 Customer Service Chains

```python
class CustomerServiceChain:
    """
    Intent → Entities → Retrieve → Generate → Tone → Deliver
    """
    
    def execute(self, message: str, customer_context: dict) -> dict:
        # Step 1: Intent classification
        intent = self.classify_intent(message)
        
        # Step 2: Entity extraction
        entities = self.extract_entities(message)
        
        # Step 3: Route by intent
        if intent["category"] == "billing":
            knowledge = self.retrieve_billing_docs(entities)
        elif intent["category"] == "technical":
            knowledge = self.retrieve_technical_docs(entities)
        elif intent["category"] == "complaint":
            knowledge = self.retrieve_escalation_policy(entities)
            # Trigger escalation check
            if intent["severity"] == "high":
                return self.escalate_to_human(message, customer_context)
        else:
            knowledge = self.retrieve_general_docs(entities)
        
        # Step 4: Generate response
        response = self.generate_response(
            message, intent, entities, knowledge, customer_context
        )
        
        # Step 5: Tone adjustment
        adjusted = self.adjust_tone(
            response,
            customer_sentiment=intent.get("sentiment", "neutral"),
            brand_voice="professional_empathetic"
        )
        
        # Step 6: Quality check
        quality = self.quality_check(adjusted, message, knowledge)
        if quality["score"] < 0.7:
            # Regenerate with quality feedback
            adjusted = self.regenerate_with_feedback(
                message, adjusted, quality["feedback"]
            )
        
        return {
            "response": adjusted,
            "intent": intent,
            "entities": entities,
            "sources": knowledge.get("sources", []),
            "quality_score": quality["score"],
            "escalated": False
        }
```

---

### 1.15.6 Scientific Research Chains

```
Literature Search → Summarize → Gap Analysis → Hypothesis → Experimental Design

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Literature  │─▶│  Summarize   │─▶│    Gap       │
│   Search     │  │  Key Papers  │  │  Analysis    │
│              │  │              │  │              │
│• Semantic    │  │• Per-paper   │  │• What's      │
│  Scholar API │  │  summary     │  │  known       │
│• arXiv       │  │• Cross-paper │  │• What's      │
│• PubMed      │  │  synthesis   │  │  unknown     │
└──────────────┘  └──────────────┘  │• Contradictions│
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐  ┌──────────────┐
                                    │  Hypothesis  │─▶│ Experimental │
                                    │  Generation  │  │   Design     │
                                    │              │  │              │
                                    │• Testable    │  │• Variables   │
                                    │• Novel       │  │• Controls    │
                                    │• Grounded    │  │• Sample size │
                                    │  in evidence │  │• Metrics     │
                                    └──────────────┘  └──────────────┘
```
