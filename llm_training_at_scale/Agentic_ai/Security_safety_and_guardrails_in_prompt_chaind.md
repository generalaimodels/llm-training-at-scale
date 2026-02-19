## 1.14 Security, Safety, and Guardrails in Prompt Chains

### 1.14.1 Threat Model for Prompt Chains

Prompt chains **expand the attack surface** compared to single-prompt systems. Each step is a potential injection point, and inter-step data flow creates **transitive trust violations** where untrusted data from one step propagates as trusted input to subsequent steps.

#### Attack Surface Analysis

```
                    Attack Surface Map
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  User Input ──▶ [Step 1] ──▶ [Step 2] ──▶ [Step 3]    │
│       ▲              │            │            │        │
│       │              ▼            ▼            ▼        │
│  ①Injection    ②Output        ③Retrieved   ④Tool       │
│   at entry      carries        content      execution   │
│                 payload        injection    hijacking    │
│                                                         │
│  Attack Vectors:                                        │
│  ① Direct prompt injection via user input               │
│  ② Poisoned output from Step 1 manipulates Step 2      │
│  ③ Adversarial content in retrieved documents           │
│  ④ Malicious tool calls or code execution               │
│  ⑤ Data exfiltration via output channels                │
│  ⑥ Chain flow hijacking via conditional manipulation    │
└─────────────────────────────────────────────────────────┘
```

#### Formal Threat Taxonomy

**1. Direct Prompt Injection**: Adversary crafts $x_{\text{adv}}$ such that:

$$
\mathcal{C}(x_{\text{adv}}) = y_{\text{attacker-desired}} \neq y_{\text{intended}}
$$

The injection payload is embedded in the user-controlled input at step 1.

**2. Indirect Prompt Injection (via retrieved content)**:

In RAG chains, the retrieval step fetches document $d$ from an external corpus. An adversary poisons $d$ such that:

$$
\text{Step}_k(\text{prompt}_k \oplus d_{\text{poisoned}}) = y_{\text{malicious}}
$$

The chain trusts retrieved content as factual context, but the content contains adversarial instructions.

**3. Transitive Injection (Cross-Step Propagation)**:

Output from step $i$ contains an injection payload that activates at step $j > i$:

$$
y_i = s_i(x_i) = \text{legitimate content} \oplus \text{latent payload}
$$

$$
y_j = s_j(y_i) = \text{payload-manipulated output}
$$

This is particularly dangerous because **intermediate outputs are typically not user-visible**.

**4. Data Exfiltration**:

An adversary causes the chain to encode sensitive information (from system prompts, retrieved documents, or previous conversation) into the output:

$$
\exists f_{\text{decode}} : f_{\text{decode}}(\mathcal{C}(x_{\text{adv}})) = \text{sensitive data}
$$

**5. Chain Hijacking**:

Manipulate conditional branching logic by crafting inputs that force the chain into an unintended execution path:

$$
p_i(x_{\text{adv}}) = \text{true} \implies \text{execute expensive/dangerous branch}
$$

**6. Privilege Escalation**:

Step $k$ has access to a tool (database, API) that earlier steps should not be able to invoke. By injecting instructions at step 1, the adversary causes step $k$ to execute unintended tool calls.

---

### 1.14.2 Defense Mechanisms

#### Input Sanitization Per Chain Step

```python
import re
from typing import Optional

class InputSanitizer:
    """Multi-layer input sanitization for chain steps."""
    
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"```\s*(system|instruction)",
        r"IMPORTANT:\s*override",
        r"\[INST\]",
        r"<<SYS>>",
    ]
    
    def __init__(self, max_length: int = 10000, 
                 allow_code: bool = False):
        self.max_length = max_length
        self.allow_code = allow_code
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """
        Returns (sanitized_text, list_of_warnings).
        """
        warnings = []
        
        # 1. Length enforcement
        if len(text) > self.max_length:
            text = text[:self.max_length]
            warnings.append(f"Input truncated to {self.max_length} chars")
        
        # 2. Injection pattern detection
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                warnings.append(f"Potential injection detected: {pattern.pattern}")
                # Replace rather than block — preserve benign content
                text = pattern.sub("[FILTERED]", text)
        
        # 3. Control character removal
        text = ''.join(
            c for c in text 
            if c.isprintable() or c in '\n\t'
        )
        
        # 4. Code block handling
        if not self.allow_code:
            text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK_REMOVED]', text)
        
        return text, warnings
```

#### Guardrail Chain Pattern

Implement safety checks as **dedicated chain steps** rather than inline logic:

```python
class GuardrailChain:
    """Chain wrapper that adds pre/post safety checks."""
    
    def __init__(self, core_chain, safety_classifier, 
                 content_policy, pii_detector):
        self.core_chain = core_chain
        self.safety_classifier = safety_classifier
        self.content_policy = content_policy
        self.pii_detector = pii_detector
    
    def invoke(self, input_data: dict) -> dict:
        # ═══ PRE-CHAIN SAFETY ═══
        
        # 1. Input safety classification
        safety_result = self.safety_classifier.classify(input_data["text"])
        if safety_result["category"] in ("harmful", "illegal", "explicit"):
            return {
                "response": "I cannot process this request.",
                "blocked": True,
                "reason": safety_result["category"],
                "trace_id": generate_trace_id()
            }
        
        # 2. PII detection and redaction in input
        redacted_input, pii_entities = self.pii_detector.redact(input_data["text"])
        input_data["text"] = redacted_input
        
        # ═══ CORE CHAIN EXECUTION ═══
        result = self.core_chain.invoke(input_data)
        
        # ═══ POST-CHAIN SAFETY ═══
        
        # 3. Output content policy check
        policy_result = self.content_policy.check(result["response"])
        if not policy_result["passes"]:
            return {
                "response": "Response filtered due to content policy.",
                "blocked": True,
                "reason": policy_result["violation"],
                "trace_id": result.get("trace_id")
            }
        
        # 4. Output PII detection (prevent leakage)
        output_pii = self.pii_detector.detect(result["response"])
        if output_pii:
            result["response"] = self.pii_detector.redact(result["response"])[0]
            result["pii_redacted"] = True
        
        # 5. Restore PII placeholders if needed
        if pii_entities:
            result["pii_detected_in_input"] = len(pii_entities)
        
        return result
```

#### Canary Token Injection Detection

Insert **canary tokens** into system prompts to detect extraction attempts:

```python
import hashlib
import time

class CanaryTokenDetector:
    """Detect system prompt extraction via canary tokens."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_canary(self, step_id: str) -> str:
        """Generate a unique canary token for a chain step."""
        raw = f"{self.secret_key}:{step_id}:{int(time.time() // 3600)}"
        token = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"INTERNAL_REF_{token}"
    
    def inject_canary(self, system_prompt: str, step_id: str) -> str:
        """Inject canary into system prompt."""
        canary = self.generate_canary(step_id)
        # Embed canary as a seemingly natural part of the prompt
        injection = f"\n[Internal reference ID: {canary}]\n"
        return system_prompt + injection
    
    def check_output(self, output: str, step_id: str) -> bool:
        """Check if output contains the canary token (indicates leakage)."""
        canary = self.generate_canary(step_id)
        return canary in output
```

#### Principle of Least Privilege Per Chain Step

```python
@dataclass
class StepPermissions:
    """Fine-grained permissions for each chain step."""
    allowed_tools: set[str] = field(default_factory=set)
    allowed_models: set[str] = field(default_factory=set)
    max_tokens: int = 4096
    can_access_internet: bool = False
    can_execute_code: bool = False
    can_read_filesystem: bool = False
    allowed_api_endpoints: set[str] = field(default_factory=set)
    max_cost_usd: float = 0.10
    timeout_seconds: int = 30

class PermissionEnforcer:
    def __init__(self, permissions: dict[str, StepPermissions]):
        self.permissions = permissions
    
    def check(self, step_id: str, action: str, resource: str) -> bool:
        perms = self.permissions.get(step_id)
        if perms is None:
            return False  # Deny by default
        
        if action == "call_tool":
            return resource in perms.allowed_tools
        elif action == "call_model":
            return resource in perms.allowed_models
        elif action == "execute_code":
            return perms.can_execute_code
        elif action == "access_url":
            return perms.can_access_internet
        
        return False  # Deny unknown actions
```

---

### 1.14.3 Safety Patterns

#### Architecture: Defense-in-Depth for Chains

```
User Input
    │
    ▼
┌──────────────────┐
│  Input Shield     │ ← Content classification, injection detection
└────────┬─────────┘
         │ (sanitized input)
         ▼
┌──────────────────┐
│  PII Redactor     │ ← Replace PII with placeholders
└────────┬─────────┘
         │
    ┌────▼────────────────────────────────────────┐
    │           CORE CHAIN (sandboxed)             │
    │  ┌────────┐    ┌────────┐    ┌────────┐     │
    │  │ Step 1 │───▶│ Step 2 │───▶│ Step 3 │     │
    │  └────────┘    └───┬────┘    └────────┘     │
    │                    │                         │
    │              ┌─────▼─────┐                   │
    │              │Checkpoint │ ← Mid-chain       │
    │              │  Safety   │   safety check    │
    │              └───────────┘                   │
    └────────────────────┬────────────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Output Shield    │ ← Content policy, PII leak check
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Audit Logger     │ ← Compliance logging
                └────────┬─────────┘
                         │
                         ▼
                    Final Output
```

#### Intermediate Safety Checkpoints

For long chains, insert safety verification **between critical steps**:

```python
class SafetyCheckpoint:
    """Mid-chain safety verification step."""
    
    def __init__(self, safety_model, policies: list[str]):
        self.safety_model = safety_model
        self.policies = policies
    
    def check(self, intermediate_output: str, chain_context: dict) -> dict:
        """
        Verify intermediate output before it becomes input to next step.
        """
        checks = {
            "contains_harmful_instructions": self._check_harmful(intermediate_output),
            "contains_pii": self._check_pii(intermediate_output),
            "contains_injection_payload": self._check_injection(intermediate_output),
            "within_scope": self._check_scope(intermediate_output, chain_context),
        }
        
        passed = all(not v for v in checks.values())
        
        if not passed:
            failed_checks = [k for k, v in checks.items() if v]
            return {
                "passed": False,
                "failed_checks": failed_checks,
                "action": "halt_chain",
                "sanitized_output": self._sanitize(intermediate_output, failed_checks)
            }
        
        return {"passed": True, "action": "continue"}
```

---

### 1.14.4 Access Control and Governance

#### Role-Based Access Control (RBAC) for Chains

```python
from enum import Flag, auto

class ChainPermission(Flag):
    VIEW = auto()
    EXECUTE = auto()
    MODIFY = auto()
    DEPLOY = auto()
    DELETE = auto()
    VIEW_TRACES = auto()
    APPROVE_DEPLOYMENT = auto()

@dataclass
class ChainACL:
    chain_id: str
    owner: str
    role_permissions: dict[str, ChainPermission]

class ChainGovernance:
    def __init__(self):
        self.acls: dict[str, ChainACL] = {}
        self.deployment_approvers: set[str] = set()
    
    def can_deploy(self, chain_id: str, user: str) -> bool:
        acl = self.acls.get(chain_id)
        if acl is None:
            return False
        user_perms = acl.role_permissions.get(user, ChainPermission(0))
        return ChainPermission.DEPLOY in user_perms
    
    def request_deployment(self, chain_id: str, requester: str,
                          chain_version: str, test_results: dict) -> dict:
        """
        Deployment requires:
        1. All tests passing
        2. Safety evaluation passing
        3. Approval from designated approver
        """
        checks = {
            "tests_passing": test_results.get("all_passed", False),
            "safety_eval": test_results.get("safety_score", 0) > 0.95,
            "has_permission": self.can_deploy(chain_id, requester),
            "version_tagged": chain_version is not None,
        }
        
        if all(checks.values()):
            return {
                "status": "approved",
                "deployment_id": str(uuid4()),
                "approved_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "rejected",
                "failed_checks": [k for k, v in checks.items() if not v]
            }
```

#### Compliance Logging

```python
class ComplianceLogger:
    """GDPR/HIPAA-aware chain execution logging."""
    
    def __init__(self, log_store, retention_days: int = 90):
        self.log_store = log_store
        self.retention_days = retention_days
    
    def log_execution(self, trace_id: str, chain_id: str,
                      user_id: str, input_data: dict,
                      output_data: dict, pii_detected: bool):
        
        record = {
            "trace_id": trace_id,
            "chain_id": chain_id,
            "user_id": self._hash_user_id(user_id),  # Pseudonymize
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),  # Hash, don't store raw input
            "output_length": len(str(output_data)),
            "pii_detected": pii_detected,
            "retention_expiry": (
                datetime.utcnow() + timedelta(days=self.retention_days)
            ).isoformat(),
            # Do NOT log raw input/output for compliance
            # Log only metadata sufficient for audit
        }
        
        self.log_store.write(record)
    
    def handle_deletion_request(self, user_id: str):
        """GDPR right to erasure — delete all traces for user."""
        hashed_id = self._hash_user_id(user_id)
        self.log_store.delete_by_user(hashed_id)
        return {"status": "deleted", "user_hash": hashed_id}
```
