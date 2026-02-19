

# Chapter 2: Tools — Comprehensive SOTA Reference

## 2.4 Understanding the Model Context Protocol (MCP)

---

### 2.4.1 The "N × M" Integration Problem and the Need for Standardization

#### 2.4.1.1 The Combinatorial Explosion Problem

Every modern AI system that aspires to agentic behavior must interface with external tools—databases, APIs, file systems, code interpreters, search engines, and domain-specific services. The fundamental integration challenge arises from a simple combinatorial reality.

**Formal Problem Statement.** Let $N$ denote the number of distinct AI applications (hosts/clients)—Claude Desktop, ChatGPT, a custom LangChain agent, an IDE copilot, a customer-support bot—and let $M$ denote the number of distinct tool/service providers—GitHub, Slack, PostgreSQL, a weather API, a code sandbox. Without any standardization layer, connecting every application to every tool requires building a bespoke, point-to-point connector for each $(application_i, tool_j)$ pair. The total number of unique integration artifacts is:

$$
C_{\text{naive}} = N \times M
$$

Each connector must handle:

1. **Schema translation** — converting the tool's native API schema into the format the AI application's function-calling interface expects.
2. **Authentication plumbing** — managing API keys, OAuth tokens, or session cookies specific to that tool for that application.
3. **Error mapping** — translating tool-specific error codes into the host application's error-handling paradigm.
4. **Data serialization** — marshalling inputs and unmarshalling outputs between the AI application's internal representation and the tool's wire format.
5. **Lifecycle management** — handling connection setup, health checking, reconnection, and teardown.

**Growth dynamics.** As either $N$ or $M$ increases, the integration surface grows multiplicatively:

$$
\frac{\partial C_{\text{naive}}}{\partial N} = M, \qquad \frac{\partial C_{\text{naive}}}{\partial M} = N
$$

Adding a single new AI application requires $M$ new connectors; adding a single new tool requires $N$ new connectors. In a realistic enterprise with $N = 10$ applications and $M = 50$ services, this yields $500$ bespoke connectors—each requiring independent development, testing, versioning, and maintenance.

**Maintenance amplification.** When tool $j$ updates its API (e.g., a breaking schema change, endpoint deprecation, or authentication protocol migration), all $N$ connectors to that tool must be updated simultaneously. Conversely, when application $i$ upgrades its function-calling format (e.g., OpenAI migrating from `functions` to `tools` parameter format), all $M$ connectors from that application must be refactored. The total maintenance events per API change scale as $O(N)$ or $O(M)$ respectively, making the system operationally fragile at scale.

**Analogy: USB standardization.** Prior to the Universal Serial Bus (USB) standard, peripheral manufacturers had to build device-specific interfaces for each computer platform (serial ports, parallel ports, proprietary connectors). USB collapsed this $N \times M$ matrix into $N + M$: each computer implements one USB host controller, each peripheral implements one USB device interface. MCP aims to achieve an identical architectural simplification for AI-tool connectivity.

---

#### 2.4.1.2 Pre-MCP Integration Landscape

Before MCP, the AI-tool integration ecosystem consisted of multiple incompatible, partially overlapping approaches, each creating its own integration silo.

**Provider-Specific Function Calling Formats**

Each LLM provider defined its own proprietary schema for tool/function specification:

| Provider | Format Name | Schema Location | Key Differences |
|----------|------------|-----------------|-----------------|
| OpenAI | `tools` array with `function` type | Request body parameter | Uses `strict` mode for constrained decoding; `function_call` → `tool_choice` migration |
| Anthropic | `tools` array | Request body parameter | Uses `cache_control` annotations; `tool_use` content blocks |
| Google (Gemini) | `function_declarations` in `tools` | Nested within `tool_config` | Uses `function_calling_config` with mode enum |
| Cohere | `tools` array | Request body parameter | Supports `parameter_definitions` with custom types |
| Mistral | `tools` array | OpenAI-compatible but divergent edge cases | Partial OpenAI compatibility with subtle schema deviations |

Each format prescribes slightly different JSON structures for identical semantic content: the tool's name, description, and parameter schema. A tool developer who wants their service accessible across all providers must maintain separate schema definitions and result-parsing logic for each.

**Framework-Specific Tool Abstractions**

Orchestration frameworks introduced their own abstraction layers, further fragmenting the ecosystem:

- **LangChain** — `BaseTool` class with `_run()` / `_arun()` methods; `ToolMessage` type; tool schemas derived via Pydantic model introspection.
- **LlamaIndex** — `FunctionTool` / `QueryEngineTool` wrappers; tools embedded within `AgentRunner` abstractions.
- **AutoGen** — `register_for_llm()` / `register_for_execution()` decorators; tool registration tied to agent conversation protocols.
- **CrewAI** — `@tool` decorator; tool definitions coupled to crew/task configuration YAML.
- **Semantic Kernel (Microsoft)** — `KernelFunction` with plugin architecture; OpenAPI-based tool import.

Each framework defines its own tool interface, registration mechanism, invocation protocol, and result format. A tool written for LangChain cannot be used in CrewAI without a translation layer.

**Impedance Mismatch**

The term *impedance mismatch*—borrowed from electrical engineering where it describes the inefficiency of power transfer between circuits of differing impedance—precisely characterizes this situation. Translating between tool formats requires:

- Schema transformers (e.g., LangChain `ToolMessage` → OpenAI `tool` response format)
- Result adapters (e.g., converting Anthropic `tool_result` content blocks into LlamaIndex `ToolOutput`)
- Error code mapping tables
- Authentication credential forwarding across abstraction boundaries

Each translation layer introduces potential information loss, subtle behavioral differences, and additional failure modes.

**Maintenance Burden**

The total engineering cost can be modeled as:

$$
\text{Cost}_{\text{total}} = \sum_{i=1}^{N} \sum_{j=1}^{M} \left( c_{\text{dev}}^{(i,j)} + c_{\text{test}}^{(i,j)} + c_{\text{maintain}}^{(i,j)} \cdot T \right)
$$

where $c_{\text{dev}}^{(i,j)}$ is the one-time development cost, $c_{\text{test}}^{(i,j)}$ is the testing cost, $c_{\text{maintain}}^{(i,j)}$ is the per-period maintenance cost, and $T$ is the time horizon. This sum scales as $O(N \cdot M \cdot T)$, making it economically unsustainable for large ecosystems.

---

#### 2.4.1.3 The Standardization Solution

MCP eliminates the combinatorial explosion by introducing a single, shared protocol interface between applications and tools.

**Architectural Reduction**

With MCP, each of the $N$ applications implements exactly one MCP client, and each of the $M$ tools implements exactly one MCP server. The total number of integration artifacts becomes:

$$
C_{\text{MCP}} = N + M
$$

The improvement ratio is:

$$
\frac{C_{\text{naive}}}{C_{\text{MCP}}} = \frac{N \times M}{N + M}
$$

For $N = M = k$, this simplifies to $\frac{k^2}{2k} = \frac{k}{2}$, meaning the standardized approach is $\frac{k}{2}$-times more efficient. For $N = 10, M = 50$:

$$
\frac{10 \times 50}{10 + 50} = \frac{500}{60} \approx 8.3\times \text{ reduction}
$$

**Protocol-Level Interoperability**

Any MCP-compliant client can connect to any MCP-compliant server without custom integration code. This is achieved through:

1. **Uniform wire protocol** — JSON-RPC 2.0 over standardized transports (stdio, HTTP+SSE, Streamable HTTP).
2. **Standardized primitive types** — Tools, Resources, Prompts, Sampling, Elicitation defined with fixed schema contracts.
3. **Capability negotiation** — clients and servers dynamically discover each other's supported features during connection initialization.
4. **Schema-driven tool discovery** — tools self-describe via JSON Schema input specifications, enabling any client to construct valid invocations.

**Open Specification**

MCP is published as an open specification (not proprietary to any single vendor), enabling:

- Community-driven evolution via public RFC process
- Vendor-neutral governance
- Multiple independent implementations (reference SDKs exist in TypeScript and Python)
- Third-party conformance testing and validation
- Ecosystem-wide interoperability guarantees

---

#### 2.4.1.4 Historical Precedents for Standardization

MCP does not exist in a vacuum; it draws from a lineage of successful standardization protocols that solved analogous $N \times M$ problems in other domains.

**Language Server Protocol (LSP)**

| Aspect | LSP | MCP |
|--------|-----|-----|
| **Problem** | $N$ IDEs × $M$ programming languages = $N \times M$ language support plugins | $N$ AI apps × $M$ tools = $N \times M$ tool connectors |
| **Solution** | Standardized protocol between editors and language servers | Standardized protocol between AI hosts and tool servers |
| **Wire Format** | JSON-RPC 2.0 | JSON-RPC 2.0 |
| **Architecture** | Client (IDE) ↔ Server (language analyzer) | Client (AI host) ↔ Server (tool provider) |
| **Capability Negotiation** | `initialize` handshake with `capabilities` exchange | `initialize` handshake with `capabilities` exchange |
| **Key Innovation** | Separated editor UX from language analysis | Separated AI orchestration from tool implementation |

MCP's design is directly inspired by LSP. The `initialize` / `initialized` handshake pattern, the capability negotiation structure, and the use of JSON-RPC 2.0 as the wire format are all borrowed from LSP. The key conceptual difference: LSP enables code intelligence features (completion, diagnostics, refactoring) across editors, while MCP enables tool invocation across AI applications.

**ODBC/JDBC**

Open Database Connectivity (ODBC) and Java Database Connectivity (JDBC) standardized database access across applications and database engines. Before ODBC, each application needed database-specific drivers. ODBC introduced:

- A driver manager as the mediation layer (analogous to the MCP host)
- Standardized SQL wire protocol (analogous to JSON-RPC in MCP)
- Driver-based extensibility (analogous to MCP servers)

MCP borrows the concept of **driver-based extensibility**: just as a new database only needs one ODBC driver to be accessible by all ODBC-compliant applications, a new tool only needs one MCP server to be accessible by all MCP-compliant hosts.

**OpenAPI / Swagger**

OpenAPI provides a machine-readable description of REST API endpoints, enabling automatic client generation, documentation, and testing. MCP's tool definition schema (name, description, JSON Schema input) serves an analogous role: it provides a machine-readable tool description that enables AI models to understand tool capabilities and construct valid invocations.

However, MCP goes beyond OpenAPI in several ways:

- **Bidirectional communication** — OpenAPI describes one-way client→server API calls; MCP supports server→client communication (sampling, elicitation, notifications).
- **Stateful sessions** — OpenAPI describes stateless REST endpoints; MCP maintains stateful sessions with capability negotiation.
- **AI-native design** — MCP tool descriptions include `annotations` (destructive, idempotent, read-only hints) specifically designed for AI model reasoning, which OpenAPI does not natively provide.
- **Multi-primitive model** — OpenAPI describes only API endpoints; MCP defines multiple primitive types (tools, resources, prompts) with different access control semantics.

---

### 2.4.2 Core Architectural Components: Hosts, Clients, and Servers

#### 2.4.2.1 Architecture Overview

The MCP architecture follows a three-tier model with strict separation of concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                      HOST APPLICATION                        │
│  (Claude Desktop, IDE Copilot, Custom Agent, etc.)           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  MCP Client  │  │  MCP Client  │  │  MCP Client  │       │
│  │  (session 1) │  │  (session 2) │  │  (session 3) │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│  ┌──────┴─────────────────┴─────────────────┴──────┐         │
│  │            Host Orchestration Layer              │         │
│  │  (Security policy, consent UI, model routing)    │         │
│  └──────────────────────────────────────────────────┘         │
└─────────┼─────────────────┼─────────────────┼────────────────┘
          │ stdio           │ HTTP+SSE        │ Streamable HTTP
   ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐
   │  MCP Server  │  │  MCP Server  │  │  MCP Server  │
   │  (local:     │  │  (remote:    │  │  (remote:    │
   │   filesystem)│  │   GitHub API)│  │   database)  │
   └──────────────┘  └──────────────┘  └──────────────┘
```

**Key architectural invariants:**

1. **One client per server** — Each MCP client maintains exactly one session with exactly one MCP server. This 1:1 mapping ensures session isolation, simplifies state management, and provides security boundaries.

2. **Many clients per host** — A single host application can instantiate multiple MCP clients, each connecting to a different server. The host orchestrates across all clients.

3. **Host as trust root** — The host application is the ultimate authority for security decisions: which servers to connect to, which tool calls to approve, what data to expose, and what sampling requests to honor.

4. **Server as capability provider** — Servers expose capabilities (tools, resources, prompts) but do not initiate unsolicited actions. The two exceptions—sampling and elicitation—are explicitly server-initiated *requests* that the host may approve or deny.

---

#### 2.4.2.2 Host

**Definition.** The host is the user-facing AI application that orchestrates the end-to-end interaction between the user, the language model, and external tools. Examples include Claude Desktop, VS Code Copilot extensions, custom agentic applications built with orchestration frameworks, and enterprise AI platforms.

**Responsibilities:**

| Responsibility | Description |
|---------------|-------------|
| **User Interaction** | Rendering chat interfaces, displaying tool results, managing conversation history, and presenting consent dialogs for tool invocations |
| **Model Invocation** | Sending prompts to the LLM (local or API-based), managing context windows, and handling model responses including tool-call decisions |
| **Client Orchestration** | Instantiating MCP clients, managing their lifecycles, routing tool calls to the appropriate client, and aggregating results |
| **Security Policy Enforcement** | Deciding which MCP servers to connect to, what permissions to grant each server, whether to approve tool calls (especially destructive ones), and what data to expose via sampling |
| **Context Assembly** | Aggregating tool descriptions from all connected servers, formatting them for the model's function-calling interface, and injecting tool results back into the conversation context |
| **Consent Management** | Presenting human-in-the-loop approval dialogs for sensitive operations, managing user preferences for auto-approval of low-risk actions |

**Host as the trust boundary.** This is the most critical architectural principle. The host is the only component that:

- Has direct access to the user (can request consent)
- Has direct access to the model (controls what context the model sees)
- Knows the full set of connected servers (can enforce cross-server policies)
- Can enforce rate limits, spending caps, and operational boundaries

Servers cannot bypass the host's security policies. Even when a server requests sampling (asking the host to run LLM inference), the host retains full control over whether to honor the request, what model to use, and whether to modify the request before execution.

---

#### 2.4.2.3 Client

**Definition.** An MCP client is a protocol-layer component that lives inside the host application and manages a single, stateful connection to one MCP server. The client handles all protocol mechanics: transport management, JSON-RPC message serialization/deserialization, capability negotiation, and session state tracking.

**Responsibilities:**

| Responsibility | Description |
|---------------|-------------|
| **Protocol Negotiation** | Executing the `initialize` → `initialized` handshake, exchanging protocol versions and capabilities |
| **Capability Exchange** | Advertising client capabilities (e.g., support for sampling, roots, elicitation) and recording server capabilities (e.g., which primitives the server supports) |
| **Message Routing** | Serializing outbound requests (`tools/call`, `resources/read`, etc.) and dispatching inbound responses/notifications to appropriate handlers |
| **Lifecycle Management** | Managing connection establishment, health monitoring (ping/pong), reconnection after failures, and graceful shutdown |
| **Session State** | Maintaining the set of negotiated capabilities, tracking outstanding request IDs, managing subscription state for resources |

**Isolation property.** Each client is isolated from other clients within the same host. This design ensures:

- **Fault isolation** — If one MCP server crashes, only its corresponding client is affected; other clients continue operating normally.
- **Security isolation** — A compromised server cannot access data flowing through other client connections.
- **State isolation** — Each client maintains independent session state; there is no shared mutable state between clients.

**Client vs. Host distinction.** The client is a protocol-level construct; the host is an application-level construct. The host contains business logic (user interaction, model invocation, policy enforcement), while the client contains only protocol mechanics. This separation enables:

- Reusable MCP client libraries (SDK implementations) shared across different host applications
- Host-specific customization without modifying protocol handling
- Testability: clients can be tested against mock servers independently of host logic

---

#### 2.4.2.4 Server

**Definition.** An MCP server is a lightweight process or service that exposes tools, resources, and/or prompts to MCP clients. The server wraps existing functionality—an API, a database, a filesystem, a computation engine—behind the standardized MCP interface.

**Responsibilities:**

| Responsibility | Description |
|---|---|
| **Capability Advertisement** | Responding to `initialize` with the set of supported primitives (tools, resources, prompts, etc.) |
| **Tool Exposure** | Responding to `tools/list` with tool definitions; executing `tools/call` requests and returning structured results |
| **Resource Exposure** | Responding to `resources/list` and `resources/read`; managing subscriptions and sending update notifications |
| **Prompt Exposure** | Responding to `prompts/list` and `prompts/get` with parameterized prompt templates |
| **Request Handling** | Validating incoming requests against declared schemas, executing the requested operation, and formatting results per the MCP content model |

**Server deployment topologies:**

**Local servers (stdio transport):**

```
Host Process
  ├── MCP Client ──stdio──▶ MCP Server (child process)
  │                         ├── reads from stdin
  │                         └── writes to stdout
  └── (server runs as subprocess with inherited permissions)
```

- The host spawns the server as a child process.
- Communication occurs over the child process's stdin/stdout streams.
- The server inherits the host process's filesystem permissions and environment variables.
- Use cases: local file operations, code execution, development tools.
- Advantages: zero network configuration, low latency, simple security model (process-level sandboxing).
- Limitations: cannot serve multiple hosts; tightly coupled to host process lifecycle.

**Remote servers (HTTP+SSE or Streamable HTTP transport):**

```
Host Process                              Network
  ├── MCP Client ──HTTP POST──▶ ┌──────────────────┐
  │              ◀──SSE stream── │  MCP Server      │
  │                              │  (remote service) │
  └──                            └──────────────────┘
```

- The server runs as an independent network service (potentially on a different machine, in a container, or in the cloud).
- Client → Server: HTTP POST requests carrying JSON-RPC messages.
- Server → Client: Server-Sent Events (SSE) stream for push notifications and streaming responses.
- Use cases: shared tool services, cloud-hosted APIs, multi-tenant environments.
- Advantages: can serve multiple hosts simultaneously, independent scaling, remote deployment.
- Requirements: network configuration, TLS, authentication, CORS handling.

**Server lifecycle state machine:**

$$
\text{Uninitialized} \xrightarrow{\texttt{initialize}} \text{Initializing} \xrightarrow{\texttt{initialized}} \text{Operational} \xrightarrow{\texttt{shutdown / disconnect}} \text{Terminated}
$$

| State | Behavior |
|-------|----------|
| **Uninitialized** | Server is started but has not received `initialize`. Rejects all requests except `initialize`. |
| **Initializing** | Server has received `initialize`, is preparing capabilities. Sends `initialize` response with capabilities. |
| **Operational** | Server has received `initialized` notification. Accepts all supported request types. |
| **Terminated** | Connection closed. Server releases resources and may exit. |

---

#### 2.4.2.5 Capability Negotiation

The capability negotiation mechanism ensures that clients and servers can interoperate even when they support different subsets of the protocol's features.

**Negotiation flow:**

```
Client                                    Server
  │                                         │
  │──── initialize ────────────────────────▶│
  │     {                                   │
  │       protocolVersion: "2025-06-18",    │
  │       capabilities: {                   │
  │         sampling: {},                   │
  │         roots: { listChanged: true }    │
  │       },                                │
  │       clientInfo: {                     │
  │         name: "MyAgent", version: "1.0" │
  │       }                                 │
  │     }                                   │
  │                                         │
  │◀─── initialize response ───────────────│
  │     {                                   │
  │       protocolVersion: "2025-06-18",    │
  │       capabilities: {                   │
  │         tools: { listChanged: true },   │
  │         resources: {                    │
  │           subscribe: true,              │
  │           listChanged: true             │
  │         },                              │
  │         prompts: { listChanged: true }  │
  │       },                                │
  │       serverInfo: {                     │
  │         name: "GitHubServer",           │
  │         version: "2.1"                  │
  │       }                                 │
  │     }                                   │
  │                                         │
  │──── initialized (notification) ────────▶│
  │                                         │
  │     [Session is now operational]        │
```

**Capability categories:**

*Client capabilities* (advertised by the client, consumed by the server):

| Capability | Meaning |
|-----------|---------|
| `sampling` | Client supports `sampling/createMessage` requests from the server |
| `roots` | Client provides filesystem roots; `listChanged` indicates dynamic root updates |
| `elicitation` | Client supports `elicitation/create` requests from the server |

*Server capabilities* (advertised by the server, consumed by the client):

| Capability | Meaning |
|-----------|---------|
| `tools` | Server exposes callable tools; `listChanged` indicates dynamic tool set updates |
| `resources` | Server exposes readable resources; `subscribe` enables change subscriptions; `listChanged` indicates dynamic resource set updates |
| `prompts` | Server exposes prompt templates; `listChanged` indicates dynamic prompt set updates |
| `logging` | Server supports structured log emission |

**Progressive capability discovery.** After the handshake, the client adapts its behavior based on the server's declared capabilities:

- If the server does not declare `tools`, the client will not send `tools/list` or `tools/call`.
- If the server declares `resources.subscribe: true`, the client knows it can register for resource change notifications.
- If the client does not declare `sampling`, the server will not attempt to send `sampling/createMessage`.

**Version negotiation.** The `protocolVersion` field follows a date-based versioning scheme (e.g., `"2025-06-18"`). The negotiation rule:

- The client proposes the latest protocol version it supports.
- The server responds with a version it supports that is ≤ the client's proposed version.
- If no compatible version exists, the server rejects the connection.

This ensures **backward compatibility**: a newer client can connect to an older server (which will respond with an older protocol version), and the client degrades gracefully to the older feature set.

**Forward compatibility.** Unknown capabilities in the negotiation response are ignored (not treated as errors). This allows older clients to connect to newer servers that advertise capabilities the client does not understand—the client simply does not use those capabilities.

---

### 2.4.3 The Communication Layer: JSON-RPC, Transports, and Message Types

#### 2.4.3.1 JSON-RPC 2.0 as the Wire Protocol

MCP adopts JSON-RPC 2.0 as its wire protocol—the format and semantics of individual messages exchanged between client and server.

**Why JSON-RPC 2.0?**

| Property | Benefit for MCP |
|----------|----------------|
| **Simplicity** | Minimal specification (4 pages); trivial to implement in any language |
| **Language agnosticism** | JSON is universally supported; no language-specific serialization dependencies |
| **Stateless message format** | Each message is self-contained; transport-agnostic |
| **Wide tooling support** | Mature libraries in Python, TypeScript, Go, Rust, Java, etc. |
| **Proven track record** | Used by LSP, Ethereum JSON-RPC, and numerous other protocols |
| **Request-response + notification model** | Supports both bidirectional RPC and fire-and-forget notifications |

**Message structures:**

**Request** (expects a response):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_github",
    "arguments": {
      "query": "MCP specification",
      "max_results": 10
    }
  }
}
```

- `jsonrpc`: Protocol version identifier (always `"2.0"`)
- `id`: Request identifier (integer or string); used to correlate responses with requests
- `method`: The RPC method name (MCP defines specific method names for each operation)
- `params`: Method-specific parameters (object or array)

**Response** (success):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 3 repositories matching 'MCP specification'..."
      }
    ],
    "isError": false
  }
}
```

- `id`: Matches the request's `id` for correlation
- `result`: Method-specific result payload

**Error response:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params: 'max_results' must be a positive integer",
    "data": {
      "field": "max_results",
      "received": -5
    }
  }
}
```

- `error.code`: Numeric error code (JSON-RPC defines standard codes; MCP may extend)
- `error.message`: Human-readable error description
- `error.data`: Optional structured error details

**Notification** (no response expected):

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed",
  "params": {}
}
```

- No `id` field — this is the distinguishing characteristic of notifications
- The sender does not expect or wait for a response
- Used for events: tool list changes, resource updates, progress updates, cancellation

**Standard JSON-RPC error codes used by MCP:**

| Code | Name | Meaning |
|------|------|---------|
| $-32700$ | Parse error | Server received invalid JSON |
| $-32600$ | Invalid request | JSON is valid but not a valid JSON-RPC request |
| $-32601$ | Method not found | The requested method does not exist or is not supported |
| $-32602$ | Invalid params | Method parameters are invalid (type mismatch, missing required fields) |
| $-32603$ | Internal error | Unspecified server-side error during method execution |

---

#### 2.4.3.2 Transport Mechanisms

The transport layer defines *how* JSON-RPC messages are physically delivered between client and server. MCP defines three transport mechanisms, each optimized for different deployment scenarios.

**Stdio Transport**

```
┌────────────┐     stdin (JSON-RPC messages, newline-delimited)
│   Host     │ ──────────────────────────────────────────────▶ ┌─────────────┐
│ (MCP Client│ ◀────────────────────────────────────────────── │ MCP Server  │
│  inside)   │     stdout (JSON-RPC messages, newline-delimited│ (child proc)│
└────────────┘                                                 └─────────────┘
               stderr: reserved for diagnostic logging (not protocol messages)
```

**Mechanism:**
- The host spawns the MCP server as a child process (e.g., `subprocess.Popen` in Python, `child_process.spawn` in Node.js).
- Messages from client to server are written to the child process's `stdin`.
- Messages from server to client are read from the child process's `stdout`.
- Each message is a single line of JSON (newline-delimited JSON, NDJSON).
- `stderr` is reserved for human-readable diagnostic logs; it is NOT part of the protocol.

**Message framing:**
```
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n
{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}\n
```

Each JSON-RPC message is terminated by a newline character (`\n`). The receiver reads line by line, parsing each line as a complete JSON-RPC message.

**Characteristics:**
- **Latency**: Minimal (inter-process communication, no network stack).
- **Security**: Server inherits the host process's user permissions, environment variables, and filesystem access. No network exposure.
- **Lifecycle coupling**: Server lifetime is bound to the host process. When the host exits, the server's stdin is closed, signaling shutdown.
- **Concurrency**: Single connection; multiplexing achieved via JSON-RPC `id` correlation.
- **Use cases**: Local development tools, filesystem access, code execution sandboxes, desktop applications.

**HTTP + Server-Sent Events (SSE) Transport**

```
┌────────────┐                                    ┌─────────────┐
│   Host     │ ── HTTP POST (JSON-RPC request) ──▶│ MCP Server  │
│ (MCP Client│ ◀── HTTP Response (JSON-RPC resp)──│ (remote svc)│
│  inside)   │                                    │             │
│            │ ◀── SSE stream (notifications) ────│             │
└────────────┘                                    └─────────────┘
```

**Mechanism:**
- Client → Server communication: Standard HTTP POST requests, each carrying a JSON-RPC message in the request body.
- Server → Client communication (push): Server-Sent Events (SSE) stream. The client opens a long-lived HTTP connection to the server's SSE endpoint; the server pushes notifications and streaming results as SSE events.
- Request-response correlation: The client sends a POST request and receives the JSON-RPC response in the HTTP response body.

**SSE event format:**
```
event: message
data: {"jsonrpc":"2.0","method":"notifications/resources/updated","params":{"uri":"file:///data.csv"}}

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"abc","progress":50,"total":100}}
```

**Characteristics:**
- **Latency**: Network-dependent; typically 1-100ms for cloud services.
- **Security**: Supports TLS (HTTPS), authentication headers (Bearer tokens, API keys), CORS policies.
- **Lifecycle**: Server is independently deployed; connection lifecycle managed by HTTP keep-alive and SSE reconnection.
- **Concurrency**: Multiple concurrent POST requests supported; SSE stream handles all server-initiated messages.
- **Use cases**: Cloud-hosted tool services, shared team tools, multi-tenant deployments.

**Streamable HTTP Transport (Newer Specification)**

The Streamable HTTP transport unifies the request-response and streaming patterns into a single HTTP-based mechanism:

- Client sends HTTP POST requests with JSON-RPC messages.
- Server can respond with either:
  - A single JSON-RPC response (standard HTTP response body)
  - A streaming response (chunked transfer encoding or SSE within the response body)
- Server-initiated messages are delivered via a dedicated SSE endpoint or via response streaming.
- Supports session affinity via `Mcp-Session-Id` header for stateful connections.

**Characteristics:**
- **Simplification**: Eliminates the need for separate SSE endpoint management.
- **Session management**: Explicit session IDs enable load-balanced deployments with session affinity.
- **Compatibility**: Works with standard HTTP infrastructure (proxies, load balancers, CDNs).
- **Use cases**: Production enterprise deployments, Kubernetes-based architectures.

**Transport comparison matrix:**

| Feature | stdio | HTTP+SSE | Streamable HTTP |
|---------|-------|----------|-----------------|
| Deployment | Local subprocess | Remote service | Remote service |
| Latency | ~μs | ~ms | ~ms |
| Authentication | Process-level | HTTP headers (OAuth, API keys) | HTTP headers (OAuth, API keys) |
| Encryption | N/A (local) | TLS | TLS |
| Server push | stdout | SSE stream | SSE / chunked response |
| Session management | Implicit (process lifetime) | SSE connection | Explicit session ID |
| Multi-tenant | No | Yes | Yes |
| Scalability | Single host | Multiple clients | Multiple clients + load balancing |

---

#### 2.4.3.3 Message Types in Detail

MCP defines a comprehensive set of JSON-RPC methods organized by functional category. Each method has a specific direction (client→server or server→client) and semantics (request or notification).

**Initialization Messages**

The initialization handshake establishes the session and negotiates capabilities. It MUST be the first exchange on any new connection.

| Step | Direction | Type | Method | Purpose |
|------|-----------|------|--------|---------|
| 1 | Client → Server | Request | `initialize` | Client proposes protocol version, advertises client capabilities, sends client metadata |
| 2 | Server → Client | Response | (response to `initialize`) | Server confirms protocol version, advertises server capabilities, sends server metadata |
| 3 | Client → Server | Notification | `initialized` | Client confirms initialization is complete; session enters operational state |

**Detailed `initialize` request payload:**

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "sampling": {},
      "roots": {
        "listChanged": true
      },
      "elicitation": {}
    },
    "clientInfo": {
      "name": "CustomAgentHost",
      "version": "3.2.1"
    }
  }
}
```

**Detailed `initialize` response payload:**

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "prompts": {
        "listChanged": true
      },
      "logging": {}
    },
    "serverInfo": {
      "name": "GitHubMCPServer",
      "version": "1.4.0"
    }
  }
}
```

**Tool Messages**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `tools/list` | Client → Server | Request | Enumerate all tools the server exposes |
| `tools/call` | Client → Server | Request | Invoke a specific tool with provided arguments |
| `notifications/tools/list_changed` | Server → Client | Notification | Signal that the tool set has changed (tools added/removed) |

`tools/list` supports pagination via an optional `cursor` parameter. The response includes a `nextCursor` field if more tools are available:

```json
// Request
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"cursor":"page2token"}}

// Response
{
  "jsonrpc":"2.0","id":2,
  "result":{
    "tools":[...],
    "nextCursor":"page3token"
  }
}
```

**Resource Messages**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `resources/list` | Client → Server | Request | Enumerate available resources |
| `resources/read` | Client → Server | Request | Retrieve content of a specific resource by URI |
| `resources/subscribe` | Client → Server | Request | Subscribe to changes on a specific resource |
| `resources/unsubscribe` | Client → Server | Request | Cancel a resource subscription |
| `resources/templates/list` | Client → Server | Request | List parameterized resource URI templates |
| `notifications/resources/list_changed` | Server → Client | Notification | Resource set has changed |
| `notifications/resources/updated` | Server → Client | Notification | A subscribed resource's content has changed |

**Prompt Messages**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `prompts/list` | Client → Server | Request | Enumerate available prompt templates |
| `prompts/get` | Client → Server | Request | Retrieve and expand a prompt template with arguments |
| `notifications/prompts/list_changed` | Server → Client | Notification | Prompt set has changed |

**Sampling Messages (Server → Client)**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `sampling/createMessage` | Server → Client | Request | Server requests the host to perform LLM inference |

This is a reverse-direction request: the server asks the client to do something (invoke the LLM). The host retains full authority to approve, modify, or reject the request.

**Elicitation Messages (Server → Client)**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `elicitation/create` | Server → Client | Request | Server requests structured user input |

**Lifecycle Messages**

| Method | Direction | Type | Purpose |
|--------|-----------|------|---------|
| `ping` | Either | Request | Health check; expects `pong` response (empty `result: {}`) |
| `notifications/cancelled` | Either | Notification | Cancel a previously-issued request by `id` |
| `notifications/progress` | Server → Client | Notification | Report progress on a long-running operation |

**Progress notification structure:**

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "op-12345",
    "progress": 75,
    "total": 100,
    "message": "Processing file 75 of 100..."
  }
}
```

The `progressToken` is provided by the client in the original request's `_meta.progressToken` field. Only requests that include a progress token will receive progress notifications.

---

#### 2.4.3.4 Session Management and Stateful Connections

**Session establishment.** A session begins with the `initialize` / `initialized` handshake and persists until either party disconnects. During the session:

- Both parties maintain the negotiated capability set.
- The client tracks outstanding request IDs (requests sent but not yet responded to).
- Resource subscriptions remain active.
- The server may cache per-session state (e.g., authenticated user identity, workspace context).

**Session identification.** For HTTP-based transports, the Streamable HTTP specification introduces an explicit `Mcp-Session-Id` header. For stdio transport, the session is implicitly defined by the process lifetime.

**Reconnection semantics.** If the transport connection drops:

- **stdio**: The child process has likely terminated. The host must respawn the server process and re-initialize the session from scratch (full `initialize` handshake, re-list tools/resources, re-establish subscriptions).
- **HTTP+SSE**: The SSE connection may drop due to network issues. The client should:
  1. Attempt to reconnect to the SSE endpoint.
  2. If the server supports session resumption, include the previous session ID.
  3. If session state is lost, perform a full re-initialization.
- **Streamable HTTP**: Session affinity via `Mcp-Session-Id` enables transparent reconnection to load-balanced server instances.

**Timeout policies:**

- **Request timeout**: The client should implement per-request timeouts. If a response is not received within the timeout, the client may:
  - Send a `notifications/cancelled` notification for the outstanding request.
  - Mark the request as failed and propagate the error to the host.
- **Session idle timeout**: The server may disconnect sessions that have been idle beyond a configured threshold. Clients should send periodic `ping` requests as keep-alives.
- **Initialization timeout**: If the server does not respond to `initialize` within a reasonable time (e.g., 30 seconds), the client should abort the connection.

**Concurrent request handling.** JSON-RPC 2.0 supports concurrent (non-blocking) requests via unique `id` values. The client may send multiple requests without waiting for responses; the server may respond out of order. Ordering guarantees:

- **No ordering guarantee** between independent requests: The server may process and respond to requests in any order.
- **Causal ordering** for related operations: If the client needs sequential execution (e.g., read-then-write), it must wait for the first response before sending the second request.
- **Notification ordering**: Notifications from the server are delivered in the order they are emitted, but interleaving with request responses is possible.

---

### 2.4.4 Key Primitives: Tools and Others

#### 2.4.4.1 Primitives Overview

MCP defines six core primitives, each with distinct semantics, directionality, and access control characteristics:

| Primitive | Direction | Initiated By | Controlled By | Primary Purpose |
|-----------|-----------|-------------|---------------|-----------------|
| **Tools** | Server exposes → Client invokes | Model (via host) | Model decides when to call | Executable actions: API calls, computations, file operations |
| **Resources** | Server exposes → Client reads | Application (host) | Application decides when to fetch | Contextual data: files, database records, live feeds |
| **Prompts** | Server exposes → Client retrieves | User | User selects which prompt to use | Structured interaction templates: workflows, best practices |
| **Sampling** | Server requests ← Client provides | Server | Host approves/modifies/rejects | LLM inference: server needs model reasoning capability |
| **Elicitation** | Server requests ← Client provides | Server | User provides input | User input: credentials, confirmations, parameter selection |
| **Roots** | Client provides → Server consumes | Client/Host | Host defines boundaries | Operational scope: filesystem paths, workspace boundaries |

**Control semantics distinction (critical for understanding MCP's design philosophy):**

- **Model-controlled (Tools)**: The LLM decides whether and when to invoke a tool based on the user's query and the tool's description. The host orchestrates the actual call, but the decision-making authority rests with the model.

- **Application-controlled (Resources)**: The host application decides when to fetch resource content and inject it into the model's context. The model does not directly request resource reads; the application may pre-fetch resources based on conversation context or user actions.

- **User-controlled (Prompts)**: The user explicitly selects a prompt template (e.g., via a slash command `/summarize` or a UI menu). The model does not autonomously choose prompts.

This three-way control separation enforces a principle: *different types of external interactions require different levels of agency and different approval workflows.*

---

#### 2.4.4.2 Design Philosophy

**Separation of concerns.** Each primitive addresses a distinct interaction pattern:

- **Tools** = *doing* (executing actions that may change state)
- **Resources** = *knowing* (accessing data for contextual awareness)
- **Prompts** = *structuring* (organizing interaction patterns into reusable templates)

This separation prevents conflation: a tool that reads data is semantically different from a resource that provides data. The tool might have side effects (logging, rate limiting, authentication); the resource is a passive data source.

**Principle of least privilege.** Each primitive has the minimum access control necessary for its function:

- Tools have `annotations` (destructive, idempotent, read-only hints) that enable graduated approval policies.
- Resources are read-only by design; they cannot mutate state.
- Prompts are pure templates; they execute no actions.
- Sampling requests are explicitly gated by host approval.
- Elicitation requests require user consent.

**Compositional design.** Primitives can be composed for complex workflows:

1. User selects a **prompt** template (e.g., "analyze repository")
2. The prompt references **resources** (e.g., `repo://main/README.md`)
3. The model, reasoning with the prompt and resources, decides to invoke **tools** (e.g., `run_tests`, `search_code`)
4. A tool server needs further reasoning, so it requests **sampling** from the host
5. The sampling result requires user confirmation, so the server requests **elicitation**
6. All operations are scoped within **roots** defined by the host

---

### 2.4.5 Tool Definition (MCP)

#### 2.4.5.1 Tool Definition Schema

An MCP tool is defined by a JSON object that provides all information necessary for an LLM to understand the tool's purpose, determine when to use it, and construct valid invocations.

**Complete tool definition schema:**

```json
{
  "name": "create_github_issue",
  "description": "Creates a new issue in a GitHub repository. Use this when the user wants to file a bug report, feature request, or task in a specific repository. Requires the repository owner, repository name, issue title, and optionally a body and labels.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "owner": {
        "type": "string",
        "description": "The GitHub username or organization that owns the repository"
      },
      "repo": {
        "type": "string",
        "description": "The repository name (not the full URL)"
      },
      "title": {
        "type": "string",
        "description": "The title of the issue (concise, descriptive)"
      },
      "body": {
        "type": "string",
        "description": "The detailed body of the issue in Markdown format"
      },
      "labels": {
        "type": "array",
        "items": { "type": "string" },
        "description": "List of label names to apply to the issue"
      }
    },
    "required": ["owner", "repo", "title"]
  },
  "annotations": {
    "title": "Create GitHub Issue",
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": true
  }
}
```

**Field semantics:**

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `name` | string | Yes | Machine-readable identifier; must be unique within the server; used in `tools/call` |
| `description` | string | Yes | Natural language description for the LLM; critical for model's tool selection accuracy |
| `inputSchema` | JSON Schema object | Yes | Formal specification of accepted input parameters |
| `annotations` | object | No | Metadata hints for the host's security and UX policies |

**Description engineering.** The `description` field is the primary mechanism by which the LLM determines whether a tool is relevant to the current task. High-quality descriptions should:

1. State the tool's purpose in the first sentence.
2. Specify when the tool should be used (positive cases).
3. Specify when the tool should NOT be used (negative cases), to prevent misuse.
4. Mention key parameters and their semantics.
5. Note any prerequisites or side effects.

The description is effectively a natural language "API documentation" targeted at an LLM reader, not a human developer.

---

#### 2.4.5.2 Tool Annotations and Metadata

Annotations provide machine-readable hints about a tool's behavioral characteristics. They inform the host's policy engine and UX decisions but are NOT security controls in themselves.

| Annotation | Type | Default | Semantics |
|-----------|------|---------|-----------|
| `title` | string | — | Human-readable display name for UI rendering |
| `readOnlyHint` | boolean | `false` | If `true`: tool does not modify any state. Safe for speculative/preview execution. |
| `destructiveHint` | boolean | `true` | If `true`: tool may cause irreversible changes (delete files, drop tables, send emails). Warrants user confirmation. |
| `idempotentHint` | boolean | `false` | If `true`: calling the tool multiple times with the same arguments produces the same effect as calling it once. Safe to retry on failure. |
| `openWorldHint` | boolean | `true` | If `true`: tool interacts with systems outside the server's control (external APIs, internet). If `false`: tool operates only on local/contained data. |

**Policy engine integration.** The host can use annotations to implement graduated approval policies:

$$
\text{approval\_required}(t) = \begin{cases}
\text{auto-approve} & \text{if } t.\texttt{readOnlyHint} = \text{true} \\
\text{auto-approve with logging} & \text{if } t.\texttt{idempotentHint} = \text{true} \wedge t.\texttt{destructiveHint} = \text{false} \\
\text{user confirmation required} & \text{if } t.\texttt{destructiveHint} = \text{true} \\
\text{elevated approval} & \text{if } t.\texttt{destructiveHint} = \text{true} \wedge t.\texttt{openWorldHint} = \text{true}
\end{cases}
$$

**Critical caveat.** Annotations are *hints* declared by the server developer; they are NOT enforced by the protocol. A malicious or buggy server could declare `readOnlyHint: true` for a tool that actually deletes data. The host MUST NOT rely solely on annotations for security; they should be treated as advisory input to a defense-in-depth security strategy that includes sandboxing, permission systems, and monitoring.

---

#### 2.4.5.3 Tool Listing Protocol

**Basic listing:**

```json
// Request
{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}

// Response
{
  "jsonrpc":"2.0","id":3,
  "result":{
    "tools":[
      {
        "name":"search_files",
        "description":"Search for files matching a pattern...",
        "inputSchema":{...},
        "annotations":{...}
      },
      {
        "name":"read_file",
        "description":"Read the contents of a file...",
        "inputSchema":{...},
        "annotations":{...}
      }
    ]
  }
}
```

**Pagination.** For servers exposing large numbers of tools, the `tools/list` method supports cursor-based pagination:

```json
// First page request
{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}

// First page response
{
  "jsonrpc":"2.0","id":3,
  "result":{
    "tools":[...first 50 tools...],
    "nextCursor":"eyJwYWdlIjoxfQ=="
  }
}

// Second page request
{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{"cursor":"eyJwYWdlIjoxfQ=="}}

// Second page response
{
  "jsonrpc":"2.0","id":4,
  "result":{
    "tools":[...next 50 tools...],
    "nextCursor": null  // no more pages
  }
}
```

**Dynamic tool sets.** If the server's capabilities declare `tools.listChanged: true`, the server can emit `notifications/tools/list_changed` notifications when tools are dynamically added, removed, or modified. Upon receiving this notification, the client should re-invoke `tools/list` to refresh its cached tool definitions.

```json
// Server → Client notification
{
  "jsonrpc":"2.0",
  "method":"notifications/tools/list_changed",
  "params":{}
}
```

Use cases for dynamic tool sets:
- A server that discovers available tools at runtime based on installed plugins
- A server that exposes different tools based on the authenticated user's permissions
- A server that adds/removes tools based on external configuration changes

**Caching strategies:**
- Client caches the tool list after `tools/list` and only refreshes upon `notifications/tools/list_changed`.
- For static tool sets (no `listChanged` capability), the client can cache indefinitely for the session duration.
- Tool descriptions are injected into the LLM's context; caching avoids redundant context assembly.

---

#### 2.4.5.4 Input Schema Design for MCP Tools

MCP uses **JSON Schema** (draft 2020-12 compatible) as the universal language for describing tool input parameters. This choice provides:

- **Formal validation**: Inputs can be validated against the schema before tool invocation.
- **Rich type system**: Supports primitive types, arrays, objects, enums, patterns, constraints.
- **LLM compatibility**: JSON Schema is the same format used by OpenAI, Anthropic, and other providers for function calling.
- **Self-documentation**: `description` fields in the schema serve as parameter-level documentation for the LLM.

**Complex nested structure example:**

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "object",
      "properties": {
        "filters": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "field": { "type": "string", "description": "The field to filter on" },
              "operator": {
                "type": "string",
                "enum": ["eq", "neq", "gt", "lt", "gte", "lte", "contains"],
                "description": "The comparison operator"
              },
              "value": {
                "description": "The value to compare against",
                "oneOf": [
                  { "type": "string" },
                  { "type": "number" },
                  { "type": "boolean" }
                ]
              }
            },
            "required": ["field", "operator", "value"]
          },
          "description": "Array of filter conditions (combined with AND logic)"
        },
        "sort_by": { "type": "string", "description": "Field to sort results by" },
        "sort_order": { "type": "string", "enum": ["asc", "desc"], "default": "asc" },
        "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 }
      },
      "required": ["filters"]
    }
  },
  "required": ["query"]
}
```

**Cross-field validation and conditional schemas:**

JSON Schema supports `if/then/else` for expressing parameter dependencies:

```json
{
  "type": "object",
  "properties": {
    "action": { "type": "string", "enum": ["create", "update", "delete"] },
    "id": { "type": "string" },
    "data": { "type": "object" }
  },
  "required": ["action"],
  "if": {
    "properties": { "action": { "const": "create" } }
  },
  "then": {
    "required": ["data"]
  },
  "else": {
    "required": ["id"]
  }
}
```

This schema expresses: "if `action` is `create`, then `data` is required; otherwise, `id` is required."

**Schema evolution and backward compatibility:**

When a tool's input schema changes (e.g., adding new optional parameters, changing constraints), the server should:

1. **Additive changes** (adding optional fields): backward compatible. Existing clients that omit the new field continue to work.
2. **Constraint tightening** (adding `required` fields, narrowing `enum` values): breaking change. Must be communicated via `notifications/tools/list_changed` so clients refresh their cached schemas.
3. **Type changes** (changing a field from `string` to `integer`): breaking change. Requires careful version management.

---

### 2.4.6 Tool Results

#### 2.4.6.1 Result Structure

Every `tools/call` response follows a standardized result structure:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Successfully created issue #42: 'Fix login bug' in user/repo"
    }
  ],
  "isError": false
}
```

**Fields:**

| Field | Type | Required | Semantics |
|-------|------|----------|-----------|
| `content` | array of content objects | Yes | The tool's output, potentially multi-part and multi-modal |
| `isError` | boolean | No (default `false`) | Whether the result represents an error condition |

The `content` array allows tools to return rich, multi-part responses. For example, a data analysis tool might return both a textual summary and a chart image.

---

#### 2.4.6.2 Content Types in Results

MCP defines several content types that can appear in the `content` array:

**Text content:**

```json
{
  "type": "text",
  "text": "The query returned 42 results. Top result: ..."
}
```

The most common content type. Text is interpreted by the LLM as natural language (or structured text like JSON, Markdown, CSV, etc.). The text should be informative and structured for LLM consumption.

**Image content:**

```json
{
  "type": "image",
  "data": "iVBORw0KGgoAAAANSUhEUg...",
  "mimeType": "image/png"
}
```

Binary image data, Base64-encoded for JSON transport. Supported MIME types include `image/png`, `image/jpeg`, `image/gif`, `image/webp`. The host injects the image into the model's context (for multimodal models) or displays it to the user.

**Audio content:**

```json
{
  "type": "audio",
  "data": "UklGRi4AAABXQVZFZm10...",
  "mimeType": "audio/wav"
}
```

Binary audio data, Base64-encoded. Useful for speech synthesis, audio analysis, and music generation tools.

**Embedded resource content:**

```json
{
  "type": "resource",
  "resource": {
    "uri": "file:///project/src/main.py",
    "mimeType": "text/x-python",
    "text": "import os\n\ndef main():\n    ..."
  }
}
```

A resource reference embedded in a tool result. This allows tools to return data that is semantically a "resource" (with a URI for identification) rather than raw text. The host can cache, display, or re-use the resource in subsequent interactions.

**Multi-part results:**

A single tool call can return multiple content items:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Analysis complete. Found 3 anomalies in the dataset."
    },
    {
      "type": "image",
      "data": "iVBORw0KGgoAAAA...",
      "mimeType": "image/png"
    },
    {
      "type": "text",
      "text": "| Anomaly | Timestamp | Severity |\n|---------|-----------|----------|\n| #1 | 2024-03-15 | High | ..."
    }
  ],
  "isError": false
}
```

---

#### 2.4.6.3 Result Semantics

**Model-directed results.** Tool results in MCP are fundamentally directed at the model, not the user. The result content is injected back into the LLM's context window as part of the ongoing conversation, enabling the model to:

1. **Reason** over the result (e.g., analyze returned data, identify patterns).
2. **Plan** next actions based on the result (e.g., call another tool, ask clarifying questions).
3. **Synthesize** the result into a user-facing response (e.g., summarize API output in natural language).

This design principle means that tool results should be **machine-readable first, human-readable second**. For example:

- Return structured data (JSON, CSV, Markdown tables) rather than prose paragraphs.
- Include all relevant details; the model will select what to surface to the user.
- Use consistent formatting so the model can reliably parse the output.

**Context window budget.** Tool results consume tokens in the model's context window. The total context budget must accommodate:

$$
T_{\text{total}} = T_{\text{system}} + T_{\text{history}} + T_{\text{tool\_defs}} + T_{\text{tool\_results}} + T_{\text{generation}}
$$

where $T_{\text{tool\_results}}$ is the cumulative token count of all tool results in the current turn. Large tool results (e.g., full file contents, extensive API responses) should be truncated, summarized, or paginated to stay within budget.

---

### 2.4.7 Structured Content

#### 2.4.7.1 Multi-Modal Content Representation

MCP's content model is designed for multi-modal AI systems that can process text, images, audio, and structured data within a unified interaction.

**Unified content array.** All content, regardless of modality, is represented as elements of a single `content` array. This design:

- Enables interleaving of modalities (text explanation followed by image, followed by more text)
- Provides a consistent interface for hosts to iterate over and render content items
- Supports progressive adoption: hosts that don't support images can filter to `type: "text"` items

**MIME type specification.** Non-text content items include a `mimeType` field that enables:

- Proper rendering by the host UI
- Correct encoding/decoding
- Model compatibility checking (e.g., some models support only certain image formats)

**Base64 encoding for binary data.** Since the transport is JSON-based, binary data (images, audio) must be encoded as text. MCP uses Base64 encoding:

$$
\text{size}_{\text{base64}} = \left\lceil \frac{4}{3} \cdot \text{size}_{\text{binary}} \right\rceil
$$

This imposes approximately 33% overhead on binary content. For large binary payloads, this overhead can be significant; in practice, MCP tool results should keep binary content reasonably sized (< 10 MB as a guideline) or use resource URIs for large files.

**Size limits and chunking.** The MCP specification does not prescribe hard size limits for individual content items or total result payloads. However, practical limits arise from:

- JSON parser memory limits
- Transport-layer payload limits (HTTP request body size)
- Model context window capacity
- Network bandwidth and latency

For large content, servers should implement pagination, truncation, or return resource URIs that the client can fetch on demand.

---

#### 2.4.7.2 Content Annotations

Content items can include annotations that provide rendering and prioritization hints:

```json
{
  "type": "text",
  "text": "Internal analysis: confidence score 0.73, p-value 0.002",
  "annotations": {
    "priority": 0.8,
    "audience": ["assistant"]
  }
}
```

**Priority hints:**

The `priority` field is a floating-point value in $[0.0, 1.0]$ indicating the relative importance of the content item:

- $\text{priority} = 1.0$: Highest priority; should always be included in the model context.
- $\text{priority} = 0.0$: Lowest priority; can be omitted if context space is limited.

The host can use priority hints to implement intelligent context window management:

$$
\text{include}(c_i) = \begin{cases}
\text{true} & \text{if } \sum_{j : p_j \geq p_i} T_j \leq T_{\text{budget}} \\
\text{false} & \text{otherwise}
\end{cases}
$$

where $p_i$ is the priority of content item $c_i$, $T_j$ is its token count, and $T_{\text{budget}}$ is the remaining context budget.

**Audience hints:**

The `audience` field specifies the intended recipients:

| Value | Meaning |
|-------|---------|
| `["user"]` | Content intended for the user's display; should be shown in the UI but may not need to be in the model's context |
| `["assistant"]` | Content intended for the model's reasoning; should be in the model's context but may not need to be shown to the user |
| `["user", "assistant"]` | Content for both; shown to user AND included in model context |

This enables selective information routing:

- Debug information (`audience: ["assistant"]`): injected into model context for reasoning but not cluttering the user's view.
- Visual results (`audience: ["user"]`): displayed to the user as rich content but not consuming model context tokens.
- Primary results (`audience: ["user", "assistant"]`): both displayed and reasoned over.

---

### 2.4.8 Error Handling

#### 2.4.8.1 Error Levels in MCP

MCP has a three-tier error model, each addressing failures at different abstraction levels:

```
┌────────────────────────────────────────────┐
│  Tier 3: Tool Execution Errors              │
│  (isError: true in tool result)             │
├────────────────────────────────────────────┤
│  Tier 2: Protocol-Level Errors              │
│  (JSON-RPC error response)                  │
├────────────────────────────────────────────┤
│  Tier 1: Transport-Level Errors             │
│  (connection failures, timeouts, TLS)       │
└────────────────────────────────────────────┘
```

**Tier 1: Transport-Level Errors**

These occur below the JSON-RPC layer and prevent message delivery entirely:

| Error | Cause | Recovery |
|-------|-------|----------|
| Connection refused | Server not running, wrong port | Retry with backoff; check server status |
| Connection reset | Server crashed, network partition | Reconnect; re-initialize session |
| TLS handshake failure | Certificate mismatch, expired cert | Fix certificate configuration |
| DNS resolution failure | Server hostname unresolvable | Check network configuration |
| Timeout | Server unresponsive, network congestion | Retry with increased timeout; circuit breaker |

**Tier 2: Protocol-Level Errors (JSON-RPC)**

These are valid JSON-RPC error responses indicating that the request could not be processed at the protocol level:

| Code | Name | Typical Cause | Example |
|------|------|---------------|---------|
| $-32700$ | Parse error | Malformed JSON in request body | Missing closing brace, invalid unicode escape |
| $-32600$ | Invalid request | Valid JSON but not conforming to JSON-RPC 2.0 | Missing `jsonrpc` field, wrong version string |
| $-32601$ | Method not found | Requested method not supported by server | Calling `resources/read` on a server that doesn't expose resources |
| $-32602$ | Invalid params | Parameters don't match expected schema | Wrong argument types, missing required parameters |
| $-32603$ | Internal error | Unspecified server-side failure during processing | Unhandled exception in server code |

Protocol-level errors are reported via JSON-RPC error responses and typically indicate a bug in the client, the server, or a mismatch in expectations. They are NOT passed to the model; the host handles them internally (retry, error display, fallback).

**Tier 3: Tool Execution Errors**

These represent failures during the actual tool execution (the tool ran, but the operation it attempted failed):

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Repository 'user/nonexistent-repo' not found. The repository may not exist or you may not have access permissions. HTTP 404 from GitHub API."
    }
  ],
  "isError": true
}
```

Tool execution errors are semantically part of the tool's result. They are returned to the model as context, enabling the model to reason about the failure and take corrective action.

---

#### 2.4.8.2 Error Reporting for Tool Execution

**Design principle.** Tool execution errors are fundamentally different from protocol errors. A `tools/call` that results in a tool execution error is a *successful* JSON-RPC request—the request was properly received, parsed, routed, and executed. The tool simply produced an error result. Therefore:

- The JSON-RPC response has `result` (not `error`).
- The result contains `isError: true` and a descriptive error message in `content`.
- The error message is injected into the model's context for reasoning.

**Model-driven error recovery.** The model, upon receiving an error result, can:

1. **Retry with corrected arguments**: "The file path was wrong; let me try with the correct path."
2. **Try an alternative tool**: "File search failed; let me try the database query tool instead."
3. **Ask the user for clarification**: "I couldn't find that repository. Could you check the name?"
4. **Report the failure**: "I was unable to create the issue because the repository doesn't exist."

This design enables autonomous error recovery in agentic loops without requiring hard-coded retry logic in the host.

**Error message quality guidelines:**

- **Be specific**: "Repository 'user/repo' not found (HTTP 404)" not "An error occurred."
- **Include actionable information**: Suggest what went wrong and how to fix it.
- **Don't leak sensitive data**: Don't include internal stack traces, database connection strings, or authentication tokens.
- **Use structured format**: If the error has multiple aspects, use a structured format the model can parse.

---

#### 2.4.8.3 Retry and Recovery Strategies

**Idempotent tools** (`idempotentHint: true`):
- Safe to retry automatically on transient failures (network timeouts, server overload).
- The host can implement automatic retry with exponential backoff:

$$
t_{\text{wait}}(n) = \min\left(t_{\text{base}} \cdot 2^n + \text{jitter}(n), t_{\text{max}}\right)
$$

where $n$ is the retry attempt number, $t_{\text{base}}$ is the base delay, and $t_{\text{max}}$ is the maximum delay cap.

**Non-idempotent tools** (`idempotentHint: false`):
- Retrying may cause duplicate side effects (e.g., creating two identical issues, sending two emails).
- The host should NOT automatically retry. Instead:
  - Surface the error to the model for decision-making.
  - Or surface the error to the user for manual retry approval.

**Circuit breaker pattern.** For tools that fail repeatedly, the host should implement circuit breaker logic:

$$
\text{state} = \begin{cases}
\text{CLOSED} & \text{normal operation; requests pass through} \\
\text{OPEN} & \text{after } k \text{ consecutive failures; requests immediately rejected for } t_{\text{cool}} \text{ seconds} \\
\text{HALF-OPEN} & \text{after cooldown; one test request allowed to check recovery}
\end{cases}
$$

This prevents cascading failures and reduces load on failing servers.

**Graceful degradation.** When a tool is unavailable:

- The host can remove it from the tool list provided to the model.
- The host can replace it with a stub that returns a helpful error message.
- The host can suggest alternative tools that provide similar functionality.

---

### 2.4.9 Other Capabilities

#### 2.4.9.1 Resources

**Definition.** Resources are server-controlled data sources that provide contextual information to the AI model. Unlike tools (which perform actions), resources are passive data providers.

**URI-based addressing.** Every resource is identified by a URI:

```
file:///home/user/project/README.md
https://api.example.com/data/reports/q4-2024
postgres://db.internal/sales/customers
custom://myserver/live-metrics
```

The URI scheme indicates the resource type; the server is responsible for resolving URIs to content.

**Resource listing and reading:**

```json
// List resources
{"jsonrpc":"2.0","id":5,"method":"resources/list","params":{}}

// Response
{
  "jsonrpc":"2.0","id":5,
  "result":{
    "resources":[
      {
        "uri":"file:///project/README.md",
        "name":"Project README",
        "description":"Main project documentation",
        "mimeType":"text/markdown"
      },
      {
        "uri":"file:///project/config.yaml",
        "name":"Configuration",
        "description":"Application configuration file",
        "mimeType":"application/yaml"
      }
    ]
  }
}

// Read a specific resource
{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"file:///project/README.md"}}

// Response
{
  "jsonrpc":"2.0","id":6,
  "result":{
    "contents":[
      {
        "uri":"file:///project/README.md",
        "mimeType":"text/markdown",
        "text":"# My Project\n\nThis is a project that..."
      }
    ]
  }
}
```

**Subscription model.** For resources that change over time (live data feeds, files being edited), MCP supports subscriptions:

```json
// Subscribe to changes
{"jsonrpc":"2.0","id":7,"method":"resources/subscribe","params":{"uri":"file:///project/config.yaml"}}

// Server sends notification when resource changes
{
  "jsonrpc":"2.0",
  "method":"notifications/resources/updated",
  "params":{"uri":"file:///project/config.yaml"}
}

// Client re-reads the resource to get updated content
{"jsonrpc":"2.0","id":8,"method":"resources/read","params":{"uri":"file:///project/config.yaml"}}
```

**Resource templates.** For dynamic resources parameterized by variables:

```json
{
  "uriTemplate": "github://repos/{owner}/{repo}/issues/{issue_number}",
  "name": "GitHub Issue",
  "description": "A specific GitHub issue",
  "mimeType": "application/json"
}
```

The client can construct concrete URIs by substituting template variables.

**Resources vs. Tools — the critical distinction:**

| Aspect | Resources | Tools |
|--------|-----------|-------|
| **Control** | Application-controlled: the host/application decides when to fetch | Model-controlled: the model decides when to invoke |
| **Side effects** | None (read-only by definition) | May have side effects (create, update, delete) |
| **Context injection** | Pre-fetched and injected into context before model reasoning | Invoked during model reasoning as needed |
| **Use case** | Providing background context, reference data | Executing actions, querying dynamic data |

---

#### 2.4.9.2 Prompts

**Definition.** Prompts are server-provided templates that structure LLM interactions into reusable, parameterized patterns. They encode best practices, domain-specific workflows, and standardized interaction formats.

**Prompt listing and retrieval:**

```json
// List prompts
{"jsonrpc":"2.0","id":9,"method":"prompts/list","params":{}}

// Response
{
  "jsonrpc":"2.0","id":9,
  "result":{
    "prompts":[
      {
        "name":"code_review",
        "description":"Generate a thorough code review for a given file",
        "arguments":[
          {
            "name":"file_path",
            "description":"Path to the file to review",
            "required":true
          },
          {
            "name":"review_focus",
            "description":"Specific aspect to focus on (security, performance, readability)",
            "required":false
          }
        ]
      }
    ]
  }
}

// Get expanded prompt
{
  "jsonrpc":"2.0","id":10,
  "method":"prompts/get",
  "params":{
    "name":"code_review",
    "arguments":{
      "file_path":"/src/auth.py",
      "review_focus":"security"
    }
  }
}

// Response: expanded prompt messages
{
  "jsonrpc":"2.0","id":10,
  "result":{
    "description":"Code review for /src/auth.py focusing on security",
    "messages":[
      {
        "role":"system",
        "content":{
          "type":"text",
          "text":"You are an expert code reviewer specializing in security analysis..."
        }
      },
      {
        "role":"user",
        "content":{
          "type":"text",
          "text":"Please review the following file for security vulnerabilities:\n\n```python\nimport os\n...\n```"
        }
      }
    ]
  }
}
```

**User-controlled selection.** Prompts are typically surfaced to the user via slash commands (e.g., `/code_review`) or UI menus. The user explicitly selects a prompt; the model does not autonomously choose prompts. This ensures that prompt selection remains a human decision aligned with user intent.

**Multi-turn prompt sequences.** A single prompt can expand into multiple messages with different roles (system, user, assistant), enabling complex interaction setups:

```json
"messages": [
  {"role": "system", "content": {"type": "text", "text": "System instructions..."}},
  {"role": "user", "content": {"type": "text", "text": "Initial user context..."}},
  {"role": "assistant", "content": {"type": "text", "text": "Primed assistant response..."}},
  {"role": "user", "content": {"type": "text", "text": "Follow-up with specific task..."}}
]
```

---

#### 2.4.9.3 Sampling

**Definition.** Sampling is the mechanism by which an MCP server requests the host to perform LLM inference on the server's behalf. This is a *reverse-direction* operation: the server is asking the client to do something, rather than the client asking the server.

**Why does a server need LLM inference?**

1. **Agentic sub-loops**: The server is implementing a multi-step workflow that requires intermediate reasoning (e.g., analyzing search results to refine a query).
2. **Content generation**: The server needs to generate text (summaries, translations, code) as part of its tool execution.
3. **Decision making**: The server needs the model to choose between options based on complex criteria.
4. **Recursive processing**: A tool result needs further analysis before final output.

**Sampling request structure:**

```json
{
  "jsonrpc":"2.0",
  "id":"server-req-1",
  "method":"sampling/createMessage",
  "params":{
    "messages":[
      {
        "role":"user",
        "content":{
          "type":"text",
          "text":"Analyze the following error log and identify the root cause:\n\n[ERROR] 2024-03-15 Connection timeout to db-primary..."
        }
      }
    ],
    "modelPreferences":{
      "hints":[
        {"name":"claude-sonnet-4-20250514"}
      ],
      "intelligencePriority": 0.8,
      "speedPriority": 0.5,
      "costPriority": 0.3
    },
    "systemPrompt":"You are a systems reliability engineer...",
    "maxTokens": 500,
    "temperature": 0.2,
    "includeContext": "thisServer"
  }
}
```

**Host control.** The host retains absolute authority over sampling requests:

- **Approve or reject**: The host may refuse the sampling request entirely (e.g., rate limit exceeded, policy violation).
- **Modify the request**: The host may alter the messages, system prompt, model selection, or parameters before executing.
- **Choose the model**: The `modelPreferences` are hints; the host decides which model to actually use.
- **Filter the response**: The host may redact or modify the model's response before returning it to the server.
- **Human-in-the-loop**: The host may present the sampling request to the user for approval before execution.

**Privacy consideration.** The server should not receive raw model responses that contain information from other servers' contexts or private user data. The `includeContext` parameter controls what context is shared:

| Value | Meaning |
|-------|---------|
| `"none"` | Only the messages in the sampling request are used; no additional context |
| `"thisServer"` | Include context from this server's tools/resources but not from other servers |
| `"allServers"` | Include context from all connected servers (broader context, higher privacy risk) |

---

#### 2.4.9.4 Elicitation

**Definition.** Elicitation allows an MCP server to request structured input directly from the user, mediated by the client/host. Unlike sampling (which requests LLM inference), elicitation requests *human* input.

**Use cases:**

- Requesting authentication credentials (username, password, API key)
- Asking for confirmation before a destructive operation
- Collecting parameter values that the model cannot determine
- Disambiguating between multiple interpretation options

**Elicitation request structure:**

```json
{
  "jsonrpc":"2.0",
  "id":"server-req-2",
  "method":"elicitation/create",
  "params":{
    "message":"Please provide your GitHub personal access token to proceed with repository access.",
    "requestedSchema":{
      "type":"object",
      "properties":{
        "token":{
          "type":"string",
          "description":"GitHub personal access token",
          "minLength": 40
        }
      },
      "required":["token"]
    }
  }
}
```

**Host rendering.** The host/client receives the elicitation request and renders an appropriate UI widget:

- Text input field for string values
- Dropdown/select for enum values
- Checkbox for boolean values
- Secure/masked input for password/token fields
- Multi-field form for complex schemas

**Elicitation response:**

```json
{
  "jsonrpc":"2.0",
  "id":"server-req-2",
  "result":{
    "action":"accept",
    "content":{
      "token":"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    }
  }
}
```

| Action | Meaning |
|--------|---------|
| `"accept"` | User provided the requested input |
| `"decline"` | User explicitly refused to provide input |
| `"dismiss"` | User dismissed the dialog without responding |

**Security.** The host may:

- Refuse to render the elicitation request (e.g., disallow credential collection from untrusted servers).
- Validate the collected input against the schema before returning it to the server.
- Mask sensitive input in logs and audit trails.
- Rate-limit elicitation requests to prevent UI spam.

---

#### 2.4.9.5 Roots

**Definition.** Roots are URIs provided by the client to the server that define the operational boundaries—the "workspace" within which the server should operate.

**Example:**

```json
{
  "roots": [
    {
      "uri": "file:///home/user/projects/my-app",
      "name": "My Application"
    },
    {
      "uri": "file:///home/user/shared-libs",
      "name": "Shared Libraries"
    }
  ]
}
```

This tells the server: "You should only operate on files within `my-app/` and `shared-libs/`. Do not access files outside these directories."

**Informational, not enforced.** Roots are guidance signals, not hard security boundaries. A well-behaved server respects roots; a malicious server might ignore them. The host must enforce actual filesystem permissions via OS-level sandboxing, container isolation, or permission systems.

**Dynamic updates.** If the client declares `roots.listChanged: true` during capability negotiation, it can send `notifications/roots/list_changed` when the workspace changes (e.g., user opens a new project folder). The server should re-query roots and adjust its behavior accordingly.

---

## 2.5 Model Context Protocol: For and Against

### 2.5.1 Capabilities and Strategic Advantages

#### 2.5.1.1 Accelerating Development and Fostering a Reusable Ecosystem

**Integration effort reduction.** As formally derived in §2.4.1.3, MCP reduces integration complexity from $O(N \times M)$ to $O(N + M)$. For a concrete analysis, consider the expected engineering effort:

Let $c$ be the average cost (person-days) of building and maintaining one custom integration. The total ecosystem cost without MCP:

$$
\text{Cost}_{\text{without}} = c \cdot N \cdot M
$$

With MCP, each application builds one client (cost $c_{\text{client}}$) and each tool builds one server (cost $c_{\text{server}}$). Additionally, both can leverage shared SDK libraries (cost $c_{\text{SDK}}$, amortized):

$$
\text{Cost}_{\text{with}} = N \cdot c_{\text{client}} + M \cdot c_{\text{server}} + c_{\text{SDK}}
$$

The break-even condition occurs when $\text{Cost}_{\text{with}} < \text{Cost}_{\text{without}}$:

$$
N \cdot c_{\text{client}} + M \cdot c_{\text{server}} + c_{\text{SDK}} < c \cdot N \cdot M
$$

For $N, M \gg 1$, this is almost always satisfied because the left side grows linearly while the right side grows quadratically.

**Ecosystem effects.** MCP enables a "write-once, use-everywhere" tool ecosystem:

- A GitHub MCP server written by one developer is immediately usable by every MCP-compliant host (Claude Desktop, custom agents, IDE extensions).
- Community-maintained registries of MCP servers emerge (analogous to npm for Node.js packages or pip for Python packages).
- Tool quality improves through community feedback and contributions, as opposed to siloed internal integrations.

**Open-source MCP server ecosystem (current state):**

| Domain | Example MCP Servers | Functionality |
|--------|-------------------|---------------|
| Version control | GitHub, GitLab | Issue management, PR review, code search |
| Communication | Slack, Discord | Message sending, channel management |
| Databases | PostgreSQL, SQLite | Query execution, schema inspection |
| Filesystems | Local filesystem | File read/write, directory navigation |
| Search | Brave Search, Google | Web search, result retrieval |
| Cloud | AWS, GCP | Resource management, deployment |
| Development | Docker, Kubernetes | Container management, deployment |

**Rapid prototyping.** The standardized interface reduces time-to-first-tool from days (building a custom integration) to minutes (implementing an MCP server with the SDK).

**Cross-platform compatibility.** An MCP tool server developed for use with Claude Desktop automatically works with any other MCP-compliant host, including:

- Custom agentic applications
- IDE extensions
- Enterprise AI platforms
- Research prototypes

**Lower barrier to entry.** Tool developers only need to understand the MCP specification; they do not need expertise in each AI platform's specific function-calling format, SDK, or authentication system.

---

#### 2.5.1.2 Architectural Flexibility and Future-Proofing

**Model agnosticism.** MCP is deliberately model-agnostic. The protocol defines how tools are described and invoked, but does not prescribe:

- Which LLM is used (GPT-4, Claude, Gemini, Llama, Mistral, or any future model)
- How the LLM processes tool descriptions (system prompt injection, special tokens, etc.)
- The LLM's tool-calling format (the host translates between MCP tool definitions and the model's native format)

This means the same MCP server works regardless of model changes, upgrades, or migrations.

**Transport flexibility.** The three-transport architecture (stdio, HTTP+SSE, Streamable HTTP) covers the full spectrum of deployment scenarios:

$$
\text{deployment scenario} \mapsto \text{optimal transport} = \begin{cases}
\text{local development, desktop apps} & \rightarrow \text{stdio} \\
\text{cloud services, existing infrastructure} & \rightarrow \text{HTTP+SSE} \\
\text{enterprise, production, load-balanced} & \rightarrow \text{Streamable HTTP}
\end{cases}
$$

New transport mechanisms can be added in future protocol versions without breaking existing implementations.

**Progressive capability adoption.** Servers implement only the primitives they need:

- A simple calculator server: only `tools` capability.
- A file browser server: `tools` + `resources` capabilities.
- A sophisticated development server: `tools` + `resources` + `prompts` + `sampling` capabilities.

This progressivity prevents the protocol from being "all or nothing" and allows incremental adoption.

**Version negotiation.** The date-based versioning scheme and the negotiation handshake ensure that:

- Newer clients can work with older servers (graceful degradation to older feature set).
- Newer servers can work with older clients (unknown capabilities are ignored).
- Breaking changes are explicitly versioned, not silently introduced.

**Abstraction stability.** Because tool interfaces are defined at the MCP protocol level (name, description, JSON Schema), they remain stable even when the underlying implementation changes completely. A tool server can migrate from a REST API backend to a GraphQL backend, or from Python to Rust, without any change to the MCP interface.

**Multi-model architectures.** A single MCP server can simultaneously serve tools to multiple different hosts using different models. This enables:

- A/B testing of different models with the same tool set
- Ensemble architectures where multiple models collaborate via shared tools
- Migration scenarios where old and new models run in parallel

---

#### 2.5.1.3 Foundations for Governance and Control

**Centralized policy enforcement.** The host-as-trust-boundary architecture provides a single point of control for:

| Policy Type | Implementation Point | Mechanism |
|------------|---------------------|-----------|
| **Tool approval** | Host | Consent dialog before destructive tool calls |
| **Rate limiting** | Host | Cap on tool calls per minute/session |
| **Data access** | Host | Filtering which resources are exposed to which servers |
| **Sampling approval** | Host | Review/modify sampling requests before execution |
| **Audit logging** | Host | Recording all tool calls, arguments, and results |
| **Cost control** | Host | Spending caps on API-backed tools |

**Tool annotation-driven policies.** As formalized in §2.4.5.2, annotations enable automated policy decisions:

- Auto-approve `readOnlyHint: true` tools (no state change risk)
- Require confirmation for `destructiveHint: true` tools
- Allow automatic retry for `idempotentHint: true` tools
- Apply enhanced monitoring for `openWorldHint: true` tools

**Audit trail.** The standardized JSON-RPC message format enables comprehensive audit logging:

```json
{
  "timestamp": "2024-03-15T14:30:22Z",
  "session_id": "sess-abc123",
  "direction": "client→server",
  "method": "tools/call",
  "params": {
    "name": "delete_file",
    "arguments": {"path": "/tmp/old-data.csv"}
  },
  "user_id": "user-456",
  "approval": "user-confirmed",
  "result": {"isError": false, "content": [{"type":"text","text":"File deleted"}]},
  "latency_ms": 45
}
```

Every tool invocation, argument, result, approval decision, and timing can be logged in a standardized format across all tools and all hosts.

**Human-in-the-loop integration.** The sampling and elicitation primitives explicitly encode human oversight into the protocol:

- Sampling requests can be presented to the user for review before execution.
- Elicitation requests directly involve the user in the interaction loop.
- The host can configure approval policies per tool, per server, or per user.

**Scope limitation via roots.** Roots provide an explicit mechanism for constraining server operations to defined boundaries, supporting the principle of least privilege at the workspace level.

---

### 2.5.2 Critical Risks and Challenges

#### 2.5.2.1 Specification Maturity and Stability

**Rapid evolution risk.** MCP's specification is actively evolving, with the protocol version advancing through date-based releases (e.g., `2024-11-05` → `2025-03-26` → `2025-06-18`). Each release may introduce:

- New primitives (e.g., elicitation was added in later versions)
- Modified message formats
- Changed capability negotiation semantics
- New transport specifications (Streamable HTTP was added after the initial spec)

For production deployments, this means:

$$
\text{maintenance\_cost}(t) = \int_0^T \mathbb{1}[\text{spec\_change}(t)] \cdot c_{\text{update}} \, dt
$$

where $\mathbb{1}[\text{spec\_change}(t)]$ is an indicator for specification changes at time $t$ and $c_{\text{update}}$ is the cost of updating implementations.

**Underspecified areas.** Several critical aspects remain insufficiently specified:

| Area | Gap | Impact |
|------|-----|--------|
| **Authentication** | No standard auth mechanism defined in the base protocol | Each server implements its own auth, breaking interoperability |
| **Authorization** | No RBAC, ABAC, or capability-based access control standard | No way to express "user X can call tool Y but not tool Z" |
| **Multi-tenancy** | No tenant isolation semantics defined | Shared servers cannot safely serve multiple organizations |
| **Error taxonomy** | Limited standard error codes beyond JSON-RPC defaults | Inconsistent error reporting across servers |
| **Content size limits** | No standard maximum payload sizes | Risk of oversized responses causing client failures |

**Conformance validation.** As of the current state:

- No official conformance test suite exists.
- Interoperability between different MCP SDK implementations (TypeScript vs. Python) may have subtle behavioral differences.
- Edge case handling (malformed messages, unexpected disconnections, concurrent requests) varies across implementations.

---

#### 2.5.2.2 Complexity Overhead

**Abstraction layer cost.** MCP adds an abstraction layer between the AI application and the tool. For use cases where a single model calls a single tool in a single application, this abstraction provides no benefit but adds:

| Overhead | Source | Magnitude |
|----------|--------|-----------|
| Serialization | JSON-RPC encoding/decoding | ~0.1-1ms per message |
| Transport | Process spawning (stdio) or HTTP round-trip | ~1-100ms |
| Handshake | `initialize` / `initialized` exchange | ~10-500ms (one-time) |
| Tool listing | `tools/list` request-response | ~5-50ms per server |
| Context assembly | Converting MCP tool definitions to model format | ~1-5ms |

**When MCP is overkill.** MCP's value proposition depends on the ecosystem context:

$$
\text{value}(\text{MCP}) = \underbrace{(N-1) \cdot M + N \cdot (M-1)}_{\text{avoided integrations}} \cdot c_{\text{integration}} - \underbrace{(N + M)}_{\text{MCP implementations}} \cdot c_{\text{MCP}} - c_{\text{overhead}} \cdot T
$$

For $N = 1, M = 1$: $\text{value} = 0 - 2 \cdot c_{\text{MCP}} - c_{\text{overhead}} \cdot T < 0$. MCP adds cost with no benefit.

For $N = 1, M = 3$: marginal. The overhead may or may not be justified depending on tool complexity.

For $N = 5, M = 20$: strongly positive. The avoided integration cost ($100 \cdot c_{\text{integration}}$ reduced to $25 \cdot c_{\text{MCP}}$) dominates.

**Server lifecycle management.** Operating MCP servers introduces operational complexity:

- **Process management**: For stdio servers, the host must manage child process spawning, monitoring, and cleanup. Zombie processes, orphaned servers, and resource leaks are operational risks.
- **Health monitoring**: Periodic ping/pong, restart-on-failure policies, and alerting.
- **Configuration management**: Each server may require its own configuration (API keys, database credentials, feature flags).
- **Dependency management**: Servers may have their own runtime dependencies (Python virtualenvs, Node.js versions, system libraries).

---

#### 2.5.2.3 Performance Considerations

**Serialization overhead.** JSON-RPC messages require:

1. Serialization (object → JSON string): CPU cost proportional to message size.
2. Transport (send bytes over pipe or network): I/O cost.
3. Deserialization (JSON string → object): CPU cost proportional to message size.

For text-heavy tool results, this overhead is minimal. For binary-heavy results (images, audio), the Base64 encoding adds 33% size overhead:

$$
\text{overhead}_{\text{base64}} = \frac{\text{size}_{\text{encoded}} - \text{size}_{\text{original}}}{\text{size}_{\text{original}}} = \frac{4/3 \cdot s - s}{s} = \frac{1}{3} \approx 33\%
$$

**Context window consumption.** MCP tool definitions consume context tokens. For $M$ servers with an average of $k$ tools each, with average description length $d$ tokens:

$$
T_{\text{tool\_defs}} = M \cdot k \cdot d
$$

For $M = 5$ servers, $k = 10$ tools each, $d = 150$ tokens per tool: $T_{\text{tool\_defs}} = 7{,}500$ tokens. This is a significant fraction of a 128K-token context window (~6%) and a very significant fraction of a 4K-token context window (~188%—exceeding the entire window).

**Mitigation strategies:**

- **Dynamic tool selection**: Only inject tool definitions relevant to the current query.
- **Tool description compression**: Use concise descriptions optimized for token efficiency.
- **Hierarchical tool namespaces**: Present tool categories first; expand to individual tools on demand.
- **Model-side tool caching**: Some models support persistent tool definitions that don't consume per-turn context tokens.

**Latency comparison:**

| Approach | Typical Latency | Components |
|----------|----------------|------------|
| Direct function call (in-process) | ~μs | Function dispatch |
| MCP via stdio | ~1-10ms | Process IPC + JSON parse |
| MCP via HTTP (local) | ~5-50ms | HTTP stack + JSON parse |
| MCP via HTTP (remote) | ~10-500ms | Network RTT + HTTP stack + JSON parse |
| Direct API call (no MCP) | ~10-500ms | Network RTT + HTTP stack + custom parse |

Note: For remote tools, MCP adds minimal overhead compared to a direct API call, since the network round-trip dominates. The overhead is most significant for local tools where a direct function call would be near-instantaneous.

---

#### 2.5.2.4 Enterprise Readiness Gaps

**Authentication and Authorization**

MCP's base specification does not define a standard authentication or authorization mechanism. This is arguably the most significant gap for enterprise adoption.

**Current state:**

- OAuth 2.1 integration has been proposed in the specification and some SDKs, but adoption is inconsistent.
- Most existing MCP servers rely on environment variables for API keys (e.g., `GITHUB_TOKEN`), which are:
  - Not dynamically rotatable
  - Not scoped to individual users
  - Not auditable
  - Shared across all sessions on a host

**Required capabilities for enterprise auth:**

| Requirement | Description | MCP Support |
|------------|-------------|-------------|
| **User-level authentication** | Each user authenticates to each server independently | Not standardized |
| **Token rotation** | Credentials expire and are refreshed transparently | Not standardized |
| **Scoped permissions** | Different users get different tool access on the same server | Not standardized |
| **SSO integration** | Federated identity via SAML, OIDC | Not standardized |
| **Credential isolation** | One server's credentials are not accessible to other servers | Partially (client isolation) |

**Scalability Concerns**

The 1:1 client-server connection model creates scalability challenges:

$$
\text{connections}_{\text{total}} = N_{\text{users}} \times M_{\text{servers}}
$$

For $N_{\text{users}} = 1{,}000$ concurrent users, each connecting to $M_{\text{servers}} = 10$ servers: $10{,}000$ concurrent connections. Each connection consumes:

- Server-side memory for session state
- A process or thread (for stdio servers) or an HTTP connection (for remote servers)
- Capability negotiation overhead at connection establishment

**Scalability mitigation approaches (not yet standardized):**

- Connection pooling: Multiple users share a pool of server connections (requires session multiplexing, which MCP's 1:1 model does not natively support).
- Stateless server mode: Servers that don't require per-session state could handle requests without persistent connections (partially supported by Streamable HTTP).
- Server-side load balancing: Multiple instances of the same MCP server behind a load balancer, with session affinity via `Mcp-Session-Id`.

**Monitoring and Observability**

Production MCP deployments require comprehensive observability, which is not yet standardized:

| Requirement | Description | MCP Support |
|------------|-------------|-------------|
| **Metrics** | Request rate, latency percentiles, error rates per tool/server | Not standardized |
| **Distributed tracing** | Trace a user request through host → client → server → external API | Not standardized |
| **Structured logging** | Standardized log format with correlation IDs | MCP defines `logging` capability but minimally |
| **Alerting** | Automated alerts on anomalies (error spikes, latency degradation) | Not standardized |
| **SLA management** | Availability and performance guarantees per server | Not standardized |

**Integration with OpenTelemetry** (the de facto observability standard) would require:

- Propagating trace context (`traceparent` header) through MCP messages.
- Emitting spans for each `tools/call` invocation with standardized attributes.
- Recording metrics (histograms for latency, counters for calls) per tool.

This integration is technically feasible but requires convention or specification-level standardization.

**Deployment and Operations**

| Operational Concern | Challenge | Current State |
|--------------------|-----------|---------------|
| **Server discovery** | How does a host find available MCP servers? | Manual configuration; no standard registry |
| **Configuration management** | How are server configs managed across environments? | Environment variables; no standard format |
| **Rolling updates** | How to update a server without disrupting active sessions? | Not addressed; sessions break on server restart |
| **Container orchestration** | How to run MCP servers in Kubernetes? | No standard patterns; community-developed |
| **Secret management** | How to securely provide API keys/credentials to servers? | Environment variables; no Vault/KMS integration standard |

**Container orchestration pattern (emerging practice):**

```yaml
# Example: MCP server as a Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: github-mcp-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mcp-server
        image: mcp-servers/github:v1.4
        ports:
        - containerPort: 8080
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: github-credentials
              key: token
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

This pattern is not standardized; each organization develops its own deployment conventions. Future MCP specification extensions or community standards may address these operational gaps.

---

**Summary of MCP's Position in the Maturity Curve:**

$$
\text{MCP maturity} = f(\text{specification completeness}, \text{ecosystem size}, \text{enterprise readiness})
$$

| Dimension | Current State | Required for Production |
|-----------|--------------|----------------------|
| Core protocol | Mature (tools, resources, prompts) | ✓ |
| Transport layer | Mature (stdio, HTTP+SSE, Streamable HTTP) | ✓ |
| SDK support | Good (TypeScript, Python; others emerging) | ✓ |
| Authentication | Nascent (OAuth 2.1 proposed) | ✗ |
| Authorization | Absent | ✗ |
| Multi-tenancy | Absent | ✗ |
| Observability | Minimal | ✗ |
| Deployment patterns | Ad hoc | ✗ |
| Conformance testing | Absent | ✗ |
| Ecosystem size | Growing rapidly (100+ open-source servers) | ✓ |

MCP is architecturally sound and strategically well-positioned, but significant engineering effort remains to bridge the gap between its current state and enterprise-grade production readiness. Organizations adopting MCP today should treat it as a foundational protocol layer while building supplementary infrastructure for authentication, monitoring, and deployment on top of it.