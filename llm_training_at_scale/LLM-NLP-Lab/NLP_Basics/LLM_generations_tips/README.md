# LLM Inference Parameters: Detailed Technical Explanation

## 1. System Instruction (system_instruction)

**Definition:** A parameter that provides high-level directives to the language model, establishing the context, persona, or behavioral framework for the model's responses.

**Mathematical Representation:**
The system instruction modifies the conditional probability distribution:
$$P(y|x, s)$$

Where:
- $y$ = generated text
- $x$ = user input
- $s$ = system instruction

**Importance:**
- Establishes behavioral guardrails and operational parameters
- Controls model tone, personality, and response style
- Enables specialized capabilities without model fine-tuning
- Critical for alignment with intended use cases and safety requirements

## 2. Temperature (temperature)

**Definition:** A hyperparameter controlling the randomness in token selection during sampling.

**Mathematical Representation:**
The temperature-adjusted probability distribution is:
$$P_T(x_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Where:
- $T$ = temperature value
- $z_i$ = logit for token $i$
- $P_T(x_i)$ = probability of selecting token $i$ at temperature $T$

**Importance:**
- $T < 1.0$: More deterministic outputs, higher confidence tokens prioritized
- $T = 1.0$: Standard softmax distribution
- $T > 1.0$: More diverse and random outputs, flattens probability distribution
- Critical for balancing creativity vs. predictability in generated content

## 3. Top-p Sampling (top_p)

**Definition:** Nucleus sampling technique that dynamically limits the set of tokens considered during generation to only those whose cumulative probability exceeds a threshold $p$.

**Mathematical Representation:**
$$V^{(p)} = \min\{k \mid \sum_{i=1}^{k} P_{\text{sorted}}(x_i) \geq p\}$$

The sampling subset is defined as:
$$S^{(p)} = \{x_1, x_2, ..., x_{V^{(p)}}\}$$

Where:
- $P_{\text{sorted}}$ = sorted probability distribution in descending order
- $V^{(p)}$ = minimum number of tokens needed to exceed threshold $p$
- $S^{(p)}$ = the subset of tokens considered for sampling

**Importance:**
- Adaptively adjusts the candidate token pool based on probability distribution
- Prevents generation from the long tail of improbable tokens
- Provides more consistent quality than top-k across varying contexts
- Balances diversity and quality more effectively than fixed cutoffs

## 4. Top-k Sampling (top_k)

**Definition:** Sampling technique that restricts token selection to only the $k$ most probable next tokens.

**Mathematical Representation:**
$$S^{(k)} = \{x_1, x_2, ..., x_k\}$$

Where:
- $S^{(k)}$ = subset of top $k$ tokens by probability
- Probabilities are renormalized within this subset:
$$P'(x_i) = \begin{cases}
\frac{P(x_i)}{\sum_{x_j \in S^{(k)}} P(x_j)} & \text{if } x_i \in S^{(k)} \\
0 & \text{otherwise}
\end{cases}$$

**Importance:**
- Provides hard cutoff for low-probability tokens
- Prevents sampling from the long distribution tail
- Significantly reduces the likelihood of generating nonsensical text
- Simple implementation with consistent computational requirements

## 5. Candidate Count (candidate_count)

**Definition:** The number of alternative completions generated for a single prompt.

**Mathematical Representation:**
$$\{y_1, y_2, ..., y_n\} \sim P(y|x)$$

Where:
- $n$ = candidate_count
- $y_i$ = the $i$-th generated completion
- $P(y|x)$ = probability distribution of completions given prompt $x$

**Importance:**
- Enables exploring alternative generation paths
- Critical for applications requiring diverse options
- Facilitates quality selection through post-generation filtering
- Supports reliability in mission-critical applications

## 6. Maximum Output Tokens (max_output_tokens)

**Definition:** The upper bound on the number of tokens that can be generated in a model response.

**Mathematical Representation:**
$$|y| \leq M$$

Where:
- $|y|$ = length of generated text in tokens
- $M$ = max_output_tokens

**Importance:**
- Prevents unbounded or runaway generation
- Controls computational resource usage
- Ensures predictable response latency
- Protects against token quota depletion

## 7. Stop Sequences (stop_sequences)

**Definition:** A list of text patterns that, when generated, cause the model to terminate generation immediately.

**Mathematical Representation:**
Generation terminates when:
$$\exists s \in S \text{ such that } y_{t-|s|+1:t} = s$$

Where:
- $S$ = set of stop sequences
- $y_{i:j}$ = substring of generated text from position $i$ to $j$
- $t$ = current generation position
- $|s|$ = length of stop sequence $s$

**Importance:**
- Enables fine-grained control over generation termination
- Crucial for formats requiring specific boundaries (code, dialogue)
- Prevents generation beyond logical completion points
- Improves efficiency by avoiding unnecessary computation

## 8. Response Log Probabilities (response_logprobs)

**Definition:** Boolean parameter that controls whether token-level log probabilities are returned with the generated text.

**Mathematical Representation:**
For each token $t_i$ in response, return:
$$\log P(t_i | t_1, t_2, ..., t_{i-1}, x)$$

Where:
- $t_i$ = token at position $i$
- $x$ = input prompt
- $P(t_i | ...)$ = conditional probability of token $t_i$

**Importance:**
- Enables analysis of model confidence in generated content
- Critical for uncertainty quantification
- Supports advanced filtering and post-processing
- Essential for research applications and model debugging

## 9. Log Probabilities (logprobs)

**Definition:** Integer parameter specifying how many alternative token probabilities to return at each generation step.

**Mathematical Representation:**
For each position $i$, return the top $k$ tokens by probability:
$$\{(t_{i,1}, \log P(t_{i,1})), (t_{i,2}, \log P(t_{i,2})), ..., (t_{i,k}, \log P(t_{i,k}))\}$$

Where:
- $k$ = logprobs parameter value
- $t_{i,j}$ = $j$-th most probable token at position $i$
- $\log P(t_{i,j})$ = log probability of token $t_{i,j}$

**Importance:**
- Provides visibility into model uncertainty
- Enables analysis of alternative generation paths
- Essential for uncertainty quantification research
- Supports advanced model interpretation techniques

## 10. Presence Penalty (presence_penalty)

**Definition:** A parameter that penalizes tokens based on their presence in the generated text so far, regardless of frequency.

**Mathematical Representation:**
The adjusted logits for token $w$ are:
$$\text{logit}'(w) = \text{logit}(w) - \beta \cdot \mathbb{1}[w \in \text{generated}]$$

Where:
- $\beta$ = presence_penalty value
- $\mathbb{1}[w \in \text{generated}]$ = indicator function (1 if token $w$ appears in generated text, 0 otherwise)

**Importance:**
- Reduces exact repetition of phrases and concepts
- Increases topic diversity in longer generations
- Critical for maintaining user engagement
- Produces more natural and varied text

## 11. Frequency Penalty (frequency_penalty)

**Definition:** A parameter that penalizes tokens proportionally to their frequency in the generated text.

**Mathematical Representation:**
The adjusted logits for token $w$ are:
$$\text{logit}'(w) = \text{logit}(w) - \alpha \cdot \text{count}(w)$$

Where:
- $\alpha$ = frequency_penalty value
- $\text{count}(w)$ = number of occurrences of token $w$ in generated text

**Importance:**
- Reduces overuse of specific vocabulary
- Prevents pathological repetition patterns
- Crucial for long-form content generation
- Balances topical coherence with linguistic diversity

## 12. Seed (seed)

**Definition:** An integer value that initializes the pseudo-random number generator used during sampling.

**Mathematical Representation:**
$$\text{RNG} \leftarrow \text{Initialize}(\text{seed})$$
$$P(y|x) \sim \text{Sampling using RNG}$$

Where:
- $\text{RNG}$ = random number generator state
- $\text{seed}$ = initialization value

**Importance:**
- Enables deterministic and reproducible generation
- Essential for debugging and testing
- Critical for scientific applications requiring reproducibility
- Supports A/B testing and controlled experiments

## 13. Response MIME Type (response_mime_type)

**Definition:** Parameter specifying the format in which the model's response should be returned.

**Mathematical Representation:**
Not directly expressible mathematically, but conceptually:
$$\text{Format}(y) \rightarrow \text{response\_mime\_type}$$

Where:
- $y$ = raw model generation
- $\text{Format}$ = conversion function to specified format

**Importance:**
- Enables integration with diverse application frameworks
- Critical for multi-modal applications
- Supports structured data parsing workflows
- Facilitates efficient client-side processing

## 14. Response Schema (response_schema)

**Definition:** A structured definition that constrains the format and content of the model's output.

**Mathematical Representation:**
$$P(y|x, \text{schema}) = \begin{cases}
P(y|x) & \text{if } y \text{ conforms to schema} \\
0 & \text{otherwise}
\end{cases}$$

**Importance:**
- Ensures consistent and parseable outputs
- Critical for API integration use cases
- Reduces post-processing complexity
- Enables reliable automated workflows

## 15. Routing Configuration (routing_config)

**Definition:** Specification for how generation requests are directed within distributed model serving infrastructure.

**Mathematical Representation:**
$$\text{Server} = \text{RouteSelection}(\text{request}, \text{routing\_config})$$

Where:
- $\text{RouteSelection}$ = function mapping request to appropriate servers

**Importance:**
- Enables model specialization for different request types
- Critical for optimizing resource utilization
- Supports load balancing in high-traffic deployments
- Essential for multi-model orchestration

## 16. Safety Settings (safety_settings)

**Definition:** Configuration parameters controlling the model's content filtering behavior across various safety categories.

**Mathematical Representation:**
$$P(y|x, \text{safety}) = \begin{cases}
P(y|x) & \text{if } \text{SafetyScore}(y) \leq \text{threshold} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\text{SafetyScore}(y)$ = function evaluating content risk across defined categories
- $\text{threshold}$ = acceptable safety threshold defined in settings

**Importance:**
- Customizes content filtering to application requirements
- Critical for maintaining appropriate content boundaries
- Enables use in sensitive domains and regulated industries
- Reduces liability and reputation risks in deployment


# Advanced LLM API Parameters: Tools and Multimodal Capabilities

## 1. Tools (tools)

**Definition:** A collection of function specifications that the model can reference or invoke during generation, enabling the LLM to interact with external systems, retrieve information, or perform specific operations beyond text generation.

**Mathematical Representation:**
$$\text{tools} = \{T_1, T_2, ..., T_n\}$$

Where each tool $T_i$ is defined as:
$$T_i = (f_i, \text{schema}_i, \text{description}_i)$$

The tool-augmented generation process can be modeled as:
$$P(y|x, \text{tools}) = P(y_{\text{text}}|x) \times \prod_{i} P(c_i|x, y_{1:i-1}, \text{tools})$$

Where:
- $c_i$ = tool call decision at position $i$
- $y_{\text{text}}$ = pure text generation
- $P(c_i|...)$ = probability of making specific tool calls

**Importance:**
- Expands model capabilities beyond text generation to actionable operations
- Enables interaction with external APIs, databases, and services
- Critical for creating agent-based systems and autonomous workflows
- Provides structured interfaces for complex operations like data retrieval and computation

## 2. Tool Configuration (tool_config)

**Definition:** Control parameters that determine how tools are utilized during generation, including execution behavior, calling conventions, and integration protocols.

**Mathematical Representation:**
$$\text{tool\_config} = (\text{execution\_mode}, \text{filtering\_rules}, \text{timeout}, \text{retry\_policy})$$

The tool selection probability under configuration constraints:
$$P(T_i|\text{context}, \text{tool\_config}) = \frac{\text{relevance}(T_i, \text{context}) \times \text{permission}(T_i, \text{tool\_config})}{\sum_j \text{relevance}(T_j, \text{context}) \times \text{permission}(T_j, \text{tool\_config})}$$

Where:
- $\text{relevance}(T_i, \text{context})$ = contextual relevance score of tool $T_i$
- $\text{permission}(T_i, \text{tool\_config})$ = binary function (0/1) based on tool authorization

**Importance:**
- Governs security boundaries for tool invocation
- Controls synchronous vs. asynchronous execution models
- Establishes timeout and failure handling protocols
- Critical for reliable integration with production systems

## 3. Cached Content (cached_content)

**Definition:** A pre-computed or previously generated content buffer that is provided to the model to avoid redundant computation, enable continuity across requests, or reference prior context.

**Mathematical Representation:**
The generation with cached content modifies the standard autoregressive process:
$$P(y_t|x, y_{<t}, \text{cached\_content}) = f_{\theta}(x, y_{<t}, \text{cached\_content})$$

Where:
- $f_{\theta}$ = model function with parameters $\theta$
- $y_t$ = token at position $t$
- $y_{<t}$ = tokens before position $t$

**Importance:**
- Reduces computational overhead in multi-turn interactions
- Enables efficient handling of long contexts without repeated processing
- Critical for maintaining state across interaction boundaries
- Supports streaming and incremental generation workflows

## 4. Response Modalities (response_modalities)

**Definition:** A specification of the output formats the model should generate beyond text, such as images, audio, structured data, or other media types.

**Mathematical Representation:**
$$\text{response\_modalities} = \{M_1, M_2, ..., M_k\}$$

The multimodal generation probability:
$$P(y|x) = P(y_{\text{text}}|x) \times \prod_{i \in \text{modalities}} P(y_i|x, y_{\text{text}})$$

Where:
- $M_i$ = modality type (e.g., "text", "image", "audio")
- $y_i$ = generated content in modality $i$
- $P(y_i|...)$ = conditional generation probability for modality $i$

**Importance:**
- Enables true multimodal generation capabilities
- Critical for applications requiring integrated media types
- Supports rich interactive experiences across domains
- Facilitates automated content creation across formats

## 5. Media Resolution (media_resolution)

**Definition:** Configuration parameters controlling the spatial, temporal, or quality dimensions of generated multimedia content.

**Mathematical Representation:**
$$\text{media\_resolution} = (w, h, \text{quality}, \text{format})$$

The computational complexity scales with resolution parameters:
$$\text{Complexity} \propto w \times h \times \text{quality\_factor}$$

Where:
- $w$ = width in pixels or equivalent unit
- $h$ = height in pixels or equivalent unit
- $\text{quality\_factor}$ = computational multiplier for quality level

**Importance:**
- Balances quality against computational requirements
- Critical for optimizing bandwidth in distributed applications
- Determines fidelity and detail in generated visual content
- Enables tailoring outputs to device and application constraints

## 6. Speech Configuration (speech_config)

**Definition:** Parameters controlling the generation of spoken audio, including voice characteristics, prosody, speed, and acoustic features.

**Mathematical Representation:**
$$\text{speech\_config} = (v, r, p, t, m)$$

The speech synthesis process can be modeled as:
$$A = S(T, \text{speech\_config})$$

Where:
- $v$ = voice identifier or characteristics
- $r$ = speaking rate
- $p$ = pitch parameters
- $t$ = timbre settings
- $m$ = model settings
- $S$ = speech synthesis function
- $T$ = text input
- $A$ = audio output

**Importance:**
- Controls acoustic characteristics of generated speech
- Critical for naturalistic human-machine interaction
- Enables personalization of voice interfaces
- Supports accessibility requirements across applications

## 7. Automatic Function Calling (automatic_function_calling)

**Definition:** Configuration that determines when and how the model autonomously invokes tools without explicit user direction, based on contextual understanding of user intent.

**Mathematical Representation:**
$$\text{auto\_call\_decision} = D(\text{context}, \text{tools}, \text{threshold})$$

The decision function can be formalized as:
$$D(\text{context}, \text{tools}, \text{threshold}) = \begin{cases}
\text{call}(T_i) & \text{if } \max_i P(T_i|\text{context}) > \text{threshold} \\
\text{generate\_text} & \text{otherwise}
\end{cases}$$

Where:
- $D$ = decision function
- $P(T_i|\text{context})$ = probability that tool $T_i$ is appropriate
- $\text{threshold}$ = confidence threshold for automatic invocation

**Importance:**
- Enables proactive problem-solving without explicit commands
- Reduces interaction friction in complex workflows
- Critical for creating truly autonomous assistant systems
- Balances autonomy with predictability in agent behavior
- Facilitates zero-shot tool usage based on natural language intent