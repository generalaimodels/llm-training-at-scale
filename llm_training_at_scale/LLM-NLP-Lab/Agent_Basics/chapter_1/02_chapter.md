## 1 Formal Definition of an AI Agent  
### 1.1 Definition  
An AI agent is a tuple  
$$\mathcal A = \langle \mathcal S,\;\mathcal P,\;\pi,\;\mathcal U,\;\mathcal M\rangle$$  
where:  
* $ \mathcal S $ – continuous or discrete environmental state‐space.  
* $ \mathcal P: \mathcal S \times \mathcal A \times \mathcal S \rightarrow [0,1] $ – transition kernel $P(s'|s,a)$.  
* $ \pi: \mathcal H \rightarrow \Delta(\mathcal A) $ – policy mapping histories $h_t=(s_0,a_0,\ldots,s_t)$ to a distribution over actions.  
* $ \mathcal U: \mathcal H \rightarrow \mathbb R $ – utility (objective) functional.  
* $ \mathcal M $ – internal model set (world model, belief state, or learned parameters).  

The agent maximises expected cumulative utility  
$$\max_{\pi}\;\mathbb E_{\pi,\mathcal P}\Big[\sum_{t=0}^{T} \gamma^t \, \mathcal U(h_t)\Big]$$  
with discount factor $ \gamma\in[0,1]$.

### 1.2 Pertinent Equations  
1. Perception map: $$o_t = f_{\text{sens}}(s_t)$$  
2. Belief update (Bayesian filtering):  
$$b_t = \eta\,P(o_t|s_t)\sum_{s_{t-1}} P(s_t|s_{t-1},a_{t-1})\,b_{t-1}$$  
3. Optimal action (Bellman):  
$$Q^{\pi}(s,a)=\mathcal U(s,a)+\gamma\sum_{s'}\!P(s'|s,a)\,V^{\pi}(s'),\quad a^*=\arg\max_{a}Q^{\pi}(s,a)$$  
4. Learning update (TD):  
$$\theta \leftarrow \theta + \alpha\big(r_{t+1}+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\big)\nabla_\theta V_\theta(s_t)$$  

### 1.3 Key Principles  
* Rationality: $ \pi^* = \arg\max_\pi \mathbb E[\sum_t \gamma^t \mathcal U] $.  
* Autonomy: closed perception–action loop without external micro-management.  
* Reactivity vs deliberation determined by inference horizon $h$.  
* Learning: $\partial \mathcal M /\partial t \neq 0$.  
* Embodiment: physical or virtual actuators embedded in $\mathcal E$.  

### 1.4 Detailed Concept Analysis  
* Environment classes: fully observable MDP, POMDP, adversarial games, multi-objective MDP.  
* Optimality criteria: expected return, risk‐sensitive return, regret minimisation.  
* Memory complexity: $O(|\mathcal S|)$ for tabular vs $O(\text{params})$ for function approximators.  
* Computational complexity: generally PSPACE‐complete; tractable subclasses via factored dynamics or linear MDP.  

### 1.5 Importance  
* Unifies perception, reasoning, and action under a single optimisation formalism.  
* Enables lifelong adaptation in non-stationary settings.  
* Supports decomposable architectures (modular agents, hierarchical agents).  

### 1.6 Pros vs Cons  
Pros  
• Continual improvement via feedback.  
• Scalability across tasks when combined with self-supervised models.  
Cons  
• Sample inefficiency in sparse-reward regimes.  
• Safety risks from specification gaming or distributional shift.  

### 1.7 Cutting-Edge Advances  
* Foundation‐model agents (e.g., LLM-driven planners using chain-of-thought search).  
* Differentiable simulators enabling end-to-end gradient flow through $ \mathcal P $.  
* Causal RL agents leveraging structural causal models for counterfactual reasoning.  

---

## 2 Taxonomy of AI Agents  
### 2.1 Definition  
Classification function  
$$\tau: \mathcal A \rightarrow \{\text{reactive},\text{model-based},\ldots\}$$  

### 2.2 Pertinent Equations (Representative)  
1. Reactive agent: $a_t = \pi(s_t)$ with $\pi$ stateless.  
2. Model-based planning value:  
$$V(s)=\max_{a}\Big[\mathcal U(s,a)+\gamma\sum_{s'}P(s'|s,a)\,V(s')\Big]$$  
3. Multi-agent payoff matrix: $u_i(s,a_1,\dots,a_n)$ for agent $i$.  

### 2.3 Key Principles  
* Information availability ($I$): $I_{\text{reactive}}<I_{\text{model-based}}$.  
* Utility scope: goal-based vs utility-based.  
* Coordination protocol: competitive, cooperative, mixed.  

### 2.4 Detailed Concept Analysis  
| Type | Memory | Model | Example Algorithms | Use Context |  
|------|--------|-------|--------------------|-------------|  
| Reactive | None | None | $\varepsilon$-greedy, reflex rules | Low-latency control |  
| Model-Based | Internal $P$ | Learned/known | Dyna, MuZero | Data-scarce domains |  
| Goal-Based | Targets $g$ | Planner | Graph search | Task planning |  
| Utility-Based | Scalar utility | Decision theory | MCTS, UCT | Trade-off tasks |  
| Learning | $\theta$ adaptive | Gradient/EV | DQN, PPO | Non-stationary env. |  
| Communication | $\{m_t\}$ msgs | Language model | MADDPG‐Comm | Swarm robotics |  
| Multi-Agent | $\{P_i\}$ | Joint game | QMIX, MARL | Markets, games |  
| Cognitive | $\langle WM,EM,SM\rangle$ | Symbolic + sub‐symbolic | ACT-R, SOAR | Explainability |  

### 2.5 Importance  
Provides design heuristics mapping task properties to suitable agent architecture.  

### 2.6 Pros vs Cons  
Utility-Based  
• Pros: fine-grained trade-offs  
• Cons: utility specification bottleneck  
Multi-Agent  
• Pros: scalability, redundancy  
• Cons: non-stationarity, emergent conflict  

### 2.7 Cutting-Edge Advances  
* Large Language Model (LLM) agents enabling zero-shot tool use.  
* Graph-Neural‐Network-augmented swarm agents with relational inductive bias.  
* Equilibrium-finding MARL using differentiable game solvers.  

---

## 3 When to Use AI Agents  
### 3.1 Definition  
Decision criterion  
$$\text{Deploy}(\mathcal A)=\mathbf 1\big[\,\mathcal C_\text{env}\ge\theta\,\wedge\,\mathcal V_\text{ROI}\ge 0\,\big]$$  

### 3.2 Pertinent Equations  
Expected ROI  
$$\mathcal V_\text{ROI}= \mathbb E\big[\Delta \text{value} - \text{cost}_\text{compute} - \text{cost}_\text{risk}\big]$$  

### 3.3 Key Principles  
* Dynamism: high temporal variability ⇒ agentic control.  
* Partial observability: POMDP advantage over rule systems.  
* Sequential decision depth: long‐horizon credit assignment.  

### 3.4 Detailed Concept Analysis  
Suitable scenarios  
• Adaptive control in stochastic processes.  
• Tasks with uncertain, evolving goals.  
• Environments where data collection is continuous and feedback rich.  

### 3.5 Importance  
Maximises automation ROI while maintaining safety margins.  

### 3.6 Pros vs Cons  
Pros  
• Continuous optimisation, robustness to change.  
Cons  
• Overhead in monitoring, engineering, and alignment.  

### 3.7 Cutting-Edge Advances  
Auto-gating frameworks that toggle between scripted pipelines and agentic loops based on uncertainty estimators.  

---

## 4 Basics of Agentic Solutions  
### 4.1 Definition  
Agentic solution = orchestrated stack  
$$\langle \text{Perception},\text{Memory},\text{Reason},\text{Action},\text{Learning}\rangle$$  

### 4.2 Pertinent Equations  
End-to-end differentiable objective  
$$\min_\Theta \sum_{i=1}^N \Big(\ell_{\text{task}}^{(i)}+\lambda \,\ell_{\text{constraint}}^{(i)}\Big)$$  
with $\Theta$ aggregating sensory encoders, world model, policy network.  

### 4.3 Key Principles  
* Modular decoupling: $f_{\text{sens}}$, $f_{\text{model}}$, $f_{\text{policy}}$.  
* Closed-loop latency budget $t_{\text{loop}}<\tau_{\text{env}}$.  
* Safety layer enforcing $\Pr(\text{violate})<\epsilon$.  

### 4.4 Detailed Concept Analysis  
Pipeline stages  
1. Sensor fusion → latent $z_t$.  
2. Belief propagation: $b_t = g(b_{t-1},z_t)$.  
3. Planning/search over latent rollouts.  
4. Actuation with safety overrides.  
Infrastructure patterns  
• Micro-service containers for perception and control.  
• Replay buffer + off-policy learner GPUs.  

### 4.5 Importance  
Reduces coupling, aids testability, facilitates continuous deployment.  

### 4.6 Pros vs Cons  
Pros  
• Parallel scalability, component reusability.  
Cons  
• Interface brittleness, latency overhead.  

### 4.7 Cutting-Edge Advances  
Fully differentiable agent stacks (Diffuser, R2D3) and large-context memory agents using vector databases + retrieval-augmented generation.  

---

## 5 Representative Agent Use Cases  
### 5.1 Definition  
Use case $u$ characterised by $(\mathcal E_u,\mathcal U_u,\mathcal C_u)$ where $\mathcal C_u$ are constraints.  

### 5.2 Pertinent Equations  
Trajectory‐level KPI  
$$\text{KPI}(u)=\frac{1}{T}\sum_{t=0}^{T}\kappa(s_t,a_t)$$  

### 5.3 Key Principles  
Match agent architecture to KPI gradients and constraint surfaces.  

### 5.4 Detailed Concept Analysis  
| Domain | Agent Role | Mathematical Formulation | KPI |  
|--------|-----------|--------------------------|-----|  
| Robotics | $\pi_\theta: \mathbb R^{n}\!\rightarrow\!\mathbb R^{m}$ | Optimal control, joint‐space dynamics $M(q)\ddot q + C(q,\dot q)=\tau$ | Energy + task time |  
| Finance | RL trader in market MDP | Price process $dS_t=\mu S_t dt+\sigma S_t dW_t$ | Sharpe ratio |  
| NLP Tool Use | LLM agent planning API calls | Function calling graph search | Task success rate |  
| Healthcare | Treatment policy in POMDP | Patient state latent SSM | Survival/side‐effects |  
| Cyber-security | Defender agent in stochastic game | $\min_\pi \max_\alpha$ payoff | Breach probability |  

### 5.5 Importance  
Demonstrates cross-disciplinary versatility and economic value.  

### 5.6 Pros vs Cons  
Pros  
• Automates complex sequential workflows.  
Cons  
• Domain shift, regulatory hurdles.  

### 5.7 Cutting-Edge Advances  
* Retrieval-augmented LLM agents for legal reasoning.  
* Vision-language actuation agents (Gato-style) spanning 600+ tasks.  
* Graph RL in drug-discovery pipelines integrating quantum chemistry oracles.