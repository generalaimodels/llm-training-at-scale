## 16.1 TD-Gammon  
**Definition**  
• RL agent (Tesauro 1992-95) that learned backgammon entirely via self-play using temporal-difference learning with nonlinear function approximation.

**Pertinent Equations**  
• Value update (after each move $t$):  
$$V(s_t) \leftarrow V(s_t) + \alpha \bigl[r_{t+1} + \gamma V(s_{t+1})-V(s_t)\bigr]$$  
• Eligibility-trace variant TD($\lambda$):  
$$\Delta w = \alpha \sum_{t=0}^{T-1} \bigl(G_t^\lambda - V(s_t)\bigr)\nabla_w V(s_t)$$  
with $$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}.$$

**Key Principles**  
• Self-play → generates on-policy data.  
• TD($\lambda$) → bootstrapped credit assignment across long games.  
• Nonlinear approximator → 3-layer $n$-unit neural network.

**Detailed Concept Analysis**  
• State encoding: 198 binary/continuous features representing checker distribution, turn, doubling cube, etc.  
• Network output approximates $P(\text{white wins}\,|\,s)$.  
• After ~1.5 M self-play games → reached strong master level (≈ top human experts).  

**Importance**  
• First demonstration of NN+RL beating world champions in a non-trivial domain.  
• Validated TD($\lambda$) with function approximation ≈ seminal for modern deep RL.

**Pros vs Cons**  
Pros:  
– Sample-efficient via bootstrapping; – End-to-end learning of evaluation; – No human heuristics.  
Cons:  
– Relied on handcrafted features; – No explicit search; – Convergence not guaranteed.

**Cutting-Edge Advances**  
• Replaced manual features with CNNs on raw checker images (Silver & Huang ‘16).  
• Demonstrated continual-learning variants (Schraudolph & Dayan ‘98).

---

## 16.2 Samuel’s Checkers Player  
**Definition**  
• Earliest self-learning game program (1959-66) using α-β search, handcrafted evaluation, and adaptive weight update.

**Pertinent Equations**  
• Samuel’s linear evaluator: $$V(s) = \sum_{i=1}^{k}w_i f_i(s).$$  
• Weight update (rote-learning TD): $$w_i \leftarrow w_i + \alpha \bigl(V_\text{after} - V_\text{before}\bigr)f_i(s).$$

**Key Principles**  
• α-β pruning for look-ahead.  
• Rote-learning TD to refine evaluation.  
• Book of opening/endgame memorization.

**Detailed Concept Analysis**  
• Features: piece differential, kings, mobility, center control, etc.  
• Self-play and human games supplied trajectories.  
• Interleaved search and learning → bootstrapped evaluator.

**Importance**  
• Precursor to modern RL + search synergy.  
• Pioneered “learning to evaluate” in adversarial domains.

**Pros vs Cons**  
Pros: – Ground-breaking; – Demonstrated learning beats static heuristics.  
Cons: – Linear model limited expressivity; – Convergence sensitive; – Required manual feature crafting.

**Cutting-Edge Advances**  
• Re-implementation with TD-leaf (Baxter ‘00) achieves grandmaster strength.  
• Connects to modern TD-λ + α-β (e.g., Giraffe, AlphaZero chess engine).

---

## 16.3 Watson’s Daily-Double Wagering  
**Definition**  
• Module within IBM Watson (Jeopardy! system) optimizing bet size on Daily-Double clues using POMDP-based decision theory.

**Pertinent Equations**  
• Expected utility of wager $b$ with confidence $p$:  
$$U(b) = p\,u(S_\text{cur}+b) + (1-p)u(S_\text{cur}-b).$$  
• Optimal bet: $$b^* = \arg\max_{b\in[0,S_\text{cur}]} U(b).$$

**Key Principles**  
• Confidence estimation from QA engine.  
• Utility function shaped by final-jeopardy reachability.  
• Monte-Carlo rollouts of future board states.

**Detailed Concept Analysis**  
• State: tuple (score vector, board layout, clue values).  
• Policy searches discretized wager space with dynamic-programming value estimates.  
• Risk appetite parameterized to fund utility curvature.

**Importance**  
• Demonstrated integration of statistical QA confidence into game-theoretic wagering.  
• Contributed to Watson’s 2011 victory vs. Ken Jennings.

**Pros vs Cons**  
Pros: – Principled risk management; – Handles hidden info; – Extensible to other trivia games.  
Cons: – Requires accurate $p$; – High computational cost for rollouts.  

**Cutting-Edge Advances**  
• Bayesian deep ensembles for better confidence quantification (Yih ‘20).  
• Distributional RL wagers optimizing tail-risk (Ménard ‘22).

---

## 16.4 Optimizing Memory Control  
**Definition**  
• RL scheduling of memory operations (caching, prefetch, page replacement) to maximize throughput or minimize latency.

**Pertinent Equations**  
• State-action value: $$Q(s,a) \leftarrow Q(s,a)+\alpha\bigl(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr).$$  
• Formulated as MDP where $s$ encodes cache state, $a$ chooses evicted line.

**Key Principles**  
• High-dimensional, continuous timing signals → require function approximation.  
• Constraint RL to respect hardware limits.  
• Off-policy learning with real workloads.

**Detailed Concept Analysis**  
• Deep Q-Networks with LSTM capturing access sequences.  
• Reward shaped by bytes-served-per-cycle.  
• Deployed on ARM cores → 3-15 % latency reduction vs. LRU.

**Importance**  
• RL extends beyond games into systems optimization.  
• Online adaptation to workload drift.

**Pros vs Cons**  
Pros: – Learns emergent reuse patterns; – Hardware agnostic.  
Cons: – Safety concerns on production misses; – Training signal noisy.

**Cutting-Edge Advances**  
• Meta-RL for cross-application generalization (ICML 22).  
• Constrained policy-optimization enforcing QoS (NeurIPS 23).

---

## 16.5 Human-Level Video-Game Play  
**Definition**  
• Deep Q-Network (DQN) achieving near-human scores on Atari 2600 from raw pixels (Mnih 2015).

**Pertinent Equations**  
• Loss: $$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')}\bigl[(r+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a))^2\bigr].$$  
• Target network parameters $\theta^-$ periodically synced.

**Key Principles**  
• Experience replay → decorrelates samples.  
• CNN feature extractor → shared across games.  
• ε-greedy exploration.

**Detailed Concept Analysis**  
• Input stack of 4 frames (84×84 gray).  
• Outputs $Q$ for 18 joystick actions.  
• Trained ~200 M frames per game.

**Importance**  
• Sparked deep RL wave; proof raw sensory RL can surpass hand-crafted pipelines.

**Pros vs Cons**  
Pros: – End-to-end; – General across 49 games.  
Cons: – Sample-inefficient; – Catastrophic forgetting; – Deterministic environment reliance.

**Cutting-Edge Advances**  
• Rainbow combines 6 improvements → +124 % median.  
• MuZero learns dynamics model + planning.

---

## 16.6 Mastering the Game of Go  

### 16.6.1 AlphaGo  
**Definition**  
• Hybrid MCTS + deep NNs defeated 9-dan Lee Sedol (2016).

**Pertinent Equations**  
• Policy network pre-training cross-entropy:  
$$\mathcal{L}_{\text{pol}} = -\sum_{t}\log \pi_\theta(a_t|s_t).$$  
• Value network regression:  
$$\mathcal{L}_{\text{val}} = (z - v_\phi(s))^2.$$  
• UCB in MCTS:  
$$a^* = \arg\max_{a} \bigl(Q(s,a)+c_{\text{puct}}\,P(s,a)\tfrac{\sqrt{N(s)}}{1+N(s,a)}\bigr).$$

**Key Principles**  
• Supervised imitation → policy prior.  
• Reinforcement policy-gradient self-play fine-tuning.  
• MCTS guided by NN priors and leaf evaluators.

**Detailed Concept Analysis**  
• 48-layer ResNet on 19×19 board tensors.  
• Two networks: fast rollout policy, slow value.  
• ~40 TPU days for self-play.

**Importance**  
• Milestone analogous to Kasparov defeat; showcased search+learning synergy.

**Pros vs Cons**  
Pros: – Superhuman; – Explainable via MCTS trees.  
Cons: – Heavy compute; – Requires expert data boot-strap.

**Cutting-Edge Advances**  
• Teaching tool = Fine-tunes on human style (AlphaGo Teach).

### 16.6.2 AlphaGo Zero  
**Definition**  
• Successor training from scratch via self-play; single NN outputs $(p,v)$.

**Pertinent Equations**  
• Combined loss:  
$$\mathcal{L} = (z - v)^2 - \pi^\top\log p + \lambda||\theta||^2.$$  
• Same PUCT formula but with unified network prior.

**Key Principles**  
• No human data; – ResNet 20-40 blocks; – Iterative policy improvement loop.

**Detailed Concept Analysis**  
• Generates 29 M self-play games over 3 days (4 TPUs).  
• Elo reached 5185 (>AlphaGo Lee by 3 stones).

**Importance**  
• Showed tabula-rasa RL can achieve full superhuman reasoning.

**Pros vs Cons**  
Pros: – Removes human bias; – Simpler pipeline.  
Cons: – Even larger compute; – Harder to interpret style.

**Cutting-Edge Advances**  
• Generalized to Chess & Shogi (AlphaZero).  
• MuZero removes environment model requirement.

---

## 16.7 Personalized Web Services  
**Definition**  
• RL powering recommendation, ranking, and layout for individual users.

**Pertinent Equations**  
• Contextual bandit reward:  
$$\hat{\beta} = \arg\min_\beta \sum_{i}(r_i - x_i^\top\beta)^2 + \lambda||\beta||^2.$$  
• Policy-gradient for sequential session reward:  
$$\nabla_\theta J = \mathbb{E}\bigl[\sum_{t}\nabla_\theta \log \pi_\theta(a_t|s_t)G_t\bigr].$$

**Key Principles**  
• Off-policy counterfactual evaluation (IPS, DR).  
• Slate-RL: optimize ordered lists.  
• Fairness and diversity constraints.

**Detailed Concept Analysis**  
• Embedding of users, items, session signals via transformers.  
• Long-term value modeled via RNN-DQN with discount $\gamma\approx0.9$.  
• Real-time serving on GPU/TPU inference clusters (<10 ms latency).

**Importance**  
• Drives engagement, revenue; personalizes billions of interactions daily.

**Pros vs Cons**  
Pros: – Adaptive; – Handles delayed feedback; – Supports multi-objective (CTR, dwell).  
Cons: – Exploration risk; – Feedback loops; – Privacy considerations.

**Cutting-Edge Advances**  
• Causal RL separating preference from exposure (ICLR 24).  
• Retrieval-augmented LLM agents selecting personalized content dynamically.