
## 14.1 Prediction and Control

### Definition
- **Prediction and Control** in psychology refer to the ability to anticipate future events (prediction) and influence outcomes (control) based on learned associations between stimuli and responses.

### Pertinent Equations
- **General Predictive Model:**
  $$
  \hat{y} = f(x; \theta)
  $$
  where $ \hat{y} $ is the predicted outcome, $ x $ is the input (stimulus), and $ \theta $ are model parameters.

- **Control Equation (Optimal Control):**
  $$
  u^* = \arg\max_u \mathbb{E}[R(s, u)]
  $$
  where $ u^* $ is the optimal action, $ s $ is the state, and $ R $ is the reward function.

### Key Principles
- **Associative Learning:** Organisms learn to predict outcomes based on associations between stimuli.
- **Contingency:** The degree to which one event predicts another.
- **Feedback Loops:** Control is achieved by adjusting behavior based on feedback from the environment.

### Detailed Concept Analysis
- **Prediction** enables organisms to prepare for future events, increasing survival.
- **Control** allows organisms to manipulate their environment to achieve desired outcomes.
- **Learning Theories** (e.g., classical and instrumental conditioning) formalize how prediction and control are acquired.

### Importance
- Central to adaptive behavior, decision-making, and survival.
- Underpins all forms of learning and behavioral modification.

### Pros vs. Cons
- **Pros:** Enhances adaptability, efficiency, and survival.
- **Cons:** Overgeneralization or misprediction can lead to maladaptive behaviors.

### Cutting-edge Advances
- Integration of **reinforcement learning** models in neuroscience to map prediction and control mechanisms in the brain.
- Use of **deep learning** to model complex prediction/control tasks.

---

## 14.2 Classical Conditioning

### Definition
- **Classical Conditioning** is a learning process where a neutral stimulus becomes associated with a meaningful stimulus, eliciting a conditioned response.

### Pertinent Equations
- **Pavlovian Association:**
  $$
  CS + US \rightarrow UR
  $$
  $$
  CS \rightarrow CR
  $$
  where $ CS $ = conditioned stimulus, $ US $ = unconditioned stimulus, $ UR $ = unconditioned response, $ CR $ = conditioned response.

### Key Principles
- **Acquisition:** Formation of the CS-US association.
- **Extinction:** Reduction of CR when CS is presented without US.
- **Spontaneous Recovery:** Reappearance of CR after extinction.

### Detailed Concept Analysis
- **Temporal Contiguity:** CS and US must be presented closely in time.
- **Stimulus Generalization:** CR occurs to stimuli similar to CS.
- **Stimulus Discrimination:** Ability to distinguish between different stimuli.

### Importance
- Fundamental mechanism for learning in animals and humans.
- Basis for understanding phobias, preferences, and aversions.

### Pros vs. Cons
- **Pros:** Explains a wide range of learned behaviors.
- **Cons:** Limited in explaining complex, voluntary behaviors.

### Cutting-edge Advances
- Neural correlates of classical conditioning identified using fMRI and electrophysiology.
- Computational models (e.g., Rescorla–Wagner, TD) formalize learning dynamics.

---

## 14.2.1 Blocking and Higher-order Conditioning

### Definition
- **Blocking:** Prior learning of a CS-US association prevents learning about a new CS when both are paired with the US.
- **Higher-order Conditioning:** A CS becomes associated with another CS, not directly with the US.

### Pertinent Equations
- **Blocking (Rescorla–Wagner):**
  $$
  \Delta V_{B} = \alpha \beta (\lambda - V_{A} - V_{B})
  $$
  If $ V_{A} $ is maximal, $ \Delta V_{B} \approx 0 $.

### Key Principles
- **Prediction Error:** Learning occurs only when there is a discrepancy between expected and actual outcomes.

### Detailed Concept Analysis
- **Blocking** demonstrates that mere contiguity is insufficient; prediction error drives learning.
- **Higher-order Conditioning** extends associative learning beyond direct CS-US pairings.

### Importance
- Refines understanding of associative learning mechanisms.

### Pros vs. Cons
- **Pros:** Explains limitations of associative learning.
- **Cons:** Not all phenomena fit blocking/higher-order models.

### Cutting-edge Advances
- Neural evidence for prediction error signals in dopamine neurons.

---

## 14.2.2 The Rescorla–Wagner Model

### Definition
- A mathematical model describing how associative strengths are updated during conditioning.

### Pertinent Equations
- **Rescorla–Wagner Rule:**
  $$
  \Delta V = \alpha \beta (\lambda - V)
  $$
  where:
  - $ \Delta V $: Change in associative strength
  - $ \alpha $: Salience of CS
  - $ \beta $: Salience of US
  - $ \lambda $: Maximum associative strength (US magnitude)
  - $ V $: Current associative strength

### Key Principles
- **Prediction Error:** Learning is proportional to the difference between expected and actual outcomes.

### Detailed Concept Analysis
- **Acquisition:** $ V $ increases as $ (\lambda - V) $ is large.
- **Extinction:** $ \lambda = 0 $, so $ V $ decreases.
- **Blocking:** If $ V $ is maximal, $ \Delta V \approx 0 $ for new CS.

### Importance
- Quantitative framework for classical conditioning.
- Predicts phenomena like blocking, overshadowing, and extinction.

### Pros vs. Cons
- **Pros:** Simple, predictive, widely validated.
- **Cons:** Cannot explain all learning phenomena (e.g., latent inhibition).

### Cutting-edge Advances
- Extensions to account for attentional and contextual effects.

---

## 14.2.3 The TD Model

### Definition
- **Temporal Difference (TD) Model:** A reinforcement learning algorithm modeling how predictions about future rewards are updated over time.

### Pertinent Equations
- **TD Error:**
  $$
  \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
  $$
  - $ \delta_t $: TD error
  - $ r_{t+1} $: Reward at time $ t+1 $
  - $ \gamma $: Discount factor
  - $ V(s) $: Value of state $ s $

- **Value Update:**
  $$
  V(s_t) \leftarrow V(s_t) + \alpha \delta_t
  $$

### Key Principles
- **Bootstrapping:** Updates predictions based on other predictions.
- **Online Learning:** Updates occur at each time step.

### Detailed Concept Analysis
- **TD Learning** bridges classical conditioning and reinforcement learning.
- **Dopamine Neurons:** Empirically shown to encode TD error signals.

### Importance
- Explains real-time learning and prediction in animals and humans.

### Pros vs. Cons
- **Pros:** Models sequential prediction, aligns with neural data.
- **Cons:** Requires parameter tuning, may not capture all learning dynamics.

### Cutting-edge Advances
- Deep TD learning (e.g., Deep Q-Networks) for complex environments.

---

## 14.2.4 TD Model Simulations

### Definition
- Computational simulations of the TD model to replicate and predict learning behaviors.

### Pertinent Equations
- **TD Update (as above):**
  $$
  V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
  $$

### Key Principles
- **Simulation:** Iterative application of TD updates to model learning curves.

### Detailed Concept Analysis
- **Parameter Sensitivity:** Learning rate ($ \alpha $) and discount factor ($ \gamma $) affect convergence and stability.
- **Behavioral Prediction:** Simulations reproduce empirical learning curves and blocking effects.

### Importance
- Validates theoretical models against experimental data.

### Pros vs. Cons
- **Pros:** Enables hypothesis testing, parameter exploration.
- **Cons:** May oversimplify biological complexity.

### Cutting-edge Advances
- Integration with neural network models for high-dimensional state spaces.

---

## 14.3 Instrumental Conditioning

### Definition
- **Instrumental (Operant) Conditioning:** Learning process where behavior is modified by its consequences (rewards or punishments).

### Pertinent Equations
- **Reinforcement Learning Update:**
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
  $$
  where $ Q(s, a) $ is the value of action $ a $ in state $ s $.

### Key Principles
- **Law of Effect:** Behaviors followed by positive outcomes are strengthened.
- **Contingency and Contiguity:** Outcome must be contingent and temporally close to behavior.

### Detailed Concept Analysis
- **Schedules of Reinforcement:** Fixed/variable ratio/interval schedules affect learning rates and behavior patterns.
- **Shaping:** Gradual reinforcement of successive approximations to target behavior.

### Importance
- Foundation for behavior modification, therapy, and animal training.

### Pros vs. Cons
- **Pros:** Explains voluntary, goal-directed behavior.
- **Cons:** Less effective for involuntary or reflexive responses.

### Cutting-edge Advances
- Model-based vs. model-free RL distinctions in neuroscience.

---

## 14.4 Delayed Reinforcement

### Definition
- **Delayed Reinforcement:** The consequence (reward/punishment) is delivered after a temporal delay following the behavior.

### Pertinent Equations
- **Discounted Reward:**
  $$
  R_{delayed} = \gamma^d R
  $$
  where $ d $ is the delay, $ \gamma $ is the discount factor ($ 0 < \gamma < 1 $).

### Key Principles
- **Temporal Discounting:** Value of reward decreases with delay.
- **Impulsivity:** Preference for immediate over delayed rewards.

### Detailed Concept Analysis
- **Delay of Gratification:** Ability to wait for larger, delayed rewards is linked to self-control.
- **Neural Correlates:** Prefrontal cortex involvement in delayed reinforcement processing.

### Importance
- Critical for understanding addiction, self-control, and decision-making.

### Pros vs. Cons
- **Pros:** Models real-world decision-making.
- **Cons:** Individual differences in discounting rates.

### Cutting-edge Advances
- Computational psychiatry: modeling impulsivity and addiction.

---

## 14.5 Cognitive Maps

### Definition
- **Cognitive Map:** Mental representation of spatial relationships and environmental layout.

### Pertinent Equations
- **Graph Representation:**
  $$
  G = (V, E)
  $$
  where $ V $ are locations (nodes), $ E $ are paths (edges).

- **Shortest Path (Dijkstra’s Algorithm):**
  $$
  d(v) = \min_{u \in neighbors(v)} [d(u) + w(u, v)]
  $$

### Key Principles
- **Latent Learning:** Learning occurs without immediate reinforcement, revealed when needed.
- **Spatial Navigation:** Use of internal maps for efficient movement.

### Detailed Concept Analysis
- **Place Cells:** Neurons in hippocampus encoding specific locations.
- **Grid Cells:** Neurons encoding metric properties of space.

### Importance
- Explains flexible, goal-directed navigation in animals and humans.

### Pros vs. Cons
- **Pros:** Accounts for complex, flexible behavior.
- **Cons:** Hard to directly observe cognitive maps.

### Cutting-edge Advances
- Neural decoding of spatial maps using fMRI and electrophysiology.

---

## 14.6 Habitual and Goal-directed Behavior

### Definition
- **Habitual Behavior:** Actions performed automatically, triggered by cues, insensitive to outcome value.
- **Goal-directed Behavior:** Actions performed with consideration of expected outcomes and their value.

### Pertinent Equations
- **Model-free (Habitual) RL:**
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
  $$
- **Model-based (Goal-directed) RL:**
  $$
  Q(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q(s', a')]
  $$

### Key Principles
- **Dual-process Theory:** Coexistence of habitual and goal-directed systems.
- **Outcome Sensitivity:** Goal-directed actions are sensitive to outcome devaluation; habits are not.

### Detailed Concept Analysis
- **Transition from Goal-directed to Habitual:** With repetition, control shifts from goal-directed to habitual.
- **Neural Substrates:** Prefrontal cortex (goal-directed), dorsolateral striatum (habitual).

### Importance
- Explains flexibility vs. efficiency in behavior.
- Relevant to addiction, OCD, and behavioral interventions.

### Pros vs. Cons
- **Pros:** Explains both flexible and automatic behaviors.
- **Cons:** Boundary between systems can be ambiguous.

### Cutting-edge Advances
- Computational models integrating arbitration between systems.
- Neuroimaging studies mapping transitions in real time.

---