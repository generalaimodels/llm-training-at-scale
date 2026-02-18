### 14.1 Prediction and Control  
**Definition**  
Prediction = estimating the future state $s_{t+1}$ given current history $H_t$;  
Control = selecting an action $a_t$ that maximizes expected utility of future states.  

**Pertinent Equations**  
• State-transition: $$p(s_{t+1}\mid H_t,a_t)$$  
• Optimal action (Bellman): $$a_t^{\star}=\arg\max_{a}\; \mathbb{E}\!\left[\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+1}\mid H_t,a\right]$$  

**Key Principles**  
• Determinism vs. stochasticity of $p(s_{t+1}|\,\cdot)$  
• Value functions $V(s)$ and $Q(s,a)$ as compressed predictors  
• Feedback loops: prediction error $\delta_t\!\equiv\! r_{t+1}+\gamma V(s_{t+1})-V(s_t)$ drives control refinement  

**Detailed Concept Analysis**  
Animals (and agents) maintain forward models $f_\theta(H_t,a_t)\!\rightarrow\!s_{t+1}$; inverse models $g_\phi(H_t,s_{t+1})\!\rightarrow\!a_t$ implement control. Biologically, cortico-striatal circuits instantiate $V/Q$ learning; cerebellum supports forward prediction.  

**Importance**  
Links associative learning with goal-directed behavior; formalizes Skinner’s “operant conditioning” and Thorndike’s “law of effect.”  

**Pros vs. Cons**  
+ Unifies perception, action, and reward under one computational umbrella  
− Requires extensive sampling; model mis-specification ⇒ maladaptive control  

**Cutting-Edge Advances**  
• World-model RL (DreamerV3, MuZero)  
• Meta-learning of predictive errors in prefrontal cortex analogues  

---

### 14.2 Classical Conditioning  
**Definition**  
Learning a predictive association between a neutral conditioned stimulus (CS) and a biologically significant unconditioned stimulus (US).  

**Pertinent Equations**  
Pavlovian contingency: $$\Delta P = p(\text{US}\mid\text{CS})-p(\text{US}\mid\neg\text{CS})$$  

**Key Principles**  
• Temporal contiguity: CS precedes US within interval $\tau$  
• Contingency not mere co-occurrence; CS must reduce uncertainty about US  

**Detailed Concept Analysis**  
Phases: acquisition, extinction, spontaneous recovery. Neural substrates: amygdala (CS-US convergence), VTA dopamine (prediction error).  

**Importance**  
Foundation for affective forecasting, phobia genesis, drug tolerance cues.  

**Pros vs. Cons**  
+ Simple, rapid learning  
− Limited to involuntary responses; stimulus specificity constraints  

**Cutting-Edge Advances**  
• Optogenetic replacement of CS with patterned neural stimulation  
• Computational psychiatry exploiting aberrant Pavlovian processes  

---

#### 14.2.1 Blocking and Higher-Order Conditioning  
**Definition**  
Blocking: prior CS$_1$-US pairing prevents CS$_2$ learning when introduced in compound (CS$_1$+CS$_2$).  
Higher-order: CS$_1$ becomes US-like, allowing CS$_2$→CS$_1$ pairing in absence of primary US.  

**Equation (Blocking Criterion)**  
$$\Delta V_{\text{CS}_2}=0\quad\text{if}\quad V_{\text{CS}_1}\approx\lambda_{\text{US}}$$  

**Key Principles**  
Learning driven by surprise (prediction error); if US already predicted, no error ⇒ no new learning.  

**Detailed Concept Analysis**  
Explains stimulus salience hierarchy; predicts overshadowing when simultaneous CSs differ in intensity.  

**Importance**  
Demonstrates associative competition, supporting error-correction theories.  

**Pros vs. Cons**  
+ Clarifies boundary conditions for cue competition  
− Cannot explain sensory pre-conditioning without extensions  

**Cutting-Edge Advances**  
• Neural imaging shows dopaminergic attenuation during blocked trials  

---

#### 14.2.2 The Rescorla–Wagner Model  
**Definition**  
Linear-error-correction rule updating associative strength $V_i$ for each CS$_i$.  

**Equation**  
$$\Delta V_i = \alpha_i\beta\,(\lambda - \sum_j V_j)$$  

Symbols: $\alpha_i$ = CS salience, $\beta$ = learning-rate tied to US, $\lambda$ = US magnitude.  

**Key Principles**  
• Global pooled prediction $\,\sum_j V_j$ drives individual updates  
• Converges when total associative strength equals US magnitude ($\lambda$)  

**Detailed Concept Analysis**  
Replicates blocking, overshadowing, extinction ($\lambda\!=\!0$). Does not capture time within trial, nor latent inhibition.  

**Importance**  
Cornerstone of quantitative learning theory; precursor to TD learning.  

**Pros vs. Cons**  
+ Analytical tractability  
− Lacks temporal dynamics; assumes constant $\alpha_i$  

**Cutting-Edge Advances**  
• Extensions: Pearce-Hall, Mackintosh models with dynamic salience  
• Bayesian RW reinterpretations integrating uncertainty  

---

#### 14.2.3 The TD (Temporal-Difference) Model  
**Definition**  
Online algorithm updating value predictions based on successive temporal differences.  

**Equations**  
Prediction error: $$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$  
Weight update: $$\theta \leftarrow \theta + \alpha\,\delta_t\,\nabla_\theta V(s_t)$$  

**Key Principles**  
• Bootstrapping: uses own prediction $V(s_{t+1})$ as proxy for future return  
• Eligibility traces ($\lambda$) distribute $\delta_t$ backward across states  

**Detailed Concept Analysis**  
TD($\lambda$) bridges Monte-Carlo ($\lambda\!=\!1$) and RW ($\lambda\!=\!0$). Dopamine neurons fire in proportion to $\delta_t$; shift from US to CS over training mirrors predictive temporal propagation.  

**Importance**  
Unified account of Pavlovian learning, dopamine signaling, and RL algorithms (Q-learning, SARSA).  

**Pros vs. Cons**  
+ Captures timing; scalable to continuous tasks  
− Requires Markov assumption; over-generalizes in partial observability  

**Cutting-Edge Advances**  
• Distributional TD (C51, QR-DQN) models full return distribution  
• TD with representation learning (Deep RL) → richer feature extraction  

---

#### 14.2.4 TD Model Simulations  
**Definition**  
Computational experiments implementing TD to replicate empirical conditioning curves.  

**Equations Implemented**  
Same as §14.2.3 plus eligibility trace:  
$$e_t = \gamma\lambda e_{t-1} + \nabla_\theta V(s_t)$$  
$$\theta \leftarrow \theta + \alpha\,\delta_t\,e_t$$  

**Key Principles**  
Parameter sweep over $(\alpha,\gamma,\lambda)$ reproduces acquisition speed, blocking, second-order conditioning.  

**Detailed Concept Analysis**  
Simulations on semi-Markov chains enable interval timing. Models extend to continuous latent states via radial basis functions or neural networks.  

**Importance**  
Bridges theoretical TD with behavioral data; guides parameter inference from neural recordings.  

**Pros vs. Cons**  
+ Generates quantitative predictions testable in labs  
− Sensitive to feature selection; simulation-realism gap  

**Cutting-Edge Advances**  
• TD-learning embedded in spiking neural networks  
• Inverse-RL to recover latent reward structures from animal trajectories  

---

### 14.3 Instrumental Conditioning  
**Definition**  
Learning action-outcome contingencies where action probability changes as a function of its consequences.  

**Equation (Law of Effect)**  
$$\Delta Q(s_t,a_t)=\alpha\,[r_{t+1}-\bar{r}]$$  

**Key Principles**  
• Response–reward contingency; outcome devaluation tests habit vs. goal-directed control  

**Detailed Concept Analysis**  
Dorsomedial striatum ⇒ goal-directed; dorsolateral ⇒ habitual. Schedules (VR, VI) modulate learning rate and resistance to extinction.  

**Importance**  
Underpins adaptive decision-making; dysregulated in addiction and OCD.  

**Pros vs. Cons**  
+ Flexible; links voluntary behavior to motivational systems  
− Credit assignment more complex than in Pavlovian tasks  

**Cutting-Edge Advances**  
• Model-based vs. model-free arbitration theories  
• Hierarchical RL capturing action chunking (options framework)  

---

### 14.4 Delayed Reinforcement  
**Definition**  
Reinforcer delivered after temporal gap $\Delta t$ following action.  

**Equation (Hyperbolic Discounting)**  
$$V = \frac{R}{1 + k\Delta t}$$  

**Key Principles**  
• Discounting curve shape (hyperbolic > exponential) captures preference reversals  
• Bridging mechanisms: secondary reinforcers, conditioned reinforcement  

**Detailed Concept Analysis**  
Prefrontal cortex sustains working-memory traces; dopamine ramping observed during delay.  

**Importance**  
Explains impulsivity, procrastination, and clinical pathologies (ADHD).  

**Pros vs. Cons**  
+ Models real-world delayed gratification  
− Hard to tease apart delay vs. probability discounting empirically  

**Cutting-Edge Advances**  
• Successor representation bridging delay via predictive states  
• Neurofeedback training to attenuate steep discounting  

---

### 14.5 Cognitive Maps  
**Definition**  
Internal representations encoding relational structure of environment beyond mere stimulus–response chains.  

**Equation (Successor Representation, SR)**  
$$M(s,s') = \mathbb{E}\!\left[\sum_{t=0}^{\infty}\gamma^{t}\,\mathbb{1}\{s_{t}=s'\}\mid s_0=s\right]$$  

**Key Principles**  
• Separates transition structure (SR) from reward vector $r$  
• Grid cells, place cells implement basis functions for SR in hippocampus/EC  

**Detailed Concept Analysis**  
SR enables rapid revaluation when rewards change but transitions remain stable.  

**Importance**  
Accounts for detour behavior, flexible planning.  

**Pros vs. Cons**  
+ Balances efficiency of model-free with flexibility of model-based RL  
− Grows quadratically with state space; needs compression  

**Cutting-Edge Advances**  
• Deep-SR using convolutional encoders  
• Graph neural nets for abstract relational maps  

---

### 14.6 Habitual vs. Goal-Directed Behavior  
**Definition**  
Habitual: stimulus-response, insensitive to outcome devaluation; Goal-directed: action selected via expected outcome value.  

**Equation (Arbitration Weight)**  
$$\pi(a|s) = w\,\pi_{\text{MB}}(a|s) + (1-w)\,\pi_{\text{MF}}(a|s)$$  
with $$w = \frac{\eta}{\eta + \kappa}$$, $\eta$ = model-based reliability, $\kappa$ = cost.  

**Key Principles**  
• Computational cost vs. flexibility trade-off  
• Chronic stress, over-training shift control toward habitual system  

**Detailed Concept Analysis**  
Posterior putamen & sensorimotor cortex mediate habits; ventromedial prefrontal & caudate support goal-directed inference.  

**Importance**  
Explains behavioral rigidity, informs interventions for compulsive disorders.  

**Pros vs. Cons**  
+ Dual-system accounts fit neurobehavioral data  
− Boundary conditions (e.g., partial habits) still debated  

**Cutting-Edge Advances**  
• Real-time fMRI to bias arbitration  
• Neuromorphic chips implementing parallel MF/MB learning pathways