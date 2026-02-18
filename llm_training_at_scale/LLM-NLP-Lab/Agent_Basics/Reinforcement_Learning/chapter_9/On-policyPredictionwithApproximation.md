# On-Policy Prediction with Function Approximation  

---

## 9.1 Value-Function Approximation  

### Definition  
Estimate the state-value $v_\pi(s)\approx \hat v(s,\boldsymbol w)$ or action-value $q_\pi(s,a)\approx \hat q(s,a,\boldsymbol w)$ using a parameter vector $\boldsymbol w\in\mathbb R^d$ when $|\mathcal S|$ is large/continuous.

### Pertinent Equations  
$$\hat v(s,\boldsymbol w)=\sum_{i=1}^{d} w_i\,\phi_i(s)=\boldsymbol w^\top\boldsymbol\phi(s)$$  
$$\hat q(s,a,\boldsymbol w)=\sum_{i=1}^{d} w_i\,\phi_i(s,a)$$

### Key Principles  
• Compress infinite/large state spaces into $d\ll|\mathcal S|$ parameters.  
• Bias–variance trade-off arises from approximation.  

### Detailed Concept Analysis  
Linear, nonlinear (NNs), kernel, memory-based, etc. Representation affects convergence & sample efficiency.

### Importance  
Enables RL in continuous/high-dimensional environments (robotics, games, finance).  

### Pros vs Cons  
+ Scalability, generalization  
− Approximation error, potential divergence  

### Cutting-Edge Advances  
Deep value networks, residual networks, transformer-based state encoders integrating with actor–critic architectures.

---

## 9.2 Prediction Objective (VE)  

### Definition  
Minimize mean-squared value error under on-policy state distribution $d_\pi$.

### Pertinent Equations  
$$\text{VE}(\boldsymbol w)=\sum_{s} d_\pi(s)\big[v_\pi(s)-\hat v(s,\boldsymbol w)\big]^2$$  
Gradient: $$\nabla_{\boldsymbol w}\text{VE}= -2\sum_{s} d_\pi(s)\big[v_\pi(s)-\hat v(s,\boldsymbol w)\big]\boldsymbol\phi(s)$$

### Key Principles  
• True $v_\pi$ unknown; use bootstrapping to obtain targets.  
• On-policy sampling yields $d_\pi$ naturally.

### Detailed Concept Analysis  
Monte-Carlo target: $G_t$; TD target: $R_{t+1}+\gamma\hat v(S_{t+1},\boldsymbol w)$; eligibility traces blend both.

### Importance  
Defines learning criterion aligning estimator with policy’s visitation frequencies.

### Pros vs Cons  
+ Natural weighting  
− Ignores rarely visited but important states in off-policy tasks.

### Cutting-Edge Advances  
Emphasis weighting & interest functions to re-shape $d_\pi$; risk-aware objectives.

---

## 9.3 Stochastic-Gradient & Semi-Gradient Methods  

### Definition  
Incremental updates of $\boldsymbol w$ via sampled targets; semi-gradient ignores target’s dependency on $\boldsymbol w$.

### Pertinent Equations  
TD(0) semi-gradient:  
$$\delta_t=R_{t+1}+\gamma \hat v(S_{t+1},\boldsymbol w_t)-\hat v(S_t,\boldsymbol w_t)$$  
$$\boldsymbol w_{t+1}= \boldsymbol w_t+\alpha\,\delta_t\,\boldsymbol\phi(S_t)$$  
With eligibility traces ($\lambda$):  
$$\boldsymbol e_t=\gamma\lambda\boldsymbol e_{t-1}+ \boldsymbol\phi(S_t)$$  
$$\boldsymbol w_{t+1}= \boldsymbol w_t+\alpha\,\delta_t\,\boldsymbol e_t$$

### Key Principles  
• Semi-gradient converges for linear $\hat v$ if $\alpha$ small, $\gamma<1$.  
• True gradient methods include an extra term $\gamma\nabla_{\boldsymbol w}\hat v(S_{t+1})$.

### Detailed Concept Analysis  
Bias from bootstrapping + function approximation can cause divergence if off-policy/ nonlinear.

### Importance  
Foundation of tabular TD extended to approximation.

### Pros vs Cons  
+ Low computation/memory  
− Potential instability, sensitivity to $\alpha$

### Cutting-Edge Advances  
Adam, RMSProp for adaptive step-sizes; emphatic TD ensuring convergence off-policy.

---

## 9.4 Linear Methods  

### Definition  
Approximation is linear in parameters: $\hat v = \boldsymbol w^\top\boldsymbol\phi(s)$.

### Pertinent Equations  
Weight update identical to §9.3; convergence proven because objective is convex quadratic.

### Key Principles  
• Basis functions $\boldsymbol\phi$ determine representational power.  
• Least-Squares TD provides closed-form.

### Detailed Concept Analysis  
Projection onto span of features; error = orthogonal in $d_\pi$ metric.

### Importance  
Mathematically tractable, fast; baseline for RL research.

### Pros vs Cons  
+ Convergence guarantees, interpretability  
− Limited expressiveness

### Cutting-Edge Advances  
Sparse coding; linear features learned via unsupervised contrastive methods.

---

## 9.5 Feature Construction for Linear Methods  

### 9.5.1 Polynomials  
Definition: $\phi_{i}(s)=\prod_{j} s_j^{k_{ij}}$ up to degree $n$.  
Pros: captures smooth variations; Cons: curse of dimensionality.

### 9.5.2 Fourier Basis  
$\phi_{\mathbf c}(s)=\cos(\pi\mathbf c^\top s)$, $\mathbf c\in\{0,\dots,n\}^d$.  
Pros: orthogonal, uniform frequency coverage; Cons: global support.

### 9.5.3 Coarse Coding  
Overlapping receptive fields; each $\phi_i(s)=1$ if $s$ in region $i$.  
Pros: local generalization; Cons: fixed grids.

### 9.5.4 Tile Coding  
Multiple tilings offset; binary features.  
Equation: $\phi_i(s)=\mathbf 1\{s\in \text{tile}_i\}$.  
Pros: constant update cost; easy hashing; Cons: manual design.

### 9.5.5 Radial Basis Functions  
$\phi_i(s)=\exp\!\big(-\tfrac{\|s-\mu_i\|^2}{2\sigma_i^2}\big)$.  
Pros: smooth locality; Cons: bandwidth selection.

---

## 9.6 Selecting Step-Size Parameters Manually  

Definition: Choose $\alpha$ to balance speed/stability.  
Key Equations: Optimal scalar $\alpha^*=\dfrac{1}{\text{trace}(A)}$ for linear TD where $A$ is expected Hessian.  
Principles: Normalizing features helps.  
Advances: Auto-tuning rules, line search, meta-gradient $\nabla_{\alpha}J$.

---

## 9.7 Nonlinear Function Approximation: Artificial Neural Networks  

Definition: $\hat v(s,\boldsymbol w)=f_{\text{NN}}(s;\boldsymbol w)$ where $f$ is multi-layer perceptron, CNN, RNN, Transformer.

Equations: Back-prop update  
$$\boldsymbol w_{t+1}= \boldsymbol w_t+\alpha\,\delta_t\,\nabla_{\boldsymbol w}\hat v(S_t,\boldsymbol w_t)$$

Key Principles  
• Universal approximation;  
• Non-convex optimization;  
• Target network stabilization in DQN.

Importance  
Enabled deep RL breakthroughs (Atari, Go).

Pros vs Cons  
+ High expressiveness  
− Instability, sample inefficiency

Cutting-Edge Advances  
Implicit actor–critic preconditioning, Recurrent memory, Attention for long-horizon tasks.

---

## 9.8 Least-Squares TD (LSTD)  

Definition  
Solves for $\boldsymbol w$ via linear system using batch data.

Equations  
$$A=\sum_t \boldsymbol\phi(S_t)\big[\boldsymbol\phi(S_t)-\gamma\boldsymbol\phi(S_{t+1})\big]^\top$$  
$$\boldsymbol b=\sum_t \boldsymbol\phi(S_t) R_{t+1}$$  
$$\boldsymbol w=A^{-1}\boldsymbol b$$  

Key Principles  
• Exact solution in feature space;  
• Data-efficient.

Pros vs Cons  
+ Fast convergence, no $\alpha$ tuning  
− $O(d^3)$ invert cost; numerical issues.

Advances  
LSTD-Q, incrementally maintained inverse via Sherman–Morrison.

---

## 9.9 Memory-Based Function Approximation  

Definition  
Store transitions, approximate value via nearest neighbours or kernel regression.

Equations  
$$\hat v(s)=\frac{\sum_{i} K(s,s_i) G_i}{\sum_{i} K(s,s_i)}$$  

Key Principles  
• Non-parametric; adapts with data.

Pros vs Cons  
+ No model bias  
− Scaling with memory

Advances  
Experience replay with prioritized sampling; differentiable neural dictionaries.

---

## 9.10 Kernel-Based Function Approximation  

Definition  
Use RKHS: $\hat v(s)=\sum_{i} \alpha_i k(s,s_i)$.

Equation  
TD kernel update for coefficient $\alpha$ vectors analogous to §9.3.

Principles  
• Implicit high-dimensional mapping;  
• Sparsification techniques control complexity.

Advances  
Gaussian process TD; random Fourier features for scalability.

---

## 9.11 Looking Deeper at On-Policy Learning: Interest and Emphasis  

Definition  
Weight updates by user-defined interest $i_t$ and derived emphasis $m_t$ to guarantee convergence off-policy with function approximation.

Equations  
$$m_t=\lambda i_t + (1-\lambda)\sum_{k=0}^{t} (\gamma\lambda)^{t-k} i_k$$  
Update  
$$\boldsymbol w_{t+1}= \boldsymbol w_t+ \alpha\,m_t\,\delta_t\,\boldsymbol e_t$$

Principles  
• Re-weights states to modulate learning focus;  
• Emphatic TD stabilizes off-policy linear TD.

Pros vs Cons  
+ Convergent with arbitrary interest  
− Additional computation, parameter tuning

Advances  
Extension to nonlinear nets, actor–critic versions with emphatic weightings.