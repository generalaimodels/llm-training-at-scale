### Deep Belief Networks (DBN)
#### 1. Definition  
Stacked, generative, probabilistic model composed of multiple $K$ Restricted Boltzmann Machines (RBMs) trained greedily layer-wise, then fine-tuned by supervised back-propagation.

#### 2. Pertinent Equations  
RBM energy (binary‐visible, binary-hidden):  
$$E(\mathbf{v},\mathbf{h})=-\mathbf{v}^\top W\mathbf{h}-\mathbf{b}^\top\mathbf{v}-\mathbf{c}^\top\mathbf{h}$$  
Joint distribution:  
$$p(\mathbf{v},\mathbf{h})=\frac{1}{Z}\exp(-E(\mathbf{v},\mathbf{h}))$$  
Conditional posteriors (sigmoid units):  
$$p(h_j=1\mid\mathbf{v})=\sigma\big((W^\top\mathbf{v}+ \mathbf{c})_j\big),\quad p(v_i=1\mid\mathbf{h})=\sigma\big((W\mathbf{h}+ \mathbf{b})_i\big)$$  
Contrastive Divergence-$k$ weight update:  
$$\Delta W=\eta\Big(\langle\mathbf{v}\mathbf{h}^\top\rangle_{0}-\langle\mathbf{v}\mathbf{h}^\top\rangle_{k}\Big)-\lambda W$$  
DBN generative model (top two layers undirected, lower directed):  
$$p(\mathbf{v})=\sum_{\mathbf{h}^1,\dots,\mathbf{h}^K}p(\mathbf{h}^{K-1},\mathbf{h}^{K})\prod_{l=1}^{K-2}p(\mathbf{h}^{l}\mid\mathbf{h}^{l+1})\,p(\mathbf{v}\mid\mathbf{h}^1)$$  
Fine-tune via supervised loss (e.g., cross-entropy):  
$$\mathcal{L}_{CE}=-\tfrac{1}{N}\sum_{n=1}^{N}\mathbf{y}_n^\top\log\hat{\mathbf{y}}_n$$

#### 3. Key Principles  
• Layer-wise unsupervised pre-training initializes weights near optimum.  
• Each RBM learns a higher-level latent representation.  
• Greedy training mitigates vanishing gradients in deep nets.  
• Fine-tuning converts generative stacks into discriminative models.

#### 4. Detailed Concept Analysis  
• Architecture: $\mathbf{v}\!\to\!\mathbf{h}^1\!\to\!\dots\!\to\!\mathbf{h}^K$, weight matrices $W^{l}\in\mathbb{R}^{d_{l}\times d_{l+1}}$.  
• Pre-training objective approximates maximum-likelihood via CD or PCD.  
• Inference: Gibbs sampling or mean-field for generative tasks; forward pass for discrimination.  
• Regularisation: weight decay $\lambda$, sparsity constraints $||\hat{\mathbf{h}}-\rho||_2$, dropout.

#### 5. Importance  
Early breakthrough enabling very deep networks (2006). Still relevant for generative pre-training, anomaly detection, and initializing spiking DBNs.

#### 6. Pros vs Cons  
Pros:  
• Unsupervised leverage of unlabeled data.  
• Eases optimization of deep nets.  
• Generative + discriminative capability.  
Cons:  
• CD provides biased gradient.  
• Slow MCMC chains for large data.  
• Outperformed by modern autoencoders / Transformers in many tasks.

#### 7. Cutting-Edge Advances  
• Persistent CD with parallel tempering.  
• Variational/mean-field DBNs with continuous units.  
• Quantum-inspired DBNs (D-Wave).  
• Hybrid DBN-CNN for medical imaging.

#### 8. Industrial Implementation Snippets  
PyTorch RBM module (binary units):  
```python
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid)*0.01)
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))
    def sample_h(self, v):
        p = torch.sigmoid(v @ self.W + self.bh)
        return p.bernoulli(), p
    def sample_v(self, h):
        p = torch.sigmoid(h @ self.W.t() + self.bv)
        return p.bernoulli(), p
```
Layer-wise stack → `nn.ModuleList([RBM(...), ...])`, then convert to `nn.Sequential` and fine-tune with `CrossEntropyLoss`.

#### 9. End-to-End Workflow  
• Data Pre-processing: standardize/whiten $\mathbf{x}$, optionally binarise for binary RBMs: $\mathbf{v}_i=\mathbb{I}[x_i>\tau]$.  
• Model Construction: choose depths $K$, dims $d_l$, initialize $W^{l}\sim\mathcal{N}(0,\sigma^2)$.  
• Training Algorithm (pseudo-code):

```
for l in 1…K:
    for epoch in 1…E_pre:
        for minibatch v0:
            h0 ~ p(h|v0)
            vk,hk = CD_k(v0)
            ΔW = η((v0 h0ᵀ) - (vk hkᵀ))/B - λW
            update W,b,c
# stack weights into feed-forward net
for epoch in 1…E_fine:
    for minibatch (x,y):
        ŷ = f_DB N(x)
        back-prop CE loss
```

• Post-training: generative sampling via Gibbs; compress weights with pruning/quantization.  
• Best Practices: use $k=1$ CD early, increase $k$ later; monitor reconstruction error; implement momentum.

#### 10. Evaluation Metrics  
Classification: accuracy $$\text{Acc}=\frac{1}{N}\sum_{n}\mathbb{I}[\hat{y}_n=y_n]$$, F1, AUC.  
Generative: negative log-likelihood (estimated by AIS), perplexity $$\exp\big(-\tfrac{1}{N}\sum\log p(\mathbf{v}_n)\big)$$.  
Reconstruction MSE $$\tfrac{1}{N}\sum||\mathbf{v}_n-\hat{\mathbf{v}}_n||_2^2$$.


---

### Extreme Learning Machines (ELM)
#### 1. Definition  
Single-hidden-layer feed-forward network (SLFN) where input-to-hidden weights $W$ and biases $\mathbf{b}$ are randomly assigned and fixed; only output weights $\beta$ are learned analytically in one step.

#### 2. Pertinent Equations  
Hidden layer output matrix:  
$$H_{n,j}=g\big(W_j^\top\mathbf{x}_n + b_j\big),\quad n\le N,\,j\le L$$  
Closed-form output weights (ridge-regularised):  
$$\beta = (H^\top H + \lambda I)^{-1}H^\top T = H^\dagger T$$  
Prediction:  
$$\hat{\mathbf{y}} = H\beta$$  
Common $g$: sigmoid $g(z)=\frac{1}{1+\exp(-z)}$, ReLU, $\tanh$.

#### 3. Key Principles  
• Random projection to high-dimensional space makes data (near) linearly separable (Cover’s theorem).  
• Least-squares fit minimises empirical error and norm of weights, yielding good generalisation.  
• Training complexity $O(NL^2 + L^3)$ (via SVD or Cholesky).

#### 4. Detailed Concept Analysis  
• Hyper-parameters: hidden dimension $L$, weight initialisation $\mathcal{U}(-a,a)$ or orthogonal.  
• Regularisation $\lambda$ tunes bias-variance.  
• Variants: Kernel ELM (KELM) uses kernel trick, Online Sequential ELM, Hierarchical ELM.  
• Implementation: form $H$ (can be streamed), compute $\beta$ via `torch.linalg.lstsq`.

#### 5. Importance  
Provides extremely fast training (milliseconds) useful for real-time systems, embedded devices, and as baseline model.

#### 6. Pros vs Cons  
Pros:  
• No iterative weight updates.  
• Convex optimisation with global optimum.  
• Scales to big data via incremental / block ELM.  
Cons:  
• Requires large $L$ for complex tasks → memory blow-up.  
• Random weights may miss structure → performance variance.  
• Limited expressivity compared with deep nets.

#### 7. Cutting-Edge Advances  
• Deep ELM (stacked ELMs).  
• Bayesian ELM (uncertainty).  
• Hardware-friendly FPGA/ASIC ELM accelerators.  
• Sparse-ELM with $L_1$ regularisation.

#### 8. Industrial Implementation Snippets  
TensorFlow (closed-form via `tf.linalg.pinv`):  
```python
W = tf.random.uniform([d_in, L], -1., 1.)
b = tf.random.uniform([L])
H = tf.nn.relu(tf.matmul(X_train, W) + b)
beta = tf.linalg.pinv(H) @ Y_train   # Moore–Penrose
Y_pred = tf.matmul(tf.nn.relu(tf.matmul(X_test, W)+b), beta)
```

#### 9. End-to-End Workflow  
• Data Pre-processing: scale to $[-1,1]$ for sigmoids; add bias term if $g$ lacks bias.  
• Model Construction: set $L$, choose $g$, sample $W,b$.  
• Training Algorithm:

```
# build hidden layer
H = g(X @ W + b)
# compute output weights
beta = (HᵀH + λI)⁻¹ Hᵀ T   # or H† T
```

• Post-training: prune neurons with small $||\beta_j||_2$, retrain closed-form.  
• Best Practices: use orthogonal random weights for stability; cross-validate $L,\lambda$.

#### 10. Evaluation Metrics  
Regression: RMSE $$\sqrt{\tfrac{1}{N}\sum ||\hat{\mathbf{y}}_n-\mathbf{y}_n||_2^2}$$, $R^2$.  
Classification: accuracy, macro-F1, hinge loss for multiclass $$\mathcal{L}_{hinge}=\sum_n\max(0,1-\mathbf{y}_n^\top\hat{\mathbf{y}}_n)$$.  
Time-critical: throughput (samples/s), latency (ms).

---

Best Practices & Pitfalls (both models)  
• Monitor overfitting via validation set; employ early stopping.  
• Ensure numerical stability in matrix inverses (use SVD, add $\epsilon I$).  
• For DBN, initialise with Xavier/He; for ELM, keep weight variance $1/d_{in}$.  
• Reproducibility: fix RNG seeds, log hyper-params, package complete config.