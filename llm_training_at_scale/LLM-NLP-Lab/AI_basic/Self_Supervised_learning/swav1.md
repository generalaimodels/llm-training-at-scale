
# SwAV (Swapping Assignments between Views)

---

## Definition
Self-supervised visual representation learner that jointly performs online clustering and contrastive prediction by matching prototype assignments between multiple augmented views of the same image without explicit negative samples.

---

## Pertinent Equations

### Embedding & Prototype Similarity
$$
\mathbf{z}_v = \frac{f_\theta(\mathbf{x}_v)}{\lVert f_\theta(\mathbf{x}_v) \rVert_2},\quad
s_{vk} = \frac{\mathbf{c}_k^\top \mathbf{z}_v}{\tau}
$$

### Probability over Prototypes
$$
p_{vk} = \frac{\exp(s_{vk})}{\sum_{k'=1}^{K}\exp(s_{vk'})}
$$

### Optimal Transport Assignment (Sinkhorn-Knopp, $\epsilon$-entropy reg.)
$$
\max_{Q \in \mathcal{S}} \;\langle Q , S \rangle - \epsilon \sum_{i,k} Q_{ik}\log Q_{ik}
$$
subject to $\mathcal{S}=\{Q\!\in\!\mathbb{R}_{+}^{B\times K}\mid Q\mathbf{1}= \tfrac{1}{B}\mathbf{1},Q^\top\mathbf{1}= \tfrac{1}{K}\mathbf{1}\}$

### Swapped Prediction Loss (per batch)
$$
\mathcal{L}_{\text{SwAV}} = -\frac{1}{V_\text{pos}}\sum_{\substack{(v_a,v_b)\\\text{same image}}}\sum_{i=1}^{B}\sum_{k=1}^{K}Q^{(v_a)}_{ik}\,\log p^{(v_b)}_{ik}
$$

### Prototype Normalization
$$
\mathbf{c}_k \leftarrow \frac{\mathbf{c}_k}{\lVert \mathbf{c}_k\rVert_2}
$$

---

## Key Principles
- Online clustering via balanced assignments prevents representation collapse.  
- Multi-crop augmentations yield scale-consistent embeddings at low memory.  
- Swapped assignments enforce view-invariant prototype prediction.  
- No explicit negatives → memory-efficient, faster convergence.

---

## Detailed Concept Analysis

### Architecture

| Block | Symbol | Role |
|-------|--------|------|
| Backbone CNN/ViT | $f_\theta$ | Maps image to $d$-dim embedding |
| Prototype Layer | $C\in\mathbb{R}^{K\times d}$ | Learnable cluster centroids ($K\!\approx$3-6k) |
| Optional Projection MLP | $g_\theta$ | Non-linear map before $\ell_2$-norm |

### Data Pre-processing
- $T_H$: two high-res crops (e.g., $224^2$).  
- $T_L$: $(V-2)$ low-res crops (e.g., $96^2$).  
$$
\{\mathbf{x}_v\}_{v=1}^{V} = \{T_v(\mathbf{x})\}
$$
Augmentations: random resize-crop, color jitter, grayscale, blur, flip.

### Assignment Computation (Sinkhorn Pseudo-code)
```
Input S = exp(Scores/ε), total row=B, col=K
u ← 1/B · 1_B ; v ← 1/K · 1_K
for it=1..T_sinkhorn:
    u ← 1_B ./ (S v)          # Row normalization
    v ← 1_K ./ (S^T u)        # Column normalization
Q ← diag(u) · S · diag(v)
Return Q / sum(Q)             # Doubly stochastic
```

### Training Algorithm (1 iteration)
```
Sample batch {x_i}_{i=1}^B
Generate V crops per image
Compute normalized embeddings {z_v}
For high-res views only:
    Q_v ← Sinkhorn(S_v)               # Stop-grad on z_v
For every pair (v_a,v_b) from same image where v_a has Q:
    p_{v_b} ← softmax(C z_{v_b} / τ)
    Accumulate L += - Σ_i Σ_k Q^{(v_a)}_{ik} log p^{(v_b)}_{ik}
θ,C ← OptimizerStep(∇_θ,C L)
Normalize prototypes C
```

---

## Importance
- Achieves SOTA linear-probe accuracy with small batch sizes.  
- Removes need for memory banks/queues.  
- Applicable to varied modalities (audio, video, medical).

---

## Pros versus Cons

### Pros
- No negative pairs → reduced batch/memory.  
- Balanced assignments mitigate collapse.  
- Multi-crop aug. speeds training.

### Cons
- Sinkhorn adds computational overhead.  
- Prototype collapse if $K$ too small or temperature $\tau$ too low.  
- Sensitive to crop-scale design.

---

## Cutting-Edge Advances
- **SwAV-ViT**: adaptation to vision transformers with token pooling.  
- **DINO** & **SwaV-Momentum**: merge EMA teacher with swapped-assignment loss.  
- **Multi-Modal SwAV**: prototypes shared across image-text pairs.  
- **Online k-means SwAV**: replaces Sinkhorn with EMA k-means for scalability.

---

## Post-Training Procedures

### Linear Evaluation
Freeze $f_\theta$, train linear classifier $W$:
$$
\hat{\mathbf{y}} = \text{softmax}(W f_\theta(\mathbf{x}))
$$
$$
\mathcal{L}_{\text{CE}} = -\sum_{c} y_c \log \hat{y}_c
$$

### Fine-tuning
Unfreeze entire network; use lower LR, weight decay.

---

## Evaluation Metrics

### Top-$k$ Accuracy
$$
\text{Top-}k = \tfrac{1}{N}\sum_{i=1}^{N}\mathbb{I}\big(y_i \in \text{rank}_k(\hat{\mathbf{y}}_i)\big)
$$

### k-NN Accuracy
Embeddings → cosine nearest neighbors ($k\!=\!20$) → majority vote.

### Transfer Detection mAP
$$
\text{mAP} = \tfrac{1}{|\mathcal{C}|}\sum_{c}\int_{0}^{1}\text{Prec}_c(r)\,dr
$$

### Loss Tracking
$$
\overline{\mathcal{L}} = \frac{1}{P}\sum_{p=1}^{P}\mathcal{L}_{\text{SwAV}}^{(p)}
$$

---

## Best Practices
- $K\!>\!$3k, $\tau\!\in\![0.1,0.2]$, $\epsilon\!=\!0.05$.  
- Prototype re-init if assignment imbalance detected.  
- Cosine LR schedule + weight decay $1\!\times\!10^{-6}$.  
- Sinkhorn iterations $T_{sink}\!=\!3$–$5$.

---

## Potential Pitfalls
- Collapse when all samples map to few prototypes.  
- Too many low-res crops diminish high-level semantics.  
- Not normalizing prototypes each step leads to training drift.

---