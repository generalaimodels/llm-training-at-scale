# I. Definition  
- **Backpropagation:** algorithm to compute $\nabla_\theta L$ for a feed-forward network via recursive application of the chain rule on its computational graph.  

# II. Pertinent Equations  
1. Forward pass per layer $l$:  
   $$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},\quad a^{(l)}=f\bigl(z^{(l)}\bigr),\;a^{(0)}=x$$  
2. Output-layer error:  
   $$\delta^{(L)}=\nabla_{z^{(L)}}L\;=\;\frac{\partial L}{\partial a^{(L)}}\odot f'\bigl(z^{(L)}\bigr)$$  
3. Recursive error for $l=L-1,\dots,1$:  
   $$\delta^{(l)}=\bigl(W^{(l+1)}\bigr)^{T}\,\delta^{(l+1)}\odot f'\bigl(z^{(l)}\bigr)$$  
4. Parameter gradients:  
   $$\frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}\bigl(a^{(l-1)}\bigr)^{T},\quad \frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}$$  

# III. Key Principles  
- Chain rule on directed acyclic graph  
- Local gradients $f'(z^{(l)})$  
- Reuse intermediate activations/gradients (dynamic programming)  

# IV. Detailed Concept Analysis  
- **Shapes:** $W^{(l)}\in\mathbb R^{n_l\times n_{l-1}},\;a^{(l)}\in\mathbb R^{n_l},\;\delta^{(l)}\in\mathbb R^{n_l}$.  
- **Complexity:** $\mathcal O\bigl(\sum_l n_l\,n_{l-1}\bigr)$ per sample.  
- **Numerical Stability:** saturating $f(\cdot)$ (e.g.\ sigmoid) → vanishing $\delta^{(l)}$.  

# V. Importance  
- Enables efficient gradient computation in $\mathcal O(\text{parameters})$  
- Foundation of deep learning optimizers and autodiff  

# VI. Pros vs Cons  
• Pros:  
  - Generality: any differentiable DAG  
  - Efficiency: reuses computations  
• Cons:  
  - Memory intensive (stores $a^{(l)},z^{(l)}$)  
  - Vanishing/exploding gradients in deep/RNNs  

# VII. Cutting-Edge Advances  
- **Gradient checkpointing:** trades compute for memory by recomputing activations  
- **Reversible networks:** eliminate activation storage  
- **Higher-order backprop:** compute Hessian–vector products (Pearlmutter’s R-operator)  

# VIII. Training Pseudo-Algorithm  
```
Input: data {(x_i,y_i)}_{i=1}^N, epochs E, lr η
Initialize: {W^{(l)},b^{(l)}} via suitable scheme
for epoch=1…E:
  for each batch B:
    # Forward
    for l=1…L:
      z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
      a^{(l)}=f(z^{(l)})
    # Loss
    L=ℓ(a^{(L)},y)
    # Backward
    δ^{(L)}=∂_aℓ⊙f'(z^{(L)})
    for l=L−1…1:
      δ^{(l)}=(W^{(l+1)})^Tδ^{(l+1)}⊙f'(z^{(l)})
    # Gradients
    for l=1…L:
      ∂W^{(l)}=δ^{(l)}(a^{(l-1)})^T ; ∂b^{(l)}=δ^{(l)}
      W^{(l)}←W^{(l)}−η∂W^{(l)}
      b^{(l)}←b^{(l)}−η∂b^{(l)}
```

# IX. Post-Training Procedures  
- **Fine-tuning:** freeze lower layers, backpropagate on new task  
- **Calibration:** temperature scaling  
  $$\hat p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$  

# X. Evaluation Metrics  
- **Cross-entropy loss:**  
  $$L=-\frac1N\sum_{i=1}^N\sum_c y_{i,c}\log\hat y_{i,c}$$  
- **Accuracy:**  
  $$\mathrm{Acc}=\frac1N\sum_{i=1}^N\mathbf1(\hat y_i=y_i)$$  
- **Precision/Recall/F1:**  
  $$P=\frac{TP}{TP+FP},\;R=\frac{TP}{TP+FN},\;F1=2\frac{P\,R}{P+R}$$  
- **Domain-specific (e.g. BLEU in MT):**  
  $$\mathrm{BLEU}=BP\exp\Bigl(\sum_nw_n\log p_n\Bigr)$$