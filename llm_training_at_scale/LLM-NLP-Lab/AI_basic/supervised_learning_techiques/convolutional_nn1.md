# Convolutional Neural Networks (CNN)

## 1. Definition  
A CNN is a feed-forward deep neural network that applies learnable convolutional kernels to exploit spatial or temporal locality in data (1-D signals, 2-D images, 3-D videos/volumes).

---

## 2. Pertinent Equations  

### 2.1 Output Spatial Size  
$$
\begin{aligned}
L_{out} &= \Big\lfloor \tfrac{L_{in} + 2P - D\,(K-1)-1}{S} \Big\rfloor + 1\\
H_{out} &= \Big\lfloor \tfrac{H_{in} + 2P_h - D_h\,(K_h-1)-1}{S_h} \Big\rfloor + 1\\
W_{out} &= \Big\lfloor \tfrac{W_{in} + 2P_w - D_w\,(K_w-1)-1}{S_w} \Big\rfloor + 1
\end{aligned}
$$
Variables  
$L,H,W$ – signal, height, width • $P$ – padding • $D$ – dilation • $K$ – kernel size • $S$ – stride.

### 2.2 Parameter Count  
$$\text{params}=K_h K_w K_d\,C_{in}C_{out}+C_{out}$$

### 2.3 Convolution Operation (NCHW)  
$$
Y_{n,c_o,i,j,k}= \sum_{c_i=1}^{C_{in}}\sum_{p=1}^{K_h}\sum_{q=1}^{K_w}\sum_{r=1}^{K_d}
W_{c_o,c_i,p,q,r}\;X_{n,c_i,S_h i + D_h p - P_h,\; S_w j + D_w q - P_w,\; S_d k + D_d r - P_d}
$$

### 2.4 Max-Pooling  
$$
Y_{n,c,i,j,k}= \max_{\substack{0\le p<K_h\\0\le q<K_w\\0\le r<K_d}}
X_{n,c,S_h i+p,\; S_w j+q,\; S_d k+r}
$$  

---

## 3. Key Principles  
- Local receptive fields  
- Weight sharing  
- Translation equivariance  
- Hierarchical feature composition  
- Dimensional consistency via padding/stride/dilation

---

## 4. Detailed Concept Analysis  

### 4.1 1-D Convolution  
Tasks: audio, DNA, sensor streams  
- Shapes: $(N,C_{in},L_{in})\rightarrow(N,C_{out},L_{out})$  
- PyTorch: `nn.Conv1d(C_in, C_out, kernel_size=K, stride=S, padding=P, dilation=D)`  

### 4.2 2-D Convolution  
Tasks: image classification, segmentation  
- Shapes: $(N,C_{in},H_{in},W_{in})\rightarrow(N,C_{out},H_{out},W_{out})`  
- TensorFlow: `tf.keras.layers.Conv2D(filters=C_out, kernel_size=(K_h,K_w), strides=(S_h,S_w), padding='same|valid', dilation_rate=(D_h,D_w))`

### 4.3 3-D Convolution  
Tasks: video, medical volumes  
- Shapes: $$(N,C_{in},D_{in},H_{in},W_{in})\rightarrow(N,C_{out},D_{out},H_{out},W_{out})$$`  
- Heavy memory cost ⇒ use `groups`, separable, or (2+1)D factorization.

### 4.4 Padding Strategies  
- SAME: choose $P$ to keep $L_{out}=L_{in}$  
- VALID: $P=0$  
- Reflect/Replicate/Zero padding trade-offs: edge artifacts vs. computational ease.

### 4.5 Stride & Dilation  
- Large $S$ ↓ resolution ↑ speed, may lose fine detail.  
- Dilation enlarges FOV without extra params; may cause gridding artifacts.

### 4.6 Pooling Variants  
- Max-pool: preserves strongest activation; equation §2.4.  
- Avg-pool: $$Y = \frac1{K_hK_wK_d}\sum X$$  
- Global-avg-pool: collapses $(H,W,D)$ ⇒ channel descriptor.

### 4.7 Parameter & FLOPs Calculation  
FLOPs per conv layer  
$$
\text{FLOPs}=2\,K_hK_wK_d\,C_{in}C_{out}\,H_{out}W_{out}D_{out}
$$
(×2 for multiply-add).

### 4.8 Post-Training Procedures  
- Quantization: $$\hat{x}= \text{round}(x/\Delta)$$ with step $\Delta$.  
- Pruning: zero out weights with $|w|<\tau$.  
- Knowledge Distillation: $$\mathcal{L}_{KD}= \alpha\,\mathcal{L}_{CE}(y,\hat{y}) + (1-\alpha)T^2 \, \text{KL}( \sigma(z_s/T)\,\|\,\sigma(z_t/T))$$.

---

## 5. Importance  
- State-of-the-art in vision/audio tasks  
- Efficient local computation enabling edge deployment  
- Foundation for Vision Transformers (via patch embedding)  

---

## 6. Pros vs Cons  
Pros  
• Translation equivariance • Parameter efficiency • Scalability • Hardware-friendly  
Cons  
• Limited global context • Fixed kernel size • Sensitive to spatial transformations • Heavy 3-D memory footprint  

---

## 7. Cutting-Edge Advances  
- Depthwise-separable $$\downarrow$$ params: 
  $$\approx \tfrac1{C_{out}}$$
- Grouped & Shuffle Convs  
- Dilated-Residual-Dense blocks  
- Dynamic convolution (input-conditioned kernels)  
- Neural Architecture Search (e.g., EfficientNet)  
- Squeeze-and-Excitation / Attention pooling  
- ConvNeXt: conv-only models rivaling transformers  

---

## 8. Training Pseudo-Algorithm  

```python
# Pytorch-style high-level
for epoch in range(E):
    for X, y in dataloader:                 # ---- Stage-1: Data I/O
        X = augment(X)                      # ---- Stage-2: Pre-processing
        logits = model(X)                   # ---- Stage-3: Forward
        loss = criterion(logits, y)         # L = CE + λ·Reg
        loss.backward()                     # ---- Stage-4: Back-prop  (∇W)
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()                        # ---- Stage-5: LR schedule
    validate(model, val_loader)             # ---- Stage-6: Eval loop
save(model.state_dict())                    # ---- Stage-7: Checkpoint
```
Mathematically  
$$
\begin{aligned}
\hat{y} &= f_\theta(X) \\
\mathcal{L} &= -\sum_{c}y_c\log \sigma(\hat{y}_c) + \lambda\lVert\theta\rVert_2^2 \\
\theta &\leftarrow \theta - \eta \nabla_\theta\mathcal{L}
\end{aligned}
$$  

---

## 9. Evaluation Metrics  

| Task | Metric | Equation |
|------|--------|----------|
| Classification | Top-k Accuracy | $$\text{Acc}_k=\frac1N\sum_{i=1}^N\mathbb{1}[y_i\in\text{Top-}k(\hat{p}_i)]$$ |
| Detection | mAP | $$\text{mAP}=\frac1{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\int_0^1 \text{Prec}_c(r)dr$$ |
| Segmentation | IoU | $$\text{IoU}= \frac{|P\cap G|}{|P\cup G|}$$ |
| Action Recognition | Top-1, mAP (per-clip) | same as above |

Loss functions  
- Cross-Entropy: $$\mathcal{L}_{CE}$$  
- Focal: $$\mathcal{L}_{FL} = -(1-p_t)^\gamma \log p_t$$ (class-imbalance)  
- Dice: $$\mathcal{L}_{Dice}=1-\frac{2|P\cap G|}{|P|+|G|}$$ (segmentation)

Best practices  
• Balanced batches • Cosine LR decay • Weight decay 1e-4 • Mixed-precision (FP16) • EMA weights • Early-stopping on $\text{ValLoss}$.

Pitfalls  
• Padding mis-alignment • Stride-padding parity errors • Vanishing gradients in very deep convs (mitigated via residual/BN) • Over-fitting to background.

---