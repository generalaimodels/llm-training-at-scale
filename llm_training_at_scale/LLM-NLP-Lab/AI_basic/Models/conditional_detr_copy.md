# Conditional DETR – Comprehensive Technical Breakdown  

## 1. Definition  
Transformer-based end-to-end object detector that (i) decomposes each object query $q$ into a content vector $q^{c}$ and a spatial (anchor) vector $q^{s}$ and (ii) conditions cross-attention on $q^{s}$, enabling fast convergence without post-processing (NMS).

---

## 2. Pertinent Equations  

### 2.1 Backbone  
Input image $x\!\in\!\mathbb R^{H_{0}\times W_{0}\times3}$ → CNN → feature map  
$$f=\text{Backbone}(x)\in\mathbb R^{C\times H\times W}$$  

### 2.2 Flattening & Positional Encoding  
$$z_{0} = f\_\text{flat} + p\_\text{2D},\;\; f\_\text{flat}\!\in\!\mathbb R^{(HW)\times C}$$  

### 2.3 Transformer Encoder ( $L_{e}$ layers )  
$$z_{\ell+1} = \text{EncoderLayer}_{\ell}(z_\ell),\;\; \ell=0\dots L_{e}-1$$  

### 2.4 Query Decomposition  
For $N$ learned object queries:  
$$q = q^{c} \oplus q^{s},\;\; q^{c},q^{s}\!\in\!\mathbb R^{d}$$  

### 2.5 Conditional Cross-Attention  
Given decoder input $y_{\ell}$ and encoded features $Z$:  
$$\alpha_{ij} = \text{Softmax}\!\left(\frac{\left(W_{Q}y_{\ell,i}^{c}\right)\left(W_{K}(Z_{j}+g(q^{s}_{i}))\right)^{\!\top}}{\sqrt{d}}\right)$$  
$$\text{CA}(y_{\ell,i}) = \sum_{j} \alpha_{ij}\, W_{V} Z_{j}$$  
$g(\cdot)$ transforms spatial anchor to key space.

### 2.6 Decoder Update ( $L_{d}$ layers )  
$$y_{\ell+1} = \text{DecoderLayer}_{\ell}\big(y_{\ell}^{c}\!\oplus\!q^{s},\, Z\big),\;\; \ell=0\dots L_{d}-1$$  

### 2.7 Prediction Heads  
Class logits: $$\hat{p}_{i} = \text{FC}_{\text{cls}}(y_{L_{d},i})$$  
Box: $$\hat{b}_{i} = \sigma\big(\text{FC}_{\text{box}}(y_{L_{d},i})\big)\in[0,1]^{4}$$  

---

## 3. Key Principles  
• Anchored queries accelerate convergence by giving spatial priors.  
• Set-prediction reformulation removes NMS.  
• Hungarian matching ensures one-to-one assignment of predictions to GT boxes.  
• Unified encoder–decoder allows global reasoning over all objects.

---

## 4. Detailed Concept Analysis  

### 4.1 Pre-processing  
• Resize to fixed shorter side $s$; keep aspect ratio.  
• Normalize: $x'=(x-\mu)/\sigma$.  
• Data aug.: random horizontal flip $p=0.5$, color-jitter, multi-scale training.  

### 4.2 Architecture Components  
1. Backbone (ResNet-50/101, Swin, ConvNeXt) → multi-scale features.  
2. $1\times1$ conv to project to $C\!=\!256$.  
3. Positional encodings: sine-cosine 2-D.  
4. Transformer encoder $L_{e}=6$.  
5. Decoder $L_{d}=6$ with conditional cross-attention.  
6. Shared MLP heads (class & box).  

### 4.3 Training Loss (“SetCriterion”)  
For ground-truth set $\mathcal{G}$, predicted set $\hat{\mathcal{Y}}$:  

1. Hungarian cost for permutation $\pi$:  
$$\mathcal{C}(i,\;g) = \lambda_{\text{cls}}\!\cdot\! \text{CE}\!\big(\hat{p}_{i},\, y_{g}\big) + 
\lambda_{L1}\!\cdot\! \lVert \hat{b}_{i}-b_{g}\rVert_{1} + 
\lambda_{\text{GIoU}}\!\cdot\! (1-\text{GIoU}(\hat{b}_{i},b_{g}))$$  

2. Optimal matching: $$\pi^{\star} = \arg\min_{\pi}\sum_{g\in\mathcal G}\mathcal C(\pi(g),g)$$  

3. Final loss:  
$$\mathcal{L}= \sum_{g\in\mathcal G} \big[\lambda_{\text{cls}}\;\text{CE}(\hat{p}_{\pi^{\star}(g)},y_{g}) + 
\lambda_{L1}\; \lVert \hat{b}_{\pi^{\star}(g)} - b_{g}\rVert_{1} + 
\lambda_{\text{GIoU}}\; \big(1-\text{GIoU}(\hat{b}_{\pi^{\star}(g)},b_{g})\big)\big]$$  

### 4.4 Optimizer & Schedules  
• AdamW, base LR $=1\text{e-}4$, weight decay $=1\text{e-}4$.  
• Separate LR for backbone (0.1×).  
• 300 epochs; cosine or step LR decay.  

### 4.5 Post-training Procedures  
• Exponential Moving Average (EMA) of weights (optional).  
• Model quantization (INT8) respecting activation outliers (per-channel).  
• Knowledge distillation from stronger teacher DETR variant.  

---

## 5. Importance  
• 10× faster convergence than vanilla DETR (50–75 epochs vs 500).  
• Keeps end-to-end pipeline (no proposals, no NMS).  
• Simpler hyper-parameter space than two-stage detectors.  

---

## 6. Pros vs Cons  
Pros  
• Global context via transformer → superior long-range relationships.  
• Anchors preserve spatial inductive bias → stability on small objects.  
• Single loss; fully differentiable.  

Cons  
• Higher memory than CNN-only detectors.  
• Still slower FPS than YOLO family.  
• Fixed $N$ queries may limit recall under extreme crowded scenes.  

---

## 7. Cutting-Edge Advances  
• DINO/Deformable-DETR: multi-scale deformable attention; improves small-object AP.  
• H-DETR: hierarchical queries.  
• Conditional-DINO: merges conditional queries with focal distillation.  
• QD-DETR: dynamic anchor point refinement.  

---

## 8. Step-by-Step Training Pseudo-Algorithm (PyTorch-style)  

```python
# Inputs: dataloader, model, SetCriterion, optimizer, lr_scheduler
for epoch in range(EPOCHS):
    for imgs, targets in dataloader:
        imgs = preprocess(imgs).to(device)              # §4.1
        outputs = model(imgs)                           # §§2–4
        loss_dict = criterion(outputs, targets)         # §4.3
        loss = sum(loss_dict.values())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    lr_scheduler.step()
    if use_ema: ema.update(model)                       # §4.5
```

Mathematical justification: gradient descent on $\mathcal L$ yields parameters $\theta_{t+1}=\theta_{t}-\eta \nabla_{\theta}\mathcal L$; EMA maintains $\tilde\theta_{t}=\alpha\tilde\theta_{t-1}+(1-\alpha)\theta_{t}$ to smooth updates.

---

## 9. Evaluation Phase  

### 9.1 Metrics  
• COCO mAP:  
$$\text{AP} = \frac1{|\mathcal I|}\sum_{i\in\mathcal I}\int_{0}^{1} \text{Prec}_{i}(\text{Rec})\,d\text{Rec}$$  
reported at IoU thresholds $0.50:0.05:0.95$.  

• mAP\_{50}, mAP\_{75}.  

• AR\_{1}, AR\_{10}, AR\_{100}.  

• FPS: $$\text{FPS}=\frac{|\mathcal I|}{\sum_{i} t_{i}}$$  

• Param. & GFLOPs.  

### 9.2 Evaluation Workflow  
1. Run model in eval mode; disable dropout & gradient computation.  
2. Collect $\hat{p}_{i},\hat{b}_{i}$; apply $\arg\max$ on class logits.  
3. Filter ‘no-object’ class; confidence threshold $\tau$.  
4. Directly compute IoU w.r.t. GT; no NMS required.  
5. Aggregate statistics → COCO API → AP, AR.  

### 9.3 Best Practices & Pitfalls  
• Use identical image sizes between train/test for fair FPS.  
• Ensure matching of $\hat{b}$ format (cx,cy,w,h) vs (x\_min,…).  
• Warm-up LR for first 10 epochs to stabilize transformer.  
• Monitor Hungarian cost distribution; exploding costs imply LR too high.  

---