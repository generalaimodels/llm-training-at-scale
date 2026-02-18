# Patch Embedding
## Definition  
Patch Embedding converts an input image $$\mathbf{I} \in \mathbb{R}^{C \times H \times W}$$ into a sequence of fixed-length vectors $$\{\mathbf{e}_1,\dots,\mathbf{e}_N\}$$ (tokens) suitable for transformer-based processing by partitioning (possibly overlapping) spatial regions (“patches”) and projecting each patch to an embedding space of dimension $$D$$.

---

## Relevant Equations  

$$
\begin{aligned}
&\textbf{Patch size:}\quad P_h \times P_w,\; N=\frac{H}{P_h}\cdot\frac{W}{P_w} \\
&\text{Patch flattening:}\quad
\mathbf{p}_i=\operatorname{reshape}\!\left(\mathbf{I}_{\text{patch}(i)},\,C\!\cdot\!P_h\!\cdot\!P_w\right) \\
&\text{Linear projection:}\quad
\mathbf{e}_i = \mathbf{W}_e\,\mathbf{p}_i + \mathbf{b}_e,\;\;\mathbf{W}_e\in\mathbb{R}^{D\times (C P_h P_w)} \\
&\text{Overlapping conv stem:}\quad
\mathbf{E}= \operatorname{Conv2D}(\mathbf{I};k=P,s=S,p=\lfloor\frac{k-S}{2}\rfloor) \\
&\text{Projection via $1\times1$ conv:}\quad
\mathbf{E}'=\operatorname{Conv2D}(\mathbf{E};k=1,s=1) \\
&\text{Positional encoding:}\quad
\hat{\mathbf{e}}_i=\mathbf{e}_i+\mathbf{p}_i^{\text{pos}}
\end{aligned}
$$

---

## Core Principles  

- Spatial tokenization: map 2-D pixels to 1-D token sequence.  
- Information preservation: maintain local semantics inside each patch.  
- Dimensional homogenization: produce uniform $D$-dimensional vectors compatible with transformer blocks.  
- Positional awareness: add absolute, relative, or implicit spatial cues.  
- Efficiency–fidelity trade-off: choose patch size, stride, and projection depth to balance computational cost and representation quality.

---

## Detailed Concept Analysis  

### 1. Standard Non-overlapping Linear Patchify  
1. Reshape image into $$\mathbb{R}^{N \times (C P_h P_w)}$$ by contiguous memory view.  
2. Apply single shared weight matrix $$\mathbf{W}_e$$ to all flattened patches.  
3. Add 1-D learnable absolute positional embeddings.

### 2. Overlapping Convolutional Stem  
• Use a single conv layer with kernel $$k=P$$, stride $$S<P$$ (e.g., $$k=7,S=4$$) to yield overlapping tokens $$N=\left\lceil\frac{H-k+1}{S}\right\rceil \!\cdot\! \left\lceil\frac{W-k+1}{S}\right\rceil$$.  
• Padding preserves spatial alignment; overlap captures fine detail lost in rigid partitioning.  

### 3. Multi-stage / Hierarchical Patch Embedding  
• Successive conv/down-sampling blocks (e.g., Swin, PVT, ViT-Hybrid) gradually reduce spatial resolution while increasing channel dimension $$C\!\rightarrow\!D$$.  
• Enables pyramid features and linear complexity self-attention in local windows.

### 4. Hybrid CNN–ViT Embedding  
• Replace first $$L_{\text{cnn}}$$ layers of a CNN backbone (ResNet, ConvNeXt) as feature extractor  
  $$\mathbf{F}_0 = \operatorname{CNN}_{1:L_{\text{cnn}}}(\mathbf{I})$$  
• Flatten $$\mathbf{F}_0$$ to tokens and linearly project to $$D$$.  
• Retains inductive biases (translation equivariance, locality).

### 5. Dynamic / Adaptive Patchification  
• Allocate variable patch size per region via saliency or reinforcement learning; preserves high-detail zones (faces, text) with smaller patches.  
• Method: Gumbel-softmax sampling of patch sizes, differentiable routing to multiple projection heads.

### 6. Learnable Patch Tokens  
• Instead of hard partitioning, learn a set of $$K$$ content-aware aggregation kernels $$\{\mathbf{A}_k\}$$:  
  $$\mathbf{e}_k = \sum_{x,y} \alpha_{k,x,y}\,\mathbf{I}_{:,x,y},\quad \alpha_{k,x,y} = \frac{\exp(\mathbf{q}_k^\top \mathbf{I}_{:,x,y})}{\sum_{x',y'}\exp(\mathbf{q}_k^\top \mathbf{I}_{:,x',y'})}$$  
• Produces soft, content-driven tokens (e.g., Slot Attention, TokenLearner).

### 7. Fourier / Wavelet Patch Embedding  
• Transform patches into frequency domain prior to projection; preserves global texture statistics and enables low-pass/high-pass token mixing.

---

## Importance  

- Foundation of Vision Transformers; governs computational footprint $$\mathcal{O}(N^2)$$ in self-attention.  
- Directly influences model accuracy, inductive bias, and downstream transferability.  
- Crucial for multimodal fusion; harmonizes spatial scale with text token length.

---

## Pros vs. Cons  

| Aspect | Pros | Cons |
|--------|------|------|
| Non-overlapping linear | Simplest, GPU-friendly, weight sharing | Ignores inter-patch continuity, coarse granularity |
| Overlapping conv | Captures local edge info, mitigates block artifacts | Higher memory, slight translational bias |
| Hierarchical | Multi-scale, linear attention windows | Added architectural complexity |
| CNN-Hybrid | Strong low-level inductive bias, better small-data performance | Deviates from pure transformer paradigm |
| Adaptive/Dynamic | Content-aware resolution, efficient token budget | Requires controller training, non-uniform token sizes |
| Learnable tokens | Compact, focuses on informative regions | Additional attention/selection module overhead |

---

## Cutting-edge Developments  

- Masked Autoencoders (MAE): restrict encoder to visible patch subset; patchify stem unchanged but leverage 75–90 % masking.  
- PatchDrop / PatchOut: stochastic dropping of redundant tokens during training for efficiency.  
- Multi-resolution Tokenizer (ColT5-Imag): joint grid + random crops fed into shared embedding table.  
- PatchMerger / Token Merging: learnable pooling after early layers to reduce sequence length on-the-fly.  
- Deformable Patch Embedding (DiNAT, DPT): spatially deformable conv kernels adapt to object geometry.  
- PEPE (Positional Encoding for Patch Embedding): learnable polynomial functions generate continuous 2-D positions, outperforming fixed sinusoids.  
- Spectral Patch Embedding: hybrid discrete cosine basis + linear projection; enhances compression for high-res imagery.  
- Hardware-aware design: FP16-friendly conv-stem with stride-mixing to exploit tensor cores; deployed in ViT-GPU inferencing libraries (TensorRT-ViT).


-----
### example


## 1. Input Image

* **I** ∈ ℝ^\[B × C × H × W]
* Example: B=1, C=3, H=224, W=224

---

## 2A. Non-Overlapping Patch Flattening (ViT-Style)

1. **Cut image into patches**

   * Patch size Pₕ = Pₗ = 16
   * Number of patches per image:

     $$
       N = \frac{H}{Pₕ} \times \frac{W}{Pₗ}
         = \frac{224}{16} \times \frac{224}{16}
         = 14 \times 14
         = 196.
     $$
2. **Flatten each patch**

   * pᵢ ∈ ℝ^\[C·Pₕ·Pₗ] = ℝ^\[3·16·16] = ℝ^768
   * i = 1…196
3. **Linear projection**

   $$
     eᵢ = Wₑ\,pᵢ + bₑ,
     \quad
     Wₑ ∈ ℝ^{\,D×(C·Pₕ·Pₗ)},\;
     bₑ ∈ ℝ^D
   $$

   * D = 768
   * → each eᵢ ∈ ℝ^768
4. **Stack embeddings** → E\_flat ∈ ℝ^\[B × N × D]

---

## 2B. Overlapping Conv Stem (ConvViT-Style)

1. **“Patch” extraction via Conv2D**

   ```
   E = Conv2D(
         in_channels=3,
         out_channels=D,
         kernel_size=(Pₕ,Pₗ)=16×16,
         stride=S,              # often S=16 for non-overlap, S<16 for overlap
         padding=⌊(2P − S)/2⌋
       )
   ```

   * Example (no overlap): S=16 → padding = ⌊(32−16)/2⌋=8
   * Output spatial size:

     $$
       H_{\text{out}}
       = \Big\lfloor\frac{H + 2p - Pₕ}{S} + 1\Big\rfloor
       = \Big\lfloor\frac{224 + 16 - 16}{16} + 1\Big\rfloor
       = 14
     $$

     likewise W\_out = 14.
   * E ∈ ℝ^\[B × D × 14 × 14]

2. **1×1 Conv projection**

   ```
   E′ = Conv2D(
           in_channels=D,
           out_channels=D,
           kernel_size=1,
           stride=1,
           padding=0
         )
   ```

   * E′ ∈ ℝ^\[B × D × 14 × 14]

3. **Flatten spatial dims**

   * Reshape E′ → E\_conv ∈ ℝ^\[B × N × D], where N=14·14=196

---

## 3. Add Positional Encodings

For both variants we now have a sequence of N embeddings E\_{…×N×D}. We add a learnable (or fixed) positional vector to each slot:

$$
\hat e_i \;=\; e_i \;+\; p^\text{pos}_i,
\quad
p^\text{pos}_i ∈ ℝ^D,\; i=1…196.
$$

Result: an input sequence

$$
\hat E ∈ ℝ^{B × N × D}
$$

ready for the Transformer encoder.

---

## 4. Summary of Dimensions

| Step                     | Shape (example)   |
| ------------------------ | ----------------- |
| Input image I            | \[1, 3, 224, 224] |
| Non-overlap patches pᵢ   | \[196, 768]       |
| Flat‐patch projection eᵢ | \[196, 768]       |
| → E\_flat                | \[1, 196, 768]    |
| Conv stem output E       | \[1, 768, 14, 14] |
| 1×1 conv → E′            | \[1, 768, 14, 14] |
| Flatten → E\_conv        | \[1, 196, 768]    |
| + Positional → ĤE        | \[1, 196, 768]    |

---

## 1. Sanity-check & tiny nits ✅

| Spot | Comment |
|------|---------|
| Patch count (non-overlap) | `N = (H / P_h) × (W / P_w)` assumes `H` and `W` divisible by patch size. In practice ‑> floor division + optional padding. |
| Overlapping padding formula | For stride `S < P`, the “same-size” padding is `p = ⌊(P – S)/2⌋`, exactly what you wrote; just note that odd `(P – S)` yields asymm. padding (PyTorch’s `padding` takes the floor value and loses 1 px on the right/bottom). |
| 1 × 1 conv after conv-stem | Totally optional (some papers merge it with the stem’s `out_channels=D`). |
| Positional encoding symbol clash | You reuse `p_i` both for flattened pixels *and* positional vec. Maybe rename the latter to `\pi_i^{pos}` or similar to avoid confusion in slides. |
| Complexity note | `𝒪(N²)` is for *global* self-attention; Swin/PVT cut it down with local windows or spatial reduction. Might be worth one line. |



---

## 2. PyTorch reference code 🐒

```python
import math
import torch
import torch.nn as nn
from einops import rearrange

# ---------------------------------------------------------------------
# 2A. Flat, non-overlapping patch embedding (ViT style)
# ---------------------------------------------------------------------
class PatchEmbedFlat(nn.Module):
    """
    Parameters
    ----------
    img_size  : int or tuple  – input strides are assumed divisible
    patch_size: int or tuple
    in_chans  : int          – RGB = 3
    embed_dim : int          – token dimension D
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()

        img_size   = (img_size, img_size)   if isinstance(img_size, int)   else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.H, self.W = img_size
        self.P_h, self.P_w = patch_size

        assert self.H % self.P_h == 0 and self.W % self.P_w == 0, \
            'Image size must be divisible by patch size (or pad beforehand).'

        self.num_patches = (self.H // self.P_h) * (self.W // self.P_w)

        # weights = D × (C·P_h·P_w)
        self.proj = nn.Linear(in_chans * self.P_h * self.P_w, embed_dim, bias=True)

    def forward(self, x):
        """
        x : [B, C, H, W]
        Returns sequence: [B, N, D]
        """
        B, C, H, W = x.shape
        # ① reshape: [B, C, H, W] -> [B, C, P_h, N_h, P_w, N_w]
        x = x.view(B, C,
                   H // self.P_h, self.P_h,
                   W // self.P_w, self.P_w)
        # ② move P dims next to C and flatten -> [B, N_h, N_w, C·P_h·P_w]
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(3)
        # ③ merge spatial -> [B, N, C·P_h·P_w]
        x = x.flatten(1, 2)
        # ④ linear proj
        x = self.proj(x)                          # [B, N, D]
        return x


# ---------------------------------------------------------------------
# 2B. Overlapping convolutional stem
# ---------------------------------------------------------------------
class PatchEmbedConv(nn.Module):
    def __init__(self,
                 img_size=224,
                 kernel_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        pad = (kernel_size - stride) // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=pad),
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=1, stride=1, padding=0)  # optional
        )

    def forward(self, x):
        """
        x : [B, C, H, W]
        -> [B, N, D], where
        N = ceil((H-k+1)/s) * ceil((W-k+1)/s)
        """
        x = self.proj(x)          # [B, D, H', W']
        B, D, H_out, W_out = x.shape
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


# ---------------------------------------------------------------------
# 2C. Simple Hierarchical patch embed block (stage inside Swin/PVT)
#     Each stage halves resolution and doubles channels.
# ---------------------------------------------------------------------
class PatchEmbedHierarchical(nn.Module):
    """
    Example: 4-stage hierarchy used inside Swin-Tiny.
    You can treat this as a *stem* + down-sampling between stages.
    """
    def __init__(self, in_chans=3,
                 embed_dims=(96, 192, 384, 768)):
        super().__init__()
        layers = []
        prev_c = in_chans
        for dim in embed_dims:
            layers.append(
                nn.Conv2d(prev_c, dim,
                          kernel_size=3, stride=2, padding=1)  # ↓2
            )
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.GELU())
            prev_c = dim
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns list of feature maps at each level (before flattening),
        which Swin then splits into windows.
        """
        feats = []
        for op in self.proj:
            x = op(x)
            if isinstance(op, nn.GELU):      # after each mini-block
                feats.append(x)
        # feats[i] : [B, C_i, H_i, W_i]
        return feats
```

Quick usage demo:

```python
img = torch.randn(2, 3, 224, 224)

flat_tokens = PatchEmbedFlat()(img)          # [2, 196, 768]
conv_tokens = PatchEmbedConv()(img)          # [2, N≈3136/16 ≈ 3136/16?], depends on stride
pyramid_feats = PatchEmbedHierarchical()(img)
print([f.shape for f in pyramid_feats])
# e.g. [ [2,  96, 112,112],
#        [2, 192, 56, 56],
#        [2, 384, 28, 28],
#        [2, 768, 14, 14] ]
```

• Add positional embeddings right after these calls:  
`tokens = flat_tokens + pos_table[:N]` (learnable or sinusoidal).

---

## 3. Hyper-parameter quick-look 📋

| Variant | Typical values in papers |
|---------|--------------------------|
| ViT-B   | P = 16, D=768, Layers =12, Heads =12, MLP-dim =3072 |
| ViT-L   | P = 16, D=1024, Layers =24 |
| MAE     | Same patchify (16×16) but mask = 75 % |
| DeiT-T  | P = 16, D=192, Layers =12 (data-efficient) |
| Swin-T  | Conv 3×3 stride 2 → windows 7×7, dim = 96/192/384/768 |
| PVTv2-B0| Overlap stem 7×7 s=4, then SR-ratio = 8/4/2/1 per stage |

---

### Where to go next?

• Replace fixed `patch_size` with *dynamic* Gumbel sizes (your § 5) – drop-in controller can wrap the class above.  
• Play with frequency-domain: feed DCT-coeff maps into `PatchEmbedFlat`.  
• Benchmark inference cost: `torch.profiler` on A100 for different strides.  

Let me know if you need TensorFlow/JAX variants, integration into HuggingFace’s `transformers`, or further deep-dive!