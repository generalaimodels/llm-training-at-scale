---

# Chapter 8 Transformer-Based Language Models

---

## 8.1 GPT-n Series

### Definition

The GPT-n series refers to a family of large-scale, autoregressive language models based on the Transformer decoder architecture, where $n$ denotes the version (e.g., GPT-2, GPT-3, GPT-4). These models are trained to predict the next token in a sequence, enabling generative capabilities for text.

### Mathematical Formulation

- **Autoregressive Objective**:  
  Given a sequence of tokens $x = (x_1, x_2, ..., x_T)$, the model maximizes the likelihood:
  $$
  P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)
  $$
  where $\theta$ are the model parameters.

- **Transformer Decoder Block**:  
  Each block consists of masked multi-head self-attention and position-wise feed-forward layers:
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
  $$
  where $M$ is a mask to prevent attending to future tokens.

- **Layer Output**:  
  $$
  h^{(l+1)} = \text{LayerNorm}\left(h^{(l)} + \text{Dropout}(\text{FFN}(\text{LayerNorm}(h^{(l)} + \text{Dropout}(\text{Attention}(h^{(l)})))))\right)
  $$

### Step-by-Step Explanation

#### 1. Data Collection and Preprocessing

- **Corpus**: Large-scale, diverse text datasets (e.g., Common Crawl, Wikipedia).
- **Tokenization**: Byte-Pair Encoding (BPE) or similar subword tokenization.
- **Preprocessing**: Lowercasing, normalization, removal of non-textual artifacts.

#### 2. Model Architecture

- **Input Embedding**:  
  $$
  e_t = W_e x_t + p_t
  $$
  where $W_e$ is the embedding matrix, $x_t$ is the token index, $p_t$ is the positional encoding.

- **Stacked Decoder Blocks**:  
  $N$ layers of masked self-attention and feed-forward networks.

- **Output Layer**:  
  $$
  P(x_{t+1} \mid x_{\leq t}) = \text{softmax}(W_o h_t)
  $$
  where $W_o$ is the output projection matrix.

#### 3. Training

- **Objective**: Minimize negative log-likelihood:
  $$
  \mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
  $$
- **Optimization**: Adam optimizer with learning rate scheduling and gradient clipping.
- **Parallelization**: Data, model, and pipeline parallelism for large-scale training.

#### 4. Inference

- **Autoregressive Generation**:  
  Sequentially sample or decode tokens using greedy, beam search, or sampling strategies.

#### 5. Evaluation

- **Metrics**: Perplexity, BLEU, ROUGE, human evaluation for coherence and relevance.

---

## 8.2 Sparse Transformer

### Definition

Sparse Transformer is a variant of the standard Transformer that introduces sparsity in the self-attention mechanism, reducing computational and memory complexity from $O(L^2)$ to $O(L \sqrt{L})$ or better, where $L$ is the sequence length.

### Mathematical Formulation

- **Sparse Attention Pattern**:  
  For each query position $i$, attention is computed only over a subset $S(i)$ of key positions:
  $$
  \text{Attention}(Q, K, V)_{i} = \sum_{j \in S(i)} \alpha_{ij} V_j
  $$
  where
  $$
  \alpha_{ij} = \frac{\exp\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)}{\sum_{k \in S(i)} \exp\left(\frac{Q_i K_k^T}{\sqrt{d_k}}\right)}
  $$

### Step-by-Step Explanation

#### 1. Motivation

- **Quadratic Complexity**: Standard self-attention scales as $O(L^2)$.
- **Sparsity**: Only attend to a subset of positions, e.g., local windows, strided patterns, or block patterns.

#### 2. Sparse Attention Patterns

- **Local Attention**: Attend to a fixed window around each position.
- **Strided Attention**: Attend to every $k$-th position.
- **Block Attention**: Divide sequence into blocks and attend within/between blocks.

#### 3. Implementation

- **Masking**: Construct binary masks $M_{ij}$ indicating allowed attention connections.
- **Efficient Computation**: Use sparse matrix operations to reduce memory and compute.

#### 4. Training and Inference

- **Same as Standard Transformer**: Loss, optimization, and generation are unchanged.
- **Scalability**: Enables training on much longer sequences.

#### 5. Evaluation

- **Metrics**: Compare perplexity and efficiency (FLOPs, memory) to dense models.

---

## 8.3 Reformer

### Definition

Reformer is a Transformer variant designed for efficient memory and computation, using locality-sensitive hashing (LSH) for sparse attention and reversible residual layers to reduce memory usage.

### Mathematical Formulation

- **LSH Attention**:  
  Partition queries and keys into buckets using LSH, attend only within buckets:
  $$
  \text{LSHAttention}(Q, K, V) = \bigcup_{b=1}^{B} \text{Attention}(Q_b, K_b, V_b)
  $$
  where $Q_b, K_b, V_b$ are the queries, keys, and values in bucket $b$.

- **Reversible Residuals**:  
  For input $(x, y)$, the reversible block computes:
  $$
  y' = y + \mathcal{F}(x) \\
  x' = x + \mathcal{G}(y')
  $$
  allowing reconstruction of $(x, y)$ from $(x', y')$.

### Step-by-Step Explanation

#### 1. LSH Attention

- **Hashing**:  
  Use random projections to hash queries and keys into buckets.
- **Within-Bucket Attention**:  
  Compute standard attention only among tokens in the same bucket.
- **Complexity**:  
  Reduces attention complexity from $O(L^2)$ to $O(L \log L)$.

#### 2. Reversible Layers

- **Memory Efficiency**:  
  Store only activations at the input and output of each block; intermediate activations are recomputed during backpropagation.
- **Forward/Backward Pass**:  
  Enables training of very deep models with constant memory per layer.

#### 3. Chunked Feed-Forward

- **Chunking**:  
  Process feed-forward layers in chunks to further reduce memory footprint.

#### 4. Training and Inference

- **Objective**: Same as standard Transformer.
- **Optimization**: Adam or similar optimizers.
- **Scalability**: Enables training on sequences with length $L \gg 1024$.

#### 5. Evaluation

- **Metrics**: Perplexity, memory usage, training speed, and scalability.

---

# Summary Table

| Model         | Key Innovation                | Attention Complexity | Memory Optimization      | Sequence Length |
|---------------|------------------------------|---------------------|-------------------------|----------------|
| GPT-n         | Large-scale, dense decoder   | $O(L^2)$            | Model/data parallelism  | $<4096$        |
| Sparse Trans. | Sparse attention patterns    | $O(L \sqrt{L})$     | Sparse ops              | $>4096$        |
| Reformer      | LSH, reversible layers       | $O(L \log L)$       | Reversible computation  | $>64k$         |

---