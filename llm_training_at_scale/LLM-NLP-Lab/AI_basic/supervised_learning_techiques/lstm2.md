# Long Short-Term Memory (LSTM): Complete Technical Architecture

## 1. **Definition**

Long Short-Term Memory (LSTM) is a specialized recurrent neural network architecture that addresses the vanishing gradient problem in traditional RNNs through gated memory mechanisms. LSTMs maintain both short-term hidden states and long-term cell states, enabling effective modeling of sequential dependencies across extended temporal horizons.

## 2. **Mathematical Formulation**

### 2.1 **Core LSTM Equations**

For time step $t$ with input $x_t \in \mathbb{R}^{d_{in}}$, previous hidden state $h_{t-1} \in \mathbb{R}^{d_h}$, and previous cell state $c_{t-1} \in \mathbb{R}^{d_h}$:

#### **Forget Gate**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

#### **Input Gate**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

#### **Candidate Values**
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

#### **Cell State Update**
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

#### **Output Gate**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

#### **Hidden State Update**
$$h_t = o_t * \tanh(C_t)$$

### 2.2 **Vectorized Implementation Form**

$$\begin{bmatrix} f_t \\ i_t \\ o_t \\ \tilde{C}_t \end{bmatrix} = \begin{bmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{bmatrix} \left( W \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b \right)$$

Where $W \in \mathbb{R}^{4d_h \times (d_h + d_{in})}$ and $b \in \mathbb{R}^{4d_h}$.

### 2.3 **Gradient Flow Equations**

#### **Cell State Gradient**
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

#### **Hidden State Gradient**
$$\frac{\partial h_t}{\partial h_{t-1}} = o_t \cdot (1 - \tanh^2(C_t)) \cdot \frac{\partial C_t}{\partial h_{t-1}} + \frac{\partial o_t}{\partial h_{t-1}} \cdot \tanh(C_t)$$

## 3. **Key Principles**

### 3.1 **Gating Mechanism**
- **Forget Gate**: Controls information removal from cell state
- **Input Gate**: Regulates new information incorporation
- **Output Gate**: Determines hidden state exposure

### 3.2 **Memory Architecture**
- **Cell State**: Long-term memory pathway with minimal transformations
- **Hidden State**: Short-term working memory for immediate computations

### 3.3 **Gradient Preservation**
- Constant error carousel through cell state
- Multiplicative gates prevent gradient vanishing/exploding

## 4. **Detailed Architecture Analysis**

### 4.1 **Parameter Calculation**

For LSTM with input dimension $d_{in}$, hidden dimension $d_h$, and $L$ layers:

#### **Single Layer Parameters**
- Weight matrices: $W \in \mathbb{R}^{4d_h \times (d_h + d_{in})}$
- Bias vectors: $b \in \mathbb{R}^{4d_h}$
- **Total per layer**: $4d_h(d_h + d_{in}) + 4d_h = 4d_h(d_h + d_{in} + 1)$

#### **Multi-layer Parameters**
$$\text{Total} = 4d_h(d_{in} + d_h + 1) + (L-1) \cdot 4d_h(2d_h + 1)$$

### 4.2 **Computational Complexity**

#### **Time Complexity**
- Forward pass: $O(T \cdot d_h^2)$ where $T$ is sequence length
- Backward pass: $O(T \cdot d_h^2)$

#### **Space Complexity**
- Memory: $O(T \cdot d_h)$ for storing hidden states during BPTT

### 4.3 **Initialization Strategies**

#### **Xavier/Glorot Initialization**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

#### **Forget Gate Bias Initialization**
$$b_f = \mathbf{1} \text{ (bias toward remembering)}$$

## 5. **Pre-processing Pipeline**

### 5.1 **Sequence Preprocessing**

#### **Padding**
$$X_{padded} = \text{pad}(X, \text{max\_len}, \text{pad\_value}=0)$$

#### **Normalization**
$$x_{norm} = \frac{x - \mu}{\sigma + \epsilon}$$

#### **Tokenization (NLP)**
$$\text{tokens} = \text{tokenizer}(\text{text}) \rightarrow \mathbb{Z}^+$$

### 5.2 **Embedding Layer**
$$E = \text{Embedding}(\text{vocab\_size}, d_{emb})$$
$$x_t = E[\text{token}_t]$$

## 6. **Training Algorithm**

### 6.1 **Forward Pass Pseudocode**

```python
def lstm_forward(x, h_0, c_0, weights):
    """
    x: (batch_size, seq_len, input_size)
    h_0, c_0: (batch_size, hidden_size)
    """
    batch_size, seq_len, _ = x.shape
    h_t, c_t = h_0, c_0
    outputs = []
    
    for t in range(seq_len):
        # Concatenate h_{t-1} and x_t
        combined = torch.cat([h_t, x[:, t, :]], dim=1)
        
        # Compute gates
        gates = torch.matmul(combined, weights.W) + weights.b
        f_t = torch.sigmoid(gates[:, :hidden_size])
        i_t = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
        o_t = torch.sigmoid(gates[:, 2*hidden_size:3*hidden_size])
        g_t = torch.tanh(gates[:, 3*hidden_size:])
        
        # Update cell and hidden states
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        outputs.append(h_t)
    
    return torch.stack(outputs, dim=1), (h_t, c_t)
```

### 6.2 **Backpropagation Through Time (BPTT)**

```python
def bptt_lstm(outputs, targets, seq_len):
    """
    Truncated BPTT implementation
    """
    total_loss = 0
    
    for t in range(seq_len-1, -1, -1):
        # Compute loss at time t
        loss_t = criterion(outputs[t], targets[t])
        total_loss += loss_t
        
        # Compute gradients
        if t == seq_len - 1:
            dh_next = torch.autograd.grad(loss_t, outputs[t])[0]
            dc_next = torch.zeros_like(dh_next)
        
        # Backpropagate through LSTM cell
        dh_t, dc_t, dW, db = lstm_backward_step(
            dh_next, dc_next, cache[t]
        )
        
        # Accumulate gradients
        gradients.W += dW
        gradients.b += db
        
        dh_next, dc_next = dh_t, dc_t
    
    return total_loss, gradients
```

### 6.3 **Training Loop**

```python
def train_lstm(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Initialize hidden states
            h_0 = torch.zeros(batch_size, hidden_size)
            c_0 = torch.zeros(batch_size, hidden_size)
            
            # Forward pass
            outputs, (h_n, c_n) = model(data, (h_0, c_0))
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader)}')
```

## 7. **Loss Functions**

### 7.1 **Sequence-to-Sequence Loss**
$$\mathcal{L}_{seq2seq} = \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}(y_t, \hat{y}_t)$$

### 7.2 **Cross-Entropy Loss**
$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### 7.3 **Mean Squared Error**
$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 7.4 **Connectionist Temporal Classification (CTC)**
$$\mathcal{L}_{CTC} = -\log P(y|x) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t|x)$$

## 8. **Evaluation Metrics**

### 8.1 **Classification Metrics**

#### **Accuracy**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### **F1-Score**
$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### **Matthews Correlation Coefficient**
$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 8.2 **Regression Metrics**

#### **Root Mean Square Error**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

#### **Mean Absolute Percentage Error**
$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

### 8.3 **Sequence-Specific Metrics**

#### **BLEU Score**
$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

#### **Word Error Rate (WER)**
$$WER = \frac{S + D + I}{N}$$

Where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total words.

#### **Perplexity**
$$PPL = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i|w_{<i})\right)$$

## 9. **Post-Training Procedures**

### 9.1 **Model Quantization**

#### **Post-Training Quantization**
$$W_{int8} = \text{round}\left(\frac{W_{fp32}}{s}\right)$$

Where $s = \frac{\max(|W|)}{127}$ is the scaling factor.

#### **Quantization-Aware Training**
$$W_{fake\_quant} = s \cdot \text{round}\left(\frac{W}{s}\right)$$

### 9.2 **Knowledge Distillation**

#### **Distillation Loss**
$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \mathcal{L}_{KL}(\sigma(z_t/T), \sigma(z_s/T))$$

Where $z_s$, $z_t$ are student and teacher logits, $T$ is temperature.

### 9.3 **Pruning**

#### **Magnitude-Based Pruning**
$$\text{mask}_i = \begin{cases} 
1 & \text{if } |W_i| > \theta \\
0 & \text{otherwise}
\end{cases}$$

#### **Structured Pruning**
$$\text{Importance}(W_i) = \frac{1}{n} \sum_{j=1}^{n} \left|\frac{\partial \mathcal{L}}{\partial W_{i,j}}\right|$$

## 10. **Industrial Implementation**

### 10.1 **PyTorch Implementation**

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = torch.zeros(self.num_layers * num_directions, 
                            batch_size, self.hidden_size).to(x.device)
            c_0 = torch.zeros(self.num_layers * num_directions, 
                            batch_size, self.hidden_size).to(x.device)
            hidden = (h_0, c_0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Get output from last time step
        output = self.fc(lstm_out[:, -1, :])
        
        return output, hidden
```

