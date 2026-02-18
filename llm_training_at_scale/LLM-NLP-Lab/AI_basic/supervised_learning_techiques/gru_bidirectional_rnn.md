### I. Gated Recurrent Units (GRU)

#### A. Definition
A Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture that utilizes gating mechanisms to manage and control the flow of information between cells in the neural network. GRUs are designed to overcome the vanishing gradient problem, which can impair the learning capability of traditional RNNs over long sequences. They achieve this by selectively updating the hidden state, effectively deciding what information to retain from past states and what new information to incorporate.

#### B. Pertinent Equations
Let $x_t \in \mathbb{R}^{d_x}$ be the input vector at time step $t$, and $h_{t-1} \in \mathbb{R}^{d_h}$ be the hidden state from the previous time step $t-1$. $d_x$ is the input feature dimension and $d_h$ is the hidden state dimension. The GRU computations are as follows:

1.  **Reset Gate ($r_t$)**: Determines how much of the previous hidden state to forget.
    $$ r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) $$
    where $W_r \in \mathbb{R}^{d_h \times d_x}$, $U_r \in \mathbb{R}^{d_h \times d_h}$ are weight matrices, $b_r \in \mathbb{R}^{d_h}$ is a bias vector, and $\sigma$ is the sigmoid activation function, $\sigma(z) = \frac{1}{1 + e^{-z}}$.

2.  **Update Gate ($z_t$)**: Determines how much of the previous hidden state to keep and how much of the new candidate hidden state to incorporate.
    $$ z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) $$
    where $W_z \in \mathbb{R}^{d_h \times d_x}$, $U_z \in \mathbb{R}^{d_h \times d_h}$ are weight matrices, and $b_z \in \mathbb{R}^{d_h}$ is a bias vector.

3.  **Candidate Hidden State ($\tilde{h}_t$)**: Represents new information or content to be potentially added to the hidden state. It is calculated using the reset gate's output to selectively incorporate information from the previous hidden state.
    $$ \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) $$
    where $W_h \in \mathbb{R}^{d_h \times d_x}$, $U_h \in \mathbb{R}^{d_h \times d_h}$ are weight matrices, $b_h \in \mathbb{R}^{d_h}$ is a bias vector, $\tanh$ is the hyperbolic tangent activation function, and $\odot$ denotes element-wise multiplication.

4.  **Hidden State ($h_t$)**: The final hidden state for time step $t$. It is a linear interpolation between the previous hidden state $h_{t-1}$ and the candidate hidden state $\tilde{h}_t$, controlled by the update gate $z_t$.
    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$
    This equation shows how the GRU updates its hidden state: $z_t$ decides how much of $h_{t-1}$ to "forget" and how much of $\tilde{h}_t$ to "add".

5.  **Vectorized Form (Industrial Standard Implementation like PyTorch/TensorFlow):**
    Frameworks often implement these operations using combined matrix multiplications for efficiency. For instance, the gate computations can be grouped:
    $$ [r_t, z_t]^T = \sigma(W_{rz} x_t + U_{rz} h_{t-1} + b_{rz}) $$
    $$ \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) $$
    where $W_{rz} \in \mathbb{R}^{2d_h \times d_x}$, $U_{rz} \in \mathbb{R}^{2d_h \times d_h}$, $b_{rz} \in \mathbb{R}^{2d_h}$.
    Alternatively, some implementations compute matrix products involving $x_t$ first:
    $$ G_x = W_r x_t + b_{rx}, \quad N_x = W_h x_t + b_{hx} $$
    $$ G_h = U_r h_{t-1} + b_{rh}, \quad N_h = U_h (r_t \odot h_{t-1}) + b_{hh} $$
    Then:
    $$ r_t = \sigma(W_{xr} x_t + b_{xr} + W_{hr} h_{t-1} + b_{hr}) $$
    $$ z_t = \sigma(W_{xz} x_t + b_{xz} + W_{hz} h_{t-1} + b_{hz}) $$
    $$ \tilde{h}_t = \tanh(W_{xh} x_t + b_{xh} + r_t \odot (W_{hh} h_{t-1} + b_{hh})) $$
    Here, $W_{xr}, W_{xz}, W_{xh}$ are input-to-hidden weights, and $W_{hr}, W_{hz}, W_{hh}$ are hidden-to-hidden weights. $b$ terms are corresponding biases.

#### C. Key Principles
*   **Gating Mechanisms**: The core of GRUs lies in the reset and update gates. These gates, implemented as sigmoid-activated neural networks, dynamically control information flow.
*   **Adaptive Forgetting and Memory Update**: The reset gate allows the model to discard information from the past hidden state that is irrelevant for future predictions. The update gate controls how much of the past information is carried forward and how much of the newly computed information (candidate state) is incorporated.
*   **Mitigation of Vanishing Gradients**: By allowing information to bypass multiple time steps largely unchanged (when $z_t$ is close to 0, $h_t \approx h_{t-1}$), GRUs maintain long-range dependencies more effectively than simple RNNs.

#### D. Detailed Concept Analysis
1.  **Role of Gates**:
    *   **Reset Gate ($r_t$)**: When $r_t$ values are close to 0, the GRU effectively "forgets" the previous hidden state $h_{t-1}$ when computing the candidate hidden state $\tilde{h}_t$. This allows the model to drop information that is found to be irrelevant for the current context. If $r_t$ is close to 1, most of the previous hidden state is used.
    *   **Update Gate ($z_t$)**: This gate acts as a controller for information retention versus new information incorporation. When $z_t$ values are close to 1, the new candidate state $\tilde{h}_t$ is predominantly used to form $h_t$, effectively updating the memory with new information. When $z_t$ values are close to 0, the previous hidden state $h_{t-1}$ is mostly preserved, allowing the model to carry information over many time steps.

2.  **Information Flow**:
    *   Input $x_t$ and previous hidden state $h_{t-1}$ are fed into both the reset and update gates.
    *   The reset gate $r_t$ modulates the contribution of $h_{t-1}$ to the candidate state $\tilde{h}_t$.
    *   The candidate state $\tilde{h}_t$ is computed using $x_t$ and the modulated $h_{t-1}$.
    *   The update gate $z_t$ then determines the convex combination of $h_{t-1}$ and $\tilde{h}_t$ to produce the current hidden state $h_t$. This structure ensures that if the update gate is mostly "open" for the old state (i.e., $z_t$ is small), gradients related to $h_{t-1}$ can flow back relatively unimpeded.

3.  **Comparison with LSTM**:
    *   GRUs have a simpler architecture than LSTMs (Long Short-Term Memory units). LSTMs use three gates (input, forget, output) and a separate cell state $C_t$ in addition to the hidden state $h_t$.
    *   GRUs combine the functionality of LSTM's input and forget gates into a single update gate $z_t$. They also merge the cell state and hidden state.
    *   This simplification means GRUs have fewer parameters than LSTMs for the same hidden dimension size, potentially leading to faster training and less overfitting on smaller datasets.

#### E. Importance
*   **Effective for Sequential Data**: GRUs excel at modeling sequential data where context and order are crucial, such as natural language text, time series, and speech.
*   **Computational Efficiency**: Compared to LSTMs, GRUs are often computationally less expensive due to their simpler structure and fewer parameters, making them attractive for resource-constrained applications or when faster training is desired.
*   **Strong Performance**: GRUs have demonstrated performance comparable to, and sometimes exceeding, LSTMs on various tasks, particularly when dataset size is limited or sequence lengths are moderate.

#### F. Pros versus Cons
*   **Pros**:
    *   **Fewer Parameters**: Requires fewer trainable parameters than LSTM for the same hidden layer size, leading to faster training times and reduced memory footprint.
    *   **Reduced Overfitting**: The simpler architecture can be less prone to overfitting, especially on smaller datasets.
    *   **Good Performance**: Achieves strong performance on many sequence modeling tasks, often comparable to LSTMs.
    *   **Effective Gradient Flow**: The gating mechanism helps mitigate vanishing/exploding gradients, allowing learning of longer-range dependencies compared to simple RNNs.
*   **Cons**:
    *   **Potentially Less Expressive Power**: On tasks requiring extremely long-range dependencies or very complex temporal patterns, LSTMs might offer superior performance due to their additional gate and separate cell state, which can provide finer-grained control over memory.
    *   **Empirical Choice**: The choice between GRU and LSTM is often empirical; one may outperform the other depending on the specific dataset and task. There is no universal superiority.

#### G. Cutting-edge Advances
*   **Variations**: Architectural variations like Minimal Gated Unit (MGU) have been proposed to further simplify GRUs while retaining performance.
*   **Attention Mechanisms**: GRUs are frequently combined with attention mechanisms, enabling models to focus on specific parts of the input sequence when producing an output, significantly improving performance in tasks like machine translation and question answering.
*   **Hybrid Models**: Integration of GRUs with other architectures, such as Convolutional Neural Networks (CNNs) for feature extraction from sequences (e.g., CNN-GRU for text classification) or Graph Neural Networks (GNNs) for spatio-temporal data.
*   **Applications in Large Models**: While transformers have become dominant in NLP, GRU/LSTM components are still relevant in specific contexts or as part of larger, hybrid architectures, particularly where sequential inductive bias is beneficial or computational resources are a concern for full transformer models.
*   **Lightweight GRUs**: Research into further reducing parameters and computational cost for on-device AI and edge computing.

### II. Bidirectional LSTM (BiLSTM)

#### A. Definition
A Bidirectional Long Short-Term Memory (BiLSTM) network is an extension of the standard LSTM that processes sequence data in both forward (past to future) and backward (future to past) directions. It consists of two separate LSTM layers: one processes the input sequence as is, and the other processes a reversed copy of the input sequence. The outputs from these two LSTMs at each time step are then combined to produce a single output representation that incorporates information from both past and future contexts.

#### B. Pertinent Equations

1.  **LSTM Cell (Prerequisite)**:
    An LSTM cell at time step $t$ takes an input $x_t \in \mathbb{R}^{d_x}$, previous hidden state $h_{t-1} \in \mathbb{R}^{d_h}$, and previous cell state $C_{t-1} \in \mathbb{R}^{d_h}$.
    *   **Forget Gate ($f_t$)**:
        $$ f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$
    *   **Input Gate ($i_t$)**:
        $$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$
    *   **Candidate Cell State ($\tilde{C}_t$)**:
        $$ \tilde{C}_t = \tanh(W_C x_t + U_C h_{t-1} + b_C) $$
    *   **Cell State ($C_t$)**:
        $$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
    *   **Output Gate ($o_t$)**:
        $$ o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$
    *   **Hidden State ($h_t$)**:
        $$ h_t = o_t \odot \tanh(C_t) $$
    All $W \in \mathbb{R}^{d_h \times d_x}$, $U \in \mathbb{R}^{d_h \times d_h}$ are weight matrices, $b \in \mathbb{R}^{d_h}$ are bias vectors. $\sigma$ is sigmoid, $\tanh$ is hyperbolic tangent, $\odot$ is element-wise multiplication.

2.  **Forward LSTM Pass**:
    The forward LSTM processes the input sequence $X = (x_1, x_2, ..., x_T)$ from $t=1$ to $T$.
    $$ \overrightarrow{f_t} = \sigma(\overrightarrow{W_f} x_t + \overrightarrow{U_f} \overrightarrow{h_{t-1}} + \overrightarrow{b_f}) $$
    $$ \overrightarrow{i_t} = \sigma(\overrightarrow{W_i} x_t + \overrightarrow{U_i} \overrightarrow{h_{t-1}} + \overrightarrow{b_i}) $$
    $$ \overrightarrow{\tilde{C}_t} = \tanh(\overrightarrow{W_C} x_t + \overrightarrow{U_C} \overrightarrow{h_{t-1}} + \overrightarrow{b_C}) $$
    $$ \overrightarrow{C_t} = \overrightarrow{f_t} \odot \overrightarrow{C_{t-1}} + \overrightarrow{i_t} \odot \overrightarrow{\tilde{C}_t} $$
    $$ \overrightarrow{o_t} = \sigma(\overrightarrow{W_o} x_t + \overrightarrow{U_o} \overrightarrow{h_{t-1}} + \overrightarrow{b_o}) $$
    $$ \overrightarrow{h_t} = \overrightarrow{o_t} \odot \tanh(\overrightarrow{C_t}) $$
    The hidden states are $\overrightarrow{H} = (\overrightarrow{h_1}, \overrightarrow{h_2}, ..., \overrightarrow{h_T})$.

3.  **Backward LSTM Pass**:
    The backward LSTM processes the input sequence in reverse order, from $t=T$ to $1$. This can be viewed as processing $X' = (x_T, x_{T-1}, ..., x_1)$.
    $$ \overleftarrow{f_t} = \sigma(\overleftarrow{W_f} x_t + \overleftarrow{U_f} \overleftarrow{h_{t+1}} + \overleftarrow{b_f}) $$
    $$ \overleftarrow{i_t} = \sigma(\overleftarrow{W_i} x_t + \overleftarrow{U_i} \overleftarrow{h_{t+1}} + \overleftarrow{b_i}) $$
    $$ \overleftarrow{\tilde{C}_t} = \tanh(\overleftarrow{W_C} x_t + \overleftarrow{U_C} \overleftarrow{h_{t+1}} + \overleftarrow{b_C}) $$
    $$ \overleftarrow{C_t} = \overleftarrow{f_t} \odot \overleftarrow{C_{t+1}} + \overleftarrow{i_t} \odot \overleftarrow{\tilde{C}_t} $$
    $$ \overleftarrow{o_t} = \sigma(\overleftarrow{W_o} x_t + \overleftarrow{U_o} \overleftarrow{h_{t+1}} + \overleftarrow{b_o}) $$
    $$ \overleftarrow{h_t} = \overleftarrow{o_t} \odot \tanh(\overleftarrow{C_t}) $$
    The hidden states are $\overleftarrow{H} = (\overleftarrow{h_1}, \overleftarrow{h_2}, ..., \overleftarrow{h_T})$. Note that $\overleftarrow{h_{T+1}}$ is initialized (e.g., to zeros). The parameters ($\overrightarrow{W}, \overrightarrow{U}, \overrightarrow{b}$) and ($\overleftarrow{W}, \overleftarrow{U}, \overleftarrow{b}$) are distinct sets of weights.

4.  **Output Combination**:
    The final hidden state $y_t$ at each time step $t$ is typically a combination of the forward hidden state $\overrightarrow{h_t}$ and the backward hidden state $\overleftarrow{h_t}$. Common combination methods include:
    *   **Concatenation (most common)**:
        $$ y_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}] $$
        The resulting $y_t \in \mathbb{R}^{2d_h}$.
    *   **Summation**:
        $$ y_t = \overrightarrow{h_t} + \overleftarrow{h_t} $$
        Requires $\overrightarrow{h_t}$ and $\overleftarrow{h_t}$ to have the same dimension.
    *   **Average**:
        $$ y_t = \frac{1}{2} (\overrightarrow{h_t} + \overleftarrow{h_t}) $$
    *   **Multiplication**:
        $$ y_t = \overrightarrow{h_t} \odot \overleftarrow{h_t} $$

#### C. Key Principles
*   **Bidirectional Context**: BiLSTMs process sequences in both temporal directions, allowing the hidden state at any time step $t$ to encapsulate information from both past ($x_1, ..., x_{t-1}$) and future ($x_{t+1}, ..., x_T$) inputs.
*   **Independent LSTMs**: The forward and backward LSTMs operate independently with separate sets of parameters. This allows each LSTM to learn different aspects of the sequential dependencies.
*   **Information Fusion**: The outputs of the two LSTMs are combined (e.g., concatenated) to provide a richer representation of the input at each time step.

#### D. Detailed Concept Analysis
1.  **Forward Propagation**:
    The forward LSTM ($\text{LSTM}_f$) processes the input sequence $x_1, x_2, \ldots, x_T$ from left to right. At each time step $t$, $\overrightarrow{h_t}$ captures information from $x_1, \ldots, x_t$.

2.  **Backward Propagation (of sequence processing, not to be confused with BPTT for training)**:
    The backward LSTM ($\text{LSTM}_b$) processes the input sequence $x_T, x_{T-1}, \ldots, x_1$ from right to left. At each time step $t$, $\overleftarrow{h_t}$ captures information from $x_T, \ldots, x_t$. When implementing, this often means feeding the reversed sequence to a standard LSTM and then reversing its hidden state outputs to align them with the original sequence order.

3.  **Contextual Representation**:
    By combining $\overrightarrow{h_t}$ and $\overleftarrow{h_t}$, the BiLSTM generates a representation $y_t$ that is informed by the entire input sequence. For example, in Named Entity Recognition (NER), knowing the words that follow a particular word can be crucial for determining its entity type. A word like "Washington" can be a person or a location, and the subsequent words (e.g., "D.C." or "spoke") help disambiguate.

#### E. Importance
*   **Enhanced Contextual Understanding**: BiLSTMs provide a more comprehensive understanding of the context in sequential data by considering information from both directions. This is critical for tasks where the meaning of an element depends on its surrounding elements.
*   **Improved Performance**: They often achieve superior performance compared to unidirectional LSTMs on various NLP tasks, including NER, Part-of-Speech (POS) tagging, sentiment analysis, and machine translation.
*   **Foundation for Advanced Models**: BiLSTMs have served as fundamental building blocks in more complex architectures, such as attention-based models and pre-trained contextual embeddings like ELMo (Embeddings from Language Models).

#### F. Pros versus Cons
*   **Pros**:
    *   **Access to Future Context**: The primary advantage is the ability to incorporate information from future time steps, leading to richer and more accurate representations.
    *   **Improved Accuracy**: Generally results in better performance on tasks where context from both directions is beneficial.
    *   **Captures Complex Dependencies**: Effective at modeling intricate patterns and long-range dependencies within sequences.
*   **Cons**:
    *   **Increased Computational Cost**: Training and inference are more computationally intensive (roughly double) compared to unidirectional LSTMs due to the two separate LSTM passes and more parameters.
    *   **Not Suitable for Online/Real-time Processing**: Requires the entire input sequence to be available before processing can be completed for any time step. This makes them unsuitable for applications where predictions must be made as data arrives sequentially (e.g., real-time speech recognition for live streaming).
    *   **Higher Latency**: The need to process the sequence in both directions can introduce latency.

#### G. Cutting-edge Advances
*   **Stacked BiLSTMs**: Using multiple layers of BiLSTMs, where the output of one BiLSTM layer serves as the input to the next, can learn more abstract and hierarchical features.
*   **BiLSTMs with Attention**: Combining BiLSTMs with attention mechanisms allows the model to weigh the importance of different parts of the sequence (both past and future contexts) when making predictions, further enhancing performance. The `BiLSTM-CRF` architecture is a common example for sequence labeling tasks.
*   **Pre-trained Models**: BiLSTMs were instrumental in early contextual embedding models like ELMo. While transformer models have largely superseded them for general-purpose pre-training, BiLSTMs remain relevant in specific applications or hybrid architectures.
*   **Efficient Variants**: Research into making BiLSTMs more efficient, for example, through parameter sharing schemes or approximations, to reduce computational overhead.
*   **Integration with Transformers**: Some recent architectures explore hybrid approaches combining the local sequential modeling strengths of BiLSTMs with the global context capturing abilities of Transformers.

### III. Common Components and Procedures for GRU & BiLSTM

#### A. Data Pre-processing

1.  **Tokenization**:
    *   **Definition**: The process of segmenting raw text into smaller units called tokens (e.g., words, subwords, characters).
    *   **Methods**:
        *   Word-level: Splitting by spaces or punctuation.
        *   Character-level: Treating each character as a token.
        *   Subword-level: Using algorithms like Byte Pair Encoding (BPE), WordPiece, or SentencePiece to break words into smaller, meaningful units. This handles out-of-vocabulary (OOV) words and rare words better.
    *   **Mathematical Representation**: Input text $S$ is transformed into a sequence of tokens $T = (t_1, t_2, ..., t_L)$.

2.  **Numericalization & Embedding**:
    *   **a. Integer Encoding**:
        *   **Definition**: Mapping each unique token to a unique integer ID. A vocabulary (mapping from token to ID) is built.
        *   **Equation**: $t_i \rightarrow id_i$, where $id_i \in \{0, 1, ..., |V|-1\}$, and $|V|$ is the vocabulary size.
    *   **b. Embedding Layer**:
        *   **Definition**: Converting integer-encoded tokens into dense vector representations (embeddings). These embeddings capture semantic relationships between tokens.
        *   **Equation**: For a token with ID $id_i$, its embedding $e_i$ is obtained by looking up the $id_i$-th row in an embedding matrix $W_{emb} \in \mathbb{R}^{|V| \times d_{emb}}$.
            $$ e_i = W_{emb}[id_i, :] $$
            Or, if $x_{i,one\_hot}$ is a one-hot vector for token $i$:
            $$ e_i = W_{emb}^T x_{i,one\_hot} $$
            where $d_{emb}$ is the embedding dimension. $W_{emb}$ is typically learned during training.

3.  **Padding & Truncation**:
    *   **Definition**: Ensuring all sequences in a batch have the same length. RNNs can technically handle variable-length sequences, but batching for efficient GPU computation requires fixed-length inputs.
    *   **Padding**: Shorter sequences are padded with a special padding token (e.g., with ID 0) until they reach the maximum length $L_{max}$ in the batch or a predefined maximum length.
    *   **Truncation**: Longer sequences are truncated to $L_{max}$.
    *   **Masking**: A corresponding mask sequence is often created to indicate which elements are real data and which are padding, so that padding elements are ignored by subsequent layers or loss calculation. $m_t=1$ if $x_t$ is actual data, $m_t=0$ if $x_t$ is padding.

4.  **Batching**:
    *   **Definition**: Grouping multiple pre-processed sequences into a single tensor (batch) for parallel processing during training and inference.
    *   **Representation**: A batch of $B$ sequences, each of length $L_{max}$ and embedding dimension $d_{emb}$, forms a tensor of shape $(B, L_{max}, d_{emb})$.

#### B. Model Architecture (General Recurrent Network Structure)

1.  **Input Layer**:
    *   Consists of the embedding layer that transforms token IDs into dense vectors: $x_t \in \mathbb{R}^{d_{emb}}$. These are the inputs to the recurrent layer.

2.  **Recurrent Layer (GRU/BiLSTM)**:
    *   This layer processes the sequence of embeddings $x_1, x_2, ..., x_T$.
    *   For GRU: Produces a sequence of hidden states $h_1, h_2, ..., h_T$, where $h_t \in \mathbb{R}^{d_h}$.
    *   For BiLSTM: Produces a sequence of combined hidden states $y_1, y_2, ..., y_T$, where $y_t \in \mathbb{R}^{2d_h}$ (if concatenating).

3.  **Output Layer**:
    *   Transforms the recurrent layer's output(s) into the desired format for the task.
    *   **Sequence Classification (e.g., sentiment analysis)**: Typically uses the final hidden state ($h_T$ or $y_T$) or an aggregation (e.g., max-pooling) of all hidden states, followed by one or more fully connected layers and a softmax/sigmoid activation.
        $$ \text{logits} = W_{out} h_T + b_{out} $$
        $$ \hat{p} = \text{softmax}(\text{logits}) \quad \text{(for multi-class)} $$
    *   **Sequence Labeling (e.g., NER, POS tagging)**: Applies a fully connected layer and softmax to each hidden state $h_t$ (or $y_t$) in the sequence.
        $$ \text{logits}_t = W_{out} h_t + b_{out} $$
        $$ \hat{p}_t = \text{softmax}(\text{logits}_t) $$
    *   $W_{out}$ and $b_{out}$ are parameters of the output layer.

#### C. Training

1.  **Initialization**:
    *   Model parameters (weights $W$ and biases $b$ for embedding, recurrent, and output layers) are initialized, often using schemes like Xavier/Glorot initialization or Kaiming/He initialization to promote stable gradient flow.
    *   Hidden states ($h_0$, $C_0$ for LSTM) are typically initialized to zeros.

2.  **Training Loop Pseudo-algorithm**:
    Let $\theta$ be the set of all model parameters. Learning rate is $\eta$.
    ```
    Initialize parameters θ
    For epoch = 1 to num_epochs:
        Shuffle training data D
        For each batch (X_batch, Y_batch) in D:
            // X_batch: input sequences, Y_batch: target labels
            // 1. Pre-processing (if not done already, e.g. on-the-fly embedding)
            //    input_embeddings = EmbeddingLookup(X_batch)

            // 2.a. Forward Pass:
            //    Initialize h_0 (and C_0 for LSTM)
            //    hidden_states = []
            //    For t = 1 to sequence_length:
            //        If GRU:
            //            h_t = GRU_Cell(input_embeddings[:, t, :], h_{t-1}; θ_gru)
            //        If BiLSTM:
            //            Compute forward_h_t using LSTM_f_Cell(input_embeddings[:, t, :], forward_h_{t-1}; θ_lstm_f)
            //            Compute backward_h_t using LSTM_b_Cell(input_embeddings_reversed[:, t, :], backward_h_{t-1}; θ_lstm_b)
            //            (Note: backward pass usually computed on reversed sequence then reordered)
            //    (For BiLSTM, combine forward and backward hidden states for each time step: y_t = Combine(forward_h_t, backward_h_t))
            //    output_sequence = RecurrentLayer(input_embeddings; θ_recurrent) // e.g., h_1, ..., h_T or y_1, ..., y_T

            //    predictions = OutputLayer(output_sequence; θ_output) // E.g., softmax(W_out * h_T + b_out) for seq classification
            //                                                            or softmax(W_out * h_t + b_out) for each t for seq labeling

            // 2.b. Loss Computation:
            //    loss = ComputeLoss(predictions, Y_batch) // E.g., CrossEntropyLoss
            //    Mathematically justified: The loss function quantifies the discrepancy between model predictions and true labels.
            //    L = L(predictions(X_batch; θ), Y_batch)

            // 2.c. Backward Pass (Backpropagation Through Time - BPTT):
            //    Zero out gradients: ∇θ L = 0
            //    Compute gradients: ∇θ L = ∂L / ∂θ
            //    This involves unrolling the recurrent network through time and applying chain rule.
            //    For RNNs: ∂L/∂W = Σ_t (∂L_t/∂W), where L_t is loss at time t.
            //    Modern frameworks (PyTorch, TensorFlow) automate this via autodifferentiation.

            // 2.d. Parameter Update & Gradient Clipping:
            //    (Optional) Gradient Clipping:
            //    If ||∇θ L||_2 > max_grad_norm:
            //        ∇θ L = (max_grad_norm / ||∇θ L||_2) * ∇θ L
            //    Update parameters using an optimizer (e.g., Adam, SGD):
            //    θ = θ - η * ∇θ L  (for SGD)
            //    θ = OptimizerUpdate(θ, ∇θ L, η, optimizer_state) (for Adam, etc.)
            //    Mathematically justified: Parameters are updated in the direction that minimizes the loss function.
            //    Gradient clipping prevents exploding gradients by rescaling gradients if their norm exceeds a threshold.
    ```

#### D. Post-training Procedures

1.  **Model Persistence**:
    *   **Saving**: Storing the trained model's architecture and learned parameters ($\theta$) to disk.
    *   **Loading**: Retrieving the saved model for inference or further training.
    *   **Formats**: Common formats include framework-specific (e.g., PyTorch `.pt`, TensorFlow SavedModel) or interoperable formats (e.g., ONNX).

2.  **Quantization**:
    *   **Definition**: Reducing the precision of model weights and/or activations (e.g., from 32-bit floating point to 8-bit integer).
    *   **Mathematical Underpinning**: $w_{quantized} = \text{round}(w_{float} / S) + Z$, where $S$ is a scale factor and $Z$ is a zero-point.
    *   **Benefits**: Reduces model size, memory footprint, and can accelerate inference, especially on hardware with optimized low-precision arithmetic support.
    *   **Types**: Post-training quantization (PTQ), quantization-aware training (QAT).

3.  **Pruning**:
    *   **Definition**: Removing less important weights or connections from the trained model to reduce its size and computational complexity.
    *   **Mathematical Underpinning**: Identifying weights $w_{ij}$ with small magnitudes ($|w_{ij}| < \text{threshold}$) or low saliency scores and setting them to zero.
    *   **Benefits**: Can lead to significant model compression and faster inference with minimal accuracy loss if done carefully. Often requires fine-tuning after pruning.

#### E. Evaluation

1.  **Loss Functions**:
    *   **a. Cross-Entropy Loss ($L_{CE}$)**: For classification tasks.
        $$ L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c}) $$
        where $N$ is the number of samples, $C$ is the number of classes, $y_{i,c}$ is a binary indicator if sample $i$ belongs to class $c$, and $\hat{p}_{i,c}$ is the predicted probability of sample $i$ belonging to class $c$. For single-label multi-class classification, often simplified to $L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \log(\hat{p}_{i, y_i})$, where $y_i$ is the true class label.
    *   **b. Mean Squared Error (MSE, $L_{MSE}$)**: For regression tasks.
        $$ L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$
        where $y_i$ is the true value and $\hat{y}_i$ is the predicted value.

2.  **Performance Metrics**:
    *   **a. Classification Metrics**:
        *   **Accuracy**: $\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$ (TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives)
        *   **Precision**: $\text{Precision} = \frac{TP}{TP+FP}$
        *   **Recall (Sensitivity)**: $\text{Recall} = \frac{TP}{TP+FN}$
        *   **F1-Score**: $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
        *   **AUC-ROC**: Area Under the Receiver Operating Characteristic curve. Plots True Positive Rate (Recall) vs. False Positive Rate ($FP/(FP+TN)$) at various threshold settings.
    *   **b. Sequence Labeling Metrics (e.g., NER, POS)**:
        *   **Token-level F1-score**: Calculated over all tokens, often micro-averaged (aggregating TPs, FPs, FNs globally) or macro-averaged (averaging per-class F1-scores).
        *   **CoNLL evaluation script**: Standard for NER, considers entire entity spans.
    *   **c. Language Modeling Metrics**:
        *   **Perplexity (PP)**: $PP(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i | w_1, ..., w_{i-1})\right)$. Lower is better. Often computed as $\exp(\text{average cross-entropy loss})$.
    *   **d. Other Domain-Specific Metrics**:
        *   **BLEU, ROUGE, METEOR**: For machine translation, text summarization.
        *   **Word Error Rate (WER)**: For Automatic Speech Recognition (ASR). $WER = \frac{S+D+I}{N_{ref}}$, where $S$ is substitutions, $D$ deletions, $I$ insertions, $N_{ref}$ is number of words in reference.

#### F. Parameter Calculation
Let $d_x$ be the input feature dimension (embedding dimension) and $d_h$ be the hidden state dimension.

1.  **GRU Cell Parameters**:
    A GRU cell has three main transformations (for reset gate, update gate, candidate hidden state), each involving an input weight matrix, a hidden-state weight matrix, and a bias vector.
    *   Reset gate ($r_t$): $W_r \in \mathbb{R}^{d_h \times d_x}$, $U_r \in \mathbb{R}^{d_h \times d_h}$, $b_r \in \mathbb{R}^{d_h}$. Parameters: $d_h d_x + d_h^2 + d_h$.
    *   Update gate ($z_t$): $W_z \in \mathbb{R}^{d_h \times d_x}$, $U_z \in \mathbb{R}^{d_h \times d_h}$, $b_z \in \mathbb{R}^{d_h}$. Parameters: $d_h d_x + d_h^2 + d_h$.
    *   Candidate hidden state ($\tilde{h}_t$): $W_h \in \mathbb{R}^{d_h \times d_x}$, $U_h \in \mathbb{R}^{d_h \times d_h}$, $b_h \in \mathbb{R}^{d_h}$. Parameters: $d_h d_x + d_h^2 + d_h$.
    *   **Total parameters for one GRU cell**: $3 \times (d_h d_x + d_h^2 + d_h) = 3(d_x d_h + d_h^2 + d_h)$.
    (Note: PyTorch/TensorFlow often implement this with two bias terms per gate transformation, one for $Wx$ and one for $Uh$. If bias is applied after $Wx+Uh$, it's one bias. The formulation above uses one bias per gate. If two biases are used for each candidate/gate, it would be $3(d_x d_h + d_h^2 + 2d_h)$ or often $3((d_x+d_h)d_h + d_h)$ if biases are applied after matrix sum. Standard PyTorch implementation: input-hidden weights $W_{ih}, W_{rh}$ (each $3 d_h \times d_x$ and $3 d_h \times d_h$) and biases $b_{ih}, b_{rh}$ (each $3 d_h$). So, $3(d_x d_h + d_h d_h + d_h + d_h) = 3(d_h(d_x+d_h) + 2d_h)$.)
    A common simplification for parameter count is $3 \times (d_h (d_x + d_h) + d_h)$ when considering a single bias vector for each of the three main components. If biases are split for input and recurrent parts, it's $3 \times (d_h (d_x + d_h) + 2d_h)$. Using the standard definition from the equations:
    $P_{GRU} = (d_x d_h + d_h^2 + d_h)_{r} + (d_x d_h + d_h^2 + d_h)_{z} + (d_x d_h + d_h^2 + d_h)_{h} = 3(d_x d_h + d_h^2 + d_h)$.

2.  **LSTM Cell Parameters**:
    An LSTM cell has four main transformations (forget, input, output gates, and candidate cell state).
    *   Forget gate ($f_t$): $d_h d_x + d_h^2 + d_h$.
    *   Input gate ($i_t$): $d_h d_x + d_h^2 + d_h$.
    *   Candidate cell state ($\tilde{C}_t$): $d_h d_x + d_h^2 + d_h$.
    *   Output gate ($o_t$): $d_h d_x + d_h^2 + d_h$.
    *   **Total parameters for one LSTM cell**: $4 \times (d_h d_x + d_h^2 + d_h) = 4(d_x d_h + d_h^2 + d_h)$.
    Similar to GRU, PyTorch/TensorFlow parameterization might slightly differ but total count is usually represented as $4(d_h(d_x+d_h) + 2d_h)$.
    Using the standard definition from the equations:
    $P_{LSTM} = 4(d_x d_h + d_h^2 + d_h)$.

3.  **BiLSTM Layer Parameters**:
    A BiLSTM layer consists of two independent LSTM layers (one forward, one backward). Each LSTM has its own set of parameters. If both LSTMs have the same hidden dimension $d_h$:
    *   **Total parameters for BiLSTM layer**: $2 \times P_{LSTM} = 2 \times 4(d_x d_h + d_h^2 + d_h) = 8(d_x d_h + d_h^2 + d_h)$.
    The output dimension of the BiLSTM layer (if concatenating) will be $2d_h$.

**Note on Parameter Calculation in Frameworks (e.g., PyTorch `nn.GRU`, `nn.LSTM`):**
PyTorch's `nn.LSTM` and `nn.GRU` layers have parameters `weight_ih_l[k]` (input-hidden weights for layer `k`), `weight_hh_l[k]` (hidden-hidden weights), `bias_ih_l[k]`, and `bias_hh_l[k]`.
*   For LSTM:
    `weight_ih`: shape ($4d_h, d_x$)
    `weight_hh`: shape ($4d_h, d_h$)
    `bias_ih`: shape ($4d_h$)
    `bias_hh`: shape ($4d_h$)
    Total parameters: $4d_h d_x + 4d_h^2 + 4d_h + 4d_h = 4(d_h d_x + d_h^2 + 2d_h)$.
*   For GRU:
    `weight_ih`: shape ($3d_h, d_x$)
    `weight_hh`: shape ($3d_h, d_h$)
    `bias_ih`: shape ($3d_h$)
    `bias_hh`: shape ($3d_h$)
    Total parameters: $3d_h d_x + 3d_h^2 + 3d_h + 3d_h = 3(d_h d_x + d_h^2 + 2d_h)$.
These are the standard industrial implementations. The equations provided earlier are for conceptual understanding of individual gates; frameworks often fuse these for efficiency.