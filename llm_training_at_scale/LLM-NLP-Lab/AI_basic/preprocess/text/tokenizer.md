### Tokenizers

#### 1. Definition
A **Tokenizer** is a crucial component in Natural Language Processing (NLP) pipelines responsible for converting a raw input text string into a sequence of smaller units called **tokens**. These tokens can be words, characters, or sub-word units. The primary output is typically a sequence of numerical IDs, where each ID corresponds to a token in a predefined vocabulary. Tokenization is a foundational step for most downstream NLP tasks, enabling models to process and understand textual data.

The process can be abstractly represented as a function $ T: \mathcal{S} \rightarrow \mathcal{I}^* $, where $ \mathcal{S} $ is the space of input text strings, and $ \mathcal{I}^* $ is the set of sequences of integer IDs. Each ID $ i \in \mathcal{I} $ maps to a unique token $ t \in \mathcal{V} $, where $ \mathcal{V} $ is the tokenizer's vocabulary.

#### 2. Pertinent Equations and Mathematical Representations

While many tokenization methods are algorithmic, subword tokenization techniques often involve optimizing an objective function.

*   **Vocabulary Mapping:**
    *   Token to ID: $ \text{id} = \text{map}_{\mathcal{V}}(t) $
    *   ID to Token: $ t = \text{map}_{\mathcal{V}}^{-1}(\text{id}) $

*   **Byte Pair Encoding (BPE):** Implicitly optimizes by greedily choosing the most frequent pair of symbols (bytes or characters) for merging.
    *   Let $ C(s_1, s_2) $ be the count of an adjacent pair of symbols $ (s_1, s_2) $.
    *   At each step, find $ (s_a, s_b) = \arg\max_{(s_1, s_2)} C(s_1, s_2) $.
    *   Merge $ s_a $ and $ s_b $ into a new symbol $ s_{ab} $.

*   **WordPiece:** Aims to build a vocabulary $ \mathcal{V} $ that maximizes the likelihood of the training data when tokenized. Given a word, it is segmented into subword units $ w_1, \dots, w_k $. The score of a segmentation is often the sum of the scores of its constituent WordPieces.
    *   If $ \text{score}(w_i) $ is related to $ \log P(w_i) $, then for a word $ W $ segmented as $ w_1w_2...w_k $:
        $$ \text{Score}(W) = \sum_{i=1}^{k} \text{score}(w_i) $$
*  During vocabulary construction, pairs are merged if the likelihood of the corpus increases, i.e.:
$$  \log P(\text{new_unit}) - (\log P(\text{unit1}) + \log P(\text{unit2})) $$ 
is maximized.

*   **Unigram Language Model (ULM) Tokenizer:** Explicitly trains a unigram language model on subword units. The vocabulary and probabilities $ P(t) $ for each token $ t \in \mathcal{V} $ are optimized to maximize the likelihood of the training corpus.
    *   For a sentence $ X = x_1x_2...x_M $ (sequence of characters), let $ S(X) $ be the set of all possible segmentations into tokens from $ \mathcal{V} $. For a segmentation $ \mathbf{t} = (t_1, t_2, \dots, t_k) \in S(X) $:
        $$ P(\mathbf{t}) = \prod_{i=1}^{k} P(t_i) $$
    *   The marginal likelihood of the sentence $ X $ is:
        $$ P(X) = \sum_{\mathbf{t} \in S(X)} P(\mathbf{t}) $$
    *   The training objective is to maximize the total log-likelihood over the corpus $ D $:
        $$ \mathcal{L}(\mathcal{V}, P) = \sum_{X^{(s)} \in D} \log P(X^{(s)}) $$
    *   This is typically optimized using the Expectation-Maximization (EM) algorithm. For tokenizing a new sentence, the Viterbi algorithm is used to find the most probable segmentation:
        $$ \mathbf{t}^* = \arg\max_{\mathbf{t} \in S(X)} \prod_{i=1}^{k} P(t_i) $$

#### 3. Key Principles

*   **Vocabulary Construction:** Defining the set of unique tokens the tokenizer can recognize.
*   **Segmentation Strategy:** The rules or algorithms used to break down text into tokens.
*   **Out-of-Vocabulary (OOV) Handling:** Mechanism for dealing with tokens not present in the vocabulary (e.g., `[UNK]` token, subword decomposition).
*   **Normalization:** Pre-processing text to a canonical form (e.g., lowercasing, Unicode normalization) to reduce vocabulary size and improve consistency.
*   **Pre-tokenization:** An initial splitting of text (e.g., by whitespace or punctuation) before more complex subword tokenization.
*   **Reversibility (Detokenization):** The ability to convert a sequence of tokens back into a human-readable string, ideally recovering the original text or a close approximation.
*   **Efficiency:** Computational performance of both tokenization and detokenization processes.
*   **Balance between Vocabulary Size and Sequence Length:** A smaller vocabulary might lead to longer token sequences, while a larger vocabulary might shorten sequences but increase model parameter size and OOV sparsity for rare words. Subword tokenizers aim to strike this balance.
*   **Handling of Special Tokens:** Incorporating tokens with specific meanings for model operations (e.g., `[CLS]`, `[SEP]`, `[PAD]`).

#### 4. Detailed Concept Analysis and Methodologies

##### 4.1. Fundamental Concepts

*   **4.1.1. Vocabulary ($ \mathcal{V} $):**
    *   Definition: The finite set of all unique tokens that the tokenizer recognizes and can map to numerical IDs.
    *   Construction: Derived from a training corpus. Its size is a critical hyperparameter.
*   **4.1.2. Token ID Mapping:**
    *   Process: Each token in $ \mathcal{V} $ is assigned a unique integer ID. This numerical representation is the input to machine learning models.
*   **4.1.3. Out-of-Vocabulary (OOV) Handling:**
    *   Problem: Input text may contain words or symbols not present in $ \mathcal{V} $.
    *   Solutions:
        *   Assigning a special `[UNK]` (unknown) token.
        *   For subword tokenizers: Decomposing OOV words into known subword units.
        *   Character-level decomposition as a fallback.

##### 4.2. Categorization of Tokenization Techniques

*   **4.2.1. Word-Level Tokenization:**
    Segments text based on explicit word boundaries.
    *   **4.2.1.1. Whitespace Tokenization:**
        *   Methodology: Splits text based on whitespace characters (spaces, tabs, newlines).
        *   Pseudocode:
            ```pseudocode
            Function Whitespace_Tokenize(text):
              tokens = text.split(whitespace_delimiters)
              Return tokens
            ```
    *   **4.2.1.2. Punctuation-based Tokenization:**
        *   Methodology: Splits text by whitespace and also treats punctuation marks as separate tokens. Often uses regular expressions.
    *   **4.2.1.3. Rule-Based Tokenization (e.g., Penn Treebank, Moses):**
        *   Methodology: Employs a comprehensive set of manually crafted rules to handle various linguistic phenomena like contractions (e.g., "don't" -> "do", "n't"), hyphens, and specific punctuation patterns.

*   **4.2.2. Character-Level Tokenization:**
    *   Methodology: Treats each character as an individual token. The vocabulary consists of all unique characters in the corpus.
    *   Pseudocode:
        ```pseudocode
        Function Character_Tokenize(text):
          tokens = list(text) // Converts string to list of characters
          Return tokens
        ```

*   **4.2.3. Subword-Level Tokenization:**
    Strikes a balance between word and character tokenization. Common words remain single tokens, while rare words are broken down into smaller, meaningful subword units. This helps manage vocabulary size and OOV issues.

    *   **4.2.3.1. Byte Pair Encoding (BPE):**
        Originally a data compression algorithm, adapted for tokenization.
        *   **Vocabulary Construction Algorithm:**
            1.  **Initialization:**
                *   Define an initial vocabulary consisting of all individual characters present in the training corpus.
                *   Represent words in the corpus as sequences of these characters, often adding a special end-of-word symbol (e.g., `</w>`) to each word after pre-tokenization (e.g., by whitespace).
            2.  **Iterative Merging:**
                *   Repeatedly count all adjacent pairs of symbols in the current representation of the corpus.
                *   Identify the most frequent pair (e.g., `('a', 'b')`).
                *   Merge this pair into a new, single symbol (e.g., `'ab'`).
                *   Add this new symbol to the vocabulary.
                *   Replace all occurrences of the pair `('a', 'b')` in the corpus representation with the new symbol `'ab'`.
            3.  **Termination:** Stop when the desired vocabulary size is reached or no more pairs can be merged (or frequency drops below a threshold). The learnable merge operations are stored in order of priority (their learning order).
        *   **Vocabulary Construction Pseudocode:**
            ```pseudocode
            Function BPE_Build_Vocab(corpus, target_vocab_size):
              // Pre-tokenize corpus into words, split words into characters, add counts
              word_freqs = Pretokenize_And_Count_Words(corpus)
              vocab = Initialize_Char_Vocab(word_freqs)
              merges = {} // Stores merge operations

              // Represent words as char sequences + </w> with frequencies
              // E.g., {"l o w </w>": 5, "n e w e r </w>": 2, ...}
              splits = {tuple(word) + ('</w>',) : freq for word, freq in word_freqs.items()}

              num_merges_to_perform = target_vocab_size - len(vocab)
              For i from 1 to num_merges_to_perform:
                pair_freqs = Compute_Pair_Frequencies(splits)
                If not pair_freqs: Break // No more pairs to merge
                best_pair = ArgMax(pair_freqs) // Pair with highest frequency
                
                new_token = best_pair[0] + best_pair[1]
                vocab.add(new_token)
                merges[best_pair] = new_token // Store merge rule

                // Update splits by applying the new merge
                new_splits = {}
                For sequence, freq in splits.items():
                  updated_sequence = Replace_Pair_In_Sequence(sequence, best_pair, new_token)
                  new_splits[updated_sequence] = new_splits.get(updated_sequence, 0) + freq
                splits = new_splits
              
              Return vocab, merges (ordered by learning sequence)
            ```
        *   **Tokenization Algorithm (using learned merges):**
            1.  Pre-tokenize the input text into words.
            2.  For each word, convert it into a sequence of characters, appending an end-of-word symbol.
            3.  Iteratively apply the learned merge operations (highest priority first, i.e., those learned earlier) to the character sequence until no more merges are possible.
        *   **Tokenization Pseudocode:**
            ```pseudocode
            Function BPE_Tokenize(text, ordered_merges, vocab):
              // Pre-tokenize by space or other rules
              raw_words = Pretokenize_Text(text)
              all_tokens = []
              For each raw_word in raw_words:
                // Add end-of-word symbol, if BPE model uses it
                if raw_word + '</w>' in vocab: // Handle pre-existing full words
                     tokens = [raw_word + '</w>']
                else:
                     tokens = list(raw_word) + ['</w>'] 
                
                // Iteratively apply merges
                while True:
                  has_merged_in_iteration = False
                  j = 0
                  temp_tokens = []
                  while j < len(tokens):
                    found_merge_for_pair = False
                    // Check against learned merge operations (ordered_merges)
                    // This part can be optimized significantly
                    // A common implementation iterates through ordered_merges
                    // and applies the first possible one.
                    // For simplicity: find best merge in current tokens
                    best_current_merge_pair = null
                    highest_priority_for_best_merge = -1

                    // Iterate through possible merges (from ordered_merges list)
                    for k from 0 to len(ordered_merges) - 1:
                        pair_to_merge = ordered_merges[k] // This is (s1, s2)
                        if j < len(tokens) - 1 and tokens[j] == pair_to_merge[0] and tokens[j+1] == pair_to_merge[1]:
                           best_current_merge_pair = pair_to_merge
                           // priority is higher for earlier merges (lower k)
                           break // Found highest priority merge applicable at tokens[j], tokens[j+1]
                    
                    if best_current_merge_pair != null:
                        merged_token = best_current_merge_pair[0] + best_current_merge_pair[1]
                        temp_tokens.append(merged_token)
                        j += 2
                        has_merged_in_iteration = True
                    else:
                        temp_tokens.append(tokens[j])
                        j += 1
                  tokens = temp_tokens
                  if not has_merged_in_iteration:
                     break

                // Fallback for OOV characters if any token is not in vocab
                final_word_tokens = []
                for tok in tokens:
                    if tok in vocab:
                        final_word_tokens.append(tok)
                    else:
                        // Handle OOV characters or sub-tokens not in vocab explicitly (map to UNK or char-by-char)
                        final_word_tokens.extend(Handle_OOV_Subtoken(tok, vocab))


                all_tokens.extend(final_word_tokens)
              Return all_tokens
            ```
            *Note: Efficient BPE tokenization often caches merge operations or uses different data structures for faster lookups.*

    *   **4.2.3.2. WordPiece:**
        Used in models like BERT. Similar to BPE, but merge decisions are based on maximizing the likelihood of the training data, rather than raw frequency.
        *   **Vocabulary Construction Principle:**
            1.  **Initialization:** Start with a vocabulary of all individual characters.
            2.  **Iterative Merging:** Iteratively combine pairs of existing tokens to form new tokens. The pair chosen for merging is the one that increases the likelihood of the training corpus the most when added to the model.
                The score for merging two subwords $ t_1, t_2 $ to form $ t_{12} $ is often $ \text{score}(t_{12}) = \text{count}(t_{12}) / (\text{count}(t_1) \times \text{count}(t_2)) $ (or log-likelihood variant).
            3.  **Termination:** Stop when the vocab size is reached or likelihood gain falls below a threshold.
        *   **Tokenization Algorithm (Greedy Longest Match):**
            1.  Given a word, try to find the longest subword in the vocabulary that is a prefix of the word.
            2.  If found, this subword becomes a token, and the process is repeated for the remainder of the word.
            3.  If no prefix is found in the vocabulary (other than single characters, if they are not the start of a multi-character token), the word might be tokenized using an `[UNK]` token or broken down into characters that are in the vocabulary. WordPiece often uses `##` to denote subwords that are part of a larger word but not at the beginning.
        *   **Tokenization Pseudocode:**
            ```pseudocode
            Function WordPiece_Tokenize(word, vocab):
              if word in vocab:
                Return [word]
              
              tokens = []
              current_pos = 0
              while current_pos < len(word):
                best_match = ""
                best_match_len = 0
                // Iterate from longest possible subword to shortest
                for sub_len from len(word) - current_pos down to 1:
                  substring = word[current_pos : current_pos + sub_len]
                  // Check for ## prefix for non-initial parts
                  token_to_check = substring
                  if current_pos > 0:
                    token_to_check = "##" + substring
                  
                  if token_to_check in vocab:
                    best_match = token_to_check
                    best_match_len = sub_len
                    break // Found longest valid prefix
                
                if best_match_len > 0:
                  tokens.append(best_match)
                  current_pos += best_match_len
                else:
                  // Fallback: if no subword found, mark as UNK or handle character-wise
                  tokens.append("[UNK]") 
                  Return tokens // Or break word into constituent known characters if possible
              Return tokens
            
            // Main Tokenizer
            Function Full_WordPiece_Tokenize(text, vocab):
                pre_tokenized_words = Pretokenize_Text(text) // e.g., by whitespace
                all_output_tokens = []
                for word_orig in pre_tokenized_words:
                    // Apply normalization (e.g. lowercasing, accent stripping) if model expects it
                    word = Normalize(word_orig) 
                    all_output_tokens.extend(WordPiece_Tokenize(word, vocab))
                Return all_output_tokens
            ```

    *   **4.2.3.3. Unigram Language Model (ULM):**
        Used by SentencePiece, T5. Assumes each subword token occurs independently.
        *   **Vocabulary Construction Principle:**
            1.  **Initialization:** Start with a large vocabulary of candidate subwords (e.g., all substrings from the corpus, or results from BPE).
            2.  **EM Algorithm:** Estimate $ P(t_i) $ for each subword $ t_i $ using EM.
                *   E-step: Given current $ P(t_i) $, for each word in the training corpus, find the most likely segmentation using Viterbi algorithm. Calculate expected counts for each subword.
                *   M-step: Re-estimate $ P(t_i) $ based on these expected counts.
            3.  **Iterative Pruning:** Iteratively remove a certain percentage (e.g., 10-20%) of subwords whose removal leads to the smallest increase in the total corpus perplexity (or decrease in log-likelihood). The loss associated with removing a subword $ t_x $ is $ \sum P(\text{data} | \mathcal{V}) - P(\text{data} | \mathcal{V} - \{t_x\}) $.
            4.  **Termination:** Repeat EM and pruning until the desired vocabulary size is reached.
        *   **Tokenization Algorithm (Viterbi Algorithm):**
            Given an input string (word or sentence), find the segmentation $ \mathbf{t}^* = (t_1, \dots, t_k) $ that maximizes $ \prod P(t_i) $ (or $ \sum \log P(t_i) $).
            *   Let $ \text{text} $ be the input string of length $ N $.
            *   $ DP[i] $ stores the maximum log-probability of segmenting the prefix $ \text{text}[0 \dots i-1] $.
            *   $ DP[0] = 0 $.
            *   For $ i = 1 \dots N $:
                $$ DP[i] = \max_{1 \le j \le i} (DP[i-j] + \log P(\text{text}[i-j \dots i-1])) $$
                (provided $ \text{text}[i-j \dots i-1] $ is in $ \mathcal{V} $)
            *   Backtrack to find the actual token sequence.
        *   **Pseudocode for Viterbi Tokenization:**
            ```pseudocode
            Function Unigram_Tokenize_Viterbi(text, vocab_with_log_probs):
              N = len(text)
              // Min score for unsegmentable string (or use very small log_prob for UNK)
              min_log_prob = -float('inf') 
              
              scores = [min_log_prob] * (N + 1)
              backpointers = [0] * (N + 1) // Stores start index of last token in best path
              scores[0] = 0.0

              For i from 1 to N: // Current end position (exclusive)
                For j from 0 to i-1: // Potential start position (inclusive)
                  subword = text[j:i]
                  if subword in vocab_with_log_probs:
                    log_prob = vocab_with_log_probs[subword]
                    current_score = scores[j] + log_prob
                    if current_score > scores[i]:
                      scores[i] = current_score
                      backpointers[i] = j
                  elif len(subword) == 1 and "[UNK]" in vocab_with_log_probs: // Handle unknown chars as UNK
                      log_prob = vocab_with_log_probs["[UNK]"]
                      current_score = scores[j] + log_prob
                      if current_score > scores[i]: # Better path to i via UNK
                          scores[i] = current_score
                          backpointers[i] = j


              if scores[N] == min_log_prob:
                Return ["[UNK]"] // Cannot segment text

              tokens = []
              current_end = N
              while current_end > 0:
                prev_end = backpointers[current_end]
                token = text[prev_end:current_end]
                if token not in vocab_with_log_probs: // If an UNK char was chosen
                    token = "[UNK]" 
                tokens.insert(0, token)
                current_end = prev_end
              
              Return tokens
            ```

    *   **4.2.3.4. SentencePiece:**
        A library developed by Google that implements BPE and Unigram tokenization directly from raw text.
        *   **Key Features:**
            *   **Treats text as a sequence of Unicode characters:** Eliminates the need for language-specific pre-tokenizers. Whitespace is handled as a normal symbol (e.g., ` ` (U+2581)) and included in the subword segmentation.
            *   **Normalization:** Provides customizable normalization rules (e.g., NFKC).
            *   **Reversible:** Guarantees lossless conversion between text and token sequences by design because whitespace is part of the tokens.
            *   **Vocabulary and Model:** Combines vocabulary and algorithm (BPE or Unigram) into a single `​.model` file.

##### 4.3. Pre-processing and Post-processing

*   **4.3.1. Normalization:**
    Standardizing text before tokenization.
    *   **Unicode Normalization:** E.g., NFC (canonical composition), NFD (canonical decomposition), NFKC, NFKD. NFKC is common for converting visually similar characters or pre-composed characters.
    *   **Lowercasing:** Converting all text to lowercase.
    *   **Accent Removal:** Stripping accents from characters (e.g., 'é' -> 'e').
    *   **Whitespace Cleaning:** Standardizing multiple spaces, removing leading/trailing whitespace.

*   **4.3.2. Pre-tokenization:**
    An initial, often simpler, tokenization step before subword algorithms are applied.
    *   **Definition:** Splits input text into a list of "words" or preliminary segments. Subword algorithms then operate on these segments.
    *   **Common Strategies:**
        *   Whitespace splitting.
        *   Rule-based splitting on punctuation and whitespace (e.g., GPT-2/BERT pre-tokenizers).
        *   Specialized pre-tokenizers for specific scripts or cases.
    *   For SentencePiece, pre-tokenization is minimal as it operates nearly on raw streams, but it can include normalization, and handling of special markers like digits.

*   **4.3.3. Special Tokens:**
    Tokens with predefined meanings, essential for many model architectures.
    *   `[UNK]` (Unknown): Represents OOV tokens.
    *   `[PAD]` (Padding): Used to make input sequences of uniform length in a batch.
    *   `[CLS]` (Classification): Prepended to input for sequence classification tasks (e.g., in BERT).
    *   `[SEP]` (Separator): Separates segments/sentences in the input (e.g., in BERT for sentence-pair tasks).
    *   `[MASK]` (Masking): Used in Masked Language Modeling (MLM) pre-training (e.g., in BERT).
    *   `<s>`, `</s>` or `BOS`, `EOS`: Denote start and end of a sentence.
    *   Language codes (e.g., `<en>`, `<fr>`): For multilingual models to specify input/output language.

*   **4.3.4. Detokenization:**
    The process of converting a sequence of tokens (or their IDs) back into a human-readable text string.
    *   **Process:** Involves joining tokens, removing special end-of-word markers (like `</w>` or `##`), handling special tokens, and potentially restoring original casing or spacing if information was preserved or rules are available.
    *   **Challenges:** Accurate reconstruction of original spacing and punctuation, especially if aggressive normalization or non-reversible tokenization steps were applied. SentencePiece excels here due to its handling of whitespace.

#### 5. Importance

*   **Fundamental for NLP:** Tokenization is the first step in transforming unstructured text into a format suitable for machine learning models.
*   **Impact on Model Performance:** The choice of tokenizer profoundly affects downstream model performance, vocabulary size, and computational efficiency.
*   **Handling Morphological Richness:** Subword tokenizers can decompose morphologically complex words (e.g., in German, Turkish) into constituent morphemes or meaningful parts, improving generalization.
*   **Managing Vocabulary Size:** Subword methods control vocabulary explosion compared to word-based tokenizers, especially for large and diverse corpora.
*   **Reducing OOV Problem:** Subword tokenization significantly reduces the number of unknown words by breaking them into known sub-units.
*   **Enabling Cross-lingual Models:** Shared subword vocabularies allow models to process and learn from multiple languages simultaneously, facilitating cross-lingual transfer.
*   **Input Segmentation for Transformers:** Transformer models rely on sequences of tokens (embeddings) as input; tokenization provides this discrete representation.

#### 6. Pros versus Cons

| Feature           | Word-Level Tokenizers                                  | Character-Level Tokenizers                                 | Subword-Level Tokenizers (BPE, WordPiece, Unigram)            |
|-------------------|--------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------|
| **Pros**          | - Intuitive, human-readable tokens.                   | - Small, fixed vocabulary.                               | - Balances vocabulary size and sequence length.            |
|                   | - Preserves word semantics directly.                  | - No OOV words.                                            | - Handles OOV words gracefully by decomposition.             |
|                   | - Simple algorithms.                                   | - Captures character-level patterns (e.g., misspellings).  | - Effective for morphologically rich languages.            |
|                   |                                                        |                                                            | - Good empirical performance in SOTA models.                 |
|                   |                                                        |                                                            | - Can create shared vocabularies for multiple languages.     |
| **Cons**          | - Large vocabulary size for diverse corpora.          | - Very long token sequences.                               | - Tokenization can be computationally more intensive (training). |
|                   | - Severe OOV problem for rare/new words.              | - Destroys word-level semantics at input.                  | - Tokens may not always be linguistically meaningful.        |
|                   | - Poor handling of morphology and word variations.    | - Computationally demanding for models due to seq. length. | - Segmentation ambiguity (though ULM can produce multiple).   |
|                   | - Language-dependent rules often required.            | - Slower information propagation over long distances.      | - Vocabulary construction methods can be complex.            |
|                   | - Inconsistent tokenization across different tools.   |                                                            | - Detokenization requires care to be lossless.               |

#### 7. Cutting-edge Advances

*   **Learned Pre-tokenization:** Utilizing neural models or learned rules to perform the initial split of text into "words" more adaptively before subword tokenization, rather than relying solely on hard-coded rules.
*   **Byte-level Tokenizers (e.g., ByT5, Charformer, CANINE):**
    *   Operate directly on UTF-8 byte sequences.
    *   The vocabulary can be as small as 256 (all possible bytes).
    *   Truly language-agnostic and robust to noise, code-switching, and unseen characters without `[UNK]`.
    *   Results in longer sequences but potentially more fundamental representations.
    *   Some models (e.g., Charformer) use byte-level inputs but learn to group them into latent subword-like units.
*   **Tokenization-Free / Near Tokenization-Free Models:** Models aiming to process raw text (bytes or characters) directly, or with minimal, fixed tokenization (e.g., splitting into characters), pushing the segmentation learning entirely into the neural network.
*   **Adaptive/Dynamic Tokenization:** Tokenizers or vocabularies that can be updated or adapt during training or even inference, potentially adding new frequent subwords or specializing for a specific domain.
*   **Improved Multilingual Tokenization:**
    *   Techniques for better balancing script diversity and vocabulary sharing in multilingual settings.
    *   Methods to ensure fair representation of low-resource languages in shared vocabularies.
*   **Controllable Tokenization:** Research into tokenization schemes where the granularity or style of tokenization can be controlled based on downstream task requirements or other input signals.
*   **Contextual Tokenization:** Tokenization decisions that might depend not just on the word itself but also its context, though this blurs the line with early model layers.
*   **Joint Learning of Tokenization and Model Parameters:** End-to-end training paradigms where tokenizer parameters (e.g., vocabulary in Unigram, merge rules) are optimized jointly with the main model parameters, although this is computationally challenging.
*   **Semantic Tokenization:** Efforts to produce tokens that align more closely with semantic units rather than purely statistical or frequency-based segments, potentially using external linguistic knowledge or learned semantic representations.