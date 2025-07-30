# MSL Transformer Project Plan (M3 Max Optimization)

This document outlines the plan for developing a Transformer model using Metal Shading Language (MSL), optimized for an Apple M3 Max with 36GB Unified Memory. The primary dataset will be BookCorpus.

## Overall Architecture
We will begin by planning for a **decoder-only Transformer** architecture (similar to GPT models), suitable for language modeling tasks.

## Memory Strategy (M3 Max - 36GB Unified Memory)
-   **Unified Memory:** Leverage `MTLStorageModeShared` for MSL buffers to allow efficient CPU/GPU data access.
-   **Data Types:** Prioritize `half` (16-bit float) for weights and activations where possible to reduce memory footprint and improve performance. Use `float` (32-bit float) for precision-critical operations.
-   **Buffer Management:** Implement careful allocation, reuse, and deallocation of `MTLBuffer` objects.

## Phase 1: Data Preprocessing & Tokenization (CPU-side: C++/Swift)

### Task 1.1: Data Loading & Cleaning
-   **Core Plan:**
    -   Load text data from `data/bookcorpus/books_large_p1.txt` and `data/bookcorpus/books_large_p2.txt`.
    -   Implement stream processing for large files to manage memory usage.
    -   Perform basic cleaning: normalize whitespace, potentially lowercase. configurable cleaning steps.
-   **Alternative Options/Considerations:**
    -   Explore different text cleaning libraries or techniques based on initial data analysis.
    -   Assess the impact of aggressive cleaning (e.g., removing all punctuation) vs. minimal cleaning on downstream performance and tokenization.

### Task 1.2: Tokenization
-   **Core Plan:**
    -   Implement Byte Pair Encoding (BPE) as a starting point.
    -   Train the BPE tokenizer on a representative subset of the BookCorpus data.
    -   Define and handle special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`.
    -   Convert text into sequences of integer token IDs.
    -   Save the trained tokenizer (vocabulary and merge rules) for consistent use.
-   **Alternative Options/Considerations:**
    -   **Other Tokenization Methods:**
        -   WordPiece (used by BERT).
        -   SentencePiece (language-agnostic, treats text as a sequence of Unicode characters).
        -   Character-level tokenization (simpler, larger sequences, potentially less effective representation).
    -   **Pre-trained Tokenizers:** Consider using a pre-trained tokenizer (e.g., from Hugging Face tokenizers library, if its vocabulary is suitable) and adapt its usage. This would require mapping its output to our system.
    -   **Vocabulary Size:** Experiment with different vocabulary sizes. A larger vocab captures more words but increases embedding table size and final softmax layer complexity. Common sizes range from 30k to 50k.
    -   **Normalization:** Unicode normalization (NFC, NFKC) before tokenization.

### Task 1.3: Data Formatting for MSL
-   **Core Plan:**
    -   Convert tokenized sequences into arrays of `uint32` (or `int32`) for MSL buffers.
    -   Implement padding to create fixed-length sequences for batching. Truncate longer sequences.
-   **Alternative Options/Considerations:**
    -   **Integer Types:** If vocabulary size is small enough (e.g., < 65536), `uint16` could be used for token IDs to save memory, but `uint32` is generally safer and aligns well.
    -   **Dynamic Sequence Lengths:** Handling truly dynamic sequence lengths in MSL batches is complex. Padding is standard. Techniques like bucketing (grouping sequences of similar lengths) can reduce padding overhead but add complexity to batching logic. For a first pass, fixed-length padding is recommended.

## Phase 2: MSL Kernel Implementation (Core Transformer Components)

### Task 2.1: Embedding Layer
-   **Core Plan:**
    -   MSL kernel to look up token embeddings from a weight matrix (`MTLBuffer`).
    -   Input: Token IDs, Embedding Weight Matrix.
    -   Output: Token Embeddings (vectors of `half` or `float`).
-   **Alternative Options/Considerations:**
    -   **Storage Mode for Weights:** While `MTLStorageModeShared` is generally best for M-series, if the embedding matrix is exceptionally large and strictly read-only by the GPU post-initialization, `MTLStorageModePrivate` (populated via a blit command) could be benchmarked for potential minor performance differences, though shared is usually optimal due to avoiding the blit.

### Task 2.2: Positional Encoding
-   **Core Plan:**
    -   Add sinusoidal positional encodings to token embeddings. These can be pre-calculated and stored in an `MTLBuffer` or calculated on-the-fly in the kernel. Pre-calculation is common.
-   **Alternative Options/Considerations:**
    -   **Learned Positional Embeddings:** Implement as another embedding layer (parameters to be learned). Increases model size but can sometimes offer better performance. Requires changes to the backward pass.
    -   **Relative Positional Encoding (e.g., Transformer-XL, T5):** More complex to implement in MSL, involves modifying the attention mechanism to consider relative positions. Potentially better for very long sequences.
    -   **No Positional Encoding:** Some newer architectures experiment without explicit PEs, relying on the model to learn order. For a standard Transformer, PEs are crucial.

### Task 2.3: Multi-Head Self-Attention (MHSA)
This is a critical and complex component.

#### Task 2.3.1: QKV Projection
-   **Core Plan:**
    -   MSL kernel for matrix multiplication: input embeddings with weight matrices to produce Query (Q), Key (K), and Value (V) vectors for each attention head.
    -   Use `half` precision for weights and activations where possible.
    -   Efficiently split embedding dimension into multiple heads.
-   **Alternative Options/Considerations:**
    -   **Fused QKV Projection:** Perform a single larger matrix multiplication (input with a combined `W_qkv` matrix) and then split the result into Q, K, V. This can sometimes be more efficient due to better GPU utilization for the larger matmul.
    -   **Matrix Multiplication Optimization:**
        -   Utilize `simdgroup_matrix` or `threadgroup_matrix` operations available in newer Metal versions for optimized matrix multiplications.
        -   Manually optimize using threadgroup memory for blocking if not using the above primitives or for very specific matrix sizes.

#### Task 2.3.2: Scaled Dot-Product Attention
-   **Core Plan:**
    1.  Compute `Q @ K^T / sqrt(d_k)` (scaled dot products).
    2.  Apply causal masking for decoder architecture (set future positions to a large negative number before Softmax).
    3.  Implement a numerically stable Softmax function.
    4.  Compute `Softmax_output @ V` to get context vectors.
-   **Alternative Options/Considerations:**
    -   **FlashAttention-style Kernels:** If memory bandwidth for QK^T and V matrices becomes a bottleneck for large sequence lengths, implementing a fused kernel inspired by FlashAttention (which avoids materializing the full attention matrix and uses tiling and recomputation) would be a significant optimization. This is an advanced task requiring careful MSL implementation.
    -   **Softmax Optimization:** Different ways to implement softmax in parallel, ensuring numerical stability (subtracting max element in each row).
    -   **Head Dimension (`d_k`):** Typically `embedding_dim / num_heads`.

#### Task 2.3.3: Concatenation and Output Projection
-   **Core Plan:**
    -   Concatenate context vectors from all attention heads.
    -   Apply a final linear projection (matrix multiplication) with a weight matrix `W_o`.
-   **Alternative Options/Considerations:**
    -   Memory layout for head outputs prior to concatenation can impact performance. Ensure coalesced access.

### Task 2.4: Add & Norm (Layer Normalization)
-   **Core Plan:**
    -   MSL kernel to implement Layer Normalization: `output = gamma * (x - mean) / sqrt(variance + epsilon) + beta`.
    -   Residual connection: `output = LayerNorm(input_to_sublayer + sublayer_output)`.
-   **Alternative Options/Considerations:**
    -   **RMSNorm:** `output = x / sqrt(mean_of_squares(x) + epsilon) * gamma`. Simpler, fewer parameters, often performs comparably or better.
    -   **Pre-LN vs. Post-LN:**
        -   **Post-LN (this plan):** `x + SubLayer(LayerNorm(x))`. Original Transformer architecture.
        -   **Pre-LN:** `x + LayerNorm(SubLayer(x))`. Often leads to more stable training. This would change the order of operations.
    -   **Normalization Epsilon:** `1e-5` or `1e-6` is common.

### Task 2.5: Feed-Forward Network (FFN)
-   **Core Plan:**
    -   Two linear layers with a non-linearity in between.
    -   Typically: `Linear -> GELU -> Linear`.
    -   Expansion factor for the intermediate layer (e.g., 4x embedding_dim).
-   **Alternative Options/Considerations:**
    -   **Activation Functions:**
        -   ReLU: Simpler, faster, but GELU often performs better.
        -   SwiGLU / GEGLU: More complex (involves splitting activation and element-wise product) but can improve quality. E.g., `FFN_SwiGLU(x, W, V, W2) = (Swish(xW) * xV)W2`. Increases parameter count in the FFN.
    -   **Bias:** FFN layers typically include biases.

### Task 2.6: Final Linear Layer & Softmax (Output Layer)
-   **Core Plan:**
    -   A linear layer to project the final Transformer block's output to vocabulary size (logits).
    -   Softmax is typically applied within the cross-entropy loss function during training for numerical stability, rather than as a standalone layer producing probabilities. For inference, a Softmax kernel might be needed for generating probabilities if required before sampling.
-   **Alternative Options/Considerations:**
    -   **Weight Tying:** Tie the weights of this final linear layer with the input embedding matrix (`Task 2.1`). This reduces model parameters significantly and often improves performance. The dimensions must match (embedding_dim = hidden_dim). If they don't, an additional projection might be needed or weight tying isn't directly applicable.

## Phase 3: Model Assembly & Training Orchestration (Host Code: C++/Swift + Metal API)

### Task 3.1: Model Definition (Host Code)
-   **Core Plan:**
    -   Define the overall Transformer architecture (number of layers, heads, dimensions) in C++ or Swift.
    -   Manage `MTLDevice`, `MTLCommandQueue`, `MTLLibrary`, `MTLComputePipelineState` objects for each MSL kernel.
-   **Alternative Options/Considerations:**
    -   **Higher-Level Metal Wrappers:** While direct Metal API offers maximum control, a thin C++ wrapper around Metal objects could simplify boilerplate if the project grows very large. For this specific project, direct API is fine.

### Task 3.2: Parameter Initialization
-   **Core Plan:**
    -   Initialize model weights (e.g., Xavier/Glorot, Kaiming initialization).
    -   Load weights into `MTLBuffer`s using `MTLStorageModeShared`.
-   **Alternative Options/Considerations:**
    -   Specific initialization schemes tailored for Transformers (e.g., smaller variance for layers deeper in the network).
    -   Loading pre-trained weights (if adapting an existing model, not our primary goal here).

### Task 3.3: Training Loop
-   **Core Plan:**
    -   Implement the main training loop:
        1.  Batch generation from preprocessed data.
        2.  Forward Pass: Encode commands for MSL kernels in sequence on a `MTLCommandBuffer`.
        3.  Loss Calculation (Cross-Entropy Loss): Ideally an MSL kernel for performance. Input: logits, target token IDs. Output: loss value.
        4.  **Backward Pass (Autodiff in MSL):** This is the most challenging part.
            -   Derive gradient formulas for each operation (matrix multiply, softmax, layer norm, activations, embedding lookup).
            -   Implement corresponding MSL kernels for each gradient calculation.
            -   Manage buffers for activations (saved from forward pass) and gradients.
        5.  Optimizer Step (e.g., Adam, AdamW): Update weights using gradients. This will also be one or more MSL kernels.
-   **Alternative Options/Considerations:**
    -   **Autodiff Strategy:**
        -   **Full MSL Autodiff (Target):** Provides maximum control and performance. Requires significant effort in deriving and implementing gradient kernels.
        -   **CPU Gradients (Fallback/Debug):** Transfer necessary data to CPU, compute gradients (e.g., using a C++ autodiff library), transfer gradients back. Very slow for large models, not viable for final solution but can be useful for verifying MSL gradient calculations.
        -   **Mixed Approach:** Certain complex gradients could initially be prototyped on CPU while core MSL grad kernels are developed.
    -   **Optimizers:**
        -   Adam / AdamW (common for Transformers).
        -   SGD with momentum.
        -   Learning rate schedulers (e.g., linear warmup with cosine decay). Optimizer state (moment vectors for Adam) must be stored in `MTLBuffer`s.
    -   **Gradient Clipping:** Implement if gradients explode.
    -   **Gradient Accumulation:** If batch sizes are limited by memory, accumulate gradients over multiple mini-batches before an optimizer step.

### Task 3.4: Inference/Text Generation
-   **Core Plan:**
    -   Implement autoregressive decoding:
        1.  Input prompt (tokenized).
        2.  Forward pass through the model to get logits for the next token.
        3.  Sample a token from the logits (e.g., greedy sampling, top-k sampling, nucleus sampling).
        4.  Append the new token to the sequence and repeat.
    -   **KV Caching:** Crucial for efficient inference. Cache Key (K) and Value (V) tensors from self-attention layers for previous tokens, so they don't need to be recomputed for each new token. This requires specific MSL kernels to manage and update these caches.
-   **Alternative Options/Considerations:**
    -   **Sampling Strategies:**
        -   Greedy: Simplest, picks highest probability token.
        -   Top-k: Sample from the k most probable tokens.
        -   Top-p (Nucleus): Sample from the smallest set of tokens whose cumulative probability exceeds p.
        -   Temperature scaling: Adjust sharpness of probability distribution.
    -   **Beam Search:** Maintain multiple candidate sequences (beams) for higher quality generation, but more computationally expensive.

## Phase 4: Optimization & Profiling

### Task 4.1: Performance Profiling
-   **Core Plan:**
    -   Use Metal System Trace in Xcode to identify GPU time, memory bandwidth issues, shader occupancy, and other bottlenecks.
    -   Analyze individual kernel execution times.
-   **Alternative Options/Considerations:**
    -   `MTLCounterSampleBuffer` for more fine-grained GPU counters within code.
    -   Custom timing around command buffer submission and completion (coarser but useful).

### Task 4.2: Kernel Optimization
-   **Core Plan:**
    -   Based on profiling, refine MSL kernels:
        -   Optimize memory access patterns (coalescing, minimizing bank conflicts in threadgroup memory).
        -   Tune threadgroup sizes (`threads_per_threadgroup`) per kernel.
        -   Maximize usage of `half` precision.
        -   Leverage SIMD-group operations and `simdgroup_matrix` where applicable.
        -   Reduce register spilling.
-   **Alternative Options/Considerations:**
    -   Manual loop unrolling in MSL if the compiler doesn't optimize critical loops sufficiently.
    -   Exploring different data layouts for intermediate tensors if it improves access patterns.

### Task 4.3: Memory Optimization
-   **Core Plan:**
    -   Minimize `MTLBuffer` allocations; reuse buffers where possible (aliasing).
    -   Ensure timely deallocation of unused resources.
    -   If training hits memory limits:
        -   Gradient accumulation (already mentioned).
        -   Activation recomputation (checkpointing): Instead of storing all activations for backward pass, recompute some on the fly. Trades compute for memory.
-   **Alternative Options/Considerations:**
    -   **Model Parallelism:** If the model is too large even for 36GB (e.g., for extremely large future versions), consider model parallelism (splitting layers or tensors across multiple GPUs, not applicable for single M3 Max but a general technique). This is highly complex.
    -   **Quantization:** Post-training or quantization-aware training to use even lower precision (e.g., `int8`) for weights and/or activations. Advanced optimization for inference, adds significant complexity.

## General Project Considerations
-   **Host Language:** C++ or Swift. Swift offers tighter integration with Apple frameworks, C++ offers wider portability and existing ML library ecosystem (though we're building MSL from scratch). Let's assume **C++** for initial planning unless specified otherwise.
-   **Build System:** CMake or Swift Package Manager.
-   **Version Control:** Git.
-   **Testing Strategy (Test-Driven Development - TDD):**
    -   **Core Principle:** Before writing functional code for a component (e.g., an MSL kernel, a data processing step, a host-side logic unit), tests that define and verify its desired behavior will be written first.
    -   **Unit Tests:** Each MSL kernel will have unit tests. For kernels, this involves:
        -   Defining small, specific input data (e.g., small matrices, specific token ID sequences).
        -   Pre-calculating expected output data, often using a trusted CPU-based implementation (e.g., Python with NumPy/PyTorch, or a simple C++ equivalent).
        -   Running the MSL kernel with the test input.
        -   Comparing the kernel's output `MTLBuffer` content against the expected output, considering floating-point precision (e.g., allowing for small epsilon differences for `half` and `float`).
    -   **Host Code Unit Tests:** CPU-side logic (data loading, tokenization, batching, Metal object setup, training loop orchestration logic) will also have unit tests using a C++ testing framework (e.g., Google Test).
    -   **Integration Tests:** After individual components are unit-tested, integration tests will verify their interaction (e.g., data flow through a sequence of MSL kernels representing a Transformer block).
    -   **End-to-End Tests (Simplified):** Small-scale tests running a tiny version of the model through a few training steps or inference examples to ensure the overall pipeline is connected.
    -   **Documentation:** Test cases will also serve as a form of documentation, illustrating how components are intended to be used and their expected behavior.
-   **Testing:** Unit tests for individual MSL kernels (e.g., by comparing output with CPU-based reference implementations for small inputs) and integration tests for the full model.
-   **Documentation:** Consistent code comments (MSL, C++/Swift) and this plan document. 