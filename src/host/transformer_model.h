#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#include <vector>
#include <memory>
#include <string>
#include <atomic>

// Model configuration structure
struct TransformerConfig {
    // Model architecture
    uint32_t vocab_size = 1776;        // From our BPE tokenizer
    uint32_t embedding_dim = 512;      // Model dimension
    uint32_t num_layers = 6;           // Number of transformer blocks
    uint32_t num_heads = 8;            // Number of attention heads
    uint32_t ffn_hidden_dim = 2048;    // FFN intermediate dimension (4x embedding_dim)
    uint32_t max_sequence_length = 512; // Maximum sequence length
    
    // Training parameters
    uint32_t batch_size = 32;
    float learning_rate = 1e-4f;
    float epsilon = 1e-5f;             // Layer norm epsilon
    float dropout_rate = 0.1f;         // Dropout probability (for future implementation)
    
    // Optimizer parameters (AdamW)
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    float weight_decay = 0.01f;
    
    // Training settings
    uint32_t pad_token_id = 0;         // Padding token ID for loss masking
    bool gradient_clipping = true;
    float max_grad_norm = 1.0f;
    
    // Precision settings
    bool use_half_precision = true;    // Use fp16 for weights/activations
    bool float_logits = true;          // Keep logits in fp32 for stability
};

// Model weights structure for a single transformer block
struct TransformerBlockWeights {
    // Multi-Head Self-Attention weights
    id<MTLBuffer> qkv_weights;         // Combined Q,K,V projection weights
    id<MTLBuffer> qkv_bias;            // Combined Q,K,V projection bias
    id<MTLBuffer> attention_output_weights; // Attention output projection
    id<MTLBuffer> attention_output_bias;    // Attention output bias
    
    // Layer normalization 1 (before attention)
    id<MTLBuffer> ln1_gamma;           // Scale parameters
    id<MTLBuffer> ln1_beta;            // Shift parameters
    
    // Feed-Forward Network weights
    id<MTLBuffer> ffn_w1;              // First linear layer weights
    id<MTLBuffer> ffn_b1;              // First linear layer bias
    id<MTLBuffer> ffn_w2;              // Second linear layer weights
    id<MTLBuffer> ffn_b2;              // Second linear layer bias
    
    // Layer normalization 2 (before FFN)
    id<MTLBuffer> ln2_gamma;           // Scale parameters
    id<MTLBuffer> ln2_beta;            // Shift parameters
};

// Complete model weights
struct ModelWeights {
    // Embedding layers
    id<MTLBuffer> token_embeddings;    // Token embedding table
    id<MTLBuffer> positional_encodings; // Pre-computed positional encodings
    
    // Transformer blocks
    std::vector<TransformerBlockWeights> blocks;
    
    // Final layer normalization
    id<MTLBuffer> final_ln_gamma;
    id<MTLBuffer> final_ln_beta;
    
    // Output projection (can be tied to token_embeddings)
    id<MTLBuffer> output_weights;      // Vocabulary projection weights
    id<MTLBuffer> output_bias;         // Vocabulary projection bias
};

// Gradient buffers for training (mirror ModelWeights structure)
struct GradientBuffers {
    // Embedding layer gradients
    id<MTLBuffer> token_embeddings_grad;
    // Note: positional encodings are not learned (sinusoidal), so no gradients
    
    // Transformer block gradients
    std::vector<TransformerBlockWeights> blocks_grad; // Reuse same structure for gradients
    
    // Final layer norm gradients
    id<MTLBuffer> final_ln_gamma_grad;
    id<MTLBuffer> final_ln_beta_grad;
    
    // Output projection gradients
    id<MTLBuffer> output_weights_grad;
    id<MTLBuffer> output_bias_grad;
};

// Optimizer state buffers (AdamW)
struct OptimizerState {
    // First moment (momentum) buffers
    std::vector<id<MTLBuffer>> m_buffers;
    
    // Second moment (variance) buffers  
    std::vector<id<MTLBuffer>> v_buffers;
    
    // Current timestep
    uint32_t timestep;
    
    // Learning rate schedule state
    float current_learning_rate;
};

// MSL compute pipeline states for all kernels
struct MSLKernels {
    // Forward pass kernels
    id<MTLComputePipelineState> embedding_lookup;
    id<MTLComputePipelineState> positional_encoding;
    id<MTLComputePipelineState> qkv_projection;
    id<MTLComputePipelineState> scaled_dot_product_attention;
    id<MTLComputePipelineState> mhsa_output_projection;
    id<MTLComputePipelineState> layer_norm;
    id<MTLComputePipelineState> feed_forward_network;
    id<MTLComputePipelineState> output_logits_projection;
    id<MTLComputePipelineState> softmax;
    
    // Training kernels
    id<MTLComputePipelineState> cross_entropy_loss;
    id<MTLComputePipelineState> loss_gradient;
    id<MTLComputePipelineState> adamw_optimizer;
    id<MTLComputePipelineState> gradient_clipping;
    id<MTLComputePipelineState> zero_gradients;
    
    // Backward pass kernels
    id<MTLComputePipelineState> output_projection_backward;
    id<MTLComputePipelineState> layer_norm_backward;
    id<MTLComputePipelineState> ffn_backward;
    id<MTLComputePipelineState> mhsa_output_projection_backward;
    id<MTLComputePipelineState> scaled_dot_product_attention_backward;
    id<MTLComputePipelineState> attention_backward;
    id<MTLComputePipelineState> qkv_projection_backward;
    id<MTLComputePipelineState> embedding_layer_backward;
    
    // Utility kernels for data format conversion
    id<MTLComputePipelineState> extract_qkv_from_concatenated;
    id<MTLComputePipelineState> scaled_dot_product_attention_with_weights_save;

    // Inference-specific kernels
    id<MTLComputePipelineState> qkv_projection_inference;
    id<MTLComputePipelineState> scaled_dot_product_attention_inference;
};

// Working buffers for intermediate computations
struct WorkingBuffers {
    // Input/output buffers
    id<MTLBuffer> input_tokens;        // Token IDs
    id<MTLBuffer> target_tokens;       // Target token IDs for training
    id<MTLBuffer> embeddings;          // Token embeddings + positional encoding
    id<MTLBuffer> final_logits;        // Output logits
    
    // Loss and gradients
    id<MTLBuffer> loss_buffer;         // Computed loss value
    id<MTLBuffer> logits_grad;         // Gradient w.r.t. logits (start of backprop)
    
    // Per-layer intermediate buffers
    std::vector<id<MTLBuffer>> layer_inputs;     // Input to each transformer block
    std::vector<id<MTLBuffer>> attention_qkv;
    std::vector<id<MTLBuffer>> attention_output;
    std::vector<id<MTLBuffer>> mhsa_projection_outputs_saved;
    std::vector<id<MTLBuffer>> attention_normed;
    std::vector<id<MTLBuffer>> ffn_output;
    std::vector<id<MTLBuffer>> block_output;    // Final block output
    
    // FFN intermediate buffers (for backward pass)
    std::vector<id<MTLBuffer>> ffn_h_linear;    // Linear layer output before GELU
    std::vector<id<MTLBuffer>> ffn_h_activated; // GELU output (activated values)
    
    // Buffers for LayerNorm forward pass (to be used in backward pass)
    std::vector<id<MTLBuffer>> ln_mean;         // Mean per instance for each LN
    std::vector<id<MTLBuffer>> ln_rsqrt_variance; // 1/sqrt(var+eps) per instance for each LN
    id<MTLBuffer> final_ln_mean;                // For the final LN after all blocks
    id<MTLBuffer> final_ln_rsqrt_variance;      // For the final LN after all blocks

    // Gradient buffers for activations (for backward pass)
    std::vector<id<MTLBuffer>> layer_inputs_grad;
    std::vector<id<MTLBuffer>> attention_qkv_grad;
    std::vector<id<MTLBuffer>> attention_output_grad;
    std::vector<id<MTLBuffer>> attention_normed_grad;
    std::vector<id<MTLBuffer>> ffn_output_grad;
    std::vector<id<MTLBuffer>> block_output_grad;
    id<MTLBuffer> final_hidden_grad; // Gradient w.r.t. input of final output projection
    
    // Attention intermediate buffers
    std::vector<id<MTLBuffer>> attention_scores; // Attention weights (for analysis)
    
    // Separate Q, K, V buffers for backward pass (extracted from concatenated QKV)
    std::vector<id<MTLBuffer>> attention_Q;     // [B, S, H, D] Query tensors
    std::vector<id<MTLBuffer>> attention_K;     // [B, S, H, D] Key tensors  
    std::vector<id<MTLBuffer>> attention_V;     // [B, S, H, D] Value tensors
    
    // Attention weights for backward pass (saved from forward pass)
    std::vector<id<MTLBuffer>> attention_weights; // [B, H, S, S] Attention weights from softmax

    // KV Cache for inference
    std::vector<id<MTLBuffer>> kv_cache_K;        // Per-layer K cache for generation
    std::vector<id<MTLBuffer>> kv_cache_V;        // Per-layer V cache for generation
};

class TransformerModel {
protected:
    // Core Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // Configuration and state
    TransformerConfig config;
    
    // MSL kernels and data structures
    MSLKernels kernels;
    ModelWeights weights;
    GradientBuffers gradients;
    OptimizerState optimizer_state;
    WorkingBuffers buffers;
    
    // Helper methods
    bool initializeMetal();
    bool loadMSLKernels();
    bool allocateModelWeights();
    bool allocateGradientBuffers();
    bool allocateOptimizerState();
    bool allocateWorkingBuffers();
    void initializeWeights();
    void initializeOptimizerState();
    
    // Training helper methods
    bool zeroGradients();
    bool computeLoss(const std::vector<uint32_t>& input_tokens,
                    const std::vector<uint32_t>& target_tokens,
                    float& loss_value);
    bool backwardPass(size_t current_sequence_index = 0);
    bool optimizerStep();
    float calculateLearningRate(uint32_t step);
    
    // Conversion utilities
    uint16_t floatToHalf(float f);
    float halfToFloat(uint16_t h);

    // Helper for inference sampling
    uint32_t sampleGreedy(const float* logits, uint32_t vocab_size);

    // Inference helper methods
    bool clearKVCache();
    bool populateKVCacheWithPrompt(const std::vector<uint32_t>& prompt_tokens);
    bool generateNextToken(uint32_t current_token, uint32_t& next_token, float temperature, uint32_t current_position);
    bool generateWithTrainingKernels(const std::vector<uint32_t>& prompt_tokens,
                                    uint32_t max_new_tokens,
                                    std::vector<uint32_t>& generated_sequence,
                                    float temperature);

    bool m_is_diagnostic_run; // For targeted diagnostics
    uint32_t m_current_kv_cache_pos; // For KV cache management in inference

    // Diagnostic Canary Buffers
    id<MTLBuffer> m_canary_buffer_before;
    id<MTLBuffer> m_canary_buffer_after;

    std::atomic<bool> m_critical_gpu_error_occurred; // For command buffer error detection

private:
    // Metal objects
    // ... existing code ...

public:
    TransformerModel(const TransformerConfig& config);
    ~TransformerModel();
    
    // Model lifecycle
    bool initialize();
    void cleanup();
    
    // Forward pass
    bool forward(const std::vector<uint32_t>& input_tokens, 
                std::vector<float>& output_logits);
    
    // Training interface
    bool trainStep(const std::vector<uint32_t>& input_tokens,
                  const std::vector<uint32_t>& target_tokens,
                  float& loss);
    
    bool trainBatch(const std::vector<std::vector<uint32_t>>& input_batch,
                   const std::vector<std::vector<uint32_t>>& target_batch,
                   float& avg_loss);
    
    // Evaluation (forward pass only, no gradients)
    bool evaluate(const std::vector<uint32_t>& input_tokens,
                 const std::vector<uint32_t>& target_tokens,
                 float& loss);
    
    // Inference utilities
    bool generateNext(const std::vector<uint32_t>& context,
                     uint32_t& next_token,
                     float temperature = 1.0f);
    
    std::string generateText(const std::string& prompt,
                           size_t max_tokens = 100,
                           float temperature = 1.0f);
    
    // Model management
    bool saveWeights(const std::string& filepath);
    bool loadWeights(const std::string& filepath);
    bool saveCheckpoint(const std::string& filepath, uint32_t epoch, uint32_t step);
    bool loadCheckpoint(const std::string& filepath, uint32_t& epoch, uint32_t& step);
    
    // Getters
    const TransformerConfig& getConfig() const { return config; }
    size_t getParameterCount() const;
    size_t getMemoryUsage() const;
    float getCurrentLearningRate() const { return optimizer_state.current_learning_rate; }
    uint32_t getOptimizerTimestep() const { return optimizer_state.timestep; }

    // Inference method
    bool generate(const std::vector<uint32_t>& prompt_tokens,
                  uint32_t max_new_tokens,
                  std::vector<uint32_t>& generated_sequence,
                  float temperature = 1.0f, // For future sampling strategies
                  uint32_t top_k = 0         // For future sampling strategies
                  );
};

#endif // TRANSFORMER_MODEL_H 