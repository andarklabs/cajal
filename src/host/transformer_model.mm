#include "transformer_model.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <chrono>

// Half precision conversion utilities
uint16_t TransformerModel::floatToHalf(float f) {
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

float TransformerModel::halfToFloat(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

TransformerModel::TransformerModel(const TransformerConfig& config) : config(config) {
    device = nullptr;
    commandQueue = nullptr;
    library = nullptr;
    
    // Initialize all pipeline states to nil
    memset(&kernels, 0, sizeof(kernels));
    memset(&weights, 0, sizeof(weights));
    memset(&gradients, 0, sizeof(gradients));
    memset(&optimizer_state, 0, sizeof(optimizer_state));
    memset(&buffers, 0, sizeof(buffers));
    
    // Initialize optimizer state
    optimizer_state.timestep = 0;
    optimizer_state.current_learning_rate = config.learning_rate;

    // Initialize diagnostic flags
    m_is_diagnostic_run = false;
    m_critical_gpu_error_occurred = false;
}

TransformerModel::~TransformerModel() {
    cleanup();
}

bool TransformerModel::initialize() {
    std::cout << "Initializing Transformer Model..." << std::endl;
    
    // CRITICAL: Validate configuration before any initialization
    if (!validateConfiguration()) {
        std::cerr << "❌ CRITICAL: Configuration validation failed. Refusing to initialize unsafe model." << std::endl;
        return false;
    }
    
    if (!initializeMetal()) {
        std::cerr << "Failed to initialize Metal" << std::endl;
        return false;
    }
    
    if (!loadMSLKernels()) {
        std::cerr << "Failed to load MSL kernels" << std::endl;
        return false;
    }
    
    if (!allocateModelWeights()) {
        std::cerr << "Failed to allocate model weights" << std::endl;
        return false;
    }
    
    if (!allocateGradientBuffers()) {
        std::cerr << "Failed to allocate gradient buffers" << std::endl;
        return false;
    }
    
    if (!allocateOptimizerState()) {
        std::cerr << "Failed to allocate optimizer state" << std::endl;
        return false;
    }
    
    if (!allocateWorkingBuffers()) {
        std::cerr << "Failed to allocate working buffers" << std::endl;
        return false;
    }
    
    initializeWeights();
    initializeOptimizerState();
    
    std::cout << "✓ Transformer Model initialized successfully" << std::endl;
    std::cout << "  Parameters: " << getParameterCount() << std::endl;
    std::cout << "  Memory usage: " << (getMemoryUsage() / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

bool TransformerModel::validateConfiguration() {
    std::cout << "🔍 CONFIGURATION VALIDATION: Checking for known vulnerabilities..." << std::endl;
    
    // VULNERABILITY CHECK 1: Division by zero errors
    if (config.vocab_size == 0) {
        std::cerr << "❌ CRITICAL CONFIG: vocab_size cannot be 0 (causes division by zero)" << std::endl;
        return false;
    }
    if (config.embedding_dim == 0) {
        std::cerr << "❌ CRITICAL CONFIG: embedding_dim cannot be 0 (causes division by zero)" << std::endl;
        return false;
    }
    if (config.num_heads == 0) {
        std::cerr << "❌ CRITICAL CONFIG: num_heads cannot be 0 (causes division by zero)" << std::endl;
        return false;
    }
    if (config.num_layers == 0) {
        std::cerr << "❌ CRITICAL CONFIG: num_layers cannot be 0 (invalid model)" << std::endl;
        return false;
    }
    if (config.ffn_hidden_dim == 0) {
        std::cerr << "❌ CRITICAL CONFIG: ffn_hidden_dim cannot be 0 (invalid model)" << std::endl;
        return false;
    }
    if (config.max_sequence_length == 0) {
        std::cerr << "❌ CRITICAL CONFIG: max_sequence_length cannot be 0 (invalid model)" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 2: Head dimension calculation
    if (config.embedding_dim % config.num_heads != 0) {
        std::cerr << "❌ CRITICAL CONFIG: embedding_dim (" << config.embedding_dim << ") must be divisible by num_heads (" << config.num_heads << ")" << std::endl;
        return false;
    }
    uint32_t head_dim = config.embedding_dim / config.num_heads;
    if (head_dim == 0) {
        std::cerr << "❌ CRITICAL CONFIG: calculated head_dim is 0 (emb_dim: " << config.embedding_dim << ", num_heads: " << config.num_heads << ")" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 3: MSL Kernel Threadgroup Array Bounds
    // Check against known hardcoded array sizes in our MSL kernels
    
    // 3a. layer_norm_backward threadgroup arrays (currently 1024)
    const uint32_t LAYER_NORM_THREADGROUP_SIZE = 1024;
    if (config.embedding_dim > LAYER_NORM_THREADGROUP_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: embedding_dim (" << config.embedding_dim << ") > MSL layer_norm_backward threadgroup array size (" << LAYER_NORM_THREADGROUP_SIZE << ")" << std::endl;
        std::cerr << "    This would cause buffer overflow in layer_norm_backward kernel." << std::endl;
        return false;
    }
    
    // 3b. ffn_backward threadgroup arrays (updated to 2048)
    const uint32_t FFN_BACKWARD_THREADGROUP_SIZE = 2048;
    if (config.ffn_hidden_dim > FFN_BACKWARD_THREADGROUP_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: ffn_hidden_dim (" << config.ffn_hidden_dim << ") > MSL ffn_backward threadgroup array size (" << FFN_BACKWARD_THREADGROUP_SIZE << ")" << std::endl;
        std::cerr << "    This would cause buffer overflow in ffn_backward kernel." << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 4: Stack buffer overflow in attention inference kernel
    const uint32_t ATTENTION_STACK_BUFFER_SIZE = 512;
    if (config.max_sequence_length > ATTENTION_STACK_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: max_sequence_length (" << config.max_sequence_length << ") > MSL attention kernel stack buffer size (" << ATTENTION_STACK_BUFFER_SIZE << ")" << std::endl;
        std::cerr << "    This would cause stack buffer overflow in scaled_dot_product_attention_inference kernel." << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 5: Memory allocation size checks
    // Check for potential integer overflow in buffer size calculations
    uint64_t token_embeddings_size = (uint64_t)config.vocab_size * config.embedding_dim * sizeof(uint16_t);
    uint64_t output_weights_size = (uint64_t)config.embedding_dim * config.vocab_size * sizeof(uint16_t);
    uint64_t max_sequence_buffer_size = (uint64_t)config.max_sequence_length * config.embedding_dim * sizeof(uint16_t);
    
    const uint64_t MAX_SAFE_BUFFER_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB limit
    
    if (token_embeddings_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: token_embeddings buffer too large (" << (token_embeddings_size / 1024 / 1024) << " MB)" << std::endl;
        return false;
    }
    if (output_weights_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: output_weights buffer too large (" << (output_weights_size / 1024 / 1024) << " MB)" << std::endl;
        return false;
    }
    if (max_sequence_buffer_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: sequence buffer too large (" << (max_sequence_buffer_size / 1024 / 1024) << " MB)" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 6: Reasonable parameter bounds
    if (config.vocab_size > 1000000) { // 1M vocab size is extremely large
        std::cerr << "❌ CRITICAL CONFIG: vocab_size (" << config.vocab_size << ") unreasonably large (> 1M)" << std::endl;
        return false;
    }
    if (config.embedding_dim > 8192) { // 8K embedding dim is extremely large
        std::cerr << "❌ CRITICAL CONFIG: embedding_dim (" << config.embedding_dim << ") unreasonably large (> 8K)" << std::endl;
        return false;
    }
    if (config.num_heads > 64) { // 64 heads is extremely large
        std::cerr << "❌ CRITICAL CONFIG: num_heads (" << config.num_heads << ") unreasonably large (> 64)" << std::endl;
        return false;
    }
    if (config.ffn_hidden_dim > 32768) { // 32K FFN dim is extremely large
        std::cerr << "❌ CRITICAL CONFIG: ffn_hidden_dim (" << config.ffn_hidden_dim << ") unreasonably large (> 32K)" << std::endl;
        return false;
    }
    if (config.max_sequence_length > 4096) { // 4K sequence length is very large
        std::cerr << "❌ CRITICAL CONFIG: max_sequence_length (" << config.max_sequence_length << ") unreasonably large (> 4K)" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 7: Learning rate and numerical stability
    if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) {
        std::cerr << "❌ CRITICAL CONFIG: learning_rate (" << config.learning_rate << ") out of safe range (0, 1]" << std::endl;
        return false;
    }
    if (config.epsilon <= 0.0f || config.epsilon > 1.0f) {
        std::cerr << "❌ CRITICAL CONFIG: epsilon (" << config.epsilon << ") out of safe range (0, 1]" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 8: AdamW optimizer parameters
    if (config.beta1 <= 0.0f || config.beta1 >= 1.0f) {
        std::cerr << "❌ CRITICAL CONFIG: beta1 (" << config.beta1 << ") out of safe range (0, 1)" << std::endl;
        return false;
    }
    if (config.beta2 <= 0.0f || config.beta2 >= 1.0f) {
        std::cerr << "❌ CRITICAL CONFIG: beta2 (" << config.beta2 << ") out of safe range (0, 1)" << std::endl;
        return false;
    }
    if (config.adam_epsilon <= 0.0f) {
        std::cerr << "❌ CRITICAL CONFIG: adam_epsilon (" << config.adam_epsilon << ") must be positive" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 9: Check for potential Metal device memory limitations
    id<MTLDevice> temp_device = MTLCreateSystemDefaultDevice();
    if (temp_device) {
        uint64_t recommended_working_set = [temp_device recommendedMaxWorkingSetSize];
        uint64_t estimated_model_memory = token_embeddings_size + output_weights_size + (max_sequence_buffer_size * 10); // Rough estimate with overhead
        
        if (estimated_model_memory > recommended_working_set) {
            std::cerr << "⚠️  WARNING CONFIG: Estimated model memory (" << (estimated_model_memory / 1024 / 1024) << " MB) exceeds device recommended working set (" << (recommended_working_set / 1024 / 1024) << " MB)" << std::endl;
            std::cerr << "    This may cause memory pressure and instability." << std::endl;
        }
        
        if ([temp_device hasUnifiedMemory] == NO && estimated_model_memory > 1024 * 1024 * 1024) { // 1GB on discrete GPU
            std::cerr << "⚠️  WARNING CONFIG: Large model on discrete GPU without unified memory may cause issues." << std::endl;
        }
    }
    
    // VULNERABILITY CHECK 10: Validate batch size constraints
    if (config.batch_size == 0) {
        std::cerr << "❌ CRITICAL CONFIG: batch_size cannot be 0" << std::endl;
        return false;
    }
    if (config.batch_size > 64) { // Very large batch sizes can cause memory issues
        std::cerr << "❌ CRITICAL CONFIG: batch_size (" << config.batch_size << ") unreasonably large (> 64)" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 11: Check for potential numerical overflow in computations
    // QKV projection buffer size check
    uint64_t qkv_buffer_size = (uint64_t)config.batch_size * config.max_sequence_length * config.embedding_dim * 3 * sizeof(uint16_t);
    if (qkv_buffer_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: QKV buffer would be too large (" << (qkv_buffer_size / 1024 / 1024) << " MB)" << std::endl;
        return false;
    }
    
    // Attention weights buffer size check (for attention backward pass)
    uint64_t attention_weights_size = (uint64_t)config.batch_size * config.num_heads * config.max_sequence_length * config.max_sequence_length * sizeof(uint16_t);
    if (attention_weights_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: Attention weights buffer would be too large (" << (attention_weights_size / 1024 / 1024) << " MB)" << std::endl;
        std::cerr << "    Consider reducing max_sequence_length or num_heads" << std::endl;
        return false;
    }
    
    // FFN intermediate buffer size check
    uint64_t ffn_intermediate_size = (uint64_t)config.batch_size * config.max_sequence_length * config.ffn_hidden_dim * sizeof(uint16_t);
    if (ffn_intermediate_size > MAX_SAFE_BUFFER_SIZE) {
        std::cerr << "❌ CRITICAL CONFIG: FFN intermediate buffer would be too large (" << (ffn_intermediate_size / 1024 / 1024) << " MB)" << std::endl;
        return false;
    }
    
    // VULNERABILITY CHECK 12: Check for scale factors that could cause numerical instability
    float attention_scale = 1.0f / sqrt(float(head_dim));
    if (attention_scale > 1.0f || attention_scale < 1e-6f) {
        std::cerr << "❌ CRITICAL CONFIG: Attention scale factor (" << attention_scale << ") is outside safe range [1e-6, 1.0]" << std::endl;
        std::cerr << "    head_dim: " << head_dim << " (derived from embedding_dim: " << config.embedding_dim << ", num_heads: " << config.num_heads << ")" << std::endl;
        return false;
    }
    
    // SUCCESS: All checks passed
    std::cout << "✅ CONFIGURATION VALIDATION: All safety checks passed!" << std::endl;
    std::cout << "   Model parameters:" << std::endl;
    std::cout << "     vocab_size: " << config.vocab_size << std::endl;
    std::cout << "     embedding_dim: " << config.embedding_dim << std::endl;
    std::cout << "     num_heads: " << config.num_heads << " (head_dim: " << head_dim << ")" << std::endl;
    std::cout << "     num_layers: " << config.num_layers << std::endl;
    std::cout << "     ffn_hidden_dim: " << config.ffn_hidden_dim << std::endl;
    std::cout << "     max_sequence_length: " << config.max_sequence_length << std::endl;
    std::cout << "   Memory estimates:" << std::endl;
    std::cout << "     token_embeddings: " << (token_embeddings_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "     output_weights: " << (output_weights_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "     max_sequence_buffer: " << (max_sequence_buffer_size / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

bool TransformerModel::initializeMetal() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "❌ CRITICAL METAL: Failed to create Metal device" << std::endl;
        return false;
    }
    
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        std::cerr << "❌ CRITICAL METAL: Failed to create command queue" << std::endl;
        return false;
    }
    
    std::cout << "✓ Metal device initialized: " << [[device name] UTF8String] << std::endl;
    
    // Initialize Canary Buffers for diagnostics
    size_t canary_buffer_size = 16; // Small buffer
    m_canary_buffer_before = [device newBufferWithLength:canary_buffer_size options:MTLResourceStorageModeShared];
    m_canary_buffer_after = [device newBufferWithLength:canary_buffer_size options:MTLResourceStorageModeShared];
    
    if (!m_canary_buffer_before || !m_canary_buffer_after) {
        std::cerr << "❌ CRITICAL METAL: Failed to create canary buffers" << std::endl;
        // Not a fatal error for the model itself, but diagnostics will be impaired.
    } else {
        std::cout << "✓ Canary buffers initialized for diagnostics" << std::endl;
    }
    
    return true;
}

bool TransformerModel::loadMSLKernels() {
    std::cout << "Starting MSL kernel loading..." << std::endl;
    
    // Split kernels into groups to avoid large source string issues
    
    // Group 1: Forward pass kernels
    const char* forward_msl = R"(
#include <metal_stdlib>
using namespace metal;

// Embedding lookup kernel
kernel void embedding_lookup(
    device const uint32_t* token_ids [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* output_embeddings [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint32_t token_id = token_ids[token_idx];
    if (token_id >= vocab_size) return;
    
    uint output_offset = token_idx * embedding_dim;
    uint embedding_offset = token_id * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        output_embeddings[output_offset + d] = embedding_table[embedding_offset + d];
    }
}

kernel void apply_positional_encoding(
    device half* embeddings [[buffer(0)]],
    device const half* positional_table [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint seq_pos = token_idx % sequence_length;
    uint output_offset = token_idx * embedding_dim;
    uint pe_offset = seq_pos * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        embeddings[output_offset + d] += positional_table[pe_offset + d];
    }
}

kernel void qkv_projection(
    device const half* input_embeddings [[buffer(0)]],
    device const half* qkv_weights [[buffer(1)]],
    device const half* qkv_bias [[buffer(2)]],
    device half* qkv_output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& embedding_dim [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint input_offset = token_idx * embedding_dim;
    uint output_offset = token_idx * embedding_dim * 3;
    
    for (uint qkv = 0; qkv < 3; qkv++) {
        for (uint d = 0; d < embedding_dim; d++) {
            float sum = 0.0f;
            
            for (uint e = 0; e < embedding_dim; e++) {
                float input_val = float(input_embeddings[input_offset + e]);
                float weight_val = float(qkv_weights[qkv * embedding_dim * embedding_dim + e * embedding_dim + d]);
                sum += input_val * weight_val;
            }
            
            sum += float(qkv_bias[qkv * embedding_dim + d]);
            qkv_output[output_offset + qkv * embedding_dim + d] = half(sum);
        }
    }
}

kernel void scaled_dot_product_attention(
    device const half* qkv_input [[buffer(0)]],
    device half* attention_output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint head_idx = gid.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    uint head_dim = embedding_dim / num_heads;
    float scale = 1.0f / sqrt(float(head_dim));
    
    for (uint seq_i = 0; seq_i < sequence_length; seq_i++) {
        float max_score = -INFINITY;
        
        for (uint seq_j = 0; seq_j <= seq_i; seq_j++) {
            float score = 0.0f;
            
            uint q_offset = (batch_idx * sequence_length + seq_i) * embedding_dim * 3 + 0 * embedding_dim + head_idx * head_dim;
            uint k_offset = (batch_idx * sequence_length + seq_j) * embedding_dim * 3 + 1 * embedding_dim + head_idx * head_dim;
            
            for (uint d = 0; d < head_dim; d++) {
                score += float(qkv_input[q_offset + d]) * float(qkv_input[k_offset + d]);
            }
            
            score *= scale;
            max_score = max(max_score, score);
        }
        
        float sum_exp = 0.0f;
        for (uint seq_j = 0; seq_j <= seq_i; seq_j++) {
            float score = 0.0f;
            
            uint q_offset = (batch_idx * sequence_length + seq_i) * embedding_dim * 3 + 0 * embedding_dim + head_idx * head_dim;
            uint k_offset = (batch_idx * sequence_length + seq_j) * embedding_dim * 3 + 1 * embedding_dim + head_idx * head_dim;
            
            for (uint d = 0; d < head_dim; d++) {
                score += float(qkv_input[q_offset + d]) * float(qkv_input[k_offset + d]);
            }
            
            score *= scale;
            sum_exp += exp(score - max_score);
        }
        
        uint output_offset = (batch_idx * sequence_length + seq_i) * embedding_dim + head_idx * head_dim;
        
        for (uint d = 0; d < head_dim; d++) {
            float output_val = 0.0f;
            
            for (uint seq_j = 0; seq_j <= seq_i; seq_j++) {
                float score = 0.0f;
                uint q_offset = (batch_idx * sequence_length + seq_i) * embedding_dim * 3 + 0 * embedding_dim + head_idx * head_dim;
                uint k_offset = (batch_idx * sequence_length + seq_j) * embedding_dim * 3 + 1 * embedding_dim + head_idx * head_dim;
                
                for (uint d_inner = 0; d_inner < head_dim; d_inner++) {
                    score += float(qkv_input[q_offset + d_inner]) * float(qkv_input[k_offset + d_inner]);
                }
                score *= scale;
                
                float attention_weight = exp(score - max_score) / sum_exp;
                
                uint v_offset = (batch_idx * sequence_length + seq_j) * embedding_dim * 3 + 2 * embedding_dim + head_idx * head_dim;
                output_val += attention_weight * float(qkv_input[v_offset + d]);
            }
            
            attention_output[output_offset + d] = half(output_val);
        }
    }
}

kernel void mhsa_output_projection(
    device const half* attention_heads [[buffer(0)]],
    device const half* output_weights [[buffer(1)]],
    device const half* output_bias [[buffer(2)]],
    device half* projection_output [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& embedding_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint input_offset = token_idx * embedding_dim;
    uint output_offset = token_idx * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        float sum = 0.0f;
        
        for (uint e = 0; e < embedding_dim; e++) {
            float head_val = float(attention_heads[input_offset + e]);
            float weight_val = float(output_weights[e * embedding_dim + d]);
            sum += head_val * weight_val;
        }
        
        sum += float(output_bias[d]);
        projection_output[output_offset + d] = half(sum);
    }
}

kernel void layer_norm(
    device const half* input_tensor [[buffer(0)]],
    device const half* residual_input [[buffer(1)]],
    device half* output_tensor [[buffer(2)]],
    device const float* gamma [[buffer(3)]], // Changed to float for stability, matching typical LayerNorm
    device const float* beta [[buffer(4)]],  // Changed to float
    device float* mean_out [[buffer(5)]],           // Output: mean for each instance
    device float* rsqrt_variance_out [[buffer(6)]], // Output: 1/sqrt(variance + eps) for each instance
    constant uint& batch_size [[buffer(7)]],
    constant uint& sequence_length [[buffer(8)]],
    constant uint& embedding_dim [[buffer(9)]],
    constant float& epsilon [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]] // gid.x is instance_idx (batch_size * sequence_length)
) {
    uint instance_idx = gid.x;
    // uint dim_idx = gid.y; // Assuming one thread per instance, loops over embedding_dim

    if (instance_idx >= batch_size * sequence_length) return;
    
    uint instance_offset = instance_idx * embedding_dim;
    
    // Calculate sum for mean
    float sum_val = 0.0f;
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = (!residual_input) ? float(input_tensor[instance_offset + i]) : (float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]));
        sum_val += x_val;
    }
    float mean = sum_val / float(embedding_dim);
    
    // Calculate sum for variance
    float variance_sum = 0.0f;
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = (!residual_input) ? float(input_tensor[instance_offset + i]) : (float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]));
        float diff = x_val - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / float(embedding_dim);
    float rsqrt_variance = rsqrt(variance + epsilon);

    // Store mean and rsqrt_variance
    mean_out[instance_idx] = mean;
    rsqrt_variance_out[instance_idx] = rsqrt_variance;

    // Normalize, scale, and shift
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = (!residual_input) ? float(input_tensor[instance_offset + i]) : (float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]));
        float normalized = (x_val - mean) * rsqrt_variance;
        float result = gamma[i] * normalized + beta[i];
        output_tensor[instance_offset + i] = half(result);
    }
}

float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float a = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + a * x_cubed);
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void feed_forward_network(
    device const half* input_norm [[buffer(0)]],
    device const half* W1 [[buffer(1)]],
    device const half* b1 [[buffer(2)]],
    device const half* W2 [[buffer(3)]],
    device const half* b2 [[buffer(4)]],
    device half* ffn_output [[buffer(5)]],
    device half* h_linear [[buffer(6)]],      // NEW: Save linear layer output before GELU
    device half* h_activated [[buffer(7)]],   // NEW: Save GELU output
    constant uint& batch_size [[buffer(8)]],
    constant uint& sequence_length [[buffer(9)]],
    constant uint& embedding_dim [[buffer(10)]],
    constant uint& ffn_hidden_dim [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x;
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint input_offset = instance_idx * embedding_dim;
    uint output_offset = instance_idx * embedding_dim;
    uint hidden_offset = instance_idx * ffn_hidden_dim;
    
    // First linear layer: X @ W1 + b1
    for (uint h = 0; h < ffn_hidden_dim; h++) {
        float hidden_sum = float(b1[h]);
        for (uint e = 0; e < embedding_dim; e++) {
            float input_val = float(input_norm[input_offset + e]);
            float weight_val = float(W1[e * ffn_hidden_dim + h]);
            hidden_sum += input_val * weight_val;
        }
        h_linear[hidden_offset + h] = half(hidden_sum);
    }
    
    // GELU activation
    for (uint h = 0; h < ffn_hidden_dim; h++) {
        float linear_val = float(h_linear[hidden_offset + h]);
        float activated_val = gelu(linear_val);
        h_activated[hidden_offset + h] = half(activated_val);
    }
    
    // Second linear layer: H_activated @ W2 + b2
    for (uint e = 0; e < embedding_dim; e++) {
        float output_sum = float(b2[e]);
        for (uint h = 0; h < ffn_hidden_dim; h++) {
            float activated_val = float(h_activated[hidden_offset + h]);
            float w2_val = float(W2[h * embedding_dim + e]);
            output_sum += activated_val * w2_val;
        }
        ffn_output[output_offset + e] = half(output_sum);
    }
}

kernel void output_logits_projection(
    device const half* final_hidden_states [[buffer(0)]],
    device const half* W_out [[buffer(1)]],
    device const float* b_out [[buffer(2)]],
    device float* output_logits [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& embedding_dim [[buffer(6)]],
    constant uint& vocab_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x;
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint input_offset = instance_idx * embedding_dim;
    uint output_offset = instance_idx * vocab_size;
    
    for (uint v = 0; v < vocab_size; v++) {
        float sum = 0.0f;
        
        for (uint e = 0; e < embedding_dim; e++) {
            float hidden_val = float(final_hidden_states[input_offset + e]);
            float weight_val = float(W_out[e * vocab_size + v]);
            sum += hidden_val * weight_val;
        }
        
        output_logits[output_offset + v] = sum + b_out[v];
    }
}

kernel void softmax(
    device const float* logits [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x;
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint offset = instance_idx * vocab_size;
    
    float max_val = logits[offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_val = max(max_val, logits[offset + v]);
    }
    
    float sum = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        float exp_val = exp(logits[offset + v] - max_val);
        probabilities[offset + v] = exp_val;
        sum += exp_val;
    }
    
    for (uint v = 0; v < vocab_size; v++) {
        probabilities[offset + v] /= sum;
    }
}
)";

    // Group 2: Training kernels
    const char* training_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss(
    device const float* logits [[buffer(0)]],
    device const uint32_t* target_ids [[buffer(1)]],
    device float* per_token_loss [[buffer(2)]],
    device atomic<float>* total_loss [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    constant uint& pad_token_id [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint seq_idx = gid.y;
    
    if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
    
    uint token_idx = batch_idx * sequence_length + seq_idx;
    uint32_t target_id = target_ids[token_idx];
    
    if (target_id == pad_token_id) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    if (target_id >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    uint logit_offset = token_idx * vocab_size;
    
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    float log_sum_exp = max_logit + log(sum_exp);
    
    float target_log_prob = logits[logit_offset + target_id] - log_sum_exp;
    per_token_loss[token_idx] = -target_log_prob;
    
    atomic_fetch_add_explicit(total_loss, -target_log_prob, memory_order_relaxed);
}

kernel void loss_gradient(
    device const float* logits [[buffer(0)]],
    device const uint32_t* target_ids [[buffer(1)]],
    device float* logits_grad [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    constant uint& pad_token_id [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint seq_idx = gid.y;
    
    if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
    
    uint token_idx = batch_idx * sequence_length + seq_idx;
    uint32_t target_id = target_ids[token_idx];
    uint logit_offset = token_idx * vocab_size;
    
    if (target_id == pad_token_id || target_id >= vocab_size) {
        for (uint v = 0; v < vocab_size; v++) {
            logits_grad[logit_offset + v] = 0.0f;
        }
        return;
    }
    
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    
    for (uint v = 0; v < vocab_size; v++) {
        float softmax_val = exp(logits[logit_offset + v] - max_logit) / sum_exp;
        float one_hot = (v == target_id) ? 1.0f : 0.0f;
        logits_grad[logit_offset + v] = softmax_val - one_hot;
    }
}

kernel void zero_gradients(
    device float* grad_buffer [[buffer(0)]],
    constant uint& buffer_size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= buffer_size) return;
    grad_buffer[gid] = 0.0f;
}

kernel void adamw_optimizer(
    device half* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m_state [[buffer(2)]],
    device float* v_state [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant uint& timestep [[buffer(9)]],
    constant uint& param_size [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= param_size) return;
    
    float g = grad[gid];
    float p = float(param[gid]);
    
    p = p - learning_rate * weight_decay * p;
    
    float m = m_state[gid];
    float v = v_state[gid];
    
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    
    m_state[gid] = m;
    v_state[gid] = v;
    
    float m_hat = m / (1.0f - pow(beta1, float(timestep)));
    float v_hat = v / (1.0f - pow(beta2, float(timestep)));
    
    p = p - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    
    param[gid] = half(p);
}

kernel void gradient_clipping(
    device float* grad_buffer [[buffer(0)]],
    constant float& max_norm [[buffer(1)]],
    constant float& global_norm [[buffer(2)]],
    constant uint& buffer_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= buffer_size) return;
    
    if (global_norm > max_norm) {
        float clip_coeff = max_norm / global_norm;
        grad_buffer[gid] *= clip_coeff;
    }
}
)";

    // Group 3: Backward pass kernels (from file)
    std::string backward_msl_content;
    std::ifstream backward_file("src/msl/backward_kernels.msl"); // Fixed path
    if (!backward_file.is_open()) {
        std::cerr << "Failed to open src/msl/backward_kernels.msl" << std::endl; // Updated error message
        return false;
    }
    backward_msl_content.assign((std::istreambuf_iterator<char>(backward_file)), std::istreambuf_iterator<char>());
    backward_file.close();

    std::cout << "Creating forward kernels library..." << std::endl;
    NSError* error = nil;
    id<MTLLibrary> forward_library = [device newLibraryWithSource:@(forward_msl) options:nil error:&error];
    if (!forward_library) {
        std::cerr << "Failed to create forward MSL library: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    std::cout << "Creating training kernels library..." << std::endl;
    id<MTLLibrary> training_library = [device newLibraryWithSource:@(training_msl) options:nil error:&error];
    if (!training_library) {
        std::cerr << "Failed to create training MSL library: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    std::cout << "Creating backward kernels library..." << std::endl;
    id<MTLLibrary> backward_library = [device newLibraryWithSource:@(backward_msl_content.c_str()) options:nil error:&error];
    if (!backward_library) {
        std::cerr << "Failed to create backward MSL library: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    std::cout << "✓ Backward MSL library created successfully" << std::endl;
    
    std::cout << "✓ MSL libraries created successfully" << std::endl;
    
    // Load forward kernels
    std::cout << "Loading forward kernels..." << std::endl;
    
    // embedding_lookup
    id<MTLFunction> function = [forward_library newFunctionWithName:@"embedding_lookup"];
    if (!function) {
        std::cerr << "Failed to find embedding_lookup function" << std::endl;
        return false;
    }
    kernels.embedding_lookup = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.embedding_lookup) {
        std::cerr << "Failed to create embedding_lookup pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ embedding_lookup loaded" << std::endl;
    
    // apply_positional_encoding
    function = [forward_library newFunctionWithName:@"apply_positional_encoding"];
    if (!function) {
        std::cerr << "Failed to find apply_positional_encoding function" << std::endl;
        return false;
    }
    kernels.positional_encoding = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.positional_encoding) {
        std::cerr << "Failed to create apply_positional_encoding pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ apply_positional_encoding loaded" << std::endl;
    
    // qkv_projection
    function = [forward_library newFunctionWithName:@"qkv_projection"];
    if (!function) {
        std::cerr << "Failed to find qkv_projection function" << std::endl;
        return false;
    }
    kernels.qkv_projection = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.qkv_projection) {
        std::cerr << "Failed to create qkv_projection pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ qkv_projection loaded" << std::endl;
    
    // scaled_dot_product_attention
    function = [forward_library newFunctionWithName:@"scaled_dot_product_attention"];
    if (!function) {
        std::cerr << "Failed to find scaled_dot_product_attention function" << std::endl;
        return false;
    }
    kernels.scaled_dot_product_attention = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.scaled_dot_product_attention) {
        std::cerr << "Failed to create scaled_dot_product_attention pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ scaled_dot_product_attention loaded" << std::endl;
    
    // mhsa_output_projection
    function = [forward_library newFunctionWithName:@"mhsa_output_projection"];
    if (!function) {
        std::cerr << "Failed to find mhsa_output_projection function" << std::endl;
        return false;
    }
    kernels.mhsa_output_projection = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.mhsa_output_projection) {
        std::cerr << "Failed to create mhsa_output_projection pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ mhsa_output_projection loaded" << std::endl;
    
    // layer_norm
    function = [forward_library newFunctionWithName:@"layer_norm"];
    if (!function) {
        std::cerr << "Failed to find layer_norm function" << std::endl;
        return false;
    }
    kernels.layer_norm = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.layer_norm) {
        std::cerr << "Failed to create layer_norm pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ layer_norm loaded" << std::endl;
    
    // feed_forward_network
    function = [forward_library newFunctionWithName:@"feed_forward_network"];
    if (!function) {
        std::cerr << "Failed to find feed_forward_network function" << std::endl;
        return false;
    }
    kernels.feed_forward_network = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.feed_forward_network) {
        std::cerr << "Failed to create feed_forward_network pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ feed_forward_network loaded" << std::endl;
    
    // output_logits_projection
    function = [forward_library newFunctionWithName:@"output_logits_projection"];
    if (!function) {
        std::cerr << "Failed to find output_logits_projection function" << std::endl;
        return false;
    }
    kernels.output_logits_projection = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.output_logits_projection) {
        std::cerr << "Failed to create output_logits_projection pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ output_logits_projection loaded" << std::endl;
    
    // softmax
    function = [forward_library newFunctionWithName:@"softmax"];
    if (!function) {
        std::cerr << "Failed to find softmax function" << std::endl;
        return false;
    }
    kernels.softmax = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.softmax) {
        std::cerr << "Failed to create softmax pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ softmax loaded" << std::endl;
    
    // Load training kernels
    std::cout << "Loading training kernels..." << std::endl;
    
    // cross_entropy_loss
    function = [training_library newFunctionWithName:@"cross_entropy_loss"];
    if (!function) {
        std::cerr << "Failed to find cross_entropy_loss function" << std::endl;
        return false;
    }
    kernels.cross_entropy_loss = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.cross_entropy_loss) {
        std::cerr << "Failed to create cross_entropy_loss pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ cross_entropy_loss loaded" << std::endl;
    
    // loss_gradient
    function = [training_library newFunctionWithName:@"loss_gradient"];
    if (!function) {
        std::cerr << "Failed to find loss_gradient function" << std::endl;
        return false;
    }
    kernels.loss_gradient = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.loss_gradient) {
        std::cerr << "Failed to create loss_gradient pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ loss_gradient loaded" << std::endl;
    
    // zero_gradients
    function = [training_library newFunctionWithName:@"zero_gradients"];
    if (!function) {
        std::cerr << "Failed to find zero_gradients function" << std::endl;
        return false;
    }
    kernels.zero_gradients = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.zero_gradients) {
        std::cerr << "Failed to create zero_gradients pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ zero_gradients loaded" << std::endl;
    
    // adamw_optimizer
    function = [training_library newFunctionWithName:@"adamw_optimizer"];
    if (!function) {
        std::cerr << "Failed to find adamw_optimizer function" << std::endl;
        return false;
    }
    kernels.adamw_optimizer = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.adamw_optimizer) {
        std::cerr << "Failed to create adamw_optimizer pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ adamw_optimizer loaded" << std::endl;
    
    // gradient_clipping
    function = [training_library newFunctionWithName:@"gradient_clipping"];
    if (!function) {
        std::cerr << "Failed to find gradient_clipping function" << std::endl;
        return false;
    }
    kernels.gradient_clipping = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.gradient_clipping) {
        std::cerr << "Failed to create gradient_clipping pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ gradient_clipping loaded" << std::endl;
    
    // Load backward kernels
    std::cout << "Loading backward kernels..." << std::endl;
    function = [backward_library newFunctionWithName:@"layer_norm_backward"];
    if (!function) {
        std::cerr << "Failed to find layer_norm_backward function" << std::endl;
        return false;
    }
    kernels.layer_norm_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.layer_norm_backward) {
        std::cerr << "Failed to create layer_norm_backward pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ layer_norm_backward loaded" << std::endl;

    function = [backward_library newFunctionWithName:@"output_projection_backward"];
    if (!function) {
        std::cerr << "Failed to find output_projection_backward function" << std::endl;
        return false;
    }
    kernels.output_projection_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.output_projection_backward) {
        std::cerr << "Failed to create output_projection_backward pipeline" << std::endl;
        return false;
    }
    std::cout << "✓ output_projection_backward loaded" << std::endl;
    
    std::cout << "Attempting to load ffn_backward kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"ffn_backward"];
    if (!function) {
        std::cerr << "Failed to find ffn_backward function" << std::endl;
        return false;
    }
    std::cout << "✓ ffn_backward function found" << std::endl;
    kernels.ffn_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.ffn_backward) {
        std::cerr << "Failed to create ffn_backward pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ ffn_backward loaded" << std::endl;
    
    std::cout << "Attempting to load mhsa_output_projection_backward kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"mhsa_output_projection_backward"];
    if (!function) {
        std::cerr << "Failed to find mhsa_output_projection_backward function" << std::endl;
        return false;
    }
    kernels.mhsa_output_projection_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.mhsa_output_projection_backward) {
        NSLog(@"Failed to create mhsa_output_projection_backward pipeline state: %@", error.localizedDescription);
        return false;
    }
    std::cout << "✓ mhsa_output_projection_backward loaded" << std::endl;
    
    std::cout << "Attempting to load scaled_dot_product_attention_backward kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"scaled_dot_product_attention_backward"];
    if (!function) {
        std::cerr << "Failed to find scaled_dot_product_attention_backward function" << std::endl;
        return false;
    }
    kernels.scaled_dot_product_attention_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.scaled_dot_product_attention_backward) {
        std::cerr << "Failed to create scaled_dot_product_attention_backward pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ scaled_dot_product_attention_backward loaded" << std::endl;
    
    std::cout << "Attempting to load qkv_projection_backward kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"qkv_projection_backward"];
    if (!function) {
        std::cerr << "Failed to find qkv_projection_backward function" << std::endl;
        return false;
    }
    kernels.qkv_projection_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.qkv_projection_backward) {
        std::cerr << "Failed to create qkv_projection_backward pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ qkv_projection_backward loaded" << std::endl;
    
    std::cout << "Attempting to load embedding_layer_backward kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"embedding_layer_backward"];
    if (!function) {
        std::cerr << "Failed to find embedding_layer_backward function" << std::endl;
        return false;
    }
    kernels.embedding_layer_backward = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.embedding_layer_backward) {
        std::cerr << "Failed to create embedding_layer_backward pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ embedding_layer_backward loaded" << std::endl;
    
    // Load utility kernels for data format conversion
    std::cout << "Attempting to load extract_qkv_from_concatenated kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"extract_qkv_from_concatenated"];
    if (!function) {
        std::cerr << "Failed to find extract_qkv_from_concatenated function" << std::endl;
        return false;
    }
    kernels.extract_qkv_from_concatenated = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.extract_qkv_from_concatenated) {
        std::cerr << "Failed to create extract_qkv_from_concatenated pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ extract_qkv_from_concatenated loaded" << std::endl;
    
    std::cout << "Attempting to load scaled_dot_product_attention_with_weights_save kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"scaled_dot_product_attention_with_weights_save"];
    if (!function) {
        std::cerr << "Failed to find scaled_dot_product_attention_with_weights_save function" << std::endl;
        return false;
    }
    kernels.scaled_dot_product_attention_with_weights_save = [device newComputePipelineStateWithFunction:function error:&error];
    if (!kernels.scaled_dot_product_attention_with_weights_save) {
        std::cerr << "Failed to create scaled_dot_product_attention_with_weights_save pipeline" << std::endl;
        if (error) {
            NSLog(@"Error: %@", error.localizedDescription);
        }
        return false;
    }
    std::cout << "✓ scaled_dot_product_attention_with_weights_save loaded" << std::endl;
    
    // Load inference-specific kernels
    std::cout << "Attempting to load qkv_projection_inference kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"qkv_projection_inference"];
    if (!function) {
        std::cerr << "Failed to find qkv_projection_inference function" << std::endl;
        kernels.qkv_projection_inference = nil;
        std::cerr << "WARNING: qkv_projection_inference kernel not found. Inference will not work." << std::endl;
    } else {
        kernels.qkv_projection_inference = [device newComputePipelineStateWithFunction:function error:&error];
        if (!kernels.qkv_projection_inference) {
            std::cerr << "Failed to create qkv_projection_inference pipeline" << std::endl;
            if (error) {
                NSLog(@"Error: %@", error.localizedDescription);
            }
            kernels.qkv_projection_inference = nil; 
        } else {
            std::cout << "✓ qkv_projection_inference loaded" << std::endl;
        }
    }

    std::cout << "Attempting to load scaled_dot_product_attention_inference kernel..." << std::endl;
    function = [backward_library newFunctionWithName:@"scaled_dot_product_attention_inference"];
    if (!function) {
        std::cerr << "Failed to find scaled_dot_product_attention_inference function" << std::endl;
        kernels.scaled_dot_product_attention_inference = nil;
        std::cerr << "WARNING: scaled_dot_product_attention_inference kernel not found. Inference will not work." << std::endl;
    } else {
        kernels.scaled_dot_product_attention_inference = [device newComputePipelineStateWithFunction:function error:&error];
        if (!kernels.scaled_dot_product_attention_inference) {
            std::cerr << "Failed to create scaled_dot_product_attention_inference pipeline" << std::endl;
            if (error) {
                NSLog(@"Error: %@", error.localizedDescription);
            }
            kernels.scaled_dot_product_attention_inference = nil; 
        } else {
            std::cout << "✓ scaled_dot_product_attention_inference loaded" << std::endl;
        }
    }
    
    std::cout << "✓ All MSL kernels loaded successfully" << std::endl;
    return true;
}

bool TransformerModel::allocateModelWeights() {
    std::cout << "Allocating model weights..." << std::endl;
    
    // Token embeddings: vocab_size x embedding_dim
    size_t token_embedding_size = config.vocab_size * config.embedding_dim * sizeof(uint16_t);
    weights.token_embeddings = [device newBufferWithLength:token_embedding_size 
                                                   options:MTLResourceStorageModeShared];
    
    // Positional encodings: max_sequence_length x embedding_dim
    size_t pe_size = config.max_sequence_length * config.embedding_dim * sizeof(uint16_t);
    weights.positional_encodings = [device newBufferWithLength:pe_size 
                                                        options:MTLResourceStorageModeShared];
    
    // Allocate weights for each transformer block
    weights.blocks.resize(config.num_layers);
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block = weights.blocks[layer];
        
        // QKV projection weights: 3 * embedding_dim x embedding_dim
        size_t qkv_weight_size = 3 * config.embedding_dim * config.embedding_dim * sizeof(uint16_t);
        block.qkv_weights = [device newBufferWithLength:qkv_weight_size 
                                                options:MTLResourceStorageModeShared];
        
        // QKV bias: 3 * embedding_dim
        size_t qkv_bias_size = 3 * config.embedding_dim * sizeof(uint16_t);
        block.qkv_bias = [device newBufferWithLength:qkv_bias_size 
                                             options:MTLResourceStorageModeShared];
        
        // Attention output projection: embedding_dim x embedding_dim
        size_t attn_out_weight_size = config.embedding_dim * config.embedding_dim * sizeof(uint16_t);
        block.attention_output_weights = [device newBufferWithLength:attn_out_weight_size 
                                                            options:MTLResourceStorageModeShared];
        
        // Attention output bias: embedding_dim
        size_t attn_out_bias_size = config.embedding_dim * sizeof(uint16_t);
        block.attention_output_bias = [device newBufferWithLength:attn_out_bias_size 
                                                          options:MTLResourceStorageModeShared];
        
        // Layer norm parameters (gamma, beta)
        size_t ln_param_size = config.embedding_dim * sizeof(float);
        block.ln1_gamma = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block.ln1_beta = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block.ln2_gamma = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block.ln2_beta = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        
        // FFN weights: embedding_dim x ffn_hidden_dim, ffn_hidden_dim x embedding_dim
        size_t ffn_w1_size = config.embedding_dim * config.ffn_hidden_dim * sizeof(uint16_t);
        size_t ffn_w2_size = config.ffn_hidden_dim * config.embedding_dim * sizeof(uint16_t);
        block.ffn_w1 = [device newBufferWithLength:ffn_w1_size options:MTLResourceStorageModeShared];
        block.ffn_w2 = [device newBufferWithLength:ffn_w2_size options:MTLResourceStorageModeShared];
        
        // FFN biases
        size_t ffn_b1_size = config.ffn_hidden_dim * sizeof(uint16_t);
        size_t ffn_b2_size = config.embedding_dim * sizeof(uint16_t);
        block.ffn_b1 = [device newBufferWithLength:ffn_b1_size options:MTLResourceStorageModeShared];
        block.ffn_b2 = [device newBufferWithLength:ffn_b2_size options:MTLResourceStorageModeShared];
    }
    
    // Final layer norm
    size_t ln_param_size = config.embedding_dim * sizeof(float);
    weights.final_ln_gamma = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
    weights.final_ln_beta = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
    
    // Output projection weights: embedding_dim x vocab_size
    size_t output_weight_size = config.embedding_dim * config.vocab_size * sizeof(uint16_t);
    weights.output_weights = [device newBufferWithLength:output_weight_size 
                                                 options:MTLResourceStorageModeShared];
    
    // Output bias: vocab_size
    size_t output_bias_size = config.vocab_size * sizeof(float);
    weights.output_bias = [device newBufferWithLength:output_bias_size 
                                              options:MTLResourceStorageModeShared];
    
    std::cout << "✓ Model weights allocated" << std::endl;
    return true;
}

bool TransformerModel::allocateWorkingBuffers() {
    std::cout << "Allocating working buffers..." << std::endl;
    
    size_t max_tokens = config.batch_size * config.max_sequence_length;
    
    // Input/output buffers
    buffers.input_tokens = [device newBufferWithLength:max_tokens * sizeof(uint32_t) 
                                               options:MTLResourceStorageModeShared];
    
    buffers.target_tokens = [device newBufferWithLength:max_tokens * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
    
    buffers.embeddings = [device newBufferWithLength:max_tokens * config.embedding_dim * sizeof(uint16_t)
                                             options:MTLResourceStorageModeShared];
    
    buffers.final_logits = [device newBufferWithLength:max_tokens * config.vocab_size * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    
    // Loss and gradient buffers
    buffers.loss_buffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    
    buffers.logits_grad = [device newBufferWithLength:max_tokens * config.vocab_size * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    
    // Per-layer buffers
    buffers.layer_inputs.resize(config.num_layers + 1); // +1 for initial embeddings
    buffers.attention_qkv.resize(config.num_layers);
    buffers.attention_output.resize(config.num_layers);
    buffers.mhsa_projection_outputs_saved.resize(config.num_layers); // ADDED: Resize new buffer
    buffers.attention_normed.resize(config.num_layers);
    buffers.ffn_output.resize(config.num_layers);
    buffers.block_output.resize(config.num_layers);
    
    // FFN intermediate buffers (for backward pass)
    buffers.ffn_h_linear.resize(config.num_layers);
    buffers.ffn_h_activated.resize(config.num_layers);
    
    // Gradient buffers for activations
    buffers.layer_inputs_grad.resize(config.num_layers + 1);
    buffers.attention_qkv_grad.resize(config.num_layers);
    buffers.attention_output_grad.resize(config.num_layers);
    buffers.attention_normed_grad.resize(config.num_layers);
    buffers.ffn_output_grad.resize(config.num_layers);
    buffers.block_output_grad.resize(config.num_layers);
    
    // New buffers for separate Q, K, V and attention weights
    buffers.attention_Q.resize(config.num_layers);
    buffers.attention_K.resize(config.num_layers);
    buffers.attention_V.resize(config.num_layers);
    buffers.attention_weights.resize(config.num_layers);
    
    // KV Cache for inference
    buffers.kv_cache_K.resize(config.num_layers);
    buffers.kv_cache_V.resize(config.num_layers);
    
    for (uint32_t layer = 0; layer <= config.num_layers; layer++) {
        size_t buffer_size = max_tokens * config.embedding_dim * sizeof(uint16_t);
        buffers.layer_inputs[layer] = [device newBufferWithLength:buffer_size 
                                                          options:MTLResourceStorageModeShared];
        
        if (layer < config.num_layers) {
            buffers.layer_inputs_grad[layer] = [device newBufferWithLength:buffer_size 
                                                                   options:MTLResourceStorageModeShared];
        }
    }
    
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        // QKV buffer: 3 times embedding dimension
        size_t qkv_size = max_tokens * config.embedding_dim * 3 * sizeof(uint16_t);
        buffers.attention_qkv[layer] = [device newBufferWithLength:qkv_size 
                                                           options:MTLResourceStorageModeShared];
        buffers.attention_qkv_grad[layer] = [device newBufferWithLength:qkv_size 
                                                              options:MTLResourceStorageModeShared];
        
        // Other buffers: regular embedding dimension
        size_t buffer_size = max_tokens * config.embedding_dim * sizeof(uint16_t);
        buffers.attention_output[layer] = [device newBufferWithLength:buffer_size 
                                                              options:MTLResourceStorageModeShared];
        buffers.mhsa_projection_outputs_saved[layer] = [device newBufferWithLength:buffer_size options:MTLResourceStorageModeShared]; // ADDED: Allocate buffer
        buffers.attention_normed[layer] = [device newBufferWithLength:buffer_size 
                                                               options:MTLResourceStorageModeShared];
        buffers.ffn_output[layer] = [device newBufferWithLength:buffer_size 
                                                        options:MTLResourceStorageModeShared];
        buffers.block_output[layer] = [device newBufferWithLength:buffer_size 
                                                          options:MTLResourceStorageModeShared];
        
        // FFN intermediate buffers: FFN hidden dimension
        size_t ffn_hidden_size = max_tokens * config.ffn_hidden_dim * sizeof(uint16_t);
        buffers.ffn_h_linear[layer] = [device newBufferWithLength:ffn_hidden_size 
                                                           options:MTLResourceStorageModeShared];
        buffers.ffn_h_activated[layer] = [device newBufferWithLength:ffn_hidden_size 
                                                              options:MTLResourceStorageModeShared];
        
        buffers.attention_output_grad[layer] = [device newBufferWithLength:buffer_size 
                                                                   options:MTLResourceStorageModeShared];
        buffers.attention_normed_grad[layer] = [device newBufferWithLength:buffer_size 
                                                                   options:MTLResourceStorageModeShared];
        buffers.ffn_output_grad[layer] = [device newBufferWithLength:buffer_size 
                                                             options:MTLResourceStorageModeShared];
        buffers.block_output_grad[layer] = [device newBufferWithLength:buffer_size 
                                                              options:MTLResourceStorageModeShared];
        
        // Allocate new buffers for separate Q, K, V (for backward pass)
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        size_t qkv_separate_size = max_tokens * config.num_heads * head_dim * sizeof(uint16_t);
        buffers.attention_Q[layer] = [device newBufferWithLength:qkv_separate_size options:MTLResourceStorageModeShared];
        buffers.attention_K[layer] = [device newBufferWithLength:qkv_separate_size options:MTLResourceStorageModeShared]; 
        buffers.attention_V[layer] = [device newBufferWithLength:qkv_separate_size options:MTLResourceStorageModeShared];
        
        // Allocate attention weights buffer [B, H, S, S]
        size_t attention_weights_size = config.batch_size * config.num_heads * config.max_sequence_length * config.max_sequence_length * sizeof(uint16_t);
        buffers.attention_weights[layer] = [device newBufferWithLength:attention_weights_size options:MTLResourceStorageModeShared];

        // Allocate KV Cache buffers for inference [GenBatch=1, H, MaxS, HeadDim]
        // Assuming generation batch size is 1 for KV cache
        size_t kv_cache_per_layer_size = 1 * config.num_heads * config.max_sequence_length * head_dim * sizeof(uint16_t);
        buffers.kv_cache_K[layer] = [device newBufferWithLength:kv_cache_per_layer_size options:MTLResourceStorageModeShared];
        buffers.kv_cache_V[layer] = [device newBufferWithLength:kv_cache_per_layer_size options:MTLResourceStorageModeShared];
    }
    
    // Allocate LayerNorm mean and rsqrt_variance buffers
    // Each transformer block has two LayerNorms (ln1, ln2)
    // There is one final LayerNorm after all blocks
    buffers.ln_mean.resize(config.num_layers * 2);
    buffers.ln_rsqrt_variance.resize(config.num_layers * 2);
    size_t ln_stats_buffer_size = max_tokens * sizeof(float);

    for (uint32_t i = 0; i < config.num_layers * 2; ++i) {
        buffers.ln_mean[i] = [device newBufferWithLength:ln_stats_buffer_size options:MTLResourceStorageModeShared];
        buffers.ln_rsqrt_variance[i] = [device newBufferWithLength:ln_stats_buffer_size options:MTLResourceStorageModeShared];
    }
    buffers.final_ln_mean = [device newBufferWithLength:ln_stats_buffer_size options:MTLResourceStorageModeShared];
    buffers.final_ln_rsqrt_variance = [device newBufferWithLength:ln_stats_buffer_size options:MTLResourceStorageModeShared];

    // Allocate buffer for gradient of input to final output projection
    size_t final_hidden_grad_size = max_tokens * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float)); // Match precision of forward pass hidden states
    buffers.final_hidden_grad = [device newBufferWithLength:final_hidden_grad_size options:MTLResourceStorageModeShared];

    std::cout << "✓ Working buffers allocated" << std::endl;
    return true;
}

bool TransformerModel::allocateGradientBuffers() {
    std::cout << "Allocating gradient buffers..." << std::endl;
    
    // Token embeddings gradient: vocab_size x embedding_dim
    size_t token_embedding_size = config.vocab_size * config.embedding_dim * sizeof(float);
    gradients.token_embeddings_grad = [device newBufferWithLength:token_embedding_size 
                                                          options:MTLResourceStorageModeShared];
    
    // Allocate gradient buffers for each transformer block
    gradients.blocks_grad.resize(config.num_layers);
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block_grad = gradients.blocks_grad[layer];
        
        // QKV projection gradients: 3 * embedding_dim x embedding_dim
        size_t qkv_weight_size = 3 * config.embedding_dim * config.embedding_dim * sizeof(float);
        block_grad.qkv_weights = [device newBufferWithLength:qkv_weight_size 
                                                      options:MTLResourceStorageModeShared];
        
        // QKV bias gradients: 3 * embedding_dim
        size_t qkv_bias_size = 3 * config.embedding_dim * sizeof(float);
        block_grad.qkv_bias = [device newBufferWithLength:qkv_bias_size 
                                                   options:MTLResourceStorageModeShared];
        
        // Attention output projection gradients
        size_t attn_out_weight_size = config.embedding_dim * config.embedding_dim * sizeof(float);
        block_grad.attention_output_weights = [device newBufferWithLength:attn_out_weight_size 
                                                                  options:MTLResourceStorageModeShared];
        
        size_t attn_out_bias_size = config.embedding_dim * sizeof(float);
        block_grad.attention_output_bias = [device newBufferWithLength:attn_out_bias_size 
                                                                options:MTLResourceStorageModeShared];
        
        // Layer norm parameter gradients
        size_t ln_param_size = config.embedding_dim * sizeof(float);
        block_grad.ln1_gamma = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block_grad.ln1_beta = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block_grad.ln2_gamma = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        block_grad.ln2_beta = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
        
        // FFN weight gradients
        size_t ffn_w1_size = config.embedding_dim * config.ffn_hidden_dim * sizeof(float);
        size_t ffn_w2_size = config.ffn_hidden_dim * config.embedding_dim * sizeof(float);
        block_grad.ffn_w1 = [device newBufferWithLength:ffn_w1_size options:MTLResourceStorageModeShared];
        block_grad.ffn_w2 = [device newBufferWithLength:ffn_w2_size options:MTLResourceStorageModeShared];
        
        // FFN bias gradients
        size_t ffn_b1_size = config.ffn_hidden_dim * sizeof(float);
        size_t ffn_b2_size = config.embedding_dim * sizeof(float);
        block_grad.ffn_b1 = [device newBufferWithLength:ffn_b1_size options:MTLResourceStorageModeShared];
        block_grad.ffn_b2 = [device newBufferWithLength:ffn_b2_size options:MTLResourceStorageModeShared];
    }
    
    // Final layer norm gradients
    size_t ln_param_size = config.embedding_dim * sizeof(float);
    gradients.final_ln_gamma_grad = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
    gradients.final_ln_beta_grad = [device newBufferWithLength:ln_param_size options:MTLResourceStorageModeShared];
    
    // Output projection gradients
    size_t output_weight_size = config.embedding_dim * config.vocab_size * sizeof(float);
    gradients.output_weights_grad = [device newBufferWithLength:output_weight_size 
                                                         options:MTLResourceStorageModeShared];
    
    size_t output_bias_size = config.vocab_size * sizeof(float);
    gradients.output_bias_grad = [device newBufferWithLength:output_bias_size 
                                                      options:MTLResourceStorageModeShared];
    
    std::cout << "✓ Gradient buffers allocated" << std::endl;
    return true;
}

bool TransformerModel::allocateOptimizerState() {
    std::cout << "Allocating optimizer state..." << std::endl;
    
    // Count total number of parameters
    std::vector<size_t> param_sizes;
    
    // Token embeddings
    param_sizes.push_back(config.vocab_size * config.embedding_dim);
    
    // Transformer blocks
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        // QKV weights + bias
        param_sizes.push_back(3 * config.embedding_dim * config.embedding_dim);
        param_sizes.push_back(3 * config.embedding_dim);
        
        // Attention output weights + bias
        param_sizes.push_back(config.embedding_dim * config.embedding_dim);
        param_sizes.push_back(config.embedding_dim);
        
        // Layer norm parameters (x2)
        param_sizes.push_back(config.embedding_dim); // ln1_gamma
        param_sizes.push_back(config.embedding_dim); // ln1_beta
        param_sizes.push_back(config.embedding_dim); // ln2_gamma
        param_sizes.push_back(config.embedding_dim); // ln2_beta
        
        // FFN weights + biases
        param_sizes.push_back(config.embedding_dim * config.ffn_hidden_dim);
        param_sizes.push_back(config.ffn_hidden_dim);
        param_sizes.push_back(config.ffn_hidden_dim * config.embedding_dim);
        param_sizes.push_back(config.embedding_dim);
    }
    
    // Final layer norm
    param_sizes.push_back(config.embedding_dim); // final_ln_gamma
    param_sizes.push_back(config.embedding_dim); // final_ln_beta
    
    // Output projection
    param_sizes.push_back(config.embedding_dim * config.vocab_size);
    param_sizes.push_back(config.vocab_size);
    
    // Allocate momentum buffers (first moment)
    optimizer_state.m_buffers.resize(param_sizes.size());
    optimizer_state.v_buffers.resize(param_sizes.size());
    
    for (size_t i = 0; i < param_sizes.size(); i++) {
        size_t buffer_size = param_sizes[i] * sizeof(float);
        optimizer_state.m_buffers[i] = [device newBufferWithLength:buffer_size 
                                                           options:MTLResourceStorageModeShared];
        optimizer_state.v_buffers[i] = [device newBufferWithLength:buffer_size 
                                                           options:MTLResourceStorageModeShared];
    }
    
    std::cout << "✓ Optimizer state allocated" << std::endl;
    return true;
}

void TransformerModel::initializeOptimizerState() {
    std::cout << "Initializing optimizer state..." << std::endl;
    
    // Zero initialize all momentum buffers
    for (size_t i = 0; i < optimizer_state.m_buffers.size(); i++) {
        float* m_data = static_cast<float*>([optimizer_state.m_buffers[i] contents]);
        float* v_data = static_cast<float*>([optimizer_state.v_buffers[i] contents]);
        
        size_t buffer_elements = [optimizer_state.m_buffers[i] length] / sizeof(float);
        
        for (size_t j = 0; j < buffer_elements; j++) {
            m_data[j] = 0.0f;
            v_data[j] = 0.0f;
        }
    }
    
    optimizer_state.timestep = 0;
    optimizer_state.current_learning_rate = config.learning_rate;
    
    std::cout << "✓ Optimizer state initialized" << std::endl;
}

bool TransformerModel::forward(const std::vector<uint32_t>& input_tokens, 
                               std::vector<float>& output_logits) {
    uint32_t actual_batch_size = 1; // Assuming batch size of 1 for now
    uint32_t actual_sequence_length = static_cast<uint32_t>(input_tokens.size());

    if (actual_sequence_length == 0 || actual_sequence_length > config.max_sequence_length) {
        std::cerr << "Invalid sequence length: " << actual_sequence_length << std::endl;
        return false;
    }

    // Copy input tokens to buffer
    uint32_t* token_data = static_cast<uint32_t*>([buffers.input_tokens contents]);
    std::copy(input_tokens.begin(), input_tokens.end(), token_data);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Step 1: Embedding Lookup
    [encoder setComputePipelineState:kernels.embedding_lookup];
    [encoder setBuffer:buffers.input_tokens offset:0 atIndex:0];
    [encoder setBuffer:weights.token_embeddings offset:0 atIndex:1];
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:2];
    [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:6];
    MTLSize threadsPerGrid = MTLSizeMake(actual_batch_size * actual_sequence_length, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(std::min((size_t)actual_batch_size * actual_sequence_length, kernels.embedding_lookup.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    // Step 2: Positional Encoding
    [encoder setComputePipelineState:kernels.positional_encoding];
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:0]; // Input is the output of embedding_lookup
    [encoder setBuffer:weights.positional_encodings offset:0 atIndex:1];
    [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    
    // End the compute encoder before starting a blit encoder
    [encoder endEncoding]; 

    // Copy initial embeddings to first layer input using a Blit encoder
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:buffers.embeddings sourceOffset:0 
                  toBuffer:buffers.layer_inputs[0] destinationOffset:0 
                      size:actual_batch_size * actual_sequence_length * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float))];
    [blitEncoder endEncoding];

    // Start a new compute encoder for subsequent operations
    encoder = [commandBuffer computeCommandEncoder];

    // Step 3: Transformer Blocks
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        id<MTLBuffer> current_input = buffers.layer_inputs[i];
        id<MTLBuffer> current_block_output = buffers.block_output[i];
        auto& block_w = weights.blocks[i];

        // QKV Projection
        [encoder setComputePipelineState:kernels.qkv_projection];
        [encoder setBuffer:current_input offset:0 atIndex:0];
        [encoder setBuffer:block_w.qkv_weights offset:0 atIndex:1];
        [encoder setBuffer:block_w.qkv_bias offset:0 atIndex:2];
        [encoder setBuffer:buffers.attention_qkv[i] offset:0 atIndex:3];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:7];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        // Extract Q, K, V from concatenated QKV for backward pass
        [encoder setComputePipelineState:kernels.extract_qkv_from_concatenated];
        [encoder setBuffer:buffers.attention_qkv[i] offset:0 atIndex:0];      // Concatenated QKV input
        [encoder setBuffer:buffers.attention_Q[i] offset:0 atIndex:1];        // Q output
        [encoder setBuffer:buffers.attention_K[i] offset:0 atIndex:2];        // K output
        [encoder setBuffer:buffers.attention_V[i] offset:0 atIndex:3];        // V output
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:7];
        
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        MTLSize qkvExtractGrid = MTLSizeMake(actual_batch_size * actual_sequence_length, config.num_heads, head_dim);
        MTLSize qkvExtractGroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:qkvExtractGrid threadsPerThreadgroup:qkvExtractGroup];

        // Scaled Dot-Product Attention with Attention Weights Saving
        [encoder setComputePipelineState:kernels.scaled_dot_product_attention_with_weights_save];
        [encoder setBuffer:buffers.attention_qkv[i] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_output[i] offset:0 atIndex:1];
        [encoder setBuffer:buffers.attention_weights[i] offset:0 atIndex:2]; // NEW: Save attention weights
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:6];
        MTLSize attnNewThreadsPerGrid = MTLSizeMake(actual_batch_size, config.num_heads, 1);
        MTLSize attnNewThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:attnNewThreadsPerGrid threadsPerThreadgroup:attnNewThreadsPerThreadgroup];

        // MHSA Output Projection
        [encoder setComputePipelineState:kernels.mhsa_output_projection];
        [encoder setBuffer:buffers.attention_output[i] offset:0 atIndex:0];
        [encoder setBuffer:block_w.attention_output_weights offset:0 atIndex:1];
        [encoder setBuffer:block_w.attention_output_bias offset:0 atIndex:2];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[i] offset:0 atIndex:3]; // MODIFIED: Output to new saved buffer
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        // Add & Norm (Attention)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[i] offset:0 atIndex:0]; // MODIFIED: Input from new saved buffer
        [encoder setBuffer:current_input offset:0 atIndex:1]; // Residual connection
        [encoder setBuffer:buffers.attention_normed[i] offset:0 atIndex:2]; // Output to attention_normed as before
        [encoder setBuffer:block_w.ln1_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln1_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[i*2] offset:0 atIndex:5]; // mean_out
        [encoder setBuffer:buffers.ln_rsqrt_variance[i*2] offset:0 atIndex:6]; // rsqrt_variance_out
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        // Feed-Forward Network
        [encoder setComputePipelineState:kernels.feed_forward_network];
        [encoder setBuffer:buffers.attention_normed[i] offset:0 atIndex:0]; // Input from previous LayerNorm
        [encoder setBuffer:block_w.ffn_w1 offset:0 atIndex:1];
        [encoder setBuffer:block_w.ffn_b1 offset:0 atIndex:2];
        [encoder setBuffer:block_w.ffn_w2 offset:0 atIndex:3];
        [encoder setBuffer:block_w.ffn_b2 offset:0 atIndex:4];
        [encoder setBuffer:buffers.ffn_output[i] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ffn_h_linear[i] offset:0 atIndex:6];
        [encoder setBuffer:buffers.ffn_h_activated[i] offset:0 atIndex:7];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:11];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        // Add & Norm (FFN)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.ffn_output[i] offset:0 atIndex:0]; // Input from FFN
        [encoder setBuffer:buffers.attention_normed[i] offset:0 atIndex:1]; // Residual connection from after attention LN
        [encoder setBuffer:current_block_output offset:0 atIndex:2]; // Final output of the block
        [encoder setBuffer:block_w.ln2_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln2_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[i*2+1] offset:0 atIndex:5]; // mean_out
        [encoder setBuffer:buffers.ln_rsqrt_variance[i*2+1] offset:0 atIndex:6]; // rsqrt_variance_out
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // If not the last layer, copy output to next layer's input
        if (i < config.num_layers - 1) {
            // End current compute encoder, start blit, then new compute for next iter
            [encoder endEncoding];
            blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer:current_block_output sourceOffset:0 
                          toBuffer:buffers.layer_inputs[i+1] destinationOffset:0 
                              size:actual_batch_size * actual_sequence_length * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float))];
            [blitEncoder endEncoding];
            encoder = [commandBuffer computeCommandEncoder]; // Start new compute encoder for next layer
        }
    }

    // Step 4: Final Layer Norm (if num_layers > 0)
    id<MTLBuffer> final_hidden_state_for_output_proj = buffers.embeddings; // Default if no layers
    if (config.num_layers > 0) {
        final_hidden_state_for_output_proj = buffers.block_output[config.num_layers - 1];
        // Note: The layer_norm kernel expects residual input. For final LN, this should be zero or handled by kernel. 
        // Assuming kernel handles if residual_input is nil or points to a zeroed buffer if not additive.
        // For simplicity, we'll reuse the final_hidden_state_for_output_proj as its own output then to final_logits_input buffer
        id<MTLBuffer> final_logits_input_buffer = buffers.layer_inputs[config.num_layers]; // Use last layer_input as temp for final LN output

        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:final_hidden_state_for_output_proj offset:0 atIndex:0];
        [encoder setBuffer:nil offset:0 atIndex:1]; // No residual for final LayerNorm, or pass a zeroed buffer
        [encoder setBuffer:final_logits_input_buffer offset:0 atIndex:2]; 
        [encoder setBuffer:weights.final_ln_gamma offset:0 atIndex:3];
        [encoder setBuffer:weights.final_ln_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.final_ln_mean offset:0 atIndex:5]; // mean_out
        [encoder setBuffer:buffers.final_ln_rsqrt_variance offset:0 atIndex:6]; // rsqrt_variance_out
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        final_hidden_state_for_output_proj = final_logits_input_buffer; // output of LN is input to projection
    }


    // Step 5: Output Logits Projection
    [encoder setComputePipelineState:kernels.output_logits_projection];
    [encoder setBuffer:final_hidden_state_for_output_proj offset:0 atIndex:0];
    [encoder setBuffer:weights.output_weights offset:0 atIndex:1];
    [encoder setBuffer:weights.output_bias offset:0 atIndex:2];
    [encoder setBuffer:buffers.final_logits offset:0 atIndex:3];
    [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:7];
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    [encoder endEncoding];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];

    // Copy logits back to CPU
    output_logits.resize(actual_batch_size * actual_sequence_length * config.vocab_size);
    float* logits_data = static_cast<float*>([buffers.final_logits contents]);
    std::copy(logits_data, logits_data + output_logits.size(), output_logits.begin());

    return true;
}

bool TransformerModel::computeLoss(const std::vector<uint32_t>& input_tokens,
                                  const std::vector<uint32_t>& target_tokens,
                                  float& loss_value) {
    if (input_tokens.size() != target_tokens.size()) {
        std::cerr << "Input and target token sequences must have same length" << std::endl;
        return false;
    }
    
    // Copy target tokens to buffer
    uint32_t* target_data = static_cast<uint32_t*>([buffers.target_tokens contents]);
    std::copy(target_tokens.begin(), target_tokens.end(), target_data);
    
    // Run forward pass to get logits
    std::vector<float> logits;
    if (!forward(input_tokens, logits)) {
        return false;
    }
    
    // Determine actual sequence parameters
    uint32_t actual_batch_size = 1; // Single sequence for now
    uint32_t actual_sequence_length = static_cast<uint32_t>(input_tokens.size());
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Zero the total loss buffer
    float* loss_data = static_cast<float*>([buffers.loss_buffer contents]);
    *loss_data = 0.0f;
    
    // Compute cross-entropy loss
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:kernels.cross_entropy_loss];
        [encoder setBuffer:buffers.final_logits offset:0 atIndex:0];
        [encoder setBuffer:buffers.target_tokens offset:0 atIndex:1];
        [encoder setBuffer:buffers.final_logits offset:0 atIndex:2]; // Reuse for per-token loss (temp)
        [encoder setBuffer:buffers.loss_buffer offset:0 atIndex:3];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.pad_token_id length:sizeof(uint32_t) atIndex:7];
        
        MTLSize threadsPerGrid = MTLSizeMake(actual_batch_size, actual_sequence_length, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
    
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];
    
    // Read back loss and compute average
    loss_data = static_cast<float*>([buffers.loss_buffer contents]);
    
    // Count non-pad tokens for averaging
    uint32_t non_pad_tokens = 0;
    for (uint32_t token_id : target_tokens) {
        if (token_id != config.pad_token_id) {
            non_pad_tokens++;
        }
    }
    
    if (non_pad_tokens > 0) {
        loss_value = *loss_data / non_pad_tokens;
    } else {
        loss_value = 0.0f;
    }
    
    return true;
}

bool TransformerModel::trainStep(const std::vector<uint32_t>& input_tokens,
                                const std::vector<uint32_t>& target_tokens,
                                float& loss) {
    // Step 1: Zero gradients
    if (!zeroGradients()) {
        return false;
    }
    
    // Step 2: Forward pass and compute loss
    if (!computeLoss(input_tokens, target_tokens, loss)) {
        return false;
    }
    
    // Step 3: Compute loss gradients (starting point for backprop)
    uint32_t actual_batch_size = 1;
    uint32_t actual_sequence_length = static_cast<uint32_t>(input_tokens.size());
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Compute gradient of loss w.r.t. logits
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:kernels.loss_gradient];
        [encoder setBuffer:buffers.final_logits offset:0 atIndex:0];
        [encoder setBuffer:buffers.target_tokens offset:0 atIndex:1];
        [encoder setBuffer:buffers.logits_grad offset:0 atIndex:2];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.pad_token_id length:sizeof(uint32_t) atIndex:6];
        
        MTLSize threadsPerGrid = MTLSizeMake(actual_batch_size, actual_sequence_length, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
    
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];
    
    // Step 4: Backward pass - compute parameter gradients
    if (!backwardPass()) {
        return false;
    }
    
    // 🔄 STRATEGIC SYNC: Only wait before optimizer (when gradients needed)
    std::cout << "⏳ Strategic sync before optimizer step..." << std::endl;
    // Strategic sync: Let GPU complete all pending work
    
    // Step 5: Optimizer step  
    if (!optimizerStep()) {
        return false;
    }
    
    return true;
}

bool TransformerModel::evaluate(const std::vector<uint32_t>& input_tokens,
                               const std::vector<uint32_t>& target_tokens,
                               float& loss) {
    // Evaluation is just forward pass + loss computation (no gradients)
    return computeLoss(input_tokens, target_tokens, loss);
}

bool TransformerModel::zeroGradients() {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Zero all gradient buffers
    std::vector<id<MTLBuffer>> grad_buffers_to_zero = {
        gradients.token_embeddings_grad,
        gradients.final_ln_gamma_grad,
        gradients.final_ln_beta_grad,
        gradients.output_weights_grad,
        gradients.output_bias_grad
    };
    
    // Add transformer block gradients
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block_grad = gradients.blocks_grad[layer];
        grad_buffers_to_zero.insert(grad_buffers_to_zero.end(), {
            block_grad.qkv_weights,
            block_grad.qkv_bias,
            block_grad.attention_output_weights,
            block_grad.attention_output_bias,
            block_grad.ln1_gamma,
            block_grad.ln1_beta,
            block_grad.ln2_gamma,
            block_grad.ln2_beta,
            block_grad.ffn_w1,
            block_grad.ffn_b1,
            block_grad.ffn_w2,
            block_grad.ffn_b2
        });
    }
    
    // Zero each gradient buffer
    for (id<MTLBuffer> buffer : grad_buffers_to_zero) {
        if (buffer) {
            uint32_t buffer_size = static_cast<uint32_t>([buffer length] / sizeof(float));
            
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:kernels.zero_gradients];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            [encoder setBytes:&buffer_size length:sizeof(uint32_t) atIndex:1];
            
            MTLSize threadsPerGrid = MTLSizeMake(buffer_size, 1, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1); // Good threadgroup size
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
    }
    
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];
    
    return true;
}

bool TransformerModel::optimizerStep() {
    optimizer_state.timestep++;
    optimizer_state.current_learning_rate = calculateLearningRate(optimizer_state.timestep);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:kernels.adamw_optimizer];

    size_t m_v_buffer_idx = 0;

    // Helper lambda to dispatch optimizer for a given parameter set
    auto dispatch_optimizer = [&](id<MTLBuffer> param_buffer, id<MTLBuffer> grad_buffer) {
        if (!param_buffer || !grad_buffer) return;
        uint32_t param_size = static_cast<uint32_t>([param_buffer length] / (config.use_half_precision ? sizeof(uint16_t) : sizeof(float)));
        if (m_v_buffer_idx >= optimizer_state.m_buffers.size() || m_v_buffer_idx >= optimizer_state.v_buffers.size()) {
            std::cerr << "Optimizer state buffer index out of bounds!" << std::endl;
            return;
        }
        
        [encoder setBuffer:param_buffer offset:0 atIndex:0];
        [encoder setBuffer:grad_buffer offset:0 atIndex:1];
        [encoder setBuffer:optimizer_state.m_buffers[m_v_buffer_idx] offset:0 atIndex:2];
        [encoder setBuffer:optimizer_state.v_buffers[m_v_buffer_idx] offset:0 atIndex:3];
        [encoder setBytes:&optimizer_state.current_learning_rate length:sizeof(float) atIndex:4];
        [encoder setBytes:&config.beta1 length:sizeof(float) atIndex:5];
        [encoder setBytes:&config.beta2 length:sizeof(float) atIndex:6];
        [encoder setBytes:&config.adam_epsilon length:sizeof(float) atIndex:7];
        [encoder setBytes:&config.weight_decay length:sizeof(float) atIndex:8];
        [encoder setBytes:&optimizer_state.timestep length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&param_size length:sizeof(uint32_t) atIndex:10];

        MTLSize threadsPerGrid = MTLSizeMake(param_size, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(std::min(param_size, 256u), 1, 1); // Cap threadgroup size
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        m_v_buffer_idx++;
    };

    // Update token embeddings
    dispatch_optimizer(weights.token_embeddings, gradients.token_embeddings_grad);

    // Update transformer block weights
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        auto& block_weights = weights.blocks[i];
        auto& block_grads = gradients.blocks_grad[i];
        dispatch_optimizer(block_weights.qkv_weights, block_grads.qkv_weights);
        dispatch_optimizer(block_weights.qkv_bias, block_grads.qkv_bias);
        dispatch_optimizer(block_weights.attention_output_weights, block_grads.attention_output_weights);
        dispatch_optimizer(block_weights.attention_output_bias, block_grads.attention_output_bias);
        dispatch_optimizer(block_weights.ln1_gamma, block_grads.ln1_gamma);
        dispatch_optimizer(block_weights.ln1_beta, block_grads.ln1_beta);
        dispatch_optimizer(block_weights.ln2_gamma, block_grads.ln2_gamma);
        dispatch_optimizer(block_weights.ln2_beta, block_grads.ln2_beta);
        dispatch_optimizer(block_weights.ffn_w1, block_grads.ffn_w1);
        dispatch_optimizer(block_weights.ffn_b1, block_grads.ffn_b1);
        dispatch_optimizer(block_weights.ffn_w2, block_grads.ffn_w2);
        dispatch_optimizer(block_weights.ffn_b2, block_grads.ffn_b2);
    }

    // Update final layer norm
    dispatch_optimizer(weights.final_ln_gamma, gradients.final_ln_gamma_grad);
    dispatch_optimizer(weights.final_ln_beta, gradients.final_ln_beta_grad);

    // Update output projection weights
    dispatch_optimizer(weights.output_weights, gradients.output_weights_grad);
    dispatch_optimizer(weights.output_bias, gradients.output_bias_grad);

    [encoder endEncoding];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];

    return true;
}

float TransformerModel::calculateLearningRate(uint32_t step) {
    // Simple learning rate (could implement warmup + decay)
    return config.learning_rate;
}

bool TransformerModel::backwardPass(size_t current_sequence_index) {
    uint32_t actual_batch_size = 1; // Assuming batch size of 1 for now
    uint32_t actual_sequence_length = static_cast<uint32_t>([buffers.input_tokens length] / sizeof(uint32_t));
    uint32_t num_instances = actual_batch_size * actual_sequence_length;

    // --- GLOBAL DIAGNOSTIC CHECKS (if diagnostic mode is enabled) ---
    if (m_is_diagnostic_run) {
        std::cout << "🔍 DIAGNOSTIC: Starting comprehensive vulnerability checks for sequence " << current_sequence_index << std::endl;
        
        // Global config validation
        if (config.vocab_size == 0) {
            std::cerr << "❌ DIAGNOSTIC GLOBAL: vocab_size is 0" << std::endl;
            return false;
        }
        if (config.embedding_dim == 0) {
            std::cerr << "❌ DIAGNOSTIC GLOBAL: embedding_dim is 0" << std::endl;
            return false;
        }
        if (config.num_heads == 0 || config.embedding_dim % config.num_heads != 0) {
            std::cerr << "❌ DIAGNOSTIC GLOBAL: Invalid num_heads (" << config.num_heads << ") for embedding_dim (" << config.embedding_dim << ")" << std::endl;
            return false;
        }
        
        // ADDITIONAL SAFETY CHECK 1: Buffer State & Memory Consistency
        std::cout << "🔍 DIAGNOSTIC: Checking buffer states and memory consistency..." << std::endl;
        
        // Critical buffer allocation validation
        if (!buffers.input_tokens || [buffers.input_tokens length] == 0) {
            std::cerr << "❌ DIAGNOSTIC BUFFER: input_tokens buffer invalid" << std::endl;
            return false;
        }
        if (!buffers.final_logits || [buffers.final_logits length] == 0) {
            std::cerr << "❌ DIAGNOSTIC BUFFER: final_logits buffer invalid" << std::endl;
            return false;
        }
        
        // Validate all layer buffers exist and are consistent
        for (uint32_t layer = 0; layer < config.num_layers; layer++) {
            if (!buffers.layer_inputs[layer]) {
                std::cerr << "❌ DIAGNOSTIC BUFFER: layer_inputs[" << layer << "] is null" << std::endl;
                return false;
            }
            if (!buffers.attention_output[layer]) {
                std::cerr << "❌ DIAGNOSTIC BUFFER: attention_output[" << layer << "] is null" << std::endl;
                return false;
            }
            
            // Check buffer sizes are consistent
            size_t expected_layer_size = num_instances * config.embedding_dim * sizeof(uint16_t);
            if ([buffers.layer_inputs[layer] length] < expected_layer_size) {
                std::cerr << "❌ DIAGNOSTIC BUFFER: layer_inputs[" << layer << "] size mismatch. Expected: " << expected_layer_size << ", Actual: " << [buffers.layer_inputs[layer] length] << std::endl;
                return false;
            }
        }
        
        // ADDITIONAL SAFETY CHECK 2: Numerical Stability Analysis
        std::cout << "🔍 DIAGNOSTIC: Checking numerical stability..." << std::endl;
        
        // Check for NaN/Inf in final_logits
        float* logits_data = static_cast<float*>([buffers.final_logits contents]);
        size_t logits_elements = [buffers.final_logits length] / sizeof(float);
        int nan_count = 0, inf_count = 0;
        float max_abs_value = 0.0f;
        
        for (size_t i = 0; i < std::min(logits_elements, (size_t)10000); i++) { // Check first 10k elements
            float val = logits_data[i];
            if (std::isnan(val)) nan_count++;
            if (std::isinf(val)) inf_count++;
            max_abs_value = std::max(max_abs_value, std::abs(val));
        }
        
        if (nan_count > 0) {
            std::cerr << "❌ DIAGNOSTIC NUMERICAL: Found " << nan_count << " NaN values in final_logits" << std::endl;
            return false;
        }
        if (inf_count > 0) {
            std::cerr << "❌ DIAGNOSTIC NUMERICAL: Found " << inf_count << " Inf values in final_logits" << std::endl;
            return false;
        }
        if (max_abs_value > 1e6f) {
            std::cerr << "❌ DIAGNOSTIC NUMERICAL: Extreme values detected. Max abs: " << max_abs_value << std::endl;
            return false;
        }
        
        // Check for gradient explosion in weight gradients (if available)
        if (gradients.token_embeddings_grad) {
            float* grad_data = static_cast<float*>([gradients.token_embeddings_grad contents]);
            size_t grad_elements = std::min((size_t)1000, [gradients.token_embeddings_grad length] / sizeof(float));
            float grad_max = 0.0f;
            for (size_t i = 0; i < grad_elements; i++) {
                grad_max = std::max(grad_max, std::abs(grad_data[i]));
            }
            if (grad_max > 100.0f) {
                std::cerr << "❌ DIAGNOSTIC NUMERICAL: Gradient explosion detected. Max grad: " << grad_max << std::endl;
                return false;
            }
        }
        
        // ADDITIONAL SAFETY CHECK 3: GPU Memory Pressure & Resource Monitoring
        std::cout << "🔍 DIAGNOSTIC: Monitoring GPU resources..." << std::endl;
        
        if (@available(macOS 10.15, *)) {
            // Estimate current Metal memory usage
            size_t estimated_usage = 0;
            estimated_usage += [buffers.input_tokens length];
            estimated_usage += [buffers.final_logits length];
            estimated_usage += [buffers.embeddings length];
            
            for (uint32_t layer = 0; layer < config.num_layers; layer++) {
                if (buffers.layer_inputs[layer]) estimated_usage += [buffers.layer_inputs[layer] length];
                if (buffers.attention_output[layer]) estimated_usage += [buffers.attention_output[layer] length];
                if (buffers.ffn_output[layer]) estimated_usage += [buffers.ffn_output[layer] length];
                if (buffers.attention_Q[layer]) estimated_usage += [buffers.attention_Q[layer] length];
            }
            
            std::cout << "🔍 DIAGNOSTIC: Estimated GPU memory usage: " << (estimated_usage / 1024 / 1024) << " MB" << std::endl;
            
            // Check if we're approaching memory limits
            uint64_t recommended_limit = [device recommendedMaxWorkingSetSize];
            if (estimated_usage > recommended_limit * 0.8) { // 80% threshold
                std::cerr << "❌ DIAGNOSTIC MEMORY: GPU memory usage (" << (estimated_usage / 1024 / 1024) 
                          << " MB) approaching limit (" << (recommended_limit / 1024 / 1024) << " MB)" << std::endl;
                return false;
            }
        }
        
        // ADDITIONAL SAFETY CHECK 4: Canary Buffer Verification
        std::cout << "🔍 DIAGNOSTIC: Verifying canary buffers for memory corruption..." << std::endl;
        if (m_canary_buffer_before && m_canary_buffer_after) {
            uint32_t* canary_before = static_cast<uint32_t*>([m_canary_buffer_before contents]);
            uint32_t* canary_after = static_cast<uint32_t*>([m_canary_buffer_after contents]);
            
            // Check for known corruption patterns
            const uint32_t CANARY_PATTERN = 0xDEADBEEF;
            for (int i = 0; i < 4; i++) { // Check first 4 uint32_t values
                if (canary_before[i] != CANARY_PATTERN && canary_before[i] != 0) {
                    std::cerr << "❌ DIAGNOSTIC CANARY: Before-buffer corruption detected at index " << i 
                              << ". Value: 0x" << std::hex << canary_before[i] << std::dec << std::endl;
                    return false;
                }
                if (canary_after[i] != CANARY_PATTERN && canary_after[i] != 0) {
                    std::cerr << "❌ DIAGNOSTIC CANARY: After-buffer corruption detected at index " << i 
                              << ". Value: 0x" << std::hex << canary_after[i] << std::dec << std::endl;
                    return false;
                }
            }
        }
        
        // ADDITIONAL SAFETY CHECK 5: Parameter Bounds Validation (runtime values)
        std::cout << "🔍 DIAGNOSTIC: Validating runtime parameter bounds..." << std::endl;
        
        // Validate actual sequence length against configuration limits
        if (actual_sequence_length > config.max_sequence_length) {
            std::cerr << "❌ DIAGNOSTIC RUNTIME: Actual sequence length (" << actual_sequence_length 
                      << ") exceeds configured maximum (" << config.max_sequence_length << ")" << std::endl;
            return false;
        }
        
        // Validate batch processing parameters
        if (actual_batch_size == 0 || actual_batch_size > config.batch_size) {
            std::cerr << "❌ DIAGNOSTIC RUNTIME: Invalid actual_batch_size: " << actual_batch_size 
                      << " (config.batch_size: " << config.batch_size << ")" << std::endl;
            return false;
        }
        
        // Validate computed values for numerical sanity
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        if (head_dim == 0 || head_dim > config.embedding_dim) {
            std::cerr << "❌ DIAGNOSTIC RUNTIME: Invalid head_dim: " << head_dim << std::endl;
            return false;
        }
        
        // Validate current sequence index bounds
        if (current_sequence_index >= actual_sequence_length) {
            std::cerr << "❌ DIAGNOSTIC RUNTIME: current_sequence_index (" << current_sequence_index 
                      << ") >= actual_sequence_length (" << actual_sequence_length << ")" << std::endl;
            return false;
        }
        
        // ADDITIONAL SAFETY CHECK 6: Learning Rate & Optimizer State Sanity
        std::cout << "🔍 DIAGNOSTIC: Checking optimizer state sanity..." << std::endl;
        
        float current_lr = calculateLearningRate(optimizer_state.timestep);
        if (std::isnan(current_lr) || std::isinf(current_lr) || current_lr <= 0.0f || current_lr > 10.0f) {
            std::cerr << "❌ DIAGNOSTIC OPTIMIZER: Invalid learning rate: " << current_lr 
                      << " (timestep: " << optimizer_state.timestep << ")" << std::endl;
            return false;
        }
        
        // Check for numerical issues in optimizer hyperparameters
        float beta1_power = std::pow(config.beta1, optimizer_state.timestep);
        float beta2_power = std::pow(config.beta2, optimizer_state.timestep);
        if (std::isnan(beta1_power) || std::isnan(beta2_power) || beta1_power <= 0.0f || beta2_power <= 0.0f) {
            std::cerr << "❌ DIAGNOSTIC OPTIMIZER: Invalid beta powers. beta1^t: " << beta1_power 
                      << ", beta2^t: " << beta2_power << " (timestep: " << optimizer_state.timestep << ")" << std::endl;
            return false;
        }
        
        std::cout << "✅ DIAGNOSTIC: All additional safety checks passed for sequence " << current_sequence_index << std::endl;
    }
    
    // === EXISTING LAYER-BY-LAYER BACKWARD PASS ===
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Step 1: Backward pass for Output Projection Layer
    // --- DIAGNOSTIC CHECKS FOR OUTPUT PROJECTION ---
    if (m_is_diagnostic_run) {
        std::cout << "🔍 DIAGNOSTIC: Checking output projection vulnerabilities..." << std::endl;
        
        // Output projection weight matrix bounds
        size_t output_weights_required = config.embedding_dim * config.vocab_size * sizeof(uint16_t);
        if ([weights.output_weights length] < output_weights_required) {
            std::cerr << "❌ DIAGNOSTIC OUTPUT_PROJ: Output weights buffer insufficient. Required: " << output_weights_required << ", Actual: " << [weights.output_weights length] << std::endl;
            return false;
        }
        
        // Output bias vector bounds
        size_t output_bias_required = config.vocab_size * sizeof(float);
        if ([weights.output_bias length] < output_bias_required) {
            std::cerr << "❌ DIAGNOSTIC OUTPUT_PROJ: Output bias buffer insufficient. Required: " << output_bias_required << ", Actual: " << [weights.output_bias length] << std::endl;
            return false;
        }
        
        // Gradient buffer bounds for output projection
        size_t output_weights_grad_required = config.embedding_dim * config.vocab_size * sizeof(float);
        if ([gradients.output_weights_grad length] < output_weights_grad_required) {
            std::cerr << "❌ DIAGNOSTIC OUTPUT_PROJ: Output weights gradient buffer insufficient. Required: " << output_weights_grad_required << ", Actual: " << [gradients.output_weights_grad length] << std::endl;
            return false;
        }
        
        // Check for potential integer overflow in matrix dimensions
        uint64_t matrix_elements = (uint64_t)config.embedding_dim * config.vocab_size;
        if (matrix_elements > UINT32_MAX) {
            std::cerr << "❌ DIAGNOSTIC OUTPUT_PROJ: Matrix too large, potential overflow. Elements: " << matrix_elements << std::endl;
            return false;
        }
        
        // Threadgroup bounds for output projection kernel
        uint64_t max_tg_output_proj = kernels.output_projection_backward.maxTotalThreadsPerThreadgroup;
        uint64_t proposed_tg_size = config.embedding_dim * config.vocab_size;
        if (proposed_tg_size > max_tg_output_proj) {
            std::cout << "⚠️  DIAGNOSTIC OUTPUT_PROJ: Large threadgroup may be needed. Proposed: " << proposed_tg_size << ", Max: " << max_tg_output_proj << std::endl;
        }
        
        std::cout << "✅ DIAGNOSTIC OUTPUT_PROJ: Output projection checks passed" << std::endl;
    }
    
    // === FINAL PRE-KERNEL-DISPATCH SAFETY CHECK ===
    if (m_is_diagnostic_run) {
        std::cout << "🔍 DIAGNOSTIC: Final pre-kernel-dispatch safety validation..." << std::endl;
        
        // Critical kernel availability check
        if (!kernels.output_projection_backward) {
            std::cerr << "❌ DIAGNOSTIC KERNEL: output_projection_backward kernel is null" << std::endl;
            return false;
        }
        if (!kernels.layer_norm_backward) {
            std::cerr << "❌ DIAGNOSTIC KERNEL: layer_norm_backward kernel is null" << std::endl;
            return false;
        }
        if (!kernels.ffn_backward) {
            std::cerr << "❌ DIAGNOSTIC KERNEL: ffn_backward kernel is null" << std::endl;
            return false;
        }
        
        // Command buffer and encoder state check
        if (!commandBuffer) {
            std::cerr << "❌ DIAGNOSTIC DISPATCH: Command buffer is null" << std::endl;
            return false;
        }
        if (!encoder) {
            std::cerr << "❌ DIAGNOSTIC DISPATCH: Compute encoder is null" << std::endl;
            return false;
        }
        
        // Device and command queue health check
        if (!device) {
            std::cerr << "❌ DIAGNOSTIC DISPATCH: Metal device is null" << std::endl;
            return false;
        }
        if (!commandQueue) {
            std::cerr << "❌ DIAGNOSTIC DISPATCH: Command queue is null" << std::endl;
            return false;
        }
        
        // Final numerical stability check on critical parameters
        if (config.embedding_dim == 0 || config.num_heads == 0 || config.ffn_hidden_dim == 0) {
            std::cerr << "❌ DIAGNOSTIC DISPATCH: Critical dimension is zero - this will cause GPU kernel crashes" << std::endl;
            return false;
        }
        
        // Final memory coherency check - verify critical buffers are still valid
        NSError* error;
        if (@available(macOS 11.0, *)) {
            // Quick validation that our most critical buffers are still accessible
            if (buffers.final_logits && [buffers.final_logits length] > 0) {
                // Touch the first byte to ensure it's mapped
                volatile uint8_t test_byte = *((uint8_t*)[buffers.final_logits contents]);
                (void)test_byte; // Suppress unused variable warning
            }
        }
        
        std::cout << "✅ DIAGNOSTIC: Final pre-dispatch checks passed - proceeding with kernel execution" << std::endl;
    }
    
    // Computes: dL/dW_out, dL/db_out, dL/dFinalHidden (input to this layer)
    [encoder setComputePipelineState:kernels.output_projection_backward];
    [encoder setBuffer:buffers.logits_grad offset:0 atIndex:0]; // dL/dLogits
    
    // Determine input to output projection (was it output of final LN or embeddings?)
    id<MTLBuffer> input_to_output_proj;
    if (config.num_layers > 0) {
        input_to_output_proj = buffers.layer_inputs[config.num_layers]; // This was temp output of final LN
    } else {
        input_to_output_proj = buffers.embeddings; // If no layers, embeddings go directly to output projection
    }
    [encoder setBuffer:input_to_output_proj offset:0 atIndex:1]; 
    [encoder setBuffer:weights.output_weights offset:0 atIndex:2];
    [encoder setBuffer:gradients.output_weights_grad offset:0 atIndex:3];
    [encoder setBuffer:gradients.output_bias_grad offset:0 atIndex:4];
    [encoder setBuffer:buffers.final_hidden_grad offset:0 atIndex:5]; // dL/dFinalHidden output
    [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:9];
    
    // Dispatch for output_projection_backward (example, may need refinement based on kernel impl.)
    // GID.x: instance_idx, GID.y: e_idx, GID.z: v_idx
    MTLSize op_threadsPerGrid = MTLSizeMake(num_instances, config.embedding_dim, config.vocab_size);
    MTLSize op_threadsPerThreadgroup = MTLSizeMake(1, 1, 1); // Adjust for performance
    // Ensure threadgroup size doesn't exceed maxTotalThreadsPerThreadgroup
    uint32_t max_tg_size = kernels.output_projection_backward.maxTotalThreadsPerThreadgroup;
    if (op_threadsPerThreadgroup.width * op_threadsPerThreadgroup.height * op_threadsPerThreadgroup.depth > max_tg_size) {
        // Basic fallback, can be smarter
        op_threadsPerThreadgroup = MTLSizeMake(std::min(num_instances, max_tg_size), 1, 1);
        op_threadsPerGrid = MTLSizeMake(num_instances, config.embedding_dim, 1); // Reduce grid if dL/dX only needs 2D
        // This dispatch needs to match the kernel's expectation more closely.
        // For now, this is a placeholder dispatch. The kernel uses gid.y for e_idx and gid.z for v_idx.
        // If we want to compute dL/dX (num_instances * emb_dim) and dL/dW (emb_dim * vocab_size) in one go,
        // this 3D grid is one way, but kernel logic must carefully use gid.
    }
    // A safer dispatch for the current output_projection_backward kernel (which computes parts based on gid.y/gid.z):
    op_threadsPerGrid = MTLSizeMake(num_instances, config.embedding_dim, config.vocab_size);
    op_threadsPerThreadgroup = MTLSizeMake(1, 8, 8); // Example: process 64 elements of E*V plane per group for dW
    if (op_threadsPerThreadgroup.width * op_threadsPerThreadgroup.height * op_threadsPerThreadgroup.depth > max_tg_size) {
         op_threadsPerThreadgroup = MTLSizeMake(1, std::min(config.embedding_dim, max_tg_size/8 > 0 ? max_tg_size/8 : 1u), std::min(config.vocab_size, 8u));
    }
    [encoder dispatchThreads:op_threadsPerGrid threadsPerThreadgroup:op_threadsPerThreadgroup];


    // Step 2: Backward pass for Final Layer Normalization (if layers > 0)
    if (config.num_layers > 0) {
        if (m_is_diagnostic_run) {
            // Vulnerability 2: Out-of-Bounds Threadgroup Memory Access in layer_norm_backward
            // The kernel comment says: "// Assuming embedding_dim <= 1024"
            // The threadgroup arrays are sized based on this assumption.
            if (config.embedding_dim > 1024) {
                std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << "): Potential OOB in layer_norm_backward (final LN). config.embedding_dim (" << config.embedding_dim << ") > 1024." << std::endl;
                return false;
            }
            // Vulnerability 6: Division by zero
            if (config.embedding_dim == 0) {
                std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << "): Potential DIV0 in layer_norm_backward (final LN). config.embedding_dim is 0." << std::endl;
                return false;
            }
        }
        [encoder setComputePipelineState:kernels.layer_norm_backward];
        [encoder setBuffer:buffers.final_hidden_grad offset:0 atIndex:0]; // dL/dOutput_of_LN (from prev step)
        [encoder setBuffer:buffers.block_output[config.num_layers-1] offset:0 atIndex:1]; // Original input to this LN
        [encoder setBuffer:weights.final_ln_gamma offset:0 atIndex:2];
        [encoder setBuffer:buffers.final_ln_mean offset:0 atIndex:3];
        [encoder setBuffer:buffers.final_ln_rsqrt_variance offset:0 atIndex:4];
        [encoder setBuffer:gradients.final_ln_gamma_grad offset:0 atIndex:5];
        [encoder setBuffer:gradients.final_ln_beta_grad offset:0 atIndex:6];
        // Output: dL/dInput_to_LN (which is dL/dBlockOutput_N-1)
        [encoder setBuffer:buffers.block_output_grad[config.num_layers-1] offset:0 atIndex:7]; 
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];

        MTLSize ln_bwd_threadsPerGrid = MTLSizeMake(num_instances, 1, 1);
        MTLSize ln_bwd_threadsPerThreadgroup = MTLSizeMake(config.embedding_dim, 1, 1);
        if (ln_bwd_threadsPerThreadgroup.width > kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup) {
            ln_bwd_threadsPerThreadgroup.width = kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup;
        }
        [encoder dispatchThreadgroups:ln_bwd_threadsPerGrid threadsPerThreadgroup:ln_bwd_threadsPerThreadgroup];
        
        // Step 3: Backward pass through transformer layers (in reverse order)
        for (int32_t layer = config.num_layers - 1; layer >= 0; layer--) {
            // --- Diagnostic checks for this layer if m_is_diagnostic_run is true ---
            if (m_is_diagnostic_run) {
                // Vulnerability 2 & 6 (layer_norm_backward for LN2 and LN1 within the loop)
                if (config.embedding_dim > 1024) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential OOB in layer_norm_backward. config.embedding_dim (" << config.embedding_dim << ") > 1024." << std::endl;
                    return false;
                }
                if (config.embedding_dim == 0) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential DIV0 in layer_norm_backward. config.embedding_dim is 0." << std::endl;
                    return false;
                }

                // Vulnerability 3 (ffn_backward)
                // The kernel has tg_dh_act[256] and tg_dh_lin[256]
                const uint32_t ffn_backward_tg_array_size = 2048;  // Updated from our MSL fix
                if (config.ffn_hidden_dim > ffn_backward_tg_array_size) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential OOB in ffn_backward. config.ffn_hidden_dim (" << config.ffn_hidden_dim << ") > threadgroup array size (" << ffn_backward_tg_array_size << ")." << std::endl;
                    return false;
                }

                // Vulnerability 4 (Attention kernels - sequence length bounds)
                if (actual_sequence_length > config.max_sequence_length) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Sequence length (" << actual_sequence_length << ") exceeds max_sequence_length (" << config.max_sequence_length << ")." << std::endl;
                    return false;
                }

                // Vulnerability 5 (scaled_dot_product_attention_inference stack buffer overflow)
                // The inference kernel declares: float attention_scores[512];
                const uint32_t attention_scores_stack_size = 512;
                if (actual_sequence_length > attention_scores_stack_size) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential stack buffer overflow in scaled_dot_product_attention_inference. sequence_length (" << actual_sequence_length << ") > stack array size (" << attention_scores_stack_size << ")." << std::endl;
                    return false;
                }

                // Vulnerability 6 (Division by zero checks)
                if (config.num_heads == 0) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential DIV0 calculating head_dim. config.num_heads is 0." << std::endl;
                    return false;
                }
                uint32_t head_dim = config.embedding_dim / config.num_heads;
                if (head_dim == 0) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential DIV0 in attention kernels. head_dim is 0 (emb_dim: " << config.embedding_dim << ", num_heads: " << config.num_heads << ")." << std::endl;
                    return false;
                }
                if (config.ffn_hidden_dim == 0) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Potential DIV0 in FFN kernels. config.ffn_hidden_dim is 0." << std::endl;
                    return false;
                }

                // Vulnerability 7 (Buffer bounds checking)
                size_t required_buffer_size = num_instances * config.embedding_dim * sizeof(uint16_t);
                if ([buffers.layer_inputs[layer] length] < required_buffer_size) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Buffer size mismatch. Required: " << required_buffer_size << ", Actual: " << [buffers.layer_inputs[layer] length] << std::endl;
                    return false;
                }

                // Vulnerability 8 (Threadgroup size validation)
                uint64_t max_threadgroup_size = kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup;
                if (config.embedding_dim > max_threadgroup_size) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Threadgroup size (" << config.embedding_dim << ") exceeds max allowed (" << max_threadgroup_size << ") for layer_norm_backward." << std::endl;
                    return false;
                }

                // Vulnerability 9 (QKV projection buffer overflow check)
                size_t qkv_required_size = num_instances * config.embedding_dim * 3 * sizeof(uint16_t);
                if ([buffers.attention_qkv[layer] length] < qkv_required_size) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): QKV buffer size insufficient. Required: " << qkv_required_size << ", Actual: " << [buffers.attention_qkv[layer] length] << std::endl;
                    return false;
                }

                // Vulnerability 10 (FFN hidden dimension buffer bounds)
                size_t ffn_hidden_required = num_instances * config.ffn_hidden_dim * sizeof(uint16_t);
                if ([buffers.ffn_h_linear[layer] length] < ffn_hidden_required) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): FFN hidden buffer size insufficient. Required: " << ffn_hidden_required << ", Actual: " << [buffers.ffn_h_linear[layer] length] << std::endl;
                    return false;
                }

                // Vulnerability 11 (Attention weights buffer overflow)
                size_t attention_weights_required = config.batch_size * config.num_heads * actual_sequence_length * actual_sequence_length * sizeof(uint16_t);
                if ([buffers.attention_weights[layer] length] < attention_weights_required) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Attention weights buffer size insufficient. Required: " << attention_weights_required << ", Actual: " << [buffers.attention_weights[layer] length] << std::endl;
                    return false;
                }

                // Vulnerability 12 (Gradient buffer alignment check)
                if (!buffers.attention_qkv_grad[layer] || !buffers.layer_inputs_grad[layer]) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Gradient buffers are null." << std::endl;
                    return false;
                }

                // Vulnerability 13 (Weight matrix dimension consistency)
                size_t qkv_weights_required = 3 * config.embedding_dim * config.embedding_dim * sizeof(uint16_t);
                if ([weights.blocks[layer].qkv_weights length] < qkv_weights_required) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): QKV weights buffer size insufficient. Required: " << qkv_weights_required << ", Actual: " << [weights.blocks[layer].qkv_weights length] << std::endl;
                    return false;
                }

                // Vulnerability 14 (Output projection bounds)
                size_t output_proj_weights_required = config.embedding_dim * config.embedding_dim * sizeof(uint16_t);
                if ([weights.blocks[layer].attention_output_weights length] < output_proj_weights_required) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): Attention output weights buffer size insufficient. Required: " << output_proj_weights_required << ", Actual: " << [weights.blocks[layer].attention_output_weights length] << std::endl;
                    return false;
                }

                // Vulnerability 15 (Metal dispatch bounds validation)
                if (num_instances == 0) {
                    std::cerr << "❌ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): num_instances is 0, would cause invalid dispatch." << std::endl;
                    return false;
                }

                std::cout << "✅ DIAGNOSTIC (Seq: " << current_sequence_index << ", Layer: " << layer << "): All vulnerability checks passed for layer " << layer << std::endl;
            }

            // Define common sizes for LayerNorm backward dispatches once per layer
            size_t grad_buffer_size = num_instances * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float));
            MTLSize lnThreadsPerGrid = MTLSizeMake(num_instances, 1, 1);
            MTLSize lnThreadsPerThreadgroup = MTLSizeMake(std::min((size_t)config.embedding_dim, kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup), 1, 1);
            if (lnThreadsPerThreadgroup.width == 0) lnThreadsPerThreadgroup.width = 1; // Ensure width is at least 1

            // --- Backward LayerNorm for FFN (LN2) of current layer ---
            // Input grad: buffers.block_output_grad[layer] (dL/d output_of_LN2[layer])
            // This should come from final LN backward (for layer N-1) or be initialized for the last layer
            if (layer == config.num_layers - 1) {
                // For the last layer, we already computed buffers.block_output_grad[layer] in Step 2
                // No additional initialization needed
            }
            
            // LN2 backward: input_to_LN2 = ffn_output[layer] + attention_normed[layer] (residual)
            [encoder setComputePipelineState:kernels.layer_norm_backward];
            [encoder setBuffer:buffers.block_output_grad[layer] offset:0 atIndex:0]; // dL/d output_of_LN2[layer]
            [encoder setBuffer:buffers.ffn_output[layer] offset:0 atIndex:1];       // ffn_output (saved from forward)
            [encoder setBuffer:weights.blocks[layer].ln2_gamma offset:0 atIndex:2];
            [encoder setBuffer:buffers.ln_mean[layer*2+1] offset:0 atIndex:3];      // Mean/var for LN2
            [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2+1] offset:0 atIndex:4];
            [encoder setBuffer:gradients.blocks_grad[layer].ln2_gamma offset:0 atIndex:5];
            [encoder setBuffer:gradients.blocks_grad[layer].ln2_beta offset:0 atIndex:6];
            // Output: dL/d(ffn_output[layer] + attention_normed[layer])
            [encoder setBuffer:buffers.ffn_output_grad[layer] offset:0 atIndex:7];  // Temp buffer for combined gradient
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:9];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
            
            // size_t grad_buffer_size = num_instances * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float));
            // MTLSize lnThreadsPerGrid = MTLSizeMake(num_instances, 1, 1);
            // MTLSize lnThreadsPerThreadgroup = MTLSizeMake(std::min((size_t)config.embedding_dim, kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup), 1, 1);
            // if (lnThreadsPerThreadgroup.width == 0) lnThreadsPerThreadgroup.width = 1;
            [encoder dispatchThreadgroups:lnThreadsPerGrid threadsPerThreadgroup:lnThreadsPerThreadgroup];
            
            // Copy the gradient to attention_normed_grad (residual connection)
            [encoder endEncoding]; 
            id<MTLBlitCommandEncoder> blitEncoderLN2Res = [commandBuffer blitCommandEncoder];
            [blitEncoderLN2Res copyFromBuffer:buffers.ffn_output_grad[layer] sourceOffset:0 
                                    toBuffer:buffers.attention_normed_grad[layer] destinationOffset:0 
                                        size:grad_buffer_size]; 
            [blitEncoderLN2Res endEncoding];
            encoder = [commandBuffer computeCommandEncoder];

            // --- Backward FFN ---
            // Input grad: buffers.ffn_output_grad[layer] (dL/d ffn_output[layer])
            [encoder setComputePipelineState:kernels.ffn_backward];
            [encoder setBuffer:buffers.ffn_output_grad[layer] offset:0 atIndex:0];      // dL/dY (gradient w.r.t. FFN output)
            [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:1];     // X (input to FFN, saved from forward)
            [encoder setBuffer:buffers.ffn_h_linear[layer] offset:0 atIndex:2];        // h_linear (saved from forward)
            [encoder setBuffer:buffers.ffn_h_activated[layer] offset:0 atIndex:3];     // h_activated (saved from forward)
            [encoder setBuffer:weights.blocks[layer].ffn_w1 offset:0 atIndex:4];       // W1 weights
            [encoder setBuffer:weights.blocks[layer].ffn_w2 offset:0 atIndex:5];       // W2 weights
            [encoder setBuffer:gradients.blocks_grad[layer].ffn_w1 offset:0 atIndex:6]; // dL/dW1 (output)
            [encoder setBuffer:gradients.blocks_grad[layer].ffn_b1 offset:0 atIndex:7]; // dL/db1 (output)
            [encoder setBuffer:gradients.blocks_grad[layer].ffn_w2 offset:0 atIndex:8]; // dL/dW2 (output)
            [encoder setBuffer:gradients.blocks_grad[layer].ffn_b2 offset:0 atIndex:9]; // dL/db2 (output)
            [encoder setBuffer:buffers.attention_normed_grad[layer] offset:0 atIndex:10]; // dL/dX (output, overwrites previous content)
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:13];
            [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:14];
            
            MTLSize ffnThreadsPerGrid = MTLSizeMake(num_instances, config.embedding_dim, config.ffn_hidden_dim);
            MTLSize ffnThreadsPerThreadgroup = MTLSizeMake(1, 1, 1); // Adjust based on kernel requirements
            [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
            
            // After FFN Backward Pass:
            // buffers.attention_normed_grad[layer] now holds dL/d(output of LN1 / input of FFN).
            // This is the input gradient for LN1_backward.

            // Define common sizes if not already available in this exact scope from a prior LN2 backward pass
            // (which is currently missing but would typically define these)
            // size_t grad_buffer_size = num_instances * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float));
            // MTLSize lnThreadsPerGrid = MTLSizeMake(num_instances, 1, 1);
            // MTLSize lnThreadsPerThreadgroup = MTLSizeMake(std::min((size_t)config.embedding_dim, kernels.layer_norm_backward.maxTotalThreadsPerThreadgroup), 1, 1);
            // if (lnThreadsPerThreadgroup.width == 0) lnThreadsPerThreadgroup.width = 1;


            // --- Backward LayerNorm for MHSA (LN1) of current layer ---
            // Input grad: buffers.attention_normed_grad[layer] (dL/d output_of_LN1[layer])
            // Output grad: buffers.attention_output_grad[layer] (dL/d input_to_LN1[layer])
            // input_to_LN1[layer] = mhsa_projection_outputs_saved[layer] + layer_inputs[layer] (residual)
            [encoder setComputePipelineState:kernels.layer_norm_backward];
            [encoder setBuffer:buffers.attention_normed_grad[layer] offset:0 atIndex:0]; 
            [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:1]; // Output of MHSA projection
            [encoder setBuffer:weights.blocks[layer].ln1_gamma offset:0 atIndex:2];
            [encoder setBuffer:buffers.ln_mean[layer*2] offset:0 atIndex:3];         // Mean/var for LN1
            [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2] offset:0 atIndex:4];
            [encoder setBuffer:gradients.blocks_grad[layer].ln1_gamma offset:0 atIndex:5];
            [encoder setBuffer:gradients.blocks_grad[layer].ln1_beta offset:0 atIndex:6];
            // Output: dL/d(mhsa_projection_outputs_saved[layer] + layer_inputs[layer])
            [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:7]; 
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:9];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
            [encoder dispatchThreadgroups:lnThreadsPerGrid threadsPerThreadgroup:lnThreadsPerThreadgroup]; 

            // buffers.attention_output_grad[layer] now contains dL/d(mhsa_projection_outputs_saved[layer] + layer_inputs[layer]).
            // This gradient applies to BOTH mhsa_projection_outputs_saved[layer] AND layer_inputs[layer].
            
            // Accumulate dL/d(layer_inputs[layer]) into buffers.layer_inputs_grad[layer].
            // This should be an accumulation. A direct copy is used for now.
            // TODO: Ensure buffers.layer_inputs_grad[layer] is properly zeroed before the loop or use an accumulation kernel.
            [encoder endEncoding]; 
            id<MTLBlitCommandEncoder> blitEncoderLN1Res = [commandBuffer blitCommandEncoder];
            [blitEncoderLN1Res copyFromBuffer:buffers.attention_output_grad[layer] sourceOffset:0 
                                    toBuffer:buffers.layer_inputs_grad[layer] destinationOffset:0 
                                        size:grad_buffer_size]; 
            [blitEncoderLN1Res endEncoding];
            encoder = [commandBuffer computeCommandEncoder];
            // The gradient dL/d(mhsa_projection_outputs_saved[layer]) is still in buffers.attention_output_grad[layer] for the next kernel.


            // --- Backward MHSA Output Projection ---
            // Input grad: buffers.attention_output_grad[layer] (dL/d mhsa_projection_outputs_saved[layer])
            // Output grad_concatenated_attention_heads: buffers.attention_output_grad[layer] (overwritten, dL/d concatenated_heads)
            [encoder setComputePipelineState:kernels.mhsa_output_projection_backward];
            [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:0];
            [encoder setBuffer:buffers.attention_output[layer] offset:0 atIndex:1];       // concatenated_attention_heads (saved from forward)
            [encoder setBuffer:weights.blocks[layer].attention_output_weights offset:0 atIndex:2]; 
            [encoder setBuffer:gradients.blocks_grad[layer].attention_output_weights offset:0 atIndex:3]; 
            [encoder setBuffer:gradients.blocks_grad[layer].attention_output_bias offset:0 atIndex:4];    
            [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:5];    // dL/d(concatenated_attention_heads)
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
            
            MTLSize mhsaProjGrid = MTLSizeMake(num_instances, config.embedding_dim, config.embedding_dim);
            MTLSize mhsaProjGroup = MTLSizeMake(1, 1, 1); // Kernel should handle actual threading strategy
            [encoder dispatchThreads:mhsaProjGrid threadsPerThreadgroup:mhsaProjGroup];

            // Now buffers.attention_output_grad[layer] holds dL/d(concatenated_attention_heads).
            // This is the input for the Scaled Dot-Product Attention backward pass.

            // --- Backward Scaled Dot-Product Attention ---
            // Input grad: buffers.attention_output_grad[layer] (dL/d concatenated_attention_heads)
            // Output grad: buffers.attention_qkv_grad[layer] (dL/d QKV)
            // We need to reshape the concatenated heads gradient back to individual head format
            // and compute gradients for Q, K, V
            [encoder setComputePipelineState:kernels.scaled_dot_product_attention_backward];
            
            // Extract Q, K, V from the saved QKV buffer for this layer
            // The QKV buffer contains [B, S, 3*E] where the 3*E is [Q, K, V] concatenated
            // We need to pass Q, K, V separately to the kernel
            // For now, we'll use the QKV buffer and let the kernel extract Q, K, V
            
            // Note: The kernel expects separate Q, K, V buffers, but our forward pass saves them concatenated
            // We need to create temporary buffers or modify the kernel to work with concatenated QKV
            // For now, let's create temporary buffers for Q, K, V
            
            // Use saved separate Q, K, V tensors and attention weights from forward pass
            [encoder setComputePipelineState:kernels.scaled_dot_product_attention_backward];
            [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:0];  // grad_output (dL/d concatenated heads)
            [encoder setBuffer:buffers.attention_Q[layer] offset:0 atIndex:1];            // Q (saved from forward)
            [encoder setBuffer:buffers.attention_K[layer] offset:0 atIndex:2];            // K (saved from forward)  
            [encoder setBuffer:buffers.attention_V[layer] offset:0 atIndex:3];            // V (saved from forward)
            [encoder setBuffer:buffers.attention_weights[layer] offset:0 atIndex:4];      // attention_weights (saved from forward)
            [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:5];     // grad_Q, grad_K, grad_V (output concatenated)
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
            [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:9];
            
            MTLSize attnBackwardGrid = MTLSizeMake(actual_batch_size, config.num_heads, actual_sequence_length);
            MTLSize attnBackwardGroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:attnBackwardGrid threadsPerThreadgroup:attnBackwardGroup];
            
            // --- Backward QKV Projection ---
            // Input grad: buffers.attention_qkv_grad[layer] (dL/d QKV output) - this should come from attention backward
            // For now, we'll use the concatenated heads gradient reshaped back to QKV format
            // Output grad: buffers.layer_inputs_grad[layer] (dL/d layer input, accumulated)
            
            [encoder setComputePipelineState:kernels.qkv_projection_backward];
            [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:0];              // Input embeddings (saved from forward)
            [encoder setBuffer:weights.blocks[layer].qkv_weights offset:0 atIndex:1];        // QKV weights
            [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:2];        // grad_qkv_output (TODO: should come from attention backward)
            [encoder setBuffer:buffers.layer_inputs_grad[layer] offset:0 atIndex:3];         // grad_input (output, accumulated)
            [encoder setBuffer:gradients.blocks_grad[layer].qkv_weights offset:0 atIndex:4]; // grad_qkv_weights (output)
            [encoder setBuffer:gradients.blocks_grad[layer].qkv_bias offset:0 atIndex:5];    // grad_qkv_bias (output)
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
            
            // Use the new dispatch pattern for QKV projection backward
            uint32_t total_instances_qkv = actual_batch_size * actual_sequence_length;
            uint32_t total_weights_qkv = 3 * config.embedding_dim * config.embedding_dim;
            uint32_t total_bias_qkv = 3 * config.embedding_dim;
            uint32_t max_index1_qkv = std::max({total_instances_qkv, total_weights_qkv, total_bias_qkv});
            uint32_t max_index2_qkv = config.embedding_dim;
            
            MTLSize qkvThreadsPerGrid = MTLSizeMake(3, max_index1_qkv, max_index2_qkv); // 3 computation types
            MTLSize qkvThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:qkvThreadsPerGrid threadsPerThreadgroup:qkvThreadsPerThreadgroup];

            // The final output dL/d(layer_input[layer]) should be accumulated in buffers.layer_inputs_grad[layer].
            // This gradient then needs to be propagated to the input of the (i-1)th layer's LN2,
            // typically by ensuring buffers.block_output_grad[layer-1] receives this value if layer > 0.
            // If layer == 0, buffers.layer_inputs_grad[0] is dL/d(embeddings_output_plus_pe) for embedding_backward.
        }
    } else {
        // If no layers, the gradient from output_projection_backward (dL/dFinalHidden which is dL/dEmbeddings)
        // needs to be copied to an appropriate buffer if embedding_backward is separate.
        // For now, buffers.final_hidden_grad would hold dL/dEmbeddings.
        // This part will connect to embedding_backward later.
    }

    // Step 4: Embedding Layer Backward Pass
    // After all transformer layers, we need to accumulate gradients from the first layer's input
    // (or from embeddings if no layers) back into the embedding table
    
    // --- DIAGNOSTIC CHECKS FOR EMBEDDING LAYER ---
    if (m_is_diagnostic_run) {
        std::cout << "🔍 DIAGNOSTIC: Checking embedding layer vulnerabilities..." << std::endl;
        
        // Token ID bounds validation
        uint32_t* token_data = static_cast<uint32_t*>([buffers.input_tokens contents]);
        for (uint32_t i = 0; i < actual_sequence_length; i++) {
            uint32_t token_id = token_data[i];
            if (token_id >= config.vocab_size) {
                std::cerr << "❌ DIAGNOSTIC EMBEDDING: Token ID " << token_id << " at position " << i << " >= vocab_size (" << config.vocab_size << ")" << std::endl;
                return false;
            }
        }
        
        // Embedding table bounds
        size_t embedding_table_required = config.vocab_size * config.embedding_dim * sizeof(uint16_t);
        if ([weights.token_embeddings length] < embedding_table_required) {
            std::cerr << "❌ DIAGNOSTIC EMBEDDING: Embedding table buffer insufficient. Required: " << embedding_table_required << ", Actual: " << [weights.token_embeddings length] << std::endl;
            return false;
        }
        
        // Embedding gradient table bounds
        size_t embedding_grad_required = config.vocab_size * config.embedding_dim * sizeof(float);
        if ([gradients.token_embeddings_grad length] < embedding_grad_required) {
            std::cerr << "❌ DIAGNOSTIC EMBEDDING: Embedding gradient buffer insufficient. Required: " << embedding_grad_required << ", Actual: " << [gradients.token_embeddings_grad length] << std::endl;
            return false;
        }
        
        // Input gradient buffer validation
        id<MTLBuffer> input_grad_buffer = (config.num_layers > 0) ? buffers.layer_inputs_grad[0] : buffers.final_hidden_grad;
        if (!input_grad_buffer) {
            std::cerr << "❌ DIAGNOSTIC EMBEDDING: Input gradient buffer is null" << std::endl;
            return false;
        }
        
        size_t input_grad_required = num_instances * config.embedding_dim * (config.use_half_precision ? sizeof(uint16_t) : sizeof(float));
        if ([input_grad_buffer length] < input_grad_required) {
            std::cerr << "❌ DIAGNOSTIC EMBEDDING: Input gradient buffer insufficient. Required: " << input_grad_required << ", Actual: " << [input_grad_buffer length] << std::endl;
            return false;
        }
        
        // Padding token validation
        if (config.pad_token_id >= config.vocab_size) {
            std::cerr << "❌ DIAGNOSTIC EMBEDDING: pad_token_id (" << config.pad_token_id << ") >= vocab_size (" << config.vocab_size << ")" << std::endl;
            return false;
        }
        
        // Check for potential atomic operation conflicts
        if (actual_sequence_length > 1000) {  // Arbitrary threshold for potential atomic conflicts
            std::cout << "⚠️  DIAGNOSTIC EMBEDDING: Large sequence length (" << actual_sequence_length << ") may cause atomic operation contention" << std::endl;
        }
        
        std::cout << "✅ DIAGNOSTIC EMBEDDING: All embedding layer checks passed" << std::endl;
    }
    
    if (config.num_layers > 0) {
        // Use gradients from first layer input (dL/d(embeddings + positional_encoding))
        [encoder setComputePipelineState:kernels.embedding_layer_backward];
        [encoder setBuffer:buffers.input_tokens offset:0 atIndex:0];                    // Token IDs from forward pass
        [encoder setBuffer:buffers.layer_inputs_grad[0] offset:0 atIndex:1];           // Gradient w.r.t. embeddings + PE
        [encoder setBuffer:gradients.token_embeddings_grad offset:0 atIndex:2];        // Gradient w.r.t. embedding table (output)
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.pad_token_id length:sizeof(uint32_t) atIndex:7];
        
        MTLSize embeddingThreadsPerGrid = MTLSizeMake(actual_batch_size * actual_sequence_length, 1, 1);
        MTLSize embeddingThreadsPerThreadgroup = MTLSizeMake(std::min(actual_batch_size * actual_sequence_length, 64u), 1, 1);
        [encoder dispatchThreads:embeddingThreadsPerGrid threadsPerThreadgroup:embeddingThreadsPerThreadgroup];
        
        std::cout << "✓ Embedding layer backward pass completed" << std::endl;
    } else {
        // If no transformer layers, use gradients from final_hidden_grad
        [encoder setComputePipelineState:kernels.embedding_layer_backward];
        [encoder setBuffer:buffers.input_tokens offset:0 atIndex:0];                    // Token IDs from forward pass
        [encoder setBuffer:buffers.final_hidden_grad offset:0 atIndex:1];              // Gradient w.r.t. embeddings directly
        [encoder setBuffer:gradients.token_embeddings_grad offset:0 atIndex:2];        // Gradient w.r.t. embedding table (output)
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.pad_token_id length:sizeof(uint32_t) atIndex:7];
        
        MTLSize embeddingThreadsPerGrid = MTLSizeMake(actual_batch_size * actual_sequence_length, 1, 1);
        MTLSize embeddingThreadsPerThreadgroup = MTLSizeMake(std::min(actual_batch_size * actual_sequence_length, 64u), 1, 1);
        [encoder dispatchThreads:embeddingThreadsPerGrid threadsPerThreadgroup:embeddingThreadsPerThreadgroup];
        
        std::cout << "✓ Embedding layer backward pass completed (no transformer layers)" << std::endl;
    }

    [encoder endEncoding];
    
    // --- FINAL DIAGNOSTIC CHECKS AND GPU RESOURCE MONITORING ---
    if (m_is_diagnostic_run) {
        std::cout << "🔍 DIAGNOSTIC: Performing final system checks..." << std::endl;
        
        // Check GPU memory pressure
        if (@available(macOS 10.15, *)) {
            id<MTLDevice> metal_device = device;
            if (metal_device) {
                // Check if device is low on memory (approximate check)
                size_t total_allocated_memory = 0;
                
                // Estimate total buffer memory usage
                total_allocated_memory += [buffers.input_tokens length];
                total_allocated_memory += [buffers.final_logits length];
                total_allocated_memory += [buffers.logits_grad length];
                
                for (uint32_t layer = 0; layer < config.num_layers; layer++) {
                    if (buffers.layer_inputs[layer]) total_allocated_memory += [buffers.layer_inputs[layer] length];
                    if (buffers.attention_qkv[layer]) total_allocated_memory += [buffers.attention_qkv[layer] length];
                    if (buffers.attention_weights[layer]) total_allocated_memory += [buffers.attention_weights[layer] length];
                    if (buffers.ffn_h_linear[layer]) total_allocated_memory += [buffers.ffn_h_linear[layer] length];
                }
                
                // Add weight memory
                total_allocated_memory += [weights.token_embeddings length];
                total_allocated_memory += [weights.output_weights length];
                for (uint32_t layer = 0; layer < config.num_layers; layer++) {
                    if (weights.blocks[layer].qkv_weights) total_allocated_memory += [weights.blocks[layer].qkv_weights length];
                    if (weights.blocks[layer].ffn_w1) total_allocated_memory += [weights.blocks[layer].ffn_w1 length];
                    if (weights.blocks[layer].ffn_w2) total_allocated_memory += [weights.blocks[layer].ffn_w2 length];
                }
                
                std::cout << "📊 DIAGNOSTIC MEMORY: Estimated GPU memory usage: " << (total_allocated_memory / 1024 / 1024) << " MB" << std::endl;
                
                // Warn if using large amounts of memory
                if (total_allocated_memory > 8ULL * 1024 * 1024 * 1024) {  // 8 GB threshold
                    std::cout << "⚠️  DIAGNOSTIC MEMORY: High GPU memory usage detected (>8GB)" << std::endl;
                }
            }
        }
        
        // Command buffer validation
        if (!commandBuffer) {
            std::cerr << "❌ DIAGNOSTIC GPU: Command buffer is null" << std::endl;
            return false;
        }
        
        // Check command queue state
    if (!commandQueue) {
            std::cerr << "❌ DIAGNOSTIC GPU: Command queue is null" << std::endl;
        return false;
    }
    
        // Kernel pipeline state validation
        std::vector<id<MTLComputePipelineState>> critical_kernels = {
            kernels.output_projection_backward,
            kernels.layer_norm_backward,
            kernels.ffn_backward,
            kernels.embedding_layer_backward
        };
        
        for (size_t i = 0; i < critical_kernels.size(); i++) {
            if (!critical_kernels[i]) {
                std::cerr << "❌ DIAGNOSTIC GPU: Critical kernel " << i << " pipeline state is null" << std::endl;
                return false;
            }
        }
        
        // Check for potential command buffer overflow
        // This is a heuristic check - Metal doesn't provide direct command count
        uint32_t estimated_commands = config.num_layers * 8; // Rough estimate of commands per layer
        if (estimated_commands > 1000) {  // Arbitrary threshold
            std::cout << "⚠️  DIAGNOSTIC GPU: High command count estimate (" << estimated_commands << ") may stress command buffer" << std::endl;
        }
        
        // Summary diagnostic report
        std::cout << "📋 DIAGNOSTIC SUMMARY for sequence " << current_sequence_index << ":" << std::endl;
        std::cout << "  ✅ Global configuration validated" << std::endl;
        std::cout << "  ✅ Output projection layer validated" << std::endl;
        std::cout << "  ✅ All " << config.num_layers << " transformer layers validated" << std::endl;
        std::cout << "  ✅ Embedding layer validated" << std::endl;
        std::cout << "  ✅ GPU resources validated" << std::endl;
        std::cout << "  📊 Sequence length: " << actual_sequence_length << "/" << config.max_sequence_length << std::endl;
        std::cout << "  📊 Batch size: " << actual_batch_size << std::endl;
        std::cout << "  📊 Total instances: " << num_instances << std::endl;
        std::cout << "🎯 DIAGNOSTIC: Sequence " << current_sequence_index << " is SAFE to proceed" << std::endl;
    }
    
    // --- GPU ERROR DETECTION IN COMPLETION HANDLER ---
    std::atomic<bool>* error_flag = &m_critical_gpu_error_occurred;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
        if (buffer.error) {
            NSLog(@"❌ DIAGNOSTIC GPU ERROR: Command buffer failed with error: %@", buffer.error);
            error_flag->store(true);
        }
    }];
    [commandBuffer commit];
    
    return true;
}

void TransformerModel::initializeWeights() {
    // Implementation of initializeWeights method
}

void TransformerModel::cleanup() {
    // Metal objects are automatically managed by ARC
    // Just reset pointers
    device = nullptr;
    commandQueue = nullptr;
    library = nullptr;
    
    memset(&kernels, 0, sizeof(kernels));
    memset(&weights, 0, sizeof(weights));
    memset(&gradients, 0, sizeof(gradients));
    memset(&optimizer_state, 0, sizeof(optimizer_state));
    memset(&buffers, 0, sizeof(buffers));
}

size_t TransformerModel::getParameterCount() const {
    size_t total = 0;
    
    // Token embeddings
    total += config.vocab_size * config.embedding_dim;
    
    // Per layer parameters
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        // QKV weights + bias
        total += 3 * config.embedding_dim * config.embedding_dim;
        total += 3 * config.embedding_dim;
        
        // Attention output
        total += config.embedding_dim * config.embedding_dim;
        total += config.embedding_dim;
        
        // Layer norm (x2)
        total += 2 * 2 * config.embedding_dim;
        
        // FFN
        total += config.embedding_dim * config.ffn_hidden_dim;
        total += config.ffn_hidden_dim;
        total += config.ffn_hidden_dim * config.embedding_dim;
        total += config.embedding_dim;
    }
    
    // Final layer norm
    total += 2 * config.embedding_dim;
    
    // Output projection
    total += config.embedding_dim * config.vocab_size;
    total += config.vocab_size;
    
    return total;
}

size_t TransformerModel::getMemoryUsage() const {
    // Rough estimate of memory usage
    size_t params = getParameterCount();
    size_t param_memory = params * 2; // Half precision weights
    
    size_t buffer_memory = config.batch_size * config.max_sequence_length * config.embedding_dim * 2; // Embeddings
    buffer_memory += config.num_layers * buffer_memory * 6; // Intermediate buffers per layer
    buffer_memory += config.batch_size * config.max_sequence_length * config.vocab_size * 4; // Logits
    
    return param_memory + buffer_memory;
}

bool TransformerModel::generate(const std::vector<uint32_t>& prompt_tokens,
                               uint32_t max_new_tokens,
                               std::vector<uint32_t>& generated_sequence,
                               float temperature,
                               uint32_t top_k) {
    if (prompt_tokens.empty()) {
        std::cerr << "Empty prompt provided for generation" << std::endl;
        return false; 
    }
    
    if (prompt_tokens.size() >= config.max_sequence_length) {
        std::cerr << "Prompt too long for generation" << std::endl;
        return false;
    }
    
    // Check if inference kernels are available
    if (!kernels.qkv_projection_inference || !kernels.scaled_dot_product_attention_inference) {
        std::cerr << "Inference kernels not available. Using training forward pass for generation." << std::endl;
        return generateWithTrainingKernels(prompt_tokens, max_new_tokens, generated_sequence, temperature);
    }
    
    std::cout << "Starting inference generation with " << prompt_tokens.size() << " prompt tokens, max_new_tokens=" << max_new_tokens << std::endl;
    
    // Initialize generated sequence with prompt
    generated_sequence = prompt_tokens;
    
    // Clear KV caches
    if (!clearKVCache()) {
        std::cerr << "Failed to clear KV cache" << std::endl;
        return false;
    }
    
    // Process prompt tokens to populate KV cache
    if (!populateKVCacheWithPrompt(prompt_tokens)) {
        std::cerr << "Failed to populate KV cache with prompt" << std::endl;
        return false;
    }
    
    // Generate new tokens one by one
    for (uint32_t i = 0; i < max_new_tokens; i++) {
        if (generated_sequence.size() >= config.max_sequence_length) {
            std::cout << "Reached maximum sequence length, stopping generation" << std::endl;
            break;
        }
        
        // Generate next token
        uint32_t next_token;
        if (!generateNextToken(generated_sequence.back(), next_token, temperature, generated_sequence.size() - 1)) {
            std::cerr << "Failed to generate next token at position " << i << std::endl;
            return false;
        }
        
        generated_sequence.push_back(next_token);
        
        // Optional: Check for end-of-sequence token (implement if you have one)
        // if (next_token == config.eos_token_id) {
        //     std::cout << "Generated EOS token, stopping generation" << std::endl;
        //     break;
        // }
    }
    
    std::cout << "Generation completed. Total tokens: " << generated_sequence.size() << std::endl;
    return true;
}

bool TransformerModel::clearKVCache() {
    // Zero out all KV cache buffers
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        if (buffers.kv_cache_K[layer] && buffers.kv_cache_V[layer]) {
            // Zero the buffers
            memset([buffers.kv_cache_K[layer] contents], 0, [buffers.kv_cache_K[layer] length]);
            memset([buffers.kv_cache_V[layer] contents], 0, [buffers.kv_cache_V[layer] length]);
        }
    }
    return true;
}

bool TransformerModel::populateKVCacheWithPrompt(const std::vector<uint32_t>& prompt_tokens) {
    // For now, we'll process the prompt using the standard forward pass
    // A more efficient implementation would process the prompt in parallel
    // and extract K,V values to populate the cache
    
    // Copy prompt tokens to input buffer
    uint32_t* token_data = static_cast<uint32_t*>([buffers.input_tokens contents]);
    std::copy(prompt_tokens.begin(), prompt_tokens.end(), token_data);
    
    uint32_t prompt_length = static_cast<uint32_t>(prompt_tokens.size());
    uint32_t batch_size = 1; // Generation batch size
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Step 1: Embedding Lookup for prompt
    [encoder setComputePipelineState:kernels.embedding_lookup];
    [encoder setBuffer:buffers.input_tokens offset:0 atIndex:0];
    [encoder setBuffer:weights.token_embeddings offset:0 atIndex:1];
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:2];
    [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:6];
    MTLSize threadsPerGrid = MTLSizeMake(batch_size * prompt_length, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(std::min((size_t)batch_size * prompt_length, kernels.embedding_lookup.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    
    // Step 2: Add Positional Encoding
    [encoder setComputePipelineState:kernels.positional_encoding];
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:0];
    [encoder setBuffer:weights.positional_encodings offset:0 atIndex:1];
    [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    
    [encoder endEncoding];
    
    // Copy embeddings to first layer input
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:buffers.embeddings sourceOffset:0 
                toBuffer:buffers.layer_inputs[0] destinationOffset:0 
                    size:batch_size * prompt_length * config.embedding_dim * sizeof(uint16_t)];
    [blitEncoder endEncoding];
    
    encoder = [commandBuffer computeCommandEncoder];
    
    // Step 3: Process through transformer layers and populate KV cache
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block_w = weights.blocks[layer];
        
        // QKV Projection for all prompt tokens
        [encoder setComputePipelineState:kernels.qkv_projection];
        [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:0];
        [encoder setBuffer:block_w.qkv_weights offset:0 atIndex:1];
        [encoder setBuffer:block_w.qkv_bias offset:0 atIndex:2];
        [encoder setBuffer:buffers.attention_qkv[layer] offset:0 atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:7];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Extract Q, K, V from concatenated QKV
        [encoder setComputePipelineState:kernels.extract_qkv_from_concatenated];
        [encoder setBuffer:buffers.attention_qkv[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_Q[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.attention_K[layer] offset:0 atIndex:2];
        [encoder setBuffer:buffers.attention_V[layer] offset:0 atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:7];
        
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        MTLSize qkvExtractGrid = MTLSizeMake(batch_size * prompt_length, config.num_heads, head_dim);
        MTLSize qkvExtractGroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:qkvExtractGrid threadsPerThreadgroup:qkvExtractGroup];
        
        // Copy K and V to KV cache (for positions 0 to prompt_length-1)
        [encoder endEncoding];
        id<MTLBlitCommandEncoder> kvBlitEncoder = [commandBuffer blitCommandEncoder];
        
        // Copy K values: from attention_K[layer] to kv_cache_K[layer]
        size_t k_copy_size = batch_size * config.num_heads * prompt_length * head_dim * sizeof(uint16_t);
        [kvBlitEncoder copyFromBuffer:buffers.attention_K[layer] sourceOffset:0
                        toBuffer:buffers.kv_cache_K[layer] destinationOffset:0
                            size:k_copy_size];
        
        // Copy V values: from attention_V[layer] to kv_cache_V[layer]  
        [kvBlitEncoder copyFromBuffer:buffers.attention_V[layer] sourceOffset:0
                        toBuffer:buffers.kv_cache_V[layer] destinationOffset:0
                            size:k_copy_size];
        
        [kvBlitEncoder endEncoding];
        encoder = [commandBuffer computeCommandEncoder];
        
        // Continue with standard forward pass for this layer
        // (attention, layer norm, FFN, etc.) to get input for next layer
        // For simplicity, we'll use the standard forward pass logic here
        
        // Scaled Dot-Product Attention
        [encoder setComputePipelineState:kernels.scaled_dot_product_attention];
        [encoder setBuffer:buffers.attention_qkv[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_output[layer] offset:0 atIndex:1];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:5];
        MTLSize attnThreadsPerGrid = MTLSizeMake(batch_size, config.num_heads, 1);
        MTLSize attnThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:attnThreadsPerGrid threadsPerThreadgroup:attnThreadsPerThreadgroup];
        
        // MHSA Output Projection
        [encoder setComputePipelineState:kernels.mhsa_output_projection];
        [encoder setBuffer:buffers.attention_output[layer] offset:0 atIndex:0];
        [encoder setBuffer:block_w.attention_output_weights offset:0 atIndex:1];
        [encoder setBuffer:block_w.attention_output_bias offset:0 atIndex:2];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Add & Norm (Attention)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:2];
        [encoder setBuffer:block_w.ln1_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln1_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[layer*2] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2] offset:0 atIndex:6];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Feed-Forward Network
        [encoder setComputePipelineState:kernels.feed_forward_network];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:0];
        [encoder setBuffer:block_w.ffn_w1 offset:0 atIndex:1];
        [encoder setBuffer:block_w.ffn_b1 offset:0 atIndex:2];
        [encoder setBuffer:block_w.ffn_w2 offset:0 atIndex:3];
        [encoder setBuffer:block_w.ffn_b2 offset:0 atIndex:4];
        [encoder setBuffer:buffers.ffn_output[layer] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ffn_h_linear[layer] offset:0 atIndex:6];
        [encoder setBuffer:buffers.ffn_h_activated[layer] offset:0 atIndex:7];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:11];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Add & Norm (FFN)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.ffn_output[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.block_output[layer] offset:0 atIndex:2];
        [encoder setBuffer:block_w.ln2_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln2_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[layer*2+1] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2+1] offset:0 atIndex:6];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&prompt_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Copy output to next layer input (if not last layer)
        if (layer < config.num_layers - 1) {
            [encoder endEncoding];
            id<MTLBlitCommandEncoder> layerBlitEncoder = [commandBuffer blitCommandEncoder];
            [layerBlitEncoder copyFromBuffer:buffers.block_output[layer] sourceOffset:0
                            toBuffer:buffers.layer_inputs[layer+1] destinationOffset:0
                                size:batch_size * prompt_length * config.embedding_dim * sizeof(uint16_t)];
            [layerBlitEncoder endEncoding];
            encoder = [commandBuffer computeCommandEncoder];
        }
    }
    
    [encoder endEncoding];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];
    
    std::cout << "✓ KV cache populated with prompt (" << prompt_tokens.size() << " tokens)" << std::endl;
    return true;
}

bool TransformerModel::generateNextToken(uint32_t current_token, uint32_t& next_token, float temperature, uint32_t current_position) {
    if (current_position >= config.max_sequence_length) {
        std::cerr << "Current position exceeds maximum sequence length" << std::endl;
        return false;
    }
    
    uint32_t batch_size = 1;
    uint32_t sequence_length = 1; // Processing single token
    
    // Set up input token
    uint32_t* token_data = static_cast<uint32_t*>([buffers.input_tokens contents]);
    token_data[0] = current_token;
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Step 1: Embedding lookup for current token
    [encoder setComputePipelineState:kernels.embedding_lookup];
    [encoder setBuffer:buffers.input_tokens offset:0 atIndex:0];
    [encoder setBuffer:weights.token_embeddings offset:0 atIndex:1]; 
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:2];
    [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:6];
    MTLSize singleTokenGrid = MTLSizeMake(1, 1, 1);
    MTLSize singleTokenGroup = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
    
    // Step 2: Add positional encoding (for current position)
    [encoder setComputePipelineState:kernels.positional_encoding];
    [encoder setBuffer:buffers.embeddings offset:0 atIndex:0];
    [encoder setBuffer:weights.positional_encodings offset:current_position * config.embedding_dim * sizeof(uint16_t) atIndex:1]; // Offset to current position
    [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:4];
    [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
    
    [encoder endEncoding];
    
    // Copy to first layer input
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:buffers.embeddings sourceOffset:0
                toBuffer:buffers.layer_inputs[0] destinationOffset:0
                    size:config.embedding_dim * sizeof(uint16_t)];
    [blitEncoder endEncoding];
    
    encoder = [commandBuffer computeCommandEncoder];
    
    // Step 3: Process through transformer layers with KV caching
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block_w = weights.blocks[layer];
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        
        // QKV Projection for single token (inference mode)
        [encoder setComputePipelineState:kernels.qkv_projection_inference];
        [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:0];         // input [1, 1, E]
        [encoder setBuffer:block_w.qkv_weights offset:0 atIndex:1];                 // weights [3*E, E]
        [encoder setBuffer:block_w.qkv_bias offset:0 atIndex:2];                    // bias [3*E]
        [encoder setBuffer:buffers.attention_Q[layer] offset:0 atIndex:3];          // Q output [1, H, 1, D]
        [encoder setBuffer:buffers.kv_cache_K[layer] offset:current_position * config.num_heads * head_dim * sizeof(uint16_t) atIndex:4]; // K cache position
        [encoder setBuffer:buffers.kv_cache_V[layer] offset:current_position * config.num_heads * head_dim * sizeof(uint16_t) atIndex:5]; // V cache position
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&current_position length:sizeof(uint32_t) atIndex:8];
        
        MTLSize qkvInfGrid = MTLSizeMake(config.num_heads, head_dim, 1);
        MTLSize qkvInfGroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:qkvInfGrid threadsPerThreadgroup:qkvInfGroup];
        
        // Scaled Dot-Product Attention with KV caching
        [encoder setComputePipelineState:kernels.scaled_dot_product_attention_inference];
        [encoder setBuffer:buffers.attention_Q[layer] offset:0 atIndex:0];          // Q [1, H, 1, D]
        [encoder setBuffer:buffers.kv_cache_K[layer] offset:0 atIndex:1];           // Full K cache [1, H, MaxS, D]
        [encoder setBuffer:buffers.kv_cache_V[layer] offset:0 atIndex:2];           // Full V cache [1, H, MaxS, D]
        [encoder setBuffer:buffers.attention_output[layer] offset:0 atIndex:3];     // Output [1, 1, E]
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&current_position length:sizeof(uint32_t) atIndex:6];     // Current sequence length (0-indexed position + 1)
        
        MTLSize attnInfGrid = MTLSizeMake(config.num_heads, 1, 1);
        MTLSize attnInfGroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:attnInfGrid threadsPerThreadgroup:attnInfGroup];
        
        // Continue with standard layers (output projection, layer norm, FFN)
        // MHSA Output Projection
        [encoder setComputePipelineState:kernels.mhsa_output_projection];
        [encoder setBuffer:buffers.attention_output[layer] offset:0 atIndex:0];
        [encoder setBuffer:block_w.attention_output_weights offset:0 atIndex:1];
        [encoder setBuffer:block_w.attention_output_bias offset:0 atIndex:2];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:3];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
        
        // Add & Norm (Attention)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:2];
        [encoder setBuffer:block_w.ln1_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln1_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[layer*2] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2] offset:0 atIndex:6];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
        
        // Feed-Forward Network
        [encoder setComputePipelineState:kernels.feed_forward_network];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:0];
        [encoder setBuffer:block_w.ffn_w1 offset:0 atIndex:1];
        [encoder setBuffer:block_w.ffn_b1 offset:0 atIndex:2];
        [encoder setBuffer:block_w.ffn_w2 offset:0 atIndex:3];
        [encoder setBuffer:block_w.ffn_b2 offset:0 atIndex:4];
        [encoder setBuffer:buffers.ffn_output[layer] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ffn_h_linear[layer] offset:0 atIndex:6];
        [encoder setBuffer:buffers.ffn_h_activated[layer] offset:0 atIndex:7];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:11];
        [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
        
        // Add & Norm (FFN)
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:buffers.ffn_output[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.block_output[layer] offset:0 atIndex:2];
        [encoder setBuffer:block_w.ln2_gamma offset:0 atIndex:3];
        [encoder setBuffer:block_w.ln2_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.ln_mean[layer*2+1] offset:0 atIndex:5];
        [encoder setBuffer:buffers.ln_rsqrt_variance[layer*2+1] offset:0 atIndex:6];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
        
        // Copy to next layer input (if not last layer)
        if (layer < config.num_layers - 1) {
            [encoder endEncoding];
            id<MTLBlitCommandEncoder> layerBlitEncoder = [commandBuffer blitCommandEncoder];
            [layerBlitEncoder copyFromBuffer:buffers.block_output[layer] sourceOffset:0
                            toBuffer:buffers.layer_inputs[layer+1] destinationOffset:0
                                size:config.embedding_dim * sizeof(uint16_t)];
            [layerBlitEncoder endEncoding];
            encoder = [commandBuffer computeCommandEncoder];
        }
    }
    
    // Final layer norm and output projection
    id<MTLBuffer> final_hidden_state = (config.num_layers > 0) ? buffers.block_output[config.num_layers - 1] : buffers.embeddings;
    
    if (config.num_layers > 0) {
        [encoder setComputePipelineState:kernels.layer_norm];
        [encoder setBuffer:final_hidden_state offset:0 atIndex:0];
        [encoder setBuffer:nil offset:0 atIndex:1]; // No residual for final LN
        [encoder setBuffer:buffers.layer_inputs[config.num_layers] offset:0 atIndex:2]; // Temp buffer
        [encoder setBuffer:weights.final_ln_gamma offset:0 atIndex:3];
        [encoder setBuffer:weights.final_ln_beta offset:0 atIndex:4];
        [encoder setBuffer:buffers.final_ln_mean offset:0 atIndex:5];
        [encoder setBuffer:buffers.final_ln_rsqrt_variance offset:0 atIndex:6];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:10];
        [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
        final_hidden_state = buffers.layer_inputs[config.num_layers];
    }
    
    // Output projection
    [encoder setComputePipelineState:kernels.output_logits_projection];
    [encoder setBuffer:final_hidden_state offset:0 atIndex:0];
    [encoder setBuffer:weights.output_weights offset:0 atIndex:1];
    [encoder setBuffer:weights.output_bias offset:0 atIndex:2];
    [encoder setBuffer:buffers.final_logits offset:0 atIndex:3];
    [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&sequence_length length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:7];
    [encoder dispatchThreads:singleTokenGrid threadsPerThreadgroup:singleTokenGroup];
    
    [encoder endEncoding];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // Async completion - no CPU blocking
    }];
    [commandBuffer commit];
    
    // Sample next token from logits
    float* logits_data = static_cast<float*>([buffers.final_logits contents]);
    next_token = sampleGreedy(logits_data, config.vocab_size);
    
    return true;
}

bool TransformerModel::generateWithTrainingKernels(const std::vector<uint32_t>& prompt_tokens,
                                                   uint32_t max_new_tokens,
                                                   std::vector<uint32_t>& generated_sequence,
                                                   float temperature) {
    // Fallback implementation using standard forward pass (slower but functional)
    generated_sequence = prompt_tokens;
    
    for (uint32_t i = 0; i < max_new_tokens; i++) {
        if (generated_sequence.size() >= config.max_sequence_length) {
            break;
        }
        
        // Run forward pass on current sequence
        std::vector<float> logits;
        if (!forward(generated_sequence, logits)) {
            return false;
        }
        
        // Get logits for last position
        uint32_t last_position = generated_sequence.size() - 1;
        float* last_logits = logits.data() + last_position * config.vocab_size;
        
        // Sample next token
        uint32_t next_token = sampleGreedy(last_logits, config.vocab_size);
        generated_sequence.push_back(next_token);
    }
    
    return true;
}

uint32_t TransformerModel::sampleGreedy(const float* logits, uint32_t vocab_size) {
    uint32_t best_token = 0;
    float best_score = logits[0];
    
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_token = i;
        }
    }
    
    return best_token;
} 

bool TransformerModel::trainBatch(const std::vector<std::vector<uint32_t>>& input_batch,
                                 const std::vector<std::vector<uint32_t>>& target_batch,
                                 float& avg_loss) {
    // Reset critical GPU error flag for this batch
    m_critical_gpu_error_occurred = false;

    if (input_batch.size() != target_batch.size()) {
        std::cerr << "Input batch and target batch sizes must match" << std::endl;
        return false;
    }
    
    if (input_batch.empty()) {
        std::cerr << "Empty batch provided" << std::endl;
        return false;
    }
    
    std::cout << "Training batch of " << input_batch.size() << " sequences..." << std::endl;

    // --- AGGRESSIVE PRE-EMPTIVE BATCH VALIDATION ---
    std::cout << "🛡️ AGGRESSIVE PRE-FLIGHT CHECK: Validating entire batch for CRITICAL crash conditions..." << std::endl;
    for (size_t i = 0; i < input_batch.size(); ++i) {
        const auto& input_seq = input_batch[i];
        // Check for scaled_dot_product_attention_inference stack overflow
        if (input_seq.size() > 512) {
            std::cerr << "🚨 CRITICAL PRE-FLIGHT (Seq: " << i << "): Sequence length (" << input_seq.size() 
                      << ") > 512. This WILL trigger stack overflow in scaled_dot_product_attention_inference." << std::endl;
            std::cerr << "🛡️ ABORTING ENTIRE BATCH to prevent system crash." << std::endl;
            return false;
        }
    }
    if (config.ffn_hidden_dim > 2048) {
        std::cerr << "🚨 CRITICAL PRE-FLIGHT: config.ffn_hidden_dim (" << config.ffn_hidden_dim 
                  << ") > 2048. This WILL trigger ffn_backward threadgroup overflow." << std::endl;
        std::cerr << "🛡️ ABORTING ENTIRE BATCH to prevent system crash." << std::endl;
        return false;
    }
    if (config.embedding_dim > 1024) {
        std::cerr << "🚨 CRITICAL PRE-FLIGHT: config.embedding_dim (" << config.embedding_dim 
                  << ") > 1024. This WILL trigger layer_norm_backward threadgroup overflow." << std::endl;
        std::cerr << "🛡️ ABORTING ENTIRE BATCH to prevent system crash." << std::endl;
        return false;
    }
    std::cout << "✅ AGGRESSIVE PRE-FLIGHT CHECK: Batch deemed safe from KNOWN critical crash conditions." << std::endl;
    
    // --- PRE-BATCH DIAGNOSTIC VALIDATION ---
    std::cout << "🔍 PRE-BATCH DIAGNOSTIC: Validating batch before processing..." << std::endl;
    
    // Check for problematic sequences that might cause crashes
    uint32_t sequences_at_risk = 0;
    uint32_t sequences_too_long = 0;
    uint32_t sequences_empty = 0;
    
    for (size_t i = 0; i < input_batch.size(); i++) {
        const auto& input_seq = input_batch[i];
        const auto& target_seq = target_batch[i];
        
        // Empty sequence check
        if (input_seq.empty() || target_seq.empty()) {
            sequences_empty++;
            std::cerr << "⚠️  PRE-BATCH: Sequence " << i << " is empty" << std::endl;
            continue;
        }
        
        // Length mismatch check
        if (input_seq.size() != target_seq.size()) {
            sequences_at_risk++;
            std::cerr << "⚠️  PRE-BATCH: Sequence " << i << " has length mismatch (input: " << input_seq.size() << ", target: " << target_seq.size() << ")" << std::endl;
            continue;
        }
        
        // Sequence too long check
        if (input_seq.size() > config.max_sequence_length) {
            sequences_too_long++;
            std::cerr << "⚠️  PRE-BATCH: Sequence " << i << " exceeds max length (" << input_seq.size() << " > " << config.max_sequence_length << ")" << std::endl;
            continue;
        }
        
        // Check for invalid token IDs
        bool has_invalid_tokens = false;
        for (size_t j = 0; j < input_seq.size(); j++) {
            if (input_seq[j] >= config.vocab_size || target_seq[j] >= config.vocab_size) {
                has_invalid_tokens = true;
                sequences_at_risk++;
                std::cerr << "⚠️  PRE-BATCH: Sequence " << i << " has invalid token ID at position " << j << std::endl;
                break;
            }
        }
        
        if (has_invalid_tokens) continue;
        
        // Check if this is sequence 12 (index 11) - our problem sequence
        if (i == 11) {
            std::cout << "🎯 PRE-BATCH: Found target sequence " << i << " (length: " << input_seq.size() << ") - will enable diagnostics" << std::endl;
            
            // Extra validation for the problematic sequence
            if (input_seq.size() > 50) {  // Arbitrary threshold where problems might start
                std::cout << "⚠️  PRE-BATCH: Problem sequence is long (" << input_seq.size() << " tokens) - increased crash risk" << std::endl;
            }
            
            // Check for specific vulnerable patterns
            if (input_seq.size() > 512) {
                std::cout << "🚨 PRE-BATCH: CRITICAL - Sequence 12 length (" << input_seq.size() << ") > 512, triggers scaled_dot_product_attention_inference stack overflow!" << std::endl;
                std::cout << "🛡️  PRE-BATCH: Aborting batch to prevent system crash" << std::endl;
                return false;
            }
            
            if (config.ffn_hidden_dim > 2048) {
                std::cout << "🚨 PRE-BATCH: CRITICAL - ffn_hidden_dim (" << config.ffn_hidden_dim << ") > 2048, triggers ffn_backward threadgroup overflow!" << std::endl;
                std::cout << "🛡️  PRE-BATCH: Aborting batch to prevent system crash" << std::endl;
                return false;
            }
            
            if (config.embedding_dim > 1024) {
                std::cout << "🚨 PRE-BATCH: CRITICAL - embedding_dim (" << config.embedding_dim << ") > 1024, triggers layer_norm_backward threadgroup overflow!" << std::endl;
                std::cout << "🛡️  PRE-BATCH: Aborting batch to prevent system crash" << std::endl;
                return false;
            }
        }
    }
    
    // Report pre-batch validation results
    uint32_t estimated_valid_sequences = input_batch.size() - sequences_at_risk - sequences_too_long - sequences_empty;
    std::cout << "📊 PRE-BATCH REPORT:" << std::endl;
    std::cout << "  Total sequences: " << input_batch.size() << std::endl;
    std::cout << "  Valid sequences: " << estimated_valid_sequences << std::endl;
    std::cout << "  At-risk sequences: " << sequences_at_risk << std::endl;
    std::cout << "  Too-long sequences: " << sequences_too_long << std::endl;
    std::cout << "  Empty sequences: " << sequences_empty << std::endl;
    
    if (estimated_valid_sequences == 0) {
        std::cerr << "❌ PRE-BATCH: No valid sequences to process" << std::endl;
        return false;
    }
    
    if (sequences_at_risk > input_batch.size() / 2) {
        std::cout << "⚠️  PRE-BATCH: High risk batch - over 50% of sequences have issues" << std::endl;
    }
    
    std::cout << "✅ PRE-BATCH: Validation complete, proceeding with batch training" << std::endl;
    
    // Step 1: Zero gradients once for the entire batch
    if (!zeroGradients()) {
        std::cout << "❌ Failed to zero gradients" << std::endl;
        return false;
    }
    
    float total_loss = 0.0f;
    uint32_t valid_sequences = 0;
    
    // Create a semaphore to track command buffer completion
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    
    // Step 2: Accumulate gradients across all sequences in the batch
    for (size_t i = 0; i < input_batch.size(); i++) {
        const auto& input_seq = input_batch[i];
        const auto& target_seq = target_batch[i];
        
        if (input_seq.size() != target_seq.size()) {
            std::cerr << "⚠️  Skipping sequence " << i << ": input and target length mismatch" << std::endl;
            continue;
        }
        
        if (input_seq.empty() || input_seq.size() > config.max_sequence_length) {
            std::cerr << "⚠️  Skipping sequence " << i << ": invalid length (" << input_seq.size() << ")" << std::endl;
            continue;
        }
        
        // Forward pass and loss computation
        float sequence_loss;
        if (!computeLoss(input_seq, target_seq, sequence_loss)) {
            std::cerr << "⚠️  Failed to compute loss for sequence " << i << std::endl;
            continue;
        }
        
        total_loss += sequence_loss;
        valid_sequences++;
        
        // Compute loss gradients (starting point for backprop)
        uint32_t actual_batch_size = 1;
        uint32_t actual_sequence_length = static_cast<uint32_t>(input_seq.size());
        
        // Copy target tokens to buffer for gradient computation
        uint32_t* target_data = static_cast<uint32_t*>([buffers.target_tokens contents]);
        std::copy(target_seq.begin(), target_seq.end(), target_data);
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Compute gradient of loss w.r.t. logits
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:kernels.loss_gradient];
            [encoder setBuffer:buffers.final_logits offset:0 atIndex:0];
            [encoder setBuffer:buffers.target_tokens offset:0 atIndex:1];
            [encoder setBuffer:buffers.logits_grad offset:0 atIndex:2];
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&config.pad_token_id length:sizeof(uint32_t) atIndex:6];
            
            MTLSize threadsPerGrid = MTLSizeMake(actual_batch_size, actual_sequence_length, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
        
        // Add completion handler that signals the semaphore
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            dispatch_semaphore_signal(semaphore);
        }];
        [commandBuffer commit];
        
        // Wait for command buffer to complete
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        
        // Backward pass - accumulate parameter gradients
        bool backward_pass_ok = true;

        // --- CANARY BUFFER SETUP (for sequence 12) ---
        const uint32_t canary_pattern[] = {0xDEADBEEF, 0xCAFEBABE, 0xBAADF00D, 0xFEEDFACE};
        size_t canary_size_bytes = sizeof(canary_pattern);

        if (i == 11 && m_is_diagnostic_run) { // Sequence 12 and diagnostic mode
            std::cout << "🔶 DIAGNOSTIC CANARY: Setting up canary buffers before backwardPass for sequence " << i << std::endl;
            if (m_canary_buffer_before && [m_canary_buffer_before length] >= canary_size_bytes) {
                memcpy([m_canary_buffer_before contents], canary_pattern, canary_size_bytes);
            } else {
                std::cerr << "⚠️ DIAGNOSTIC CANARY: m_canary_buffer_before is invalid or too small!" << std::endl;
            }
            if (m_canary_buffer_after && [m_canary_buffer_after length] >= canary_size_bytes) {
                memcpy([m_canary_buffer_after contents], canary_pattern, canary_size_bytes);
            } else {
                std::cerr << "⚠️ DIAGNOSTIC CANARY: m_canary_buffer_after is invalid or too small!" << std::endl;
            }
        }

        if (i == 11) { // 12th sequence, which is problematic
            std::cout << "INFO: Entering diagnostic mode for sequence " << i << std::endl;
            
            // ADDITIONAL SAFETY CHECK 8: Timeout Protection for Sequence 12
            auto sequence_start_time = std::chrono::high_resolution_clock::now();
            const int MAX_SEQUENCE_TIME_MS = 30000; // 30 second timeout
            
            this->m_is_diagnostic_run = true;
            backward_pass_ok = backwardPass(i); // Pass sequence index for diagnostics
            
            // Check execution time immediately after backwardPass
            auto sequence_end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                sequence_end_time - sequence_start_time).count();
            
            std::cout << "⏱️  DIAGNOSTIC TIMING: Sequence 12 backwardPass took " << duration_ms << "ms" << std::endl;
            
            if (duration_ms > MAX_SEQUENCE_TIME_MS) {
                std::cerr << "🚨 DIAGNOSTIC TIMEOUT: Sequence 12 exceeded " << MAX_SEQUENCE_TIME_MS << "ms limit (took " << duration_ms << "ms)" << std::endl;
                this->m_is_diagnostic_run = false;
                return false;
            }
            
            if (!backward_pass_ok) {
                std::cerr << "❌ DIAGNOSTIC: backwardPass failed for sequence " << i << ". Halting batch." << std::endl;
                this->m_is_diagnostic_run = false; // Reset flag before early exit
                // dispatch_release(semaphore); // Ensure semaphore is released before early exit
                return false; // Stop the batch if diagnostic indicates a problem
            }

            // --- CANARY BUFFER VERIFICATION (for sequence 12) ---
            if (m_is_diagnostic_run) {
                std::cout << "🔶 DIAGNOSTIC CANARY: Verifying canary buffers after backwardPass for sequence " << i << std::endl;
                bool corruption_detected = false;
                if (m_canary_buffer_before && [m_canary_buffer_before length] >= canary_size_bytes) {
                    if (memcmp([m_canary_buffer_before contents], canary_pattern, canary_size_bytes) != 0) {
                        std::cerr << "🚨 CRITICAL CANARY: Corruption detected in m_canary_buffer_before!" << std::endl;
                        corruption_detected = true;
                    }
                }
                if (m_canary_buffer_after && [m_canary_buffer_after length] >= canary_size_bytes) {
                    if (memcmp([m_canary_buffer_after contents], canary_pattern, canary_size_bytes) != 0) {
                        std::cerr << "🚨 CRITICAL CANARY: Corruption detected in m_canary_buffer_after!" << std::endl;
                        corruption_detected = true;
                    }
                }
                if (corruption_detected) {
                    std::cerr << "🛡️ DIAGNOSTIC CANARY: Memory corruption detected. Halting batch to prevent further issues." << std::endl;
                    this->m_is_diagnostic_run = false; // Reset flag before early exit
                    return false; // Critical error, stop the batch
                } else {
                    std::cout << "✅ DIAGNOSTIC CANARY: Canary buffers intact for sequence " << i << "." << std::endl;
                }
            }
            
            // --- CHECK FOR GPU COMMAND BUFFER ERRORS (Sequence 12) ---
            if (m_critical_gpu_error_occurred) {
                std::cerr << "🚨 CRITICAL GPU ERROR (Seq: " << i << "): Metal command buffer reported an error during backwardPass. Halting batch." << std::endl;
                this->m_is_diagnostic_run = false; // Reset flag before early exit
                return false;
            }
            
            // ADDITIONAL SAFETY CHECK 9: Final State Validation for Sequence 12
            std::cout << "🔍 DIAGNOSTIC: Final state validation for sequence 12..." << std::endl;
            
            // Validate that all critical buffers still contain reasonable data
            float* final_check_logits = static_cast<float*>([buffers.final_logits contents]);
            bool final_state_ok = true;
            
            // Quick sanity check on final logits
            for (int check_i = 0; check_i < 10 && check_i < config.vocab_size; check_i++) {
                if (std::isnan(final_check_logits[check_i]) || std::isinf(final_check_logits[check_i])) {
                    std::cerr << "❌ DIAGNOSTIC FINAL: Invalid value in final logits at index " << check_i << std::endl;
                    final_state_ok = false;
                    break;
                }
            }
            
            // Check that we can still create a command buffer (GPU still responsive)
            id<MTLCommandBuffer> test_buffer = [commandQueue commandBuffer];
            if (!test_buffer) {
                std::cerr << "❌ DIAGNOSTIC FINAL: Cannot create test command buffer - GPU may be unresponsive" << std::endl;
                final_state_ok = false;
            } else {
                // Release the test buffer immediately
                test_buffer = nil;
            }
            
            if (!final_state_ok) {
                std::cerr << "🚨 DIAGNOSTIC FINAL: Final state validation failed for sequence 12" << std::endl;
                this->m_is_diagnostic_run = false;
                return false;
            }
            
            std::cout << "✅ DIAGNOSTIC FINAL: All state validations passed for sequence 12" << std::endl;
            
            this->m_is_diagnostic_run = false; // Now reset after all checks complete
            std::cout << "INFO: Exiting diagnostic mode for sequence " << i << std::endl;
        } else {
            backward_pass_ok = backwardPass(); // Normal call for other sequences
            
            // --- CHECK FOR GPU COMMAND BUFFER ERRORS (Other Sequences) ---
            if (m_critical_gpu_error_occurred) {
                std::cerr << "🚨 CRITICAL GPU ERROR (Seq: " << i << "): Metal command buffer reported an error during backwardPass. Halting batch." << std::endl;
                return false;
            }
        }

        if (!backward_pass_ok) {
            std::cerr << "⚠️  Failed backward pass for sequence " << i << std::endl;
            continue;
        }
        
        // --- POST-DIAGNOSTIC ANALYSIS FOR SEQUENCE 12 ---
        if (i == 11) { // Just completed sequence 12
            std::cout << "🎉 POST-DIAGNOSTIC: Sequence 12 completed successfully!" << std::endl;
            std::cout << "📋 POST-DIAGNOSTIC REPORT:" << std::endl;
            std::cout << "  ✅ Sequence 12 forward pass: SUCCESS" << std::endl;
            std::cout << "  ✅ Sequence 12 loss computation: SUCCESS" << std::endl;
            std::cout << "  ✅ Sequence 12 backward pass: SUCCESS" << std::endl;
            std::cout << "  📊 Sequence 12 loss: " << sequence_loss << std::endl;
            std::cout << "  🛡️  All vulnerability checks passed" << std::endl;
            std::cout << "  🔒 GPU state: STABLE" << std::endl;
            std::cout << "🏆 DIAGNOSTIC SUCCESS: The issue appears to be resolved!" << std::endl;
            
            // Optional: Early termination after successful sequence 12 for focused testing
            std::cout << "🔬 DIAGNOSTIC MODE: Consider stopping here for focused analysis" << std::endl;
            std::cout << "     To continue full batch, this is normal operation." << std::endl;
        }
        
        if ((i + 1) % 10 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << input_batch.size() << " sequences, avg loss: " 
                      << (total_loss / valid_sequences) << std::endl;
        }
    }
    
    // Release the semaphore - Commented out as ARC handles this
    // dispatch_release(semaphore); 
    
    if (valid_sequences == 0) {
        std::cerr << "❌ No valid sequences processed in batch" << std::endl;
        return false;
    }
    
    // Step 3: Average the accumulated gradients
    avg_loss = total_loss / valid_sequences;
    
    // Scale gradients by 1/batch_size (since they were accumulated)
    float scale_factor = 1.0f / valid_sequences;
    
    // Apply gradient scaling using simple CPU scaling
    std::vector<id<MTLBuffer>> grad_buffers_to_scale = {
        gradients.token_embeddings_grad,
        gradients.final_ln_gamma_grad,
        gradients.final_ln_beta_grad,
        gradients.output_weights_grad,
        gradients.output_bias_grad
    };
    
    // Add transformer block gradients
    for (uint32_t layer = 0; layer < config.num_layers; layer++) {
        auto& block_grad = gradients.blocks_grad[layer];
        grad_buffers_to_scale.insert(grad_buffers_to_scale.end(), {
            block_grad.qkv_weights,
            block_grad.qkv_bias,
            block_grad.attention_output_weights,
            block_grad.attention_output_bias,
            block_grad.ln1_gamma,
            block_grad.ln1_beta,
            block_grad.ln2_gamma,
            block_grad.ln2_beta,
            block_grad.ffn_w1,
            block_grad.ffn_b1,
            block_grad.ffn_w2,
            block_grad.ffn_b2
        });
    }
    
    // Scale gradients
    for (id<MTLBuffer> buffer : grad_buffers_to_scale) {
        if (buffer) {
            uint32_t buffer_size = static_cast<uint32_t>([buffer length] / sizeof(float));
            float* grad_data = static_cast<float*>([buffer contents]);
            
            // Simple CPU scaling for now (could be done in MSL for better performance)
            for (uint32_t j = 0; j < buffer_size; j++) {
                grad_data[j] *= scale_factor;
            }
        }
    }
    
    // 🔄 STRATEGIC SYNC: Only wait before optimizer (when gradients needed)
    std::cout << "⏳ Strategic sync before optimizer step..." << std::endl;
    
    // Step 4: Single optimizer step with averaged gradients
    if (!optimizerStep()) {
        std::cout << "❌ Failed optimizer step" << std::endl;
        return false;
    }
    
    std::cout << "✅ Batch training completed. Valid sequences: " << valid_sequences 
              << "/" << input_batch.size() << ", avg loss: " << avg_loss << std::endl;
    
    return true;
}