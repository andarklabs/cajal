#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <cassert>

// Helper functions for half-precision conversion
uint16_t floatToHalf(float f) {
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

float halfToFloat(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

// Test dimensions (small for clarity)
const uint32_t BATCH_SIZE = 1;
const uint32_t SEQ_LEN = 2;    // Small sequence for manual verification
const uint32_t EMBEDDING_DIM = 4;
const uint32_t NUM_HEADS = 2;
const uint32_t HEAD_DIM = EMBEDDING_DIM / NUM_HEADS; // 2

// CPU reference implementation for scaled dot-product attention forward pass
void scaled_attention_forward_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& output, std::vector<float>& attention_weights,
    uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    
    float scale = 1.0f / std::sqrt(float(head_dim));
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t h = 0; h < num_heads; h++) {
            for (uint32_t i = 0; i < seq_len; i++) {
                // Find max for numerical stability
                float max_score = -INFINITY;
                for (uint32_t j = 0; j <= i; j++) { // Causal masking
                    float score = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        uint32_t q_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                        uint32_t k_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        score += Q[q_idx] * K[k_idx];
                    }
                    score *= scale;
                    max_score = std::max(max_score, score);
                }
                
                // Compute softmax with causal masking
                float sum_exp = 0.0f;
                for (uint32_t j = 0; j <= i; j++) {
                    float score = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        uint32_t q_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                        uint32_t k_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        score += Q[q_idx] * K[k_idx];
                    }
                    score *= scale;
                    float attention_prob = std::exp(score - max_score);
                    
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    attention_weights[attn_idx] = attention_prob;
                    sum_exp += attention_prob;
                }
                
                // Normalize attention weights
                for (uint32_t j = 0; j <= i; j++) {
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    attention_weights[attn_idx] /= sum_exp;
                }
                
                // Future positions get zero attention (causal masking)
                for (uint32_t j = i + 1; j < seq_len; j++) {
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    attention_weights[attn_idx] = 0.0f;
                }
                
                // Compute output: sum of attention_weight * value
                for (uint32_t d = 0; d < head_dim; d++) {
                    float output_val = 0.0f;
                    for (uint32_t j = 0; j <= i; j++) {
                        uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        uint32_t v_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        output_val += attention_weights[attn_idx] * V[v_idx];
                    }
                    uint32_t out_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                    output[out_idx] = output_val;
                }
            }
        }
    }
}

// CPU reference implementation for scaled dot-product attention backward pass
void scaled_attention_backward_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    const std::vector<float>& attention_weights, const std::vector<float>& grad_output,
    std::vector<float>& grad_Q, std::vector<float>& grad_K, std::vector<float>& grad_V,
    uint32_t batch_size, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) {
    
    float scale = 1.0f / std::sqrt(float(head_dim));
    
    // Initialize gradients to zero
    std::fill(grad_Q.begin(), grad_Q.end(), 0.0f);
    std::fill(grad_K.begin(), grad_K.end(), 0.0f);
    std::fill(grad_V.begin(), grad_V.end(), 0.0f);
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t h = 0; h < num_heads; h++) {
            for (uint32_t i = 0; i < seq_len; i++) {
                
                // Step 1: Gradient through V multiplication
                // grad_attention_weights[i,j] = sum_d(grad_output[i,d] * V[j,d])
                std::vector<float> grad_attention_weights(seq_len, 0.0f);
                for (uint32_t j = 0; j <= i; j++) {
                    for (uint32_t d = 0; d < head_dim; d++) {
                        uint32_t out_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                        uint32_t v_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        grad_attention_weights[j] += grad_output[out_idx] * V[v_idx];
                    }
                }
                
                // grad_V[j,d] += attention_weights[i,j] * grad_output[i,d]
                for (uint32_t j = 0; j <= i; j++) {
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        uint32_t out_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                        uint32_t v_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        grad_V[v_idx] += attention_weights[attn_idx] * grad_output[out_idx];
                    }
                }
                
                // Step 2: Gradient through softmax
                // For softmax: if y = softmax(x), then dy/dx_k = y_k * (grad_y_k - sum_j(y_j * grad_y_j))
                float weighted_sum = 0.0f;
                for (uint32_t j = 0; j <= i; j++) {
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    weighted_sum += attention_weights[attn_idx] * grad_attention_weights[j];
                }
                
                std::vector<float> grad_scores(seq_len, 0.0f);
                for (uint32_t j = 0; j <= i; j++) {
                    uint32_t attn_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    grad_scores[j] = attention_weights[attn_idx] * (grad_attention_weights[j] - weighted_sum);
                }
                
                // Step 3: Gradient through scaled QK^T
                // score[i,j] = scale * sum_d(Q[i,d] * K[j,d])
                // grad_Q[i,d] += scale * sum_j(grad_scores[j] * K[j,d])
                // grad_K[j,d] += scale * sum_i(grad_scores[j] * Q[i,d])  (but only for j <= i due to causal)
                
                for (uint32_t d = 0; d < head_dim; d++) {
                    uint32_t q_idx = ((b * seq_len + i) * num_heads + h) * head_dim + d;
                    for (uint32_t j = 0; j <= i; j++) {
                        uint32_t k_idx = ((b * seq_len + j) * num_heads + h) * head_dim + d;
                        grad_Q[q_idx] += scale * grad_scores[j] * K[k_idx];
                        grad_K[k_idx] += scale * grad_scores[j] * Q[q_idx];
                    }
                }
            }
        }
    }
}

bool test_scaled_attention_backward_reference() {
    std::cout << "=== Testing Scaled Dot-Product Attention Backward Reference ===" << std::endl;
    
    // Create test data with known values for manual verification
    std::vector<float> Q = {
        // Head 0, Seq 0: [0.1, 0.2]
        // Head 1, Seq 0: [0.3, 0.4] 
        // Head 0, Seq 1: [0.5, 0.6]
        // Head 1, Seq 1: [0.7, 0.8]
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    std::vector<float> K = {
        0.2f, 0.1f, 0.4f, 0.3f,  // seq 0
        0.6f, 0.5f, 0.8f, 0.7f   // seq 1
    };
    
    std::vector<float> V = {
        0.1f, 0.3f, 0.5f, 0.7f,  // seq 0
        0.2f, 0.4f, 0.6f, 0.8f   // seq 1
    };
    
    // Forward pass to get attention weights
    std::vector<float> output(BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> attention_weights(BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN, 0.0f);
    
    scaled_attention_forward_cpu(Q, K, V, output, attention_weights, 
                                BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);
    
    // Create gradient w.r.t. output
    std::vector<float> grad_output = {
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    // Backward pass
    std::vector<float> grad_Q(BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> grad_K(BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> grad_V(BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM, 0.0f);
    
    scaled_attention_backward_cpu(Q, K, V, attention_weights, grad_output,
                                 grad_Q, grad_K, grad_V,
                                 BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);
    
    // Print results for verification
    std::cout << "Q (" << Q.size() << "): [";
    for (size_t i = 0; i < Q.size(); i++) {
        std::cout << Q[i];
        if (i < Q.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_output (" << grad_output.size() << "): [";
    for (size_t i = 0; i < grad_output.size(); i++) {
        std::cout << grad_output[i];
        if (i < grad_output.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_Q (" << grad_Q.size() << "): [";
    for (size_t i = 0; i < grad_Q.size(); i++) {
        std::cout << grad_Q[i];
        if (i < grad_Q.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_K (" << grad_K.size() << "): [";
    for (size_t i = 0; i < grad_K.size(); i++) {
        std::cout << grad_K[i];
        if (i < grad_K.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_V (" << grad_V.size() << "): [";
    for (size_t i = 0; i < grad_V.size(); i++) {
        std::cout << grad_V[i];
        if (i < grad_V.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "✅ Scaled Dot-Product Attention backward reference calculations completed." << std::endl;
    return true;
}

bool test_scaled_attention_backward_msl() {
    std::cout << "\n=== Testing Scaled Dot-Product Attention Backward MSL Kernel ===" << std::endl;
    
    // Initialize Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "❌ Failed to create Metal device" << std::endl;
        return false;
    }
    
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        std::cerr << "❌ Failed to create command queue" << std::endl;
        return false;
    }
    
    // Load the backward kernels MSL library
    std::string kernelPath = "../src/msl/backward_kernels.msl";
    std::ifstream kernelFile(kernelPath);
    if (!kernelFile.is_open()) {
        std::cerr << "❌ Failed to open " << kernelPath << std::endl;
        std::cerr << "❌ Current working directory should be tests/" << std::endl;
        // Try absolute path
        kernelPath = "/Users/andrewceniccola/Desktop/cajal/src/msl/backward_kernels.msl";
        kernelFile.open(kernelPath);
        if (!kernelFile.is_open()) {
            std::cerr << "❌ Also failed to open absolute path: " << kernelPath << std::endl;
            return false;
        } else {
            std::cout << "✓ Successfully opened using absolute path" << std::endl;
        }
    }
    
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            std::istreambuf_iterator<char>());
    kernelFile.close();
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kernelSource.c_str()) options:nil error:&error];
    if (!library) {
        std::cerr << "❌ Failed to create MSL library: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"scaled_dot_product_attention_backward"];
    if (!function) {
        std::cerr << "❌ Failed to find scaled_dot_product_attention_backward function" << std::endl;
        return false;
    }
    
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        std::cerr << "❌ Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create test data (same as reference test)
    std::vector<float> Q = {
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    std::vector<float> K = {
        0.2f, 0.1f, 0.4f, 0.3f,  // seq 0
        0.6f, 0.5f, 0.8f, 0.7f   // seq 1
    };
    
    std::vector<float> V = {
        0.1f, 0.3f, 0.5f, 0.7f,  // seq 0
        0.2f, 0.4f, 0.6f, 0.8f   // seq 1
    };
    
    std::vector<float> grad_output = {
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    // Get forward pass attention weights for backward computation
    std::vector<float> output_ref(BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> attention_weights(BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN, 0.0f);
    scaled_attention_forward_cpu(Q, K, V, output_ref, attention_weights, 
                                BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);
    
    // Convert to half precision for GPU
    std::vector<uint16_t> Q_half(Q.size());
    std::vector<uint16_t> K_half(K.size());
    std::vector<uint16_t> V_half(V.size());
    std::vector<uint16_t> grad_output_half(grad_output.size());
    std::vector<uint16_t> attention_weights_half(attention_weights.size());
    
    for (size_t i = 0; i < Q.size(); i++) Q_half[i] = floatToHalf(Q[i]);
    for (size_t i = 0; i < K.size(); i++) K_half[i] = floatToHalf(K[i]);
    for (size_t i = 0; i < V.size(); i++) V_half[i] = floatToHalf(V[i]);
    for (size_t i = 0; i < grad_output.size(); i++) grad_output_half[i] = floatToHalf(grad_output[i]);
    for (size_t i = 0; i < attention_weights.size(); i++) attention_weights_half[i] = floatToHalf(attention_weights[i]);
    
    // Create Metal buffers
    id<MTLBuffer> Q_buffer = [device newBufferWithBytes:Q_half.data() 
                                                length:Q_half.size() * sizeof(uint16_t) 
                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> K_buffer = [device newBufferWithBytes:K_half.data() 
                                                length:K_half.size() * sizeof(uint16_t) 
                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> V_buffer = [device newBufferWithBytes:V_half.data() 
                                                length:V_half.size() * sizeof(uint16_t) 
                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_output_buffer = [device newBufferWithBytes:grad_output_half.data() 
                                                          length:grad_output_half.size() * sizeof(uint16_t) 
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> attention_weights_buffer = [device newBufferWithBytes:attention_weights_half.data() 
                                                                length:attention_weights_half.size() * sizeof(uint16_t) 
                                                               options:MTLResourceStorageModeShared];
    
    // Create output buffers
    id<MTLBuffer> grad_Q_buffer = [device newBufferWithLength:Q_half.size() * sizeof(uint16_t) 
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_K_buffer = [device newBufferWithLength:K_half.size() * sizeof(uint16_t) 
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_V_buffer = [device newBufferWithLength:V_half.size() * sizeof(uint16_t) 
                                                      options:MTLResourceStorageModeShared];
    
    // Execute kernel
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:Q_buffer offset:0 atIndex:0];
    [encoder setBuffer:K_buffer offset:0 atIndex:1];
    [encoder setBuffer:V_buffer offset:0 atIndex:2];
    [encoder setBuffer:attention_weights_buffer offset:0 atIndex:3];
    [encoder setBuffer:grad_output_buffer offset:0 atIndex:4];
    [encoder setBuffer:grad_Q_buffer offset:0 atIndex:5];
    [encoder setBuffer:grad_K_buffer offset:0 atIndex:6];
    [encoder setBuffer:grad_V_buffer offset:0 atIndex:7];
    [encoder setBytes:&BATCH_SIZE length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&SEQ_LEN length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&NUM_HEADS length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&HEAD_DIM length:sizeof(uint32_t) atIndex:11];
    
    MTLSize threadsPerGrid = MTLSizeMake(BATCH_SIZE, NUM_HEADS, SEQ_LEN);
    MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Read back results
    uint16_t* grad_Q_data = (uint16_t*)[grad_Q_buffer contents];
    uint16_t* grad_K_data = (uint16_t*)[grad_K_buffer contents];
    uint16_t* grad_V_data = (uint16_t*)[grad_V_buffer contents];
    
    std::vector<float> grad_Q_msl(Q.size());
    std::vector<float> grad_K_msl(K.size());
    std::vector<float> grad_V_msl(V.size());
    
    for (size_t i = 0; i < Q.size(); i++) grad_Q_msl[i] = halfToFloat(grad_Q_data[i]);
    for (size_t i = 0; i < K.size(); i++) grad_K_msl[i] = halfToFloat(grad_K_data[i]);
    for (size_t i = 0; i < V.size(); i++) grad_V_msl[i] = halfToFloat(grad_V_data[i]);
    
    // Compare with reference
    std::vector<float> grad_Q_ref(Q.size(), 0.0f);
    std::vector<float> grad_K_ref(K.size(), 0.0f);
    std::vector<float> grad_V_ref(V.size(), 0.0f);
    
    scaled_attention_backward_cpu(Q, K, V, attention_weights, grad_output,
                                 grad_Q_ref, grad_K_ref, grad_V_ref,
                                 BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);
    
    // Print MSL results
    std::cout << "grad_Q_msl (" << grad_Q_msl.size() << "): [";
    for (size_t i = 0; i < grad_Q_msl.size(); i++) {
        std::cout << grad_Q_msl[i];
        if (i < grad_Q_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_K_msl (" << grad_K_msl.size() << "): [";
    for (size_t i = 0; i < grad_K_msl.size(); i++) {
        std::cout << grad_K_msl[i];
        if (i < grad_K_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_V_msl (" << grad_V_msl.size() << "): [";
    for (size_t i = 0; i < grad_V_msl.size(); i++) {
        std::cout << grad_V_msl[i];
        if (i < grad_V_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Verify results with tolerance
    const float tolerance = 1e-3f; // Relaxed for half precision
    bool success = true;
    
    for (size_t i = 0; i < Q.size(); i++) {
        if (std::abs(grad_Q_msl[i] - grad_Q_ref[i]) > tolerance) {
            std::cerr << "❌ grad_Q mismatch at index " << i << ": " 
                      << grad_Q_msl[i] << " vs " << grad_Q_ref[i] << std::endl;
            success = false;
        }
    }
    
    for (size_t i = 0; i < K.size(); i++) {
        if (std::abs(grad_K_msl[i] - grad_K_ref[i]) > tolerance) {
            std::cerr << "❌ grad_K mismatch at index " << i << ": " 
                      << grad_K_msl[i] << " vs " << grad_K_ref[i] << std::endl;
            success = false;
        }
    }
    
    for (size_t i = 0; i < V.size(); i++) {
        if (std::abs(grad_V_msl[i] - grad_V_ref[i]) > tolerance) {
            std::cerr << "❌ grad_V mismatch at index " << i << ": " 
                      << grad_V_msl[i] << " vs " << grad_V_ref[i] << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "✅ MSL scaled_dot_product_attention_backward kernel test passed!" << std::endl;
    } else {
        std::cout << "❌ MSL scaled_dot_product_attention_backward kernel test failed!" << std::endl;
    }
    
    return success;
}

int main() {
    std::cout << "=== Scaled Dot-Product Attention Backward Pass TDD Tests ===" << std::endl;
    
    bool success = true;
    
    // Test reference implementation
    success &= test_scaled_attention_backward_reference();
    
    // Test MSL kernel
    success &= test_scaled_attention_backward_msl();
    
    if (success) {
        std::cout << "\n✅ All Scaled Dot-Product Attention backward tests passed!" << std::endl;
    } else {
        std::cout << "\n❌ Some Scaled Dot-Product Attention backward tests failed!" << std::endl;
    }
    
    return success ? 0 : 1;
} 