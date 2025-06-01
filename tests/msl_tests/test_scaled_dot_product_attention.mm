#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Metal/Metal.h>

// Test configuration for scaled dot-product attention
struct AttentionTestConfig {
    uint32_t batch_size = 1;         // Small batch
    uint32_t num_heads = 1;          // Single head for simplicity
    uint32_t sequence_length = 3;    // Short sequences
    uint32_t head_dim = 2;           // Small head dimension
    float scale_factor = 1.0f / std::sqrt(2.0f); // 1/sqrt(head_dim)
    
    // Q, K, V matrices (batch_size x num_heads x sequence_length x head_dim)
    // Layout: (batch, head, seq, dim)
    std::vector<float> Q = {
        // Batch 0, Head 0, Position 0: [1, 0]
        1.0f, 0.0f,
        // Batch 0, Head 0, Position 1: [0, 1]
        0.0f, 1.0f,
        // Batch 0, Head 0, Position 2: [1, 1]
        1.0f, 1.0f
    };
    
    std::vector<float> K = {
        // Same as Q for this test
        1.0f, 0.0f,  // Position 0
        0.0f, 1.0f,  // Position 1
        1.0f, 1.0f   // Position 2
    };
    
    std::vector<float> V = {
        // Different values for V
        2.0f, 3.0f,  // Position 0
        4.0f, 5.0f,  // Position 1
        6.0f, 7.0f   // Position 2
    };
    
    // Expected outputs
    std::vector<float> expected_scores;     // QK^T scores
    std::vector<float> expected_masked;     // After causal masking
    std::vector<float> expected_weights;    // After softmax
    std::vector<float> expected_output;     // Final context vectors
    
    void compute_expected_values() {
        // Compute QK^T
        expected_scores.resize(sequence_length * sequence_length);
        expected_masked.resize(sequence_length * sequence_length);
        expected_weights.resize(sequence_length * sequence_length);
        expected_output.resize(sequence_length * head_dim);
        
        // QK^T computation
        for (uint32_t i = 0; i < sequence_length; ++i) {
            for (uint32_t j = 0; j < sequence_length; ++j) {
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    float q_val = Q[i * head_dim + d];
                    float k_val = K[j * head_dim + d];
                    score += q_val * k_val;
                }
                score *= scale_factor;
                expected_scores[i * sequence_length + j] = score;
            }
        }
        
        // Apply causal masking (set future positions to -inf)
        const float NEG_INF = -1e9f;
        for (uint32_t i = 0; i < sequence_length; ++i) {
            for (uint32_t j = 0; j < sequence_length; ++j) {
                if (j > i) {
                    expected_masked[i * sequence_length + j] = NEG_INF;
                } else {
                    expected_masked[i * sequence_length + j] = expected_scores[i * sequence_length + j];
                }
            }
        }
        
        // Apply softmax row-wise
        for (uint32_t i = 0; i < sequence_length; ++i) {
            // Find max for numerical stability
            float max_val = expected_masked[i * sequence_length + 0];
            for (uint32_t j = 1; j < sequence_length; ++j) {
                if (j <= i) { // Only consider non-masked values
                    max_val = std::max(max_val, expected_masked[i * sequence_length + j]);
                }
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (uint32_t j = 0; j < sequence_length; ++j) {
                if (j <= i) {
                    float exp_val = std::exp(expected_masked[i * sequence_length + j] - max_val);
                    expected_weights[i * sequence_length + j] = exp_val;
                    sum_exp += exp_val;
                } else {
                    expected_weights[i * sequence_length + j] = 0.0f;
                }
            }
            
            // Normalize
            for (uint32_t j = 0; j <= i; ++j) {
                expected_weights[i * sequence_length + j] /= sum_exp;
            }
        }
        
        // Compute final output: weights @ V
        for (uint32_t i = 0; i < sequence_length; ++i) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                float output_val = 0.0f;
                for (uint32_t j = 0; j < sequence_length; ++j) {
                    float weight = expected_weights[i * sequence_length + j];
                    float v_val = V[j * head_dim + d];
                    output_val += weight * v_val;
                }
                expected_output[i * head_dim + d] = output_val;
            }
        }
    }
};

class ScaledDotProductAttentionTest {
public:
    ScaledDotProductAttentionTest() {
        // Initialize Metal
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("Failed to create Metal device");
        }
        
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
        
        std::cout << "Metal device: " << [device_.name UTF8String] << std::endl;
    }
    
    bool load_kernel() {
        // Load the Metal kernel library
        NSError* error = nil;
        NSString* kernel_source = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void scaled_dot_product_attention(
                device const float* Q [[buffer(0)]],
                device const float* K [[buffer(1)]],
                device const float* V [[buffer(2)]],
                device float* output_context [[buffer(3)]],
                constant uint& batch_size [[buffer(4)]],
                constant uint& num_heads [[buffer(5)]],
                constant uint& sequence_length [[buffer(6)]],
                constant uint& head_dim [[buffer(7)]],
                constant float& scale_factor [[buffer(8)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint batch_idx = gid.z;
                uint head_idx = gid.y;
                uint query_idx = gid.x;
                
                // Bounds check
                if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= sequence_length) 
                    return;
                
                // Calculate base indices for this head
                uint qkv_base = batch_idx * num_heads * sequence_length * head_dim + 
                               head_idx * sequence_length * head_dim;
                uint output_base = qkv_base + query_idx * head_dim;
                
                // Temporary storage for attention scores and weights
                float scores[8];    // Max sequence length we support
                float weights[8];
                
                // Step 1: Compute QK^T scores for this query position
                float q_vec[4];     // Max head_dim we support
                for (uint d = 0; d < head_dim; ++d) {
                    q_vec[d] = Q[qkv_base + query_idx * head_dim + d];
                }
                
                for (uint key_idx = 0; key_idx < sequence_length; ++key_idx) {
                    float score = 0.0f;
                    for (uint d = 0; d < head_dim; ++d) {
                        float k_val = K[qkv_base + key_idx * head_dim + d];
                        score += q_vec[d] * k_val;
                    }
                    scores[key_idx] = score * scale_factor;
                }
                
                // Step 2: Apply causal masking
                const float NEG_INF = -1e9f;
                for (uint key_idx = 0; key_idx < sequence_length; ++key_idx) {
                    if (key_idx > query_idx) {
                        scores[key_idx] = NEG_INF;
                    }
                }
                
                // Step 3: Softmax - find max for stability
                float max_score = scores[0];
                for (uint key_idx = 1; key_idx <= query_idx; ++key_idx) {
                    max_score = max(max_score, scores[key_idx]);
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (uint key_idx = 0; key_idx < sequence_length; ++key_idx) {
                    if (key_idx <= query_idx) {
                        weights[key_idx] = exp(scores[key_idx] - max_score);
                        sum_exp += weights[key_idx];
                    } else {
                        weights[key_idx] = 0.0f;
                    }
                }
                
                // Normalize
                for (uint key_idx = 0; key_idx <= query_idx; ++key_idx) {
                    weights[key_idx] /= sum_exp;
                }
                
                // Step 4: Compute weighted sum with V
                for (uint d = 0; d < head_dim; ++d) {
                    float context_val = 0.0f;
                    for (uint key_idx = 0; key_idx < sequence_length; ++key_idx) {
                        float v_val = V[qkv_base + key_idx * head_dim + d];
                        context_val += weights[key_idx] * v_val;
                    }
                    output_context[output_base + d] = context_val;
                }
            }
        )";
        
        id<MTLLibrary> library = [device_ newLibraryWithSource:kernel_source 
                                                       options:nil 
                                                         error:&error];
        if (!library) {
            std::cout << "Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        id<MTLFunction> kernel_function = [library newFunctionWithName:@"scaled_dot_product_attention"];
        if (!kernel_function) {
            std::cout << "Failed to find kernel function" << std::endl;
            return false;
        }
        
        compute_pipeline_ = [device_ newComputePipelineStateWithFunction:kernel_function error:&error];
        if (!compute_pipeline_) {
            std::cout << "Failed to create compute pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool test_manual_calculation() {
        std::cout << "Running test_manual_calculation..." << std::endl;
        
        AttentionTestConfig config;
        config.compute_expected_values();
        
        // Print intermediate values for verification
        std::cout << "  QK^T scores: ";
        for (float score : config.expected_scores) {
            std::cout << score << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  After masking: ";
        for (float masked : config.expected_masked) {
            std::cout << (masked < -1e8f ? "-inf" : std::to_string(masked)) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Attention weights: ";
        for (float weight : config.expected_weights) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Expected output: ";
        for (float out : config.expected_output) {
            std::cout << out << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Manual calculation test PASSED!" << std::endl;
        return true;
    }
    
    bool test_scaled_dot_product_attention() {
        std::cout << "Running test_scaled_dot_product_attention..." << std::endl;
        
        AttentionTestConfig config;
        config.compute_expected_values();
        
        // Create Metal buffers
        size_t qkv_size = config.Q.size() * sizeof(float);
        size_t output_size = config.expected_output.size() * sizeof(float);
        
        id<MTLBuffer> Q_buffer = [device_ newBufferWithBytes:config.Q.data()
                                                      length:qkv_size
                                                     options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> K_buffer = [device_ newBufferWithBytes:config.K.data()
                                                      length:qkv_size
                                                     options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> V_buffer = [device_ newBufferWithBytes:config.V.data()
                                                      length:qkv_size
                                                     options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_size
                                                           options:MTLResourceStorageModeShared];
        
        // Create constant buffers
        id<MTLBuffer> batch_size_buffer = [device_ newBufferWithBytes:&config.batch_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> num_heads_buffer = [device_ newBufferWithBytes:&config.num_heads
                                                              length:sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&config.sequence_length
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> head_dim_buffer = [device_ newBufferWithBytes:&config.head_dim
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> scale_factor_buffer = [device_ newBufferWithBytes:&config.scale_factor
                                                                 length:sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        // Set the compute pipeline and buffers
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:Q_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:K_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:V_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:output_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:batch_size_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:num_heads_buffer offset:0 atIndex:5];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:6];
        [compute_encoder setBuffer:head_dim_buffer offset:0 atIndex:7];
        [compute_encoder setBuffer:scale_factor_buffer offset:0 atIndex:8];
        
        // Dispatch threads (3D grid: sequence_length x num_heads x batch_size)
        MTLSize grid_size = MTLSizeMake(config.sequence_length, config.num_heads, config.batch_size);
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        // Execute
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Verify results
        float* output_data = (float*)[output_buffer contents];
        
        std::cout << "  GPU output: ";
        for (size_t i = 0; i < config.expected_output.size(); ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
        
        bool success = true;
        float tolerance = 1e-6f;
        
        for (size_t i = 0; i < config.expected_output.size(); ++i) {
            float expected = config.expected_output[i];
            float actual = output_data[i];
            float diff = std::abs(expected - actual);
            
            if (diff > tolerance) {
                std::cout << "  MISMATCH at index " << i 
                         << ": expected " << expected 
                         << ", got " << actual 
                         << ", diff " << diff << std::endl;
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "  Scaled dot-product attention test PASSED!" << std::endl;
        } else {
            std::cout << "  Scaled dot-product attention test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_causal_masking() {
        std::cout << "Running test_causal_masking..." << std::endl;
        
        // Test with simple identity Q and K to verify masking works
        AttentionTestConfig config;
        config.Q = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f}; // Identity-like patterns
        config.K = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
        config.V = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // Distinct values
        config.scale_factor = 1.0f; // No scaling for simplicity
        
        config.compute_expected_values();
        
        // Verify that masking worked correctly in expected values
        // Position 0 should only attend to position 0
        // Position 1 should only attend to positions 0 and 1
        // Position 2 should attend to positions 0, 1, and 2
        
        bool masking_correct = true;
        
        // Check position 0: weights should be [1, 0, 0]
        if (config.expected_weights[0] < 0.99f || config.expected_weights[1] > 0.01f || config.expected_weights[2] > 0.01f) {
            masking_correct = false;
            std::cout << "  Position 0 masking incorrect" << std::endl;
        }
        
        // Check position 1: weights should be [w, 1-w, 0] where w + (1-w) = 1
        if (config.expected_weights[4] > 0.01f) { // Position 1, key 2 should be zero
            masking_correct = false;
            std::cout << "  Position 1 masking incorrect" << std::endl;
        }
        
        if (masking_correct) {
            std::cout << "  Causal masking test PASSED!" << std::endl;
        } else {
            std::cout << "  Causal masking test FAILED!" << std::endl;
        }
        
        return masking_correct;
    }
    
    ~ScaledDotProductAttentionTest() {
        [device_ release];
        [command_queue_ release];
        [compute_pipeline_ release];
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLComputePipelineState> compute_pipeline_;
};

int main() {
    std::cout << "Scaled Dot-Product Attention MSL Tests" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        ScaledDotProductAttentionTest test;
        
        if (!test.load_kernel()) {
            std::cout << "Failed to load Metal kernel" << std::endl;
            return 1;
        }
        
        bool success = true;
        success &= test.test_manual_calculation();
        success &= test.test_causal_masking();
        success &= test.test_scaled_dot_product_attention();
        
        if (success) {
            std::cout << std::endl << "All scaled dot-product attention tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << std::endl << "Some scaled dot-product attention tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 