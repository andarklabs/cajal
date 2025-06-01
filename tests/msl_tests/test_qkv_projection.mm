#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Metal/Metal.h>

// Test configuration for QKV projection
struct QKVProjectionTestConfig {
    uint32_t batch_size = 1;         // Small batch for testing
    uint32_t sequence_length = 2;    // Short sequences
    uint32_t embedding_dim = 4;      // Small embedding dimension
    uint32_t num_heads = 2;          // Multiple heads
    uint32_t head_dim = 2;           // embedding_dim / num_heads
    
    // Input embeddings (batch_size x sequence_length x embedding_dim)
    std::vector<float> input_embeddings = {
        // Batch 0, Position 0: [1, 2, 3, 4]
        1.0f, 2.0f, 3.0f, 4.0f,
        // Batch 0, Position 1: [5, 6, 7, 8]
        5.0f, 6.0f, 7.0f, 8.0f
    };
    
    // Weight matrices for Q, K, V (embedding_dim x embedding_dim for simplicity)
    // In practice: embedding_dim x (num_heads * head_dim)
    std::vector<float> Wq = {
        // Row 0: [0.1, 0.2, 0.3, 0.4]
        0.1f, 0.2f, 0.3f, 0.4f,
        // Row 1: [0.5, 0.6, 0.7, 0.8]
        0.5f, 0.6f, 0.7f, 0.8f,
        // Row 2: [0.9, 1.0, 1.1, 1.2]
        0.9f, 1.0f, 1.1f, 1.2f,
        // Row 3: [1.3, 1.4, 1.5, 1.6]
        1.3f, 1.4f, 1.5f, 1.6f
    };
    
    std::vector<float> Wk = {
        // Different weights for K
        0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f, 1.3f,
        1.4f, 1.5f, 1.6f, 1.7f
    };
    
    std::vector<float> Wv = {
        // Different weights for V
        0.3f, 0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f, 1.0f,
        1.1f, 1.2f, 1.3f, 1.4f,
        1.5f, 1.6f, 1.7f, 1.8f
    };
    
    // Expected outputs (manually calculated)
    std::vector<float> expected_Q, expected_K, expected_V;
    
    void compute_expected_values() {
        // Manual matrix multiplication: input @ Wq -> Q_flat
        // Then reshape to (batch, num_heads, seq_len, head_dim)
        
        expected_Q.resize(batch_size * num_heads * sequence_length * head_dim);
        expected_K.resize(batch_size * num_heads * sequence_length * head_dim);
        expected_V.resize(batch_size * num_heads * sequence_length * head_dim);
        
        // For each position, compute Q, K, V
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            for (uint32_t seq = 0; seq < sequence_length; ++seq) {
                // Get input vector for this position
                std::vector<float> input_vec(embedding_dim);
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    input_vec[i] = input_embeddings[batch * sequence_length * embedding_dim + 
                                                   seq * embedding_dim + i];
                }
                
                // Compute Q = input @ Wq
                std::vector<float> Q_raw(embedding_dim, 0.0f);
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    for (uint32_t j = 0; j < embedding_dim; ++j) {
                        Q_raw[i] += input_vec[j] * Wq[j * embedding_dim + i];
                    }
                }
                
                // Compute K = input @ Wk  
                std::vector<float> K_raw(embedding_dim, 0.0f);
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    for (uint32_t j = 0; j < embedding_dim; ++j) {
                        K_raw[i] += input_vec[j] * Wk[j * embedding_dim + i];
                    }
                }
                
                // Compute V = input @ Wv
                std::vector<float> V_raw(embedding_dim, 0.0f);
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    for (uint32_t j = 0; j < embedding_dim; ++j) {
                        V_raw[i] += input_vec[j] * Wv[j * embedding_dim + i];
                    }
                }
                
                // Reshape to heads: (embedding_dim) -> (num_heads, head_dim)
                // Then arrange as (batch, num_heads, seq_len, head_dim)
                for (uint32_t head = 0; head < num_heads; ++head) {
                    for (uint32_t dim = 0; dim < head_dim; ++dim) {
                        uint32_t raw_idx = head * head_dim + dim;
                        uint32_t output_idx = batch * num_heads * sequence_length * head_dim +
                                            head * sequence_length * head_dim +
                                            seq * head_dim + dim;
                        
                        expected_Q[output_idx] = Q_raw[raw_idx];
                        expected_K[output_idx] = K_raw[raw_idx];
                        expected_V[output_idx] = V_raw[raw_idx];
                    }
                }
            }
        }
    }
};

class QKVProjectionTest {
public:
    QKVProjectionTest() {
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
            
            kernel void qkv_projection(
                device const float* input_embeddings [[buffer(0)]],
                device const float* Wq [[buffer(1)]],
                device const float* Wk [[buffer(2)]],
                device const float* Wv [[buffer(3)]],
                device float* Q_out [[buffer(4)]],
                device float* K_out [[buffer(5)]],
                device float* V_out [[buffer(6)]],
                constant uint& batch_size [[buffer(7)]],
                constant uint& sequence_length [[buffer(8)]],
                constant uint& embedding_dim [[buffer(9)]],
                constant uint& num_heads [[buffer(10)]],
                constant uint& head_dim [[buffer(11)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint batch_idx = gid.y;
                uint seq_idx = gid.x;
                
                // Bounds check
                if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
                
                // Calculate input base index
                uint input_base = batch_idx * sequence_length * embedding_dim + seq_idx * embedding_dim;
                
                // Temporary storage for Q, K, V raw outputs (before reshaping)
                float Q_raw[16]; // Max embedding_dim we support in test
                float K_raw[16];
                float V_raw[16];
                
                // Matrix multiplication: input @ Wq -> Q_raw
                for (uint i = 0; i < embedding_dim; ++i) {
                    Q_raw[i] = 0.0f;
                    K_raw[i] = 0.0f;
                    V_raw[i] = 0.0f;
                    
                    for (uint j = 0; j < embedding_dim; ++j) {
                        float input_val = input_embeddings[input_base + j];
                        Q_raw[i] += input_val * Wq[j * embedding_dim + i];
                        K_raw[i] += input_val * Wk[j * embedding_dim + i];
                        V_raw[i] += input_val * Wv[j * embedding_dim + i];
                    }
                }
                
                // Reshape and store: (embedding_dim) -> (num_heads, head_dim)
                // Output layout: (batch, num_heads, seq_len, head_dim)
                for (uint head = 0; head < num_heads; ++head) {
                    for (uint dim = 0; dim < head_dim; ++dim) {
                        uint raw_idx = head * head_dim + dim;
                        uint output_idx = batch_idx * num_heads * sequence_length * head_dim +
                                        head * sequence_length * head_dim +
                                        seq_idx * head_dim + dim;
                        
                        Q_out[output_idx] = Q_raw[raw_idx];
                        K_out[output_idx] = K_raw[raw_idx];
                        V_out[output_idx] = V_raw[raw_idx];
                    }
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
        
        id<MTLFunction> kernel_function = [library newFunctionWithName:@"qkv_projection"];
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
    
    bool test_qkv_projection() {
        std::cout << "Running test_qkv_projection..." << std::endl;
        
        QKVProjectionTestConfig config;
        config.compute_expected_values();
        
        // Create Metal buffers
        size_t input_size = config.input_embeddings.size() * sizeof(float);
        size_t weight_size = config.Wq.size() * sizeof(float);
        size_t output_size = config.expected_Q.size() * sizeof(float);
        
        id<MTLBuffer> input_buffer = [device_ newBufferWithBytes:config.input_embeddings.data()
                                                          length:input_size
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> Wq_buffer = [device_ newBufferWithBytes:config.Wq.data()
                                                       length:weight_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> Wk_buffer = [device_ newBufferWithBytes:config.Wk.data()
                                                       length:weight_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> Wv_buffer = [device_ newBufferWithBytes:config.Wv.data()
                                                       length:weight_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> Q_buffer = [device_ newBufferWithLength:output_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> K_buffer = [device_ newBufferWithLength:output_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> V_buffer = [device_ newBufferWithLength:output_size
                                                      options:MTLResourceStorageModeShared];
        
        // Create constant buffers
        id<MTLBuffer> batch_size_buffer = [device_ newBufferWithBytes:&config.batch_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&config.sequence_length
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> embedding_dim_buffer = [device_ newBufferWithBytes:&config.embedding_dim
                                                                  length:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> num_heads_buffer = [device_ newBufferWithBytes:&config.num_heads
                                                              length:sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> head_dim_buffer = [device_ newBufferWithBytes:&config.head_dim
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        // Set the compute pipeline and buffers
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:input_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:Wq_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:Wk_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:Wv_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:Q_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:K_buffer offset:0 atIndex:5];
        [compute_encoder setBuffer:V_buffer offset:0 atIndex:6];
        [compute_encoder setBuffer:batch_size_buffer offset:0 atIndex:7];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:8];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:9];
        [compute_encoder setBuffer:num_heads_buffer offset:0 atIndex:10];
        [compute_encoder setBuffer:head_dim_buffer offset:0 atIndex:11];
        
        // Dispatch threads (2D grid: sequence_length x batch_size)
        MTLSize grid_size = MTLSizeMake(config.sequence_length, config.batch_size, 1);
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        // Execute
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Verify Q results
        float* Q_data = (float*)[Q_buffer contents];
        bool success = true;
        float tolerance = 1e-6f;
        
        std::cout << "  Verifying Q matrix..." << std::endl;
        for (size_t i = 0; i < config.expected_Q.size(); ++i) {
            float expected = config.expected_Q[i];
            float actual = Q_data[i];
            float diff = std::abs(expected - actual);
            
            if (diff > tolerance) {
                std::cout << "  Q MISMATCH at index " << i 
                         << ": expected " << expected 
                         << ", got " << actual 
                         << ", diff " << diff << std::endl;
                success = false;
                break;
            }
        }
        
        // Verify K results
        float* K_data = (float*)[K_buffer contents];
        std::cout << "  Verifying K matrix..." << std::endl;
        for (size_t i = 0; i < config.expected_K.size(); ++i) {
            float expected = config.expected_K[i];
            float actual = K_data[i];
            float diff = std::abs(expected - actual);
            
            if (diff > tolerance) {
                std::cout << "  K MISMATCH at index " << i 
                         << ": expected " << expected 
                         << ", got " << actual 
                         << ", diff " << diff << std::endl;
                success = false;
                break;
            }
        }
        
        // Verify V results
        float* V_data = (float*)[V_buffer contents];
        std::cout << "  Verifying V matrix..." << std::endl;
        for (size_t i = 0; i < config.expected_V.size(); ++i) {
            float expected = config.expected_V[i];
            float actual = V_data[i];
            float diff = std::abs(expected - actual);
            
            if (diff > tolerance) {
                std::cout << "  V MISMATCH at index " << i 
                         << ": expected " << expected 
                         << ", got " << actual 
                         << ", diff " << diff << std::endl;
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "  QKV projection test PASSED!" << std::endl;
        } else {
            std::cout << "  QKV projection test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_matrix_multiplication() {
        std::cout << "Running test_matrix_multiplication..." << std::endl;
        
        // Test just the matrix multiplication part with simple known values
        std::vector<float> simple_input = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity-like input
        std::vector<float> simple_Wq = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 3.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 4.0f
        };
        
        // Expected: [1, 0, 0, 1] @ simple_Wq = [1, 0, 0, 4]
        
        QKVProjectionTestConfig config;
        config.input_embeddings = simple_input;
        config.Wq = simple_Wq;
        config.Wk = simple_Wq; // Same for simplicity
        config.Wv = simple_Wq;
        config.batch_size = 1;
        config.sequence_length = 1;
        
        config.compute_expected_values();
        
        // Print expected output for debugging
        std::cout << "  Expected Q output: ";
        for (float val : config.expected_Q) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        return true; // This is just a debug test
    }
    
    ~QKVProjectionTest() {
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
    std::cout << "QKV Projection MSL Tests" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        QKVProjectionTest test;
        
        if (!test.load_kernel()) {
            std::cout << "Failed to load Metal kernel" << std::endl;
            return 1;
        }
        
        bool success = true;
        success &= test.test_matrix_multiplication();
        success &= test.test_qkv_projection();
        
        if (success) {
            std::cout << std::endl << "All QKV projection tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << std::endl << "Some QKV projection tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 