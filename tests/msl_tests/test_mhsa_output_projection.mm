#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Metal/Metal.h>

// Test configuration for MHSA output projection
struct OutputProjectionTestConfig {
    uint32_t batch_size = 1;         // Small batch
    uint32_t num_heads = 2;          // Multiple heads
    uint32_t sequence_length = 2;    // Short sequences
    uint32_t head_dim = 2;           // Small head dimension
    uint32_t embedding_dim = 4;      // num_heads * head_dim
    
    // Context vectors from attention (batch_size x num_heads x sequence_length x head_dim)
    // Layout: (batch, head, seq, dim)
    std::vector<float> context_vectors = {
        // Batch 0, Head 0, Position 0: [1, 2]
        1.0f, 2.0f,
        // Batch 0, Head 0, Position 1: [3, 4]
        3.0f, 4.0f,
        // Batch 0, Head 1, Position 0: [5, 6]
        5.0f, 6.0f,
        // Batch 0, Head 1, Position 1: [7, 8]
        7.0f, 8.0f
    };
    
    // Output projection weight matrix Wo (embedding_dim x embedding_dim)
    std::vector<float> Wo = {
        // Row 0: [0.1, 0.2, 0.3, 0.4]
        0.1f, 0.2f, 0.3f, 0.4f,
        // Row 1: [0.5, 0.6, 0.7, 0.8]
        0.5f, 0.6f, 0.7f, 0.8f,
        // Row 2: [0.9, 1.0, 1.1, 1.2]
        0.9f, 1.0f, 1.1f, 1.2f,
        // Row 3: [1.3, 1.4, 1.5, 1.6]
        1.3f, 1.4f, 1.5f, 1.6f
    };
    
    // Expected outputs
    std::vector<float> expected_concatenated; // Context after concatenation
    std::vector<float> expected_output;       // Final MHSA output
    
    void compute_expected_values() {
        // Concatenate heads: (B, H, S, D) -> (B, S, H*D)
        expected_concatenated.resize(batch_size * sequence_length * embedding_dim);
        expected_output.resize(batch_size * sequence_length * embedding_dim);
        
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            for (uint32_t seq = 0; seq < sequence_length; ++seq) {
                // Concatenate heads for this position
                for (uint32_t head = 0; head < num_heads; ++head) {
                    for (uint32_t dim = 0; dim < head_dim; ++dim) {
                        uint32_t context_idx = batch * num_heads * sequence_length * head_dim +
                                             head * sequence_length * head_dim +
                                             seq * head_dim + dim;
                        uint32_t concat_idx = batch * sequence_length * embedding_dim +
                                            seq * embedding_dim +
                                            head * head_dim + dim;
                        
                        expected_concatenated[concat_idx] = context_vectors[context_idx];
                    }
                }
            }
        }
        
        // Apply output projection: concatenated @ Wo
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            for (uint32_t seq = 0; seq < sequence_length; ++seq) {
                // Get concatenated vector for this position
                std::vector<float> concat_vec(embedding_dim);
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    concat_vec[i] = expected_concatenated[batch * sequence_length * embedding_dim +
                                                        seq * embedding_dim + i];
                }
                
                // Matrix multiplication: concat_vec @ Wo
                for (uint32_t i = 0; i < embedding_dim; ++i) {
                    float output_val = 0.0f;
                    for (uint32_t j = 0; j < embedding_dim; ++j) {
                        output_val += concat_vec[j] * Wo[j * embedding_dim + i];
                    }
                    expected_output[batch * sequence_length * embedding_dim +
                                  seq * embedding_dim + i] = output_val;
                }
            }
        }
    }
};

class MHSAOutputProjectionTest {
public:
    MHSAOutputProjectionTest() {
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
            
            kernel void mhsa_output_projection(
                device const float* context_vectors [[buffer(0)]],
                device const float* Wo [[buffer(1)]],
                device float* final_output [[buffer(2)]],
                constant uint& batch_size [[buffer(3)]],
                constant uint& sequence_length [[buffer(4)]],
                constant uint& embedding_dim [[buffer(5)]],
                constant uint& num_heads [[buffer(6)]],
                constant uint& head_dim [[buffer(7)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint batch_idx = gid.y;
                uint seq_idx = gid.x;
                
                // Bounds check
                if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
                
                // Calculate input and output base indices
                uint context_base = batch_idx * num_heads * sequence_length * head_dim;
                uint output_base = batch_idx * sequence_length * embedding_dim + seq_idx * embedding_dim;
                
                // Temporary storage for concatenated context vector
                float concat_vec[8]; // Max embedding_dim we support
                
                // Step 1: Concatenate heads for this position
                // Input layout: (batch, head, seq, dim)
                // Output layout: (batch, seq, head*dim)
                for (uint head = 0; head < num_heads; ++head) {
                    for (uint dim = 0; dim < head_dim; ++dim) {
                        uint context_idx = context_base + head * sequence_length * head_dim +
                                         seq_idx * head_dim + dim;
                        uint concat_idx = head * head_dim + dim;
                        
                        concat_vec[concat_idx] = context_vectors[context_idx];
                    }
                }
                
                // Step 2: Apply output projection (matrix multiplication)
                // concat_vec @ Wo -> final_output
                for (uint i = 0; i < embedding_dim; ++i) {
                    float output_val = 0.0f;
                    for (uint j = 0; j < embedding_dim; ++j) {
                        output_val += concat_vec[j] * Wo[j * embedding_dim + i];
                    }
                    final_output[output_base + i] = output_val;
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
        
        id<MTLFunction> kernel_function = [library newFunctionWithName:@"mhsa_output_projection"];
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
    
    bool test_concatenation() {
        std::cout << "Running test_concatenation..." << std::endl;
        
        OutputProjectionTestConfig config;
        config.compute_expected_values();
        
        // Print expected concatenated values for verification
        std::cout << "  Expected concatenated context: ";
        for (float val : config.expected_concatenated) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        // Expected for our test data:
        // Position 0: [1, 2] from head 0 + [5, 6] from head 1 = [1, 2, 5, 6]
        // Position 1: [3, 4] from head 0 + [7, 8] from head 1 = [3, 4, 7, 8]
        
        std::vector<float> expected_concat = {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f};
        
        bool success = true;
        for (size_t i = 0; i < expected_concat.size(); ++i) {
            if (std::abs(config.expected_concatenated[i] - expected_concat[i]) > 1e-6f) {
                std::cout << "  Concatenation mismatch at index " << i << std::endl;
                success = false;
            }
        }
        
        if (success) {
            std::cout << "  Concatenation test PASSED!" << std::endl;
        } else {
            std::cout << "  Concatenation test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_mhsa_output_projection() {
        std::cout << "Running test_mhsa_output_projection..." << std::endl;
        
        OutputProjectionTestConfig config;
        config.compute_expected_values();
        
        // Create Metal buffers
        size_t context_size = config.context_vectors.size() * sizeof(float);
        size_t weight_size = config.Wo.size() * sizeof(float);
        size_t output_size = config.expected_output.size() * sizeof(float);
        
        id<MTLBuffer> context_buffer = [device_ newBufferWithBytes:config.context_vectors.data()
                                                            length:context_size
                                                           options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> Wo_buffer = [device_ newBufferWithBytes:config.Wo.data()
                                                       length:weight_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_size
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
        [compute_encoder setBuffer:context_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:Wo_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:output_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:batch_size_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:5];
        [compute_encoder setBuffer:num_heads_buffer offset:0 atIndex:6];
        [compute_encoder setBuffer:head_dim_buffer offset:0 atIndex:7];
        
        // Dispatch threads (2D grid: sequence_length x batch_size)
        MTLSize grid_size = MTLSizeMake(config.sequence_length, config.batch_size, 1);
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        // Execute
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Verify results
        float* output_data = (float*)[output_buffer contents];
        
        std::cout << "  Expected output: ";
        for (float val : config.expected_output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
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
            std::cout << "  MHSA output projection test PASSED!" << std::endl;
        } else {
            std::cout << "  MHSA output projection test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_identity_projection() {
        std::cout << "Running test_identity_projection..." << std::endl;
        
        // Test with identity weight matrix
        OutputProjectionTestConfig config;
        
        // Set Wo to identity matrix
        std::fill(config.Wo.begin(), config.Wo.end(), 0.0f);
        for (uint32_t i = 0; i < config.embedding_dim; ++i) {
            config.Wo[i * config.embedding_dim + i] = 1.0f;
        }
        
        config.compute_expected_values();
        
        // With identity matrix, output should equal concatenated input
        bool matches_concat = true;
        for (size_t i = 0; i < config.expected_output.size(); ++i) {
            if (std::abs(config.expected_output[i] - config.expected_concatenated[i]) > 1e-6f) {
                matches_concat = false;
                break;
            }
        }
        
        if (matches_concat) {
            std::cout << "  Identity projection test PASSED!" << std::endl;
        } else {
            std::cout << "  Identity projection test FAILED!" << std::endl;
        }
        
        return matches_concat;
    }
    
    ~MHSAOutputProjectionTest() {
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
    std::cout << "MHSA Output Projection MSL Tests" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        MHSAOutputProjectionTest test;
        
        if (!test.load_kernel()) {
            std::cout << "Failed to load Metal kernel" << std::endl;
            return 1;
        }
        
        bool success = true;
        success &= test.test_concatenation();
        success &= test.test_identity_projection();
        success &= test.test_mhsa_output_projection();
        
        if (success) {
            std::cout << std::endl << "All MHSA output projection tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << std::endl << "Some MHSA output projection tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 