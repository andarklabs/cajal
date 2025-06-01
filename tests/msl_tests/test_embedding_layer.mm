#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Metal/Metal.h>

// Test configuration for embedding layer
struct EmbeddingTestConfig {
    uint32_t vocab_size = 5;        // Small vocabulary for testing
    uint32_t embedding_dim = 4;     // Small embedding dimension
    uint32_t batch_size = 2;        // Small batch
    uint32_t sequence_length = 3;   // Short sequences
    
    // Known test data
    std::vector<uint32_t> token_ids = {
        // Batch 0: tokens [1, 2, 0]
        1, 2, 0,
        // Batch 1: tokens [3, 1, 4]  
        3, 1, 4
    };
    
    // Known embedding weights (vocab_size=5 x embedding_dim=4)
    // Row 0 (token 0): [0.1, 0.2, 0.3, 0.4]
    // Row 1 (token 1): [0.5, 0.6, 0.7, 0.8]
    // Row 2 (token 2): [0.9, 1.0, 1.1, 1.2]
    // Row 3 (token 3): [1.3, 1.4, 1.5, 1.6]
    // Row 4 (token 4): [1.7, 1.8, 1.9, 2.0]
    std::vector<float> embedding_weights = {
        0.1f, 0.2f, 0.3f, 0.4f,  // token 0
        0.5f, 0.6f, 0.7f, 0.8f,  // token 1
        0.9f, 1.0f, 1.1f, 1.2f,  // token 2
        1.3f, 1.4f, 1.5f, 1.6f,  // token 3
        1.7f, 1.8f, 1.9f, 2.0f   // token 4
    };
    
    // Expected output embeddings (batch_size=2 x sequence_length=3 x embedding_dim=4)
    std::vector<float> expected_output = {
        // Batch 0
        0.5f, 0.6f, 0.7f, 0.8f,  // token 1 embedding
        0.9f, 1.0f, 1.1f, 1.2f,  // token 2 embedding
        0.1f, 0.2f, 0.3f, 0.4f,  // token 0 embedding
        // Batch 1
        1.3f, 1.4f, 1.5f, 1.6f,  // token 3 embedding
        0.5f, 0.6f, 0.7f, 0.8f,  // token 1 embedding
        1.7f, 1.8f, 1.9f, 2.0f   // token 4 embedding
    };
};

class EmbeddingLayerTest {
public:
    EmbeddingLayerTest() {
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
            
            kernel void embedding_lookup(
                device const uint* token_ids [[buffer(0)]],
                device const float* weights [[buffer(1)]],
                device float* output_embeddings [[buffer(2)]],
                constant uint& vocab_size [[buffer(3)]],
                constant uint& embedding_dim [[buffer(4)]],
                constant uint& sequence_length [[buffer(5)]],
                uint gid [[thread_position_in_grid]]
            ) {
                // Calculate batch index and sequence index
                uint total_tokens = gid;
                uint token_idx = total_tokens;
                
                // Bounds check
                if (token_idx >= sequence_length * 2) return;  // batch_size = 2 for test
                
                // Get the token ID
                uint token_id = token_ids[token_idx];
                
                // Bounds check for token ID
                if (token_id >= vocab_size) return;
                
                // Calculate source and destination addresses
                uint weight_offset = token_id * embedding_dim;
                uint output_offset = token_idx * embedding_dim;
                
                // Copy embedding vector
                for (uint i = 0; i < embedding_dim; ++i) {
                    output_embeddings[output_offset + i] = weights[weight_offset + i];
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
        
        id<MTLFunction> kernel_function = [library newFunctionWithName:@"embedding_lookup"];
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
    
    bool test_embedding_lookup() {
        std::cout << "Running test_embedding_lookup..." << std::endl;
        
        EmbeddingTestConfig config;
        
        // Create Metal buffers
        size_t token_ids_size = config.token_ids.size() * sizeof(uint32_t);
        size_t weights_size = config.embedding_weights.size() * sizeof(float);
        size_t output_size = config.expected_output.size() * sizeof(float);
        
        id<MTLBuffer> token_ids_buffer = [device_ newBufferWithBytes:config.token_ids.data()
                                                              length:token_ids_size
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> weights_buffer = [device_ newBufferWithBytes:config.embedding_weights.data()
                                                            length:weights_size
                                                           options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_size
                                                           options:MTLResourceStorageModeShared];
        
        // Create constant buffers for kernel parameters
        id<MTLBuffer> vocab_size_buffer = [device_ newBufferWithBytes:&config.vocab_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> embedding_dim_buffer = [device_ newBufferWithBytes:&config.embedding_dim
                                                                  length:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&config.sequence_length
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        // Set the compute pipeline and buffers
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:token_ids_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:weights_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:output_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:vocab_size_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:5];
        
        // Dispatch threads
        NSUInteger total_threads = config.batch_size * config.sequence_length;
        MTLSize grid_size = MTLSizeMake(total_threads, 1, 1);
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        // Execute
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Verify results
        float* output_data = (float*)[output_buffer contents];
        
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
            }
        }
        
        if (success) {
            std::cout << "  Embedding lookup test PASSED!" << std::endl;
        } else {
            std::cout << "  Embedding lookup test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_bounds_checking() {
        std::cout << "Running test_bounds_checking..." << std::endl;
        
        // Test with invalid token IDs
        std::vector<uint32_t> invalid_tokens = {0, 1, 10}; // token 10 is out of bounds for vocab_size=5
        EmbeddingTestConfig config;
        
        size_t token_ids_size = invalid_tokens.size() * sizeof(uint32_t);
        size_t weights_size = config.embedding_weights.size() * sizeof(float);
        size_t output_size = invalid_tokens.size() * config.embedding_dim * sizeof(float);
        
        id<MTLBuffer> token_ids_buffer = [device_ newBufferWithBytes:invalid_tokens.data()
                                                              length:token_ids_size
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> weights_buffer = [device_ newBufferWithBytes:config.embedding_weights.data()
                                                            length:weights_size
                                                           options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_size
                                                           options:MTLResourceStorageModeShared];
        
        // Zero out output buffer to detect if invalid tokens write anything
        float* output_data = (float*)[output_buffer contents];
        for (size_t i = 0; i < output_size / sizeof(float); ++i) {
            output_data[i] = -999.0f; // Sentinel value
        }
        
        // Dispatch kernel (similar to above)
        id<MTLBuffer> vocab_size_buffer = [device_ newBufferWithBytes:&config.vocab_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> embedding_dim_buffer = [device_ newBufferWithBytes:&config.embedding_dim
                                                                  length:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        
        uint32_t test_seq_len = 3;
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&test_seq_len
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:token_ids_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:weights_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:output_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:vocab_size_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:5];
        
        MTLSize grid_size = MTLSizeMake(3, 1, 1); // 3 tokens
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Check that valid tokens (0, 1) have embeddings but invalid token (10) doesn't
        bool success = true;
        
        // Token 0 should have embedding [0.1, 0.2, 0.3, 0.4]
        for (int i = 0; i < 4; ++i) {
            if (std::abs(output_data[i] - config.embedding_weights[i]) > 1e-6f) {
                std::cout << "  Token 0 embedding incorrect" << std::endl;
                success = false;
            }
        }
        
        // Token 1 should have embedding [0.5, 0.6, 0.7, 0.8]
        for (int i = 0; i < 4; ++i) {
            if (std::abs(output_data[4 + i] - config.embedding_weights[4 + i]) > 1e-6f) {
                std::cout << "  Token 1 embedding incorrect" << std::endl;
                success = false;
            }
        }
        
        // Token 10 (invalid) should leave sentinel values
        for (int i = 0; i < 4; ++i) {
            if (output_data[8 + i] != -999.0f) {
                std::cout << "  Invalid token 10 overwrote output!" << std::endl;
                success = false;
            }
        }
        
        if (success) {
            std::cout << "  Bounds checking test PASSED!" << std::endl;
        } else {
            std::cout << "  Bounds checking test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    ~EmbeddingLayerTest() {
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
    std::cout << "Embedding Layer MSL Tests" << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        EmbeddingLayerTest test;
        
        if (!test.load_kernel()) {
            std::cout << "Failed to load Metal kernel" << std::endl;
            return 1;
        }
        
        bool success = true;
        success &= test.test_embedding_lookup();
        success &= test.test_bounds_checking();
        
        if (success) {
            std::cout << std::endl << "All embedding layer tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << std::endl << "Some embedding layer tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 