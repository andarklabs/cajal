#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Metal/Metal.h>

// Test configuration for positional encoding
struct PositionalEncodingTestConfig {
    uint32_t batch_size = 2;         // Small batch
    uint32_t sequence_length = 4;    // Short sequences
    uint32_t embedding_dim = 6;      // Must be even for sinusoidal PE
    
    // Input embeddings (batch_size x sequence_length x embedding_dim)
    // Initialize to 1.0 for easy verification
    std::vector<float> input_embeddings = {
        // Batch 0
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 0
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 1
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 2
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 3
        // Batch 1
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 0
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 1
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // Position 2
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f   // Position 3
    };
    
    // Pre-computed positional encoding table (sequence_length x embedding_dim)
    std::vector<float> pe_table;
    
    // Expected output (input + PE)
    std::vector<float> expected_output;
    
    void compute_expected_values() {
        // Compute sinusoidal positional encodings
        pe_table.resize(sequence_length * embedding_dim);
        expected_output.resize(batch_size * sequence_length * embedding_dim);
        
        for (uint32_t pos = 0; pos < sequence_length; ++pos) {
            for (uint32_t i = 0; i < embedding_dim; ++i) {
                float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / embedding_dim);
                
                if (i % 2 == 0) {
                    // Even indices: sin
                    pe_table[pos * embedding_dim + i] = std::sin(angle);
                } else {
                    // Odd indices: cos
                    pe_table[pos * embedding_dim + i] = std::cos(angle);
                }
            }
        }
        
        // Compute expected output (input + PE)
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            for (uint32_t pos = 0; pos < sequence_length; ++pos) {
                for (uint32_t dim = 0; dim < embedding_dim; ++dim) {
                    uint32_t input_idx = batch * sequence_length * embedding_dim + pos * embedding_dim + dim;
                    uint32_t pe_idx = pos * embedding_dim + dim;
                    expected_output[input_idx] = input_embeddings[input_idx] + pe_table[pe_idx];
                }
            }
        }
    }
};

class PositionalEncodingTest {
public:
    PositionalEncodingTest() {
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
            
            kernel void apply_positional_encoding(
                device const float* input_embeddings [[buffer(0)]],
                device const float* pe_table [[buffer(1)]],
                device float* output_embeddings [[buffer(2)]],
                constant uint& batch_size [[buffer(3)]],
                constant uint& sequence_length [[buffer(4)]],
                constant uint& embedding_dim [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint batch_idx = gid.y;
                uint seq_idx = gid.x;
                
                // Bounds check
                if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
                
                // Calculate base indices
                uint input_base = batch_idx * sequence_length * embedding_dim + seq_idx * embedding_dim;
                uint pe_base = seq_idx * embedding_dim;
                uint output_base = input_base;
                
                // Apply positional encoding element-wise
                for (uint dim = 0; dim < embedding_dim; ++dim) {
                    output_embeddings[output_base + dim] = 
                        input_embeddings[input_base + dim] + pe_table[pe_base + dim];
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
        
        id<MTLFunction> kernel_function = [library newFunctionWithName:@"apply_positional_encoding"];
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
    
    bool test_pe_calculation() {
        std::cout << "Running test_pe_calculation..." << std::endl;
        
        // Test our manual PE calculation against known values
        PositionalEncodingTestConfig config;
        config.compute_expected_values();
        
        // Check a few specific PE values manually
        // Position 0, dimension 0: sin(0 / 10000^0) = sin(0) = 0
        float expected_00 = 0.0f;
        float actual_00 = config.pe_table[0 * config.embedding_dim + 0];
        if (std::abs(actual_00 - expected_00) > 1e-6f) {
            std::cout << "  PE[0,0] mismatch: expected " << expected_00 << ", got " << actual_00 << std::endl;
            return false;
        }
        
        // Position 0, dimension 1: cos(0 / 10000^0) = cos(0) = 1
        float expected_01 = 1.0f;
        float actual_01 = config.pe_table[0 * config.embedding_dim + 1];
        if (std::abs(actual_01 - expected_01) > 1e-6f) {
            std::cout << "  PE[0,1] mismatch: expected " << expected_01 << ", got " << actual_01 << std::endl;
            return false;
        }
        
        // Position 1, dimension 0: sin(1 / 10000^0) = sin(1) ≈ 0.8415
        float expected_10 = std::sin(1.0f);
        float actual_10 = config.pe_table[1 * config.embedding_dim + 0];
        if (std::abs(actual_10 - expected_10) > 1e-6f) {
            std::cout << "  PE[1,0] mismatch: expected " << expected_10 << ", got " << actual_10 << std::endl;
            return false;
        }
        
        std::cout << "  PE calculation test PASSED!" << std::endl;
        return true;
    }
    
    bool test_positional_encoding_application() {
        std::cout << "Running test_positional_encoding_application..." << std::endl;
        
        PositionalEncodingTestConfig config;
        config.compute_expected_values();
        
        // Create Metal buffers
        size_t input_size = config.input_embeddings.size() * sizeof(float);
        size_t pe_size = config.pe_table.size() * sizeof(float);
        size_t output_size = config.expected_output.size() * sizeof(float);
        
        id<MTLBuffer> input_buffer = [device_ newBufferWithBytes:config.input_embeddings.data()
                                                          length:input_size
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> pe_buffer = [device_ newBufferWithBytes:config.pe_table.data()
                                                       length:pe_size
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_size
                                                           options:MTLResourceStorageModeShared];
        
        // Create constant buffers for kernel parameters
        id<MTLBuffer> batch_size_buffer = [device_ newBufferWithBytes:&config.batch_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&config.sequence_length
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> embedding_dim_buffer = [device_ newBufferWithBytes:&config.embedding_dim
                                                                  length:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        // Set the compute pipeline and buffers
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:input_buffer offset:0 atIndex:0];
        [compute_encoder setBuffer:pe_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:output_buffer offset:0 atIndex:2];
        [compute_encoder setBuffer:batch_size_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:5];
        
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
                
                // Only show first few mismatches
                if (!success) break;
            }
        }
        
        if (success) {
            std::cout << "  Positional encoding application test PASSED!" << std::endl;
        } else {
            std::cout << "  Positional encoding application test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    bool test_in_place_modification() {
        std::cout << "Running test_in_place_modification..." << std::endl;
        
        // Test modifying embeddings in-place
        PositionalEncodingTestConfig config;
        config.compute_expected_values();
        
        size_t buffer_size = config.input_embeddings.size() * sizeof(float);
        size_t pe_size = config.pe_table.size() * sizeof(float);
        
        // Create buffer with input data
        id<MTLBuffer> embeddings_buffer = [device_ newBufferWithBytes:config.input_embeddings.data()
                                                               length:buffer_size
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> pe_buffer = [device_ newBufferWithBytes:config.pe_table.data()
                                                       length:pe_size
                                                      options:MTLResourceStorageModeShared];
        
        // Use the same buffer for input and output (in-place)
        id<MTLBuffer> batch_size_buffer = [device_ newBufferWithBytes:&config.batch_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> sequence_length_buffer = [device_ newBufferWithBytes:&config.sequence_length
                                                                    length:sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> embedding_dim_buffer = [device_ newBufferWithBytes:&config.embedding_dim
                                                                  length:sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
        
        [compute_encoder setComputePipelineState:compute_pipeline_];
        [compute_encoder setBuffer:embeddings_buffer offset:0 atIndex:0];  // input
        [compute_encoder setBuffer:pe_buffer offset:0 atIndex:1];
        [compute_encoder setBuffer:embeddings_buffer offset:0 atIndex:2];  // output (same buffer)
        [compute_encoder setBuffer:batch_size_buffer offset:0 atIndex:3];
        [compute_encoder setBuffer:sequence_length_buffer offset:0 atIndex:4];
        [compute_encoder setBuffer:embedding_dim_buffer offset:0 atIndex:5];
        
        MTLSize grid_size = MTLSizeMake(config.sequence_length, config.batch_size, 1);
        MTLSize threadgroup_size = MTLSizeMake(1, 1, 1);
        
        [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [compute_encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Verify results
        float* output_data = (float*)[embeddings_buffer contents];
        
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
            std::cout << "  In-place modification test PASSED!" << std::endl;
        } else {
            std::cout << "  In-place modification test FAILED!" << std::endl;
        }
        
        return success;
    }
    
    ~PositionalEncodingTest() {
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
    std::cout << "Positional Encoding MSL Tests" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        PositionalEncodingTest test;
        
        if (!test.load_kernel()) {
            std::cout << "Failed to load Metal kernel" << std::endl;
            return 1;
        }
        
        bool success = true;
        success &= test.test_pe_calculation();
        success &= test.test_positional_encoding_application();
        success &= test.test_in_place_modification();
        
        if (success) {
            std::cout << std::endl << "All positional encoding tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << std::endl << "Some positional encoding tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 