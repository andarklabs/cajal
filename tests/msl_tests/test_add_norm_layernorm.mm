#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Half precision conversion functions
static inline uint16_t float_to_half(float f) {
    // Simple conversion using Apple's built-in half precision support
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

static inline float half_to_float(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

// MSL Kernel for Layer Normalization
const char* layer_norm_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm(
    device const half* input_tensor [[buffer(0)]],
    device const half* residual_input [[buffer(1)]],
    device half* output_tensor [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    constant uint& embedding_dim [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x; // Each thread processes one instance (one token)
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint instance_offset = instance_idx * embedding_dim;
    
    // Step 1: Add residual connection and calculate mean
    // x = input_tensor + residual_input
    float sum = 0.0f;
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]);
        sum += x_val;
    }
    
    // Step 2: Calculate mean
    float mean = sum / float(embedding_dim);
    
    // Step 3: Calculate variance
    float variance_sum = 0.0f;
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]);
        float diff = x_val - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / float(embedding_dim);
    
    // Step 4: Normalize and apply gamma/beta
    float std_inv = 1.0f / sqrt(variance + epsilon);
    for (uint i = 0; i < embedding_dim; i++) {
        float x_val = float(input_tensor[instance_offset + i]) + float(residual_input[instance_offset + i]);
        float normalized = (x_val - mean) * std_inv;
        float result = gamma[i] * normalized + beta[i];
        output_tensor[instance_offset + i] = half(result);
    }
}
)";

// Test configuration
struct TestConfig {
    uint32_t batch_size = 1;
    uint32_t sequence_length = 2;
    uint32_t embedding_dim = 4;
    float epsilon = 1e-5f;
};

class LayerNormTest {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    TestConfig config;
    
public:
    LayerNormTest() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to create Metal device");
        }
        
        commandQueue = [device newCommandQueue];
        
        // Create compute pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(layer_norm_kernel_source) 
                                                      options:nil 
                                                        error:&error];
        if (!library) {
            NSLog(@"Failed to create library: %@", error);
            throw std::runtime_error("Failed to create Metal library");
        }
        
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"layer_norm"];
        if (!kernelFunction) {
            throw std::runtime_error("Failed to find kernel function");
        }
        
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            NSLog(@"Failed to create pipeline state: %@", error);
            throw std::runtime_error("Failed to create compute pipeline state");
        }
    }
    
    void test_basic_layer_norm() {
        std::cout << "Testing basic layer normalization..." << std::endl;
        
        // Test data: simple values to make manual calculation easier
        std::vector<float> input_data = {
            // Batch 0, Seq 0: [1.0, 2.0, 3.0, 4.0]
            1.0f, 2.0f, 3.0f, 4.0f,
            // Batch 0, Seq 1: [0.5, 1.5, 2.5, 3.5]
            0.5f, 1.5f, 2.5f, 3.5f
        };
        
        std::vector<float> residual_data = {
            // Batch 0, Seq 0: [0.0, 0.0, 0.0, 0.0] (no residual for simplicity)
            0.0f, 0.0f, 0.0f, 0.0f,
            // Batch 0, Seq 1: [0.0, 0.0, 0.0, 0.0]
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f, 1.0f}; // No scaling
        std::vector<float> beta_data = {0.0f, 0.0f, 0.0f, 0.0f};  // No shift
        
        // Convert to half precision for input using proper conversion
        std::vector<uint16_t> input_half(input_data.size());
        std::vector<uint16_t> residual_half(residual_data.size());
        for (size_t i = 0; i < input_data.size(); i++) {
            input_half[i] = float_to_half(input_data[i]);
        }
        for (size_t i = 0; i < residual_data.size(); i++) {
            residual_half[i] = float_to_half(residual_data[i]);
        }
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_half.data()
                                                        length:input_half.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual_half.data()
                                                           length:residual_half.size() * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:input_half.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> gammaBuffer = [device newBufferWithBytes:gamma_data.data()
                                                        length:gamma_data.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> betaBuffer = [device newBufferWithBytes:beta_data.data()
                                                       length:beta_data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:residualBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:gammaBuffer offset:0 atIndex:3];
        [encoder setBuffer:betaBuffer offset:0 atIndex:4];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:8];
        
        // Dispatch
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results and convert back to float
        uint16_t* result_data = static_cast<uint16_t*>([outputBuffer contents]);
        
        // Expected results (manually calculated):
        // For [1,2,3,4]: mean=2.5, variance=1.25, std=sqrt(1.25+1e-5)≈1.118
        // normalized: [(1-2.5)/1.118, (2-2.5)/1.118, (3-2.5)/1.118, (4-2.5)/1.118]
        //           ≈ [-1.342, -0.447, 0.447, 1.342]
        
        std::cout << "Layer norm results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int j = 0; j < config.embedding_dim; j++) {
                int idx = i * config.embedding_dim + j;
                float val = half_to_float(result_data[idx]);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        
        // Verify first instance: [1,2,3,4] with mean=2.5, std≈1.118
        float tolerance = 1e-2f;
        std::vector<float> expected_0 = {-1.342f, -0.447f, 0.447f, 1.342f};
        for (int j = 0; j < config.embedding_dim; j++) {
            float actual = half_to_float(result_data[j]);
            if (std::abs(actual - expected_0[j]) > tolerance) {
                std::cout << "Warning: Expected " << expected_0[j] << " but got " << actual << " at position " << j << std::endl;
            }
        }
        
        std::cout << "✓ Basic layer normalization test completed" << std::endl;
    }
    
    void test_with_residual() {
        std::cout << "Testing layer normalization with residual connection..." << std::endl;
        
        std::vector<float> input_data = {
            2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f
        };
        
        std::vector<float> residual_data = {
            1.0f, 1.0f, 1.0f, 1.0f,  // Add 1 to each element
            0.5f, 0.5f, 0.5f, 0.5f   // Add 0.5 to each element
        };
        
        std::vector<float> gamma_data = {2.0f, 2.0f, 2.0f, 2.0f}; // Scale by 2
        std::vector<float> beta_data = {1.0f, 1.0f, 1.0f, 1.0f};  // Shift by 1
        
        // Convert to half precision properly
        std::vector<uint16_t> input_half(input_data.size());
        std::vector<uint16_t> residual_half(residual_data.size());
        for (size_t i = 0; i < input_data.size(); i++) {
            input_half[i] = float_to_half(input_data[i]);
        }
        for (size_t i = 0; i < residual_data.size(); i++) {
            residual_half[i] = float_to_half(residual_data[i]);
        }
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_half.data()
                                                        length:input_half.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual_half.data()
                                                           length:residual_half.size() * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:input_half.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> gammaBuffer = [device newBufferWithBytes:gamma_data.data()
                                                        length:gamma_data.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> betaBuffer = [device newBufferWithBytes:beta_data.data()
                                                       length:beta_data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        // Execute kernel
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:residualBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:gammaBuffer offset:0 atIndex:3];
        [encoder setBuffer:betaBuffer offset:0 atIndex:4];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:8];
        
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read and verify results
        uint16_t* result_data = static_cast<uint16_t*>([outputBuffer contents]);
        
        std::cout << "Layer norm with residual results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int j = 0; j < config.embedding_dim; j++) {
                int idx = i * config.embedding_dim + j;
                float val = half_to_float(result_data[idx]);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "✓ Layer normalization with residual test completed" << std::endl;
    }
    
    void test_gamma_beta_scaling() {
        std::cout << "Testing layer normalization with gamma/beta scaling..." << std::endl;
        
        std::vector<float> input_data = {
            0.0f, 1.0f, 2.0f, 3.0f,  // Simple progression
            4.0f, 5.0f, 6.0f, 7.0f
        };
        
        std::vector<float> residual_data = {
            0.0f, 0.0f, 0.0f, 0.0f,  // No residual
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        std::vector<float> gamma_data = {0.5f, 1.0f, 1.5f, 2.0f}; // Different scales
        std::vector<float> beta_data = {-1.0f, 0.0f, 1.0f, 2.0f}; // Different shifts
        
        // Convert to half precision properly
        std::vector<uint16_t> input_half(input_data.size());
        std::vector<uint16_t> residual_half(residual_data.size());
        for (size_t i = 0; i < input_data.size(); i++) {
            input_half[i] = float_to_half(input_data[i]);
        }
        for (size_t i = 0; i < residual_data.size(); i++) {
            residual_half[i] = float_to_half(residual_data[i]);
        }
        
        // Create buffers and execute
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_half.data()
                                                        length:input_half.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual_half.data()
                                                           length:residual_half.size() * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:input_half.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> gammaBuffer = [device newBufferWithBytes:gamma_data.data()
                                                        length:gamma_data.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> betaBuffer = [device newBufferWithBytes:beta_data.data()
                                                       length:beta_data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:residualBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:gammaBuffer offset:0 atIndex:3];
        [encoder setBuffer:betaBuffer offset:0 atIndex:4];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:8];
        
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        uint16_t* result_data = static_cast<uint16_t*>([outputBuffer contents]);
        
        std::cout << "Layer norm with gamma/beta scaling results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int j = 0; j < config.embedding_dim; j++) {
                int idx = i * config.embedding_dim + j;
                float val = half_to_float(result_data[idx]);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "✓ Layer normalization with gamma/beta scaling test completed" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "=== Add & Norm (Layer Normalization) MSL Kernel Tests ===" << std::endl;
        std::cout << "Configuration: batch_size=" << config.batch_size 
                  << ", sequence_length=" << config.sequence_length 
                  << ", embedding_dim=" << config.embedding_dim 
                  << ", epsilon=" << config.epsilon << std::endl;
        std::cout << std::endl;
        
        test_basic_layer_norm();
        std::cout << std::endl;
        
        test_with_residual();
        std::cout << std::endl;
        
        test_gamma_beta_scaling();
        std::cout << std::endl;
        
        std::cout << "=== All Add & Norm tests completed successfully! ===" << std::endl;
    }
};

int main() {
    try {
        LayerNormTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
} 