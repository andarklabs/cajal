#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Half precision conversion functions
static inline uint16_t float_to_half(float f) {
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

static inline float half_to_float(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

// MSL Kernel for Feed-Forward Network
const char* ffn_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

// GELU activation function implementation
float gelu(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
    const float a = 0.044715f;
    
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + a * x_cubed);
    float tanh_result = tanh(inner);
    return 0.5f * x * (1.0f + tanh_result);
}

kernel void feed_forward_network(
    device const half* input_norm [[buffer(0)]],
    device const half* W1 [[buffer(1)]],
    device const half* b1 [[buffer(2)]],
    device const half* W2 [[buffer(3)]],
    device const half* b2 [[buffer(4)]],
    device half* ffn_output [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& sequence_length [[buffer(7)]],
    constant uint& embedding_dim [[buffer(8)]],
    constant uint& ffn_hidden_dim [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x; // Each thread processes one instance (one token)
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint input_offset = instance_idx * embedding_dim;
    uint output_offset = instance_idx * embedding_dim;
    
    // Step 1: First linear layer + GELU activation
    // Hidden = GELU(Input @ W1 + b1)
    for (uint h = 0; h < ffn_hidden_dim; h++) {
        float sum = 0.0f;
        
        // Matrix multiplication: input @ W1 for hidden unit h
        for (uint e = 0; e < embedding_dim; e++) {
            float input_val = float(input_norm[input_offset + e]);
            float weight_val = float(W1[e * ffn_hidden_dim + h]); // W1 is (E x H)
            sum += input_val * weight_val;
        }
        
        // Add bias and apply GELU activation
        float hidden_val = sum + float(b1[h]);
        float activated = gelu(hidden_val);
        
        // Step 2: Second linear layer
        // Output = Hidden_Activated @ W2 + b2
        for (uint e = 0; e < embedding_dim; e++) {
            float weight_val = float(W2[h * embedding_dim + e]); // W2 is (H x E)
            
            // Accumulate contribution from this hidden unit to output element e
            // We need to use atomic operations or handle this differently for parallel writes
            // For simplicity, we'll compute the full output for element e in the inner loop
            if (h == 0) {
                // Initialize output element
                float output_sum = float(b2[e]);
                
                // Sum contributions from all hidden units
                for (uint h_inner = 0; h_inner < ffn_hidden_dim; h_inner++) {
                    // Recompute hidden activation for h_inner
                    float hidden_sum = 0.0f;
                    for (uint e_inner = 0; e_inner < embedding_dim; e_inner++) {
                        float input_val = float(input_norm[input_offset + e_inner]);
                        float weight_val = float(W1[e_inner * ffn_hidden_dim + h_inner]);
                        hidden_sum += input_val * weight_val;
                    }
                    float hidden_activated = gelu(hidden_sum + float(b1[h_inner]));
                    
                    // Add contribution to output
                    float w2_val = float(W2[h_inner * embedding_dim + e]);
                    output_sum += hidden_activated * w2_val;
                }
                
                ffn_output[output_offset + e] = half(output_sum);
            }
        }
    }
}
)";

// Test configuration
struct TestConfig {
    uint32_t batch_size = 1;
    uint32_t sequence_length = 2;
    uint32_t embedding_dim = 4;
    uint32_t ffn_hidden_dim = 8; // 2x expansion factor
};

class FFNTest {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    TestConfig config;
    
public:
    FFNTest() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to create Metal device");
        }
        
        commandQueue = [device newCommandQueue];
        
        // Create compute pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(ffn_kernel_source) 
                                                      options:nil 
                                                        error:&error];
        if (!library) {
            NSLog(@"Failed to create library: %@", error);
            throw std::runtime_error("Failed to create Metal library");
        }
        
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"feed_forward_network"];
        if (!kernelFunction) {
            throw std::runtime_error("Failed to find kernel function");
        }
        
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            NSLog(@"Failed to create pipeline state: %@", error);
            throw std::runtime_error("Failed to create compute pipeline state");
        }
    }
    
    void test_basic_ffn() {
        std::cout << "Testing basic feed-forward network..." << std::endl;
        
        // Simple input data
        std::vector<float> input_data = {
            // Batch 0, Seq 0: [0.5, 1.0, 1.5, 2.0]
            0.5f, 1.0f, 1.5f, 2.0f,
            // Batch 0, Seq 1: [1.0, 2.0, 3.0, 4.0]
            1.0f, 2.0f, 3.0f, 4.0f
        };
        
        // Identity-like weights for W1 (E x H = 4 x 8)
        std::vector<float> W1_data(config.embedding_dim * config.ffn_hidden_dim);
        for (int e = 0; e < config.embedding_dim; e++) {
            for (int h = 0; h < config.ffn_hidden_dim; h++) {
                if (h < config.embedding_dim && e == h) {
                    W1_data[e * config.ffn_hidden_dim + h] = 1.0f; // Identity for first 4 units
                } else if (h >= config.embedding_dim && e == (h - config.embedding_dim)) {
                    W1_data[e * config.ffn_hidden_dim + h] = 0.5f; // Scale for next 4 units
                } else {
                    W1_data[e * config.ffn_hidden_dim + h] = 0.0f;
                }
            }
        }
        
        // Zero biases for b1
        std::vector<float> b1_data(config.ffn_hidden_dim, 0.0f);
        
        // Identity-like weights for W2 (H x E = 8 x 4)
        std::vector<float> W2_data(config.ffn_hidden_dim * config.embedding_dim);
        for (int h = 0; h < config.ffn_hidden_dim; h++) {
            for (int e = 0; e < config.embedding_dim; e++) {
                if (h < config.embedding_dim && h == e) {
                    W2_data[h * config.embedding_dim + e] = 0.5f; // Scale down from first set
                } else if (h >= config.embedding_dim && (h - config.embedding_dim) == e) {
                    W2_data[h * config.embedding_dim + e] = 0.5f; // Scale down from second set
                } else {
                    W2_data[h * config.embedding_dim + e] = 0.0f;
                }
            }
        }
        
        // Zero biases for b2
        std::vector<float> b2_data(config.embedding_dim, 0.0f);
        
        // Convert to half precision
        std::vector<uint16_t> input_half(input_data.size());
        std::vector<uint16_t> W1_half(W1_data.size());
        std::vector<uint16_t> b1_half(b1_data.size());
        std::vector<uint16_t> W2_half(W2_data.size());
        std::vector<uint16_t> b2_half(b2_data.size());
        
        for (size_t i = 0; i < input_data.size(); i++) {
            input_half[i] = float_to_half(input_data[i]);
        }
        for (size_t i = 0; i < W1_data.size(); i++) {
            W1_half[i] = float_to_half(W1_data[i]);
        }
        for (size_t i = 0; i < b1_data.size(); i++) {
            b1_half[i] = float_to_half(b1_data[i]);
        }
        for (size_t i = 0; i < W2_data.size(); i++) {
            W2_half[i] = float_to_half(W2_data[i]);
        }
        for (size_t i = 0; i < b2_data.size(); i++) {
            b2_half[i] = float_to_half(b2_data[i]);
        }
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_half.data()
                                                        length:input_half.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> W1Buffer = [device newBufferWithBytes:W1_half.data()
                                                     length:W1_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> b1Buffer = [device newBufferWithBytes:b1_half.data()
                                                     length:b1_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> W2Buffer = [device newBufferWithBytes:W2_half.data()
                                                     length:W2_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> b2Buffer = [device newBufferWithBytes:b2_half.data()
                                                     length:b2_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:input_half.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:W1Buffer offset:0 atIndex:1];
        [encoder setBuffer:b1Buffer offset:0 atIndex:2];
        [encoder setBuffer:W2Buffer offset:0 atIndex:3];
        [encoder setBuffer:b2Buffer offset:0 atIndex:4];
        [encoder setBuffer:outputBuffer offset:0 atIndex:5];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:9];
        
        // Dispatch
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results
        uint16_t* result_data = static_cast<uint16_t*>([outputBuffer contents]);
        
        std::cout << "FFN results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int j = 0; j < config.embedding_dim; j++) {
                int idx = i * config.embedding_dim + j;
                float val = half_to_float(result_data[idx]);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "✓ Basic FFN test completed" << std::endl;
    }
    
    void test_gelu_activation() {
        std::cout << "Testing GELU activation function..." << std::endl;
        
        // Simple test with known GELU values
        std::vector<float> input_data = {
            // Test GELU with specific values
            -1.0f, 0.0f, 1.0f, 2.0f,
            -0.5f, 0.5f, 1.5f, 3.0f
        };
        
        // Create simple W1 and W2 for identity transformation to test GELU
        std::vector<float> W1_data(config.embedding_dim * config.ffn_hidden_dim);
        std::vector<float> W2_data(config.ffn_hidden_dim * config.embedding_dim);
        
        // Set up identity transformation with single hidden unit per input
        for (int i = 0; i < config.embedding_dim * config.ffn_hidden_dim; i++) {
            W1_data[i] = 0.0f;
        }
        for (int i = 0; i < config.ffn_hidden_dim * config.embedding_dim; i++) {
            W2_data[i] = 0.0f;
        }
        
        // Identity mapping for first 4 units
        for (int i = 0; i < config.embedding_dim; i++) {
            W1_data[i * config.ffn_hidden_dim + i] = 1.0f;
            W2_data[i * config.embedding_dim + i] = 1.0f;
        }
        
        std::vector<float> b1_data(config.ffn_hidden_dim, 0.0f);
        std::vector<float> b2_data(config.embedding_dim, 0.0f);
        
        // Convert to half precision
        std::vector<uint16_t> input_half(input_data.size());
        std::vector<uint16_t> W1_half(W1_data.size());
        std::vector<uint16_t> b1_half(b1_data.size());
        std::vector<uint16_t> W2_half(W2_data.size());
        std::vector<uint16_t> b2_half(b2_data.size());
        
        for (size_t i = 0; i < input_data.size(); i++) input_half[i] = float_to_half(input_data[i]);
        for (size_t i = 0; i < W1_data.size(); i++) W1_half[i] = float_to_half(W1_data[i]);
        for (size_t i = 0; i < b1_data.size(); i++) b1_half[i] = float_to_half(b1_data[i]);
        for (size_t i = 0; i < W2_data.size(); i++) W2_half[i] = float_to_half(W2_data[i]);
        for (size_t i = 0; i < b2_data.size(); i++) b2_half[i] = float_to_half(b2_data[i]);
        
        // Create buffers and execute
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_half.data()
                                                        length:input_half.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> W1Buffer = [device newBufferWithBytes:W1_half.data()
                                                     length:W1_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> b1Buffer = [device newBufferWithBytes:b1_half.data()
                                                     length:b1_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> W2Buffer = [device newBufferWithBytes:W2_half.data()
                                                     length:W2_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> b2Buffer = [device newBufferWithBytes:b2_half.data()
                                                     length:b2_half.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:input_half.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:W1Buffer offset:0 atIndex:1];
        [encoder setBuffer:b1Buffer offset:0 atIndex:2];
        [encoder setBuffer:W2Buffer offset:0 atIndex:3];
        [encoder setBuffer:b2Buffer offset:0 atIndex:4];
        [encoder setBuffer:outputBuffer offset:0 atIndex:5];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:9];
        
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        uint16_t* result_data = static_cast<uint16_t*>([outputBuffer contents]);
        
        std::cout << "GELU activation results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int j = 0; j < config.embedding_dim; j++) {
                int idx = i * config.embedding_dim + j;
                float input_val = input_data[idx];
                float output_val = half_to_float(result_data[idx]);
                
                // Expected GELU values (approximately)
                // GELU(-1.0) ≈ -0.158, GELU(0.0) = 0.0, GELU(1.0) ≈ 0.841, GELU(2.0) ≈ 1.954
                std::cout << "GELU(" << input_val << ")=" << output_val << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "✓ GELU activation test completed" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "=== Feed-Forward Network (FFN) MSL Kernel Tests ===" << std::endl;
        std::cout << "Configuration: batch_size=" << config.batch_size 
                  << ", sequence_length=" << config.sequence_length 
                  << ", embedding_dim=" << config.embedding_dim 
                  << ", ffn_hidden_dim=" << config.ffn_hidden_dim << std::endl;
        std::cout << std::endl;
        
        test_basic_ffn();
        std::cout << std::endl;
        
        test_gelu_activation();
        std::cout << std::endl;
        
        std::cout << "=== All FFN tests completed successfully! ===" << std::endl;
    }
};

int main() {
    try {
        FFNTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
} 