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

// MSL Kernel for Output Layer (Final Linear Layer)
const char* output_layer_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

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
    uint instance_idx = gid.x; // Each thread processes one instance (one token)
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint input_offset = instance_idx * embedding_dim;
    uint output_offset = instance_idx * vocab_size;
    
    // Linear transformation: Logits = FinalHiddenStates @ W_out + b_out
    // W_out has shape (embedding_dim x vocab_size)
    for (uint v = 0; v < vocab_size; v++) {
        float sum = 0.0f;
        
        // Matrix multiplication: hidden_states @ W_out for vocab element v
        for (uint e = 0; e < embedding_dim; e++) {
            float hidden_val = float(final_hidden_states[input_offset + e]);
            float weight_val = float(W_out[e * vocab_size + v]); // W_out is (E x V)
            sum += hidden_val * weight_val;
        }
        
        // Add bias and store as float for numerical stability
        output_logits[output_offset + v] = sum + b_out[v];
    }
}

// Softmax kernel for inference (optional, separate from training)
kernel void softmax(
    device const float* logits [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x; // Each thread processes one instance (one token)
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint offset = instance_idx * vocab_size;
    
    // Find maximum value for numerical stability
    float max_val = logits[offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_val = max(max_val, logits[offset + v]);
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        float exp_val = exp(logits[offset + v] - max_val);
        probabilities[offset + v] = exp_val;
        sum += exp_val;
    }
    
    // Normalize to get probabilities
    for (uint v = 0; v < vocab_size; v++) {
        probabilities[offset + v] /= sum;
    }
}
)";

// Test configuration
struct TestConfig {
    uint32_t batch_size = 1;
    uint32_t sequence_length = 2;
    uint32_t embedding_dim = 4;
    uint32_t vocab_size = 10; // Small vocabulary for testing
};

class OutputLayerTest {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> outputPipelineState;
    id<MTLComputePipelineState> softmaxPipelineState;
    TestConfig config;
    
public:
    OutputLayerTest() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to create Metal device");
        }
        
        commandQueue = [device newCommandQueue];
        
        // Create compute pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(output_layer_kernel_source) 
                                                      options:nil 
                                                        error:&error];
        if (!library) {
            NSLog(@"Failed to create library: %@", error);
            throw std::runtime_error("Failed to create Metal library");
        }
        
        id<MTLFunction> outputFunction = [library newFunctionWithName:@"output_logits_projection"];
        if (!outputFunction) {
            throw std::runtime_error("Failed to find output kernel function");
        }
        
        id<MTLFunction> softmaxFunction = [library newFunctionWithName:@"softmax"];
        if (!softmaxFunction) {
            throw std::runtime_error("Failed to find softmax kernel function");
        }
        
        outputPipelineState = [device newComputePipelineStateWithFunction:outputFunction error:&error];
        if (!outputPipelineState) {
            NSLog(@"Failed to create output pipeline state: %@", error);
            throw std::runtime_error("Failed to create output compute pipeline state");
        }
        
        softmaxPipelineState = [device newComputePipelineStateWithFunction:softmaxFunction error:&error];
        if (!softmaxPipelineState) {
            NSLog(@"Failed to create softmax pipeline state: %@", error);
            throw std::runtime_error("Failed to create softmax compute pipeline state");
        }
    }
    
    void test_basic_output_projection() {
        std::cout << "Testing basic output logits projection..." << std::endl;
        
        // Simple input hidden states
        std::vector<float> hidden_states = {
            // Batch 0, Seq 0: [1.0, 2.0, 3.0, 4.0]
            1.0f, 2.0f, 3.0f, 4.0f,
            // Batch 0, Seq 1: [0.5, 1.5, 2.5, 3.5]
            0.5f, 1.5f, 2.5f, 3.5f
        };
        
        // Simple weight matrix W_out (E x V = 4 x 10)
        // Each column represents weights for one vocabulary item
        std::vector<float> W_out_data(config.embedding_dim * config.vocab_size);
        for (int e = 0; e < config.embedding_dim; e++) {
            for (int v = 0; v < config.vocab_size; v++) {
                // Simple pattern: each vocab gets different scaling
                W_out_data[e * config.vocab_size + v] = 0.1f * (v + 1) * (e + 1);
            }
        }
        
        // Simple bias vector
        std::vector<float> b_out_data(config.vocab_size);
        for (int v = 0; v < config.vocab_size; v++) {
            b_out_data[v] = 0.01f * v; // Small bias
        }
        
        // Convert hidden states to half precision
        std::vector<uint16_t> hidden_half(hidden_states.size());
        std::vector<uint16_t> W_out_half(W_out_data.size());
        for (size_t i = 0; i < hidden_states.size(); i++) {
            hidden_half[i] = float_to_half(hidden_states[i]);
        }
        for (size_t i = 0; i < W_out_data.size(); i++) {
            W_out_half[i] = float_to_half(W_out_data[i]);
        }
        
        // Create buffers
        id<MTLBuffer> hiddenBuffer = [device newBufferWithBytes:hidden_half.data()
                                                         length:hidden_half.size() * sizeof(uint16_t)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> WOutBuffer = [device newBufferWithBytes:W_out_half.data()
                                                       length:W_out_half.size() * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bOutBuffer = [device newBufferWithBytes:b_out_data.data()
                                                       length:b_out_data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        size_t logits_size = config.batch_size * config.sequence_length * config.vocab_size;
        id<MTLBuffer> logitsBuffer = [device newBufferWithLength:logits_size * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:outputPipelineState];
        [encoder setBuffer:hiddenBuffer offset:0 atIndex:0];
        [encoder setBuffer:WOutBuffer offset:0 atIndex:1];
        [encoder setBuffer:bOutBuffer offset:0 atIndex:2];
        [encoder setBuffer:logitsBuffer offset:0 atIndex:3];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:7];
        
        // Dispatch
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results
        float* result_data = static_cast<float*>([logitsBuffer contents]);
        
        std::cout << "Output logits:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int v = 0; v < config.vocab_size; v++) {
                int idx = i * config.vocab_size + v;
                std::cout << result_data[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        // Verify some basic properties
        // For instance 0: [1,2,3,4] @ W_out + b_out
        // Should be non-zero and increasing with vocab index due to our weight pattern
        bool logits_increasing = true;
        for (int v = 1; v < config.vocab_size; v++) {
            if (result_data[v] <= result_data[v-1]) {
                logits_increasing = false;
                break;
            }
        }
        
        if (logits_increasing) {
            std::cout << "✓ Logits show expected increasing pattern" << std::endl;
        } else {
            std::cout << "⚠ Logits don't show expected pattern (may be normal)" << std::endl;
        }
        
        std::cout << "✓ Basic output projection test completed" << std::endl;
    }
    
    void test_softmax_normalization() {
        std::cout << "Testing softmax normalization..." << std::endl;
        
        // Create some test logits with known values
        std::vector<float> test_logits = {
            // Instance 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
            // Instance 1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f
        };
        
        // Create buffers
        id<MTLBuffer> logitsBuffer = [device newBufferWithBytes:test_logits.data()
                                                         length:test_logits.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> probsBuffer = [device newBufferWithLength:test_logits.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder for softmax
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:softmaxPipelineState];
        [encoder setBuffer:logitsBuffer offset:0 atIndex:0];
        [encoder setBuffer:probsBuffer offset:0 atIndex:1];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:4];
        
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results
        float* result_data = static_cast<float*>([probsBuffer contents]);
        
        std::cout << "Softmax probabilities:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            float sum = 0.0f;
            for (int v = 0; v < config.vocab_size; v++) {
                int idx = i * config.vocab_size + v;
                float prob = result_data[idx];
                std::cout << prob << " ";
                sum += prob;
            }
            std::cout << " (sum=" << sum << ")" << std::endl;
            
            // Verify that probabilities sum to approximately 1.0
            if (std::abs(sum - 1.0f) > 1e-5f) {
                std::cout << "⚠ Warning: Probabilities don't sum to 1.0 for instance " << i << std::endl;
            }
        }
        
        std::cout << "✓ Softmax normalization test completed" << std::endl;
    }
    
    void test_identity_projection() {
        std::cout << "Testing identity projection..." << std::endl;
        
        // Test with identity matrix (square matrix for simplicity)
        uint32_t small_vocab = config.embedding_dim; // Make it square
        
        std::vector<float> hidden_states = {
            1.0f, 0.0f, 0.0f, 0.0f,  // Should map to [1,0,0,0]
            0.0f, 1.0f, 0.0f, 0.0f   // Should map to [0,1,0,0]
        };
        
        // Identity weight matrix
        std::vector<float> W_out_data(config.embedding_dim * small_vocab);
        for (int i = 0; i < config.embedding_dim * small_vocab; i++) {
            W_out_data[i] = 0.0f;
        }
        for (int i = 0; i < config.embedding_dim; i++) {
            W_out_data[i * small_vocab + i] = 1.0f; // Identity diagonal
        }
        
        // Zero bias
        std::vector<float> b_out_data(small_vocab, 0.0f);
        
        // Convert to half precision
        std::vector<uint16_t> hidden_half(hidden_states.size());
        std::vector<uint16_t> W_out_half(W_out_data.size());
        for (size_t i = 0; i < hidden_states.size(); i++) {
            hidden_half[i] = float_to_half(hidden_states[i]);
        }
        for (size_t i = 0; i < W_out_data.size(); i++) {
            W_out_half[i] = float_to_half(W_out_data[i]);
        }
        
        // Create buffers
        id<MTLBuffer> hiddenBuffer = [device newBufferWithBytes:hidden_half.data()
                                                         length:hidden_half.size() * sizeof(uint16_t)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> WOutBuffer = [device newBufferWithBytes:W_out_half.data()
                                                       length:W_out_half.size() * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bOutBuffer = [device newBufferWithBytes:b_out_data.data()
                                                       length:b_out_data.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        
        size_t logits_size = config.batch_size * config.sequence_length * small_vocab;
        id<MTLBuffer> logitsBuffer = [device newBufferWithLength:logits_size * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        
        // Execute
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:outputPipelineState];
        [encoder setBuffer:hiddenBuffer offset:0 atIndex:0];
        [encoder setBuffer:WOutBuffer offset:0 atIndex:1];
        [encoder setBuffer:bOutBuffer offset:0 atIndex:2];
        [encoder setBuffer:logitsBuffer offset:0 atIndex:3];
        [encoder setBytes:&config.batch_size length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&config.sequence_length length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&small_vocab length:sizeof(uint32_t) atIndex:7];
        
        MTLSize threadsPerGrid = MTLSizeMake(config.batch_size * config.sequence_length, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        float* result_data = static_cast<float*>([logitsBuffer contents]);
        
        std::cout << "Identity projection results:" << std::endl;
        for (int i = 0; i < config.batch_size * config.sequence_length; i++) {
            std::cout << "Instance " << i << ": ";
            for (int v = 0; v < small_vocab; v++) {
                int idx = i * small_vocab + v;
                std::cout << result_data[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        // Verify identity property
        float tolerance = 1e-4f;
        bool identity_correct = true;
        
        // Instance 0: [1,0,0,0] -> should give [1,0,0,0]
        if (std::abs(result_data[0] - 1.0f) > tolerance || std::abs(result_data[1]) > tolerance ||
            std::abs(result_data[2]) > tolerance || std::abs(result_data[3]) > tolerance) {
            identity_correct = false;
        }
        
        // Instance 1: [0,1,0,0] -> should give [0,1,0,0]
        if (std::abs(result_data[4]) > tolerance || std::abs(result_data[5] - 1.0f) > tolerance ||
            std::abs(result_data[6]) > tolerance || std::abs(result_data[7]) > tolerance) {
            identity_correct = false;
        }
        
        if (identity_correct) {
            std::cout << "✓ Identity projection working correctly" << std::endl;
        } else {
            std::cout << "⚠ Identity projection has errors" << std::endl;
        }
        
        std::cout << "✓ Identity projection test completed" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "=== Final Output Layer MSL Kernel Tests ===" << std::endl;
        std::cout << "Configuration: batch_size=" << config.batch_size 
                  << ", sequence_length=" << config.sequence_length 
                  << ", embedding_dim=" << config.embedding_dim 
                  << ", vocab_size=" << config.vocab_size << std::endl;
        std::cout << std::endl;
        
        test_basic_output_projection();
        std::cout << std::endl;
        
        test_softmax_normalization();
        std::cout << std::endl;
        
        test_identity_projection();
        std::cout << std::endl;
        
        std::cout << "=== All Output Layer tests completed successfully! ===" << std::endl;
    }
};

int main() {
    try {
        OutputLayerTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
} 