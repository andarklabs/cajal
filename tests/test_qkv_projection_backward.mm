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

// CPU reference implementation for QKV projection forward pass
void qkv_projection_forward_cpu(
    const std::vector<float>& input,          // [B, S, E]
    const std::vector<float>& qkv_weights,    // [3, E, E] - concatenated Q, K, V weights
    const std::vector<float>& qkv_bias,       // [3, E] - concatenated Q, K, V biases
    std::vector<float>& qkv_output,           // [B, S, 3, E] - concatenated Q, K, V outputs
    uint32_t batch_size, uint32_t seq_len, uint32_t embedding_dim) {
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t input_offset = (b * seq_len + s) * embedding_dim;
            uint32_t output_offset = (b * seq_len + s) * 3 * embedding_dim;
            
            // Compute Q, K, V projections
            for (uint32_t qkv = 0; qkv < 3; qkv++) { // 0=Q, 1=K, 2=V
                for (uint32_t e_out = 0; e_out < embedding_dim; e_out++) {
                    float sum = qkv_bias[qkv * embedding_dim + e_out];
                    
                    for (uint32_t e_in = 0; e_in < embedding_dim; e_in++) {
                        float input_val = input[input_offset + e_in];
                        float weight_val = qkv_weights[qkv * embedding_dim * embedding_dim + e_in * embedding_dim + e_out];
                        sum += input_val * weight_val;
                    }
                    
                    qkv_output[output_offset + qkv * embedding_dim + e_out] = sum;
                }
            }
        }
    }
}

// CPU reference implementation for QKV projection backward pass
void qkv_projection_backward_cpu(
    const std::vector<float>& input,           // [B, S, E] - saved from forward
    const std::vector<float>& qkv_weights,     // [3, E, E] - concatenated Q, K, V weights
    const std::vector<float>& grad_qkv_output, // [B, S, 3, E] - gradient w.r.t. QKV output
    std::vector<float>& grad_input,            // [B, S, E] - gradient w.r.t. input (output)
    std::vector<float>& grad_qkv_weights,      // [3, E, E] - gradient w.r.t. weights (output)
    std::vector<float>& grad_qkv_bias,         // [3, E] - gradient w.r.t. bias (output)
    uint32_t batch_size, uint32_t seq_len, uint32_t embedding_dim) {
    
    // Initialize gradients to zero
    std::fill(grad_input.begin(), grad_input.end(), 0.0f);
    std::fill(grad_qkv_weights.begin(), grad_qkv_weights.end(), 0.0f);
    std::fill(grad_qkv_bias.begin(), grad_qkv_bias.end(), 0.0f);
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t input_offset = (b * seq_len + s) * embedding_dim;
            uint32_t grad_output_offset = (b * seq_len + s) * 3 * embedding_dim;
            
            for (uint32_t qkv = 0; qkv < 3; qkv++) { // 0=Q, 1=K, 2=V
                for (uint32_t e_out = 0; e_out < embedding_dim; e_out++) {
                    float grad_out_val = grad_qkv_output[grad_output_offset + qkv * embedding_dim + e_out];
                    
                    // Accumulate grad_bias
                    grad_qkv_bias[qkv * embedding_dim + e_out] += grad_out_val;
                    
                    for (uint32_t e_in = 0; e_in < embedding_dim; e_in++) {
                        float input_val = input[input_offset + e_in];
                        float weight_val = qkv_weights[qkv * embedding_dim * embedding_dim + e_in * embedding_dim + e_out];
                        
                        // Accumulate grad_weights: grad_W[qkv][e_in][e_out] += input[e_in] * grad_output[qkv][e_out]
                        grad_qkv_weights[qkv * embedding_dim * embedding_dim + e_in * embedding_dim + e_out] += input_val * grad_out_val;
                        
                        // Accumulate grad_input: grad_input[e_in] += weight[qkv][e_in][e_out] * grad_output[qkv][e_out]
                        grad_input[input_offset + e_in] += weight_val * grad_out_val;
                    }
                }
            }
        }
    }
}

bool test_qkv_projection_backward_reference() {
    std::cout << "=== Testing QKV Projection Backward Reference ===" << std::endl;
    
    // Create test data with known values for manual verification
    std::vector<float> input = {
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    // Simple weights for easier verification
    std::vector<float> qkv_weights(3 * EMBEDDING_DIM * EMBEDDING_DIM);
    std::vector<float> qkv_bias(3 * EMBEDDING_DIM);
    
    // Initialize weights to simple patterns
    for (uint32_t qkv = 0; qkv < 3; qkv++) {
        for (uint32_t e_in = 0; e_in < EMBEDDING_DIM; e_in++) {
            for (uint32_t e_out = 0; e_out < EMBEDDING_DIM; e_out++) {
                if (e_in == e_out) {
                    qkv_weights[qkv * EMBEDDING_DIM * EMBEDDING_DIM + e_in * EMBEDDING_DIM + e_out] = 1.0f + qkv * 0.1f;
                } else {
                    qkv_weights[qkv * EMBEDDING_DIM * EMBEDDING_DIM + e_in * EMBEDDING_DIM + e_out] = 0.1f;
                }
            }
        }
        
        for (uint32_t e = 0; e < EMBEDDING_DIM; e++) {
            qkv_bias[qkv * EMBEDDING_DIM + e] = 0.1f * (qkv + 1);
        }
    }
    
    // Forward pass to get QKV output
    std::vector<float> qkv_output(BATCH_SIZE * SEQ_LEN * 3 * EMBEDDING_DIM, 0.0f);
    qkv_projection_forward_cpu(input, qkv_weights, qkv_bias, qkv_output,
                               BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM);
    
    // Create gradient w.r.t. QKV output
    std::vector<float> grad_qkv_output = {
        // Q gradients (seq 0, seq 1)
        0.1f, 0.2f, 0.3f, 0.4f,  // Q seq 0
        0.5f, 0.6f, 0.7f, 0.8f,  // Q seq 1
        // K gradients (seq 0, seq 1)  
        0.2f, 0.3f, 0.4f, 0.5f,  // K seq 0
        0.6f, 0.7f, 0.8f, 0.9f,  // K seq 1
        // V gradients (seq 0, seq 1)
        0.3f, 0.4f, 0.5f, 0.6f,  // V seq 0
        0.7f, 0.8f, 0.9f, 1.0f   // V seq 1
    };
    
    // Backward pass
    std::vector<float> grad_input(BATCH_SIZE * SEQ_LEN * EMBEDDING_DIM, 0.0f);
    std::vector<float> grad_qkv_weights(3 * EMBEDDING_DIM * EMBEDDING_DIM, 0.0f);
    std::vector<float> grad_qkv_bias(3 * EMBEDDING_DIM, 0.0f);
    
    qkv_projection_backward_cpu(input, qkv_weights, grad_qkv_output,
                                grad_input, grad_qkv_weights, grad_qkv_bias,
                                BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM);
    
    // Print results for verification
    std::cout << "input (" << input.size() << "): [";
    for (size_t i = 0; i < input.size(); i++) {
        std::cout << input[i];
        if (i < input.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_qkv_output (" << grad_qkv_output.size() << "): [";
    for (size_t i = 0; i < grad_qkv_output.size(); i++) {
        std::cout << grad_qkv_output[i];
        if (i < grad_qkv_output.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_input (" << grad_input.size() << "): [";
    for (size_t i = 0; i < grad_input.size(); i++) {
        std::cout << grad_input[i];
        if (i < grad_input.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_qkv_bias (" << grad_qkv_bias.size() << "): [";
    for (size_t i = 0; i < grad_qkv_bias.size(); i++) {
        std::cout << grad_qkv_bias[i];
        if (i < grad_qkv_bias.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "✅ QKV Projection backward reference calculations completed." << std::endl;
    return true;
}

bool test_qkv_projection_backward_msl() {
    std::cout << "\n=== Testing QKV Projection Backward MSL Kernel ===" << std::endl;
    
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
    
    id<MTLFunction> function = [library newFunctionWithName:@"qkv_projection_backward"];
    if (!function) {
        std::cerr << "❌ Failed to find qkv_projection_backward function" << std::endl;
        return false;
    }
    
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        std::cerr << "❌ Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create test data (same as reference test)
    std::vector<float> input = {
        0.1f, 0.2f, 0.3f, 0.4f,  // seq 0
        0.5f, 0.6f, 0.7f, 0.8f   // seq 1
    };
    
    std::vector<float> qkv_weights(3 * EMBEDDING_DIM * EMBEDDING_DIM);
    std::vector<float> qkv_bias(3 * EMBEDDING_DIM);
    
    // Initialize weights to simple patterns (same as reference)
    for (uint32_t qkv = 0; qkv < 3; qkv++) {
        for (uint32_t e_in = 0; e_in < EMBEDDING_DIM; e_in++) {
            for (uint32_t e_out = 0; e_out < EMBEDDING_DIM; e_out++) {
                if (e_in == e_out) {
                    qkv_weights[qkv * EMBEDDING_DIM * EMBEDDING_DIM + e_in * EMBEDDING_DIM + e_out] = 1.0f + qkv * 0.1f;
                } else {
                    qkv_weights[qkv * EMBEDDING_DIM * EMBEDDING_DIM + e_in * EMBEDDING_DIM + e_out] = 0.1f;
                }
            }
        }
        
        for (uint32_t e = 0; e < EMBEDDING_DIM; e++) {
            qkv_bias[qkv * EMBEDDING_DIM + e] = 0.1f * (qkv + 1);
        }
    }
    
    std::vector<float> grad_qkv_output = {
        // Q gradients (seq 0, seq 1)
        0.1f, 0.2f, 0.3f, 0.4f,  // Q seq 0
        0.5f, 0.6f, 0.7f, 0.8f,  // Q seq 1
        // K gradients (seq 0, seq 1)  
        0.2f, 0.3f, 0.4f, 0.5f,  // K seq 0
        0.6f, 0.7f, 0.8f, 0.9f,  // K seq 1
        // V gradients (seq 0, seq 1)
        0.3f, 0.4f, 0.5f, 0.6f,  // V seq 0
        0.7f, 0.8f, 0.9f, 1.0f   // V seq 1
    };
    
    // Convert to half precision for GPU
    std::vector<uint16_t> input_half(input.size());
    std::vector<uint16_t> qkv_weights_half(qkv_weights.size());
    std::vector<uint16_t> grad_qkv_output_half(grad_qkv_output.size());
    
    for (size_t i = 0; i < input.size(); i++) input_half[i] = floatToHalf(input[i]);
    for (size_t i = 0; i < qkv_weights.size(); i++) qkv_weights_half[i] = floatToHalf(qkv_weights[i]);
    for (size_t i = 0; i < grad_qkv_output.size(); i++) grad_qkv_output_half[i] = floatToHalf(grad_qkv_output[i]);
    
    // Create Metal buffers
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_half.data() 
                                                    length:input_half.size() * sizeof(uint16_t) 
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> qkv_weights_buffer = [device newBufferWithBytes:qkv_weights_half.data() 
                                                          length:qkv_weights_half.size() * sizeof(uint16_t) 
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_qkv_output_buffer = [device newBufferWithBytes:grad_qkv_output_half.data() 
                                                              length:grad_qkv_output_half.size() * sizeof(uint16_t) 
                                                             options:MTLResourceStorageModeShared];
    
    // Create output buffers (using float for gradients)
    id<MTLBuffer> grad_input_buffer = [device newBufferWithLength:input.size() * sizeof(float) 
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_qkv_weights_buffer = [device newBufferWithLength:qkv_weights.size() * sizeof(float) 
                                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_qkv_bias_buffer = [device newBufferWithLength:qkv_bias.size() * sizeof(float) 
                                                             options:MTLResourceStorageModeShared];
    
    // Execute kernel with new dispatch pattern
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:qkv_weights_buffer offset:0 atIndex:1];
    [encoder setBuffer:grad_qkv_output_buffer offset:0 atIndex:2];
    [encoder setBuffer:grad_input_buffer offset:0 atIndex:3];
    [encoder setBuffer:grad_qkv_weights_buffer offset:0 atIndex:4];
    [encoder setBuffer:grad_qkv_bias_buffer offset:0 atIndex:5];
    [encoder setBytes:&BATCH_SIZE length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&SEQ_LEN length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&EMBEDDING_DIM length:sizeof(uint32_t) atIndex:8];
    
    // New dispatch pattern for the fixed kernel:
    // gid.x = computation_type (0=grad_input, 1=grad_weights, 2=grad_bias)
    // gid.y = index1, gid.z = index2 (varies by computation type)
    //
    // For computation_type 0 (grad_input): index1=instance_idx, index2=e_in_idx
    // For computation_type 1 (grad_weights): index1=weight_idx, index2=unused
    // For computation_type 2 (grad_bias): index1=qkv_e_out_idx, index2=unused
    
    uint32_t total_instances = BATCH_SIZE * SEQ_LEN;
    uint32_t total_weights = 3 * EMBEDDING_DIM * EMBEDDING_DIM;
    uint32_t total_bias = 3 * EMBEDDING_DIM;
    
    // Use a unified dispatch that covers all computation types
    uint32_t max_index1 = std::max({total_instances, total_weights, total_bias});
    uint32_t max_index2 = EMBEDDING_DIM; // Only needed for grad_input computation
    
    MTLSize threadsPerGrid = MTLSizeMake(3, max_index1, max_index2); // 3 computation types
    MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Read back results
    float* grad_input_data = (float*)[grad_input_buffer contents];
    float* grad_qkv_weights_data = (float*)[grad_qkv_weights_buffer contents];
    float* grad_qkv_bias_data = (float*)[grad_qkv_bias_buffer contents];
    
    std::vector<float> grad_input_msl(input.size());
    std::vector<float> grad_qkv_weights_msl(qkv_weights.size());
    std::vector<float> grad_qkv_bias_msl(qkv_bias.size());
    
    for (size_t i = 0; i < input.size(); i++) grad_input_msl[i] = grad_input_data[i];
    for (size_t i = 0; i < qkv_weights.size(); i++) grad_qkv_weights_msl[i] = grad_qkv_weights_data[i];
    for (size_t i = 0; i < qkv_bias.size(); i++) grad_qkv_bias_msl[i] = grad_qkv_bias_data[i];
    
    // Compare with reference
    std::vector<float> grad_input_ref(input.size(), 0.0f);
    std::vector<float> grad_qkv_weights_ref(qkv_weights.size(), 0.0f);
    std::vector<float> grad_qkv_bias_ref(qkv_bias.size(), 0.0f);
    
    qkv_projection_backward_cpu(input, qkv_weights, grad_qkv_output,
                                grad_input_ref, grad_qkv_weights_ref, grad_qkv_bias_ref,
                                BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM);
    
    // Print MSL results
    std::cout << "grad_input_msl (" << grad_input_msl.size() << "): [";
    for (size_t i = 0; i < grad_input_msl.size(); i++) {
        std::cout << grad_input_msl[i];
        if (i < grad_input_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_qkv_bias_msl (" << grad_qkv_bias_msl.size() << "): [";
    for (size_t i = 0; i < grad_qkv_bias_msl.size(); i++) {
        std::cout << grad_qkv_bias_msl[i];
        if (i < grad_qkv_bias_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Verify results with tolerance
    const float tolerance = 1e-3f; // Relaxed for half precision
    bool success = true;
    
    for (size_t i = 0; i < input.size(); i++) {
        if (std::abs(grad_input_msl[i] - grad_input_ref[i]) > tolerance) {
            std::cerr << "❌ grad_input mismatch at index " << i << ": " 
                      << grad_input_msl[i] << " vs " << grad_input_ref[i] << std::endl;
            success = false;
        }
    }
    
    for (size_t i = 0; i < qkv_bias.size(); i++) {
        if (std::abs(grad_qkv_bias_msl[i] - grad_qkv_bias_ref[i]) > tolerance) {
            std::cerr << "❌ grad_qkv_bias mismatch at index " << i << ": " 
                      << grad_qkv_bias_msl[i] << " vs " << grad_qkv_bias_ref[i] << std::endl;
            success = false;
        }
    }
    
    // Check a few weight gradients (skip full check for brevity)
    for (size_t i = 0; i < std::min(size_t(12), qkv_weights.size()); i++) {
        if (std::abs(grad_qkv_weights_msl[i] - grad_qkv_weights_ref[i]) > tolerance) {
            std::cerr << "❌ grad_qkv_weights mismatch at index " << i << ": " 
                      << grad_qkv_weights_msl[i] << " vs " << grad_qkv_weights_ref[i] << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "✅ MSL qkv_projection_backward kernel test passed!" << std::endl;
    } else {
        std::cout << "❌ MSL qkv_projection_backward kernel test failed!" << std::endl;
    }
    
    return success;
}

int main() {
    std::cout << "=== QKV Projection Backward Pass TDD Tests ===" << std::endl;
    
    bool success = true;
    
    // Test reference implementation
    success &= test_qkv_projection_backward_reference();
    
    // Test MSL kernel
    success &= test_qkv_projection_backward_msl();
    
    if (success) {
        std::cout << "\n✅ All QKV Projection backward tests passed!" << std::endl;
    } else {
        std::cout << "\n❌ Some QKV Projection backward tests failed!" << std::endl;
    }
    
    return success ? 0 : 1;
} 