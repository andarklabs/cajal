#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <cassert>
#include <unordered_map>

// Helper functions for half-precision conversion
uint16_t floatToHalf(float f) {
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

float halfToFloat(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

// Test dimensions (small for clarity and manual verification)
const uint32_t BATCH_SIZE = 1;
const uint32_t SEQ_LEN = 3;        // Small sequence for manual verification
const uint32_t VOCAB_SIZE = 5;     // Small vocabulary for easy tracking
const uint32_t EMBEDDING_DIM = 4;  // Small embedding dimension
const uint32_t PAD_TOKEN_ID = 0;   // Padding token ID

// CPU reference implementation for embedding layer forward pass
void embedding_forward_cpu(
    const std::vector<uint32_t>& token_ids,      // [B, S] 
    const std::vector<float>& embedding_table,   // [V, E] - vocabulary x embedding_dim
    std::vector<float>& output_embeddings,       // [B, S, E] - output embeddings
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size, uint32_t embedding_dim) {
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t token_idx = b * seq_len + s;
            uint32_t token_id = token_ids[token_idx];
            
            // Skip out-of-bounds token IDs
            if (token_id >= vocab_size) continue;
            
            uint32_t output_offset = token_idx * embedding_dim;
            uint32_t embedding_offset = token_id * embedding_dim;
            
            // Copy embedding vector
            for (uint32_t e = 0; e < embedding_dim; e++) {
                output_embeddings[output_offset + e] = embedding_table[embedding_offset + e];
            }
        }
    }
}

// CPU reference implementation for embedding layer backward pass
void embedding_backward_cpu(
    const std::vector<uint32_t>& token_ids,         // [B, S] - token IDs from forward pass
    const std::vector<float>& grad_output_embeddings, // [B, S, E] - gradient w.r.t. output embeddings
    std::vector<float>& grad_embedding_table,       // [V, E] - gradient w.r.t. embedding table (output)
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size, uint32_t embedding_dim,
    uint32_t pad_token_id) {
    
    // Initialize gradient table to zero
    std::fill(grad_embedding_table.begin(), grad_embedding_table.end(), 0.0f);
    
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            uint32_t token_idx = b * seq_len + s;
            uint32_t token_id = token_ids[token_idx];
            
            // Skip padding tokens and out-of-bounds token IDs
            if (token_id == pad_token_id || token_id >= vocab_size) continue;
            
            uint32_t grad_output_offset = token_idx * embedding_dim;
            uint32_t grad_embedding_offset = token_id * embedding_dim;
            
            // Accumulate gradients into embedding table
            for (uint32_t e = 0; e < embedding_dim; e++) {
                grad_embedding_table[grad_embedding_offset + e] += grad_output_embeddings[grad_output_offset + e];
            }
        }
    }
}

bool test_embedding_layer_backward_reference() {
    std::cout << "=== Testing Embedding Layer Backward Reference ===" << std::endl;
    
    // Create test data with known values for manual verification
    std::vector<uint32_t> token_ids = {
        1, 2, 1  // sequence: token 1, token 2, token 1 (note: token 1 appears twice)
    };
    
    // Simple embedding table for easier verification
    std::vector<float> embedding_table = {
        // Token 0 (pad): 
        0.0f, 0.0f, 0.0f, 0.0f,
        // Token 1:
        0.1f, 0.2f, 0.3f, 0.4f,
        // Token 2: 
        0.5f, 0.6f, 0.7f, 0.8f,
        // Token 3:
        0.9f, 1.0f, 1.1f, 1.2f,
        // Token 4:
        1.3f, 1.4f, 1.5f, 1.6f
    };
    
    // Forward pass to get embeddings (for reference)
    std::vector<float> output_embeddings(BATCH_SIZE * SEQ_LEN * EMBEDDING_DIM, 0.0f);
    embedding_forward_cpu(token_ids, embedding_table, output_embeddings,
                          BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM);
    
    // Create gradient w.r.t. output embeddings
    std::vector<float> grad_output_embeddings = {
        // Gradients for token 1 (position 0):
        0.1f, 0.2f, 0.3f, 0.4f,
        // Gradients for token 2 (position 1):  
        0.5f, 0.6f, 0.7f, 0.8f,
        // Gradients for token 1 (position 2):
        0.9f, 1.0f, 1.1f, 1.2f
    };
    
    // Backward pass
    std::vector<float> grad_embedding_table(VOCAB_SIZE * EMBEDDING_DIM, 0.0f);
    embedding_backward_cpu(token_ids, grad_output_embeddings, grad_embedding_table,
                          BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM, PAD_TOKEN_ID);
    
    // Print results for verification
    std::cout << "token_ids (" << token_ids.size() << "): [";
    for (size_t i = 0; i < token_ids.size(); i++) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_output_embeddings (" << grad_output_embeddings.size() << "): [";
    for (size_t i = 0; i < grad_output_embeddings.size(); i++) {
        std::cout << grad_output_embeddings[i];
        if (i < grad_output_embeddings.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "grad_embedding_table (" << grad_embedding_table.size() << "): [";
    for (size_t i = 0; i < grad_embedding_table.size(); i++) {
        std::cout << grad_embedding_table[i];
        if (i < grad_embedding_table.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Manual verification: 
    // Token 1 appears at positions 0 and 2, so its gradient should be sum of gradients at those positions
    // Expected grad for token 1: [0.1+0.9, 0.2+1.0, 0.3+1.1, 0.4+1.2] = [1.0, 1.2, 1.4, 1.6]
    // Expected grad for token 2: [0.5, 0.6, 0.7, 0.8] (appears only once)
    // Expected grad for token 0, 3, 4: [0.0, 0.0, 0.0, 0.0] (not used)
    
    std::cout << "\nManual verification:" << std::endl;
    std::cout << "Token 1 grad (expected: [1.0, 1.2, 1.4, 1.6]): [";
    for (int i = 0; i < 4; i++) {
        std::cout << grad_embedding_table[1 * EMBEDDING_DIM + i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Token 2 grad (expected: [0.5, 0.6, 0.7, 0.8]): [";
    for (int i = 0; i < 4; i++) {
        std::cout << grad_embedding_table[2 * EMBEDDING_DIM + i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "✅ Embedding Layer backward reference calculations completed." << std::endl;
    return true;
}

bool test_embedding_layer_backward_msl() {
    std::cout << "\n=== Testing Embedding Layer Backward MSL Kernel ===" << std::endl;
    
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
    
    id<MTLFunction> function = [library newFunctionWithName:@"embedding_layer_backward"];
    if (!function) {
        std::cerr << "❌ Failed to find embedding_layer_backward function" << std::endl;
        return false;
    }
    
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        std::cerr << "❌ Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create test data (same as reference test)
    std::vector<uint32_t> token_ids = {
        1, 2, 1  // sequence: token 1, token 2, token 1 (note: token 1 appears twice)
    };
    
    std::vector<float> grad_output_embeddings = {
        // Gradients for token 1 (position 0):
        0.1f, 0.2f, 0.3f, 0.4f,
        // Gradients for token 2 (position 1):  
        0.5f, 0.6f, 0.7f, 0.8f,
        // Gradients for token 1 (position 2):
        0.9f, 1.0f, 1.1f, 1.2f
    };
    
    // Convert to half precision for GPU input gradients
    std::vector<uint16_t> grad_output_embeddings_half(grad_output_embeddings.size());
    for (size_t i = 0; i < grad_output_embeddings.size(); i++) {
        grad_output_embeddings_half[i] = floatToHalf(grad_output_embeddings[i]);
    }
    
    // Create Metal buffers
    id<MTLBuffer> token_ids_buffer = [device newBufferWithBytes:token_ids.data() 
                                                        length:token_ids.size() * sizeof(uint32_t) 
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> grad_output_embeddings_buffer = [device newBufferWithBytes:grad_output_embeddings_half.data() 
                                                                     length:grad_output_embeddings_half.size() * sizeof(uint16_t) 
                                                                    options:MTLResourceStorageModeShared];
    
    // Create output buffer (using float for gradients)
    id<MTLBuffer> grad_embedding_table_buffer = [device newBufferWithLength:VOCAB_SIZE * EMBEDDING_DIM * sizeof(float) 
                                                                    options:MTLResourceStorageModeShared];
    
    // Execute kernel
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:token_ids_buffer offset:0 atIndex:0];
    [encoder setBuffer:grad_output_embeddings_buffer offset:0 atIndex:1];
    [encoder setBuffer:grad_embedding_table_buffer offset:0 atIndex:2];
    [encoder setBytes:&BATCH_SIZE length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&SEQ_LEN length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&VOCAB_SIZE length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&EMBEDDING_DIM length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&PAD_TOKEN_ID length:sizeof(uint32_t) atIndex:7];
    
    // Dispatch one thread per token in the batch
    MTLSize threadsPerGrid = MTLSizeMake(BATCH_SIZE * SEQ_LEN, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(std::min(BATCH_SIZE * SEQ_LEN, 64u), 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Read back results
    float* grad_embedding_table_data = (float*)[grad_embedding_table_buffer contents];
    std::vector<float> grad_embedding_table_msl(VOCAB_SIZE * EMBEDDING_DIM);
    
    for (size_t i = 0; i < VOCAB_SIZE * EMBEDDING_DIM; i++) {
        grad_embedding_table_msl[i] = grad_embedding_table_data[i];
    }
    
    // Compare with reference
    std::vector<float> grad_embedding_table_ref(VOCAB_SIZE * EMBEDDING_DIM, 0.0f);
    embedding_backward_cpu(token_ids, grad_output_embeddings, grad_embedding_table_ref,
                          BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM, PAD_TOKEN_ID);
    
    // Print MSL results
    std::cout << "grad_embedding_table_msl (" << grad_embedding_table_msl.size() << "): [";
    for (size_t i = 0; i < grad_embedding_table_msl.size(); i++) {
        std::cout << grad_embedding_table_msl[i];
        if (i < grad_embedding_table_msl.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Verify results with tolerance
    const float tolerance = 1e-3f; // Relaxed for half precision
    bool success = true;
    
    for (size_t i = 0; i < VOCAB_SIZE * EMBEDDING_DIM; i++) {
        if (std::abs(grad_embedding_table_msl[i] - grad_embedding_table_ref[i]) > tolerance) {
            std::cerr << "❌ grad_embedding_table mismatch at index " << i << ": " 
                      << grad_embedding_table_msl[i] << " vs " << grad_embedding_table_ref[i] << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "✅ MSL embedding_layer_backward kernel test passed!" << std::endl;
    } else {
        std::cout << "❌ MSL embedding_layer_backward kernel test failed!" << std::endl;
    }
    
    return success;
}

int main() {
    std::cout << "=== Embedding Layer Backward Pass TDD Tests ===" << std::endl;
    
    bool success = true;
    
    // Test reference implementation
    success &= test_embedding_layer_backward_reference();
    
    // Test MSL kernel
    success &= test_embedding_layer_backward_msl();
    
    if (success) {
        std::cout << "\n✅ All Embedding Layer backward tests passed!" << std::endl;
    } else {
        std::cout << "\n❌ Some Embedding Layer backward tests failed!" << std::endl;
    }
    
    return success ? 0 : 1;
} 