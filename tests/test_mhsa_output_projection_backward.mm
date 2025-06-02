#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric> // For std::iota

// Helper function to convert half to float
float halfToFloat(uint16_t h) {
    __fp16* hp = (__fp16*)&h;
    return (float)(*hp);
}

// Helper function to convert float to half
uint16_t floatToHalf(float f) {
    __fp16 h = (__fp16)f;
    return *((uint16_t*)&h);
}

// Test dimensions
const uint32_t BATCH_SIZE = 1;
const uint32_t SEQ_LEN = 1; // Using 1 for simplicity in a single token test
const uint32_t EMBEDDING_DIM = 4; // Example embedding dim (num_heads * head_dim)
// NUM_HEADS and HEAD_DIM are not directly used here as we operate on concatenated heads

// Helper to print vectors
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec, int limit = 16) {
    std::cout << name << " (" << vec.size() << "): [";
    for (size_t i = 0; i < std::min((size_t)limit, vec.size()); ++i) {
        std::cout << vec[i] << (i == std::min((size_t)limit, vec.size()) - 1 ? "" : ", ");
    }
    if (vec.size() > limit) std::cout << "...";
    std::cout << "]" << std::endl;
}

// Forward pass reference: Y = X @ W + b
// X: concatenated_attention_heads [B*S, E]
// W: W_o [E, E]
// b: b_o [E]
// Y: mhsa_output [B*S, E]
std::vector<float> mhsa_output_projection_forward_cpu(
    const std::vector<float>& X,
    const std::vector<float>& W_o,
    const std::vector<float>& b_o,
    uint32_t num_instances, // BATCH_SIZE * SEQ_LEN
    uint32_t embedding_dim
) {
    std::vector<float> Y(num_instances * embedding_dim);
    for (uint32_t i = 0; i < num_instances; ++i) {
        for (uint32_t j = 0; j < embedding_dim; ++j) { // Output feature
            float sum = b_o[j];
            for (uint32_t k = 0; k < embedding_dim; ++k) { // Input feature
                sum += X[i * embedding_dim + k] * W_o[k * embedding_dim + j];
            }
            Y[i * embedding_dim + j] = sum;
        }
    }
    return Y;
}

// Backward pass reference:
// dL/dX = dL/dY @ W_o^T
// dL/dW_o = X^T @ dL/dY
// dL/db_o = sum_rows(dL/dY)
void mhsa_output_projection_backward_cpu(
    const std::vector<float>& dL_dY, // grad_mhsa_output
    const std::vector<float>& X,     // concatenated_attention_heads
    const std::vector<float>& W_o,
    uint32_t num_instances,
    uint32_t embedding_dim,
    std::vector<float>& dL_dX,       // out: grad_concatenated_attention_heads
    std::vector<float>& dL_dW_o,     // out: grad_W_o
    std::vector<float>& dL_db_o      // out: grad_b_o
) {
    dL_dX.assign(num_instances * embedding_dim, 0.0f);
    dL_dW_o.assign(embedding_dim * embedding_dim, 0.0f);
    dL_db_o.assign(embedding_dim, 0.0f);

    // dL/db_o
    for (uint32_t i = 0; i < num_instances; ++i) {
        for (uint32_t j = 0; j < embedding_dim; ++j) {
            dL_db_o[j] += dL_dY[i * embedding_dim + j];
        }
    }

    // dL/dW_o and dL/dX
    for (uint32_t i = 0; i < num_instances; ++i) { // Instance
        for (uint32_t j = 0; j < embedding_dim; ++j) { // Output feature (dL_dY index) / W_o col index
            float grad_y_ij = dL_dY[i * embedding_dim + j];
            for (uint32_t k = 0; k < embedding_dim; ++k) { // Input feature (X index) / W_o row index
                // dL/dW_o[k,j] += X[i,k] * dL/dY[i,j]
                dL_dW_o[k * embedding_dim + j] += X[i * embedding_dim + k] * grad_y_ij;
                // dL/dX[i,k] += dL/dY[i,j] * W_o[k,j]
                dL_dX[i * embedding_dim + k] += grad_y_ij * W_o[k * embedding_dim + j];
            }
        }
    }
}

bool test_mhsa_output_projection_reference() {
    std::cout << "\\n=== Testing MHSA Output Projection Backward Reference ===\\n";
    uint32_t num_instances = BATCH_SIZE * SEQ_LEN;

    std::vector<float> X(num_instances * EMBEDDING_DIM);
    std::vector<float> W_o(EMBEDDING_DIM * EMBEDDING_DIM);
    std::vector<float> b_o(EMBEDDING_DIM);
    std::vector<float> dL_dY(num_instances * EMBEDDING_DIM);

    // Initialize with some patterned data
    std::iota(X.begin(), X.end(), 0.1f);
    std::iota(W_o.begin(), W_o.end(), 0.05f);
    std::iota(b_o.begin(), b_o.end(), 0.01f);
    std::iota(dL_dY.begin(), dL_dY.end(), 1.0f);
    for(auto& val : X) val *= 0.1f;
    for(auto& val : W_o) val *= 0.1f;
    for(auto& val : b_o) val *= 0.1f;
    for(auto& val : dL_dY) val *= 0.2f;


    print_vector("X (input heads)", X);
    print_vector("W_o", W_o);
    print_vector("b_o", b_o);
    print_vector("dL/dY (incoming grad)", dL_dY);
    
    std::vector<float> Y_cpu = mhsa_output_projection_forward_cpu(X, W_o, b_o, num_instances, EMBEDDING_DIM);
    print_vector("Y_cpu (forward output)", Y_cpu);

    std::vector<float> dL_dX_cpu, dL_dW_o_cpu, dL_db_o_cpu;
    mhsa_output_projection_backward_cpu(dL_dY, X, W_o, num_instances, EMBEDDING_DIM, dL_dX_cpu, dL_dW_o_cpu, dL_db_o_cpu);

    print_vector("dL/dX_cpu", dL_dX_cpu);
    print_vector("dL/dW_o_cpu", dL_dW_o_cpu);
    print_vector("dL/db_o_cpu", dL_db_o_cpu);
    
    // Basic sanity checks (not exhaustive validation)
    if (dL_dX_cpu.empty() || dL_dW_o_cpu.empty() || dL_db_o_cpu.empty()) {
        std::cerr << "❌ CPU backward pass produced empty gradients." << std::endl;
        return false;
    }
    std::cout << "✅ MHSA Output Projection backward reference calculations completed." << std::endl;
    return true;
}


bool test_mhsa_output_projection_backward_msl() {
    std::cout << "\\n=== Testing MHSA Output Projection Backward MSL Kernel ===\\n";
    uint32_t num_instances = BATCH_SIZE * SEQ_LEN;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { std::cerr << "❌ Failed to create Metal device" << std::endl; return false; }
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) { std::cerr << "❌ Failed to create command queue" << std::endl; return false; }

    std::ifstream kernelFile("../src/msl/backward_kernels.msl");
    if (!kernelFile.is_open()) { std::cerr << "❌ Failed to open backward_kernels.msl" << std::endl; return false; }
    std::string mslSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    kernelFile.close();

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:mslSource.c_str()] options:nil error:&error];
    if (!library) { NSLog(@"❌ Failed to compile MSL library: %@", error.localizedDescription); return false; }

    id<MTLFunction> function = [library newFunctionWithName:@"mhsa_output_projection_backward"];
    if (!function) { std::cerr << "❌ Failed to find mhsa_output_projection_backward function" << std::endl; return false; }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) { NSLog(@"❌ Failed to create compute pipeline: %@", error.localizedDescription); return false; }

    // Prepare data (same as reference)
    std::vector<float> X_f(num_instances * EMBEDDING_DIM);
    std::vector<float> W_o_f(EMBEDDING_DIM * EMBEDDING_DIM);
    std::vector<float> b_o_f(EMBEDDING_DIM); // Bias is not used by this specific backward kernel but good to have for completeness
    std::vector<float> dL_dY_f(num_instances * EMBEDDING_DIM);

    std::iota(X_f.begin(), X_f.end(), 0.1f);
    std::iota(W_o_f.begin(), W_o_f.end(), 0.05f);
    std::iota(b_o_f.begin(), b_o_f.end(), 0.01f); // Not directly used in W_o or X grad by this kernel, but grad_b_o is computed
    std::iota(dL_dY_f.begin(), dL_dY_f.end(), 1.0f);
    for(auto& val : X_f) val *= 0.1f;
    for(auto& val : W_o_f) val *= 0.1f;
    // for(auto& val : b_o_f) val *= 0.1f; // b_o itself not input to backward kernel for W, X grads
    for(auto& val : dL_dY_f) val *= 0.2f;

    // Convert to half
    std::vector<uint16_t> X_h(X_f.size());
    std::vector<uint16_t> W_o_h(W_o_f.size());
    std::vector<uint16_t> dL_dY_h(dL_dY_f.size());
    for(size_t i=0; i<X_f.size(); ++i) X_h[i] = floatToHalf(X_f[i]);
    for(size_t i=0; i<W_o_f.size(); ++i) W_o_h[i] = floatToHalf(W_o_f[i]);
    for(size_t i=0; i<dL_dY_f.size(); ++i) dL_dY_h[i] = floatToHalf(dL_dY_f[i]);

    // Buffers
    id<MTLBuffer> dL_dY_buf = [device newBufferWithBytes:dL_dY_h.data() length:dL_dY_h.size()*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> X_buf = [device newBufferWithBytes:X_h.data() length:X_h.size()*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> W_o_buf = [device newBufferWithBytes:W_o_h.data() length:W_o_h.size()*sizeof(uint16_t) options:MTLResourceStorageModeShared];

    id<MTLBuffer> dL_dX_buf = [device newBufferWithLength:X_h.size()*sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> dL_dW_o_buf = [device newBufferWithLength:W_o_f.size()*sizeof(float) options:MTLResourceStorageModeShared]; // Grads for weights often float
    id<MTLBuffer> dL_db_o_buf = [device newBufferWithLength:b_o_f.size()*sizeof(float) options:MTLResourceStorageModeShared];

    // Zero grad buffers for accumulation
    memset([dL_dW_o_buf contents], 0, [dL_dW_o_buf length]);
    memset([dL_db_o_buf contents], 0, [dL_db_o_buf length]);
    
    // CPU reference for comparison
    std::vector<float> dL_dX_cpu, dL_dW_o_cpu, dL_db_o_cpu;
    mhsa_output_projection_backward_cpu(dL_dY_f, X_f, W_o_f, num_instances, EMBEDDING_DIM, dL_dX_cpu, dL_dW_o_cpu, dL_db_o_cpu);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dL_dY_buf offset:0 atIndex:0];
    [encoder setBuffer:X_buf offset:0 atIndex:1];
    [encoder setBuffer:W_o_buf offset:0 atIndex:2];
    [encoder setBuffer:dL_dW_o_buf offset:0 atIndex:3];
    [encoder setBuffer:dL_db_o_buf offset:0 atIndex:4];
    [encoder setBuffer:dL_dX_buf offset:0 atIndex:5];
    
    uint32_t b = BATCH_SIZE, s = SEQ_LEN, e_dim = EMBEDDING_DIM;
    [encoder setBytes:&b length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&s length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&e_dim length:sizeof(uint32_t) atIndex:8];

    // Dispatch strategy: One thread per (instance, output_embedding_dim_idx, input_embedding_dim_idx) for dL/dW and dL/dX parts.
    // Simpler: One thread per instance, loop internally.
    // Or, one thread per (instance, e_out) for dL/dX.
    // And for dL/dW, dL/db, use atomic adds, grid over (instance, e_out, e_in) or similar.
    // Kernel expects gid.x = instance_idx, gid.y = e_idx (for dL/dX and dL/dW row), gid.z = v_idx (for dL/db and dL/dW col)
    MTLSize gridSize = MTLSizeMake(num_instances, EMBEDDING_DIM, EMBEDDING_DIM); 
    uint32_t threads_per_group_dim = 1;
    MTLSize threadgroupSize = MTLSizeMake(threads_per_group_dim, threads_per_group_dim, threads_per_group_dim);
    // Cap threadgroup size if too large
    if (pipeline.maxTotalThreadsPerThreadgroup < threads_per_group_dim * threads_per_group_dim * threads_per_group_dim) {
        threadgroupSize = MTLSizeMake(1,1,1); // Fallback
    }
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (commandBuffer.status == MTLCommandBufferStatusError) {
        NSLog(@"❌ Command buffer execution failed: %@", commandBuffer.error.localizedDescription);
        return false;
    }

    // Extract results
    uint16_t* dL_dX_msl_h = (uint16_t*)[dL_dX_buf contents];
    float* dL_dW_o_msl_f = (float*)[dL_dW_o_buf contents];
    float* dL_db_o_msl_f = (float*)[dL_db_o_buf contents];

    std::vector<float> dL_dX_msl_f(X_f.size());
    for(size_t i=0; i<X_f.size(); ++i) dL_dX_msl_f[i] = halfToFloat(dL_dX_msl_h[i]);

    print_vector("dL/dX_msl", dL_dX_msl_f);
    print_vector("dL/dW_o_msl", std::vector<float>(dL_dW_o_msl_f, dL_dW_o_msl_f + dL_dW_o_cpu.size()));
    print_vector("dL/db_o_msl", std::vector<float>(dL_db_o_msl_f, dL_db_o_msl_f + dL_db_o_cpu.size()));
    
    bool success = true;
    float tolerance = 1e-2f; // Higher tolerance for half precision and accumulated sums

    auto compare_vectors = [&](const std::string& name, const std::vector<float>& cpu, const std::vector<float>& msl) {
        if(cpu.size() != msl.size()){
            std::cerr << "❌ " << name << " size mismatch! CPU: " << cpu.size() << ", MSL: " << msl.size() << std::endl;
            success = false;
            return;
        }
        for (size_t i = 0; i < cpu.size(); ++i) {
            if (std::abs(cpu[i] - msl[i]) > tolerance) {
                 if (std::abs(cpu[i] - msl[i]) / (std::abs(cpu[i]) + 1e-9f) > tolerance ) { // Relative error
                    std::cerr << "❌ " << name << "[" << i << "] mismatch. CPU: " << cpu[i] << ", MSL: " << msl[i] << ", Diff: " << std::abs(cpu[i] - msl[i]) << std::endl;
                    success = false;
                 }
            }
        }
    };
    
    compare_vectors("dL/dX", dL_dX_cpu, dL_dX_msl_f);
    compare_vectors("dL/dW_o", dL_dW_o_cpu, std::vector<float>(dL_dW_o_msl_f, dL_dW_o_msl_f + dL_dW_o_cpu.size()));
    compare_vectors("dL/db_o", dL_db_o_cpu, std::vector<float>(dL_db_o_msl_f, dL_db_o_msl_f + dL_db_o_cpu.size()));

    if (success) {
        std::cout << "✅ MSL mhsa_output_projection_backward kernel test passed!" << std::endl;
    } else {
        std::cout << "❌ MSL mhsa_output_projection_backward kernel test FAILED!" << std::endl;
    }
    return success;
}


int main() {
    std::cout << "=== MHSA Output Projection Backward Pass TDD Tests ===\\n";
    bool all_tests_passed = true;
    
    all_tests_passed &= test_mhsa_output_projection_reference();
    // Placeholder for MSL test call - will be enabled after kernel implementation
    all_tests_passed &= test_mhsa_output_projection_backward_msl(); 

    if (all_tests_passed) {
        std::cout << "\\n✅ All MHSA Output Projection backward tests passed!\\n";
        return 0;
    } else {
        std::cout << "\\n❌ Some MHSA Output Projection backward tests FAILED!\\n";
        return 1;
    }
} 