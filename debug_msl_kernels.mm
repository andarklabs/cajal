#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>

// Test individual MSL kernels to identify the problematic one
int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return 1;
    }
    
    std::cout << "Testing MSL kernels individually..." << std::endl;
    
    // Test 1: Basic kernels (should work)
    {
        const char* basic_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void test_kernel(device float* buffer [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    buffer[gid] = float(gid);
}
)";
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(basic_msl) options:nil error:&error];
        if (!library) {
            std::cerr << "❌ Basic kernel test failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        std::cout << "✓ Basic kernel test passed" << std::endl;
    }
    
    // Test 2: Cross-entropy loss kernel (suspect)
    {
        const char* loss_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss(
    device const float* logits [[buffer(0)]],
    device const uint32_t* target_ids [[buffer(1)]],
    device float* per_token_loss [[buffer(2)]],
    device float* total_loss [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    constant uint& pad_token_id [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint seq_idx = gid.y;
    
    if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
    
    uint token_idx = batch_idx * sequence_length + seq_idx;
    uint32_t target_id = target_ids[token_idx];
    
    if (target_id == pad_token_id) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    if (target_id >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    uint logit_offset = token_idx * vocab_size;
    
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    float log_sum_exp = max_logit + log(sum_exp);
    
    float target_log_prob = logits[logit_offset + target_id] - log_sum_exp;
    per_token_loss[token_idx] = -target_log_prob;
    
    // Test without atomic operations first
    // atomic_fetch_add_explicit((device atomic<float>*)total_loss, -target_log_prob, memory_order_relaxed);
}
)";
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(loss_msl) options:nil error:&error];
        if (!library) {
            std::cerr << "❌ Cross-entropy loss kernel failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        std::cout << "✓ Cross-entropy loss kernel (without atomic) passed" << std::endl;
    }
    
    // Test 3: Cross-entropy loss with atomic operations
    {
        const char* loss_atomic_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss_atomic(
    device const float* logits [[buffer(0)]],
    device const uint32_t* target_ids [[buffer(1)]],
    device float* per_token_loss [[buffer(2)]],
    device atomic<float>* total_loss [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    constant uint& pad_token_id [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint seq_idx = gid.y;
    
    if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
    
    uint token_idx = batch_idx * sequence_length + seq_idx;
    uint32_t target_id = target_ids[token_idx];
    
    if (target_id == pad_token_id) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    if (target_id >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    uint logit_offset = token_idx * vocab_size;
    
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    float log_sum_exp = max_logit + log(sum_exp);
    
    float target_log_prob = logits[logit_offset + target_id] - log_sum_exp;
    per_token_loss[token_idx] = -target_log_prob;
    
    atomic_fetch_add_explicit(total_loss, -target_log_prob, memory_order_relaxed);
}
)";
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(loss_atomic_msl) options:nil error:&error];
        if (!library) {
            std::cerr << "❌ Cross-entropy loss with atomic failed: " << [[error localizedDescription] UTF8String] << std::endl;
            std::cerr << "Error details: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        std::cout << "✓ Cross-entropy loss with atomic passed" << std::endl;
    }
    
    // Test 4: AdamW optimizer kernel
    {
        const char* adamw_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void adamw_optimizer(
    device half* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m_state [[buffer(2)]],
    device float* v_state [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant uint& timestep [[buffer(9)]],
    constant uint& param_size [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= param_size) return;
    
    float g = grad[gid];
    float p = float(param[gid]);
    
    p = p - learning_rate * weight_decay * p;
    
    float m = m_state[gid];
    float v = v_state[gid];
    
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    
    m_state[gid] = m;
    v_state[gid] = v;
    
    float m_hat = m / (1.0f - pow(beta1, float(timestep)));
    float v_hat = v / (1.0f - pow(beta2, float(timestep)));
    
    p = p - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    
    param[gid] = half(p);
}
)";
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(adamw_msl) options:nil error:&error];
        if (!library) {
            std::cerr << "❌ AdamW optimizer kernel failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        std::cout << "✓ AdamW optimizer kernel passed" << std::endl;
    }
    
    std::cout << "\n🎉 All individual kernel tests passed!" << std::endl;
    std::cout << "The issue might be in how kernels are combined or other aspects." << std::endl;
    
    return 0;
} 