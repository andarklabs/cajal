#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>

bool testKernelSet(id<MTLDevice> device, const char* description, const char* msl_source, const char* kernel_names[], int num_kernels) {
    std::cout << "Testing: " << description << std::endl;
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(msl_source) options:nil error:&error];
    if (!library) {
        std::cerr << "❌ Failed to create library for " << description << std::endl;
        std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Try to load each kernel function
    for (int i = 0; i < num_kernels; i++) {
        id<MTLFunction> function = [library newFunctionWithName:@(kernel_names[i])];
        if (!function) {
            std::cerr << "❌ Failed to find kernel function: " << kernel_names[i] << std::endl;
            return false;
        }
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "❌ Failed to create pipeline for: " << kernel_names[i] << std::endl;
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        std::cout << "  ✓ " << kernel_names[i] << std::endl;
    }
    
    std::cout << "✅ " << description << " - All kernels loaded successfully" << std::endl;
    return true;
}

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return 1;
    }
    
    std::cout << "Progressive MSL kernel loading test..." << std::endl;
    
    // Test 1: Basic forward kernels only
    {
        const char* forward_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void embedding_lookup(
    device const uint32_t* token_ids [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* output_embeddings [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint32_t token_id = token_ids[token_idx];
    if (token_id >= vocab_size) return;
    
    uint output_offset = token_idx * embedding_dim;
    uint embedding_offset = token_id * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        output_embeddings[output_offset + d] = embedding_table[embedding_offset + d];
    }
}

kernel void apply_positional_encoding(
    device half* embeddings [[buffer(0)]],
    device const half* positional_table [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint total_tokens = batch_size * sequence_length;
    
    if (token_idx >= total_tokens) return;
    
    uint seq_pos = token_idx % sequence_length;
    uint output_offset = token_idx * embedding_dim;
    uint pe_offset = seq_pos * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        embeddings[output_offset + d] += positional_table[pe_offset + d];
    }
}

kernel void softmax(
    device const float* logits [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& sequence_length [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint instance_idx = gid.x;
    uint total_instances = batch_size * sequence_length;
    
    if (instance_idx >= total_instances) return;
    
    uint offset = instance_idx * vocab_size;
    
    float max_val = logits[offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_val = max(max_val, logits[offset + v]);
    }
    
    float sum = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        float exp_val = exp(logits[offset + v] - max_val);
        probabilities[offset + v] = exp_val;
        sum += exp_val;
    }
    
    for (uint v = 0; v < vocab_size; v++) {
        probabilities[offset + v] /= sum;
    }
}
)";
        
        const char* forward_kernel_names[] = {
            "embedding_lookup",
            "apply_positional_encoding", 
            "softmax"
        };
        
        if (!testKernelSet(device, "Basic Forward Kernels", forward_msl, forward_kernel_names, 3)) {
            return 1;
        }
    }
    
    // Test 2: Add simple training kernels
    {
        const char* simple_training_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void zero_gradients(
    device float* grad_buffer [[buffer(0)]],
    constant uint& buffer_size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= buffer_size) return;
    grad_buffer[gid] = 0.0f;
}

kernel void loss_gradient(
    device const float* logits [[buffer(0)]],
    device const uint32_t* target_ids [[buffer(1)]],
    device float* logits_grad [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    constant uint& pad_token_id [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint seq_idx = gid.y;
    
    if (batch_idx >= batch_size || seq_idx >= sequence_length) return;
    
    uint token_idx = batch_idx * sequence_length + seq_idx;
    uint32_t target_id = target_ids[token_idx];
    uint logit_offset = token_idx * vocab_size;
    
    if (target_id == pad_token_id || target_id >= vocab_size) {
        for (uint v = 0; v < vocab_size; v++) {
            logits_grad[logit_offset + v] = 0.0f;
        }
        return;
    }
    
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    
    for (uint v = 0; v < vocab_size; v++) {
        float softmax_val = exp(logits[logit_offset + v] - max_logit) / sum_exp;
        float one_hot = (v == target_id) ? 1.0f : 0.0f;
        logits_grad[logit_offset + v] = softmax_val - one_hot;
    }
}
)";
        
        const char* simple_training_names[] = {
            "zero_gradients",
            "loss_gradient"
        };
        
        if (!testKernelSet(device, "Simple Training Kernels", simple_training_msl, simple_training_names, 2)) {
            return 1;
        }
    }
    
    // Test 3: Cross-entropy loss with atomic (suspect)
    {
        const char* loss_atomic_msl = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy_loss(
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
        
        const char* loss_names[] = {
            "cross_entropy_loss"
        };
        
        if (!testKernelSet(device, "Cross-Entropy Loss with Atomic", loss_atomic_msl, loss_names, 1)) {
            return 1;
        }
    }
    
    std::cout << "\n🎉 All progressive kernel tests passed!" << std::endl;
    std::cout << "The issue might be elsewhere in the code structure..." << std::endl;
    
    return 0;
} 