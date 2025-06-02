#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>

// Extract the exact MSL source from transformer_model.mm to test
const char* exact_msl_source = R"(
#include <metal_stdlib>
using namespace metal;

// Embedding lookup kernel
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
    if (token_id >= vocab_size) return; // Bounds check
    
    uint output_offset = token_idx * embedding_dim;
    uint embedding_offset = token_id * embedding_dim;
    
    for (uint d = 0; d < embedding_dim; d++) {
        output_embeddings[output_offset + d] = embedding_table[embedding_offset + d];
    }
}

// Test with just a few key kernels first
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
    
    // Skip padded tokens
    if (target_id == pad_token_id) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    // Bounds check for target_id
    if (target_id >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    // Compute log softmax for numerical stability
    uint logit_offset = token_idx * vocab_size;
    
    // Find max logit for stability
    float max_logit = logits[logit_offset];
    for (uint v = 1; v < vocab_size; v++) {
        max_logit = max(max_logit, logits[logit_offset + v]);
    }
    
    // Compute log sum exp
    float sum_exp = 0.0f;
    for (uint v = 0; v < vocab_size; v++) {
        sum_exp += exp(logits[logit_offset + v] - max_logit);
    }
    float log_sum_exp = max_logit + log(sum_exp);
    
    // Negative log likelihood loss
    float target_log_prob = logits[logit_offset + target_id] - log_sum_exp;
    per_token_loss[token_idx] = -target_log_prob;
    
    // Atomic add to total loss (will be averaged by number of non-pad tokens on CPU)
    atomic_fetch_add_explicit(total_loss, -target_log_prob, memory_order_relaxed);
}
)";

int main() {
    std::cout << "Testing exact MSL kernel loading..." << std::endl;
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Metal device created: " << [[device name] UTF8String] << std::endl;
    
    // Test the exact MSL compilation
    NSError* error = nil;
    std::cout << "Creating MSL library..." << std::endl;
    
    id<MTLLibrary> library = [device newLibraryWithSource:@(exact_msl_source) options:nil error:&error];
    if (!library) {
        std::cerr << "❌ Failed to create MSL library!" << std::endl;
        std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }
    
    std::cout << "✓ MSL library created successfully" << std::endl;
    
    // Test individual kernel loading
    const char* kernel_names[] = {
        "embedding_lookup",
        "apply_positional_encoding",
        "cross_entropy_loss"
    };
    
    for (int i = 0; i < 3; i++) {
        std::cout << "Loading kernel: " << kernel_names[i] << std::endl;
        
        id<MTLFunction> function = [library newFunctionWithName:@(kernel_names[i])];
        if (!function) {
            std::cerr << "❌ Failed to find kernel function: " << kernel_names[i] << std::endl;
            return 1;
        }
        
        std::cout << "Creating pipeline for: " << kernel_names[i] << std::endl;
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "❌ Failed to create pipeline for: " << kernel_names[i] << std::endl;
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        
        std::cout << "✓ " << kernel_names[i] << " loaded successfully" << std::endl;
    }
    
    std::cout << "\n🎉 All kernels loaded successfully!" << std::endl;
    std::cout << "The segfault must be elsewhere in the transformer model code." << std::endl;
    
    return 0;
} 