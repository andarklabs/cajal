// PERFORMANCE PATCH: Eliminates blocking synchronization causing 5-minute delays
// This is a targeted fix for the CPU-GPU synchronization bottleneck
// Apply these changes to your main transformer_model.mm file

// PATCH 1: Replace blocking backward pass with async version
bool TransformerModel::backwardPass() {
    uint32_t actual_batch_size = config.batch_size;
    uint32_t actual_sequence_length = config.max_sequence_length;
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"OptimizedBackwardPass";
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Process all backward operations in a single command buffer (batching optimization)
    
    // Layer backward passes (in reverse order)
    for (int layer = config.num_layers - 1; layer >= 0; layer--) {
        
        // FFN backward
        [encoder setComputePipelineState:kernels.ffn_backward];
        [encoder setBuffer:buffers.ffn_output_grad[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.ffn_h_activated[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.ffn_h_linear[layer] offset:0 atIndex:2];
        [encoder setBuffer:weights.blocks[layer].ffn_w2 offset:0 atIndex:3];
        [encoder setBuffer:weights.blocks[layer].ffn_w1 offset:0 atIndex:4];
        [encoder setBuffer:gradients.blocks_grad[layer].ffn_w2 offset:0 atIndex:5];
        [encoder setBuffer:gradients.blocks_grad[layer].ffn_b2 offset:0 atIndex:6];
        [encoder setBuffer:gradients.blocks_grad[layer].ffn_w1 offset:0 atIndex:7];
        [encoder setBuffer:gradients.blocks_grad[layer].ffn_b1 offset:0 atIndex:8];
        [encoder setBuffer:buffers.attention_normed_grad[layer] offset:0 atIndex:9];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:11];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:12];
        [encoder setBytes:&config.ffn_hidden_dim length:sizeof(uint32_t) atIndex:13];
        
        uint32_t ffn_total_elements = actual_batch_size * actual_sequence_length * config.embedding_dim;
        MTLSize ffnThreadsPerGrid = MTLSizeMake(ffn_total_elements, 1, 1);
        MTLSize ffnThreadsPerThreadgroup = MTLSizeMake(128, 1, 1); // Optimized for M3 Max
        [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
        
        // Layer norm backward (2nd one - after FFN)
        [encoder setComputePipelineState:kernels.layer_norm_backward];
        [encoder setBuffer:buffers.attention_normed_grad[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_normed[layer] offset:0 atIndex:1];
        [encoder setBuffer:weights.blocks[layer].ln2_gamma offset:0 atIndex:2];
        [encoder setBuffer:buffers.ln_mean[layer] offset:0 atIndex:3];
        [encoder setBuffer:buffers.ln_rsqrt_variance[layer] offset:0 atIndex:4];
        [encoder setBuffer:gradients.blocks_grad[layer].ln2_gamma offset:0 atIndex:5];
        [encoder setBuffer:gradients.blocks_grad[layer].ln2_beta offset:0 atIndex:6];
        [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:7];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&config.epsilon length:sizeof(float) atIndex:11];
        [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
        
        // MHSA output projection backward
        [encoder setComputePipelineState:kernels.mhsa_output_projection_backward];
        [encoder setBuffer:buffers.attention_output_grad[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.mhsa_projection_outputs_saved[layer] offset:0 atIndex:1];
        [encoder setBuffer:weights.blocks[layer].attention_output_weights offset:0 atIndex:2];
        [encoder setBuffer:gradients.blocks_grad[layer].attention_output_weights offset:0 atIndex:3];
        [encoder setBuffer:gradients.blocks_grad[layer].attention_output_bias offset:0 atIndex:4];
        [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:5];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
        [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
        
        // Scaled dot-product attention backward
        [encoder setComputePipelineState:kernels.scaled_dot_product_attention_backward];
        [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.attention_Q[layer] offset:0 atIndex:1];
        [encoder setBuffer:buffers.attention_K[layer] offset:0 atIndex:2];
        [encoder setBuffer:buffers.attention_V[layer] offset:0 atIndex:3];
        [encoder setBuffer:buffers.attention_weights[layer] offset:0 atIndex:4];
        [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:5];
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.num_heads length:sizeof(uint32_t) atIndex:8];
        uint32_t head_dim = config.embedding_dim / config.num_heads;
        [encoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:9];
        [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
        
        // QKV projection backward
        [encoder setComputePipelineState:kernels.qkv_projection_backward];
        [encoder setBuffer:buffers.attention_qkv_grad[layer] offset:0 atIndex:0];
        [encoder setBuffer:buffers.layer_inputs[layer] offset:0 atIndex:1];
        [encoder setBuffer:weights.blocks[layer].qkv_weights offset:0 atIndex:2];
        [encoder setBuffer:gradients.blocks_grad[layer].qkv_weights offset:0 atIndex:3];
        [encoder setBuffer:gradients.blocks_grad[layer].qkv_bias offset:0 atIndex:4];
        
        id<MTLBuffer> output_grad_buffer = (layer > 0) ? buffers.layer_inputs_grad[layer-1] : buffers.final_hidden_grad;
        [encoder setBuffer:output_grad_buffer offset:0 atIndex:5];
        
        [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:8];
        [encoder dispatchThreads:ffnThreadsPerGrid threadsPerThreadgroup:ffnThreadsPerThreadgroup];
    }
    
    // Final embedding layer backward (the problematic one!)
    [encoder setComputePipelineState:kernels.embedding_layer_backward];
    [encoder setBuffer:buffers.final_hidden_grad offset:0 atIndex:0];
    [encoder setBuffer:buffers.input_tokens offset:0 atIndex:1];
    [encoder setBuffer:gradients.token_embeddings_grad offset:0 atIndex:2];
    [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&config.embedding_dim length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&config.vocab_size length:sizeof(uint32_t) atIndex:6];
    
    MTLSize embeddingThreadsPerGrid = MTLSizeMake(actual_batch_size * actual_sequence_length, 1, 1);
    MTLSize embeddingThreadsPerThreadgroup = MTLSizeMake(std::min(actual_batch_size * actual_sequence_length, 64u), 1, 1);
    [encoder dispatchThreads:embeddingThreadsPerGrid threadsPerThreadgroup:embeddingThreadsPerThreadgroup];
    
    [encoder endEncoding];
    
    // 🚀 THE FIX: Replace blocking wait with async completion
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // This runs asynchronously when GPU completes - no CPU blocking!
        std::cout << "✓ Embedding layer backward pass completed asynchronously" << std::endl;
    }];
    
    [commandBuffer commit];
    // ✅ NO BLOCKING WAIT HERE - Return immediately and let GPU work async
    
    std::cout << "🚀 Backward pass submitted asynchronously, CPU continues..." << std::endl;
    return true;
}

// PATCH 2: Modify trainStep to handle async backward pass
bool TransformerModel::trainStep(const std::vector<uint32_t>& input_tokens,
                                const std::vector<uint32_t>& target_tokens,
                                float& loss) {
    // Forward pass
    std::vector<float> logits;
    if (!forward(input_tokens, logits)) {
        return false;
    }
    
    // Loss computation
    if (!computeLoss(input_tokens, target_tokens, loss)) {
        return false;
    }
    
    // Async backward pass (the fixed version)
    if (!backwardPass()) {
        return false;
    }
    
    // 🔄 STRATEGIC SYNC: Only wait before optimizer step (when we need gradients)
    // This is much more efficient than waiting after every operation
    std::cout << "⏳ Syncing before optimizer step (strategic sync point)..." << std::endl;
    [commandQueue waitUntilSheduledCommandsCompleted]; // Wait for backward pass to complete
    
    // Optimizer step (needs completed gradients)
    if (!optimizerStep()) {
        return false;
    }
    
    return true;
}

// PATCH 3: Remove other unnecessary blocking waits
// Apply these changes throughout your implementation:

// ❌ REPLACE THIS PATTERN:
// [encoder endEncoding];
// [commandBuffer commit];
// [commandBuffer waitUntilCompleted];  // <-- REMOVE THIS LINE

// ✅ WITH THIS PATTERN:
// [encoder endEncoding];
// [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
//     // Optional: async completion callback
// }];
// [commandBuffer commit];  // No blocking wait - return immediately

// PATCH 4: Comments showing where to apply fixes in your current code
/*
LOCATIONS TO PATCH in transformer_model.mm:

Line ~1618: Remove waitUntilCompleted after forward pass operations
Line ~1676: Remove waitUntilCompleted after loss computation  
Line ~1736: Remove waitUntilCompleted after backward operations
Line ~1807: Remove waitUntilCompleted after layer processing
Line ~1880: Remove waitUntilCompleted after attention processing
Line ~2218: Remove waitUntilCompleted after embedding backward (THE MAIN ONE!)
Line ~2551: Keep sync before optimizer (or make strategic)
Line ~2745: Remove waitUntilCompleted after final operations

STRATEGY:
1. Remove ALL waitUntilCompleted calls except before optimizer step
2. Add async completion handlers where needed for logging/debugging
3. Only sync when you absolutely need results (optimizer needs gradients)
*/ 