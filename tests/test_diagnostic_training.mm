#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <sstream>
#include "transformer_model.h"

class DetailedTimer {
public:
    void start(const std::string& operation) {
        start_times[operation] = std::chrono::high_resolution_clock::now();
        std::cout << "🕐 Starting: " << operation << std::endl;
    }
    
    void end(const std::string& operation) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_times[operation]).count();
        std::cout << "✅ Completed: " << operation << " (" << duration << "ms)" << std::endl;
        
        if (duration > 5000) {  // Warn if > 5 seconds
            std::cout << "⚠️  SLOW OPERATION: " << operation << " took " << duration << "ms" << std::endl;
        }
    }
    
private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
};

void checkMemoryUsage(const std::string& checkpoint) {
    std::cout << "🧠 Memory checkpoint: " << checkpoint << std::endl;
    // Simple memory check - could be enhanced with more detailed monitoring
}

bool diagnosticTrainBatch(TransformerModel& model, 
                         const std::vector<std::vector<uint32_t>>& input_batch,
                         const std::vector<std::vector<uint32_t>>& target_batch) {
    DetailedTimer timer;
    
    timer.start("Zero Gradients");
    if (!model.zeroGradients()) {
        std::cout << "❌ Failed to zero gradients" << std::endl;
        return false;
    }
    timer.end("Zero Gradients");
    
    float total_loss = 0.0f;
    uint32_t valid_sequences = 0;
    
    for (size_t i = 0; i < input_batch.size(); i++) {
        std::cout << "\n🔄 Processing sequence " << (i+1) << "/" << input_batch.size() << std::endl;
        checkMemoryUsage("Sequence " + std::to_string(i+1) + " start");
        
        const auto& input_seq = input_batch[i];
        const auto& target_seq = target_batch[i];
        
        if (input_seq.size() != target_seq.size() || input_seq.empty()) {
            std::cout << "⚠️  Skipping invalid sequence " << i << std::endl;
            continue;
        }
        
        // Forward pass with timing
        timer.start("Forward Pass " + std::to_string(i+1));
        float sequence_loss;
        if (!model.computeLoss(input_seq, target_seq, sequence_loss)) {
            std::cout << "⚠️  Failed forward pass for sequence " << i << std::endl;
            timer.end("Forward Pass " + std::to_string(i+1));
            continue;
        }
        timer.end("Forward Pass " + std::to_string(i+1));
        
        total_loss += sequence_loss;
        valid_sequences++;
        
        // Loss gradient computation with timing
        timer.start("Loss Gradient " + std::to_string(i+1));
        // Copy target tokens to buffer for gradient computation
        uint32_t* target_data = static_cast<uint32_t*>([model.buffers.target_tokens contents]);
        std::copy(target_seq.begin(), target_seq.end(), target_data);
        
        uint32_t actual_batch_size = 1;
        uint32_t actual_sequence_length = static_cast<uint32_t>(input_seq.size());
        
        id<MTLCommandBuffer> commandBuffer = [model.commandQueue commandBuffer];
        commandBuffer.label = [NSString stringWithFormat:@"LossGrad_Seq_%zu", i];
        
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:model.kernels.loss_gradient];
            [encoder setBuffer:model.buffers.final_logits offset:0 atIndex:0];
            [encoder setBuffer:model.buffers.target_tokens offset:0 atIndex:1];
            [encoder setBuffer:model.buffers.logits_grad offset:0 atIndex:2];
            [encoder setBytes:&actual_batch_size length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&actual_sequence_length length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&model.config.vocab_size length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&model.config.pad_token_id length:sizeof(uint32_t) atIndex:6];
            
            MTLSize threadsPerGrid = MTLSizeMake(actual_batch_size, actual_sequence_length, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
        
        // Add completion handler for debugging
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            std::cout << "📱 Loss gradient command buffer completed for sequence " << (i+1) << std::endl;
        }];
        [commandBuffer commit];
        timer.end("Loss Gradient " + std::to_string(i+1));
        
        // Backward pass with timing
        timer.start("Backward Pass " + std::to_string(i+1));
        if (!model.backwardPass()) {
            std::cout << "⚠️  Failed backward pass for sequence " << i << std::endl;
            timer.end("Backward Pass " + std::to_string(i+1));
            continue;
        }
        timer.end("Backward Pass " + std::to_string(i+1));
        
        checkMemoryUsage("Sequence " + std::to_string(i+1) + " end");
        
        std::cout << "✅ Sequence " << (i+1) << " completed, loss: " << sequence_loss << std::endl;
        
        // Force a brief pause to see if that helps with command buffer queue
        if (i > 0 && i % 8 == 0) {
            std::cout << "⏸️  Brief pause after 8 sequences..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    if (valid_sequences == 0) {
        std::cout << "❌ No valid sequences processed" << std::endl;
        return false;
    }
    
    float avg_loss = total_loss / valid_sequences;
    
    // Gradient scaling
    timer.start("Gradient Scaling");
    float scale_factor = 1.0f / valid_sequences;
    // ... scaling logic would go here
    timer.end("Gradient Scaling");
    
    // Optimizer step
    timer.start("Optimizer Step");
    if (!model.optimizerStep()) {
        std::cout << "❌ Failed optimizer step" << std::endl;
        return false;
    }
    timer.end("Optimizer Step");
    
    std::cout << "\n✅ Diagnostic batch completed. Valid sequences: " << valid_sequences 
              << ", avg loss: " << avg_loss << std::endl;
    
    return true;
}

int main() {
    std::cout << "🔍 Diagnostic Training Test - Analyzing Performance Bottlenecks" << std::endl;
    
    // Smaller model for debugging
    TransformerConfig config;
    config.embedding_dim = 512;
    config.num_layers = 4;
    config.num_heads = 8;
    config.ffn_hidden_dim = 2048;
    config.max_sequence_length = 256;
    config.batch_size = 16;  // Smaller batch size
    
    std::cout << "🔧 Initializing smaller model for diagnosis..." << std::endl;
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "📊 Model initialized. Parameters: " << model.getParameterCount() << std::endl;
    
    // Generate minimal test data
    std::vector<std::vector<uint32_t>> input_batch;
    std::vector<std::vector<uint32_t>> target_batch;
    
    std::cout << "📊 Generating test data..." << std::endl;
    for (int i = 0; i < 20; i++) {  // 20 sequences to see where it fails
        std::vector<uint32_t> seq;
        for (int j = 0; j < 64; j++) {  // Short sequences
            seq.push_back(rand() % 1000 + 1);  // Random tokens 1-1000
        }
        input_batch.push_back(seq);
        
        // Target is input shifted by 1
        std::vector<uint32_t> target = seq;
        target.erase(target.begin());
        target.push_back(rand() % 1000 + 1);
        target_batch.push_back(target);
    }
    
    std::cout << "🔥 Running diagnostic training..." << std::endl;
    
    if (!diagnosticTrainBatch(model, input_batch, target_batch)) {
        std::cout << "❌ Diagnostic training failed" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Diagnostic training completed successfully!" << std::endl;
    return 0;
} 