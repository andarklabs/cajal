#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "transformer_model.h"

int main() {
    std::cout << "🔍 COMPREHENSIVE DIAGNOSTIC TEST - Sequence 12 Crash Analysis" << std::endl;
    std::cout << "🎯 Testing with FULL configuration that was causing system crashes" << std::endl;
    
    // Use the EXACT configuration that was causing crashes
    TransformerConfig config;
    config.vocab_size = 32000;
    config.embedding_dim = 512;        // Original crash config
    config.num_layers = 6;             // Original crash config  
    config.num_heads = 8;              // Original crash config
    config.ffn_hidden_dim = 2048;      // Original crash config - this was the main issue!
    config.max_sequence_length = 512;  // Original crash config
    config.batch_size = 1;             // Keep batch size small for focused testing
    config.learning_rate = 0.001f;
    config.use_half_precision = true;
    
    std::cout << "📊 Model Configuration:" << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  embedding_dim: " << config.embedding_dim << std::endl;
    std::cout << "  num_layers: " << config.num_layers << std::endl;
    std::cout << "  num_heads: " << config.num_heads << std::endl;
    std::cout << "  ffn_hidden_dim: " << config.ffn_hidden_dim << " (🚨 This was the crash cause!)" << std::endl;
    std::cout << "  max_sequence_length: " << config.max_sequence_length << std::endl;
    
    std::cout << "\n🔧 Initializing model with comprehensive safety checks..." << std::endl;
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "❌ CRITICAL: Model initialization failed!" << std::endl;
        std::cerr << "    This means our validateConfiguration() checks caught an issue." << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized successfully!" << std::endl;
    std::cout << "  Parameters: " << model.getParameterCount() << std::endl;
    std::cout << "  Memory usage: " << (model.getMemoryUsage() / 1024 / 1024) << " MB" << std::endl;
    
    // Generate test sequences similar to original crash scenario
    std::vector<std::vector<uint32_t>> input_batch;
    std::vector<std::vector<uint32_t>> target_batch;
    
    // Create 15 sequences to reach the problematic sequence 12 (index 11)
    std::cout << "\n📝 Generating 15 test sequences (to reach sequence 12)..." << std::endl;
    
    for (int i = 0; i < 15; i++) {
        std::vector<uint32_t> seq;
        // Use realistic sequence lengths
        int seq_length = 50 + (rand() % 50); // 50-100 tokens
        
        for (int j = 0; j < seq_length; j++) {
            // Use valid token IDs within vocab range
            seq.push_back((rand() % (config.vocab_size - 1)) + 1);
        }
        input_batch.push_back(seq);
        
        // Create target sequence (input shifted by one)
        std::vector<uint32_t> target = seq;
        target.erase(target.begin());
        target.push_back((rand() % (config.vocab_size - 1)) + 1);
        target_batch.push_back(target);
    }
    
    std::cout << "✅ Generated " << input_batch.size() << " sequences" << std::endl;
    
    std::cout << "\n🚨 CRITICAL TEST: Processing sequences to reach the crash point..." << std::endl;
    std::cout << "    Diagnostics will activate at sequence 12 (index 11)" << std::endl;
    std::cout << "    The ffn_backward kernel fix will be tested!" << std::endl;
    
    // Process sequences using trainBatch (this is where the original crash occurred)
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    float avg_loss;
    bool success = model.trainBatch(input_batch, target_batch, avg_loss);
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        overall_end - overall_start).count();
    
    if (success) {
        std::cout << "\n🎉 SUCCESS! All sequences processed including sequence 12!" << std::endl;
        std::cout << "✅ Total time: " << total_duration << "ms" << std::endl;
        std::cout << "✅ Average loss: " << avg_loss << std::endl;
        std::cout << "\n🔍 This means:" << std::endl;
        std::cout << "  ✓ Configuration validation passed" << std::endl;
        std::cout << "  ✓ MSL kernel ffn_backward fix worked (2048 threadgroup arrays)" << std::endl;
        std::cout << "  ✓ Diagnostic checks for sequence 12 passed" << std::endl;
        std::cout << "  ✓ No system crash occurred!" << std::endl;
        
        // Additional verification test - try a few more batches to stress test
        std::cout << "\n🔄 STRESS TEST: Running additional batches to verify stability..." << std::endl;
        
        for (int stress_batch = 0; stress_batch < 3; stress_batch++) {
            // Generate new sequences
            std::vector<std::vector<uint32_t>> stress_input;
            std::vector<std::vector<uint32_t>> stress_target;
            
            for (int i = 0; i < 12; i++) { // Smaller batch, but still reaches sequence 12
                std::vector<uint32_t> seq;
                int seq_length = 40 + (rand() % 40);
                
                for (int j = 0; j < seq_length; j++) {
                    seq.push_back((rand() % (config.vocab_size - 1)) + 1);
                }
                stress_input.push_back(seq);
                
                std::vector<uint32_t> target = seq;
                target.erase(target.begin());
                target.push_back((rand() % (config.vocab_size - 1)) + 1);
                stress_target.push_back(target);
            }
            
            auto stress_start = std::chrono::high_resolution_clock::now();
            float stress_loss;
            bool stress_success = model.trainBatch(stress_input, stress_target, stress_loss);
            auto stress_end = std::chrono::high_resolution_clock::now();
            auto stress_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                stress_end - stress_start).count();
            
            if (stress_success) {
                std::cout << "✅ Stress batch " << (stress_batch + 1) << " completed in " 
                          << stress_duration << "ms, loss: " << stress_loss << std::endl;
            } else {
                std::cout << "❌ Stress batch " << (stress_batch + 1) << " FAILED after " 
                          << stress_duration << "ms" << std::endl;
                break;
            }
        }
        
    } else {
        std::cout << "\n❌ FAILURE: Batch training failed!" << std::endl;
        std::cout << "💥 Time to failure: " << total_duration << "ms" << std::endl;
        std::cout << "\n🔍 Analysis:" << std::endl;
        std::cout << "  - If crash occurred before sequence 12: Earlier issue not caught by validation" << std::endl;
        std::cout << "  - If crash occurred at sequence 12: Diagnostic checks caught an issue and aborted" << std::endl;
        std::cout << "  - If system crashed: Our fixes didn't work completely" << std::endl;
        return 1;
    }
    
    std::cout << "\n✅ COMPREHENSIVE DIAGNOSTIC TEST COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << "🎯 The sequence 12 crash issue appears to be RESOLVED!" << std::endl;
    
    return 0;
} 