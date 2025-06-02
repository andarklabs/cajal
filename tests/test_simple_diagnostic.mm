#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "transformer_model.h"

int main() {
    std::cout << "🔍 Simple Diagnostic - Analyzing 17th Sequence Issue" << std::endl;
    
    // Small model configuration for quick testing
    TransformerConfig config;
    config.embedding_dim = 256;
    config.num_layers = 2;
    config.num_heads = 4;
    config.ffn_hidden_dim = 1024;
    config.max_sequence_length = 128;
    
    std::cout << "🔧 Initializing minimal model..." << std::endl;
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized. Parameters: " << model.getParameterCount() << std::endl;
    
    // Generate test sequences
    std::vector<std::vector<uint32_t>> input_batch;
    std::vector<std::vector<uint32_t>> target_batch;
    
    for (int i = 0; i < 25; i++) {  // Test 25 sequences to exceed the problematic 17th
        std::vector<uint32_t> seq;
        for (int j = 0; j < 32; j++) {  // Short sequences
            seq.push_back((rand() % 1000) + 1);
        }
        input_batch.push_back(seq);
        
        std::vector<uint32_t> target = seq;
        target.erase(target.begin());
        target.push_back((rand() % 1000) + 1);
        target_batch.push_back(target);
    }
    
    std::cout << "📊 Testing individual sequence processing..." << std::endl;
    
    // Process sequences one by one to isolate the issue
    for (size_t i = 0; i < input_batch.size(); i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n🔄 Processing sequence " << (i+1) << "/" << input_batch.size() << std::endl;
        
        float loss;
        bool success = model.trainStep(input_batch[i], target_batch[i], loss);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        if (success) {
            std::cout << "✅ Sequence " << (i+1) << " completed in " << duration 
                      << "ms, loss: " << loss << std::endl;
        } else {
            std::cout << "❌ Sequence " << (i+1) << " FAILED after " << duration << "ms" << std::endl;
            break;
        }
        
        // Check for concerning patterns
        if (duration > 5000) {
            std::cout << "⚠️  SLOW: Sequence " << (i+1) << " took " << duration << "ms" << std::endl;
        }
        
        if (i == 15) {
            std::cout << "🎯 Reached sequence 16 - next sequence is the problematic 17th..." << std::endl;
        }
        
        if (i == 16) {
            std::cout << "🚨 This is sequence 17 - the problematic one!" << std::endl;
        }
        
        // Add a small delay every few sequences to see if that helps
        if (i > 0 && i % 5 == 0) {
            std::cout << "⏸️  Brief pause..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    std::cout << "\n🔍 Now testing batch processing..." << std::endl;
    
    // Test smaller batches to see when the issue occurs
    std::vector<int> batch_sizes = {4, 8, 12, 16, 20};
    
    for (int batch_size : batch_sizes) {
        std::cout << "\n📦 Testing batch size: " << batch_size << std::endl;
        
        std::vector<std::vector<uint32_t>> test_input(input_batch.begin(), 
                                                      input_batch.begin() + batch_size);
        std::vector<std::vector<uint32_t>> test_target(target_batch.begin(), 
                                                       target_batch.begin() + batch_size);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        float avg_loss;
        bool success = model.trainBatch(test_input, test_target, avg_loss);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        if (success) {
            std::cout << "✅ Batch size " << batch_size << " completed in " << duration 
                      << "ms, avg loss: " << avg_loss << std::endl;
        } else {
            std::cout << "❌ Batch size " << batch_size << " FAILED after " << duration << "ms" << std::endl;
            std::cout << "🎯 FOUND THE ISSUE: Batch training fails at size " << batch_size << std::endl;
            break;
        }
        
        if (duration > 10000) {
            std::cout << "⚠️  VERY SLOW: Batch size " << batch_size << " took " << duration << "ms" << std::endl;
        }
    }
    
    std::cout << "\n✅ Diagnostic completed!" << std::endl;
    return 0;
} 