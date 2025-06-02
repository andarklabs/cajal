#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "transformer_model.h"

int main() {
    std::cout << "🔍 Testing Batch Size Limits - Finding the 17th Sequence Issue" << std::endl;
    
    // Small model for quick testing
    TransformerConfig config;
    config.embedding_dim = 128;
    config.num_layers = 2;
    config.num_heads = 4;
    config.ffn_hidden_dim = 256;
    config.max_sequence_length = 64;
    config.vocab_size = 1000;
    
    std::cout << "🔧 Initializing model..." << std::endl;
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized. Parameters: " << model.getParameterCount() << std::endl;
    
    // Generate test sequences
    std::cout << "📊 Generating test sequences..." << std::endl;
    std::vector<std::vector<uint32_t>> input_batch;
    std::vector<std::vector<uint32_t>> target_batch;
    
    for (int i = 0; i < 25; i++) {
        std::vector<uint32_t> seq;
        for (int j = 0; j < 32; j++) {  // Short sequences
            seq.push_back((rand() % 900) + 100);  // Tokens 100-999
        }
        input_batch.push_back(seq);
        
        std::vector<uint32_t> target = seq;
        target.erase(target.begin());
        target.push_back((rand() % 900) + 100);
        target_batch.push_back(target);
    }
    
    // Test different batch sizes to find the breaking point
    std::vector<int> test_sizes = {1, 2, 4, 8, 12, 16, 20, 24};
    
    for (int batch_size : test_sizes) {
        std::cout << "\n📦 Testing batch size: " << batch_size << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<uint32_t>> test_input(input_batch.begin(), 
                                                      input_batch.begin() + batch_size);
        std::vector<std::vector<uint32_t>> test_target(target_batch.begin(), 
                                                       target_batch.begin() + batch_size);
        
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
            std::cout << "🎯 CRITICAL: Found failure point at batch size " << batch_size << std::endl;
            break;
        }
        
        if (duration > 30000) {  // 30 seconds
            std::cout << "⚠️  VERY SLOW: Batch size " << batch_size << " took " << duration 
                      << "ms - likely hanging" << std::endl;
            std::cout << "🛑 Stopping due to excessive time" << std::endl;
            break;
        }
        
        if (batch_size == 16) {
            std::cout << "🎯 Successfully processed 16 sequences - this is where problems start!" << std::endl;
        }
    }
    
    std::cout << "\n🔍 Now testing individual sequences around the problem area..." << std::endl;
    
    // Test individual sequences 15, 16, 17, 18 to isolate the exact problematic sequence
    for (int i = 14; i < 19 && i < input_batch.size(); i++) {
        std::cout << "\n🔄 Testing individual sequence " << (i+1) << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
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
            std::cout << "🎯 FOUND IT: Individual sequence " << (i+1) << " fails!" << std::endl;
            break;
        }
        
        if (duration > 10000) {  // 10 seconds
            std::cout << "⚠️  SLOW: Sequence " << (i+1) << " took " << duration << "ms" << std::endl;
        }
    }
    
    std::cout << "\n✅ Batch size limit test completed!" << std::endl;
    return 0;
} 