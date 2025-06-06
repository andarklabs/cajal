//
// Minimal MSL Transformer Chatbot Test
// For debugging segfault issues safely
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include "src/host/transformer_model.h"

int main() {
    std::cout << "🧪 Minimal MSL Transformer Chatbot Test" << std::endl;
    std::cout << "═══════════════════════════════════════=" << std::endl;
    
    try {
        // Create minimal configuration for safety
        TransformerConfig test_config;
        test_config.vocab_size = 1776;
        test_config.embedding_dim = 512;
        test_config.num_heads = 8;
        test_config.num_layers = 6;
        test_config.ffn_hidden_dim = 2048;
        test_config.max_sequence_length = 512;
        test_config.batch_size = 1;  // Minimal batch size
        test_config.learning_rate = 0.001f;
        
        std::cout << "📋 Test Configuration:" << std::endl;
        std::cout << "   vocab_size: " << test_config.vocab_size << std::endl;
        std::cout << "   embedding_dim: " << test_config.embedding_dim << std::endl;
        std::cout << "   num_layers: " << test_config.num_layers << std::endl;
        std::cout << "   batch_size: " << test_config.batch_size << std::endl;
        std::cout << "   max_sequence_length: " << test_config.max_sequence_length << std::endl;
        std::cout << std::endl;
        
        std::cout << "🔧 Step 1: Creating TransformerModel..." << std::endl;
        TransformerModel model(test_config);
        
        std::cout << "🔧 Step 1.5: Initializing model..." << std::endl;
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model!" << std::endl;
            return 1;
        }
        std::cout << "✅ Model initialized successfully!" << std::endl;
        
        std::cout << "📊 Model Info:" << std::endl;
        std::cout << "   Parameters: " << model.getParameterCount() << std::endl;
        std::cout << "   Memory: " << model.getMemoryUsage() << " MB" << std::endl;
        
        // Check if memory is reasonable (should be < 200 MB)
        if (model.getMemoryUsage() > 200) {
            std::cerr << "❌ Memory usage too high: " << model.getMemoryUsage() << " MB" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Memory usage looks reasonable!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "🧪 Step 2: Testing generateNext method..." << std::endl;
        
        // Create a simple test context
        std::vector<uint32_t> test_context = {100, 200, 300, 400};
        
        std::cout << "   Input context: [";
        for (size_t i = 0; i < test_context.size(); i++) {
            std::cout << test_context[i];
            if (i < test_context.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Test the generateNext method
        std::cout << "   Calling generateNext..." << std::endl;
        std::vector<float> logits = model.generateNext(test_context);
        
        if (logits.empty()) {
            std::cerr << "❌ generateNext returned empty logits" << std::endl;
            return 1;
        }
        
        std::cout << "✅ generateNext succeeded!" << std::endl;
        std::cout << "   Logits size: " << logits.size() << std::endl;
        std::cout << "   Expected size: " << test_config.vocab_size << std::endl;
        
        if (logits.size() != test_config.vocab_size) {
            std::cerr << "❌ Logits size mismatch!" << std::endl;
            return 1;
        }
        
        // Check first few logits for sanity
        std::cout << "   First 5 logits: [";
        for (int i = 0; i < 5 && i < logits.size(); i++) {
            std::cout << logits[i];
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << std::endl;
        std::cout << "🎉 SUCCESS: All tests passed!" << std::endl;
        std::cout << "   The chatbot should work safely now." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "💥 Unknown exception caught" << std::endl;
        return 1;
    }
    
    return 0;
} 