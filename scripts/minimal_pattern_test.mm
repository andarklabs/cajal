//
// Minimal Pattern Test - Find what's broken!
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🔍 MINIMAL Pattern Test - Find the Problem!" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        // TINY model for debugging
        TransformerConfig config;
        config.vocab_size = 1000;      // Much smaller vocab
        config.embedding_dim = 128;    // Much smaller model
        config.num_heads = 4;
        config.num_layers = 2;         // Fewer layers
        config.ffn_hidden_dim = 256;   // Smaller FFN
        config.max_sequence_length = 16; // Short sequences
        config.batch_size = 1;         // Single sample
        config.learning_rate = 0.1f;   // Higher LR for simple patterns
        
        std::cout << "🔧 TINY model (1K vocab, ~200K params)" << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // SUPER SIMPLE patterns
        std::vector<uint32_t> input = {5};     // Token 5
        std::vector<uint32_t> target = {6};   // Should predict token 6
        
        std::cout << "\n🎯 SIMPLE Pattern: [5] -> [6]" << std::endl;
        std::cout << "Rule: 5 always predicts 6\n" << std::endl;
        
        // Test prediction BEFORE training
        std::cout << "🧪 BEFORE Training:" << std::endl;
        std::vector<float> logits_before;
        if (model.forward(input, logits_before)) {
            uint32_t predicted = 0;
            float max_prob = logits_before[0];
            for (uint32_t i = 1; i < config.vocab_size; i++) {
                if (logits_before[i] > max_prob) {
                    max_prob = logits_before[i];
                    predicted = i;
                }
            }
            std::cout << "  Input [5] -> Predicted: " << predicted << " (Expected: 6)";
            std::cout << " | Prob of 6: " << std::fixed << std::setprecision(4) << logits_before[6] << std::endl;
        }
        
        // RAPID training - just 3 steps
        std::cout << "\n🚀 Quick Training (3 steps):" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < 3; step++) {
            auto step_start = std::chrono::high_resolution_clock::now();
            
            float loss;
            bool result = model.trainBatch({input}, {target}, loss);
            
            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
            
            std::cout << "Step " << (step + 1) << ": ";
            if (result) {
                std::cout << "✅ (" << step_duration.count() << "ms)";
                if (loss > 0) {
                    std::cout << " Loss: " << std::fixed << std::setprecision(4) << loss;
                }
            } else {
                std::cout << "❌ FAILED";
            }
            std::cout << std::endl;
            
            // If taking too long, abort
            if (step_duration.count() > 5000) {
                std::cout << "⚠️  Step took > 5 seconds - ABORTING TEST" << std::endl;
                break;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Total training time: " << duration.count() << "ms" << std::endl;
        
        // Test prediction AFTER training
        std::cout << "\n🧪 AFTER Training:" << std::endl;
        std::vector<float> logits_after;
        if (model.forward(input, logits_after)) {
            uint32_t predicted = 0;
            float max_prob = logits_after[0];
            for (uint32_t i = 1; i < config.vocab_size; i++) {
                if (logits_after[i] > max_prob) {
                    max_prob = logits_after[i];
                    predicted = i;
                }
            }
            std::cout << "  Input [5] -> Predicted: " << predicted << " (Expected: 6)";
            std::cout << " | Prob of 6: " << std::fixed << std::setprecision(4) << logits_after[6] << std::endl;
            
            // Compare probabilities
            float improvement = logits_after[6] - logits_before[6];
            std::cout << "  Improvement in P(6): " << std::showpos << improvement << std::noshowpos << std::endl;
        }
        
        // DIAGNOSIS
        std::cout << "\n📊 DIAGNOSIS:" << std::endl;
        std::cout << "==================" << std::endl;
        
        if (duration.count() > 10000) {
            std::cout << "🚨 PROBLEM: Training too slow (" << duration.count() << "ms for 3 steps)" << std::endl;
            std::cout << "   Expected: <100ms total" << std::endl;
        } else if (duration.count() > 1000) {
            std::cout << "⚠️  CONCERN: Training slower than expected (" << duration.count() << "ms)" << std::endl;
        } else {
            std::cout << "✅ SPEED: Training time acceptable (" << duration.count() << "ms)" << std::endl;
        }
        
        // Check if model learned anything
        if (logits_after[6] > logits_before[6]) {
            std::cout << "✅ LEARNING: Model improved prediction of target token" << std::endl;
        } else {
            std::cout << "❌ LEARNING: No improvement in target prediction" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 