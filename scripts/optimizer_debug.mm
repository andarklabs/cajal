//
// Optimizer Debug - Check learning rate and gradient values
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🔧 OPTIMIZER DEBUG - Check Learning Rate & Gradients" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    try {
        // Tiny model
        TransformerConfig config;
        config.vocab_size = 10;        // Tiny vocab
        config.embedding_dim = 8;      // Tiny embedding 
        config.num_heads = 2;
        config.num_layers = 1;         
        config.ffn_hidden_dim = 16;
        config.max_sequence_length = 4;
        config.batch_size = 1;
        config.learning_rate = 0.1f;   // High LR - should be obvious
        config.use_half_precision = false; // Use float for easier debugging
        
        std::cout << "🔧 NANO model (10 vocab, float precision)" << std::endl;
        std::cout << "   Learning rate: " << config.learning_rate << std::endl;
        std::cout << "   Use half precision: " << (config.use_half_precision ? "true" : "false") << std::endl;
        
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // Simple pattern
        std::vector<uint32_t> input = {1};
        std::vector<uint32_t> target = {2};
        
        std::cout << "\n🎯 Pattern: [1] -> [2]" << std::endl;
        
        // Manual training step with inspection
        std::cout << "\n🔍 MANUAL TRAINING STEP:" << std::endl;
        
        // 1. Forward pass
        std::vector<float> logits_before;
        if (!model.forward(input, logits_before)) {
            std::cout << "❌ Forward pass failed" << std::endl;
            return 1;
        }
        
        std::cout << "📊 BEFORE weights (first 5 logits): ";
        for (int i = 0; i < 5 && i < logits_before.size(); i++) {
            std::cout << std::fixed << std::setprecision(4) << logits_before[i] << " ";
        }
        std::cout << std::endl;
        
        // Skip manual steps - just use trainBatch
        
        // 5. Let's use a single trainBatch call and then inspect
        std::cout << "🎓 Using trainBatch for comparison..." << std::endl;
        
        float batch_loss;
        bool train_result = model.trainBatch({input}, {target}, batch_loss);
        
        std::cout << "📊 Training result: " << (train_result ? "✅ Success" : "❌ Failed") << std::endl;
        std::cout << "📊 Batch loss: " << batch_loss << std::endl;
        
        // 6. Check weights after training
        std::vector<float> logits_after;
        if (!model.forward(input, logits_after)) {
            std::cout << "❌ Forward pass after training failed" << std::endl;
            return 1;
        }
        
        std::cout << "📊 AFTER weights (first 5 logits): ";
        for (int i = 0; i < 5 && i < logits_after.size(); i++) {
            std::cout << std::fixed << std::setprecision(4) << logits_after[i] << " ";
        }
        std::cout << std::endl;
        
        // 7. Check for changes
        bool any_change = false;
        float max_change = 0.0f;
        for (int i = 0; i < std::min(logits_before.size(), logits_after.size()); i++) {
            float change = std::abs(logits_after[i] - logits_before[i]);
            if (change > max_change) max_change = change;
            if (change > 1e-6f) any_change = true;
        }
        
        std::cout << "\n📈 ANALYSIS:" << std::endl;
        std::cout << "   Max change: " << max_change << std::endl;
        std::cout << "   Learning rate used: " << config.learning_rate << std::endl;
        std::cout << "   Expected change: > 0.0001 (for LR=0.1)" << std::endl;
        
        if (any_change) {
            std::cout << "   ✅ WEIGHTS UPDATED!" << std::endl;
            
            // Check target token specifically
            if (logits_after[2] > logits_before[2]) {
                std::cout << "   ✅ TARGET IMPROVED! Token 2 probability increased" << std::endl;
            }
            
            return 0; // Success!
        } else {
            std::cout << "   ❌ NO WEIGHT UPDATE!" << std::endl;
            
            // Possible causes
            std::cout << "\n🔍 POSSIBLE CAUSES:" << std::endl;
            std::cout << "   1. Learning rate is effectively zero" << std::endl;
            std::cout << "   2. Gradients are zero (no learning signal)" << std::endl;
            std::cout << "   3. Optimizer not applying gradients" << std::endl;
            std::cout << "   4. Precision mismatch in buffers" << std::endl;
            
            return 1; // Failure
        }
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 