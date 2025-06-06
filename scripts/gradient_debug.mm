//
// Gradient Debug - Check if gradients flow and weights update
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🔬 GRADIENT DEBUG - Are weights actually updating?" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        // Tiny model
        TransformerConfig config;
        config.vocab_size = 100;       // Very small vocab
        config.embedding_dim = 32;     // Very small embedding
        config.num_heads = 2;
        config.num_layers = 1;         // Single layer
        config.ffn_hidden_dim = 64;
        config.max_sequence_length = 4;
        config.batch_size = 1;
        config.learning_rate = 0.01f;  // More reasonable LR
        
        std::cout << "🔧 MICRO model (100 vocab, ~10K params)" << std::endl;
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
        
        // Get model weights BEFORE training (sample a few)
        std::cout << "\n📊 WEIGHT INSPECTION:" << std::endl;
        
        // We need to add a method to inspect weights, but for now let's check predictions
        std::cout << "🧪 BEFORE Training:" << std::endl;
        std::vector<float> logits_before;
        if (model.forward(input, logits_before)) {
            std::cout << "  Logit[0]: " << std::fixed << std::setprecision(6) << logits_before[0] << std::endl;
            std::cout << "  Logit[1]: " << logits_before[1] << std::endl;
            std::cout << "  Logit[2]: " << logits_before[2] << std::endl;
            std::cout << "  Logit[3]: " << logits_before[3] << std::endl;
            
            // Check if all logits are the same (uniform distribution)
            bool all_same = true;
            for (int i = 1; i < 10; i++) {
                if (std::abs(logits_before[i] - logits_before[0]) > 0.001f) {
                    all_same = false;
                    break;
                }
            }
            
            if (all_same) {
                std::cout << "  ⚠️  UNIFORM DISTRIBUTION - Model not making predictions" << std::endl;
            } else {
                std::cout << "  ✅ NON-UNIFORM - Model making some predictions" << std::endl;
            }
        }
        
        // Single training step with detailed loss analysis
        std::cout << "\n🎓 Training Step:" << std::endl;
        float loss;
        bool result = model.trainBatch({input}, {target}, loss);
        
        std::cout << "  Result: " << (result ? "✅ Success" : "❌ Failed") << std::endl;
        std::cout << "  Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
        
        // Expected loss for uniform distribution over 100 tokens
        float expected_uniform_loss = std::log(100.0f);
        std::cout << "  Expected uniform loss: " << expected_uniform_loss << std::endl;
        
        if (std::abs(loss - expected_uniform_loss) < 0.01f) {
            std::cout << "  🚨 PROBLEM: Loss matches uniform distribution!" << std::endl;
        } else {
            std::cout << "  ✅ Loss differs from uniform - some learning" << std::endl;
        }
        
        // Check predictions AFTER training
        std::cout << "\n🧪 AFTER Training:" << std::endl;
        std::vector<float> logits_after;
        if (model.forward(input, logits_after)) {
            std::cout << "  Logit[0]: " << std::fixed << std::setprecision(6) << logits_after[0] << std::endl;
            std::cout << "  Logit[1]: " << logits_after[1] << std::endl;
            std::cout << "  Logit[2]: " << logits_after[2] << std::endl;
            std::cout << "  Logit[3]: " << logits_after[3] << std::endl;
            
            // Check for changes
            std::cout << "\n📈 CHANGES:" << std::endl;
            std::cout << "  Δ Logit[0]: " << std::showpos << (logits_after[0] - logits_before[0]) << std::noshowpos << std::endl;
            std::cout << "  Δ Logit[1]: " << std::showpos << (logits_after[1] - logits_before[1]) << std::noshowpos << std::endl;
            std::cout << "  Δ Logit[2]: " << std::showpos << (logits_after[2] - logits_before[2]) << std::noshowpos << std::endl;
            std::cout << "  Δ Logit[3]: " << std::showpos << (logits_after[3] - logits_before[3]) << std::noshowpos << std::endl;
            
            // Check if ANY logit changed significantly
            bool any_change = false;
            for (int i = 0; i < 10; i++) {
                if (std::abs(logits_after[i] - logits_before[i]) > 0.001f) {
                    any_change = true;
                    break;
                }
            }
            
            if (any_change) {
                std::cout << "  ✅ WEIGHTS UPDATED: Logits changed after training" << std::endl;
            } else {
                std::cout << "  🚨 NO WEIGHT UPDATE: Logits identical - weights not changing!" << std::endl;
            }
            
            // Check target improvement specifically
            float target_improvement = logits_after[2] - logits_before[2];
            std::cout << "  Target improvement (token 2): " << std::showpos << target_improvement << std::noshowpos << std::endl;
            
            if (target_improvement > 0.001f) {
                std::cout << "  ✅ LEARNING: Target token probability increased" << std::endl;
            } else if (target_improvement < -0.001f) {
                std::cout << "  ⚠️  ANTI-LEARNING: Target token probability decreased" << std::endl;
            } else {
                std::cout << "  🚨 NO LEARNING: Target token unchanged" << std::endl;
            }
        }
        
        // FINAL DIAGNOSIS
        std::cout << "\n🔍 FINAL DIAGNOSIS:" << std::endl;
        std::cout << "===================" << std::endl;
        
        if (std::abs(loss - expected_uniform_loss) < 0.01f) {
            std::cout << "🚨 CRITICAL: Model outputting uniform distribution" << std::endl;
            std::cout << "   Possible causes:" << std::endl;
            std::cout << "   - Gradients not computed correctly" << std::endl;
            std::cout << "   - Weights not being updated" << std::endl;
            std::cout << "   - Optimizer not working" << std::endl;
        } else {
            std::cout << "✅ Model producing non-uniform outputs" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 