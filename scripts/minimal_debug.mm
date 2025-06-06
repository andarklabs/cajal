//
// Minimal Debug - Find exactly where the issue occurs
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include "../src/host/transformer_model.h"
#include <iomanip>

int main() {
    std::cout << "🔬 MINIMAL DEBUG - Find NaN Source" << std::endl;
    
    // Ultra minimal configuration
    TransformerConfig config;
    config.vocab_size = 3;
    config.embedding_dim = 4;
    config.num_heads = 1;
    config.num_layers = 1;
    config.ffn_hidden_dim = 8;
    config.max_sequence_length = 2;
    config.use_half_precision = false;  // Use float for debugging
    
    TransformerModel model(config);
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized" << std::endl;
    
    // Simple single token test: [1] -> [2]
    std::vector<uint32_t> input_tokens = {1};
    std::vector<uint32_t> target_tokens = {2};
    
    std::cout << "\n🔍 Step 1: Forward Pass Only" << std::endl;
    
    // Try a simple forward pass
    std::vector<float> logits;
    if (!model.forward(input_tokens, logits)) {
        std::cerr << "❌ Forward pass failed!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Forward pass succeeded" << std::endl;
    std::cout << "📊 Logits: ";
    for (size_t i = 0; i < std::min(logits.size(), size_t(5)); i++) {
        std::cout << std::fixed << std::setprecision(4) << logits[i] << " ";
    }
    std::cout << std::endl;
    
    // Check for NaNs in logits
    bool has_nan = false;
    for (float logit : logits) {
        if (std::isnan(logit) || std::isinf(logit)) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FOUND NaN/Inf in forward pass logits!" << std::endl;
        return 1;
    }
    
    std::cout << "\n🔍 Step 2: Loss Computation" << std::endl;
    
    // Try loss computation manually  
    float loss = 0.0f;
    if (!model.evaluate(input_tokens, target_tokens, loss)) {
        std::cerr << "❌ Loss computation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Loss computation succeeded" << std::endl;
    std::cout << "📊 Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
    
    if (std::isnan(loss) || std::isinf(loss)) {
        std::cout << "❌ FOUND NaN/Inf in loss!" << std::endl;
        return 1;
    }
    
    std::cout << "\n✅ No NaN found in basic forward pass and loss!" << std::endl;
    std::cout << "🔍 Issue is likely in the training batch pipeline" << std::endl;
    
    return 0;
} 