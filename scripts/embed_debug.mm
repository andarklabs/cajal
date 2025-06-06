#include "../src/host/transformer_model.h"
#include <iostream>
#include <iomanip>

// Quick test: manually check that token embeddings work
int main() {
    std::cout << "🔍 EMBEDDING DEBUG" << std::endl;
    
    TransformerConfig config;
    config.vocab_size = 3;
    config.embedding_dim = 4;
    config.num_heads = 1;
    config.num_layers = 0;  // NO layers - just test embedding lookup
    config.ffn_hidden_dim = 8;
    config.max_sequence_length = 2;
    config.use_half_precision = false;  // Use float for debugging
    
    TransformerModel model(config);
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized (0 layers - just embeddings + output)" << std::endl;
    
    // Test token: [1] 
    std::vector<uint32_t> input_tokens = {1};
    
    std::cout << "\n🔍 Forward pass with 0 transformer layers..." << std::endl;
    std::vector<float> logits;
    if (!model.forward(input_tokens, logits)) {
        std::cerr << "❌ Forward pass failed!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Forward pass succeeded" << std::endl;
    std::cout << "📊 Logits: ";
    for (size_t i = 0; i < logits.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << logits[i] << " ";
    }
    std::cout << std::endl;
    
    // Check for non-zero logits
    bool has_nonzero = false;
    for (float logit : logits) {
        if (abs(logit) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    
    if (has_nonzero) {
        std::cout << "✅ SUCCESS: Non-zero logits found!" << std::endl;
        std::cout << "🎯 Embedding + output projection working correctly" << std::endl;
    } else {
        std::cout << "❌ STILL ALL ZEROS: Issue is in embedding lookup or output projection" << std::endl;
    }
    
    return 0;
} 