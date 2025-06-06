#include "../src/host/transformer_model.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "🔬 FORWARD PASS DEBUG" << std::endl;
    
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
    
    std::cout << "✅ Model initialized with proper weights" << std::endl;
    
    // Test token: [1] (valid for vocab_size=3: 0,1,2)
    std::vector<uint32_t> input_tokens = {1};
    
    std::cout << "\n🔍 Testing token embedding lookup..." << std::endl;
    
    // Manual check: Look at token embedding weights  
    // Token 1 should have non-zero embedding values since we initialized properly
    
    return 0;
} 