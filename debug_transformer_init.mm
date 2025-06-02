#include "src/host/transformer_model.h"
#include <iostream>

int main() {
    std::cout << "=== Debug Transformer Initialization ===" << std::endl;
    
    std::cout << "1. Creating config..." << std::endl;
    TransformerConfig config;
    config.vocab_size = 50;
    config.embedding_dim = 32;
    config.num_layers = 1;
    config.num_heads = 2;
    config.ffn_hidden_dim = 128;
    config.max_sequence_length = 8;
    config.batch_size = 1;
    config.learning_rate = 1e-3f;
    config.epsilon = 1e-5f;
    config.pad_token_id = 0;
    config.use_half_precision = true;
    config.float_logits = true;
    std::cout << "✓ Config created" << std::endl;
    
    std::cout << "2. Creating TransformerModel..." << std::endl;
    TransformerModel model(config);
    std::cout << "✓ TransformerModel constructor completed" << std::endl;
    
    std::cout << "3. Starting initialization..." << std::endl;
    bool success = model.initialize();
    
    if (success) {
        std::cout << "✅ Initialization completed successfully!" << std::endl;
    } else {
        std::cout << "❌ Initialization failed" << std::endl;
        return 1;
    }
    
    return 0;
} 