#include "src/host/transformer_model.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Testing Forward Pass ===" << std::endl;
    
    // Create a small test config
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
    
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Model initialized successfully" << std::endl;
    
    // Test forward pass with simple input
    std::vector<uint32_t> input_tokens = {1, 2, 3, 4, 5}; // Simple sequence
    std::vector<float> output_logits;
    
    std::cout << "Running forward pass..." << std::endl;
    bool success = model.forward(input_tokens, output_logits);
    
    if (success) {
        std::cout << "✅ Forward pass completed successfully!" << std::endl;
        std::cout << "Input tokens: " << input_tokens.size() << std::endl;
        std::cout << "Output logits: " << output_logits.size() << std::endl;
        std::cout << "Expected logits size: " << input_tokens.size() * config.vocab_size << std::endl;
        
        // Print first few logits
        std::cout << "First 10 logits: ";
        for (size_t i = 0; i < std::min(size_t(10), output_logits.size()); i++) {
            std::cout << output_logits[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "❌ Forward pass failed" << std::endl;
        return 1;
    }
    
    return 0;
} 