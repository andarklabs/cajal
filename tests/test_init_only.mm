#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include "transformer_model.h"

int main() {
    std::cout << "🔍 Testing Model Initialization Only" << std::endl;
    
    // Very small model
    TransformerConfig config;
    config.embedding_dim = 128;
    config.num_layers = 1;
    config.num_heads = 2;
    config.ffn_hidden_dim = 256;
    config.max_sequence_length = 32;
    config.vocab_size = 1000;
    
    std::cout << "Creating model..." << std::endl;
    TransformerModel model(config);
    
    std::cout << "Initializing model..." << std::endl;
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized successfully!" << std::endl;
    std::cout << "Parameters: " << model.getParameterCount() << std::endl;
    
    return 0;
} 