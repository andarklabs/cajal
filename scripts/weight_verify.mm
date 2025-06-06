#include "../src/host/transformer_model.h"
#include <iostream>
#include <iomanip>

// Quick test: verify weights are actually stored in buffers after initialization
int main() {
    std::cout << "🔍 WEIGHT VERIFICATION DEBUG" << std::endl;
    
    TransformerConfig config;
    config.vocab_size = 3;
    config.embedding_dim = 4;
    config.num_heads = 1;
    config.num_layers = 1;
    config.ffn_hidden_dim = 8;
    config.max_sequence_length = 2;
    config.use_half_precision = true;   // FORCE half precision - MSL kernels expect half
    
    TransformerModel model(config);
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized - checking if weights actually exist..." << std::endl;
    
    // This is a hacky way to access the weights, but let's check if token embeddings are non-zero
    // We can't access private members directly, but we can infer from behavior
    
    // Method: Check logits BEFORE any training - they should be non-zero due to weight initialization
    std::vector<uint32_t> input_tokens = {1}; // Valid token for vocab_size=3
    std::vector<float> logits;
    
    std::cout << "\n🔍 Forward pass immediately after initialization..." << std::endl;
    if (!model.forward(input_tokens, logits)) {
        std::cerr << "❌ Forward pass failed!" << std::endl;
        return 1;
    }
    
    std::cout << "📊 Raw logits: ";
    for (size_t i = 0; i < logits.size(); i++) {
        std::cout << std::scientific << std::setprecision(6) << logits[i] << " ";
    }
    std::cout << std::endl;
    
    // Check for non-zero logits
    bool all_zero = true;
    bool has_nan = false;
    float max_abs = 0.0f;
    
    for (float logit : logits) {
        if (std::isnan(logit) || std::isinf(logit)) {
            has_nan = true;
        }
        if (abs(logit) > 1e-10) {
            all_zero = false;
        }
        max_abs = std::max(max_abs, abs(logit));
    }
    
    std::cout << "📊 Max absolute logit: " << max_abs << std::endl;
    
    if (has_nan) {
        std::cout << "❌ NaN/Inf detected in forward pass" << std::endl;
    } else if (all_zero) {
        std::cout << "❌ ALL LOGITS ARE ZERO - weights not initialized or forward pass broken" << std::endl;
    } else {
        std::cout << "✅ Non-zero logits found - weights and forward pass working!" << std::endl;
    }
    
    // Try different token
    std::cout << "\n🔍 Testing different token [0]..." << std::endl;
    input_tokens = {0};
    if (model.forward(input_tokens, logits)) {
        max_abs = 0.0f;
        for (float logit : logits) {
            max_abs = std::max(max_abs, abs(logit));
        }
        std::cout << "📊 Token 0 max absolute logit: " << max_abs << std::endl;
    }
    
    std::cout << "\n🔍 Testing different token [2]..." << std::endl;
    input_tokens = {2};
    if (model.forward(input_tokens, logits)) {
        max_abs = 0.0f;
        for (float logit : logits) {
            max_abs = std::max(max_abs, abs(logit));
        }
        std::cout << "📊 Token 2 max absolute logit: " << max_abs << std::endl;
    }
    
    return 0;
} 