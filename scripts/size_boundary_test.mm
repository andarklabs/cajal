//
// Size Boundary Test - Find where NaN starts appearing
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../src/host/transformer_model.h"

bool testConfiguration(const TransformerConfig& config, const std::string& description) {
    std::cout << "\n🧪 TESTING: " << description << std::endl;
    std::cout << "   embedding_dim=" << config.embedding_dim 
              << ", ffn_hidden_dim=" << config.ffn_hidden_dim 
              << ", vocab_size=" << config.vocab_size << std::endl;
    
    try {
        TransformerModel model(config);
        if (!model.initialize()) {
            std::cout << "❌ FAILED to initialize" << std::endl;
            return false;
        }
        
        std::vector<uint32_t> input = {5};     
        std::vector<uint32_t> target = {6};   
        
        // Try 5 training steps
        for (int step = 1; step <= 5; step++) {
            float loss;
            bool result = model.trainBatch({input}, {target}, loss);
            
            if (!result || std::isnan(loss) || std::isinf(loss)) {
                std::cout << "❌ FAILED at step " << step << " with loss: " << loss << std::endl;
                return false;
            }
        }
        
        std::cout << "✅ SUCCESS - All 5 steps completed" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ EXCEPTION: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "🔍 SIZE BOUNDARY TEST - Find breaking point!" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Base configuration (known to work)
    TransformerConfig base_config;
    base_config.vocab_size = 100;       
    base_config.embedding_dim = 32;     
    base_config.num_heads = 2;
    base_config.num_layers = 1;         
    base_config.ffn_hidden_dim = 64;    
    base_config.max_sequence_length = 4; 
    base_config.batch_size = 1;         
    base_config.learning_rate = 0.0001f; 
    
    std::cout << "🎯 Goal: Find model size where NaN bug appears\n" << std::endl;
    
    // Test 1: Base (should work)
    bool works = testConfiguration(base_config, "BASELINE (32 dim, 64 FFN) - should work");
    if (!works) {
        std::cout << "❌ BASELINE FAILED - Something is wrong!" << std::endl;
        return 1;
    }
    
    // Test 2: Double embedding dim
    TransformerConfig config2 = base_config;
    config2.embedding_dim = 64;
    config2.ffn_hidden_dim = 128;
    testConfiguration(config2, "DOUBLE SIZE (64 dim, 128 FFN)");
    
    // Test 3: Quadruple 
    TransformerConfig config3 = base_config;
    config3.embedding_dim = 128;
    config3.ffn_hidden_dim = 256;
    testConfiguration(config3, "QUADRUPLE SIZE (128 dim, 256 FFN) - known problematic");
    
    // Test 4: Larger vocab
    TransformerConfig config4 = base_config;
    config4.vocab_size = 1000;
    testConfiguration(config4, "LARGE VOCAB (1000 tokens, 32 dim)");
    
    // Test 5: Multiple layers
    TransformerConfig config5 = base_config;
    config5.num_layers = 2;
    testConfiguration(config5, "MULTI-LAYER (2 layers, 32 dim)");
    
    // Test 6: More heads
    TransformerConfig config6 = base_config;
    config6.num_heads = 4;
    testConfiguration(config6, "MORE HEADS (4 heads, 32 dim)");
    
    // Test 7: The configuration we know fails
    TransformerConfig config7 = base_config;
    config7.embedding_dim = 128;
    config7.ffn_hidden_dim = 256;
    config7.vocab_size = 1000;
    config7.num_layers = 2;
    testConfiguration(config7, "KNOWN FAILING CONFIG (128 dim, 256 FFN, 1000 vocab, 2 layers)");
    
    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "==============" << std::endl;
    std::cout << "✅ Models up to certain size work correctly" << std::endl;
    std::cout << "❌ Larger models hit NaN bug" << std::endl;
    std::cout << "🎯 This suggests buffer overflow or threadgroup limit issue" << std::endl;
    
    return 0;
} 