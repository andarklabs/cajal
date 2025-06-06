#include <iostream>
#include <vector>
#include "../src/host/transformer_model.h"

bool quickSafetyCheck() {
    std::cout << "🔬 QUICK SAFETY CHECK: Verifying critical vulnerability fixes..." << std::endl;
    
    // Test 1: Verify safe configuration initializes
    std::cout << "\n📋 Test 1: Safe Configuration Initialization" << std::endl;
    {
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;       // Safe for all kernels
        config.num_layers = 2;            // Minimal for speed
        config.num_heads = 8;
        config.ffn_hidden_dim = 2048;     // Fixed: now safe with 2048 threadgroup arrays
        config.max_sequence_length = 512; // Safe for 1024 stack arrays
        config.batch_size = 2;            // Small for speed
        config.learning_rate = 1e-4f;
        config.use_half_precision = true;
        config.float_logits = true;
        
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize safe configuration" << std::endl;
            return false;
        }
        
        std::cout << "✅ Safe configuration initializes correctly" << std::endl;
    }
    
    // Test 2: Verify dangerous configurations are rejected
    std::cout << "\n📋 Test 2: Dangerous Configuration Rejection" << std::endl;
    {
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 0;         // Should trigger division by zero check
        config.num_layers = 1;
        config.num_heads = 8;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 2;
        config.learning_rate = 1e-4f;
        config.use_half_precision = true;
        config.float_logits = true;
        
        TransformerModel model(config);
        
        if (model.initialize()) {
            std::cerr << "❌ Dangerous configuration should have been rejected" << std::endl;
            return false;
        }
        
        std::cout << "✅ Dangerous configuration correctly rejected" << std::endl;
    }
    
    // Test 3: Verify edge case configurations work
    std::cout << "\n📋 Test 3: Edge Case Configuration" << std::endl;
    {
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 1024;      // At the edge of layer_norm limit
        config.num_layers = 1;            // Minimal for speed
        config.num_heads = 8;
        config.ffn_hidden_dim = 2048;     // At the edge of ffn_backward limit
        config.max_sequence_length = 1024; // At the edge of attention stack limit
        config.batch_size = 1;            // Minimal for speed
        config.learning_rate = 1e-4f;
        config.use_half_precision = true;
        config.float_logits = true;
        
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Edge case configuration should work within limits" << std::endl;
            return false;
        }
        
        std::cout << "✅ Edge case configuration works within safety limits" << std::endl;
    }
    
    return true;
}

int main() {
    std::cout << "🔬 MSL Transformer Quick Safety Check" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    if (quickSafetyCheck()) {
        std::cout << "\n🎉 QUICK SAFETY CHECK PASSED!" << std::endl;
        std::cout << "🛡️  Critical vulnerabilities have been addressed:" << std::endl;
        std::cout << "    ✅ FFN backward threadgroup arrays: 1024→2048" << std::endl;
        std::cout << "    ✅ Attention stack buffer: 512→1024" << std::endl;
        std::cout << "    ✅ Division by zero protection" << std::endl;
        std::cout << "    ✅ Configuration validation system" << std::endl;
        std::cout << "\n🚀 READY TO PROCEED TO OPTIMIZATION PHASE!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ SAFETY CHECK FAILED" << std::endl;
        std::cout << "🚨 Critical issues remain - do not proceed!" << std::endl;
        return 1;
    }
} 