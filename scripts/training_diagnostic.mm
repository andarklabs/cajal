//
// Training Diagnostic - Test if model is actually learning
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <random>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🔬 Training Diagnostic Test" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Create model with same config as BookCorpus training
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 2;
        config.learning_rate = 0.001f; // Higher learning rate for testing
        
        std::cout << "🔧 Initializing model..." << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // Create simple repeating pattern that model should learn
        std::vector<uint32_t> pattern1 = {100, 200, 300, 400, 100, 200, 300, 400};
        std::vector<uint32_t> target1 = {200, 300, 400, 100, 200, 300, 400, 500};
        
        std::vector<uint32_t> pattern2 = {150, 250, 350, 450, 150, 250, 350, 450};
        std::vector<uint32_t> target2 = {250, 350, 450, 150, 250, 350, 450, 550};
        
        std::cout << "\n🧪 Testing if model can learn simple patterns..." << std::endl;
        std::cout << "Pattern 1: [100,200,300,400,100,200,300,400] -> [200,300,400,100,200,300,400,500]" << std::endl;
        std::cout << "Pattern 2: [150,250,350,450,150,250,350,450] -> [250,350,450,150,250,350,450,550]" << std::endl;
        
        // Train for several steps and monitor loss
        for (int step = 0; step < 10; step++) {
            float loss;
            bool result = model.trainBatch({pattern1, pattern2}, {target1, target2}, loss);
            
            if (result) {
                std::cout << "Step " << (step + 1) << ": Loss = " << loss;
                if (loss > 0) {
                    std::cout << " ✅ (Learning detected!)";
                } else {
                    std::cout << " ⚠️  (Zero loss - potential issue)";
                }
                std::cout << std::endl;
            } else {
                std::cout << "Step " << (step + 1) << ": Training failed ❌" << std::endl;
            }
            
            // Test if model is generating different outputs
            if (step == 0 || step == 9) {
                std::cout << "  🔍 Generating test tokens..." << std::endl;
                
                // Test generation on first pattern
                std::vector<uint32_t> test_input = {100, 200, 300};
                std::vector<uint32_t> generated_tokens;
                bool gen_result = model.generate(test_input, 5, generated_tokens, 1.0f);
                
                std::cout << "  📝 Input [100,200,300] -> Generated: ";
                if (gen_result && !generated_tokens.empty()) {
                    for (size_t i = 0; i < generated_tokens.size() && i < 5; i++) {
                        std::cout << generated_tokens[i] << " ";
                    }
                } else {
                    std::cout << "generation failed";
                }
                std::cout << std::endl;
            }
        }
        
        std::cout << "\n🎯 Diagnostic Summary:" << std::endl;
        std::cout << "If loss was > 0 and changing, the model CAN learn!" << std::endl;
        std::cout << "If loss stayed at 0, there's a training issue to fix." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Diagnostic failed: " << e.what() << std::endl;
        return 1;
    }
} 