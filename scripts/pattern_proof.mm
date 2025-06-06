//
// Pattern Proof - Quick test showing model learns simple patterns
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🧠 Pattern Learning Proof Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        // Minimal config for speed
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 256;  // Smaller for speed
        config.num_heads = 4;        // Fewer heads
        config.num_layers = 2;       // Fewer layers
        config.ffn_hidden_dim = 512; // Smaller FFN
        config.max_sequence_length = 512;
        config.batch_size = 1;       // Single sequence
        config.learning_rate = 0.1f; // High LR for fast learning
        
        std::cout << "🔧 Initializing smaller model for speed..." << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized (much smaller & faster)" << std::endl;
        
        // Super simple pattern: A->B, B->C, C->D
        std::vector<uint32_t> input = {100, 101, 102};     // A, B, C
        std::vector<uint32_t> target = {101, 102, 103};    // B, C, D
        
        std::cout << "\n🎯 Learning Simple Pattern:" << std::endl;
        std::cout << "100 -> 101, 101 -> 102, 102 -> 103" << std::endl;
        std::cout << "Pattern: Each token predicts token_id + 1\n" << std::endl;
        
        // Train just a few steps
        std::cout << "📚 Training (5 steps only)..." << std::endl;
        float final_loss = 0.0f;
        
        for (int step = 0; step < 5; step++) {
            float loss;
            if (model.trainStep(input, target, loss)) {
                final_loss = loss;
                std::cout << "Step " << (step + 1) << ": Loss = " << loss << std::endl;
            } else {
                std::cout << "Step " << (step + 1) << ": Failed" << std::endl;
            }
        }
        
        std::cout << "\n🧪 Testing Pattern Recognition..." << std::endl;
        
        // Test: Can it predict 101 after seeing 100?
        std::vector<uint32_t> test_input = {100};
        std::vector<float> logits;
        
        if (model.forward(test_input, logits)) {
            // Find highest probability token
            uint32_t predicted_token = 0;
            float max_prob = logits[0];
            for (uint32_t i = 1; i < config.vocab_size; i++) {
                if (logits[i] > max_prob) {
                    max_prob = logits[i];
                    predicted_token = i;
                }
            }
            
            std::cout << "Input: 100 -> Predicted: " << predicted_token;
            if (predicted_token == 101) {
                std::cout << " ✅ CORRECT! (Expected 101)" << std::endl;
            } else {
                std::cout << " ❌ Wrong (Expected 101)" << std::endl;
            }
            
            // Check probability distribution
            float prob_101 = logits[101];
            float prob_100 = logits[100];
            float prob_102 = logits[102];
            
            std::cout << "Probabilities:" << std::endl;
            std::cout << "  Token 100: " << prob_100 << std::endl;
            std::cout << "  Token 101: " << prob_101 << " (target)" << std::endl;
            std::cout << "  Token 102: " << prob_102 << std::endl;
            
            if (prob_101 > prob_100 && prob_101 > prob_102) {
                std::cout << "✅ SUCCESS: Token 101 has highest probability!" << std::endl;
            }
            
        } else {
            std::cout << "❌ Forward pass failed" << std::endl;
        }
        
        // Test another pattern
        test_input = {101};
        if (model.forward(test_input, logits)) {
            uint32_t predicted_token = 0;
            float max_prob = logits[0];
            for (uint32_t i = 1; i < config.vocab_size; i++) {
                if (logits[i] > max_prob) {
                    max_prob = logits[i];
                    predicted_token = i;
                }
            }
            
            std::cout << "Input: 101 -> Predicted: " << predicted_token;
            if (predicted_token == 102) {
                std::cout << " ✅ CORRECT! (Expected 102)" << std::endl;
            } else {
                std::cout << " ❌ Wrong (Expected 102)" << std::endl;
            }
        }
        
        std::cout << "\n🎉 Pattern Learning Verification Complete!" << std::endl;
        std::cout << "Final Loss: " << final_loss << std::endl;
        
        if (final_loss < 2.0f) {
            std::cout << "✅ Model successfully learned the simple pattern!" << std::endl;
        } else {
            std::cout << "⚠️  Model may need more training or different parameters." << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 