//
// Pattern Recognition Test - Prove the model learns simple patterns
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🧠 Pattern Recognition Test" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Full BookCorpus-sized model
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 2;
        config.learning_rate = 0.01f; // Higher LR for pattern learning
        
        std::cout << "🔧 Full-sized model (1776 vocab, 20M params)" << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized successfully" << std::endl;
        
        // Simple arithmetic patterns
        std::vector<uint32_t> input1 = {100, 200, 300, 400};     // +100 pattern
        std::vector<uint32_t> target1 = {200, 300, 400, 500};
        std::vector<uint32_t> input2 = {150, 250, 350, 450};     // +100 pattern  
        std::vector<uint32_t> target2 = {250, 350, 450, 550};
        
        std::cout << "\n🎯 Training Patterns:" << std::endl;
        std::cout << "Pattern 1: [100,200,300,400] -> [200,300,400,500]" << std::endl;
        std::cout << "Pattern 2: [150,250,350,450] -> [250,350,450,550]" << std::endl;
        std::cout << "Rule: Each token predicts (current_token + 100), except last which adds 100\n" << std::endl;
        
        // Test BEFORE training
        std::cout << "🧪 BEFORE Training - Testing Pattern Recognition:" << std::endl;
        
        auto test_prediction = [&](const std::vector<uint32_t>& test_input, uint32_t expected) {
            std::vector<float> logits;
            if (model.forward(test_input, logits)) {
                // Find token with highest probability
                uint32_t predicted = 0;
                float max_prob = logits[0];
                for (uint32_t i = 1; i < config.vocab_size; i++) {
                    if (logits[i] > max_prob) {
                        max_prob = logits[i];
                        predicted = i;
                    }
                }
                
                std::cout << "  Input: [";
                for (size_t i = 0; i < test_input.size(); i++) {
                    std::cout << test_input[i];
                    if (i < test_input.size() - 1) std::cout << ",";
                }
                std::cout << "] -> Predicted: " << predicted << " (Expected: " << expected << ")";
                
                if (predicted == expected) {
                    std::cout << " ✅ CORRECT";
                } else {
                    std::cout << " ❌ Wrong";
                }
                
                // Show confidence
                float expected_prob = logits[expected];
                float predicted_prob = logits[predicted];
                std::cout << " | Confidence: " << std::fixed << std::setprecision(3) << predicted_prob;
                std::cout << " | Expected prob: " << expected_prob << std::endl;
                
                return (predicted == expected);
            }
            return false;
        };
        
        // Test specific predictions
        bool before_100_correct = test_prediction({100}, 200);  // 100 -> 200
        bool before_200_correct = test_prediction({200}, 300);  // 200 -> 300  
        bool before_150_correct = test_prediction({150}, 250);  // 150 -> 250
        
        int correct_before = (before_100_correct ? 1 : 0) + (before_200_correct ? 1 : 0) + (before_150_correct ? 1 : 0);
        std::cout << "Before training: " << correct_before << "/3 patterns correct\n" << std::endl;
        
        // TRAINING PHASE
        std::cout << "🎓 Training on Patterns (10 steps):" << std::endl;
        
        for (int step = 0; step < 10; step++) {
            float loss;
            bool result = model.trainBatch({input1, input2}, {target1, target2}, loss);
            
            if (result) {
                std::cout << "Step " << std::setw(2) << (step + 1) << ": Training completed";
                if (loss > 0) {
                    std::cout << " (Loss: " << std::fixed << std::setprecision(4) << loss << ")";
                } else {
                    std::cout << " (Loss reporting issue)";
                }
                std::cout << std::endl;
            } else {
                std::cout << "Step " << (step + 1) << ": Training failed ❌" << std::endl;
            }
        }
        
        // Test AFTER training
        std::cout << "\n🧪 AFTER Training - Testing Pattern Recognition:" << std::endl;
        
        bool after_100_correct = test_prediction({100}, 200);  // 100 -> 200
        bool after_200_correct = test_prediction({200}, 300);  // 200 -> 300  
        bool after_150_correct = test_prediction({150}, 250);  // 150 -> 250
        
        int correct_after = (after_100_correct ? 1 : 0) + (after_200_correct ? 1 : 0) + (after_150_correct ? 1 : 0);
        std::cout << "After training: " << correct_after << "/3 patterns correct" << std::endl;
        
        // Test some harder patterns
        std::cout << "\n🔬 Advanced Pattern Tests:" << std::endl;
        test_prediction({300}, 400);  // Should predict 300 + 100 = 400
        test_prediction({250}, 350);  // Should predict 250 + 100 = 350
        test_prediction({100, 200}, 300);  // Sequence pattern
        
        // RESULTS ANALYSIS
        std::cout << "\n📊 Learning Results:" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        int improvement = correct_after - correct_before;
        
        if (improvement > 0) {
            std::cout << "✅ SUCCESS: Model learned! Improvement: +" << improvement << " correct predictions" << std::endl;
            std::cout << "🎉 The full-sized model CAN learn simple patterns!" << std::endl;
        } else if (correct_after >= 2) {
            std::cout << "⚠️  PARTIAL: Model shows some pattern recognition (" << correct_after << "/3)" << std::endl;
        } else {
            std::cout << "❌ CONCERN: Little to no pattern learning detected" << std::endl;
            std::cout << "💡 May need: Higher learning rate, more steps, or different patterns" << std::endl;
        }
        
        if (correct_after >= 2) {
            std::cout << "\n🚀 READY FOR BOOKCORPUS: Pattern recognition confirmed!" << std::endl;
        } else {
            std::cout << "\n⚠️  RECOMMEND: Fix pattern learning before BookCorpus training" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 