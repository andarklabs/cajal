//
// Simple Step Debug - Track exactly when NaN appears
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🔍 SIMPLE STEP DEBUG - Track NaN appearance!" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        // Ultra minimal model
        TransformerConfig config;
        config.vocab_size = 100;       
        config.embedding_dim = 32;     
        config.num_heads = 2;
        config.num_layers = 1;         
        config.ffn_hidden_dim = 64;    
        config.max_sequence_length = 4; 
        config.batch_size = 1;         
        config.learning_rate = 0.0001f; 
        
        std::cout << "🔧 ULTRA-MINIMAL model for NaN debugging" << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // Simple pattern
        std::vector<uint32_t> input = {5};     
        std::vector<uint32_t> target = {6};   
        
        std::cout << "\n🎯 Testing Training Step by Step\n" << std::endl;
        
        // Step 1 - Should work
        std::cout << "STEP 1:" << std::endl;
        float loss1;
        bool result1 = model.trainBatch({input}, {target}, loss1);
        std::cout << "  Result: " << (result1 ? "✅" : "❌") << " Loss: " << std::fixed << std::setprecision(6) << loss1 << std::endl;
        
        if (std::isnan(loss1) || std::isinf(loss1)) {
            std::cout << "❌ STEP 1 FAILED - Basic training is broken!" << std::endl;
            return 1;
        }
        
        // Step 2 - Should work
        std::cout << "\nSTEP 2:" << std::endl;
        float loss2;
        bool result2 = model.trainBatch({input}, {target}, loss2);
        std::cout << "  Result: " << (result2 ? "✅" : "❌") << " Loss: " << std::fixed << std::setprecision(6) << loss2 << std::endl;
        
        if (std::isnan(loss2) || std::isinf(loss2)) {
            std::cout << "❌ STEP 2 FAILED - Intermediate failure!" << std::endl;
            return 1;
        }
        
        // Step 3 - Usually fails
        std::cout << "\nSTEP 3 (Critical Step):" << std::endl;
        float loss3;
        bool result3 = model.trainBatch({input}, {target}, loss3);
        std::cout << "  Result: " << (result3 ? "✅" : "❌") << " Loss: " << std::fixed << std::setprecision(6) << loss3 << std::endl;
        
        // ANALYZE THE PATTERN
        std::cout << "\n📊 ANALYSIS:" << std::endl;
        std::cout << "============" << std::endl;
        
        if (std::isnan(loss3) || std::isinf(loss3)) {
            std::cout << "💥 CONFIRMED: Step 3 fails with NaN/INF" << std::endl;
            
            // Check loss trend
            float loss_change_1_to_2 = loss2 - loss1;
            std::cout << "📈 Loss trend:" << std::endl;
            std::cout << "   Step 1: " << loss1 << std::endl;
            std::cout << "   Step 2: " << loss2 << " (change: " << std::showpos << loss_change_1_to_2 << std::noshowpos << ")" << std::endl;
            std::cout << "   Step 3: " << loss3 << " (NaN/INF)" << std::endl;
            
            // Hypothesis based on loss behavior
            if (std::abs(loss_change_1_to_2) > 1.0f) {
                std::cout << "\n💡 HYPOTHESIS: Large loss changes indicate gradient explosion" << std::endl;
            } else {
                std::cout << "\n💡 HYPOTHESIS: Gradual numerical instability" << std::endl;
            }
            
            // Test with even smaller learning rate
            std::cout << "\n🧪 TESTING with 10x smaller LR (0.00001):" << std::endl;
            TransformerConfig safer_config = config;
            safer_config.learning_rate = 0.00001f;
            
            TransformerModel safer_model(safer_config);
            if (safer_model.initialize()) {
                float safe_loss;
                safer_model.trainBatch({input}, {target}, safe_loss); // Step 1
                std::cout << "   Step 1: " << safe_loss << std::endl;
                safer_model.trainBatch({input}, {target}, safe_loss); // Step 2
                std::cout << "   Step 2: " << safe_loss << std::endl;
                safer_model.trainBatch({input}, {target}, safe_loss); // Step 3
                std::cout << "   Step 3: " << safe_loss << std::endl;
                
                if (std::isnan(safe_loss)) {
                    std::cout << "❌ STILL FAILS with 10x smaller LR - Not a learning rate issue!" << std::endl;
                    std::cout << "🎯 ROOT CAUSE: Backward pass computation bug" << std::endl;
                } else {
                    std::cout << "✅ WORKS with smaller LR - Learning rate was too high" << std::endl;
                }
            }
            
            // Test reproducibility
            std::cout << "\n🔄 REPRODUCIBILITY TEST (10 fresh models):" << std::endl;
            int nan_count = 0;
            for (int test = 0; test < 10; test++) {
                TransformerModel test_model(config);
                if (test_model.initialize()) {
                    float test_loss;
                    test_model.trainBatch({input}, {target}, test_loss); // Step 1
                    test_model.trainBatch({input}, {target}, test_loss); // Step 2
                    test_model.trainBatch({input}, {target}, test_loss); // Step 3
                    if (std::isnan(test_loss)) nan_count++;
                }
            }
            std::cout << "   Failure rate: " << nan_count << "/10 models" << std::endl;
            
            if (nan_count >= 8) {
                std::cout << "✅ HIGHLY REPRODUCIBLE BUG" << std::endl;
                std::cout << "🎯 DEFINITIVE: Backward pass computation bug" << std::endl;
            } else if (nan_count >= 5) {
                std::cout << "⚠️  SOMEWHAT REPRODUCIBLE - possible race condition" << std::endl;
            } else {
                std::cout << "❓ LOW REPRODUCIBILITY - environmental issue?" << std::endl;
            }
            
        } else {
            std::cout << "😮 STEP 3 SUCCEEDED - Bug not reproduced this time!" << std::endl;
            std::cout << "🔄 Trying again..." << std::endl;
            
            // Try a few more times
            for (int retry = 0; retry < 3; retry++) {
                float retry_loss;
                bool retry_result = model.trainBatch({input}, {target}, retry_loss);
                std::cout << "   Retry " << (retry + 1) << ": " << retry_loss << std::endl;
                if (std::isnan(retry_loss)) {
                    std::cout << "❌ FAILED on retry " << (retry + 1) << std::endl;
                    break;
                }
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 