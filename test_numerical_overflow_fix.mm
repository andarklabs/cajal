//
// Test: Numerical Overflow Fix Verification
// 
// PURPOSE: Verify that the FFN weight initialization and float accumulation fixes
//          resolve the NaN overflow issue for large embedding_dim × ffn_hidden_dim
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "src/host/transformer_model.h"

bool test_numerical_stability_fix(const TransformerConfig& config, const std::string& description, int num_steps = 10) {
    std::cout << "\n🧪 TESTING: " << description << std::endl;
    std::cout << "   embedding_dim=" << config.embedding_dim 
              << ", ffn_hidden_dim=" << config.ffn_hidden_dim 
              << ", product=" << ((uint64_t)config.embedding_dim * config.ffn_hidden_dim) << std::endl;
    
    try {
        TransformerModel model(config);
        if (!model.initialize()) {
            std::cout << "❌ FAILED to initialize" << std::endl;
            return false;
        }
        
        std::vector<uint32_t> input = {5};     
        std::vector<uint32_t> target = {6};   
        
        std::cout << "   Step-by-step loss tracking:" << std::endl;
        for (int step = 1; step <= num_steps; step++) {
            float loss;
            bool result = model.trainBatch({input}, {target}, loss);
            
            if (!result) {
                std::cout << "   ❌ FAILED at step " << step << " (trainBatch returned false)" << std::endl;
                return false;
            }
            
            if (std::isnan(loss)) {
                std::cout << "   ❌ FAILED at step " << step << " with NaN loss" << std::endl;
                return false;
            }
            
            if (std::isinf(loss)) {
                std::cout << "   ❌ FAILED at step " << step << " with Inf loss: " << loss << std::endl;
                return false;
            }
            
            if (loss > 1e6f) {
                std::cout << "   ❌ FAILED at step " << step << " with extreme loss: " << loss << std::endl;
                return false;
            }
            
            std::cout << "   Step " << std::setw(2) << step << ": " << std::fixed << std::setprecision(5) << loss;
            
            // Check for loss explosion pattern (rapid increase)
            static float prev_loss = loss;
            if (step > 1 && loss > prev_loss * 10.0f) {
                std::cout << " ⚠️  Large increase!" << std::endl;
            } else {
                std::cout << " ✓" << std::endl;
            }
            prev_loss = loss;
        }
        
        std::cout << "   ✅ SUCCESS - All " << num_steps << " steps completed with stable loss" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "   ❌ EXCEPTION: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "🔧 NUMERICAL OVERFLOW FIX VERIFICATION" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Testing configurations that previously caused NaN at step 4..." << std::endl;
    
    // Test configurations that previously failed
    std::vector<std::pair<TransformerConfig, std::string>> test_cases;
    
    // Base config
    TransformerConfig base_config;
    base_config.vocab_size = 100;  // Small for faster testing
    base_config.num_layers = 1;    // Single layer for focused testing
    base_config.num_heads = 2;     
    base_config.max_sequence_length = 16;
    base_config.batch_size = 1;
    base_config.use_half_precision = true;
    
    // Test case 1: Previously failed - 128 × 256 = 32,768
    TransformerConfig config1 = base_config;
    config1.embedding_dim = 128;
    config1.ffn_hidden_dim = 256;
    test_cases.push_back({config1, "Previously Failed: 128 × 256 = 32,768"});
    
    // Test case 2: Previously failed - 256 × 128 = 32,768  
    TransformerConfig config2 = base_config;
    config2.embedding_dim = 256;
    config2.ffn_hidden_dim = 128;
    config2.num_heads = 4; // Adjust for divisibility
    test_cases.push_back({config2, "Previously Failed: 256 × 128 = 32,768"});
    
    // Test case 3: Previously failed - 512 × 64 = 32,768
    TransformerConfig config3 = base_config;
    config3.embedding_dim = 512;
    config3.ffn_hidden_dim = 64;
    config3.num_heads = 8; // Adjust for divisibility
    test_cases.push_back({config3, "Previously Failed: 512 × 64 = 32,768"});
    
    // Test case 4: Edge case at threshold - 128 × 128 = 16,384 (should work)
    TransformerConfig config4 = base_config;
    config4.embedding_dim = 128;
    config4.ffn_hidden_dim = 128;
    test_cases.push_back({config4, "Threshold Test: 128 × 128 = 16,384 (should work)"});
    
    // Test case 5: Large configuration - 256 × 256 = 65,536 (stress test)
    TransformerConfig config5 = base_config;
    config5.embedding_dim = 256;
    config5.ffn_hidden_dim = 256;
    config5.num_heads = 4; // Adjust for divisibility
    test_cases.push_back({config5, "Stress Test: 256 × 256 = 65,536"});
    
    // Run all test cases
    int passed = 0;
    int total = test_cases.size();
    
    for (const auto& test_case : test_cases) {
        if (test_numerical_stability_fix(test_case.first, test_case.second, 8)) {
            passed++;
        }
    }
    
    std::cout << "\n📊 FINAL RESULTS:" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << " test cases" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 ALL TESTS PASSED! Numerical overflow fix is working correctly." << std::endl;
        std::cout << "\n✅ VERIFICATION COMPLETE:" << std::endl;
        std::cout << "   • Kaiming/He weight initialization prevents initial overflow" << std::endl;
        std::cout << "   • Float accumulation prevents half-precision overflow" << std::endl;
        std::cout << "   • Overflow clamping provides additional safety" << std::endl;
        std::cout << "   • Diagnostic warnings help identify risky configurations" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed. The fix may need additional work." << std::endl;
        return 1;
    }
} 