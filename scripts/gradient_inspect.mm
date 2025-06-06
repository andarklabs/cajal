//
// Gradient Inspection - Find which backward kernel causes NaN
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../src/host/transformer_model.h"

bool checkBufferForNaN(MTLBuffer* buffer, size_t num_elements, const std::string& name) {
    float* data = (float*)buffer.contents;
    for (size_t i = 0; i < num_elements; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            std::cout << "⚠️  " << name << " has NaN/INF at index " << i << " = " << data[i] << std::endl;
            return true;
        }
    }
    return false;
}

bool checkHalfBufferForNaN(MTLBuffer* buffer, size_t num_elements, const std::string& name) {
    uint16_t* data = (uint16_t*)buffer.contents;
    for (size_t i = 0; i < num_elements; i++) {
        float val = Float16ToFloat32(data[i]);
        if (std::isnan(val) || std::isinf(val)) {
            std::cout << "⚠️  " << name << " has NaN/INF at index " << i << " = " << val << std::endl;
            return true;
        }
    }
    return false;
}

// Convert half to float for inspection
float Float16ToFloat32(uint16_t half_val) {
    uint32_t sign = (half_val >> 15) & 0x1;
    uint32_t exp = (half_val >> 10) & 0x1F;
    uint32_t mantissa = half_val & 0x3FF;
    
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        // Denormalized number
        float val = mantissa / 1024.0f / 1024.0f;
        return sign ? -val : val;
    } else if (exp == 31) {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    
    uint32_t float_exp = (exp - 15 + 127) << 23;
    uint32_t float_mantissa = mantissa << 13;
    uint32_t float_bits = (sign << 31) | float_exp | float_mantissa;
    return *(float*)&float_bits;
}

int main() {
    std::cout << "🔍 GRADIENT INSPECTION - Find the NaN source!" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        // Ultra minimal model
        TransformerConfig config;
        config.vocab_size = 100;       // Very small vocab
        config.embedding_dim = 32;     // Very small model  
        config.num_heads = 2;
        config.num_layers = 1;         // Just ONE layer
        config.ffn_hidden_dim = 64;    // Tiny FFN
        config.max_sequence_length = 4; // Very short sequences
        config.batch_size = 1;         // Single sample
        config.learning_rate = 0.0001f; // Ultra low LR
        
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
        std::cout << "Step 1:" << std::endl;
        float loss1;
        bool result1 = model.trainBatch({input}, {target}, loss1);
        std::cout << "  Result: " << (result1 ? "✅" : "❌") << " Loss: " << loss1 << std::endl;
        
        if (std::isnan(loss1) || std::isinf(loss1)) {
            std::cout << "❌ STEP 1 FAILED - Basic training is broken!" << std::endl;
            return 1;
        }
        
        // Step 2 - Should work
        std::cout << "\nStep 2:" << std::endl;
        float loss2;
        bool result2 = model.trainBatch({input}, {target}, loss2);
        std::cout << "  Result: " << (result2 ? "✅" : "❌") << " Loss: " << loss2 << std::endl;
        
        if (std::isnan(loss2) || std::isinf(loss2)) {
            std::cout << "❌ STEP 2 FAILED - Intermediate failure!" << std::endl;
            return 1;
        }
        
        // Step 3 - Usually fails
        std::cout << "\nStep 3 (Critical Step):" << std::endl;
        
        // Do forward pass first
        std::vector<float> logits;
        if (!model.forward(input, logits)) {
            std::cout << "❌ Forward pass failed before step 3!" << std::endl;
            return 1;
        }
        
        // Check forward pass output for NaN BEFORE doing backward
        bool forward_has_nan = false;
        for (size_t i = 0; i < std::min(logits.size(), size_t(10)); i++) {
            if (std::isnan(logits[i]) || std::isinf(logits[i])) {
                std::cout << "⚠️  Forward pass logits[" << i << "] = " << logits[i] << std::endl;
                forward_has_nan = true;
            }
        }
        
        if (forward_has_nan) {
            std::cout << "❌ Forward pass already has NaN before step 3!" << std::endl;
        } else {
            std::cout << "✅ Forward pass clean, max logit: " << *std::max_element(logits.begin(), logits.end()) << std::endl;
        }
        
        // Now try the training step that usually fails
        float loss3;
        bool result3 = model.trainBatch({input}, {target}, loss3);
        std::cout << "  Result: " << (result3 ? "✅" : "❌") << " Loss: " << loss3 << std::endl;
        
        // DIAGNOSE THE FAILURE
        if (std::isnan(loss3) || std::isinf(loss3)) {
            std::cout << "\n💥 STEP 3 FAILED with NaN/INF!" << std::endl;
            std::cout << "📊 ANALYSIS:" << std::endl;
            std::cout << "  - Step 1 worked (loss: " << loss1 << ")" << std::endl;
            std::cout << "  - Step 2 worked (loss: " << loss2 << ")" << std::endl;
            std::cout << "  - Step 3 failed (loss: " << loss3 << ")" << std::endl;
            
            if (loss2 < 1.0f) {
                std::cout << "  💡 HYPOTHESIS: Model learning too fast, gradients exploding" << std::endl;
                std::cout << "  💡 LIKELY CAUSE: Gradient computation bug in backward pass" << std::endl;
            } else {
                std::cout << "  💡 HYPOTHESIS: Gradual numerical instability buildup" << std::endl;
            }
            
            // Check if it's consistent
            std::cout << "\n🔄 Testing Consistency (5 fresh models):" << std::endl;
            int nan_count = 0;
            for (int test = 0; test < 5; test++) {
                TransformerModel test_model(config);
                if (test_model.initialize()) {
                    float test_loss;
                    test_model.trainBatch({input}, {target}, test_loss); // Step 1
                    test_model.trainBatch({input}, {target}, test_loss); // Step 2
                    test_model.trainBatch({input}, {target}, test_loss); // Step 3
                    if (std::isnan(test_loss)) nan_count++;
                }
            }
            std::cout << "  Consistency: " << nan_count << "/5 models failed on step 3" << std::endl;
            
            if (nan_count >= 4) {
                std::cout << "✅ HIGHLY CONSISTENT BUG - This is reproducible!" << std::endl;
                std::cout << "🎯 ROOT CAUSE: Backward pass computation bug" << std::endl;
            }
            
        } else {
            std::cout << "😮 STEP 3 SUCCEEDED - Bug not reproduced!" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 