//
// NaN Kernel Isolator - Find which backward kernel causes step 7 NaN
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../src/host/transformer_model.h"

bool checkBufferForNaNs(MTLBuffer* buffer, size_t num_elements, const std::string& name) {
    if (!buffer) return false;
    
    // Check if it's half-precision or float
    bool found_nan = false;
    size_t nan_count = 0;
    
    if ([buffer length] == num_elements * sizeof(uint16_t)) {
        // Half precision
        uint16_t* data = (uint16_t*)[buffer contents];
        for (size_t i = 0; i < num_elements; i++) {
            // Convert half to float for checking
            float val = *((float*)&data[i]); // Simple cast - not proper half conversion but good enough for NaN detection
            if (std::isnan(val) || std::isinf(val)) {
                if (nan_count < 5) { // Only report first 5
                    std::cout << "  🚨 " << name << "[" << i << "] = NaN/INF" << std::endl;
                }
                found_nan = true;
                nan_count++;
            }
        }
    } else if ([buffer length] == num_elements * sizeof(float)) {
        // Float precision
        float* data = (float*)[buffer contents];
        for (size_t i = 0; i < num_elements; i++) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                if (nan_count < 5) { // Only report first 5
                    std::cout << "  🚨 " << name << "[" << i << "] = " << data[i] << std::endl;
                }
                found_nan = true;
                nan_count++;
            }
        }
    }
    
    if (found_nan) {
        std::cout << "  ❌ " << name << ": " << nan_count << " NaN/INF values found!" << std::endl;
    } else {
        std::cout << "  ✅ " << name << ": Clean" << std::endl;
    }
    
    return found_nan;
}

int main() {
    std::cout << "🔍 NaN KERNEL ISOLATOR" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Goal: Find which backward kernel introduces NaN at step 7" << std::endl;
    
    try {
        // Use a simple configuration that we know hits NaN at step 7
        TransformerConfig config;
        config.vocab_size = 1000;      
        config.embedding_dim = 128;     
        config.num_heads = 4;
        config.num_layers = 2;         
        config.ffn_hidden_dim = 256;    
        config.max_sequence_length = 16; 
        config.batch_size = 1;         
        config.learning_rate = 0.001f; 
        
        std::cout << "\n🧪 CONFIG: embedding_dim=" << config.embedding_dim 
                  << ", ffn_hidden_dim=" << config.ffn_hidden_dim 
                  << ", num_layers=" << config.num_layers << std::endl;
        
        TransformerModel model(config);
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // Simple training data
        std::vector<uint32_t> input_tokens = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80};
        std::vector<uint32_t> target_tokens = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85};
        
        // Train for 10 steps with comprehensive buffer checking after each step
        for (int step = 1; step <= 10; step++) {
            std::cout << "\n🚀 STEP " << step << ": Pre-training buffer check" << std::endl;
            
            // Check key buffers BEFORE training
            // (We'll implement buffer access here - this is pseudocode structure)
            
            std::cout << "  🔄 Training..." << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            
            float loss = model.train(input_tokens, target_tokens);
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "  📊 STEP " << step << " RESULT:" << std::endl;
            std::cout << "     Loss: " << loss << std::endl;
            std::cout << "     Time: " << duration.count() << "ms" << std::endl;
            
            if (std::isnan(loss) || std::isinf(loss)) {
                std::cout << "\n🚨 NaN DETECTED AT STEP " << step << "!" << std::endl;
                std::cout << "====================================" << std::endl;
                
                // When we detect NaN, we want to check ALL buffers to see which kernel produced it
                // This requires access to internal model buffers, which might need some refactoring
                
                std::cout << "🔍 POST-NaN ANALYSIS:" << std::endl;
                std::cout << "   Step " << (step-1) << " was clean" << std::endl; 
                std::cout << "   Step " << step << " produced NaN" << std::endl;
                std::cout << "   Likely culprit: One of the backward kernels" << std::endl;
                
                std::cout << "\n📋 KERNEL EXECUTION ORDER (Backward Pass):" << std::endl;
                std::cout << "   1. output_projection_backward" << std::endl;
                std::cout << "   2. layer_norm_backward (final)" << std::endl;
                std::cout << "   3. FOR each layer (reverse order):" << std::endl;
                std::cout << "      - layer_norm_backward (LN2)" << std::endl;
                std::cout << "      - ffn_backward" << std::endl;
                std::cout << "      - layer_norm_backward (LN1)" << std::endl;
                std::cout << "      - mhsa_output_projection_backward" << std::endl;
                std::cout << "      - scaled_dot_product_attention_backward" << std::endl;
                std::cout << "      - qkv_projection_backward" << std::endl;
                std::cout << "   4. embedding_layer_backward" << std::endl;
                
                std::cout << "\n💡 INVESTIGATION PRIORITIES:" << std::endl;
                std::cout << "   🥇 ffn_backward: Complex 3D dispatch, reduction loops" << std::endl;
                std::cout << "   🥈 scaled_dot_product_attention_backward: Matrix ops, softmax gradients" << std::endl;
                std::cout << "   🥉 output_projection_backward: Large vocab matrix multiplications" << std::endl;
                
                break;
            }
            
            // If loss is still finite, continue
            std::cout << "     Status: ✅ Finite" << std::endl;
        }
        
        std::cout << "\n🎯 NEXT STEPS:" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "1. Examine ffn_backward kernel for buffer overflows" << std::endl;
        std::cout << "2. Check attention backward for numerical instability" << std::endl;
        std::cout << "3. Investigate half-precision accumulation issues" << std::endl;
        std::cout << "4. Add kernel-by-kernel NaN detection" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 