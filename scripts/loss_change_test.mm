//
// Loss Change Test - Check if loss decreases with proper learning rate
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "📈 Loss Change Test" << std::endl;
    std::cout << "===================" << std::endl;
    
    try {
        // Use higher learning rate for clearer learning signal
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 2;
        config.learning_rate = 0.01f; // Higher learning rate for clear change
        
        std::cout << "🔧 Initializing model with LR = " << config.learning_rate << "..." << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        // Create simple pattern that should be learnable
        std::vector<uint32_t> input1 = {100, 200, 300, 400};
        std::vector<uint32_t> target1 = {200, 300, 400, 500};
        std::vector<uint32_t> input2 = {150, 250, 350, 450};
        std::vector<uint32_t> target2 = {250, 350, 450, 550};
        
        std::cout << "\n📊 Training Pattern Recognition Task:" << std::endl;
        std::cout << "Input:  [100,200,300,400] -> Target: [200,300,400,500]" << std::endl;
        std::cout << "Input:  [150,250,350,450] -> Target: [250,350,450,550]" << std::endl;
        std::cout << "Goal: Model should learn to predict next in sequence + final increment\n" << std::endl;
        
        float initial_loss = 0.0f;
        float final_loss = 0.0f;
        
        // Train for more steps and watch loss
        for (int step = 0; step < 20; step++) {
            float loss;
            bool result = model.trainBatch({input1, input2}, {target1, target2}, loss);
            
            if (result) {
                if (step == 0) initial_loss = loss;
                if (step == 19) final_loss = loss;
                
                std::cout << "Step " << std::setw(2) << (step + 1) << ": Loss = " << std::fixed << std::setprecision(6) << loss;
                
                if (step > 0) {
                    // Show trend
                    static float prev_loss = loss;
                    if (loss < prev_loss) {
                        std::cout << " ⬇️ (decreasing)";
                    } else if (loss > prev_loss) {
                        std::cout << " ⬆️ (increasing)";  
                    } else {
                        std::cout << " ➡️ (stable)";
                    }
                    prev_loss = loss;
                }
                std::cout << std::endl;
                
            } else {
                std::cout << "Step " << (step + 1) << ": Training failed ❌" << std::endl;
            }
        }
        
        std::cout << "\n🎯 Learning Analysis:" << std::endl;
        std::cout << "Initial Loss: " << std::fixed << std::setprecision(6) << initial_loss << std::endl;
        std::cout << "Final Loss:   " << std::fixed << std::setprecision(6) << final_loss << std::endl;
        
        float improvement = initial_loss - final_loss;
        float improvement_pct = (improvement / initial_loss) * 100.0f;
        
        std::cout << "Improvement:  " << std::fixed << std::setprecision(6) << improvement << " (" << std::setprecision(2) << improvement_pct << "%)" << std::endl;
        
        if (improvement > 0.01f) {
            std::cout << "✅ SUCCESS: Model is learning! Loss decreased significantly." << std::endl;
        } else if (improvement > 0.001f) {
            std::cout << "⚠️  MODEST: Some learning detected, but slow progress." << std::endl;
        } else {
            std::cout << "❌ CONCERN: No significant learning detected." << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 