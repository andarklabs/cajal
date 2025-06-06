//
// Quick Training Test - Train for 1 batch and save
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🧪 Quick Training Test + Save" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // Create model
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 2;  // Small batch
        config.learning_rate = 0.01f;
        
        std::cout << "🔧 Initializing model..." << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized" << std::endl;
        
        // Create simple training data
        std::vector<uint32_t> input1 = {100, 200, 300, 400};
        std::vector<uint32_t> target1 = {200, 300, 400, 500};
        std::vector<uint32_t> input2 = {150, 250, 350, 450};
        std::vector<uint32_t> target2 = {250, 350, 450, 550};
        
        std::cout << "🎓 Training 1 batch..." << std::endl;
        
        // Train single step
        float loss;
        bool train_result = model.trainBatch({input1, input2}, {target1, target2}, loss);
        
        if (train_result) {
            std::cout << "✅ Training successful, loss: " << loss << std::endl;
        } else {
            std::cout << "❌ Training failed" << std::endl;
            return 1;
        }
        
        // Save the trained model
        std::string model_path = "models/quick_trained_model.bin";
        std::cout << "💾 Saving trained model to: " << model_path << std::endl;
        
        if (!model.saveWeights(model_path)) {
            std::cerr << "❌ Failed to save model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model saved successfully!" << std::endl;
        
        // Test that we can load it
        TransformerModel model2(config);
        if (!model2.initialize()) {
            std::cerr << "❌ Failed to initialize second model" << std::endl;
            return 1;
        }
        
        if (!model2.loadWeights(model_path)) {
            std::cerr << "❌ Failed to load saved model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "🎉 Quick training + save/load complete!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 