//
// Simple Test for Model Save/Load Functionality
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "../src/host/transformer_model.h"

int main() {
    std::cout << "🧪 Testing Model Save/Load Functionality" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Create model with same config as training
        TransformerConfig config;
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 1;
        config.learning_rate = 1e-4f;
        
        std::cout << "🔧 Creating and initializing model..." << std::endl;
        TransformerModel model(config);
        
        if (!model.initialize()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model initialized successfully" << std::endl;
        std::cout << "📊 Parameters: " << model.getParameterCount() << std::endl;
        std::cout << "💾 Memory: " << model.getMemoryUsage() << " MB" << std::endl;
        
        // Test save
        std::string test_path = "models/test_model.bin";
        std::cout << "\n💾 Testing save to: " << test_path << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool save_result = model.saveWeights(test_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (save_result) {
            std::cout << "✅ Model saved successfully in " << save_duration.count() << "ms" << std::endl;
        } else {
            std::cout << "❌ Model save failed" << std::endl;
            return 1;
        }
        
        // Check file exists and size
        std::ifstream check_file(test_path, std::ios::binary | std::ios::ate);
        if (check_file.is_open()) {
            auto file_size = check_file.tellg();
            check_file.close();
            std::cout << "📁 Saved file size: " << (file_size / 1024 / 1024) << " MB" << std::endl;
        }
        
        // Test load
        std::cout << "\n📂 Testing load from: " << test_path << std::endl;
        
        // Create new model with same config
        TransformerModel model2(config);
        if (!model2.initialize()) {
            std::cerr << "❌ Failed to initialize second model" << std::endl;
            return 1;
        }
        
        start_time = std::chrono::high_resolution_clock::now();
        bool load_result = model2.loadWeights(test_path);
        end_time = std::chrono::high_resolution_clock::now();
        
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (load_result) {
            std::cout << "✅ Model loaded successfully in " << load_duration.count() << "ms" << std::endl;
        } else {
            std::cout << "❌ Model load failed" << std::endl;
            return 1;
        }
        
        std::cout << "\n🎉 Save/Load functionality verified!" << std::endl;
        std::cout << "💡 Now you can train and save models successfully" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Test failed: " << e.what() << std::endl;
        return 1;
    }
} 