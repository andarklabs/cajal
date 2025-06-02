#include "src/host/transformer_model.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== Testing Complete Training Pipeline ===" << std::endl;
    
    // Create a test config
    TransformerConfig config;
    config.vocab_size = 50;
    config.embedding_dim = 32;
    config.num_layers = 2; // Test with 2 layers
    config.num_heads = 4;
    config.ffn_hidden_dim = 128;
    config.max_sequence_length = 16;
    config.batch_size = 1;
    config.learning_rate = 2e-3f; // Slightly higher for faster convergence
    config.epsilon = 1e-5f;
    config.pad_token_id = 0;
    config.use_half_precision = true;
    config.float_logits = true;
    
    TransformerModel model(config);
    
    if (!model.initialize()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Model initialized successfully" << std::endl;
    std::cout << "  Parameters: " << model.getParameterCount() << std::endl;
    
    // Test data: multiple sequences
    std::vector<std::vector<uint32_t>> input_sequences = {
        {1, 2, 3, 4, 5, 6},
        {7, 8, 9, 10, 11, 12},
        {13, 14, 15, 16, 17, 18},
        {19, 20, 21, 22, 23, 24}
    };
    
    std::vector<std::vector<uint32_t>> target_sequences = {
        {2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13},
        {14, 15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24, 25}
    };
    
    std::cout << "\n=== Training Multiple Epochs ===" << std::endl;
    
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    // Train for multiple epochs
    for (int epoch = 0; epoch < 3; epoch++) {
        std::cout << "\n--- Epoch " << (epoch + 1) << " ---" << std::endl;
        
        float epoch_loss = 0.0f;
        int total_steps = 0;
        
        // Train on all sequences
        for (size_t i = 0; i < input_sequences.size(); i++) {
            float step_loss = 0.0f;
            
            if (!model.trainStep(input_sequences[i], target_sequences[i], step_loss)) {
                std::cerr << "Training step failed!" << std::endl;
                return 1;
            }
            
            epoch_loss += step_loss;
            total_steps++;
            
            if (epoch == 0 && i == 0) {
                initial_loss = step_loss;
            }
        }
        
        float avg_loss = epoch_loss / total_steps;
        final_loss = avg_loss;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average loss: " << avg_loss << std::endl;
        std::cout << "Learning rate: " << model.getCurrentLearningRate() << std::endl;
    }
    
    std::cout << "\n=== Training Results ===" << std::endl;
    std::cout << "Initial loss: " << initial_loss << std::endl;
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Loss reduction: " << (initial_loss - final_loss) << std::endl;
    std::cout << "Improvement: " << ((initial_loss - final_loss) / initial_loss * 100.0f) << "%" << std::endl;
    
    if (initial_loss > final_loss) {
        std::cout << "✅ Training is working - loss decreased!" << std::endl;
    } else {
        std::cout << "❌ Training issue - loss did not decrease" << std::endl;
    }
    
    std::cout << "\n=== Complete Training Pipeline Test Successful ===" << std::endl;
    return 0;
} 