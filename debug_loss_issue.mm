#include "src/host/transformer_model.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== Debugging Loss Calculation Issue ===" << std::endl;
    
    // Create a simple test config
    TransformerConfig config;
    config.vocab_size = 50;
    config.embedding_dim = 32;
    config.num_layers = 1;
    config.num_heads = 2;
    config.ffn_hidden_dim = 128;
    config.max_sequence_length = 8;
    config.batch_size = 1;
    config.learning_rate = 1e-4f; // Much lower learning rate
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
    
    // Test with the exact same sequence multiple times
    std::vector<uint32_t> input_tokens = {1, 2, 3, 4, 5};
    std::vector<uint32_t> target_tokens = {2, 3, 4, 5, 6};
    
    std::cout << "\n=== Testing Loss Consistency ===" << std::endl;
    
    // Test 1: Multiple evaluate calls (should be identical)
    std::cout << "\n--- Multiple evaluate calls (should be identical) ---" << std::endl;
    for (int i = 0; i < 3; i++) {
        float eval_loss = 0.0f;
        if (!model.evaluate(input_tokens, target_tokens, eval_loss)) {
            std::cerr << "Evaluate failed" << std::endl;
            return 1;
        }
        std::cout << "Evaluate " << (i+1) << ": " << std::fixed << std::setprecision(6) << eval_loss << std::endl;
    }
    
    // Test 2: Compare trainStep vs evaluate
    std::cout << "\n--- Compare trainStep vs evaluate ---" << std::endl;
    float eval_before = 0.0f;
    model.evaluate(input_tokens, target_tokens, eval_before);
    std::cout << "Evaluate before training: " << eval_before << std::endl;
    
    float train_loss = 0.0f;
    model.trainStep(input_tokens, target_tokens, train_loss);
    std::cout << "TrainStep loss: " << train_loss << std::endl;
    
    float eval_after = 0.0f;
    model.evaluate(input_tokens, target_tokens, eval_after);
    std::cout << "Evaluate after training: " << eval_after << std::endl;
    
    std::cout << "Difference (eval_before - train_loss): " << (eval_before - train_loss) << std::endl;
    std::cout << "Loss change (eval_after - eval_before): " << (eval_after - eval_before) << std::endl;
    
    // Test 3: Multiple sequences to see loss variation
    std::cout << "\n--- Multiple sequences loss variation ---" << std::endl;
    std::vector<std::vector<uint32_t>> test_inputs = {
        {1, 2, 3, 4, 5},
        {7, 8, 9, 10, 11},
        {13, 14, 15, 16, 17},
        {19, 20, 21, 22, 23}
    };
    
    std::vector<std::vector<uint32_t>> test_targets = {
        {2, 3, 4, 5, 6},
        {8, 9, 10, 11, 12},
        {14, 15, 16, 17, 18},
        {20, 21, 22, 23, 24}
    };
    
    for (size_t i = 0; i < test_inputs.size(); i++) {
        float seq_loss = 0.0f;
        model.evaluate(test_inputs[i], test_targets[i], seq_loss);
        std::cout << "Sequence " << (i+1) << " loss: " << seq_loss << std::endl;
    }
    
    // Test 4: Check if gradients are the problem
    std::cout << "\n--- Training multiple steps on same sequence ---" << std::endl;
    for (int step = 0; step < 5; step++) {
        float step_loss = 0.0f;
        model.trainStep(input_tokens, target_tokens, step_loss);
        std::cout << "Step " << (step+1) << " loss: " << step_loss << std::endl;
    }
    
    std::cout << "\n=== Debug Complete ===" << std::endl;
    return 0;
} 