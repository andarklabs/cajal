#include "src/host/transformer_model.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== Testing Full Training Pipeline ===" << std::endl;
    
    // Create a small test config for faster debugging
    TransformerConfig config;
    config.vocab_size = 50;
    config.embedding_dim = 32;
    config.num_layers = 1;
    config.num_heads = 2;
    config.ffn_hidden_dim = 128;
    config.max_sequence_length = 8;
    config.batch_size = 1;
    config.learning_rate = 1e-3f;
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
    
    // Test data: simple sequence for language modeling
    std::vector<uint32_t> input_tokens = {1, 2, 3, 4, 5}; // Input sequence
    std::vector<uint32_t> target_tokens = {2, 3, 4, 5, 6}; // Target (shifted by 1)
    
    std::cout << "\n=== Testing Training Steps ===" << std::endl;
    
    // Test multiple training steps to verify gradient flow
    for (int step = 0; step < 5; step++) {
        std::cout << "\n--- Training Step " << (step + 1) << " ---" << std::endl;
        
        float loss_before = 0.0f;
        
        // 1. Compute initial loss (before training step)
        if (!model.evaluate(input_tokens, target_tokens, loss_before)) {
            std::cerr << "Failed to compute initial loss" << std::endl;
            return 1;
        }
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Loss before step: " << loss_before << std::endl;
        
        // 2. Perform training step
        float training_loss = 0.0f;
        if (!model.trainStep(input_tokens, target_tokens, training_loss)) {
            std::cerr << "Training step failed!" << std::endl;
            return 1;
        }
        std::cout << "Training loss: " << training_loss << std::endl;
        
        // 3. Compute loss after training step
        float loss_after = 0.0f;
        if (!model.evaluate(input_tokens, target_tokens, loss_after)) {
            std::cerr << "Failed to compute loss after training" << std::endl;
            return 1;
        }
        std::cout << "Loss after step: " << loss_after << std::endl;
        
        // 4. Verify loss decreased (or at least didn't increase too much)
        float loss_change = loss_after - loss_before;
        std::cout << "Loss change: " << loss_change;
        if (loss_change < 0) {
            std::cout << " ✓ (decreased)" << std::endl;
        } else if (loss_change < 0.1) {
            std::cout << " ~ (slight increase, possibly due to learning dynamics)" << std::endl;
        } else {
            std::cout << " ❌ (significant increase - potential issue)" << std::endl;
        }
    }
    
    std::cout << "\n=== Testing Forward Pass Changes ===" << std::endl;
    
    // Test that forward pass outputs change after training
    std::vector<float> logits_before, logits_after;
    
    if (!model.forward(input_tokens, logits_before)) {
        std::cerr << "Failed initial forward pass" << std::endl;
        return 1;
    }
    
    // Train for a few more steps
    for (int i = 0; i < 3; i++) {
        float dummy_loss;
        model.trainStep(input_tokens, target_tokens, dummy_loss);
    }
    
    if (!model.forward(input_tokens, logits_after)) {
        std::cerr << "Failed forward pass after training" << std::endl;
        return 1;
    }
    
    // Check if outputs changed
    bool outputs_changed = false;
    float max_change = 0.0f;
    for (size_t i = 0; i < std::min(logits_before.size(), logits_after.size()); i++) {
        float change = std::abs(logits_after[i] - logits_before[i]);
        if (change > 1e-6) {
            outputs_changed = true;
        }
        max_change = std::max(max_change, change);
    }
    
    std::cout << "Maximum logit change: " << max_change << std::endl;
    if (outputs_changed) {
        std::cout << "✅ Forward pass outputs changed after training (parameters updating)" << std::endl;
    } else {
        std::cout << "❌ Forward pass outputs unchanged (parameters not updating properly)" << std::endl;
    }
    
    std::cout << "\n=== Training Pipeline Test Complete ===" << std::endl;
    return 0;
} 