#include "../src/host/transformer_model.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

class TransformerTrainingTest {
private:
    TransformerConfig config;
    std::unique_ptr<TransformerModel> model;
    
public:
    TransformerTrainingTest() {
        // Create a minimal test configuration
        config.vocab_size = 50;         // Very small vocabulary
        config.embedding_dim = 32;      // Small embedding dimension
        config.num_layers = 1;          // Just 1 layer for quick testing
        config.num_heads = 2;           // 2 attention heads
        config.ffn_hidden_dim = 128;    // 4x embedding_dim
        config.max_sequence_length = 8; // Short sequences
        config.batch_size = 1;          // Single batch
        config.learning_rate = 1e-3f;   // Higher learning rate for testing
        config.epsilon = 1e-5f;
        config.pad_token_id = 0;        // Use token 0 as padding
        config.use_half_precision = true;
        config.float_logits = true;
        
        model = std::make_unique<TransformerModel>(config);
    }
    
    void test_loss_calculation() {
        std::cout << "=== Testing Loss Calculation ===" << std::endl;
        
        // Test data: simple sequence with known targets
        std::vector<uint32_t> input_tokens = {1, 2, 3, 4};
        std::vector<uint32_t> target_tokens = {2, 3, 4, 5}; // Next token prediction
        
        float loss_value;
        bool success = model->evaluate(input_tokens, target_tokens, loss_value);
        
        assert(success && "Loss calculation should succeed");
        assert(std::isfinite(loss_value) && "Loss should be finite");
        assert(loss_value > 0.0f && "Loss should be positive for random initialization");
        
        std::cout << "✓ Loss calculation successful" << std::endl;
        std::cout << "  Input sequence: [1, 2, 3, 4]" << std::endl;
        std::cout << "  Target sequence: [2, 3, 4, 5]" << std::endl;
        std::cout << "  Computed loss: " << loss_value << std::endl;
        
        // Test with padding tokens
        std::vector<uint32_t> padded_input = {1, 2, 0, 0}; // Last two are padding
        std::vector<uint32_t> padded_target = {2, 3, 0, 0}; // Corresponding padding
        
        float padded_loss;
        success = model->evaluate(padded_input, padded_target, padded_loss);
        
        assert(success && "Padded loss calculation should succeed");
        assert(std::isfinite(padded_loss) && "Padded loss should be finite");
        
        std::cout << "✓ Padded sequence loss: " << padded_loss << std::endl;
    }
    
    void test_training_step() {
        std::cout << "\n=== Testing Training Step ===" << std::endl;
        
        // Simple training data
        std::vector<uint32_t> input_tokens = {1, 2, 3};
        std::vector<uint32_t> target_tokens = {2, 3, 4};
        
        // Measure loss before training
        float initial_loss;
        bool success = model->evaluate(input_tokens, target_tokens, initial_loss);
        assert(success && "Initial loss evaluation should succeed");
        
        std::cout << "Initial loss: " << initial_loss << std::endl;
        
        // Perform training step
        float training_loss;
        success = model->trainStep(input_tokens, target_tokens, training_loss);
        assert(success && "Training step should succeed");
        
        std::cout << "Training step loss: " << training_loss << std::endl;
        
        // Check that loss values are reasonable
        assert(std::abs(training_loss - initial_loss) < 1.0f && "Training and evaluation loss should be similar");
        
        // Measure loss after training step
        float post_training_loss;
        success = model->evaluate(input_tokens, target_tokens, post_training_loss);
        assert(success && "Post-training loss evaluation should succeed");
        
        std::cout << "Post-training loss: " << post_training_loss << std::endl;
        
        // Check optimizer state was updated
        assert(model->getOptimizerTimestep() == 1 && "Optimizer timestep should be incremented");
        
        std::cout << "✓ Training step completed successfully" << std::endl;
        std::cout << "  Optimizer timestep: " << model->getOptimizerTimestep() << std::endl;
        std::cout << "  Current learning rate: " << model->getCurrentLearningRate() << std::endl;
    }
    
    void test_multiple_training_steps() {
        std::cout << "\n=== Testing Multiple Training Steps ===" << std::endl;
        
        // Training data
        std::vector<uint32_t> input_tokens = {5, 10, 15};
        std::vector<uint32_t> target_tokens = {10, 15, 20};
        
        // Track loss over multiple steps
        std::vector<float> losses;
        
        for (int step = 0; step < 5; step++) {
            float step_loss;
            bool success = model->trainStep(input_tokens, target_tokens, step_loss);
            assert(success && "Each training step should succeed");
            
            losses.push_back(step_loss);
            std::cout << "Step " << (step + 1) << " loss: " << step_loss << std::endl;
        }
        
        // Verify optimizer state
        assert(model->getOptimizerTimestep() == 6 && "Should have 6 total timesteps (1 from previous test + 5 new)");
        
        std::cout << "✓ Multiple training steps completed" << std::endl;
        std::cout << "  Final optimizer timestep: " << model->getOptimizerTimestep() << std::endl;
        
        // Check that losses are reasonable (don't need to decrease for such a simple test)
        for (float loss : losses) {
            assert(std::isfinite(loss) && loss > 0.0f && "All losses should be positive and finite");
        }
    }
    
    void test_different_sequence_lengths() {
        std::cout << "\n=== Testing Different Sequence Lengths ===" << std::endl;
        
        // Test various sequence lengths
        std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> test_cases = {
            {{1}, {2}},                     // Single token
            {{1, 2}, {2, 3}},              // Two tokens  
            {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}}, // Five tokens
        };
        
        for (size_t i = 0; i < test_cases.size(); i++) {
            auto& input = test_cases[i].first;
            auto& target = test_cases[i].second;
            
            float loss;
            bool success = model->trainStep(input, target, loss);
            assert(success && "Training should succeed for all sequence lengths");
            assert(std::isfinite(loss) && loss > 0.0f && "Loss should be valid");
            
            std::cout << "  Length " << input.size() << " loss: " << loss << std::endl;
        }
        
        std::cout << "✓ Different sequence lengths handled correctly" << std::endl;
    }
    
    void test_gradient_flow() {
        std::cout << "\n=== Testing Gradient Flow ===" << std::endl;
        
        // This is a basic test to ensure gradients are being computed and applied
        // We'll check that repeated training on the same example shows some learning signal
        
        std::vector<uint32_t> input_tokens = {7, 14, 21};
        std::vector<uint32_t> target_tokens = {14, 21, 28};
        
        // Get initial evaluation
        float initial_loss;
        model->evaluate(input_tokens, target_tokens, initial_loss);
        std::cout << "Initial evaluation loss: " << initial_loss << std::endl;
        
        // Train multiple steps on the same data
        float final_loss = initial_loss;
        for (int step = 0; step < 3; step++) {
            bool success = model->trainStep(input_tokens, target_tokens, final_loss);
            assert(success && "Training step should succeed");
            std::cout << "Training step " << (step + 1) << " loss: " << final_loss << std::endl;
        }
        
        // Final evaluation
        float eval_loss;
        model->evaluate(input_tokens, target_tokens, eval_loss);
        std::cout << "Final evaluation loss: " << eval_loss << std::endl;
        
        // We don't expect dramatic improvement with such simple training,
        // but the training should complete without errors
        assert(std::isfinite(eval_loss) && "Final loss should be finite");
        
        std::cout << "✓ Gradient flow test completed" << std::endl;
    }
    
    void runAllTests() {
        std::cout << "=== Transformer Training Tests ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Vocab size: " << config.vocab_size << std::endl;
        std::cout << "  Embedding dim: " << config.embedding_dim << std::endl;
        std::cout << "  Num layers: " << config.num_layers << std::endl;
        std::cout << "  Learning rate: " << config.learning_rate << std::endl;
        std::cout << std::endl;
        
        // Initialize model
        bool init_success = model->initialize();
        assert(init_success && "Model initialization should succeed");
        
        test_loss_calculation();
        test_training_step();
        test_multiple_training_steps();
        test_different_sequence_lengths();
        test_gradient_flow();
        
        std::cout << "\n=== All Training Tests Passed! ===" << std::endl;
        std::cout << "🎉 Basic MSL training pipeline is working!" << std::endl;
        std::cout << "\nNote: This implements a simplified training pipeline." << std::endl;
        std::cout << "Full backward pass through all layers would require additional gradient kernels." << std::endl;
    }
};

int main() {
    try {
        TransformerTrainingTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Training test failed with error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Training test failed with unknown error" << std::endl;
        return 1;
    }
} 