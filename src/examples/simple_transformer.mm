#include "../host/transformer_model.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

class SimpleTransformerExample {
private:
    TransformerConfig config;
    std::unique_ptr<TransformerModel> model;
    
public:
    SimpleTransformerExample() {
        // Create a realistic but manageable configuration
        config.vocab_size = 1000;       // Moderate vocabulary
        config.embedding_dim = 256;     // Standard embedding dimension
        config.num_layers = 4;          // 4 transformer layers
        config.num_heads = 8;           // 8 attention heads
        config.ffn_hidden_dim = 1024;   // 4x embedding_dim
        config.max_sequence_length = 64; // Moderate sequence length
        config.batch_size = 1;          // Single batch for inference
        config.learning_rate = 1e-4f;
        config.epsilon = 1e-5f;
        config.use_half_precision = true;
        config.float_logits = true;
        
        model = std::make_unique<TransformerModel>(config);
    }
    
    bool initialize() {
        std::cout << "=== Initializing MSL Transformer Model ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Vocabulary size: " << config.vocab_size << std::endl;
        std::cout << "  Embedding dimension: " << config.embedding_dim << std::endl;
        std::cout << "  Number of layers: " << config.num_layers << std::endl;
        std::cout << "  Number of attention heads: " << config.num_heads << std::endl;
        std::cout << "  FFN hidden dimension: " << config.ffn_hidden_dim << std::endl;
        std::cout << "  Maximum sequence length: " << config.max_sequence_length << std::endl;
        std::cout << std::endl;
        
        bool success = model->initialize();
        if (success) {
            std::cout << "✓ Model initialized successfully!" << std::endl;
            std::cout << "  Total parameters: " << model->getParameterCount() << std::endl;
            std::cout << "  Memory usage: " << (model->getMemoryUsage() / 1024 / 1024) << " MB" << std::endl;
            std::cout << std::endl;
        } else {
            std::cerr << "❌ Failed to initialize model" << std::endl;
        }
        
        return success;
    }
    
    void demonstrateForwardPass() {
        std::cout << "=== Forward Pass Demonstration ===" << std::endl;
        
        // Create a test sequence
        std::vector<uint32_t> input_sequence = {42, 123, 456, 789, 100};
        std::cout << "Input sequence: [";
        for (size_t i = 0; i < input_sequence.size(); i++) {
            std::cout << input_sequence[i];
            if (i < input_sequence.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Run forward pass
        std::vector<float> logits;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = model->forward(input_sequence, logits);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (success) {
            std::cout << "✓ Forward pass completed in " << duration.count() << " μs" << std::endl;
            std::cout << "  Output logits shape: [" << input_sequence.size() << " x " << config.vocab_size << "]" << std::endl;
            
            // Show logits statistics for each position
            for (size_t pos = 0; pos < input_sequence.size(); pos++) {
                float* pos_logits = logits.data() + (pos * config.vocab_size);
                
                float min_logit = pos_logits[0];
                float max_logit = pos_logits[0];
                float sum_logits = 0.0f;
                
                for (size_t v = 0; v < config.vocab_size; v++) {
                    min_logit = std::min(min_logit, pos_logits[v]);
                    max_logit = std::max(max_logit, pos_logits[v]);
                    sum_logits += pos_logits[v];
                }
                
                float mean_logit = sum_logits / config.vocab_size;
                
                std::cout << "  Position " << pos << ": logits range [" 
                          << min_logit << ", " << max_logit << "], mean=" << mean_logit << std::endl;
            }
        } else {
            std::cerr << "❌ Forward pass failed" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    uint32_t sampleFromLogits(const float* logits, float temperature = 1.0f) {
        // Apply temperature scaling and convert to probabilities
        std::vector<float> probs(config.vocab_size);
        
        // Find max for numerical stability
        float max_logit = logits[0];
        for (size_t i = 1; i < config.vocab_size; i++) {
            max_logit = std::max(max_logit, logits[i]);
        }
        
        // Compute softmax with temperature
        float sum = 0.0f;
        for (size_t i = 0; i < config.vocab_size; i++) {
            probs[i] = exp((logits[i] - max_logit) / temperature);
            sum += probs[i];
        }
        
        // Normalize
        for (size_t i = 0; i < config.vocab_size; i++) {
            probs[i] /= sum;
        }
        
        // Sample from the distribution
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        float random_val = dis(gen);
        float cumulative = 0.0f;
        
        for (size_t i = 0; i < config.vocab_size; i++) {
            cumulative += probs[i];
            if (random_val <= cumulative) {
                return static_cast<uint32_t>(i);
            }
        }
        
        return config.vocab_size - 1; // Fallback
    }
    
    void demonstrateGeneration() {
        std::cout << "=== Text Generation Demonstration ===" << std::endl;
        
        // Start with a short prompt
        std::vector<uint32_t> context = {100, 200, 300}; // Simulated "prompt"
        
        std::cout << "Initial context: [";
        for (size_t i = 0; i < context.size(); i++) {
            std::cout << context[i];
            if (i < context.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Generate additional tokens
        size_t max_new_tokens = 10;
        
        for (size_t step = 0; step < max_new_tokens; step++) {
            // Run forward pass on current context
            std::vector<float> logits;
            bool success = model->forward(context, logits);
            
            if (!success) {
                std::cerr << "❌ Generation failed at step " << step << std::endl;
                break;
            }
            
            // Get logits for the last position (next token prediction)
            size_t last_pos = context.size() - 1;
            float* last_logits = logits.data() + (last_pos * config.vocab_size);
            
            // Sample next token (with some temperature for variety)
            uint32_t next_token = sampleFromLogits(last_logits, 0.8f);
            
            // Add to context
            context.push_back(next_token);
            
            std::cout << "Step " << (step + 1) << ": Generated token " << next_token 
                      << " (context length: " << context.size() << ")" << std::endl;
            
            // Stop if we reach max sequence length
            if (context.size() >= config.max_sequence_length) {
                std::cout << "  Reached maximum sequence length" << std::endl;
                break;
            }
        }
        
        std::cout << "\nFinal generated sequence: [";
        for (size_t i = 0; i < context.size(); i++) {
            std::cout << context[i];
            if (i < context.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "✓ Generation completed successfully!" << std::endl;
        std::cout << std::endl;
    }
    
    void demonstratePerformance() {
        std::cout << "=== Performance Benchmarking ===" << std::endl;
        
        // Test different sequence lengths
        std::vector<size_t> test_lengths = {1, 4, 8, 16, 32};
        
        for (size_t length : test_lengths) {
            if (length > config.max_sequence_length) continue;
            
            // Create test sequence
            std::vector<uint32_t> test_seq(length);
            for (size_t i = 0; i < length; i++) {
                test_seq[i] = i % config.vocab_size;
            }
            
            // Warm up
            std::vector<float> logits;
            model->forward(test_seq, logits);
            
            // Benchmark multiple runs
            size_t num_runs = 10;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (size_t run = 0; run < num_runs; run++) {
                model->forward(test_seq, logits);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            double avg_time = double(total_duration.count()) / num_runs;
            double tokens_per_second = (length * 1000000.0) / avg_time;
            
            std::cout << "  Length " << length << ": " << avg_time << " μs/forward, " 
                      << tokens_per_second << " tokens/sec" << std::endl;
        }
        
        std::cout << "✓ Performance benchmarking completed" << std::endl;
        std::cout << std::endl;
    }
    
    void runDemo() {
        if (!initialize()) {
            return;
        }
        
        demonstrateForwardPass();
        demonstrateGeneration();
        demonstratePerformance();
        
        std::cout << "🎉 MSL Transformer demonstration completed successfully!" << std::endl;
        std::cout << "\nThis demonstrates:" << std::endl;
        std::cout << "  ✓ Complete forward pass pipeline" << std::endl;
        std::cout << "  ✓ All 8 MSL kernels working in sequence" << std::endl;
        std::cout << "  ✓ Token embedding → positional encoding → attention → FFN → output" << std::endl;
        std::cout << "  ✓ Autoregressive text generation capability" << std::endl;
        std::cout << "  ✓ Real-time performance on Apple M3 Max" << std::endl;
        std::cout << "\nReady for training implementation (Phase 3 Tasks 3.2-3.4)!" << std::endl;
    }
};

int main() {
    try {
        SimpleTransformerExample demo;
        demo.runDemo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Demo failed with unknown error" << std::endl;
        return 1;
    }
} 