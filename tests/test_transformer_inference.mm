#include <iostream>
#include <vector>
#include <chrono>
#include "../src/host/transformer_model.h"

void printTokens(const std::vector<uint32_t>& tokens, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

bool testTransformerInference() {
    std::cout << "=== Testing Transformer Inference ===\n" << std::endl;
    
    // Create a small model configuration for testing
    TransformerConfig config;
    config.vocab_size = 100;        // Small vocab for testing
    config.embedding_dim = 128;     // Small embedding dimension
    config.num_layers = 2;          // Few layers
    config.num_heads = 4;           // 4 attention heads
    config.ffn_hidden_dim = 256;    // 2x embedding_dim
    config.max_sequence_length = 64; // Short sequences
    config.batch_size = 1;          // Single batch for inference
    
    std::cout << "Model Configuration:" << std::endl;
    std::cout << "  Vocab Size: " << config.vocab_size << std::endl;
    std::cout << "  Embedding Dim: " << config.embedding_dim << std::endl;
    std::cout << "  Layers: " << config.num_layers << std::endl;
    std::cout << "  Heads: " << config.num_heads << std::endl;
    std::cout << "  Max Seq Length: " << config.max_sequence_length << std::endl;
    std::cout << std::endl;
    
    // Create and initialize model
    TransformerModel model(config);
    if (!model.initialize()) {
        std::cerr << "Failed to initialize transformer model" << std::endl;
        return false;
    }
    
    std::cout << "✓ Model initialized successfully" << std::endl;
    std::cout << "  Parameters: " << model.getParameterCount() << std::endl;
    std::cout << "  Memory Usage: " << (model.getMemoryUsage() / 1024 / 1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Test 1: Simple inference with a small prompt
    std::cout << "Test 1: Basic Inference" << std::endl;
    std::vector<uint32_t> prompt = {1, 5, 10, 3};  // Simple prompt tokens
    std::vector<uint32_t> generated_sequence;
    uint32_t max_new_tokens = 10;
    
    printTokens(prompt, "Prompt");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = model.generate(prompt, max_new_tokens, generated_sequence);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!success) {
        std::cerr << "✗ Generation failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Generation successful" << std::endl;
    printTokens(generated_sequence, "Generated Sequence");
    std::cout << "Generation time: " << duration.count() << " ms" << std::endl;
    std::cout << "Tokens per second: " << (max_new_tokens * 1000.0 / duration.count()) << std::endl;
    
    // Verify the generated sequence
    if (generated_sequence.size() != prompt.size() + max_new_tokens) {
        std::cerr << "✗ Generated sequence has unexpected length: " 
                  << generated_sequence.size() << " (expected " 
                  << (prompt.size() + max_new_tokens) << ")" << std::endl;
        return false;
    }
    
    // Check that prompt is preserved
    for (size_t i = 0; i < prompt.size(); i++) {
        if (generated_sequence[i] != prompt[i]) {
            std::cerr << "✗ Prompt not preserved at position " << i << std::endl;
            return false;
        }
    }
    
    // Check that new tokens are within vocab range
    for (size_t i = prompt.size(); i < generated_sequence.size(); i++) {
        if (generated_sequence[i] >= config.vocab_size) {
            std::cerr << "✗ Generated token " << generated_sequence[i] 
                      << " exceeds vocab size " << config.vocab_size << std::endl;
            return false;
        }
    }
    
    std::cout << "✓ Generated sequence validation passed" << std::endl;
    std::cout << std::endl;
    
    // Test 2: Different prompt length
    std::cout << "Test 2: Longer Prompt" << std::endl;
    std::vector<uint32_t> long_prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint32_t> long_generated;
    uint32_t long_max_tokens = 5;
    
    printTokens(long_prompt, "Long Prompt");
    
    success = model.generate(long_prompt, long_max_tokens, long_generated);
    
    if (!success) {
        std::cerr << "✗ Long prompt generation failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Long prompt generation successful" << std::endl;
    printTokens(long_generated, "Generated from Long Prompt");
    
    if (long_generated.size() != long_prompt.size() + long_max_tokens) {
        std::cerr << "✗ Long generated sequence has unexpected length" << std::endl;
        return false;
    }
    
    std::cout << "✓ Long prompt test passed" << std::endl;
    std::cout << std::endl;
    
    // Test 3: Edge case - Single token prompt
    std::cout << "Test 3: Single Token Prompt" << std::endl;
    std::vector<uint32_t> single_prompt = {42};
    std::vector<uint32_t> single_generated;
    uint32_t single_max_tokens = 3;
    
    printTokens(single_prompt, "Single Token Prompt");
    
    success = model.generate(single_prompt, single_max_tokens, single_generated);
    
    if (!success) {
        std::cerr << "✗ Single token generation failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Single token generation successful" << std::endl;
    printTokens(single_generated, "Generated from Single Token");
    
    if (single_generated.size() != single_prompt.size() + single_max_tokens) {
        std::cerr << "✗ Single token generated sequence has unexpected length" << std::endl;
        return false;
    }
    
    std::cout << "✓ Single token test passed" << std::endl;
    std::cout << std::endl;
    
    // Test 4: Error handling - Empty prompt
    std::cout << "Test 4: Error Handling - Empty Prompt" << std::endl;
    std::vector<uint32_t> empty_prompt;
    std::vector<uint32_t> empty_generated;
    
    success = model.generate(empty_prompt, 5, empty_generated);
    
    if (success) {
        std::cerr << "✗ Empty prompt should have failed but succeeded" << std::endl;
        return false;
    }
    
    std::cout << "✓ Empty prompt correctly rejected" << std::endl;
    std::cout << std::endl;
    
    // Test 5: Error handling - Prompt too long
    std::cout << "Test 5: Error Handling - Prompt Too Long" << std::endl;
    std::vector<uint32_t> too_long_prompt(config.max_sequence_length + 1, 1);
    std::vector<uint32_t> too_long_generated;
    
    success = model.generate(too_long_prompt, 5, too_long_generated);
    
    if (success) {
        std::cerr << "✗ Too long prompt should have failed but succeeded" << std::endl;
        return false;
    }
    
    std::cout << "✓ Too long prompt correctly rejected" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== All Inference Tests Passed! ===\n" << std::endl;
    return true;
}

int main() {
    std::cout << "Starting Transformer Inference Tests...\n" << std::endl;
    
    bool success = testTransformerInference();
    
    if (success) {
        std::cout << "🎉 All tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "❌ Some tests failed!" << std::endl;
        return 1;
    }
} 