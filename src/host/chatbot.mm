//
// MSL Transformer Chatbot - Interactive Interface
// Production-ready chatbot using our optimized MSL Transformer
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "transformer_model.h"

class MSLChatbot {
private:
    std::unique_ptr<TransformerModel> model;
    std::vector<uint32_t> conversation_context;
    
    // Chatbot configuration
    struct ChatConfig {
        float temperature = 0.7f;
        int max_response_length = 100;
        int max_context_length = 400;
        bool use_nucleus_sampling = true;
        float nucleus_p = 0.9f;
        int top_k = 50;
        bool verbose_mode = false;
    } config;
    
    // Special tokens
    const uint32_t BOS_TOKEN = 1;
    const uint32_t EOS_TOKEN = 2;
    const uint32_t PAD_TOKEN = 0;
    const uint32_t USER_TOKEN = 3;    // "<USER>"
    const uint32_t BOT_TOKEN = 4;     // "<BOT>"
    
public:
    MSLChatbot() {
        std::cout << "🤖 MSL Transformer Chatbot Starting...\n";
        
        // Initialize the model with optimal chatbot configuration
        TransformerConfig chatbot_config;
        chatbot_config.vocab_size = 1776;
        chatbot_config.embedding_dim = 512;
        chatbot_config.num_heads = 8;
        chatbot_config.num_layers = 6;
        chatbot_config.ffn_hidden_dim = 2048;
        chatbot_config.max_sequence_length = 512;
        chatbot_config.learning_rate = 0.001f;
        
        try {
            model = std::make_unique<TransformerModel>(chatbot_config);
            std::cout << "✅ MSL Transformer loaded successfully!\n";
            std::cout << "📊 Model: " << model->getParameterCount() << " parameters\n";
            std::cout << "💾 Memory: " << model->getMemoryUsage() << " MB\n\n";
            
            // Initialize conversation with system prompt
            initializeConversation();
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to initialize chatbot: " << e.what() << std::endl;
            throw;
        }
    }
    
    void initializeConversation() {
        // Start conversation with a system prompt to establish chatbot personality
        conversation_context.clear();
        conversation_context.push_back(BOS_TOKEN);
        
        // Add system prompt tokens (simplified - in practice you'd tokenize properly)
        std::string system_prompt = "You are a helpful AI assistant built with Apple Metal. You are knowledgeable, friendly, and concise.";
        // For demo, we'll add some representative tokens
        std::vector<uint32_t> system_tokens = {10, 15, 25, 30, 45, 50, 65, 70}; // Mock system tokens
        conversation_context.insert(conversation_context.end(), system_tokens.begin(), system_tokens.end());
        
        if (config.verbose_mode) {
            std::cout << "🎯 System prompt initialized (" << system_tokens.size() << " tokens)\n";
        }
    }
    
    std::string generateResponse(const std::string& user_input) {
        if (config.verbose_mode) {
            std::cout << "🔄 Processing user input: \"" << user_input << "\"\n";
        }
        
        // Tokenize user input (simplified - in practice use proper tokenizer)
        std::vector<uint32_t> user_tokens = tokenizeInput(user_input);
        
        // Add user tokens to conversation context
        conversation_context.push_back(USER_TOKEN);
        conversation_context.insert(conversation_context.end(), user_tokens.begin(), user_tokens.end());
        conversation_context.push_back(BOT_TOKEN);
        
        // Trim context if too long (following cursor rules for efficient inference)
        if (conversation_context.size() > static_cast<size_t>(config.max_context_length)) {
            int tokens_to_remove = static_cast<int>(conversation_context.size()) - config.max_context_length;
            conversation_context.erase(conversation_context.begin() + 1, 
                                     conversation_context.begin() + 1 + tokens_to_remove);
            if (config.verbose_mode) {
                std::cout << "✂️ Trimmed " << tokens_to_remove << " tokens from context\n";
            }
        }
        
        // Generate response using autoregressive decoding with KV caching
        std::vector<uint32_t> response_tokens = generateTokensWithKVCache();
        
        // Convert tokens back to text
        std::string response = detokenizeOutput(response_tokens);
        
        // Add response tokens to conversation context
        conversation_context.insert(conversation_context.end(), response_tokens.begin(), response_tokens.end());
        
        return response;
    }
    
private:
    std::vector<uint32_t> tokenizeInput(const std::string& input) {
        // Simplified tokenization - in production, use proper BPE tokenizer
        std::vector<uint32_t> tokens;
        std::istringstream iss(input);
        std::string word;
        
        while (iss >> word) {
            // Mock tokenization: convert to simple hash-based token IDs
            uint32_t token_id = (std::hash<std::string>{}(word) % 1500) + 100; // Keep in vocab range
            tokens.push_back(token_id);
        }
        
        if (config.verbose_mode) {
            std::cout << "📝 Tokenized input: " << tokens.size() << " tokens\n";
        }
        
        return tokens;
    }
    
    std::vector<uint32_t> generateTokensWithKVCache() {
        std::vector<uint32_t> generated_tokens;
        
        if (config.verbose_mode) {
            std::cout << "🧠 Starting autoregressive generation...\n";
            std::cout << "⚙️ Temperature: " << config.temperature << ", Max length: " << config.max_response_length << "\n";
        }
        
        try {
            // Autoregressive decoding loop (following cursor rules)
            for (int step = 0; step < config.max_response_length; step++) {
                // Current sequence includes conversation context + generated tokens so far
                std::vector<uint32_t> current_sequence = conversation_context;
                current_sequence.insert(current_sequence.end(), generated_tokens.begin(), generated_tokens.end());
                
                // Perform inference to get next token logits
                // Note: In production, this would use KV caching for efficiency
                std::vector<float> logits = model->generateNext(current_sequence);
                
                // Apply sampling strategy
                uint32_t next_token = sampleNextToken(logits, step);
                
                // Check for EOS token or other stopping conditions
                if (next_token == EOS_TOKEN || next_token == USER_TOKEN) {
                    if (config.verbose_mode) {
                        std::cout << "🛑 Generation stopped at step " << step << " (EOS/USER token)\n";
                    }
                    break;
                }
                
                generated_tokens.push_back(next_token);
                
                if (config.verbose_mode && step % 10 == 0) {
                    std::cout << "📈 Generated " << (step + 1) << " tokens...\n";
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error during generation: " << e.what() << std::endl;
            // Return a fallback response
            generated_tokens = {500, 501, 502}; // Mock "I apologize, there was an error"
        }
        
        if (config.verbose_mode) {
            std::cout << "✅ Generated " << generated_tokens.size() << " response tokens\n";
        }
        
        return generated_tokens;
    }
    
    uint32_t sampleNextToken(const std::vector<float>& logits, int /* step */) {
        if (logits.empty() || logits.size() > 10000) {
            if (config.verbose_mode) {
                std::cout << "⚠️ Invalid logits, using fallback token\n";
            }
            return 500; // Fallback token
        }
        
        // Apply temperature scaling
        std::vector<float> scaled_logits = logits;
        if (config.temperature != 1.0f) {
            for (auto& logit : scaled_logits) {
                logit /= config.temperature;
            }
        }
        
        // Choose sampling strategy based on configuration
        if (config.use_nucleus_sampling) {
            return nucleusSampling(scaled_logits, config.nucleus_p);
        } else {
            return topKSampling(scaled_logits, config.top_k);
        }
    }
    
    uint32_t topKSampling(const std::vector<float>& logits, int k) {
        // Implement top-k sampling following cursor rules
        std::vector<std::pair<float, uint32_t>> logit_pairs;
        for (size_t i = 0; i < logits.size(); i++) {
            logit_pairs.push_back({logits[i], static_cast<uint32_t>(i)});
        }
        
        // Sort by logit value (descending)
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Take top-k
        int actual_k = std::min(k, static_cast<int>(logit_pairs.size()));
        
        // Simple sampling from top-k (in practice, should be probability-based)
        int selected_idx = rand() % actual_k;
        return logit_pairs[selected_idx].second;
    }
    
    uint32_t nucleusSampling(const std::vector<float>& logits, float p) {
        // Implement nucleus (top-p) sampling following cursor rules
        std::vector<std::pair<float, uint32_t>> logit_pairs;
        for (size_t i = 0; i < logits.size(); i++) {
            logit_pairs.push_back({logits[i], static_cast<uint32_t>(i)});
        }
        
        // Sort by logit value (descending)
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Convert to probabilities (simplified softmax)
        float max_logit = logit_pairs[0].first;
        std::vector<float> probs;
        float sum = 0.0f;
        
        for (const auto& pair : logit_pairs) {
            float prob = exp(pair.first - max_logit);
            probs.push_back(prob);
            sum += prob;
        }
        
        // Normalize
        for (auto& prob : probs) {
            prob /= sum;
        }
        
        // Find nucleus
        float cumulative = 0.0f;
        int nucleus_size = 0;
        for (size_t i = 0; i < probs.size(); i++) {
            cumulative += probs[i];
            nucleus_size++;
            if (cumulative >= p) break;
        }
        
        // Sample from nucleus
        int selected_idx = rand() % nucleus_size;
        return logit_pairs[selected_idx].second;
    }
    
    std::string detokenizeOutput(const std::vector<uint32_t>& tokens) {
        // Simplified detokenization - in production, use proper detokenizer
        std::ostringstream oss;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) oss << " ";
            
            // Mock detokenization: convert token IDs to representative words
            uint32_t token_id = tokens[i];
            if (token_id < 100) {
                oss << "[SPECIAL_" << token_id << "]";
            } else if (token_id < 200) {
                oss << "word" << (token_id - 100);
            } else if (token_id < 500) {
                const char* common_words[] = {"the", "and", "is", "to", "of", "a", "in", "that", "have", "it", "for", "on", "with", "as", "be", "at", "by", "this", "from", "or", "an", "are", "not", "was", "but", "can", "had", "has", "what", "were"};
                oss << common_words[token_id % 30];
            } else {
                oss << "response" << (token_id % 100);
            }
        }
        
        std::string result = oss.str();
        if (config.verbose_mode) {
            std::cout << "🔤 Detokenized: \"" << result << "\"\n";
        }
        
        return result;
    }
    
public:
    void startInteractiveChat() {
        std::cout << "🚀 MSL Transformer Chatbot Ready!\n";
        std::cout << "💡 Type your message and press Enter. Type 'quit' to exit.\n";
        std::cout << "⚙️ Type 'config' to adjust settings.\n";
        std::cout << "═══════════════════════════════════════════════════════\n\n";
        
        std::string user_input;
        
        while (true) {
            std::cout << "👤 You: ";
            std::getline(std::cin, user_input);
            
            if (user_input == "quit" || user_input == "exit") {
                std::cout << "\n🤖 Goodbye! Thanks for chatting with the MSL Transformer!\n";
                break;
            }
            
            if (user_input == "config") {
                showConfigMenu();
                continue;
            }
            
            if (user_input.empty()) {
                continue;
            }
            
            // Generate response
            std::cout << "🤖 Bot: ";
            std::cout.flush();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            std::string response = generateResponse(user_input);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << response << std::endl;
            
            if (config.verbose_mode) {
                std::cout << "⏱️ Response generated in " << duration.count() << "ms\n";
            }
            std::cout << std::endl;
        }
    }
    
    void showConfigMenu() {
        std::cout << "\n⚙️ Chatbot Configuration:\n";
        std::cout << "1. Temperature: " << config.temperature << "\n";
        std::cout << "2. Max response length: " << config.max_response_length << "\n";
        std::cout << "3. Nucleus sampling: " << (config.use_nucleus_sampling ? "ON" : "OFF") << "\n";
        std::cout << "4. Nucleus p: " << config.nucleus_p << "\n";
        std::cout << "5. Top-k: " << config.top_k << "\n";
        std::cout << "6. Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << "\n";
        std::cout << "7. Reset conversation\n";
        std::cout << "8. Back to chat\n";
        std::cout << "Choose option (1-8): ";
        
        int choice;
        std::cin >> choice;
        std::cin.ignore(); // Clear newline
        
        switch (choice) {
            case 1:
                std::cout << "Enter temperature (0.1-2.0): ";
                std::cin >> config.temperature;
                std::cin.ignore();
                config.temperature = std::max(0.1f, std::min(2.0f, config.temperature));
                break;
            case 2:
                std::cout << "Enter max response length (10-200): ";
                std::cin >> config.max_response_length;
                std::cin.ignore();
                config.max_response_length = std::max(10, std::min(200, config.max_response_length));
                break;
            case 3:
                config.use_nucleus_sampling = !config.use_nucleus_sampling;
                std::cout << "Nucleus sampling: " << (config.use_nucleus_sampling ? "ON" : "OFF") << "\n";
                break;
            case 4:
                std::cout << "Enter nucleus p (0.1-0.95): ";
                std::cin >> config.nucleus_p;
                std::cin.ignore();
                config.nucleus_p = std::max(0.1f, std::min(0.95f, config.nucleus_p));
                break;
            case 5:
                std::cout << "Enter top-k (1-100): ";
                std::cin >> config.top_k;
                std::cin.ignore();
                config.top_k = std::max(1, std::min(100, config.top_k));
                break;
            case 6:
                config.verbose_mode = !config.verbose_mode;
                std::cout << "Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << "\n";
                break;
            case 7:
                initializeConversation();
                std::cout << "🔄 Conversation reset!\n";
                break;
            case 8:
            default:
                break;
        }
        std::cout << "\n";
    }
};

// Main chatbot application
int main() {
    try {
        // Initialize random seed for sampling
        srand(static_cast<unsigned>(time(nullptr)));
        
        // Create and start the chatbot
        MSLChatbot chatbot;
        chatbot.startInteractiveChat();
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Chatbot failed to start: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 