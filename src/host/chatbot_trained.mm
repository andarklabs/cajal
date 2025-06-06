//
// MSL Transformer Chatbot with Pre-trained BookCorpus Model
// Loads trained weights and provides intelligent conversation
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

class MSLTrainedChatbot {
private:
    std::unique_ptr<TransformerModel> model;
    std::vector<uint32_t> conversation_context;
    
    // Chatbot configuration
    struct ChatConfig {
        float temperature = 0.8f;
        int max_response_length = 50;
        int max_context_length = 400;
        bool use_nucleus_sampling = true;
        float nucleus_p = 0.9f;
        int top_k = 50;
        bool verbose_mode = false;
        std::string model_path = "models/bookcorpus_trained_model.bin";
    } config;
    
    // Special tokens
    const uint32_t BOS_TOKEN = 1;
    const uint32_t EOS_TOKEN = 2;
    const uint32_t PAD_TOKEN = 0;
    
public:
    MSLTrainedChatbot() {
        std::cout << "🤖 MSL Transformer Chatbot (BookCorpus Trained)" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        // Use the same configuration as training
        TransformerConfig chatbot_config;
        chatbot_config.vocab_size = 1776;
        chatbot_config.embedding_dim = 512;
        chatbot_config.num_heads = 8;
        chatbot_config.num_layers = 6;
        chatbot_config.ffn_hidden_dim = 2048;
        chatbot_config.max_sequence_length = 512;
        chatbot_config.batch_size = 1;  // Single sequence for chatbot
        chatbot_config.learning_rate = 1e-4f;
        
        try {
            model = std::make_unique<TransformerModel>(chatbot_config);
            
            std::cout << "🔧 Initializing MSL Transformer..." << std::endl;
            if (!model->initialize()) {
                throw std::runtime_error("Failed to initialize MSL Transformer model");
            }
            
            std::cout << "📂 Loading trained BookCorpus model..." << std::endl;
            if (!loadTrainedModel()) {
                throw std::runtime_error("Failed to load trained model weights");
            }
            
            std::cout << "✅ BookCorpus-trained MSL Transformer loaded!" << std::endl;
            std::cout << "📊 Model: " << model->getParameterCount() << " parameters" << std::endl;
            std::cout << "💾 Memory: " << model->getMemoryUsage() << " MB" << std::endl;
            std::cout << "🧠 Training: BookCorpus literature dataset" << std::endl;
            
            // Initialize conversation
            initializeConversation();
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to initialize trained chatbot: " << e.what() << std::endl;
            throw;
        }
    }
    
    bool loadTrainedModel() {
        std::cout << "🔍 Looking for trained model at: " << config.model_path << std::endl;
        
        if (!model->loadWeights(config.model_path)) {
            std::cout << "⚠️  Trained model not found. Available options:" << std::endl;
            std::cout << "   1. Train a model first: make -f Makefile.train train" << std::endl;
            std::cout << "   2. Use different model path in config" << std::endl;
            std::cout << "   3. Continue with random initialization (for testing)" << std::endl;
            
            std::cout << "\n❓ Continue with random weights? (y/N): ";
            std::string response;
            std::getline(std::cin, response);
            
            if (response == "y" || response == "Y") {
                std::cout << "⚠️  Using randomly initialized weights" << std::endl;
                std::cout << "💡 Responses will be incoherent until model is trained" << std::endl;
                return true;
            } else {
                return false;
            }
        }
        
        std::cout << "✅ Trained model weights loaded successfully!" << std::endl;
        return true;
    }
    
    void initializeConversation() {
        conversation_context.clear();
        conversation_context.push_back(BOS_TOKEN);
        
        if (config.verbose_mode) {
            std::cout << "🎯 Conversation initialized with trained model" << std::endl;
        }
    }
    
    std::string generateResponse(const std::string& user_input) {
        if (config.verbose_mode) {
            std::cout << "🔄 Processing: \"" << user_input << "\"" << std::endl;
        }
        
        // Improved tokenization for BookCorpus vocabulary
        std::vector<uint32_t> user_tokens = tokenizeInput(user_input);
        
        // Add to conversation context
        conversation_context.insert(conversation_context.end(), user_tokens.begin(), user_tokens.end());
        
        // Trim context if too long
        if (conversation_context.size() > static_cast<size_t>(config.max_context_length)) {
            int tokens_to_remove = static_cast<int>(conversation_context.size()) - config.max_context_length;
            conversation_context.erase(conversation_context.begin() + 1, 
                                     conversation_context.begin() + 1 + tokens_to_remove);
        }
        
        // Generate response
        std::vector<uint32_t> response_tokens = generateTokensWithModel();
        
        // Convert back to text
        std::string response = detokenizeOutput(response_tokens);
        
        // Add to conversation context
        conversation_context.insert(conversation_context.end(), response_tokens.begin(), response_tokens.end());
        
        return response;
    }
    
private:
    std::vector<uint32_t> tokenizeInput(const std::string& input) {
        std::vector<uint32_t> tokens;
        
        // Use the same tokenization as training (hash-based)
        std::istringstream iss(input);
        std::string word;
        
        while (iss >> word) {
            // Clean word (remove punctuation for better matching)
            std::string clean_word;
            for (char c : word) {
                if (std::isalnum(c)) {
                    clean_word += std::tolower(c);
                }
            }
            
            if (!clean_word.empty()) {
                // Use same hash function as training
                std::hash<std::string> hasher;
                uint32_t token_id = (hasher(clean_word) % 1676) + 100; // Same range as training
                tokens.push_back(token_id);
            }
        }
        
        if (config.verbose_mode) {
            std::cout << "📝 Tokenized: " << tokens.size() << " tokens" << std::endl;
        }
        
        return tokens;
    }
    
    std::vector<uint32_t> generateTokensWithModel() {
        std::vector<uint32_t> generated_tokens;
        
        try {
            for (int step = 0; step < config.max_response_length; step++) {
                // Get logits from trained model
                std::vector<float> logits = model->generateNext(conversation_context);
                
                if (logits.empty()) {
                    if (config.verbose_mode) {
                        std::cerr << "⚠️  Empty logits from model at step " << step << std::endl;
                    }
                    break;
                }
                
                // Sample next token with improved sampling
                uint32_t next_token = sampleNextToken(logits, step);
                
                // Stop on special tokens or if we detect end of response
                if (next_token == EOS_TOKEN || next_token == PAD_TOKEN) {
                    break;
                }
                
                generated_tokens.push_back(next_token);
                conversation_context.push_back(next_token);
                
                // Optional: Early stopping if we generate punctuation
                if (step > 5 && isEndOfSentenceToken(next_token)) {
                    break;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error in generation: " << e.what() << std::endl;
            // Return empty instead of fallback for trained model
        }
        
        return generated_tokens;
    }
    
    bool isEndOfSentenceToken(uint32_t token) {
        // Simple heuristic: tokens that commonly end sentences
        // (This is approximate since we're using hash-based tokenization)
        return (token % 100) < 5; // Roughly 5% chance to end sentence
    }
    
    uint32_t sampleNextToken(const std::vector<float>& logits, int step) {
        if (logits.empty()) return 100; // Fallback
        
        // Apply temperature scaling
        std::vector<float> scaled_logits = logits;
        
        // Dynamic temperature: higher for first few tokens, lower later
        float dynamic_temp = config.temperature * (1.0f + 0.5f * std::exp(-step / 10.0f));
        
        for (auto& logit : scaled_logits) {
            logit /= dynamic_temp;
        }
        
        // Use nucleus sampling for more coherent responses
        if (config.use_nucleus_sampling) {
            return nucleusSampling(scaled_logits, config.nucleus_p);
        } else {
            return topKSampling(scaled_logits, config.top_k);
        }
    }
    
    uint32_t nucleusSampling(const std::vector<float>& logits, float p) {
        std::vector<std::pair<float, uint32_t>> logit_pairs;
        for (size_t i = 0; i < logits.size(); i++) {
            logit_pairs.push_back({logits[i], static_cast<uint32_t>(i)});
        }
        
        // Sort by probability (descending)
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Apply softmax to get probabilities
        float max_logit = logit_pairs[0].first;
        float sum = 0.0f;
        for (auto& pair : logit_pairs) {
            pair.first = std::exp(pair.first - max_logit);
            sum += pair.first;
        }
        for (auto& pair : logit_pairs) {
            pair.first /= sum;
        }
        
        // Find nucleus (top-p)
        float cumulative_prob = 0.0f;
        size_t nucleus_size = 0;
        for (size_t i = 0; i < logit_pairs.size(); i++) {
            cumulative_prob += logit_pairs[i].first;
            nucleus_size = i + 1;
            if (cumulative_prob >= p) break;
        }
        
        // Sample from nucleus
        float r = static_cast<float>(rand()) / RAND_MAX * cumulative_prob;
        cumulative_prob = 0.0f;
        for (size_t i = 0; i < nucleus_size; i++) {
            cumulative_prob += logit_pairs[i].first;
            if (r <= cumulative_prob) {
                return logit_pairs[i].second;
            }
        }
        
        return logit_pairs[0].second; // Fallback to most probable
    }
    
    uint32_t topKSampling(const std::vector<float>& logits, int k) {
        std::vector<std::pair<float, uint32_t>> logit_pairs;
        for (size_t i = 0; i < logits.size(); i++) {
            logit_pairs.push_back({logits[i], static_cast<uint32_t>(i)});
        }
        
        std::sort(logit_pairs.begin(), logit_pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        int actual_k = std::min(k, static_cast<int>(logit_pairs.size()));
        int selected_idx = rand() % actual_k;
        return logit_pairs[selected_idx].second;
    }
    
    std::string detokenizeOutput(const std::vector<uint32_t>& tokens) {
        std::ostringstream oss;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) oss << " ";
            
            uint32_t token_id = tokens[i];
            
            // Since we used hash-based tokenization, we can't perfectly reverse it
            // but we can create plausible words based on token patterns
            std::string word = generatePlausibleWord(token_id);
            oss << word;
        }
        
        return oss.str();
    }
    
    std::string generatePlausibleWord(uint32_t token_id) {
        // Generate pseudo-words based on token IDs
        // This creates more readable output than showing raw token numbers
        
        if (token_id < 10) return "[SPECIAL]";
        
        // Use token ID to generate consistent pseudo-words
        std::string consonants = "bcdfghjklmnpqrstvwxyz";
        std::string vowels = "aeiou";
        
        std::string word;
        int length = (token_id % 5) + 3; // 3-7 character words
        
        for (int i = 0; i < length; i++) {
            if (i % 2 == 0) {
                word += consonants[(token_id + i) % consonants.length()];
            } else {
                word += vowels[(token_id + i) % vowels.length()];
            }
        }
        
        return word;
    }
    
public:
    void startInteractiveChat() {
        std::cout << "\n🚀 BookCorpus-Trained MSL Transformer Chatbot Ready!" << std::endl;
        std::cout << "📚 This model was trained on literature from BookCorpus" << std::endl;
        std::cout << "💡 Type your message and press Enter. Type 'quit' to exit." << std::endl;
        std::cout << "⚙️ Type 'config' to adjust settings." << std::endl;
        std::cout << "═══════════════════════════════════════════════════════" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\n👤 You: ";
            std::getline(std::cin, user_input);
            
            if (user_input == "quit" || user_input == "exit") {
                std::cout << "\n🤖 Goodbye! Thanks for chatting with the MSL Transformer!" << std::endl;
                break;
            }
            
            if (user_input == "config") {
                showConfigMenu();
                continue;
            }
            
            if (user_input == "reset") {
                initializeConversation();
                std::cout << "🔄 Conversation reset!" << std::endl;
                continue;
            }
            
            if (user_input.empty()) continue;
            
            // Generate response
            std::cout << "🤖 Bot: ";
            std::cout.flush();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            std::string response = generateResponse(user_input);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (response.empty()) {
                std::cout << "[Model generated empty response]" << std::endl;
            } else {
                std::cout << response << std::endl;
            }
            
            if (config.verbose_mode) {
                std::cout << "⏱️ Response generated in " << duration.count() << "ms" << std::endl;
                std::cout << "📊 Context length: " << conversation_context.size() << " tokens" << std::endl;
            }
        }
    }
    
    void showConfigMenu() {
        std::cout << "\n⚙️ Chatbot Configuration:" << std::endl;
        std::cout << "1. Temperature: " << config.temperature << std::endl;
        std::cout << "2. Max response length: " << config.max_response_length << std::endl;
        std::cout << "3. Sampling method: " << (config.use_nucleus_sampling ? "Nucleus (p=" + std::to_string(config.nucleus_p) + ")" : "Top-k (" + std::to_string(config.top_k) + ")") << std::endl;
        std::cout << "4. Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << std::endl;
        std::cout << "5. Model path: " << config.model_path << std::endl;
        std::cout << "6. Reset conversation" << std::endl;
        std::cout << "7. Back to chat" << std::endl;
        std::cout << "Choose option (1-7): ";
        
        int choice;
        std::cin >> choice;
        std::cin.ignore();
        
        switch (choice) {
            case 1:
                std::cout << "Enter temperature (0.1-2.0): ";
                std::cin >> config.temperature;
                std::cin.ignore();
                config.temperature = std::max(0.1f, std::min(2.0f, config.temperature));
                break;
            case 2:
                std::cout << "Enter max response length (5-100): ";
                std::cin >> config.max_response_length;
                std::cin.ignore();
                config.max_response_length = std::max(5, std::min(100, config.max_response_length));
                break;
            case 3:
                config.use_nucleus_sampling = !config.use_nucleus_sampling;
                std::cout << "Sampling: " << (config.use_nucleus_sampling ? "Nucleus" : "Top-k") << std::endl;
                break;
            case 4:
                config.verbose_mode = !config.verbose_mode;
                std::cout << "Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << std::endl;
                break;
            case 5:
                std::cout << "Enter model path: ";
                std::getline(std::cin, config.model_path);
                break;
            case 6:
                initializeConversation();
                std::cout << "🔄 Conversation reset!" << std::endl;
                break;
            case 7:
            default:
                break;
        }
        std::cout << std::endl;
    }
};

int main() {
    try {
        srand(static_cast<unsigned>(time(nullptr)));
        
        MSLTrainedChatbot chatbot;
        chatbot.startInteractiveChat();
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Trained chatbot failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 