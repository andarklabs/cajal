//
// MSL Transformer Chatbot with Training
// Includes a quick training step to teach the model basic responses
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

class MSLTrainingChatbot {
private:
    std::unique_ptr<TransformerModel> model;
    std::vector<uint32_t> conversation_context;
    
    // Chatbot configuration
    struct ChatConfig {
        float temperature = 0.7f;
        int max_response_length = 20;  // Shorter for untrained model
        int max_context_length = 100;  // Shorter for training
        bool use_nucleus_sampling = true;
        float nucleus_p = 0.9f;
        int top_k = 50;
        bool verbose_mode = false;
    } config;
    
    // Special tokens
    const uint32_t BOS_TOKEN = 1;
    const uint32_t EOS_TOKEN = 2;
    const uint32_t PAD_TOKEN = 0;
    const uint32_t USER_TOKEN = 3;
    const uint32_t BOT_TOKEN = 4;
    
public:
    MSLTrainingChatbot() {
        std::cout << "🤖 MSL Transformer Chatbot with Training Starting...\n";
        
        // Initialize with smaller configuration for training
        TransformerConfig chatbot_config;
        chatbot_config.vocab_size = 1776;
        chatbot_config.embedding_dim = 256;  // Smaller for faster training
        chatbot_config.num_heads = 4;        // Fewer heads
        chatbot_config.num_layers = 2;       // Fewer layers
        chatbot_config.ffn_hidden_dim = 512; // Smaller FFN
        chatbot_config.max_sequence_length = 128; // Shorter sequences
        chatbot_config.batch_size = 4;       // Small batch for training
        chatbot_config.learning_rate = 0.01f; // Higher LR for faster learning
        
        try {
            model = std::make_unique<TransformerModel>(chatbot_config);
            
            std::cout << "🔧 Initializing MSL Transformer..." << std::endl;
            if (!model->initialize()) {
                throw std::runtime_error("Failed to initialize MSL Transformer model");
            }
            
            std::cout << "✅ MSL Transformer initialized!\n";
            std::cout << "📊 Model: " << model->getParameterCount() << " parameters\n";
            std::cout << "💾 Memory: " << model->getMemoryUsage() << " MB\n\n";
            
            // Train the model with basic responses
            quickTraining();
            
            // Initialize conversation
            initializeConversation();
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to initialize chatbot: " << e.what() << std::endl;
            throw;
        }
    }
    
    void quickTraining() {
        std::cout << "🎓 Quick Training: Teaching the model basic responses...\n";
        
        // Simple training data - patterns the model should learn
        std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> training_data = {
            // Pattern: hello -> hi
            {{USER_TOKEN, 100}, {BOT_TOKEN, 101}},  // hello -> hi
            {{USER_TOKEN, 102}, {BOT_TOKEN, 103}},  // how -> good  
            {{USER_TOKEN, 104}, {BOT_TOKEN, 105}},  // what -> something
            {{USER_TOKEN, 106}, {BOT_TOKEN, 107}},  // why -> because
            {{USER_TOKEN, 108}, {BOT_TOKEN, 109}},  // when -> now
            
            // Simple sequences
            {{100, 200}, {200, 300}},
            {{101, 201}, {201, 301}},
            {{102, 202}, {202, 302}},
            {{103, 203}, {203, 303}},
        };
        
        std::cout << "Training on " << training_data.size() << " examples...\n";
        
        // Quick training loop
        for (int epoch = 0; epoch < 5; epoch++) {
            float total_loss = 0.0f;
            int successful_steps = 0;
            
            for (const auto& example : training_data) {
                float loss;
                if (model->trainStep(example.first, example.second, loss)) {
                    total_loss += loss;
                    successful_steps++;
                }
            }
            
            if (successful_steps > 0) {
                float avg_loss = total_loss / successful_steps;
                std::cout << "  Epoch " << (epoch + 1) << "/5 - Loss: " << avg_loss << std::endl;
            }
        }
        
        std::cout << "✅ Quick training completed!\n";
        std::cout << "🧠 Model should now generate more meaningful responses.\n\n";
    }
    
    void initializeConversation() {
        conversation_context.clear();
        conversation_context.push_back(BOS_TOKEN);
        
        if (config.verbose_mode) {
            std::cout << "🎯 Conversation initialized\n";
        }
    }
    
    std::string generateResponse(const std::string& user_input) {
        if (config.verbose_mode) {
            std::cout << "🔄 Processing: \"" << user_input << "\"\n";
        }
        
        // Simple tokenization
        std::vector<uint32_t> user_tokens = tokenizeInput(user_input);
        
        // Add to conversation context
        conversation_context.push_back(USER_TOKEN);
        conversation_context.insert(conversation_context.end(), user_tokens.begin(), user_tokens.end());
        conversation_context.push_back(BOT_TOKEN);
        
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
        
        // Map common words to specific tokens (matching training data)
        if (input.find("hello") != std::string::npos) tokens.push_back(100);
        else if (input.find("how") != std::string::npos) tokens.push_back(102);
        else if (input.find("what") != std::string::npos) tokens.push_back(104);
        else if (input.find("why") != std::string::npos) tokens.push_back(106);
        else if (input.find("when") != std::string::npos) tokens.push_back(108);
        else {
            // Default tokenization
            std::istringstream iss(input);
            std::string word;
            while (iss >> word) {
                uint32_t token_id = (std::hash<std::string>{}(word) % 1500) + 200;
                tokens.push_back(token_id);
            }
        }
        
        if (config.verbose_mode) {
            std::cout << "📝 Tokenized: " << tokens.size() << " tokens\n";
        }
        
        return tokens;
    }
    
    std::vector<uint32_t> generateTokensWithModel() {
        std::vector<uint32_t> generated_tokens;
        
        try {
            for (int step = 0; step < config.max_response_length; step++) {
                // Get logits from model
                std::vector<float> logits = model->generateNext(conversation_context);
                
                if (logits.empty()) {
                    std::cerr << "❌ Empty logits from model\n";
                    break;
                }
                
                // Sample next token
                uint32_t next_token = sampleNextToken(logits, step);
                
                // Stop on special tokens
                if (next_token == EOS_TOKEN || next_token == USER_TOKEN) {
                    break;
                }
                
                generated_tokens.push_back(next_token);
                conversation_context.push_back(next_token);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error in generation: " << e.what() << std::endl;
            // Fallback response
            generated_tokens = {200, 201, 202}; // "I understand"
        }
        
        return generated_tokens;
    }
    
    uint32_t sampleNextToken(const std::vector<float>& logits, int /* step */) {
        if (logits.empty()) return 200; // Fallback
        
        // Apply temperature scaling
        std::vector<float> scaled_logits = logits;
        for (auto& logit : scaled_logits) {
            logit /= config.temperature;
        }
        
        // Simple top-k sampling
        return topKSampling(scaled_logits, config.top_k);
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
            
            // Map trained tokens back to words
            switch (token_id) {
                case 101: oss << "hi"; break;
                case 103: oss << "good"; break;
                case 105: oss << "something"; break;
                case 107: oss << "because"; break;
                case 109: oss << "now"; break;
                case 200: oss << "I"; break;
                case 201: oss << "understand"; break;
                case 202: oss << "you"; break;
                case 300: oss << "hello"; break;
                case 301: oss << "there"; break;
                case 302: oss << "friend"; break;
                case 303: oss << "thanks"; break;
                default:
                    if (token_id < 100) {
                        oss << "[SPECIAL_" << token_id << "]";
                    } else {
                        oss << "word" << (token_id % 100);
                    }
                    break;
            }
        }
        
        return oss.str();
    }
    
public:
    void startInteractiveChat() {
        std::cout << "🚀 MSL Transformer Chatbot with Training Ready!\n";
        std::cout << "💡 The model has been quickly trained on basic patterns.\n";
        std::cout << "🎯 Try: 'hello', 'how are you', 'what', 'why', 'when'\n";
        std::cout << "💡 Type 'quit' to exit, 'config' for settings.\n";
        std::cout << "═══════════════════════════════════════════════════════\n\n";
        
        std::string user_input;
        
        while (true) {
            std::cout << "👤 You: ";
            std::getline(std::cin, user_input);
            
            if (user_input == "quit" || user_input == "exit") {
                std::cout << "\n🤖 Goodbye! Thanks for testing the MSL Transformer!\n";
                break;
            }
            
            if (user_input == "config") {
                showConfigMenu();
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
            
            std::cout << response << std::endl;
            
            if (config.verbose_mode) {
                std::cout << "⏱️ Response in " << duration.count() << "ms\n";
            }
            std::cout << std::endl;
        }
    }
    
    void showConfigMenu() {
        std::cout << "\n⚙️ Chatbot Configuration:\n";
        std::cout << "1. Temperature: " << config.temperature << "\n";
        std::cout << "2. Max response length: " << config.max_response_length << "\n";
        std::cout << "3. Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << "\n";
        std::cout << "4. Reset conversation\n";
        std::cout << "5. Back to chat\n";
        std::cout << "Choose option (1-5): ";
        
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
                std::cout << "Enter max response length (5-50): ";
                std::cin >> config.max_response_length;
                std::cin.ignore();
                config.max_response_length = std::max(5, std::min(50, config.max_response_length));
                break;
            case 3:
                config.verbose_mode = !config.verbose_mode;
                std::cout << "Verbose mode: " << (config.verbose_mode ? "ON" : "OFF") << "\n";
                break;
            case 4:
                initializeConversation();
                std::cout << "🔄 Conversation reset!\n";
                break;
            case 5:
            default:
                break;
        }
        std::cout << "\n";
    }
};

int main() {
    try {
        srand(static_cast<unsigned>(time(nullptr)));
        
        MSLTrainingChatbot chatbot;
        chatbot.startInteractiveChat();
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Chatbot failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 