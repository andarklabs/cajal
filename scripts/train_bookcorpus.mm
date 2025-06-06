//
// BookCorpus Training Script for MSL Transformer
// Trains the model on BookCorpus data and saves the trained weights
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>
#include <random>
#include "../src/host/transformer_model.h"

class BookCorpusTrainer {
private:
    std::unique_ptr<TransformerModel> model;
    TransformerConfig config;
    std::vector<std::string> text_data;
    std::vector<std::vector<uint32_t>> tokenized_sequences;
    
public:
    BookCorpusTrainer() {
        // Setup optimal training configuration
        config.vocab_size = 1776;
        config.embedding_dim = 512;
        config.num_heads = 8;
        config.num_layers = 6;
        config.ffn_hidden_dim = 2048;
        config.max_sequence_length = 512;
        config.batch_size = 8;  // Reasonable for 36GB M3 Max
        config.learning_rate = 1e-4f;
        config.weight_decay = 0.01f;
        config.gradient_clipping = true;
        config.max_grad_norm = 1.0f;
        
        std::cout << "🎓 BookCorpus MSL Transformer Trainer" << std::endl;
        std::cout << "📊 Configuration:" << std::endl;
        std::cout << "   📝 Vocab size: " << config.vocab_size << std::endl;
        std::cout << "   🧠 Model dim: " << config.embedding_dim << std::endl;
        std::cout << "   🔄 Layers: " << config.num_layers << std::endl;
        std::cout << "   👁️  Heads: " << config.num_heads << std::endl;
        std::cout << "   🔢 FFN dim: " << config.ffn_hidden_dim << std::endl;
        std::cout << "   📏 Max seq len: " << config.max_sequence_length << std::endl;
        std::cout << "   📦 Batch size: " << config.batch_size << std::endl;
        std::cout << "   📈 Learning rate: " << config.learning_rate << std::endl;
    }
    
    bool loadBookCorpusData() {
        std::cout << "\n📚 Loading BookCorpus data..." << std::endl;
        
        std::vector<std::string> corpus_files = {
            "data/bookcorpus/books_large_p1.txt",
            "data/bookcorpus/books_large_p2.txt"
        };
        
        for (const auto& filename : corpus_files) {
            std::cout << "📖 Loading: " << filename << std::endl;
            
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "❌ Failed to open: " << filename << std::endl;
                return false;
            }
            
            std::string line;
            int line_count = 0;
            int max_lines_per_file = 10000; // Limit for faster training
            
            while (std::getline(file, line) && line_count < max_lines_per_file) {
                if (!line.empty() && line.length() > 10) { // Filter short lines
                    text_data.push_back(line);
                    line_count++;
                }
                
                if (line_count % 1000 == 0) {
                    std::cout << "  📄 Loaded " << line_count << " lines..." << std::endl;
                }
            }
            
            file.close();
            std::cout << "✅ Loaded " << line_count << " lines from " << filename << std::endl;
        }
        
        std::cout << "📊 Total text samples: " << text_data.size() << std::endl;
        return !text_data.empty();
    }
    
    bool tokenizeData() {
        std::cout << "\n🔤 Tokenizing text data..." << std::endl;
        
        tokenized_sequences.clear();
        tokenized_sequences.reserve(text_data.size());
        
        for (size_t i = 0; i < text_data.size(); i++) {
            std::vector<uint32_t> tokens = simpleTokenize(text_data[i]);
            
            if (tokens.size() >= 10 && tokens.size() <= config.max_sequence_length) {
                tokenized_sequences.push_back(tokens);
            }
            
            if (i % 1000 == 0) {
                std::cout << "  🔢 Tokenized " << i << "/" << text_data.size() << " sequences..." << std::endl;
            }
        }
        
        std::cout << "✅ Created " << tokenized_sequences.size() << " tokenized sequences" << std::endl;
        
        // Clear text data to save memory
        text_data.clear();
        
        return !tokenized_sequences.empty();
    }
    
    std::vector<uint32_t> simpleTokenize(const std::string& text) {
        std::vector<uint32_t> tokens;
        
        // Simple word-based tokenization
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Simple hash-based tokenization (could be improved with actual BPE)
            std::hash<std::string> hasher;
            uint32_t token_id = (hasher(word) % (config.vocab_size - 100)) + 100; // Reserve 0-99 for special tokens
            tokens.push_back(token_id);
            
            if (tokens.size() >= config.max_sequence_length - 1) {
                break;
            }
        }
        
        return tokens;
    }
    
    bool initializeModel() {
        std::cout << "\n🔧 Initializing MSL Transformer Model..." << std::endl;
        
        try {
            model = std::make_unique<TransformerModel>(config);
            
            if (!model->initialize()) {
                std::cerr << "❌ Failed to initialize transformer model" << std::endl;
                return false;
            }
            
            std::cout << "✅ Model initialized successfully!" << std::endl;
            std::cout << "📊 Parameters: " << model->getParameterCount() << std::endl;
            std::cout << "💾 Memory: " << model->getMemoryUsage() << " MB" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error initializing model: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool trainModel(int num_epochs = 3, int save_every_epochs = 1) {
        std::cout << "\n🎓 Starting BookCorpus Training..." << std::endl;
        std::cout << "📅 Epochs: " << num_epochs << std::endl;
        std::cout << "💾 Save every: " << save_every_epochs << " epochs" << std::endl;
        
        // Shuffle sequences for better training
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(tokenized_sequences.begin(), tokenized_sequences.end(), gen);
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            std::cout << "\n📈 Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
            
            auto epoch_start = std::chrono::high_resolution_clock::now();
            float total_loss = 0.0f;
            int successful_batches = 0;
            int batch_count = 0;
            
            // Create batches
            for (size_t i = 0; i < tokenized_sequences.size(); i += config.batch_size) {
                std::vector<std::vector<uint32_t>> input_batch;
                std::vector<std::vector<uint32_t>> target_batch;
                
                // Create batch
                for (size_t j = i; j < i + config.batch_size && j < tokenized_sequences.size(); j++) {
                    const auto& sequence = tokenized_sequences[j];
                    
                    if (sequence.size() > 1) {
                        std::vector<uint32_t> input(sequence.begin(), sequence.end() - 1);
                        std::vector<uint32_t> target(sequence.begin() + 1, sequence.end());
                        
                        input_batch.push_back(input);
                        target_batch.push_back(target);
                    }
                }
                
                if (input_batch.empty()) continue;
                
                // Train on batch
                float batch_loss;
                if (model->trainBatch(input_batch, target_batch, batch_loss)) {
                    total_loss += batch_loss;
                    successful_batches++;
                } else {
                    std::cerr << "⚠️  Failed batch " << batch_count << std::endl;
                }
                
                batch_count++;
                
                // Progress reporting
                if (batch_count % 100 == 0) {
                    float avg_loss = successful_batches > 0 ? total_loss / successful_batches : 0.0f;
                    std::cout << "  📊 Batch " << batch_count << ", avg loss: " << avg_loss << std::endl;
                }
                
                // Prevent overheating/overload (optional)
                if (batch_count % 500 == 0) {
                    std::cout << "  🌡️  Cooling break..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::minutes>(epoch_end - epoch_start);
            
            float avg_loss = successful_batches > 0 ? total_loss / successful_batches : 0.0f;
            
            std::cout << "✅ Epoch " << (epoch + 1) << " completed in " << epoch_duration.count() << " min" << std::endl;
            std::cout << "📊 Average loss: " << avg_loss << std::endl;
            std::cout << "✅ Successful batches: " << successful_batches << "/" << batch_count << std::endl;
            
            // Save checkpoint
            if ((epoch + 1) % save_every_epochs == 0) {
                std::string checkpoint_path = "models/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
                saveModelCheckpoint(checkpoint_path, epoch + 1);
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(training_end - training_start);
        
        std::cout << "\n🎉 Training completed in " << total_duration.count() << " minutes!" << std::endl;
        
        return true;
    }
    
    bool saveModelCheckpoint(const std::string& filepath, int epoch) {
        std::cout << "💾 Saving model checkpoint: " << filepath << std::endl;
        
        // Create models directory if it doesn't exist
        system("mkdir -p models");
        
        uint32_t step = epoch * (tokenized_sequences.size() / config.batch_size);
        
        if (!model->saveCheckpoint(filepath, epoch, step)) {
            std::cerr << "❌ Failed to save checkpoint" << std::endl;
            return false;
        }
        
        std::cout << "✅ Checkpoint saved successfully" << std::endl;
        return true;
    }
    
    bool saveFinalModel() {
        std::cout << "\n💾 Saving final trained model..." << std::endl;
        
        // Create models directory if it doesn't exist
        system("mkdir -p models");
        
        std::string model_path = "models/bookcorpus_trained_model.bin";
        
        if (!model->saveWeights(model_path)) {
            std::cerr << "❌ Failed to save final model" << std::endl;
            return false;
        }
        
        std::cout << "✅ Final model saved to: " << model_path << std::endl;
        
        // Also save a copy with timestamp
        auto now = std::time(nullptr);
        auto* tm = std::localtime(&now);
        char timestamp[100];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm);
        
        std::string timestamped_path = "models/bookcorpus_model_" + std::string(timestamp) + ".bin";
        if (model->saveWeights(timestamped_path)) {
            std::cout << "✅ Timestamped backup saved to: " << timestamped_path << std::endl;
        }
        
        return true;
    }
    
    void printTrainingInfo() {
        std::cout << "\n📋 Training Summary:" << std::endl;
        std::cout << "📊 Total sequences: " << tokenized_sequences.size() << std::endl;
        std::cout << "📦 Batches per epoch: " << (tokenized_sequences.size() / config.batch_size) << std::endl;
        std::cout << "🎯 Model parameters: " << model->getParameterCount() << std::endl;
        std::cout << "💾 Memory usage: " << model->getMemoryUsage() << " MB" << std::endl;
        std::cout << "📈 Learning rate: " << model->getCurrentLearningRate() << std::endl;
    }
};

int main() {
    std::cout << "🚀 MSL Transformer BookCorpus Training Script" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        BookCorpusTrainer trainer;
        
        // Step 1: Load BookCorpus data
        if (!trainer.loadBookCorpusData()) {
            std::cerr << "❌ Failed to load BookCorpus data" << std::endl;
            return 1;
        }
        
        // Step 2: Tokenize the data
        if (!trainer.tokenizeData()) {
            std::cerr << "❌ Failed to tokenize data" << std::endl;
            return 1;
        }
        
        // Step 3: Initialize the model
        if (!trainer.initializeModel()) {
            std::cerr << "❌ Failed to initialize model" << std::endl;
            return 1;
        }
        
        // Step 4: Print training info
        trainer.printTrainingInfo();
        
        // Step 5: Train the model
        std::cout << "\n🎓 Ready to start training. Press Enter to continue...";
        std::cin.get();
        
        if (!trainer.trainModel(3, 1)) { // 3 epochs, save every epoch
            std::cerr << "❌ Training failed" << std::endl;
            return 1;
        }
        
        // Step 6: Save the final model
        if (!trainer.saveFinalModel()) {
            std::cerr << "❌ Failed to save final model" << std::endl;
            return 1;
        }
        
        std::cout << "\n🎉 BookCorpus training completed successfully!" << std::endl;
        std::cout << "🤖 Your trained model is ready for the chatbot!" << std::endl;
        std::cout << "📁 Model saved in: models/bookcorpus_trained_model.bin" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Training failed with error: " << e.what() << std::endl;
        return 1;
    }
} 