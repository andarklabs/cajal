#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <sstream>
#include "transformer_model.h"

// Performance monitoring utilities
class PerformanceMonitor {
public:
    void startMeasurement(const std::string& name) {
        measurements[name].start = std::chrono::high_resolution_clock::now();
    }
    
    double endMeasurement(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - measurements[name].start).count() / 1000.0; // Convert to ms
        measurements[name].durations.push_back(duration);
        return duration;
    }
    
    void reportStatistics() {
        std::cout << "\n📊 Performance Statistics:\n";
        for (const auto& [name, data] : measurements) {
            if (data.durations.empty()) continue;
            
            // Calculate statistics
            double sum = 0;
            double min = data.durations[0];
            double max = data.durations[0];
            
            for (double d : data.durations) {
                sum += d;
                min = std::min(min, d);
                max = std::max(max, d);
            }
            
            double avg = sum / data.durations.size();
            
            // Calculate 95th percentile
            std::vector<double> sorted = data.durations;
            std::sort(sorted.begin(), sorted.end());
            size_t p95_idx = static_cast<size_t>(sorted.size() * 0.95);
            double p95 = sorted[p95_idx];
            
            std::cout << "\n🔍 " << name << ":\n"
                      << "   • Average: " << avg << " ms\n"
                      << "   • Min: " << min << " ms\n"
                      << "   • Max: " << max << " ms\n"
                      << "   • P95: " << p95 << " ms\n"
                      << "   • Samples: " << data.durations.size() << "\n";
        }
    }
    
private:
    struct MeasurementData {
        std::chrono::high_resolution_clock::time_point start;
        std::vector<double> durations;
    };
    std::map<std::string, MeasurementData> measurements;
};

// Test configuration
struct TestConfig {
    uint32_t vocab_size = 1776;
    uint32_t embedding_dim = 1024;  // Increased for real-world testing
    uint32_t num_layers = 8;       // More layers
    uint32_t num_heads = 16;       // More attention heads
    uint32_t ffn_hidden_dim = 4096; // 4x embedding_dim
    uint32_t max_sequence_length = 512;
    uint32_t batch_size = 32;
    float learning_rate = 1e-4f;
    uint32_t warmup_steps = 5;     // Warmup iterations before measurement
    uint32_t test_steps = 20;      // Training steps to measure
};

// Generate synthetic training data
std::vector<std::vector<uint32_t>> generateTrainingData(const TestConfig& config, uint32_t num_sequences) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, config.vocab_size - 1);
    
    std::vector<std::vector<uint32_t>> sequences;
    for (uint32_t i = 0; i < num_sequences; i++) {
        std::vector<uint32_t> sequence;
        for (uint32_t j = 0; j < config.max_sequence_length; j++) {
            sequence.push_back(dist(gen));
        }
        sequences.push_back(sequence);
    }
    return sequences;
}

void runTrainingTest() {
    std::cout << "🚀 Starting Real-World Training Test\n";
    std::cout << "Following TDD principles and cursor rules\n\n";
    
    // Initialize performance monitoring
    PerformanceMonitor monitor;
    
    // Configure larger model for real-world testing
    TestConfig config;
    
    // Initialize transformer model
    TransformerConfig transformer_config;
    transformer_config.vocab_size = config.vocab_size;
    transformer_config.embedding_dim = config.embedding_dim;
    transformer_config.num_layers = config.num_layers;
    transformer_config.num_heads = config.num_heads;
    transformer_config.ffn_hidden_dim = config.ffn_hidden_dim;
    transformer_config.max_sequence_length = config.max_sequence_length;
    transformer_config.batch_size = config.batch_size;
    transformer_config.learning_rate = config.learning_rate;
    
    TransformerModel model(transformer_config);
    
    std::cout << "🔧 Initializing Transformer Model...\n";
    if (!model.initialize()) {
        std::cerr << "❌ Failed to initialize model\n";
        return;
    }
    
    std::cout << "📈 Model Configuration:\n"
              << "   • Embedding Dim: " << config.embedding_dim << "\n"
              << "   • Layers: " << config.num_layers << "\n"
              << "   • Attention Heads: " << config.num_heads << "\n"
              << "   • FFN Hidden Dim: " << config.ffn_hidden_dim << "\n"
              << "   • Sequence Length: " << config.max_sequence_length << "\n"
              << "   • Batch Size: " << config.batch_size << "\n";
    
    // Generate training data
    std::cout << "\n📊 Generating training data...\n";
    uint32_t total_sequences = (config.warmup_steps + config.test_steps) * config.batch_size;
    auto training_data = generateTrainingData(config, total_sequences);
    
    // Warmup phase
    std::cout << "\n🔥 Warmup Phase: " << config.warmup_steps << " steps\n";
    for (uint32_t step = 0; step < config.warmup_steps; step++) {
        uint32_t start_idx = step * config.batch_size;
        std::vector<std::vector<uint32_t>> input_batch(
            training_data.begin() + start_idx,
            training_data.begin() + start_idx + config.batch_size
        );
        
        // Use next tokens as targets (simple language modeling)
        std::vector<std::vector<uint32_t>> target_batch = input_batch;
        for (auto& seq : target_batch) {
            std::rotate(seq.begin(), seq.begin() + 1, seq.end());
        }
        
        float loss;
        model.trainBatch(input_batch, target_batch, loss);
        std::cout << "   • Warmup step " << step + 1 << "/" << config.warmup_steps 
                  << " (loss: " << loss << ")\n";
    }
    
    // Testing phase with measurements
    std::cout << "\n📈 Testing Phase: " << config.test_steps << " steps\n";
    std::vector<float> losses;
    
    for (uint32_t step = 0; step < config.test_steps; step++) {
        uint32_t start_idx = (config.warmup_steps + step) * config.batch_size;
        std::vector<std::vector<uint32_t>> input_batch(
            training_data.begin() + start_idx,
            training_data.begin() + start_idx + config.batch_size
        );
        
        std::vector<std::vector<uint32_t>> target_batch = input_batch;
        for (auto& seq : target_batch) {
            std::rotate(seq.begin(), seq.begin() + 1, seq.end());
        }
        
        // Measure full training step
        monitor.startMeasurement("full_training_step");
        float loss;
        model.trainBatch(input_batch, target_batch, loss);
        double step_time = monitor.endMeasurement("full_training_step");
        
        losses.push_back(loss);
        std::cout << "✓ Step " << step + 1 << "/" << config.test_steps 
                  << " completed in " << step_time << "ms (loss: " << loss << ")\n";
    }
    
    // Report final statistics
    monitor.reportStatistics();
    
    // Calculate and report loss statistics
    double avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    auto [min_loss, max_loss] = std::minmax_element(losses.begin(), losses.end());
    
    std::cout << "\n📊 Training Statistics:\n"
              << "   • Average Loss: " << avg_loss << "\n"
              << "   • Min Loss: " << *min_loss << "\n"
              << "   • Max Loss: " << *max_loss << "\n"
              << "   • Total Steps: " << config.test_steps << "\n";
    
    std::cout << "\n✅ Real-world training test completed successfully!\n";
}

int main() {
    @autoreleasepool {
        std::cout << "🚀 Phase 4.2: Real-World Training Validation\n\n";
        runTrainingTest();
    }
    return 0;
} 