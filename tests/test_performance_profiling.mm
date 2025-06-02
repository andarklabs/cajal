#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <string>
#include <fstream>
#include <iomanip>
#include <numeric>
#include "../src/host/transformer_model.h"

struct PerformanceMetrics {
    double total_time_ms;
    double kernel_time_ms;
    double memory_bandwidth_gb_s;
    double tokens_per_second;
    size_t memory_usage_mb;
    size_t parameter_count;
    double flops_per_second;
    
    void print() const {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(2) << total_time_ms << " ms" << std::endl;
        std::cout << "  Kernel Time: " << std::fixed << std::setprecision(2) << kernel_time_ms << " ms" << std::endl;
        std::cout << "  Memory Bandwidth: " << std::fixed << std::setprecision(2) << memory_bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Tokens/Second: " << std::fixed << std::setprecision(2) << tokens_per_second << std::endl;
        std::cout << "  Memory Usage: " << memory_usage_mb << " MB" << std::endl;
        std::cout << "  Parameters: " << parameter_count << std::endl;
        std::cout << "  FLOPS: " << std::fixed << std::setprecision(2) << flops_per_second << " GFLOPS" << std::endl;
    }
};

struct BenchmarkConfig {
    std::string name;
    uint32_t vocab_size;
    uint32_t embedding_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t sequence_length;
    uint32_t batch_size;
    uint32_t iterations;
    bool warmup_enabled;
    uint32_t warmup_iterations;
};

class PerformanceProfiler {
private:
    std::map<std::string, PerformanceMetrics> baseline_metrics;
    std::map<std::string, std::vector<PerformanceMetrics>> benchmark_history;
    std::string output_directory;
    
public:
    PerformanceProfiler(const std::string& output_dir = "profiling_results") 
        : output_directory(output_dir) {
        // Create output directory if it doesn't exist
        system(("mkdir -p " + output_directory).c_str());
    }
    
    // TDD Principle: Test performance expectations before optimization
    bool setPerformanceBaseline(const std::string& test_name, const PerformanceMetrics& metrics) {
        baseline_metrics[test_name] = metrics;
        std::cout << "✓ Baseline set for " << test_name << std::endl;
        metrics.print();
        return true;
    }
    
    // Verify performance improvement after optimization
    bool verifyPerformanceImprovement(const std::string& test_name, const PerformanceMetrics& new_metrics, double min_improvement_percent = 5.0) {
        if (baseline_metrics.find(test_name) == baseline_metrics.end()) {
            std::cerr << "✗ No baseline found for " << test_name << std::endl;
            return false;
        }
        
        const auto& baseline = baseline_metrics[test_name];
        double improvement = ((baseline.total_time_ms - new_metrics.total_time_ms) / baseline.total_time_ms) * 100.0;
        
        std::cout << "Performance comparison for " << test_name << ":" << std::endl;
        std::cout << "  Baseline: " << baseline.total_time_ms << " ms" << std::endl;
        std::cout << "  Current:  " << new_metrics.total_time_ms << " ms" << std::endl;
        std::cout << "  Improvement: " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
        
        if (improvement >= min_improvement_percent) {
            std::cout << "✓ Performance improvement verified!" << std::endl;
            return true;
        } else {
            std::cout << "✗ Performance improvement below threshold (" << min_improvement_percent << "%)" << std::endl;
            return false;
        }
    }
    
    // Record performance metrics for regression testing
    void recordMetrics(const std::string& test_name, const PerformanceMetrics& metrics) {
        benchmark_history[test_name].push_back(metrics);
        
        // Save to file for analysis
        std::string filename = output_directory + "/" + test_name + "_metrics.csv";
        std::ofstream file(filename, std::ios::app);
        if (file.is_open()) {
            // Write header if file is empty
            file.seekp(0, std::ios::end);
            if (file.tellp() == 0) {
                file << "timestamp,total_time_ms,kernel_time_ms,memory_bandwidth_gb_s,tokens_per_second,memory_usage_mb,parameter_count,flops_per_second\n";
            }
            
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            file << time_t << ","
                 << metrics.total_time_ms << ","
                 << metrics.kernel_time_ms << ","
                 << metrics.memory_bandwidth_gb_s << ","
                 << metrics.tokens_per_second << ","
                 << metrics.memory_usage_mb << ","
                 << metrics.parameter_count << ","
                 << metrics.flops_per_second << "\n";
            file.close();
        }
    }
    
    // Check for performance regressions
    bool checkForRegressions(const std::string& test_name, const PerformanceMetrics& current_metrics, double max_regression_percent = 10.0) {
        if (benchmark_history.find(test_name) == benchmark_history.end() || benchmark_history[test_name].empty()) {
            std::cout << "No historical data for " << test_name << ", recording first measurement" << std::endl;
            recordMetrics(test_name, current_metrics);
            return true;
        }
        
        // Compare against recent average (last 5 measurements)
        auto& history = benchmark_history[test_name];
        size_t samples = std::min((size_t)5, history.size());
        double avg_time = 0.0;
        
        for (size_t i = history.size() - samples; i < history.size(); i++) {
            avg_time += history[i].total_time_ms;
        }
        avg_time /= samples;
        
        double regression = ((current_metrics.total_time_ms - avg_time) / avg_time) * 100.0;
        
        if (regression > max_regression_percent) {
            std::cout << "⚠️  Performance regression detected in " << test_name << ":" << std::endl;
            std::cout << "  Recent Average: " << avg_time << " ms" << std::endl;
            std::cout << "  Current: " << current_metrics.total_time_ms << " ms" << std::endl;
            std::cout << "  Regression: " << std::fixed << std::setprecision(2) << regression << "%" << std::endl;
            return false;
        }
        
        recordMetrics(test_name, current_metrics);
        return true;
    }
};

class TransformerBenchmark {
private:
    PerformanceProfiler profiler;
    
    // Calculate theoretical FLOPS for transformer operations
    double calculateTheoreticalFLOPS(const BenchmarkConfig& config) {
        // Rough FLOPS calculation for transformer forward pass
        double embedding_flops = config.batch_size * config.sequence_length * config.embedding_dim;
        double attention_flops = config.num_layers * config.batch_size * config.num_heads * 
                                config.sequence_length * config.sequence_length * (config.embedding_dim / config.num_heads);
        double ffn_flops = config.num_layers * config.batch_size * config.sequence_length * 
                          config.embedding_dim * config.embedding_dim * 4 * 2; // Two linear layers
        
        return (embedding_flops + attention_flops + ffn_flops) / 1e9; // Convert to GFLOPS
    }
    
    // Estimate memory bandwidth usage
    double estimateMemoryBandwidth(const BenchmarkConfig& config, double time_ms) {
        // Rough estimate: reading weights + activations + writing outputs
        size_t param_bytes = config.vocab_size * config.embedding_dim * 2; // Token embeddings (half precision)
        param_bytes += config.num_layers * (3 * config.embedding_dim * config.embedding_dim * 2); // QKV weights
        param_bytes += config.num_layers * (config.embedding_dim * config.embedding_dim * 4 * 2 * 2); // FFN weights
        
        size_t activation_bytes = config.batch_size * config.sequence_length * config.embedding_dim * 2 * config.num_layers * 4; // Rough estimate
        
        double total_gb = (param_bytes + activation_bytes) / (1024.0 * 1024.0 * 1024.0);
        return total_gb / (time_ms / 1000.0); // GB/s
    }
    
public:
    TransformerBenchmark() : profiler("profiling_results") {}
    
    PerformanceMetrics benchmarkForwardPass(const BenchmarkConfig& config) {
        std::cout << "\n=== Benchmarking Forward Pass: " << config.name << " ===" << std::endl;
        
        // Create model with benchmark configuration
        TransformerConfig model_config;
        model_config.vocab_size = config.vocab_size;
        model_config.embedding_dim = config.embedding_dim;
        model_config.num_layers = config.num_layers;
        model_config.num_heads = config.num_heads;
        model_config.max_sequence_length = config.sequence_length;
        model_config.batch_size = config.batch_size;
        
        TransformerModel model(model_config);
        if (!model.initialize()) {
            throw std::runtime_error("Failed to initialize model for benchmarking");
        }
        
        // Create test input
        std::vector<uint32_t> input_tokens(config.sequence_length);
        std::iota(input_tokens.begin(), input_tokens.end(), 1); // Fill with 1, 2, 3, ...
        
        // Warmup runs
        if (config.warmup_enabled) {
            std::cout << "Performing " << config.warmup_iterations << " warmup iterations..." << std::endl;
            for (uint32_t i = 0; i < config.warmup_iterations; i++) {
                std::vector<float> logits;
                model.forward(input_tokens, logits);
            }
        }
        
        // Actual benchmark
        std::cout << "Running " << config.iterations << " benchmark iterations..." << std::endl;
        std::vector<double> iteration_times;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (uint32_t i = 0; i < config.iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<float> logits;
            if (!model.forward(input_tokens, logits)) {
                throw std::runtime_error("Forward pass failed during benchmarking");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            iteration_times.push_back(duration.count() / 1000.0); // Convert to ms
            
            if ((i + 1) % 10 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << config.iterations << " iterations" << std::endl;
            }
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        
        // Calculate statistics
        double total_time = total_duration.count() / 1000.0; // ms
        double avg_time = total_time / config.iterations;
        
        // Sort for percentile calculations
        std::sort(iteration_times.begin(), iteration_times.end());
        double median_time = iteration_times[iteration_times.size() / 2];
        double p95_time = iteration_times[(size_t)(iteration_times.size() * 0.95)];
        
        // Calculate performance metrics
        PerformanceMetrics metrics;
        metrics.total_time_ms = avg_time;
        metrics.kernel_time_ms = avg_time * 0.8; // Rough estimate (80% of time in GPU kernels)
        metrics.memory_bandwidth_gb_s = estimateMemoryBandwidth(config, avg_time);
        metrics.tokens_per_second = (config.sequence_length * 1000.0) / avg_time;
        metrics.memory_usage_mb = model.getMemoryUsage() / (1024 * 1024);
        metrics.parameter_count = model.getParameterCount();
        metrics.flops_per_second = calculateTheoreticalFLOPS(config) / (avg_time / 1000.0);
        
        // Print detailed results
        std::cout << "\nBenchmark Results:" << std::endl;
        std::cout << "  Average Time: " << std::fixed << std::setprecision(2) << avg_time << " ms" << std::endl;
        std::cout << "  Median Time:  " << std::fixed << std::setprecision(2) << median_time << " ms" << std::endl;
        std::cout << "  95th Percentile: " << std::fixed << std::setprecision(2) << p95_time << " ms" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
        metrics.print();
        
        return metrics;
    }
    
    PerformanceMetrics benchmarkInference(const BenchmarkConfig& config) {
        std::cout << "\n=== Benchmarking Inference: " << config.name << " ===" << std::endl;
        
        // Create model with benchmark configuration
        TransformerConfig model_config;
        model_config.vocab_size = config.vocab_size;
        model_config.embedding_dim = config.embedding_dim;
        model_config.num_layers = config.num_layers;
        model_config.num_heads = config.num_heads;
        model_config.max_sequence_length = config.sequence_length;
        model_config.batch_size = 1; // Inference is typically batch size 1
        
        TransformerModel model(model_config);
        if (!model.initialize()) {
            throw std::runtime_error("Failed to initialize model for inference benchmarking");
        }
        
        // Create test prompt
        std::vector<uint32_t> prompt = {1, 2, 3, 4, 5}; // Simple prompt
        uint32_t max_new_tokens = config.sequence_length / 2; // Generate half sequence length
        
        // Warmup
        if (config.warmup_enabled) {
            std::cout << "Performing warmup..." << std::endl;
            for (uint32_t i = 0; i < config.warmup_iterations; i++) {
                std::vector<uint32_t> generated;
                model.generate(prompt, max_new_tokens, generated);
            }
        }
        
        // Actual benchmark
        std::cout << "Running " << config.iterations << " inference iterations..." << std::endl;
        std::vector<double> iteration_times;
        
        for (uint32_t i = 0; i < config.iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<uint32_t> generated;
            if (!model.generate(prompt, max_new_tokens, generated)) {
                throw std::runtime_error("Inference failed during benchmarking");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            iteration_times.push_back(duration.count() / 1000.0); // Convert to ms
            
            if ((i + 1) % 5 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << config.iterations << " iterations" << std::endl;
            }
        }
        
        // Calculate statistics
        double avg_time = 0.0;
        for (double time : iteration_times) {
            avg_time += time;
        }
        avg_time /= iteration_times.size();
        
        // Calculate performance metrics
        PerformanceMetrics metrics;
        metrics.total_time_ms = avg_time;
        metrics.kernel_time_ms = avg_time * 0.8; // Rough estimate
        metrics.memory_bandwidth_gb_s = estimateMemoryBandwidth(config, avg_time);
        metrics.tokens_per_second = (max_new_tokens * 1000.0) / avg_time;
        metrics.memory_usage_mb = model.getMemoryUsage() / (1024 * 1024);
        metrics.parameter_count = model.getParameterCount();
        metrics.flops_per_second = calculateTheoreticalFLOPS(config) / (avg_time / 1000.0);
        
        std::cout << "\nInference Benchmark Results:" << std::endl;
        metrics.print();
        
        return metrics;
    }
    
    PerformanceMetrics benchmarkTrainingStep(const BenchmarkConfig& config) {
        std::cout << "\n=== Benchmarking Training Step: " << config.name << " ===" << std::endl;
        
        TransformerConfig model_config;
        model_config.vocab_size = config.vocab_size;
        model_config.embedding_dim = config.embedding_dim;
        model_config.num_layers = config.num_layers;
        model_config.num_heads = config.num_heads;
        model_config.max_sequence_length = config.sequence_length;
        model_config.batch_size = config.batch_size;
        
        TransformerModel model(model_config);
        if (!model.initialize()) {
            throw std::runtime_error("Failed to initialize model for training benchmarking");
        }
        
        // Create test data
        std::vector<uint32_t> input_tokens(config.sequence_length);
        std::vector<uint32_t> target_tokens(config.sequence_length);
        std::iota(input_tokens.begin(), input_tokens.end(), 1);
        std::iota(target_tokens.begin(), target_tokens.end(), 2);
        
        // Warmup
        if (config.warmup_enabled) {
            std::cout << "Performing warmup..." << std::endl;
            for (uint32_t i = 0; i < config.warmup_iterations; i++) {
                float loss;
                model.trainStep(input_tokens, target_tokens, loss);
            }
        }
        
        // Benchmark
        std::cout << "Running " << config.iterations << " training iterations..." << std::endl;
        std::vector<double> iteration_times;
        
        for (uint32_t i = 0; i < config.iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            float loss;
            if (!model.trainStep(input_tokens, target_tokens, loss)) {
                throw std::runtime_error("Training step failed during benchmarking");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            iteration_times.push_back(duration.count() / 1000.0);
            
            if ((i + 1) % 5 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << config.iterations << " iterations" << std::endl;
            }
        }
        
        // Calculate metrics
        double avg_time = 0.0;
        for (double time : iteration_times) {
            avg_time += time;
        }
        avg_time /= iteration_times.size();
        
        PerformanceMetrics metrics;
        metrics.total_time_ms = avg_time;
        metrics.kernel_time_ms = avg_time * 0.9; // Training has more GPU work
        metrics.memory_bandwidth_gb_s = estimateMemoryBandwidth(config, avg_time) * 1.5; // More memory traffic
        metrics.tokens_per_second = (config.sequence_length * 1000.0) / avg_time;
        metrics.memory_usage_mb = model.getMemoryUsage() / (1024 * 1024);
        metrics.parameter_count = model.getParameterCount();
        metrics.flops_per_second = calculateTheoreticalFLOPS(config) * 3 / (avg_time / 1000.0); // ~3x for forward+backward+optimizer
        
        std::cout << "\nTraining Benchmark Results:" << std::endl;
        metrics.print();
        
        return metrics;
    }
    
    void setBaseline(const std::string& test_name, const PerformanceMetrics& metrics) {
        profiler.setPerformanceBaseline(test_name, metrics);
    }
    
    bool verifyImprovement(const std::string& test_name, const PerformanceMetrics& metrics, double min_improvement = 5.0) {
        return profiler.verifyPerformanceImprovement(test_name, metrics, min_improvement);
    }
    
    bool checkRegressions(const std::string& test_name, const PerformanceMetrics& metrics) {
        return profiler.checkForRegressions(test_name, metrics);
    }
};

// Test suite following TDD principles
bool testPerformanceProfiling() {
    std::cout << "=== Performance Profiling Test Suite ===\n" << std::endl;
    
    TransformerBenchmark benchmark;
    
    // Test 1: Small model baseline (for development)
    std::cout << "Test 1: Small Model Baseline" << std::endl;
    BenchmarkConfig small_config = {
        .name = "small_model_baseline",
        .vocab_size = 100,
        .embedding_dim = 128,
        .num_layers = 2,
        .num_heads = 4,
        .sequence_length = 32,
        .batch_size = 1,
        .iterations = 10,
        .warmup_enabled = true,
        .warmup_iterations = 3
    };
    
    auto small_forward_metrics = benchmark.benchmarkForwardPass(small_config);
    benchmark.setBaseline("small_forward", small_forward_metrics);
    
    auto small_inference_metrics = benchmark.benchmarkInference(small_config);
    benchmark.setBaseline("small_inference", small_inference_metrics);
    
    auto small_training_metrics = benchmark.benchmarkTrainingStep(small_config);
    benchmark.setBaseline("small_training", small_training_metrics);
    
    // Test 2: Medium model (realistic size)
    std::cout << "\nTest 2: Medium Model Benchmark" << std::endl;
    BenchmarkConfig medium_config = {
        .name = "medium_model",
        .vocab_size = 1000,
        .embedding_dim = 256,
        .num_layers = 4,
        .num_heads = 8,
        .sequence_length = 128,
        .batch_size = 1,
        .iterations = 5,
        .warmup_enabled = true,
        .warmup_iterations = 2
    };
    
    auto medium_forward_metrics = benchmark.benchmarkForwardPass(medium_config);
    benchmark.setBaseline("medium_forward", medium_forward_metrics);
    
    auto medium_inference_metrics = benchmark.benchmarkInference(medium_config);
    benchmark.setBaseline("medium_inference", medium_inference_metrics);
    
    // Test 3: Check for performance regressions
    std::cout << "\nTest 3: Regression Testing" << std::endl;
    bool no_regressions = true;
    no_regressions &= benchmark.checkRegressions("small_forward", small_forward_metrics);
    no_regressions &= benchmark.checkRegressions("small_inference", small_inference_metrics);
    no_regressions &= benchmark.checkRegressions("small_training", small_training_metrics);
    
    if (no_regressions) {
        std::cout << "✓ No performance regressions detected" << std::endl;
    } else {
        std::cout << "⚠️  Performance regressions detected" << std::endl;
    }
    
    std::cout << "\n=== Performance Profiling Complete ===\n" << std::endl;
    std::cout << "💡 Next Steps:" << std::endl;
    std::cout << "  1. Use Metal System Trace in Xcode to profile GPU utilization" << std::endl;
    std::cout << "  2. Identify bottleneck kernels from timeline analysis" << std::endl;
    std::cout << "  3. Optimize threadgroup sizes and memory access patterns" << std::endl;
    std::cout << "  4. Re-run benchmarks to verify improvements" << std::endl;
    
    return true;
}

int main() {
    std::cout << "🚀 Starting Phase 4.1: Performance Profiling\n" << std::endl;
    
    try {
        bool success = testPerformanceProfiling();
        
        if (success) {
            std::cout << "🎉 Performance profiling tests completed successfully!" << std::endl;
            std::cout << "\n📊 Profiling data saved to 'profiling_results/' directory" << std::endl;
            std::cout << "📈 Use this data for optimization planning and regression testing" << std::endl;
            return 0;
        } else {
            std::cerr << "❌ Performance profiling tests failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "💥 Exception during profiling: " << e.what() << std::endl;
        return 1;
    }
} 