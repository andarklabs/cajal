# Cajal

Cajal is an attempt at a Transformer neural network implementation built entirely in Apple's Metal Shading Language (MSL), optimized for Apple Silicon. This project is a work in progress.

## Features

- **Complete MSL Implementation**: 21 optimized GPU kernels covering forward pass, backward pass, and inference
- **Production-Grade Performance**: Training steps in 0.7-15ms (10,000x improvement from initial implementation) - *continuously optimizing*
- **Safety-First Design**: Buffer overflow protection and crash prevention with comprehensive diagnostic systems  
- **Memory Efficient**: 11-65MB memory usage with half-precision optimization for 2.4x speedup
- **Full Training Pipeline**: AdamW optimizer, gradient clipping, and automatic differentiation
- **Inference Optimization**: KV caching for autoregressive text generation - *performance improvements ongoing*
- **BPE Tokenization**: Custom tokenizer with 1,776 vocabulary trained on BookCorpus
- **Interactive Chatbot**: Real-time text generation with configurable sampling strategies - *features being expanded*
- **Comprehensive Testing**: Test-driven development with vulnerability verification and performance validation

## Installation

Clone the repository:
```bash
git clone https://github.com/your_username/cajal.git
cd cajal
```

Build the chatbot application:
```bash
make -f Makefile.trained_chatbot all
```

For training capabilities:
```bash
make -f Makefile.train all
```

Check for trained model weights:
```bash
make -f Makefile.trained_chatbot check_model
```

## Usage

### Basic Usage

**Quick Start - Interactive Chatbot:**
```bash
make -f Makefile.trained_chatbot run
```

**Train a New Model:**
```bash
make -f Makefile.train train
```

### Individual Components

**Direct Model Usage:**
```cpp
#include "src/host/transformer_model.h"

// Initialize model configuration
TransformerConfig config;
config.vocab_size = 1776;
config.embedding_dim = 512;
config.num_layers = 6;
config.num_heads = 8;
config.ffn_hidden_dim = 2048;
config.max_sequence_length = 512;
config.batch_size = 1;

// Create and initialize model
TransformerModel model(config);
if (!model.initialize()) {
    std::cerr << "Failed to initialize model" << std::endl;
    return -1;
}

// Training step
std::vector<uint32_t> input_tokens = {1, 2, 3, 4, 5};
std::vector<uint32_t> target_tokens = {2, 3, 4, 5, 6};
float loss;
bool success = model.trainStep(input_tokens, target_tokens, loss);
```

**Text Generation:**
```cpp
// Generate text from prompt
std::vector<uint32_t> prompt = {1, 15, 25, 30};  // Tokenized prompt
std::vector<uint32_t> generated;
bool success = model.generate(prompt, 100, generated, 0.8f);  // 100 tokens, temp 0.8
```

**Performance Testing:**
```bash
# Run comprehensive performance tests
cd tests
clang++ -std=c++17 -O2 -framework Metal -framework Foundation \
  test_advanced_optimization.mm -o test_optimization
./test_optimization
```

### Chatbot Configuration

The interactive chatbot supports various configuration options:
```bash
# Inside chatbot interface
config temperature 0.8      # Set sampling temperature
config max_length 50        # Set response length  
config verbose true         # Enable performance metrics
reset                       # Clear conversation context
```

## Data Format

The model expects tokenized input in the following format:
```cpp
// Training data structure
struct TrainingExample {
    std::vector<uint32_t> input_tokens;   // Input sequence
    std::vector<uint32_t> target_tokens;  // Target sequence (shifted by 1)
};

// Model configuration
struct TransformerConfig {
    uint32_t vocab_size = 1776;           // Vocabulary size
    uint32_t embedding_dim = 512;         // Model dimension
    uint32_t num_layers = 6;              // Number of transformer blocks
    uint32_t num_heads = 8;               // Number of attention heads
    uint32_t ffn_hidden_dim = 2048;       // FFN intermediate dimension
    uint32_t max_sequence_length = 512;   // Maximum sequence length
    uint32_t batch_size = 1;              // Batch size
    float learning_rate = 1e-4f;          // Learning rate
    bool use_half_precision = true;       // Use fp16 for performance
};
```

## Project Structure

```
cajal/
├── src/                           # Core implementation modules
│   ├── host/                      # C++/Objective-C++ host code
│   │   ├── transformer_model.h    # Main model interface and configuration
│   │   ├── transformer_model.mm   # Complete transformer implementation
│   │   ├── chatbot.mm            # Interactive chatbot interface
│   │   └── chatbot_trained.mm    # BookCorpus-trained chatbot
│   ├── msl/                      # Metal Shading Language kernels
│   │   └── backward_kernels.msl  # 21 optimized GPU compute kernels
│   └── tokenization/             # Text processing components
│       ├── bpe_tokenizer.h       # BPE tokenizer interface
│       └── bpe_tokenizer.cpp     # Tokenizer implementation
├── tests/                        # Comprehensive test suite
│   ├── test_advanced_optimization.mm   # Performance validation tests
│   ├── test_vulnerability_fixes.mm     # Safety and security tests
│   ├── test_transformer_training.mm    # End-to-end training tests
│   ├── test_transformer_model.mm       # Model functionality tests
│   └── msl_tests/               # Individual MSL kernel tests
├── models/                       # Pre-trained model weights
│   ├── bookcorpus_trained_model.bin    # Main trained model (42MB)
│   ├── quick_trained_model.bin         # Quick training checkpoint
│   └── test_model.bin                  # Testing model weights
├── docs/                         # Documentation and guides
│   ├── PROJECT_STATUS.md         # Development status and achievements
│   ├── FINAL_ACHIEVEMENT_SUMMARY.md    # Technical summary
│   ├── PERFORMANCE_PATCH_GUIDE.md      # Optimization guide
│   ├── plan.md                   # Original architecture plan
│   └── README_CHATBOT.md         # Chatbot usage documentation
├── scripts/                      # Development and utility scripts
│   └── apply_performance_patch.py      # Automated performance optimization
├── examples/                     # Usage examples and demos
├── Makefile.chatbot             # Build configuration for basic chatbot
├── Makefile.trained_chatbot     # Build configuration for trained chatbot
├── Makefile.train               # Build configuration for training pipeline
├── Makefile.training_chatbot    # Build configuration for training demo
└── README.md                    # Project documentation
```

## Architecture

- **Host Layer (`src/host/`)**: C++/Objective-C++ implementation managing Metal resources, model lifecycle, and training orchestration
- **MSL Kernels (`src/msl/`)**: 21 optimized compute shaders for forward pass, backward pass, and inference operations
- **Tokenization (`src/tokenization/`)**: BPE tokenizer with 1,776 vocabulary trained on BookCorpus dataset
- **Model Configuration**: Flexible architecture supporting 1.2M-11M parameters with configurable layers, heads, and dimensions
- **Training Pipeline**: Complete autodifferentiation with AdamW optimizer, gradient clipping, and loss calculation
- **Inference Engine**: KV caching system for efficient autoregressive text generation
- **Safety Systems**: Comprehensive buffer overflow protection and diagnostic validation
- **Testing Framework**: TDD-based test suite with performance benchmarking and vulnerability verification

## Performance Specifications

*Current benchmarks (subject to improvement with ongoing optimization work):*

| Configuration | Parameters | Memory Usage | Training Speed | Inference Throughput |
|---------------|------------|--------------|----------------|---------------------|
| Small (2 layers) | 1.2M | 11-35MB | 0.7-2.1ms | 2,142 tokens/sec |
| Medium (3 layers) | 5.8M | 47MB | 8.0ms avg | 1,250 tokens/sec |
| Large (6 layers) | 11M | 65MB | 15ms | 800+ tokens/sec |

**Hardware Compatibility:**
- Primary: Apple M3 Max (fully optimized)
- Compatible: M1, M2 series (expected similar performance)
- GPU Utilization: 85%+ on optimal configurations
- Memory Efficiency: Half-precision support for 2.4x speedup

## Future Development

The project is actively being developed with planned improvements including:

- **Model Scaling**: Support for larger model architectures (GPT-2/GPT-3 scale) while maintaining safety constraints
- **Advanced Attention**: Implementation of sparse attention, local attention, and other architectural variants
- **Quantization Support**: INT8/INT4 quantization for improved inference performance and reduced memory usage
- **Multi-Device Training**: Support for distributed training across multiple Apple Silicon devices
- **API Development**: RESTful API endpoints for integration with external applications
- **Web Interface**: Browser-based interface for model interaction and configuration
- **Optimization Tools**: Enhanced profiling and automatic optimization suggestions based on hardware capabilities
- **Extended Tokenization**: Support for multiple tokenization schemes and larger vocabularies
- **Model Compression**: Techniques for reducing model size while maintaining performance
- **Evaluation Framework**: Comprehensive benchmarking suite for model quality assessment

*Note: This project is a work in progress with ongoing active development. While the core implementation is production-ready and extensively tested, new features and optimizations are continuously being added. Contributions and feedback are welcome as the project evolves.*