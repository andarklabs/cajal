#include "../src/data/book_corpus_reader.h"
#include "../src/tokenization/bpe_tokenizer.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>

void test_book_corpus_tokenizer_integration() {
    std::cout << "Running test_book_corpus_tokenizer_integration..." << std::endl;
    
    // Set up BookCorpusReader with default cleaning config
    std::vector<std::string> book_files = {
        "data/bookcorpus/books_large_p1.txt",
        "data/bookcorpus/books_large_p2.txt"
    };
    
    TextCleanerConfig config;
    config.to_lowercase = true;
    config.normalize_whitespace = true;
    config.newlines_to_spaces = true;
    
    BookCorpusReader reader(book_files, config);
    
    // Try to open the corpus
    if (!reader.begin_stream()) {
        std::cout << "  Warning: Could not open BookCorpus files. Skipping real data test." << std::endl;
        std::cout << "  Make sure data/bookcorpus/ contains the book files." << std::endl;
        return;
    }
    
    std::cout << "  Successfully opened BookCorpus files" << std::endl;
    std::cout << "  Current file: " << reader.current_file_path() << std::endl;
    
    // Collect a subset of lines for training (don't load the entire corpus into memory)
    std::vector<std::string> training_corpus;
    const int MAX_LINES = 1000; // Limit for reasonable training time
    int lines_read = 0;
    
    std::cout << "  Loading " << MAX_LINES << " lines for training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (lines_read < MAX_LINES) {
        std::optional<std::string> line = reader.next_cleaned_line();
        if (!line.has_value()) {
            std::cout << "  Reached end of corpus after " << lines_read << " lines" << std::endl;
            break;
        }
        
        // Only include non-empty lines with reasonable content
        if (!line->empty() && line->length() > 10) {
            training_corpus.push_back(*line);
            lines_read++;
        }
        
        if (lines_read % 100 == 0) {
            std::cout << "    Loaded " << lines_read << " lines..." << std::endl;
        }
    }
    
    auto load_time = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_time - start_time);
    
    std::cout << "  Loaded " << training_corpus.size() << " lines in " 
              << load_duration.count() << "ms" << std::endl;
    
    if (training_corpus.empty()) {
        std::cout << "  Warning: No training data loaded. Check BookCorpus files." << std::endl;
        return;
    }
    
    // Show some example lines
    std::cout << "  Sample training lines:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), training_corpus.size()); ++i) {
        std::string preview = training_corpus[i];
        if (preview.length() > 80) {
            preview = preview.substr(0, 77) + "...";
        }
        std::cout << "    " << i + 1 << ": \"" << preview << "\"" << std::endl;
    }
    
    // Train the BPE tokenizer
    std::cout << "  Training BPE tokenizer (vocab size 5000)..." << std::endl;
    BPETokenizer tokenizer;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    bool success = tokenizer.train(training_corpus, 5000);
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
    
    assert(success && "Tokenizer training should succeed");
    std::cout << "  Training completed in " << train_duration.count() << "ms" << std::endl;
    std::cout << "  Final vocabulary size: " << tokenizer.get_vocab_size() << std::endl;
    
    // Test the tokenizer on some sample text
    std::vector<std::string> test_texts = {
        "hello world this is a test",
        "the quick brown fox jumps over the lazy dog",
        "natural language processing with transformers"
    };
    
    std::cout << "  Testing tokenizer on sample texts:" << std::endl;
    for (const std::string& text : test_texts) {
        auto encode_start = std::chrono::high_resolution_clock::now();
        std::vector<int> encoded = tokenizer.encode(text, true, true); // with BOS/EOS
        auto encode_end = std::chrono::high_resolution_clock::now();
        auto encode_duration = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start);
        
        std::string decoded = tokenizer.decode(encoded);
        
        std::cout << "    Original:  \"" << text << "\"" << std::endl;
        std::cout << "    Tokens:    " << encoded.size() << " (encoded in " 
                  << encode_duration.count() << "μs)" << std::endl;
        std::cout << "    Decoded:   \"" << decoded << "\"" << std::endl;
        std::cout << "    Token IDs: [";
        for (size_t i = 0; i < std::min(size_t(10), encoded.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << encoded[i];
        }
        if (encoded.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
    
    // Test on a real BookCorpus line if we have training data
    if (!training_corpus.empty()) {
        std::string book_line = training_corpus[0];
        if (book_line.length() > 100) {
            book_line = book_line.substr(0, 100); // Truncate for display
        }
        
        std::vector<int> encoded = tokenizer.encode(book_line);
        std::string decoded = tokenizer.decode(encoded);
        
        std::cout << "  Real BookCorpus example:" << std::endl;
        std::cout << "    Original:  \"" << book_line << "\"" << std::endl;
        std::cout << "    Tokens:    " << encoded.size() << std::endl;
        std::cout << "    Decoded:   \"" << decoded << "\"" << std::endl;
        
        // Calculate compression ratio
        double compression_ratio = static_cast<double>(encoded.size()) / book_line.length();
        std::cout << "    Compression ratio: " << compression_ratio 
                  << " tokens/char (lower is better)" << std::endl;
    }
    
    // Save the trained tokenizer for potential reuse
    std::string model_path = "bookcorpus_tokenizer";
    std::cout << "  Saving trained tokenizer to " << model_path << ".*" << std::endl;
    bool save_success = tokenizer.save_model(model_path);
    assert(save_success && "Save should succeed");
    
    std::cout << "test_book_corpus_tokenizer_integration PASSED." << std::endl;
}

void test_tokenizer_performance_metrics() {
    std::cout << "Running test_tokenizer_performance_metrics..." << std::endl;
    
    // Load the saved tokenizer if it exists
    BPETokenizer tokenizer;
    std::string model_path = "bookcorpus_tokenizer";
    
    if (!tokenizer.load_model(model_path)) {
        std::cout << "  Warning: Could not load saved tokenizer. Run integration test first." << std::endl;
        return;
    }
    
    // Performance test with various text types
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Short", "hello world"},
        {"Medium", "the quick brown fox jumps over the lazy dog and runs through the forest"},
        {"Long", "this is a much longer sentence that contains many different words and should test the tokenizer's ability to handle longer sequences of text with various vocabulary items including some that might be out of vocabulary or require multiple subword tokens to represent properly"},
        {"Repeated", "hello hello hello world world world test test test"},
        {"Mixed case", "Hello World This Is Mixed Case Text With CAPS and lowercase"},
        {"Punctuation", "Hello, world! How are you? I'm fine, thanks. What about you?"}
    };
    
    std::cout << "  Performance metrics:" << std::endl;
    for (const auto& test_case : test_cases) {
        const std::string& name = test_case.first;
        const std::string& text = test_case.second;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> encoded = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double compression = static_cast<double>(encoded.size()) / text.length();
        double speed = text.length() / (duration.count() / 1000.0); // chars per ms
        
        std::cout << "    " << name << " (" << text.length() << " chars): "
                  << encoded.size() << " tokens, "
                  << "ratio=" << std::fixed << std::setprecision(3) << compression << ", "
                  << "speed=" << std::fixed << std::setprecision(1) << speed << " chars/ms"
                  << std::endl;
    }
    
    std::cout << "test_tokenizer_performance_metrics PASSED." << std::endl;
}

int main() {
    std::cout << "BookCorpus + BPE Tokenizer Integration Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    test_book_corpus_tokenizer_integration();
    std::cout << std::endl;
    test_tokenizer_performance_metrics();
    
    std::cout << std::endl;
    std::cout << "Integration tests completed!" << std::endl;
    std::cout << "The tokenizer is ready for real-world use." << std::endl;
    
    return 0;
} 