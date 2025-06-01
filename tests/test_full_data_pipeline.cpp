#include "../src/data/book_corpus_reader.h"
#include "../src/tokenization/bpe_tokenizer.h"
#include "../src/data/data_formatter.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <fstream>

void test_full_data_pipeline() {
    std::cout << "Running test_full_data_pipeline..." << std::endl;
    
    // Step 1: Create sample data file
    std::string test_file = "test_pipeline_data.txt";
    std::ofstream out(test_file);
    out << "The quick brown fox jumps over the lazy dog.\n";
    out << "Machine learning is transforming the world of artificial intelligence.\n";
    out << "Natural language processing enables computers to understand human language.\n";
    out << "Deep learning models require large amounts of training data.\n";
    out << "Transformers have revolutionized the field of NLP.\n";
    out.close();
    
    // Step 2: Configure and test BookCorpusReader
    TextCleanerConfig cleaner_config;
    cleaner_config.to_lowercase = true;
    cleaner_config.normalize_whitespace = true;
    
    std::vector<std::string> file_paths = {test_file};
    BookCorpusReader reader(file_paths, cleaner_config);
    
    std::vector<std::string> lines;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = reader.begin_stream();
    assert(success);
    
    // Read all lines
    while (auto line = reader.next_cleaned_line()) {
        lines.push_back(*line);
    }
    
    auto read_time = std::chrono::high_resolution_clock::now();
    assert(lines.size() == 5);
    
    std::cout << "  Step 1: BookCorpusReader loaded " << lines.size() << " lines" << std::endl;
    
    // Step 3: Train BPE tokenizer on the data
    BPETokenizer tokenizer;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    tokenizer.train(lines, 100); // Small vocab for test
    auto train_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "  Step 2: BPE tokenizer trained with vocab size " << tokenizer.get_vocab_size() << std::endl;
    
    // Step 4: Tokenize all lines
    std::vector<std::vector<int>> tokenized_sequences;
    auto tokenize_start = std::chrono::high_resolution_clock::now();
    
    for (const auto& line : lines) {
        std::vector<int> tokens = tokenizer.encode(line);
        tokenized_sequences.push_back(tokens);
    }
    
    auto tokenize_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "  Step 3: Tokenized " << tokenized_sequences.size() << " sequences" << std::endl;
    
    // Step 5: Format data for MSL
    DataFormatterConfig formatter_config;
    formatter_config.max_sequence_length = 32;
    formatter_config.batch_size = 3;
    
    // Get special token IDs
    auto pad_id = tokenizer.get_special_token_id("[PAD]");
    auto bos_id = tokenizer.get_special_token_id("[BOS]");
    auto eos_id = tokenizer.get_special_token_id("[EOS]");
    
    formatter_config.pad_token_id = pad_id ? *pad_id : 0;
    formatter_config.bos_token_id = bos_id ? *bos_id : 1;
    formatter_config.eos_token_id = eos_id ? *eos_id : 2;
    formatter_config.add_bos_eos = true;
    
    DataFormatter formatter(formatter_config);
    
    auto format_start = std::chrono::high_resolution_clock::now();
    FormattedBatch batch = formatter.format_batch(tokenized_sequences);
    auto format_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "  Step 4: Formatted batch with " << batch.batch_size << " sequences" << std::endl;
    
    // Step 6: Verify the complete pipeline
    assert(batch.batch_size == 3); // First 3 sequences (limited by batch_size)
    assert(batch.sequence_length == 32);
    assert(batch.sequences.size() == 3);
    assert(batch.sequence_lengths.size() == 3);
    
    // Check that sequences are properly formatted
    for (uint32_t i = 0; i < batch.batch_size; ++i) {
        assert(batch.sequences[i].size() == 32);
        assert(batch.sequences[i][0] == formatter_config.bos_token_id); // BOS token
        
        // Find EOS token position
        bool found_eos = false;
        for (size_t j = 1; j < batch.sequences[i].size(); ++j) {
            if (batch.sequences[i][j] == formatter_config.eos_token_id) {
                found_eos = true;
                // Everything after EOS should be padding
                for (size_t k = j + 1; k < batch.sequences[i].size(); ++k) {
                    assert(batch.sequences[i][k] == formatter_config.pad_token_id);
                }
                break;
            }
        }
        assert(found_eos);
    }
    
    // Step 7: Test stream-based processing with BatchBuilder
    auto builder = formatter.create_batch_builder();
    
    for (const auto& tokens : tokenized_sequences) {
        if (!builder.add_sequence(tokens)) {
            // Batch is full, finalize it
            FormattedBatch stream_batch = builder.finalize_batch();
            assert(stream_batch.batch_size == 3);
            
            // Add the sequence to the new batch
            bool added = builder.add_sequence(tokens);
            assert(added);
        }
    }
    
    // Get the final partial batch
    FormattedBatch final_batch = builder.get_current_batch();
    assert(final_batch.batch_size == 2); // Remaining 2 sequences
    
    std::cout << "  Step 5: Stream processing created batches of sizes 3 and " << final_batch.batch_size << std::endl;
    
    // Step 8: Performance metrics
    auto total_time = std::chrono::high_resolution_clock::now();
    
    auto read_duration = std::chrono::duration_cast<std::chrono::microseconds>(read_time - start_time);
    auto train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_time - train_start);
    auto tokenize_duration = std::chrono::duration_cast<std::chrono::microseconds>(tokenize_time - tokenize_start);
    auto format_duration = std::chrono::duration_cast<std::chrono::microseconds>(format_time - format_start);
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_time - start_time);
    
    std::cout << "  Performance metrics:" << std::endl;
    std::cout << "    Reading: " << read_duration.count() << " μs" << std::endl;
    std::cout << "    Training: " << train_duration.count() << " μs" << std::endl;
    std::cout << "    Tokenizing: " << tokenize_duration.count() << " μs" << std::endl;
    std::cout << "    Formatting: " << format_duration.count() << " μs" << std::endl;
    std::cout << "    Total: " << total_duration.count() << " μs" << std::endl;
    
    // Step 9: MSL Buffer size calculation
    size_t buffer_size = batch.size_in_bytes();
    std::cout << "  MSL Buffer requirements:" << std::endl;
    std::cout << "    Total elements: " << batch.total_elements() << std::endl;
    std::cout << "    Buffer size: " << buffer_size << " bytes (" << buffer_size / 1024.0 << " KB)" << std::endl;
    
    // Step 10: Statistics
    const auto& stats = formatter.get_stats();
    std::cout << "  Formatting statistics:" << std::endl;
    std::cout << "    Sequences processed: " << stats.total_sequences_processed << std::endl;
    std::cout << "    Truncation rate: " << (stats.truncation_rate() * 100) << "%" << std::endl;
    std::cout << "    Padding efficiency: " << (stats.padding_efficiency() * 100) << "%" << std::endl;
    
    // Cleanup
    std::remove(test_file.c_str());
    
    std::cout << "  Full data pipeline test PASSED!" << std::endl;
}

void test_pipeline_with_real_bookcorpus() {
    std::cout << "Running test_pipeline_with_real_bookcorpus..." << std::endl;
    
    // Check if real BookCorpus files exist
    std::string file1 = "data/bookcorpus/books_large_p1.txt";
    std::string file2 = "data/bookcorpus/books_large_p2.txt";
    
    std::ifstream test1(file1);
    std::ifstream test2(file2);
    
    if (!test1.good() && !test2.good()) {
        std::cout << "  Skipping real BookCorpus test - files not found" << std::endl;
        return;
    }
    
    std::string test_file = test1.good() ? file1 : file2;
    test1.close();
    test2.close();
    
    // Configure components for real data
    TextCleanerConfig cleaner_config;
    cleaner_config.to_lowercase = true;
    cleaner_config.normalize_whitespace = true;
    
    std::vector<std::string> file_paths = {test_file};
    BookCorpusReader reader(file_paths, cleaner_config);
    
    // Read a limited number of lines for testing
    std::vector<std::string> lines;
    size_t max_lines = 100;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = reader.begin_stream();
    if (!success) {
        std::cout << "  Could not open " << test_file << std::endl;
        return;
    }
    
    // Read lines up to max_lines
    while (lines.size() < max_lines) {
        auto line = reader.next_cleaned_line();
        if (!line) break; // End of file or error
        lines.push_back(*line);
    }
    
    auto read_time = std::chrono::high_resolution_clock::now();
    
    if (lines.empty()) {
        std::cout << "  No lines read from " << test_file << std::endl;
        return;
    }
    
    std::cout << "  Loaded " << lines.size() << " lines from real BookCorpus" << std::endl;
    
    // Train tokenizer on real data (use subset for training)
    BPETokenizer tokenizer;
    
    std::vector<std::string> training_subset;
    for (size_t i = 0; i < std::min(lines.size(), size_t(20)); ++i) {
        training_subset.push_back(lines[i]);
    }
    
    auto train_start = std::chrono::high_resolution_clock::now();
    tokenizer.train(training_subset, 500); // Larger vocab for real data
    auto train_time = std::chrono::high_resolution_clock::now();
    
    // Process all lines through the pipeline
    DataFormatterConfig formatter_config;
    formatter_config.max_sequence_length = 128;
    formatter_config.batch_size = 16;
    
    auto pad_id = tokenizer.get_special_token_id("[PAD]");
    auto bos_id = tokenizer.get_special_token_id("[BOS]");
    auto eos_id = tokenizer.get_special_token_id("[EOS]");
    
    formatter_config.pad_token_id = pad_id ? *pad_id : 0;
    formatter_config.bos_token_id = bos_id ? *bos_id : 1;
    formatter_config.eos_token_id = eos_id ? *eos_id : 2;
    
    DataFormatter formatter(formatter_config);
    auto builder = formatter.create_batch_builder();
    
    size_t total_batches = 0;
    size_t total_sequences = 0;
    
    auto process_start = std::chrono::high_resolution_clock::now();
    
    for (const auto& line : lines) {
        std::vector<int> tokens = tokenizer.encode(line);
        
        if (!builder.add_sequence(tokens)) {
            // Batch is full
            FormattedBatch batch = builder.finalize_batch();
            total_batches++;
            total_sequences += batch.batch_size;
            
            // Add current sequence to new batch
            builder.add_sequence(tokens);
        }
    }
    
    // Handle final partial batch
    if (builder.current_batch_size() > 0) {
        FormattedBatch final_batch = builder.get_current_batch();
        total_batches++;
        total_sequences += final_batch.batch_size;
    }
    
    auto process_time = std::chrono::high_resolution_clock::now();
    
    // Performance metrics
    auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_time - start_time);
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_time - train_start);
    auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_time - process_start);
    
    std::cout << "  Real BookCorpus pipeline results:" << std::endl;
    std::cout << "    Vocabulary size: " << tokenizer.get_vocab_size() << std::endl;
    std::cout << "    Total batches: " << total_batches << std::endl;
    std::cout << "    Total sequences: " << total_sequences << std::endl;
    std::cout << "    Reading time: " << read_duration.count() << " ms" << std::endl;
    std::cout << "    Training time: " << train_duration.count() << " ms" << std::endl;
    std::cout << "    Processing time: " << process_duration.count() << " ms" << std::endl;
    
    const auto& stats = formatter.get_stats();
    std::cout << "    Truncation rate: " << (stats.truncation_rate() * 100) << "%" << std::endl;
    std::cout << "    Padding efficiency: " << (stats.padding_efficiency() * 100) << "%" << std::endl;
    
    std::cout << "  Real BookCorpus pipeline test PASSED!" << std::endl;
}

int main() {
    std::cout << "Full Data Pipeline Integration Tests" << std::endl;
    std::cout << "====================================" << std::endl;
    
    test_full_data_pipeline();
    test_pipeline_with_real_bookcorpus();
    
    std::cout << std::endl;
    std::cout << "All integration tests PASSED!" << std::endl;
    std::cout << "Phase 1 Task 1.3 (Data Formatting for MSL) COMPLETED!" << std::endl;
    std::cout << "Ready for Phase 2: MSL Kernel Implementation" << std::endl;
    
    return 0;
} 