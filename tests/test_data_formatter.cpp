#include "../src/data/data_formatter.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

void test_data_formatter_basic_functionality() {
    std::cout << "Running test_data_formatter_basic_functionality..." << std::endl;
    
    // Test basic configuration
    DataFormatterConfig config;
    config.max_sequence_length = 10;
    config.batch_size = 3;
    config.pad_token_id = 0;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.add_bos_eos = true;
    
    DataFormatter formatter(config);
    
    // Test single sequence formatting - short sequence with padding
    std::vector<int> short_tokens = {100, 101, 102};
    FormattedSequence formatted = formatter.format_sequence(short_tokens);
    
    // Should be: [BOS, 100, 101, 102, EOS, PAD, PAD, PAD, PAD, PAD]
    assert(formatted.token_ids.size() == 10);
    assert(formatted.token_ids[0] == 1); // BOS
    assert(formatted.token_ids[1] == 100);
    assert(formatted.token_ids[2] == 101);
    assert(formatted.token_ids[3] == 102);
    assert(formatted.token_ids[4] == 2); // EOS
    assert(formatted.token_ids[5] == 0); // PAD
    assert(formatted.actual_length == 5); // BOS + 3 tokens + EOS
    assert(!formatted.was_truncated);
    
    std::cout << "  Basic padding test PASSED" << std::endl;
}

void test_data_formatter_truncation() {
    std::cout << "Running test_data_formatter_truncation..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 5;
    config.pad_token_id = 0;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.add_bos_eos = true;
    config.truncate_long_sequences = true;
    
    DataFormatter formatter(config);
    
    // Test long sequence that needs truncation
    std::vector<int> long_tokens = {100, 101, 102, 103, 104, 105, 106, 107};
    FormattedSequence formatted = formatter.format_sequence(long_tokens);
    
    // Should be truncated to: [BOS, 100, 101, 102, EOS] (length 5)
    assert(formatted.token_ids.size() == 5);
    assert(formatted.token_ids[0] == 1); // BOS
    assert(formatted.token_ids[1] == 100);
    assert(formatted.token_ids[2] == 101);
    assert(formatted.token_ids[3] == 102);
    assert(formatted.token_ids[4] == 2); // EOS (preserved)
    assert(formatted.actual_length == 5);
    assert(formatted.was_truncated);
    
    std::cout << "  Truncation test PASSED" << std::endl;
}

void test_data_formatter_no_bos_eos() {
    std::cout << "Running test_data_formatter_no_bos_eos..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 6;
    config.pad_token_id = 0;
    config.add_bos_eos = false; // No BOS/EOS tokens
    
    DataFormatter formatter(config);
    
    std::vector<int> tokens = {100, 101, 102};
    FormattedSequence formatted = formatter.format_sequence(tokens);
    
    // Should be: [100, 101, 102, PAD, PAD, PAD]
    assert(formatted.token_ids.size() == 6);
    assert(formatted.token_ids[0] == 100);
    assert(formatted.token_ids[1] == 101);
    assert(formatted.token_ids[2] == 102);
    assert(formatted.token_ids[3] == 0); // PAD
    assert(formatted.actual_length == 3);
    assert(!formatted.was_truncated);
    
    std::cout << "  No BOS/EOS test PASSED" << std::endl;
}

void test_data_formatter_exact_length() {
    std::cout << "Running test_data_formatter_exact_length..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 5;
    config.add_bos_eos = true;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    
    DataFormatter formatter(config);
    
    // Test sequence that exactly fits with BOS/EOS
    std::vector<int> tokens = {100, 101, 102}; // +BOS+EOS = 5 total
    FormattedSequence formatted = formatter.format_sequence(tokens);
    
    assert(formatted.token_ids.size() == 5);
    assert(formatted.token_ids[0] == 1); // BOS
    assert(formatted.token_ids[1] == 100);
    assert(formatted.token_ids[2] == 101);
    assert(formatted.token_ids[3] == 102);
    assert(formatted.token_ids[4] == 2); // EOS
    assert(formatted.actual_length == 5);
    assert(!formatted.was_truncated);
    
    std::cout << "  Exact length test PASSED" << std::endl;
}

void test_data_formatter_batch_processing() {
    std::cout << "Running test_data_formatter_batch_processing..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 6;
    config.batch_size = 3;
    config.pad_token_id = 0;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    config.add_bos_eos = true;
    
    DataFormatter formatter(config);
    
    // Test batch with multiple sequences of different lengths
    std::vector<std::vector<int>> sequences = {
        {100, 101},           // Short sequence
        {200, 201, 202, 203}, // Longer sequence  
        {300}                 // Very short sequence
    };
    
    FormattedBatch batch = formatter.format_batch(sequences);
    
    assert(batch.batch_size == 3);
    assert(batch.sequence_length == 6);
    assert(batch.sequences.size() == 3);
    assert(batch.sequence_lengths.size() == 3);
    assert(batch.truncation_flags.size() == 3);
    
    // Check first sequence: [BOS, 100, 101, EOS, PAD, PAD]
    assert(batch.sequences[0][0] == 1);  // BOS
    assert(batch.sequences[0][1] == 100);
    assert(batch.sequences[0][2] == 101);
    assert(batch.sequences[0][3] == 2);  // EOS
    assert(batch.sequences[0][4] == 0);  // PAD
    assert(batch.sequences[0][5] == 0);  // PAD
    assert(batch.sequence_lengths[0] == 4); // BOS + 2 tokens + EOS
    assert(!batch.truncation_flags[0]);
    
    // Check second sequence: [BOS, 200, 201, 202, 203, EOS]
    assert(batch.sequences[1][0] == 1);  // BOS
    assert(batch.sequences[1][1] == 200);
    assert(batch.sequences[1][4] == 203);
    assert(batch.sequences[1][5] == 2);  // EOS
    assert(batch.sequence_lengths[1] == 6); // BOS + 4 tokens + EOS
    assert(!batch.truncation_flags[1]);
    
    // Check total elements and size calculation
    assert(batch.total_elements() == 18); // 3 sequences * 6 tokens
    assert(batch.size_in_bytes() == 18 * sizeof(uint32_t));
    
    std::cout << "  Batch processing test PASSED" << std::endl;
}

void test_data_formatter_batch_builder() {
    std::cout << "Running test_data_formatter_batch_builder..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 4;
    config.batch_size = 2;
    config.pad_token_id = 0;
    config.add_bos_eos = false;
    
    DataFormatter formatter(config);
    auto builder = formatter.create_batch_builder();
    
    // Test adding sequences one by one
    assert(builder.current_batch_size() == 0);
    assert(!builder.is_batch_full());
    
    bool added1 = builder.add_sequence({100, 101});
    assert(added1);
    assert(builder.current_batch_size() == 1);
    assert(!builder.is_batch_full());
    
    bool added2 = builder.add_sequence({200, 201, 202});
    assert(added2);
    assert(builder.current_batch_size() == 2);
    assert(builder.is_batch_full());
    
    // Try to add when full - should fail
    bool added3 = builder.add_sequence({300});
    assert(!added3);
    assert(builder.current_batch_size() == 2);
    
    // Finalize the batch
    FormattedBatch batch = builder.finalize_batch();
    assert(batch.batch_size == 2);
    assert(batch.sequence_length == 4);
    
    // Builder should be reset
    assert(builder.current_batch_size() == 0);
    assert(!builder.is_batch_full());
    
    // Now we can add the sequence that failed before
    bool added4 = builder.add_sequence({300});
    assert(added4);
    assert(builder.current_batch_size() == 1);
    
    std::cout << "  Batch builder test PASSED" << std::endl;
}

void test_data_formatter_validation() {
    std::cout << "Running test_data_formatter_validation..." << std::endl;
    
    // Test uint32 validation
    DataFormatterConfig config;
    config.use_uint32 = true;
    
    DataFormatter formatter(config);
    
    // Valid tokens (use reasonable large values, not max limits)
    std::vector<int> valid_tokens = {0, 1000, 50000, 100000};
    assert(formatter.validate_token_ids(valid_tokens));
    
    // Invalid tokens (negative)
    std::vector<int> invalid_tokens = {-1, 1000, 2000};
    assert(!formatter.validate_token_ids(invalid_tokens));
    
    // Test uint16 mode
    config.use_uint32 = false;
    formatter.update_config(config);
    
    std::vector<int> valid_uint16 = {0, 1000, 60000}; // Well within uint16 range
    assert(formatter.validate_token_ids(valid_uint16));
    
    std::vector<int> invalid_uint16 = {0, 1000, 70000}; // Above uint16 range
    assert(!formatter.validate_token_ids(invalid_uint16));
    
    std::cout << "  Validation test PASSED" << std::endl;
}

void test_data_formatter_statistics() {
    std::cout << "Running test_data_formatter_statistics..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 5;
    config.add_bos_eos = true;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    
    DataFormatter formatter(config);
    
    // Process some sequences to generate statistics
    formatter.format_sequence({100}); // Will be padded: [BOS, 100, EOS, PAD, PAD]
    formatter.format_sequence({200, 201, 202, 203, 204, 205}); // Will be truncated: [BOS, 200, 201, 202, EOS]
    formatter.format_sequence({300, 301, 302}); // Will be padded: [BOS, 300, 301, 302, EOS] - actually fits exactly!
    
    const auto& stats = formatter.get_stats();
    
    assert(stats.total_sequences_processed == 3);
    assert(stats.sequences_truncated == 1);
    // The third sequence (300, 301, 302) + BOS + EOS = 5 tokens, which exactly fits max_length=5
    // So only the first sequence gets padded
    assert(stats.sequences_padded == 1);
    
    double truncation_rate = stats.truncation_rate();
    assert(truncation_rate > 0.33 && truncation_rate < 0.34); // 1/3
    
    double padding_efficiency = stats.padding_efficiency();
    assert(padding_efficiency > 0.0 && padding_efficiency < 1.0);
    
    // Test reset
    formatter.reset_stats();
    const auto& reset_stats = formatter.get_stats();
    assert(reset_stats.total_sequences_processed == 0);
    
    std::cout << "  Statistics test PASSED" << std::endl;
}

void test_data_formatter_edge_cases() {
    std::cout << "Running test_data_formatter_edge_cases..." << std::endl;
    
    DataFormatterConfig config;
    config.max_sequence_length = 2;
    config.add_bos_eos = true;
    config.bos_token_id = 1;
    config.eos_token_id = 2;
    
    DataFormatter formatter(config);
    
    // Test empty sequence
    std::vector<int> empty_tokens = {};
    FormattedSequence formatted_empty = formatter.format_sequence(empty_tokens);
    
    // Should be [BOS, EOS]
    assert(formatted_empty.token_ids.size() == 2);
    assert(formatted_empty.token_ids[0] == 1); // BOS
    assert(formatted_empty.token_ids[1] == 2); // EOS
    assert(formatted_empty.actual_length == 2);
    assert(!formatted_empty.was_truncated);
    
    // Test sequence with BOS/EOS already present
    std::vector<int> with_bos_eos = {1, 100, 2}; // Already has BOS and EOS
    FormattedSequence formatted_with = formatter.format_sequence(with_bos_eos);
    
    // Should remain [BOS, 100, EOS] but might be truncated to fit max_length=2
    assert(formatted_with.token_ids.size() == 2);
    assert(formatted_with.was_truncated); // Because original is length 3 but max is 2
    
    std::cout << "  Edge cases test PASSED" << std::endl;
}

void test_data_formatter_configuration_errors() {
    std::cout << "Running test_data_formatter_configuration_errors..." << std::endl;
    
    // Test invalid configurations
    try {
        DataFormatterConfig bad_config;
        bad_config.max_sequence_length = 0;
        DataFormatter formatter(bad_config);
        assert(false && "Should have thrown exception");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    try {
        DataFormatterConfig bad_config;
        bad_config.batch_size = 0;
        DataFormatter formatter(bad_config);
        assert(false && "Should have thrown exception");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    std::cout << "  Configuration error test PASSED" << std::endl;
}

int main() {
    std::cout << "DataFormatter Unit Tests" << std::endl;
    std::cout << "========================" << std::endl;
    
    test_data_formatter_basic_functionality();
    test_data_formatter_truncation();
    test_data_formatter_no_bos_eos();
    test_data_formatter_exact_length();
    test_data_formatter_batch_processing();
    test_data_formatter_batch_builder();
    test_data_formatter_validation();
    test_data_formatter_statistics();
    test_data_formatter_edge_cases();
    test_data_formatter_configuration_errors();
    
    std::cout << std::endl;
    std::cout << "All DataFormatter tests PASSED!" << std::endl;
    std::cout << "DataFormatter is ready for MSL integration." << std::endl;
    
    return 0;
} 