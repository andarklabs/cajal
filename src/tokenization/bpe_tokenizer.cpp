#include "bpe_tokenizer.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <regex>

// Static constant definitions
const std::string BPETokenizer::DEFAULT_UNK_TOKEN = "[UNK]";
const std::string BPETokenizer::DEFAULT_PAD_TOKEN = "[PAD]";
const std::string BPETokenizer::DEFAULT_BOS_TOKEN = "[BOS]";
const std::string BPETokenizer::DEFAULT_EOS_TOKEN = "[EOS]";

BPETokenizer::BPETokenizer() {
    // Constructor initializes with empty state
    // Special token IDs will be set when tokens are added during training
}

size_t BPETokenizer::get_vocab_size() const {
    return vocab_.size();
}

std::optional<int> BPETokenizer::get_special_token_id(const std::string& token_str) const {
    if (token_str == unk_token_ && unk_token_id_ != -1) return unk_token_id_;
    if (token_str == pad_token_ && pad_token_id_ != -1) return pad_token_id_;
    if (token_str == bos_token_ && bos_token_id_ != -1) return bos_token_id_;
    if (token_str == eos_token_ && eos_token_id_ != -1) return eos_token_id_;
    return std::nullopt;
}

int BPETokenizer::token_to_id(const std::string& token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    // Return UNK token ID if token not found
    return unk_token_id_;
}

std::string BPETokenizer::id_to_token(int id) const {
    auto it = id_to_token_map_.find(id);
    if (it != id_to_token_map_.end()) {
        return it->second;
    }
    // Return UNK token if ID not found
    return unk_token_;
}

void BPETokenizer::add_token_to_vocab(const std::string& token, bool is_special) {
    if (vocab_.find(token) != vocab_.end()) {
        return; // Token already exists
    }
    
    int token_id = next_token_id_++;
    vocab_[token] = token_id;
    id_to_token_map_[token_id] = token;
    
    // Update special token IDs if this is a special token
    if (is_special) {
        if (token == unk_token_) unk_token_id_ = token_id;
        else if (token == pad_token_) pad_token_id_ = token_id;
        else if (token == bos_token_) bos_token_id_ = token_id;
        else if (token == eos_token_) eos_token_id_ = token_id;
    }
}

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    // Simple pre-tokenization: split by whitespace
    // This can be enhanced later with more sophisticated regex-based splitting
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;
    
    while (stream >> word) {
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> token_ids;
    
    // Add BOS token if requested
    if (add_bos && bos_token_id_ != -1) {
        token_ids.push_back(bos_token_id_);
    }
    
    // Pre-tokenize the text
    std::vector<std::string> words = pre_tokenize(text);
    
    // Apply BPE to each word and convert to IDs
    for (const std::string& word : words) {
        std::vector<std::string> bpe_tokens = bpe_tokenize_word(word);
        for (const std::string& token : bpe_tokens) {
            token_ids.push_back(token_to_id(token));
        }
    }
    
    // Add EOS token if requested
    if (add_eos && eos_token_id_ != -1) {
        token_ids.push_back(eos_token_id_);
    }
    
    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const {
    std::vector<std::string> tokens;
    
    for (int id : token_ids) {
        std::string token = id_to_token(id);
        // Skip special tokens in decoding (optional behavior)
        if (token != pad_token_) {
            tokens.push_back(token);
        }
    }
    
    // Simple joining with spaces - this may need refinement for subword tokens
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        result += tokens[i];
    }
    
    return result;
}

std::vector<std::string> BPETokenizer::bpe_tokenize_word(const std::string& word) const {
    // If the tokenizer hasn't been trained yet, just return character-level tokens
    if (ranked_merges_.empty()) {
        std::vector<std::string> char_tokens;
        for (char c : word) {
            char_tokens.push_back(std::string(1, c));
        }
        return char_tokens;
    }
    
    // Start with character-level tokenization
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    
    // Apply merges in order (lowest rank first)
    bool changed = true;
    while (changed && tokens.size() > 1) {
        changed = false;
        int best_rank = INT_MAX;
        int best_i = -1;
        
        // Find the highest priority merge available
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            std::pair<std::string, std::string> pair = {tokens[i], tokens[i + 1]};
            auto it = ranked_merges_.find(pair);
            if (it != ranked_merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }
        
        // Apply the best merge found
        if (best_i != -1) {
            std::string merged = tokens[best_i] + tokens[best_i + 1];
            tokens[best_i] = merged;
            tokens.erase(tokens.begin() + best_i + 1);
            changed = true;
        }
    }
    
    return tokens;
}

// Placeholder implementations for training methods - to be implemented
bool BPETokenizer::train(const std::vector<std::string>& corpus, 
                        int target_vocab_size,
                        const std::vector<std::string>& initial_special_tokens) {
    // Clear existing state
    vocab_.clear();
    id_to_token_map_.clear();
    ranked_merges_.clear();
    next_token_id_ = 0;
    
    // Reset special token IDs
    unk_token_id_ = pad_token_id_ = bos_token_id_ = eos_token_id_ = -1;
    
    // Add special tokens first
    for (const std::string& special_token : initial_special_tokens) {
        add_token_to_vocab(special_token, true);
    }
    
    if (corpus.empty()) {
        std::cout << "Warning: Training on empty corpus. Only special tokens will be available." << std::endl;
        return true; // Success with just special tokens
    }
    
    std::cout << "Starting BPE training with target vocab size: " << target_vocab_size << std::endl;
    std::cout << "Corpus size: " << corpus.size() << " documents" << std::endl;
    
    // Step 1: Build initial word counts and character vocabulary
    std::map<std::string, int> word_counts;
    std::set<char> unique_chars;
    
    for (const std::string& document : corpus) {
        std::vector<std::string> words = pre_tokenize(document);
        
        for (const std::string& word : words) {
            // Create space-separated character representation for BPE
            std::string char_word;
            for (size_t i = 0; i < word.size(); ++i) {
                if (i > 0) char_word += " ";
                char_word += word[i];
                unique_chars.insert(word[i]);
            }
            word_counts[char_word]++;
        }
    }
    
    // Add all unique characters to vocabulary
    for (char c : unique_chars) {
        std::string char_token(1, c);
        add_token_to_vocab(char_token, false);
    }
    
    std::cout << "Initial character vocabulary size: " << unique_chars.size() << std::endl;
    std::cout << "Total unique words: " << word_counts.size() << std::endl;
    
    // Step 2: Iteratively find most frequent pairs and merge them
    int merge_rank = 0;
    
    while (static_cast<int>(vocab_.size()) < target_vocab_size) {
        // Get all pair frequencies
        auto pair_freqs = get_pair_frequencies(word_counts);
        
        if (pair_freqs.empty()) {
            std::cout << "No more pairs to merge. Final vocab size: " << vocab_.size() << std::endl;
            break;
        }
        
        // Find the most frequent pair
        auto most_frequent = std::max_element(pair_freqs.begin(), pair_freqs.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
        
        if (most_frequent == pair_freqs.end() || most_frequent->second <= 1) {
            std::cout << "No frequent pairs remaining. Final vocab size: " << vocab_.size() << std::endl;
            break;
        }
        
        // Create merged symbol
        std::string merged_symbol = most_frequent->first.first + most_frequent->first.second;
        
        // Add to vocabulary
        add_token_to_vocab(merged_symbol, false);
        
        // Record the merge rule
        ranked_merges_[most_frequent->first] = merge_rank++;
        
        // Apply the merge to all words
        merge_pair_in_word_counts(word_counts, most_frequent->first, merged_symbol);
        
        if (merge_rank % 100 == 0) {
            std::cout << "Completed " << merge_rank << " merges. Vocab size: " << vocab_.size() 
                      << ". Most recent merge: (" << most_frequent->first.first 
                      << ", " << most_frequent->first.second << ") -> " << merged_symbol
                      << " (freq: " << most_frequent->second << ")" << std::endl;
        }
    }
    
    std::cout << "BPE training completed. Final vocabulary size: " << vocab_.size() << std::endl;
    std::cout << "Number of learned merges: " << ranked_merges_.size() << std::endl;
    
    return true;
}

std::map<std::pair<std::string, std::string>, int> BPETokenizer::get_pair_frequencies(
    const std::map<std::string, int>& word_counts) const {
    
    std::map<std::pair<std::string, std::string>, int> pair_freqs;
    
    for (const auto& word_count_pair : word_counts) {
        const std::string& word = word_count_pair.first;
        int count = word_count_pair.second;
        
        // Split word into symbols (space-separated for BPE representation)
        std::vector<std::string> symbols;
        std::istringstream stream(word);
        std::string symbol;
        
        while (stream >> symbol) {
            symbols.push_back(symbol);
        }
        
        // Count all adjacent pairs in this word
        for (size_t i = 0; i < symbols.size() - 1; ++i) {
            std::pair<std::string, std::string> pair = {symbols[i], symbols[i + 1]};
            pair_freqs[pair] += count;
        }
    }
    
    return pair_freqs;
}

void BPETokenizer::merge_pair_in_word_counts(
    std::map<std::string, int>& word_counts,
    const std::pair<std::string, std::string>& pair_to_merge,
    const std::string& merged_symbol) const {
    
    std::map<std::string, int> new_word_counts;
    
    for (const auto& word_count_pair : word_counts) {
        const std::string& word = word_count_pair.first;
        int count = word_count_pair.second;
        
        // Split word into symbols
        std::vector<std::string> symbols;
        std::istringstream stream(word);
        std::string symbol;
        
        while (stream >> symbol) {
            symbols.push_back(symbol);
        }
        
        // Apply the merge to this word
        std::vector<std::string> merged_symbols;
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i < symbols.size() - 1 && 
                symbols[i] == pair_to_merge.first && 
                symbols[i + 1] == pair_to_merge.second) {
                // Found the pair to merge
                merged_symbols.push_back(merged_symbol);
                ++i; // Skip the next symbol as it's part of the merged pair
            } else {
                merged_symbols.push_back(symbols[i]);
            }
        }
        
        // Reconstruct the word string with merged symbols
        std::string new_word;
        for (size_t i = 0; i < merged_symbols.size(); ++i) {
            if (i > 0) new_word += " ";
            new_word += merged_symbols[i];
        }
        
        new_word_counts[new_word] += count;
    }
    
    // Replace the original word counts
    word_counts = std::move(new_word_counts);
}

bool BPETokenizer::save_model(const std::string& path_prefix) const {
    // Save vocabulary
    std::string vocab_path = path_prefix + ".vocab";
    std::ofstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Failed to open vocab file for writing: " << vocab_path << std::endl;
        return false;
    }
    
    // Save vocab in format: token<tab>id
    for (const auto& token_id_pair : vocab_) {
        vocab_file << token_id_pair.first << "\t" << token_id_pair.second << std::endl;
    }
    vocab_file.close();
    
    // Save merges
    std::string merges_path = path_prefix + ".merges";
    std::ofstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file for writing: " << merges_path << std::endl;
        return false;
    }
    
    // Create a vector of merges sorted by rank
    std::vector<std::pair<std::pair<std::string, std::string>, int>> sorted_merges(
        ranked_merges_.begin(), ranked_merges_.end());
    
    std::sort(sorted_merges.begin(), sorted_merges.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second; // Sort by rank (ascending)
        });
    
    // Save merges in format: symbol1<space>symbol2<tab>rank
    for (const auto& merge_rank_pair : sorted_merges) {
        merges_file << merge_rank_pair.first.first << " " << merge_rank_pair.first.second 
                   << "\t" << merge_rank_pair.second << std::endl;
    }
    merges_file.close();
    
    // Save special token mappings
    std::string special_path = path_prefix + ".special";
    std::ofstream special_file(special_path);
    if (!special_file.is_open()) {
        std::cerr << "Failed to open special tokens file for writing: " << special_path << std::endl;
        return false;
    }
    
    special_file << "unk_token\t" << unk_token_ << "\t" << unk_token_id_ << std::endl;
    special_file << "pad_token\t" << pad_token_ << "\t" << pad_token_id_ << std::endl;
    special_file << "bos_token\t" << bos_token_ << "\t" << bos_token_id_ << std::endl;
    special_file << "eos_token\t" << eos_token_ << "\t" << eos_token_id_ << std::endl;
    special_file << "next_token_id\t" << next_token_id_ << std::endl;
    special_file.close();
    
    std::cout << "Model saved to: " << path_prefix << ".{vocab,merges,special}" << std::endl;
    return true;
}

bool BPETokenizer::load_model(const std::string& path_prefix) {
    // Clear existing state
    vocab_.clear();
    id_to_token_map_.clear();
    ranked_merges_.clear();
    
    // Load vocabulary
    std::string vocab_path = path_prefix + ".vocab";
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Failed to open vocab file for reading: " << vocab_path << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(vocab_file, line)) {
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string token = line.substr(0, tab_pos);
            int id = std::stoi(line.substr(tab_pos + 1));
            vocab_[token] = id;
            id_to_token_map_[id] = token;
        }
    }
    vocab_file.close();
    
    // Load merges
    std::string merges_path = path_prefix + ".merges";
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file for reading: " << merges_path << std::endl;
        return false;
    }
    
    while (std::getline(merges_file, line)) {
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string pair_str = line.substr(0, tab_pos);
            int rank = std::stoi(line.substr(tab_pos + 1));
            
            size_t space_pos = pair_str.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = pair_str.substr(0, space_pos);
                std::string second = pair_str.substr(space_pos + 1);
                ranked_merges_[{first, second}] = rank;
            }
        }
    }
    merges_file.close();
    
    // Load special token mappings
    std::string special_path = path_prefix + ".special";
    std::ifstream special_file(special_path);
    if (!special_file.is_open()) {
        std::cerr << "Failed to open special tokens file for reading: " << special_path << std::endl;
        return false;
    }
    
    while (std::getline(special_file, line)) {
        std::istringstream stream(line);
        std::string key, token_str, id_str;
        if (stream >> key >> token_str >> id_str) {
            if (key == "unk_token") {
                unk_token_ = token_str;
                unk_token_id_ = std::stoi(id_str);
            } else if (key == "pad_token") {
                pad_token_ = token_str;
                pad_token_id_ = std::stoi(id_str);
            } else if (key == "bos_token") {
                bos_token_ = token_str;
                bos_token_id_ = std::stoi(id_str);
            } else if (key == "eos_token") {
                eos_token_ = token_str;
                eos_token_id_ = std::stoi(id_str);
            } else if (key == "next_token_id") {
                next_token_id_ = std::stoi(token_str); // token_str contains the ID in this case
            }
        }
    }
    special_file.close();
    
    std::cout << "Model loaded from: " << path_prefix << ".{vocab,merges,special}" << std::endl;
    std::cout << "Vocabulary size: " << vocab_.size() << std::endl;
    std::cout << "Number of merge rules: " << ranked_merges_.size() << std::endl;
    
    return true;
} 