#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility> // For std::pair
#include <optional>

class BPETokenizer {
public:
    // Special token constants (can be overridden)
    static const std::string DEFAULT_UNK_TOKEN;
    static const std::string DEFAULT_PAD_TOKEN;
    static const std::string DEFAULT_BOS_TOKEN;
    static const std::string DEFAULT_EOS_TOKEN;

    BPETokenizer();

    // Trains the tokenizer on a corpus.
    // corpus: Vector of strings (documents or sentences).
    // target_vocab_size: The desired final vocabulary size (including initial chars and special tokens).
    // initial_special_tokens: Tokens like [UNK], [PAD], [BOS], [EOS] to be added from the start.
    bool train(const std::vector<std::string>& corpus, 
               int target_vocab_size,
               const std::vector<std::string>& initial_special_tokens = {
                   DEFAULT_UNK_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN
               });

    // Encodes a single text string into a vector of token IDs.
    // Adds BOS/EOS if configured and not already present (logic TBD).
    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;

    // Decodes a vector of token IDs back into a string.
    std::string decode(const std::vector<int>& token_ids) const;

    // Converts a single token string to its ID.
    // Returns ID of UNK_TOKEN if token is not found.
    int token_to_id(const std::string& token) const;

    // Converts a single token ID back to its string representation.
    // Returns empty string or UNK_TOKEN if ID is not found (TBD).
    std::string id_to_token(int id) const;

    // Gets the vocabulary size.
    size_t get_vocab_size() const;

    // Gets the ID for a special token (e.g., UNK, PAD).
    std::optional<int> get_special_token_id(const std::string& token_str) const;

    // Saves the tokenizer model (vocab, merges) to files.
    // path_prefix: e.g., "./my_tokenizer" -> "./my_tokenizer.vocab", "./my_tokenizer.merges"
    bool save_model(const std::string& path_prefix) const;

    // Loads the tokenizer model from files.
    bool load_model(const std::string& path_prefix);

private:
    // Pre-tokenizes text into a list of "words".
    // For BPE, a "word" is then broken into characters before merging.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Gets all unique pairs of symbols in a list of words and their frequencies.
    std::map<std::pair<std::string, std::string>, int> get_pair_frequencies(
        const std::map<std::string, int>& word_counts
    ) const;
    
    // Merges a specific pair in all words in the vocabulary.
    // word_counts: map of word (sequence of symbols) to its count in corpus.
    // pair_to_merge: the pair to merge (e.g., ("a", "b")).
    // merged_symbol: the new symbol (e.g., "ab").
    void merge_pair_in_word_counts(
        std::map<std::string, int>& word_counts,
        const std::pair<std::string, std::string>& pair_to_merge,
        const std::string& merged_symbol
    ) const;

    // Applies learned merges to a single word (sequence of symbols) to get BPE tokens.
    std::vector<std::string> bpe_tokenize_word(const std::string& word) const;

    void add_token_to_vocab(const std::string& token, bool is_special = false);

    std::map<std::string, int> vocab_;
    std::map<int, std::string> id_to_token_map_;
    // Merges: store as ranked list or map from pair to new merged token string.
    // Using a ranked list (vector of pairs) for applying merges in order.
    std::vector<std::pair<std::string, std::string>> merge_rules_; // Stores merged pairs, e.g., ("a", "b") -> then find "ab"
                                                               // Or: std::map<std::pair<std::string, std::string>, std::string> direct_merges_;
                                                               // Or: std::map<std::pair<std::string, std::string>, int> ranked_merges_ for order;
                                                               // Let's use ranked_merges for applying them correctly.
    std::map<std::pair<std::string, std::string>, int> ranked_merges_; // pair -> rank (lower rank = earlier merge)
    
    int next_token_id_ = 0;

    // Special token tracking
    std::string unk_token_ = DEFAULT_UNK_TOKEN;
    std::string pad_token_ = DEFAULT_PAD_TOKEN;
    std::string bos_token_ = DEFAULT_BOS_TOKEN;
    std::string eos_token_ = DEFAULT_EOS_TOKEN;

    int unk_token_id_ = -1;
    int pad_token_id_ = -1;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
};

#endif // BPE_TOKENIZER_H 