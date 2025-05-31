#include "../src/data/book_corpus_reader.h" // Adjust path if necessary
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio> // For std::remove (deleting temp file)

// Helper function to create a temporary file with specific content
// Returns the path to the temporary file. Caller is responsible for deleting it.
std::string create_temp_file(const std::string& content, const std::string& temp_filename = "temp_test_file.txt") {
    std::ofstream outfile(temp_filename);
    if (outfile.is_open()) {
        outfile << content;
        outfile.close();
        return temp_filename;
    } else {
        std::cerr << "Failed to create temporary file: " << temp_filename << std::endl;
        return "";
    }
}

void test_default_cleaning() {
    std::cout << "Running test_default_cleaning..." << std::endl;
    TextCleanerConfig config; // Default config
    assert(config.to_lowercase == true);
    assert(config.normalize_whitespace == true);
    assert(config.newlines_to_spaces == true);

    std::string raw_content = "  First Line with   EXTRA Spaces.\nSecond Line.\n\nAnother line with TAB\tinside.";
    std::string expected_cleaned_content = "first line with extra spaces. second line. another line with tab inside.";
    // Note: The current clean_text implementation might produce slightly different results for newlines within the string vs. end-of-line newlines read by getline.
    // This expected string assumes newlines become single spaces and everything is compact.

    std::string temp_file_path = create_temp_file(raw_content);
    if (temp_file_path.empty()) {
        assert(false && "Failed to set up temp file for test_default_cleaning");
        return;
    }

    BookCorpusReader reader({temp_file_path}, config);
    assert(reader.begin_stream() && "Failed to begin stream for test_default_cleaning");

    std::optional<std::string> line1_opt = reader.next_cleaned_line();
    // The current implementation of next_cleaned_line calls clean_text on each line read by getline.
    // So, we will test this line by line based on how getline splits the raw_content.
    
    // Line 1: "  First Line with   EXTRA Spaces."
    std::string expected_line1 = "first line with extra spaces.";
    assert(line1_opt.has_value() && "Line 1 missing");
    if (line1_opt.has_value()) {
        std::cout << "  Raw line1 (from file): '  First Line with   EXTRA Spaces.'" << std::endl;
        std::cout << "  Cleaned line1: '" << line1_opt.value() << "'" << std::endl;
        std::cout << "  Expected line1: '" << expected_line1 << "'" << std::endl;
        assert(line1_opt.value() == expected_line1 && "Line 1 cleaning failed");
    }

    // Line 2: "Second Line."
    std::optional<std::string> line2_opt = reader.next_cleaned_line();
    std::string expected_line2 = "second line.";
    assert(line2_opt.has_value() && "Line 2 missing");
    if (line2_opt.has_value()) {
        std::cout << "  Raw line2 (from file): 'Second Line.'" << std::endl;
        std::cout << "  Cleaned line2: '" << line2_opt.value() << "'" << std::endl;
        std::cout << "  Expected line2: '" << expected_line2 << "'" << std::endl;
        assert(line2_opt.value() == expected_line2 && "Line 2 cleaning failed");
    }

    // Line 3: empty line because of \n\n, getline might return an empty string.
    std::optional<std::string> line3_opt = reader.next_cleaned_line();
    std::string expected_line3 = ""; // clean_text on an empty string is empty.
    assert(line3_opt.has_value() && "Line 3 (empty) missing");
     if (line3_opt.has_value()) {
        std::cout << "  Raw line3 (from file): ''" << std::endl;
        std::cout << "  Cleaned line3: '" << line3_opt.value() << "'" << std::endl;
        std::cout << "  Expected line3: '" << expected_line3 << "'" << std::endl;
        assert(line3_opt.value() == expected_line3 && "Line 3 (empty) cleaning failed");
    }

    // Line 4: "Another line with TAB\tinside."
    std::optional<std::string> line4_opt = reader.next_cleaned_line();
    std::string expected_line4 = "another line with tab inside.";
    assert(line4_opt.has_value() && "Line 4 missing");
    if (line4_opt.has_value()) {
        std::cout << "  Raw line4 (from file): 'Another line with TAB\tinside.'" << std::endl;
        std::cout << "  Cleaned line4: '" << line4_opt.value() << "'" << std::endl;
        std::cout << "  Expected line4: '" << expected_line4 << "'" << std::endl;
        assert(line4_opt.value() == expected_line4 && "Line 4 cleaning failed");
    }

    // Should be EOF now
    std::optional<std::string> eof_opt = reader.next_cleaned_line();
    assert(!eof_opt.has_value() && "Expected EOF");

    std::remove(temp_file_path.c_str());
    std::cout << "test_default_cleaning PASSED." << std::endl;
}

void test_no_lowercase_cleaning() {
    std::cout << "Running test_no_lowercase_cleaning..." << std::endl;
    TextCleanerConfig config;
    config.to_lowercase = false;
    config.normalize_whitespace = true;
    config.newlines_to_spaces = true;

    std::string raw_content = "  MixedCase   Line ";
    std::string temp_file_path = create_temp_file(raw_content);
    BookCorpusReader reader({temp_file_path}, config);
    reader.begin_stream();
    std::optional<std::string> line_opt = reader.next_cleaned_line();
    std::string expected_line = "MixedCase Line"; // Normalization but no lowercase
    assert(line_opt.has_value());
    if(line_opt.has_value()){
        std::cout << "  Raw line: '" << raw_content << "'" << std::endl;
        std::cout << "  Cleaned line: '" << line_opt.value() << "'" << std::endl;
        std::cout << "  Expected line: '" << expected_line << "'" << std::endl;
        assert(line_opt.value() == expected_line && "No lowercase cleaning failed");
    }
    std::remove(temp_file_path.c_str());
    std::cout << "test_no_lowercase_cleaning PASSED." << std::endl;
}

void test_no_whitespace_normalization() {
    std::cout << "Running test_no_whitespace_normalization..." << std::endl;
    TextCleanerConfig config;
    config.to_lowercase = true;
    config.normalize_whitespace = false; // This is the key change
    // newlines_to_spaces is irrelevant if normalize_whitespace is false in current clean_text logic

    std::string raw_content = "  Lowercase   Line\nWith\tTabs  "; // Contains leading/trailing/multiple spaces, newline, tab
    std::string temp_file_path = create_temp_file(raw_content);
    BookCorpusReader reader({temp_file_path}, config);
    reader.begin_stream();
    std::optional<std::string> line_opt = reader.next_cleaned_line();
    // If normalize_whitespace is false, clean_text currently only does lowercasing.
    // The raw_content is read line by line. So getline will give "  Lowercase   Line"
    std::string expected_line = "  lowercase   line"; 
    assert(line_opt.has_value());
    if(line_opt.has_value()){
        std::cout << "  Raw line (from getline): '  Lowercase   Line'" << std::endl;
        std::cout << "  Cleaned line: '" << line_opt.value() << "'" << std::endl;
        std::cout << "  Expected line: '" << expected_line << "'" << std::endl;
        assert(line_opt.value() == expected_line && "No whitespace normalization failed for line 1");
    }

    line_opt = reader.next_cleaned_line();
    // getline will give "With\tTabs  "
    expected_line = "with\ttabs  ";
    assert(line_opt.has_value());
    if(line_opt.has_value()){
        std::cout << "  Raw line (from getline): 'With\tTabs  '" << std::endl;
        std::cout << "  Cleaned line: '" << line_opt.value() << "'" << std::endl;
        std::cout << "  Expected line: '" << expected_line << "'" << std::endl;
        assert(line_opt.value() == expected_line && "No whitespace normalization failed for line 2");
    }

    std::remove(temp_file_path.c_str());
    std::cout << "test_no_whitespace_normalization PASSED." << std::endl;
}

void test_preserve_newlines() {
    std::cout << "Running test_preserve_newlines..." << std::endl;
    TextCleanerConfig config;
    config.to_lowercase = true;
    config.normalize_whitespace = true;
    config.newlines_to_spaces = false; // Preserve newlines. clean_text should handle internal newlines if passed them.
                                       // However, we are testing via next_cleaned_line(), which uses getline().

    std::string raw_content_for_file = "First part. \n Second part after explicit newline.";
    // When create_temp_file writes this, it becomes two lines in the file because of the \n.
    // Line 1 in file: "First part. "
    // Line 2 in file: " Second part after explicit newline."
    
    std::string temp_file_path = create_temp_file(raw_content_for_file);
    BookCorpusReader reader({temp_file_path}, config);
    assert(reader.begin_stream() && "Failed to begin stream for test_preserve_newlines");

    // First call to next_cleaned_line() should get the first line from the file
    std::optional<std::string> line1_opt = reader.next_cleaned_line();
    std::string expected_cleaned_line1 = "first part.";
    
    assert(line1_opt.has_value() && "Preserve newlines: Line 1 missing");
    if(line1_opt.has_value()){
        // std::cout << "  Raw content in file was: First part. \\n Second part after explicit newline." << std::endl;
        std::cout << "  Cleaned line1: '" << line1_opt.value() << "'" << std::endl;
        std::cout << "  Expected line1: '" << expected_cleaned_line1 << "'" << std::endl;
        assert(line1_opt.value() == expected_cleaned_line1 && "Preserve newlines: Line 1 cleaning failed");
    }

    // Second call to next_cleaned_line() should get the second line from the file
    std::optional<std::string> line2_opt = reader.next_cleaned_line();
    std::string expected_cleaned_line2 = "second part after explicit newline."; // Leading space from " Second..." should be trimmed.

    assert(line2_opt.has_value() && "Preserve newlines: Line 2 missing");
    if(line2_opt.has_value()){
        std::cout << "  Cleaned line2: '" << line2_opt.value() << "'" << std::endl;
        std::cout << "  Expected line2: '" << expected_cleaned_line2 << "'" << std::endl;
        assert(line2_opt.value() == expected_cleaned_line2 && "Preserve newlines: Line 2 cleaning failed");
    }

    // Should be EOF now
    std::optional<std::string> eof_opt = reader.next_cleaned_line();
    assert(!eof_opt.has_value() && "Preserve newlines: Expected EOF after line 2");

    std::remove(temp_file_path.c_str());
    std::cout << "test_preserve_newlines PASSED (pending verification)." << std::endl;
}


int main() {
    test_default_cleaning();
    test_no_lowercase_cleaning();
    test_no_whitespace_normalization();
    test_preserve_newlines(); // This one is tricky with current implementation

    std::cout << "\nAll BookCorpusReader preliminary tests (potentially with failures) finished." << std::endl;
    std::cout << "Review test_preserve_newlines and the clean_text implementation for whitespace/newline handling." << std::endl;
    return 0;
} 