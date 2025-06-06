//
// Threadgroup Mismatch Test - Verify the dispatch bug
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    std::cout << "🎯 THREADGROUP DISPATCH MISMATCH TEST" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Test the exact failing scenario
    struct TestCase {
        int batch_size;
        int sequence_length; 
        int embedding_dim;
        const char* expected_result;
    };
    
    TestCase tests[] = {
        {1, 4, 32, "✅ SHOULD WORK"},   // num_instances=4, embedding_dim=32
        {1, 4, 64, "❌ SHOULD FAIL"},   // num_instances=4, embedding_dim=64 <- THE BUG!
        {1, 64, 64, "✅ SHOULD WORK"},  // num_instances=64, embedding_dim=64
        {2, 32, 64, "✅ SHOULD WORK"},  // num_instances=64, embedding_dim=64
    };
    
    std::cout << "\n🧪 DISPATCH ANALYSIS:" << std::endl;
    std::cout << "Format: batch_size × seq_len = num_instances vs embedding_dim" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    for (const auto& test : tests) {
        int num_instances = test.batch_size * test.sequence_length;
        
        std::cout << "Test: " << test.batch_size << " × " << test.sequence_length 
                  << " = " << num_instances << " instances, embedding_dim=" << test.embedding_dim << std::endl;
        
        // Analyze the dispatch configuration
        std::cout << "  ThreadsPerGrid: (" << num_instances << ", 1, 1)" << std::endl;
        std::cout << "  ThreadsPerThreadgroup: (" << test.embedding_dim << ", 1, 1)" << std::endl;
        
        int num_threadgroups = (num_instances + test.embedding_dim - 1) / test.embedding_dim; // Ceiling division
        int threads_in_last_group = num_instances % test.embedding_dim;
        if (threads_in_last_group == 0) threads_in_last_group = test.embedding_dim;
        
        std::cout << "  Actual threadgroups: " << num_threadgroups << std::endl;
        std::cout << "  Threads in threadgroup: " << (num_threadgroups == 1 ? threads_in_last_group : test.embedding_dim) << std::endl;
        
        if (num_instances < test.embedding_dim) {
            std::cout << "  🚨 CRITICAL: Only " << num_instances << " threads, but reduction needs " << test.embedding_dim << " threads!" << std::endl;
            std::cout << "  💀 Reduction algorithm will access UNINITIALIZED threadgroup memory!" << std::endl;
        } else {
            std::cout << "  ✅ Sufficient threads for reduction algorithm" << std::endl;
        }
        
        std::cout << "  Expected: " << test.expected_result << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "🎯 ROOT CAUSE ANALYSIS:" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "The layer_norm_backward kernel reduction algorithm assumes:" << std::endl;
    std::cout << "  ✅ threadgroup has EXACTLY embedding_dim threads" << std::endl;
    std::cout << "  ✅ Each thread initializes tg_sum_dL_dnorm_x[dim_idx]" << std::endl;
    std::cout << "  ✅ Reduction loop: for (s = embedding_dim/2; s > 0; s >>= 1)" << std::endl;
    std::cout << std::endl;
    std::cout << "But when num_instances < embedding_dim:" << std::endl;
    std::cout << "  ❌ Only num_instances threads are launched" << std::endl;
    std::cout << "  ❌ tg_sum_dL_dnorm_x[num_instances...embedding_dim-1] = GARBAGE" << std::endl;
    std::cout << "  ❌ Reduction reads garbage values → NaN propagation!" << std::endl;
    std::cout << std::endl;
    std::cout << "🔧 SOLUTION:" << std::endl;
    std::cout << "============" << std::endl;
    std::cout << "Change dispatch to ensure threadgroup always has embedding_dim threads:" << std::endl;
    std::cout << "  📄 Option 1: Use dispatchThreads instead of dispatchThreadgroups" << std::endl;
    std::cout << "  📄 Option 2: Zero-initialize threadgroup arrays in kernel" << std::endl;
    std::cout << "  📄 Option 3: Redesign reduction to handle variable thread counts" << std::endl;
    
    return 0;
} 