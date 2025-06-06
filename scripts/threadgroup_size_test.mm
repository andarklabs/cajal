//
// Threadgroup Size Test - Check if our FFN fix is within Metal limits
//

#include <iostream>

int main() {
    std::cout << "🔍 THREADGROUP SIZE VALIDATION" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Our configuration
    int embedding_dim = 128;
    int ffn_hidden_dim = 256;
    
    // Proposed fix: (1, embedding_dim, ffn_hidden_dim) threads per threadgroup
    int proposed_threads_per_tg = 1 * embedding_dim * ffn_hidden_dim;
    
    std::cout << "📊 CONFIGURATION:" << std::endl;
    std::cout << "   embedding_dim: " << embedding_dim << std::endl;
    std::cout << "   ffn_hidden_dim: " << ffn_hidden_dim << std::endl;
    std::cout << "   Proposed threads per threadgroup: " << proposed_threads_per_tg << std::endl;
    
    // Metal limits on M3 Max (from our earlier test)
    const int METAL_MAX_THREADS_PER_THREADGROUP = 1024;
    
    std::cout << "\n🔍 METAL LIMITS CHECK:" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Metal max threads per threadgroup: " << METAL_MAX_THREADS_PER_THREADGROUP << std::endl;
    
    if (proposed_threads_per_tg <= METAL_MAX_THREADS_PER_THREADGROUP) {
        std::cout << "✅ SAFE: " << proposed_threads_per_tg << " <= " << METAL_MAX_THREADS_PER_THREADGROUP << std::endl;
        std::cout << "   → FFN backward dispatch fix is within Metal limits!" << std::endl;
    } else {
        std::cout << "❌ EXCEEDS LIMITS: " << proposed_threads_per_tg << " > " << METAL_MAX_THREADS_PER_THREADGROUP << std::endl;
        std::cout << "   → Need alternative approach!" << std::endl;
        
        // Calculate safe alternatives
        std::cout << "\n🔧 ALTERNATIVE APPROACHES:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        // Option 1: Reduce threadgroup dimensions to fit
        int safe_embedding_dim = std::min(embedding_dim, METAL_MAX_THREADS_PER_THREADGROUP / ffn_hidden_dim);
        int safe_ffn_dim = std::min(ffn_hidden_dim, METAL_MAX_THREADS_PER_THREADGROUP / embedding_dim);
        
        std::cout << "Option 1: Clamp dimensions" << std::endl;
        std::cout << "  Safe embedding_dim: " << safe_embedding_dim << std::endl;
        std::cout << "  Safe ffn_hidden_dim: " << safe_ffn_dim << std::endl;
        std::cout << "  Threads per tg: " << (safe_embedding_dim * safe_ffn_dim) << std::endl;
        
        // Option 2: 2D approach (keep one dimension, batch the other)
        std::cout << "\nOption 2: 2D dispatch patterns" << std::endl;
        std::cout << "  (1, embedding_dim, 1) = " << embedding_dim << " threads (reduction over h_idx in device memory)" << std::endl;
        std::cout << "  (1, 1, ffn_hidden_dim) = " << ffn_hidden_dim << " threads (reduction over e_idx in device memory)" << std::endl;
        
        // Option 3: Device memory reductions instead of threadgroup
        std::cout << "\nOption 3: Device memory reductions" << std::endl;
        std::cout << "  Use (1,1,1) threadgroups with atomic device memory accumulation" << std::endl;
        std::cout << "  Sacrifice some performance for correctness" << std::endl;
    }
    
    return 0;
} 