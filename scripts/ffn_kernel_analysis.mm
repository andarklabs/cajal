//
// FFN Kernel Analysis - Analyze FFN backward kernel issues
//

#include <iostream>
#include <cmath>

int main() {
    std::cout << "🔍 FFN BACKWARD KERNEL ANALYSIS" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Our failing configuration
    int embedding_dim = 128;
    int ffn_hidden_dim = 256;
    int batch_size = 1;
    int sequence_length = 16;
    int num_instances = batch_size * sequence_length;
    
    std::cout << "📊 CONFIGURATION:" << std::endl;
    std::cout << "   embedding_dim: " << embedding_dim << std::endl;
    std::cout << "   ffn_hidden_dim: " << ffn_hidden_dim << std::endl;
    std::cout << "   num_instances: " << num_instances << std::endl;
    
    std::cout << "\n🔍 THREADGROUP BUFFER ANALYSIS:" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Check buffer sizes vs limits
    std::cout << "tg_sum_for_dLdHact[1024]:" << std::endl;
    if (embedding_dim <= 1024) {
        std::cout << "  ✅ embedding_dim (" << embedding_dim << ") <= 1024 - OK" << std::endl;
    } else {
        std::cout << "  ❌ embedding_dim (" << embedding_dim << ") > 1024 - OVERFLOW!" << std::endl;
    }
    
    std::cout << "tg_sum_for_dLdX[2048]:" << std::endl;
    if (ffn_hidden_dim <= 2048) {
        std::cout << "  ✅ ffn_hidden_dim (" << ffn_hidden_dim << ") <= 2048 - OK" << std::endl;
    } else {
        std::cout << "  ❌ ffn_hidden_dim (" << ffn_hidden_dim << ") > 2048 - OVERFLOW!" << std::endl;
    }
    
    std::cout << "\n🔍 REDUCTION ALGORITHM ANALYSIS:" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Check if dimensions are powers of 2
    auto isPowerOf2 = [](int n) {
        return n > 0 && (n & (n - 1)) == 0;
    };
    
    std::cout << "embedding_dim (" << embedding_dim << "):" << std::endl;
    if (isPowerOf2(embedding_dim)) {
        std::cout << "  ✅ Power of 2 - Reduction algorithm OK" << std::endl;
    } else {
        std::cout << "  ⚠️  NOT power of 2 - Reduction may access invalid indices!" << std::endl;
        
        // Simulate the reduction to find issues
        std::cout << "  🔍 Reduction simulation:" << std::endl;
        for (int s = embedding_dim / 2; s > 0; s >>= 1) {
            int max_access = embedding_dim - 1 + s;
            std::cout << "     s=" << s << " → max access: index " << max_access;
            if (max_access >= embedding_dim) {
                std::cout << " ❌ OUT OF BOUNDS!";
            } else {
                std::cout << " ✅ OK";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\nffn_hidden_dim (" << ffn_hidden_dim << "):" << std::endl;
    if (isPowerOf2(ffn_hidden_dim)) {
        std::cout << "  ✅ Power of 2 - Reduction algorithm OK" << std::endl;
    } else {
        std::cout << "  ⚠️  NOT power of 2 - Reduction may access invalid indices!" << std::endl;
        
        // Simulate the reduction to find issues
        std::cout << "  🔍 Reduction simulation:" << std::endl;
        for (int s = ffn_hidden_dim / 2; s > 0; s >>= 1) {
            int max_access = ffn_hidden_dim - 1 + s;
            std::cout << "     s=" << s << " → max access: index " << max_access;
            if (max_access >= ffn_hidden_dim) {
                std::cout << " ❌ OUT OF BOUNDS!";
            } else {
                std::cout << " ✅ OK";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n🔍 DISPATCH ANALYSIS:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // FFN kernel uses 3D dispatch: (instance_idx, h_idx, e_idx)
    std::cout << "FFN backward kernel dispatch:" << std::endl;
    std::cout << "  Grid: (" << num_instances << ", " << ffn_hidden_dim << ", " << embedding_dim << ")" << std::endl;
    std::cout << "  Threadgroup: (1, 1, 1)" << std::endl;
    std::cout << "  Total threads: " << (num_instances * ffn_hidden_dim * embedding_dim) << std::endl;
    std::cout << "  Total threadgroups: " << (num_instances * ffn_hidden_dim * embedding_dim) << std::endl;
    
    int total_threads = num_instances * ffn_hidden_dim * embedding_dim;
    if (total_threads > 1000000) {
        std::cout << "  ⚠️  Very high thread count - potential performance issue" << std::endl;
    }
    
    std::cout << "\n🎯 CRITICAL ISSUES IDENTIFIED:" << std::endl;
    std::cout << "===============================" << std::endl;
    
    bool has_critical_issues = false;
    
    if (!isPowerOf2(embedding_dim)) {
        std::cout << "❌ CRITICAL: embedding_dim (" << embedding_dim << ") not power of 2" << std::endl;
        std::cout << "   → Reduction over e_idx may access out-of-bounds threadgroup memory" << std::endl;
        std::cout << "   → Can cause garbage values → NaN propagation" << std::endl;
        has_critical_issues = true;
    }
    
    if (!isPowerOf2(ffn_hidden_dim)) {
        std::cout << "❌ CRITICAL: ffn_hidden_dim (" << ffn_hidden_dim << ") not power of 2" << std::endl;
        std::cout << "   → Reduction over h_idx may access out-of-bounds threadgroup memory" << std::endl;
        std::cout << "   → Can cause garbage values → NaN propagation" << std::endl;
        has_critical_issues = true;
    }
    
    if (!has_critical_issues) {
        std::cout << "✅ No obvious threadgroup buffer or reduction issues found" << std::endl;
        std::cout << "   → Problem may be in threadgroup variable synchronization" << std::endl;
        std::cout << "   → Or in atomic operations causing race conditions" << std::endl;
    }
    
    std::cout << "\n🔧 SOLUTIONS:" << std::endl;
    std::cout << "=============" << std::endl;
    std::cout << "1. Fix reduction algorithm to handle non-power-of-2 dimensions" << std::endl;
    std::cout << "2. Add bounds checking in threadgroup array access" << std::endl;
    std::cout << "3. Initialize unused threadgroup elements to 0.0" << std::endl;
    std::cout << "4. Review threadgroup variable synchronization" << std::endl;
    
    return 0;
} 