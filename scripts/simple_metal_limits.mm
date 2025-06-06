//
// Simple Metal Limits Test
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    std::cout << "🔍 METAL DEVICE LIMITS TEST" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "❌ Failed to create Metal device" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Metal Device: " << [[device name] UTF8String] << std::endl;
    
    // Check device capabilities
    std::cout << "\n🔧 DEVICE LIMITS:" << std::endl;
    
    // MTLSize is a struct with width, height, depth
    // Let's check the maxThreadsPerThreadgroup property
    MTLSize maxThreads = device.maxThreadsPerThreadgroup;
    std::cout << "Max threads per threadgroup: " << maxThreads.width << " x " << maxThreads.height << " x " << maxThreads.depth << std::endl;
    std::cout << "Max total threads: " << (maxThreads.width * maxThreads.height * maxThreads.depth) << std::endl;
    
    // Test specific threadgroup sizes that matter for our kernels
    std::cout << "\n🧪 EMBEDDING DIM ANALYSIS:" << std::endl;
    
    struct TestCase {
        int size;
        const char* description;
        bool observed_working;
    };
    
    TestCase tests[] = {
        {32, "embedding_dim=32", true},
        {64, "embedding_dim=64", false},
        {128, "embedding_dim=128", false},
        {256, "embedding_dim=256", false},
        {512, "embedding_dim=512", false},
        {1024, "embedding_dim=1024", false}
    };
    
    uint64_t max_1d_threads = maxThreads.width;
    
    for (const auto& test : tests) {
        std::string status = test.observed_working ? "✅ WORKS" : "❌ FAILS";
        std::string limit_status = (test.size <= max_1d_threads) ? "WITHIN LIMITS" : "EXCEEDS LIMITS";
        
        std::cout << test.description << ": " << status << " (" << limit_status << ")" << std::endl;
    }
    
    // Analysis
    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "============" << std::endl;
    
    if (64 <= max_1d_threads) {
        std::cout << "💡 embedding_dim=64 is WITHIN Metal limits (max: " << max_1d_threads << ")" << std::endl;
        std::cout << "🎯 The NaN bug is NOT due to basic threadgroup size limits" << std::endl;
        std::cout << "🔍 Need to investigate other potential causes:" << std::endl;
        std::cout << "   - Threadgroup memory usage with larger embedding_dim" << std::endl;
        std::cout << "   - Buffer indexing errors at higher dimensions" << std::endl;  
        std::cout << "   - Kernel dispatch configuration issues" << std::endl;
        std::cout << "   - Memory alignment or precision issues" << std::endl;
    } else {
        std::cout << "🎯 FOUND ISSUE: embedding_dim=64 exceeds Metal 1D threadgroup limit!" << std::endl;
        std::cout << "✅ Max 1D threads: " << max_1d_threads << ", but we need: 64" << std::endl;
    }
    
    // Check if our layer norm implementation might be the issue
    std::cout << "\n🔧 LAYER NORM KERNEL ANALYSIS:" << std::endl;
    std::cout << "Our layer_norm_backward uses embedding_dim as threadgroup size" << std::endl;
    std::cout << "Required: embedding_dim threads per threadgroup" << std::endl;
    std::cout << "Available: " << max_1d_threads << " threads per threadgroup (1D)" << std::endl;
    
    if (64 <= max_1d_threads) {
        std::cout << "✅ Layer norm threadgroup size should work for embedding_dim=64" << std::endl;
    } else {
        std::cout << "❌ Layer norm threadgroup size is the problem!" << std::endl;
    }
    
    return 0;
} 