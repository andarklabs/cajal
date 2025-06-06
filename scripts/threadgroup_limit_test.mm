//
// Threadgroup Limit Test - Check Metal limits on M3 Max
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    std::cout << "🔍 METAL THREADGROUP LIMITS TEST" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "❌ Failed to create Metal device" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Metal Device: " << [[device name] UTF8String] << std::endl;
    
    // Check device capabilities
    std::cout << "\n🔧 DEVICE LIMITS:" << std::endl;
    std::cout << "Max threads per threadgroup: " << device.maxThreadsPerThreadgroup << std::endl;
    
    // Create a library with our kernels
    NSError* error = nil;
    NSString* libraryPath = @"src/msl/backward_kernels.metal";
    
    // Try to compile a simple test kernel to check limits
    NSString* kernelSource = @"
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void test_kernel(
        device float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint tid [[thread_position_in_threadgroup]],
        uint gid [[threadgroup_position_in_grid]]
    ) {
        output[gid * 1024 + tid] = input[gid * 1024 + tid] * 2.0f;
    }
    ";
    
    id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
    if (!library) {
        std::cerr << "❌ Failed to create library: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"test_kernel"];
    if (!function) {
        std::cerr << "❌ Failed to create function" << std::endl;
        return 1;
    }
    
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::cerr << "❌ Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }
    
    std::cout << "\n🧪 KERNEL LIMITS:" << std::endl;
    std::cout << "Max total threads per threadgroup: " << pipeline.maxTotalThreadsPerThreadgroup << std::endl;
    std::cout << "Thread execution width: " << pipeline.threadExecutionWidth << std::endl;
    
    // Test specific threadgroup sizes
    std::cout << "\n🧪 THREADGROUP SIZE TESTS:" << std::endl;
    
    struct TestCase {
        int size;
        const char* description;
    };
    
    TestCase tests[] = {
        {32, "embedding_dim=32 (works)"},
        {64, "embedding_dim=64 (fails)"},
        {128, "embedding_dim=128 (fails)"},
        {256, "embedding_dim=256"},
        {512, "embedding_dim=512"},
        {1024, "embedding_dim=1024 (threadgroup array limit)"}
    };
    
    for (const auto& test : tests) {
        if (test.size <= pipeline.maxTotalThreadsPerThreadgroup) {
            std::cout << "✅ " << test.description << " - SUPPORTED (limit: " << pipeline.maxTotalThreadsPerThreadgroup << ")" << std::endl;
        } else {
            std::cout << "❌ " << test.description << " - EXCEEDS LIMIT (limit: " << pipeline.maxTotalThreadsPerThreadgroup << ")" << std::endl;
        }
    }
    
    // Check if 64 is problematic for some other reason
    if (64 <= pipeline.maxTotalThreadsPerThreadgroup) {
        std::cout << "\n💡 ANALYSIS: embedding_dim=64 should be supported by Metal limits" << std::endl;
        std::cout << "🎯 The NaN bug is NOT due to basic threadgroup size limits" << std::endl;
        std::cout << "🔍 Need to look for other issues (buffer overflow, indexing, etc.)" << std::endl;
    } else {
        std::cout << "\n🎯 FOUND THE ISSUE: embedding_dim=64 exceeds Metal threadgroup limits!" << std::endl;
        std::cout << "✅ This explains why the NaN bug appears exactly at embedding_dim > 32" << std::endl;
    }
    
    return 0;
} 