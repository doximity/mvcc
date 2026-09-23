// mvcc-mslc: compile an MSL file with the system Metal compiler (runtime path, no Xcode needed)
// and report diagnostics + per-kernel pipeline stats. Exit code 0 on success.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: mvcc-mslc file.metal [--metal4] [--quiet]\n"); return 2; }
        bool metal4 = true, quiet = false;
        NSString* path = nil;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--metal3")) metal4 = false;
            else if (!strcmp(argv[i], "--quiet")) quiet = true;
            else path = [NSString stringWithUTF8String:argv[i]];
        }
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read %s: %s\n", path.UTF8String, err.localizedDescription.UTF8String); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = metal4 ? MTLLanguageVersion4_0 : MTLLanguageVersion3_2;
        opts.mathMode = MTLMathModeSafe;  // we control fast-math per call site
        auto t0 = std::chrono::steady_clock::now();
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        auto t1 = std::chrono::steady_clock::now();
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        if (err && !quiet) fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);  // warnings
        int rc = 0;
        for (NSString* name in lib.functionNames) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            auto p0 = std::chrono::steady_clock::now();
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            auto p1 = std::chrono::steady_clock::now();
            if (!ps) { fprintf(stderr, "%s: pipeline error: %s\n", name.UTF8String, err.localizedDescription.UTF8String); rc = 1; continue; }
            if (!quiet)
                printf("%s: ok  maxThreads/tg=%lu  width=%lu  staticTG=%lu  pipeline=%.0fms\n", name.UTF8String,
                       (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.threadExecutionWidth,
                       (unsigned long)ps.staticThreadgroupMemoryLength, std::chrono::duration<double, std::milli>(p1 - p0).count());
        }
        if (!quiet) printf("library compile: %.0f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
        return rc;
    }
}
