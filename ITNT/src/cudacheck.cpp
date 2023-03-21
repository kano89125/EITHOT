#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
}
