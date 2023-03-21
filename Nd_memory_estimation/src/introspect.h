#pragma once

#include "cuda_runtime.h"

struct introspect {
    int device;
    cudaDeviceProp properties;
    introspect() {
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&properties, device);
    }
};

int n_sms();
size_t gpu_memory_size();
size_t shared_mem_per_block();
int max_n_threads_per_sm();
int current_sm();