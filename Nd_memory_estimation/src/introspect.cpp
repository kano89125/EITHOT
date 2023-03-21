#include "introspect.h"

namespace {
    introspect cached_properties;
}

int n_sms() {
    return cached_properties.properties.multiProcessorCount;
}

size_t gpu_memory_size() {
    return cached_properties.properties.totalGlobalMem;
}

size_t shared_mem_per_block() {
    return cached_properties.properties.sharedMemPerBlock;
}

int max_n_threads_per_sm() {
	return cached_properties.properties.maxThreadsPerMultiProcessor;
}

int current_sm() {
    return cached_properties.properties.major * 100 +
        cached_properties.properties.minor;
}
