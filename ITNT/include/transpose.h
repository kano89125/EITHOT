#pragma once

#include <cstdint>
#include <cstddef>

namespace inplace {

	void transpose(void* data, int source, int rank, void* dim, void* permutation, size_t sizeofType, int num_block, double LARGE_RATIO);

}
