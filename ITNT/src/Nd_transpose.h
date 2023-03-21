#include <cstdint>

namespace inplace {
namespace _Nd {
	template<typename T>
	void transpose(T* data, int source, void* dim, void* permutation, int rank, int num_block, double LARGE_RATIO);
	//void transpose(T* data, int source, void* dim, void* permutation, int rank);
}
}