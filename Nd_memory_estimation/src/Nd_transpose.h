#include <cstdint>

namespace inplace {
namespace _Nd {
	template<typename T>
	size_t transpose(T* data, int source, void* dim, void* permutation, int rank);
}
}