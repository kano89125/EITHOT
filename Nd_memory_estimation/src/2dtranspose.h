#include <cstdint>

namespace inplace {
namespace _2d {
	template<typename T>
	size_t transpose(T* data, int source, void* dim);
	
	namespace c2r {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2);
	}

	namespace r2c {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2);
	}
}
}
