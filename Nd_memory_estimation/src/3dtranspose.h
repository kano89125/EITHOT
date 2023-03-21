#include <cstdint>

namespace inplace {
namespace _3d {
	template<typename T>
	size_t transpose(T* data, int source, void* dim, int type);
	
	namespace _312 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3);
	}
	
	namespace _231 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3);
	}
	
	namespace _213 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3);
	}
	
	namespace _132 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3);
	}
}
}