#include <cstdint>

namespace inplace {
namespace _4d {
	template<typename T>
	size_t transpose(T* data, int source, void* dim, int type);
	
	namespace _1324 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4);
	}
	
	namespace _1432 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4);
	}
	
	namespace _3214 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4);
	}
	
}
}