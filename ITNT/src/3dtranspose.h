#include <cstdint>

namespace inplace {
namespace _3d {
	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double LARGE_RATIO);
	
	namespace _231 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double LARGE_RATIO);
	}

	namespace _312 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double LARGE_RATIO);
	}
	
	namespace _213 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double LARGE_RATIO);
	}
	
	namespace _132 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double LARGE_RATIO);
	}
}
}