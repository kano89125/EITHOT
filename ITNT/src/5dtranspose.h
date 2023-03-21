#include <cstdint>

namespace inplace {
namespace _5d {
	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double LARGE_RATIO);
	
	namespace _14325 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double LARGE_RATIO);
	}

	namespace _15432 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double LARGE_RATIO);
	}
	namespace _43215 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double LARGE_RATIO);
	}
}
}