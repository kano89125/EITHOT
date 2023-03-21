#include <cstdint>

namespace inplace {
namespace _4d {
	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double LARGE_RATIO);
	
	namespace _1324 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double LARGE_RATIO);
	}
	
	namespace _1432 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double LARGE_RATIO);
	}

	
	namespace _3214 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double LARGE_RATIO);
	}

}
}