#include <cstdint>

namespace inplace {
namespace _2d {
	template<typename T>
	void transpose(T* data, int source, void* dim, int tensor_vol, double LARGE_RATIO);
	
	namespace c2r {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2);
	}

	namespace r2c {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2);
	}
}
}
