#include <cooperative_groups.h>
#include "equations.h"
#include "util.h"
#include "smem.h"
#include "debug.h"

namespace inplace {

namespace _4d {

namespace _1324 {


template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
 	if (sizeof(T) * d1 * d3 * d4 <= smem_lim) {
		return 0;
	}
	else {
		return sizeof(T) * d1 * d3 * d4;
    	}
}

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
	if (sizeof(T) * d1 * d3 * d4 <= smem_lim) {
		return 0;
    	}
	else {
		return sizeof(T) * d1 * d3 * d4;
   	}
}

template size_t row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int, int);
template size_t row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int, int);

template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int, int);
template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int, int);


}

}

namespace _3d {

namespace _132 {
	
template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	/*if (2 * d1 * d3 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, d1, d2, d3);
    }
    else*/ if (sizeof(T) * d1 * d3 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1 * d3);
        return sizeof(T) * d1 * d3;
    }
}

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	/*if (2 * d1 * d3 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, d1, d2, d3);
    }
    else*/ if (sizeof(T) * d1 * d3 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1 * d3);
        return sizeof(T) * d1 * d3;
    }
}

template size_t row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int);
template size_t row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int);

template size_t row_gather_op(_2d::r2c::row_shuffle, float*, int, int, int, int);
template size_t row_gather_op(_2d::r2c::row_shuffle, double*, int, int, int, int);

template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int);
template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int);

}

namespace _213 {

template<typename F, typename K, typename T>
size_t gmem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	return sizeof(T) * d1;
}

template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		return 0;
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1);
        return sizeof(T) * d1;
    }
}

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	//printf("smem_lim = %zu\nrow size = %zu\n", smem_lim, sizeof(T) * (size_t)d1);
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		return 0;
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1);
        return sizeof(T) * d1;
    }
}

template size_t row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int);
template size_t row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int);

template size_t row_gather_op(_2d::r2c::row_shuffle, float*, int, int, int, int);
template size_t row_gather_op(_2d::r2c::row_shuffle, double*, int, int, int, int);

template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int);
template size_t row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int);

}
}

namespace _2d {

template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		return 0;
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1);
        return sizeof(T) * d1;
    }
}

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	//printf("smem_lim = %zu\nrow size = %zu\n", smem_lim, sizeof(T) * (size_t)d1);
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		return 0;
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		return 0;
    }
	else {
		//PRINT("Gmem Row op, size = %zu\n", sizeof(T) * d1);
        return sizeof(T) * d1;
    }
}

template size_t row_gather_op(c2r::row_shuffle, float*, int, int, int);
template size_t row_gather_op(c2r::row_shuffle, double*, int, int, int);

template size_t row_gather_op(r2c::row_shuffle, float*, int, int, int);
template size_t row_gather_op(r2c::row_shuffle, double*, int, int, int);

template size_t row_gather_op(_3d::_213::row_shuffle, float*, int, int, int);
template size_t row_gather_op(_3d::_213::row_shuffle, double*, int, int, int);

template size_t row_scatter_op(r2c::row_scatter_shuffle, float*, int, int, int);
template size_t row_scatter_op(r2c::row_scatter_shuffle, double*, int, int, int);

template size_t row_gather_op(_4d::_1324::row_shuffle, float*, int, int, int);
template size_t row_gather_op(_4d::_1324::row_shuffle, double*, int, int, int);

template size_t row_gather_op(_4d::_3214::row_shuffle, float*, int, int, int);
template size_t row_gather_op(_4d::_3214::row_shuffle, double*, int, int, int);

template size_t row_gather_op(_5d::_43215::row_shuffle, float*, int, int, int);
template size_t row_gather_op(_5d::_43215::row_shuffle, double*, int, int, int);

template size_t row_gather_op(_5d::_14325::row_shuffle, float*, int, int, int);
template size_t row_gather_op(_5d::_14325::row_shuffle, double*, int, int, int);

}

}
