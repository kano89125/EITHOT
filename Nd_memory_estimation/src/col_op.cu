#include <cooperative_groups.h>
#include "util.h"
#include "equations.h"
#include "smem.h"
#include "debug.h"

namespace inplace {


namespace _4d {

namespace _1324 {


template<typename F, typename T>
size_t col_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
	int n_threads = 256;
	_2d::c2r::rotate r1;
	_2d::r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		return 0;
	}
	else {
		return sizeof(T) * n_threads * d2;
	}
	
}

template size_t col_op(_2d::c2r::rotate, float*, int, int, int, int, int);
template size_t col_op(_2d::c2r::rotate, double*, int, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, float*, int, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, double*, int, int, int, int, int);

template size_t col_op(_2d::r2c::rotate, float*, int, int, int, int, int);
template size_t col_op(_2d::r2c::rotate, double*, int, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, float*, int, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, double*, int, int, int, int, int);


}
}

namespace _3d {

namespace _132 {

template<typename F, typename T>
size_t gmem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	int n_threads = 256;
	_2d::c2r::rotate r1;
	_2d::r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}

	size_t tmp_size = sizeof(T) * n_threads * d2;
	return tmp_size;
}

template<typename F, typename T>
size_t col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * sizeof(T) * d1 * d2 <= smem_lim / 32) {
		return 0;
	}
	else if (sizeof(T) * d1 * d2 <= smem_lim) {
		return 0;
	}
	else if (smem_lim / (sizeof(T) * d2) >= 16) {
		return 0;
	}
	else {
		size_t tmp = gmem_launch(fn, data, source, d1, d2, d3);
		//PRINT("Gmem Col op, size = %zu\n", tmp);
		return tmp;
	}
}

template size_t col_op(_2d::c2r::rotate, float*, int, int, int, int);
template size_t col_op(_2d::c2r::rotate, double*, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, float*, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, double*, int, int, int, int);

template size_t col_op(_2d::r2c::rotate, float*, int, int, int, int);
template size_t col_op(_2d::r2c::rotate, double*, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, float*, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, double*, int, int, int, int);

}

namespace _213 {

template<typename F, typename T>
size_t gmem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	int n_threads = 256;
	_2d::c2r::rotate r1;
	_2d::r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}
	size_t tmp_size = sizeof(T) * n_threads * d2;
	return tmp_size;
}

template<typename F, typename T>
size_t col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		return 0;
	}
	else {
		size_t tmp = gmem_launch(fn, data, source, d1, d2, d3);
		//PRINT("Gmem Col op, size = %zu\n", tmp);
		return tmp;
	}
}

template size_t col_op(_2d::c2r::rotate, float*, int, int, int, int);
template size_t col_op(_2d::c2r::rotate, double*, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, float*, int, int, int, int);
template size_t col_op(_2d::c2r::col_shuffle, double*, int, int, int, int);

template size_t col_op(_2d::r2c::rotate, float*, int, int, int, int);
template size_t col_op(_2d::r2c::rotate, double*, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, float*, int, int, int, int);
template size_t col_op(_2d::r2c::col_shuffle, double*, int, int, int, int);

}

} //End of namespace _3d

namespace _2d {

template<typename F, typename T>
size_t gmem_launch(F fn, T* data, int source, int d1, int d2) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	int n_threads = 256;
	c2r::rotate r1;
	r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}
	//PRINT("d2 = %d\n", d2);
	size_t tmp_size = sizeof(T) * n_threads * d2;
	return tmp_size;
}

template<typename F, typename T>
size_t col_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		return 0;
	}
	else {
		size_t tmp = gmem_launch(fn, data, source, d1, d2);
		//PRINT("Gmem Col op, size = %zu\n", tmp);
		return tmp;
	}
}

template size_t col_op(c2r::rotate, float*, int, int, int);
template size_t col_op(c2r::rotate, double*, int, int, int);
template size_t col_op(c2r::col_shuffle, float*, int, int, int);
template size_t col_op(c2r::col_shuffle, double*, int, int, int);

template size_t col_op(r2c::rotate, float*, int, int, int);
template size_t col_op(r2c::rotate, double*, int, int, int);
template size_t col_op(r2c::col_shuffle, float*, int, int, int);
template size_t col_op(r2c::col_shuffle, double*, int, int, int);

template size_t col_op(_3d::_132::row_permute, float*, int, int, int);
template size_t col_op(_3d::_132::row_permute, double*, int, int, int);

template size_t col_op(_4d::_1324::row_permute, float*, int, int, int);
template size_t col_op(_4d::_1324::row_permute, double*, int, int, int);

template size_t col_op(_4d::_1432::row_permute, float*, int, int, int);
template size_t col_op(_4d::_1432::row_permute, double*, int, int, int);

template size_t col_op(_5d::_15432::row_permute, float*, int, int, int);
template size_t col_op(_5d::_15432::row_permute, double*, int, int, int);

template size_t col_op(_5d::_14325::row_permute, float*, int, int, int);
template size_t col_op(_5d::_14325::row_permute, double*, int, int, int);

}
}
