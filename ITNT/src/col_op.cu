#include <cooperative_groups.h>
#include "util.h"
#include "equations.h"
#include "smem.h"
#include "debug.h"


/* 

all d1,d2, d3 is merged catanzaro dim in xd-transpose.cpp 

*/

namespace inplace {

namespace _4d {

namespace _1324 {

template<typename F, typename T> // 132
__global__ void smem_col_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	T* smem = shared_memory<T>();
	//size_t d3d4 = (size_t)d3 * (size_t)d4;
	size_t d1d2d3 = (size_t)d1 * (size_t)d2 * (size_t)d3;
	size_t d3d4 = (size_t)d3 * (size_t)d4;
	for (size_t id = 0; id < d3d4; id++) {
		size_t k = id % d3;
		size_t l = id / d3;
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				smem[threadIdx.x + i * blockDim.x] = data[source + j + i * d1 + kd1d2 + l * d1d2d3];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				size_t i_prime = fn(i, k);
				data[source + j + i * d1 + kd1d2 + l * d1d2d3] = smem[threadIdx.x + i_prime * blockDim.x];
			}
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	///PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2((double)x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, source, d1, d2, d3, d4);
}

template<typename F, typename T> // 132
__global__ void gmem_col_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3, int d4) {
	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();
	
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	size_t d1d3 = (size_t)d1 * (size_t)d3;
	size_t d1d2d3 = (size_t)d1 * (size_t)d2 * (size_t)d3;
	size_t d1d3d4 = d1d3 * (size_t)d4;
	for (size_t idx = threadIdx.x; idx < d1d3d4; idx += blockDim.x) {
		size_t j = idx % d1;
		size_t k = (idx / d1) % d3;
		size_t ld1d2d3 = (idx / d1d3) * d1d2d3;
		size_t remain_id = j + k * d1d2 + ld1d2d3;
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			tmp[threadIdx.x + i * blockDim.x] = data[source + i * d1 + remain_id];
		}
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			size_t i_prim = fn(i, k);
			data[source + i * d1 + remain_id] = tmp[threadIdx.x + i_prim * blockDim.x];
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
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
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	//PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3, (void *)&d4
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, source, d1, d2, d3, d4);
	}
	else {
		gmem_launch(fn, data, source, d1, d2, d3, d4);
	}
	
}

template void col_op(_2d::c2r::rotate, float*, int, int, int, int, int);
template void col_op(_2d::c2r::rotate, double*, int, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, float*, int, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, double*, int, int, int, int, int);

template void col_op(_2d::r2c::rotate, float*, int, int, int, int, int);
template void col_op(_2d::r2c::rotate, double*, int, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, float*, int, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, double*, int, int, int, int, int);

}

}

namespace _3d {

namespace _132 {

template<typename F, typename T>
__global__ void compress_col_op(F fn, T* data, int source, size_t batch_size, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1 * (size_t)d2;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d2; idx += blockDim.x) {
            smem[idx] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d2; idx += blockDim.x) {
            int u = (idx / d1d2);
	    size_t k = lv + u;
            size_t i = idx / d1 - u * d2;
            size_t j = idx % d1;
	    size_t i_prime = fn(i, k);
            data[source + offset + idx] = smem[j + i_prime * d1 + u * d1d2];
        }
    }
}

template<typename F, typename T>
void compress_col_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Compress d1d2 %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size = smem_lim / 32;
	int n_threads = 64;
	int n_blocks = min(d3, get_num_block(compress_col_op<F, T>, n_threads, smem_size));
	size_t batch_size = smem_size / (sizeof(T) * (size_t)d1 * (size_t)d2);
	//PRINT("\tBatch size = %zu\n", batch_size);
	compress_col_op<<<n_blocks, n_threads, smem_size>>>(fn, data, source, batch_size, d1, d2, d3);
}

template<typename F, typename T>
__global__ void small_d1d2_col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	T* smem = shared_memory<T>();
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = blockIdx.x; k < d3; k += gridDim.x) {
		size_t offset = k * d1d2;
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			smem[idx] = data[source + offset + idx];
		}
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t i = idx / d1;
			size_t i_prime = fn(i, k);
			data[source + offset + idx] = smem[j + i_prime * d1];
		}
	}
}

template<typename F, typename T>
void small_d1d2_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Small d1d2 %s\n", fn.getName().c_str());
	size_t smem_size = d1 * d2 * sizeof(T);
	int n_threads = 256;
	int n_blocks = min(d3, get_num_block(small_d1d2_col_op<F, T>, n_threads, smem_size));
	small_d1d2_col_op<<<n_blocks, n_threads, smem_size>>>(fn, data, source, d1, d2, d3);
}

template<typename F, typename T> // 132
__global__ void smem_col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	T* smem = shared_memory<T>();
	
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				smem[threadIdx.x + i * blockDim.x] = data[source + j + i * d1 + kd1d2];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				size_t i_prime = fn(i, k);
				data[source + j + i * d1 + kd1d2] = smem[threadIdx.x + i_prime * blockDim.x];
			}
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	///PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2((double)x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, source, d1, d2, d3);
}

template<typename F, typename T> // 132
__global__ void gmem_col_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();
	
	/*for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				tmp[threadIdx.x + i * blockDim.x] = data[soruce + j + i * d1 + kd1d2];
			}
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				size_t i_prim = fn(i, k);
				data[soruce + j + i * d1 + kd1d2] = tmp[threadIdx.x + i_prim * blockDim.x];
			}
		}
	}*/
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	size_t d1d3 = (size_t)d1 * (size_t)d3;
	for (size_t idx = threadIdx.x; idx < d1d3; idx += blockDim.x) {
		size_t j = idx % d1;
		size_t k = idx / d1;
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			tmp[threadIdx.x + i * blockDim.x] = data[source + j + i * d1 + k * d1d2];
		}
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			size_t i_prim = fn(i, k);
			data[source + j + i * d1 + k * d1d2] = tmp[threadIdx.x + i_prim * blockDim.x];
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
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
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	//PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * sizeof(T) * d1 * d2 <= smem_lim / 32) {
		compress_col_launch(fn, data, source, d1, d2, d3);
	}
	else if (sizeof(T) * d1 * d2 <= smem_lim) {
		small_d1d2_launch(fn, data, source, d1, d2, d3);
	}
	else if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, source, d1, d2, d3);
	}
	else {
		gmem_launch(fn, data, source, d1, d2, d3);
	}
}

template void col_op(_2d::c2r::rotate, float*, int, int, int, int);
template void col_op(_2d::c2r::rotate, double*, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, float*, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, double*, int, int, int, int);

template void col_op(_2d::r2c::rotate, float*, int, int, int, int);
template void col_op(_2d::r2c::rotate, double*, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, float*, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, double*, int, int, int, int);

}

namespace _213 {

template<typename F, typename T>
__global__ void smem_col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	T* smem = shared_memory<T>();
	
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				smem[threadIdx.x + i * blockDim.x] = data[source + j + i * d1 + kd1d2];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				size_t i_prime = fn(i, j);
				data[source + j + i * d1 + kd1d2] = smem[threadIdx.x + i_prime * blockDim.x];
			}
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2((double)x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, source, d1, d2, d3);
}

template<typename F, typename T>
__global__ void gmem_col_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();
	
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				tmp[threadIdx.x + i * blockDim.x] = data[source + j + i * d1 + kd1d2];
			}
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				size_t i_prim = fn(i, j);
				data[source + j + i * d1 + kd1d2] = tmp[threadIdx.x + i_prim * blockDim.x];
			}
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int source, int d1, int d2, int d3) {
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
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	//PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, source, d1, d2, d3);
	}
	else {
		gmem_launch(fn, data, source, d1, d2, d3);
	}
}

template void col_op(_2d::c2r::rotate, float*, int, int, int, int);
template void col_op(_2d::c2r::rotate, double*, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, float*, int, int, int, int);
template void col_op(_2d::c2r::col_shuffle, double*, int, int, int, int);

template void col_op(_2d::r2c::rotate, float*, int, int, int, int);
template void col_op(_2d::r2c::rotate, double*, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, float*, int, int, int, int);
template void col_op(_2d::r2c::col_shuffle, double*, int, int, int, int);

}

} //End of namespace _3d

namespace _2d {

template<typename F, typename T>
__global__ void smem_col_op(F fn, T* data, int source, int d1, int d2) {
	T* smem = shared_memory<T>();
	
	for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
		__syncthreads();
		for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
			smem[threadIdx.x + i * blockDim.x] = data[source + j + i * d1];
		}
		__syncthreads();
		for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
			size_t i_prime = fn(i, j);
			data[source + j + i * d1] = smem[threadIdx.x + i_prime * blockDim.x];
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int source, int d1, int d2) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2((double)x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);

	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, source, d1, d2);
}

template<typename F, typename T>
__global__ void gmem_col_op(F fn, T* data, int source, T* tmp, int d1, int d2) {
	namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
	
	for (size_t j = threadIdx.x; j < d1; j += blockDim.x) {
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			tmp[threadIdx.x + i * blockDim.x] = data[source + j + i * d1];
		}
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			size_t i_prim = fn(i, j);
			data[source + j + i * d1] = tmp[threadIdx.x + i_prim * blockDim.x];
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int source, int d1, int d2) {
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
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	//PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, source, d1, d2);
	}
	else {
		gmem_launch(fn, data, source, d1, d2);
	}
}

template void col_op(c2r::rotate, float*, int, int, int);
template void col_op(c2r::rotate, double*, int, int, int);
template void col_op(c2r::col_shuffle, float*, int, int, int);
template void col_op(c2r::col_shuffle, double*, int, int, int);

template void col_op(r2c::rotate, float*, int, int, int);
template void col_op(r2c::rotate, double*, int, int, int);
template void col_op(r2c::col_shuffle, float*, int, int, int);
template void col_op(r2c::col_shuffle, double*, int, int, int);

template void col_op(_3d::_132::row_permute, float*, int, int, int);
template void col_op(_3d::_132::row_permute, double*, int, int, int);

template void col_op(_4d::_1324::row_permute, float*, int, int, int);
template void col_op(_4d::_1324::row_permute, double*, int, int, int);

template void col_op(_4d::_1432::row_permute, float*, int, int, int);
template void col_op(_4d::_1432::row_permute, double*, int, int, int);

template void col_op(_5d::_15432::row_permute, float*, int, int, int);
template void col_op(_5d::_15432::row_permute, double*, int, int, int);

template void col_op(_5d::_14325::row_permute, float*, int, int, int);
template void col_op(_5d::_14325::row_permute, double*, int, int, int);

}
}
