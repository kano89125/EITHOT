#include <cooperative_groups.h>
#include "equations.h"
#include "util.h"
#include "smem.h"
#include "debug.h"


/* 

all d1,d2, d3 is merged catanzaro dim in xd-transpose.cpp 

*/

namespace inplace {

namespace _4d {

namespace _1324 {

template<typename F, typename T>
__global__ void smem_row_gather_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
    T* smem = shared_memory<T>();
    
	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	size_t d1d2d3 = d1d2 * size_t(d3);
	size_t d1d3d4 = d1d3 * size_t(d4);
	for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
		fn.set_i(i);
		size_t id1 = i * d1;
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3d4; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			smem[idx] = data[source + j + id1 + k * d1d2 + l * d1d2d3];
		}
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3d4; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			data[source + j + id1 + k * d1d2 + l * d1d2d3] = smem[j + fn(k) * d1 + l * d1d3];
		}
	}
	
}

template<typename F, typename T>
__global__ void smem_row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
    	T* smem = shared_memory<T>();
    
	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	size_t d1d2d3 = (size_t)d1d2 * (size_t)d3;
	size_t d1d3d4 = d1d3 * (size_t)d4;
	for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
		fn.set_i(i);
		size_t id1 = i * d1;
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3d4; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			smem[j + fn(k) * d1 + l * d1d3] = data[source + j + id1 + k * d1d2 + l * d1d2d3];
		}
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3d4; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			data[source + j + id1 + k * d1d2 + l * d1d2d3] = smem[idx];
		}
	}
}


template<typename F, typename T>
__global__ void gmem_row_gather_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3, int d4) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	size_t d1d2d3 = d1d2 * size_t(d3);
	size_t d1d3d4 = d1d3 * size_t(d4);
    	size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    	size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		size_t id1 = i * d1;
		g.sync();
		for (size_t idx = global_id; idx < d1d3d4; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			tmp[idx] = data[source + j + id1 + fn(k) * d1d2 + l * d1d2d3];
		}
		g.sync();
		for (size_t idx = global_id; idx < d1d3d4; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			data[source + j + id1 + k * d1d2 + l * d1d2d3] = tmp[idx];
		}
	}
}

template<typename F, typename T> // 132
__global__ void gmem_row_scatter_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3, int d4) {
    	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();

	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	size_t d1d3d4 = d1d3 * (size_t)d4;
	size_t d1d2d3 = d1d2 * (size_t)d3;
   	size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
   	size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		size_t id1 = i * d1;
		g.sync();
		for (size_t idx = global_id; idx < d1d3d4; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			tmp[j + fn(k) * d1 + l * d1d3] = data[source + j + id1 + k * d1d2 + l * d1d2d3];
		}
		g.sync();
		for (size_t idx = global_id; idx < d1d3d4; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = (idx / d1) % d3;
			size_t l = idx / d1d3;
			data[source + j + id1 + k * d1d2 + l * d1d2d3] = tmp[idx];
		}
	}
}

template<typename F, typename K, typename T>
void smem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3, int d4) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	int d1d3d4 = d1 * d3 * d4;
	size_t smem_size = sizeof(T) * (size_t)d1d3d4;
	int n_threads = get_num_thread(d1d3d4);
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, d1, d2, d3, d4);
}

template<typename F, typename K, typename T>
void gmem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3, int d4) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	T* tmp;
	size_t tmp_size = sizeof(T) * d1 * d3 * d4;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	int n_threads = 512;
	int n_blocks = get_num_block(kernel, n_threads, 0);
	//PRINT("\t# blocks = %d\n", n_blocks);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3, (void *)&d4
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)kernel, n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void row_gather_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
 	if (sizeof(T) * d1 * d3 * d4 <= smem_lim) {
		smem_row_launch(fn, smem_row_gather_op<F, T>, data, source, d1, d2, d3, d4);
    }
	else {
        gmem_row_launch(fn, gmem_row_gather_op<F, T>, data, source, d1, d2, d3, d4);
    }
}

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3, int d4) {
	size_t smem_lim = shared_mem_per_block();
	/*if (2 * d1 * d3 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, d1, d2, d3, d4);
    }
    else*/ if (sizeof(T) * d1 * d3 * d4 <= smem_lim) {
		smem_row_launch(fn, smem_row_scatter_op<F, T>, data, source, d1, d2, d3, d4);
    }
	else {
        gmem_row_launch(fn, gmem_row_scatter_op<F, T>, data, source, d1, d2, d3, d4);
    }
}

template void row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int, int);
template void row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int, int);

template void row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int, int);
template void row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int, int);


}

}

namespace _3d {

namespace _132 {

template<typename F, typename T>
__global__ void compress_row_gather_op(F fn, T* data, int source, size_t batch_size, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d2);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	size_t d1d3 = (size_t)d1 * (size_t)d3;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d3; idx += blockDim.x) {
            smem[idx] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d3; idx += blockDim.x) {
            int u = (idx / d1d3);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
			size_t k = idx / d1;
            fn.set_i(i);
            data[source + offset + idx] = smem[j + u * d1 + fn(k) * d1d2];
        }
    }
}

template<typename F, typename T>
__global__ void compress_row_scatter_op(F fn, T* data, int source, size_t batch_size, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d2);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	size_t d1d3 = (size_t)d1 * (size_t)d3;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        //size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d3; idx += blockDim.x) {
			int u = (idx / d1d3);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
			size_t k = idx / d1;
            fn.set_i(i);
            //smem[j + u * d1 + fn(k) * d1d2] = data[offset + idx];
			smem[j + u * d1 + fn(k) * d1d2] = data[source + j + i * d1 + k * d1d2];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1d3; idx += blockDim.x) {
            //data[offset + idx] = smem[idx];
			int u = (idx / d1d3);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
			size_t k = idx / d1;
			data[source + j + i * d1 + k * d1d2] = smem[idx];
        }
    }
}

template<typename F, typename K, typename T>
void compress_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Smem Compress %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size = smem_lim / 32;
	int n_threads = max_n_threads_per_sm() / 32;
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	//PRINT("\t# blocks = %d\n", n_blocks);
	size_t batch_size = smem_size / (sizeof(T) * (size_t)d1 * (size_t)d3);
	//PRINT("\tbatch size = %zu\n", batch_size);
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, batch_size, d1, d2, d3);
}

template<typename F, typename T>
__global__ void smem_row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();
    
	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
		fn.set_i(i);
		size_t id1 = i * d1;
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			smem[idx] = data[source + j + id1 + k * d1d2];
		}
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			data[source + j + id1 + k * d1d2] = smem[j + fn(k) * d1];
		}
	}
}

template<typename F, typename T>
__global__ void smem_row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();
    
	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
	for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
		fn.set_i(i);
		size_t id1 = i * d1;
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			smem[j + fn(k) * d1] = data[source + j + id1 + k * d1d2];
		}
		__syncthreads();
		for(size_t idx = threadIdx.x; idx < d1d3; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			data[source + j + id1 + k * d1d2] = smem[idx];
		}
	}
}

template<typename F, typename K, typename T>
void smem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_size = sizeof(T) * (size_t)d1 * (size_t)d3;
	int n_threads = get_num_thread(d1 * d3);
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, d1, d2, d3);
}

template<typename F, typename T>
__global__ void gmem_row_gather_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		size_t id1 = i * d1;
		g.sync();
		for (size_t idx = global_id; idx < d1d3; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			tmp[idx] = data[source + j + id1 + fn(k) * d1d2];
		}
		g.sync();
		for (size_t idx = global_id; idx < d1d3; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			data[source + j + id1 + k * d1d2] = tmp[idx];
		}
	}
}

template<typename F, typename T> // 132
__global__ void gmem_row_scatter_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

	size_t d1d2 = (size_t)d1 * size_t(d2);
	size_t d1d3 = (size_t)d1 * size_t(d3);
    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		size_t id1 = i * d1;
		g.sync();
		for (size_t idx = global_id; idx < d1d3; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			tmp[j + fn(k) * d1] = data[source + j + id1 + k * d1d2];
		}
		g.sync();
		for (size_t idx = global_id; idx < d1d3; idx += grid_size) {
			size_t j = idx % d1;
			size_t k = idx / d1;
			data[source + j + id1 + k * d1d2] = tmp[idx];
		}
	}
}

template<typename F, typename K, typename T>
void gmem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	T* tmp;
	size_t tmp_size = sizeof(T) * d1 * d3;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	int n_threads = 1024;
	int n_blocks = get_num_block(kernel, n_threads, 0);
	//PRINT("\t# blocks = %d\n", n_blocks);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)kernel,
										  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	/*if (2 * d1 * d3 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_gather_op<F, T>, data, d1, d2, d3);
    }
    else*/ if (sizeof(T) * d1 * d3 <= smem_lim) {
		smem_row_launch(fn, smem_row_gather_op<F, T>, data, source, d1, d2, d3);
    }
	else {
        gmem_row_launch(fn, gmem_row_gather_op<F, T>, data, source, d1, d2, d3);
    }
}

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	/*if (2 * d1 * d3 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, d1, d2, d3);
    }
    else*/ if (sizeof(T) * d1 * d3 <= smem_lim) {
		smem_row_launch(fn, smem_row_scatter_op<F, T>, data, source, d1, d2, d3);
    }
	else {
        gmem_row_launch(fn, gmem_row_scatter_op<F, T>, data, source, d1, d2, d3);
    }
}

template void row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int);
template void row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int);

template void row_gather_op(_2d::r2c::row_shuffle, float*, int, int, int, int);
template void row_gather_op(_2d::r2c::row_shuffle, double*, int, int, int, int);

template void row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int);
template void row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int);

}

namespace _213 {

template<typename F, typename T>
__global__ void compress_row_gather_op(F fn, T* data, int source, size_t batch_size, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();

	size_t d2d3 = (size_t)d2 * (size_t)d3;
    size_t l = chunk_left(blockIdx.x, gridDim.x, d2d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2d3);
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            smem[idx] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            fn.set_i(i);
            data[source + offset + idx] = smem[fn(j) + u * d1];
        }
    }
}

template<typename F, typename T>
__global__ void compress_row_scatter_op(F fn, T* data, int source, size_t batch_size, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();

    size_t d2d3 = (size_t)d2 * (size_t)d3;
    size_t l = chunk_left(blockIdx.x, gridDim.x, d2d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2d3);
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
			int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            fn.set_i(i);
            smem[fn(j) + u * d1] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            data[source + offset + idx] = smem[idx];
        }
    }
}

template<typename F, typename K, typename T>
void compress_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Smem Compress %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size = smem_lim / 32;
	int n_threads = max_n_threads_per_sm() / 32;
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2 * d3, get_num_block(kernel, n_threads, smem_size));
	//PRINT("\t# blocks = %d\n", n_blocks);
	size_t batch_size = smem_size / (sizeof(T) * (size_t)d1);
	//PRINT("\tbatch size = %zu\n", batch_size);
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, batch_size, d1, d2, d3);
}

template<typename F, typename T>
__global__ void smem_row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();
    
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			fn.set_i(i);
			__syncthreads();
			for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
				smem[j] = data[source + j + i * d1 + kd1d2];
			}
			__syncthreads();
			for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
				data[source + j + i * d1 + kd1d2] = smem[fn(j)];
			}
		}
	}
}

template<typename F, typename T>
__global__ void smem_row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
    T* smem = shared_memory<T>();
    
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			fn.set_i(i);
			__syncthreads();
			for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
				smem[fn(j)] = data[source + j + i * d1 + kd1d2];
			}
			__syncthreads();
			for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
				data[source + j + i * d1 + kd1d2] = smem[j];
			}
		}
	}
}

template<typename F, typename K, typename T>
void smem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_size = sizeof(T) * (size_t)d1;
	int n_threads = get_num_thread(d1);
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, d1, d2, d3);
}



template<typename F, typename T>
__global__ void gmem_row_gather_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t i = 0; i < d2; i++) {
			fn.set_i(i);
			g.sync();
			for (size_t j = global_id; j < d1; j += grid_size) {
				tmp[j] = data[source + fn(j) + i * d1 + kd1d2];
			}
			g.sync();
			for (size_t j = global_id; j < d1; j += grid_size) {
				data[source + j + i * d1 + kd1d2] = tmp[j];
			}
		}
	}
}

template<typename F, typename T> // 213
__global__ void gmem_row_scatter_op(F fn, T* data, int source, T* tmp, int d1, int d2, int d3) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t i = 0; i < d2; i++) {
			fn.set_i(i);
			g.sync();
			for (size_t j = global_id; j < d1; j += grid_size) {
				//tmp[j] = d[rm(i, s(j))];
				tmp[fn(j)] = data[source + j + i * d1 + kd1d2];
			}
			g.sync();
			for (size_t j = global_id; j < d1; j += grid_size) {
				//d[rm(i, j)] = tmp[j];
				data[source + j + i * d1 + kd1d2] = tmp[j];
			}
		}
	}
}

template<typename F, typename K, typename T>
void gmem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2, int d3) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	T* tmp;
	size_t tmp_size = sizeof(T) * d1;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	int n_threads = 1024;
	int n_blocks = get_num_block(kernel, n_threads, 0);
	//PRINT("\t# blocks = %d\n", n_blocks);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)kernel,
										  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void row_gather_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_gather_op<F, T>, data, source, d1, d2, d3);
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		smem_row_launch(fn, smem_row_gather_op<F, T>, data, source, d1, d2, d3);
    }
	/*else if (d1 * 64 / ((double)d1 * d2 * d3) < 0.1) {
		gmem_multi_row_launch(fn, gmem_multi_row_gather_op<F, T>, data, source, d1, d2, d3);
	}*/
	else {
        gmem_row_launch(fn, gmem_row_gather_op<F, T>, data, source, d1, d2, d3);
    }
}

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, source, d1, d2, d3);
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		smem_row_launch(fn, smem_row_scatter_op<F, T>, data, source, d1, d2, d3);
    }
	/*else if (d1 * 64 / ((double)d1 * d2 * d3) < 0.1) {
		gmem_multi_row_launch(fn, gmem_multi_row_scatter_op<F, T>, data, source, d1, d2, d3);
	}*/
	else {
        gmem_row_launch(fn, gmem_row_scatter_op<F, T>, data, source, d1, d2, d3);
    }
}

template void row_gather_op(_2d::c2r::row_shuffle, float*, int, int, int, int);
template void row_gather_op(_2d::c2r::row_shuffle, double*, int, int, int, int);

template void row_gather_op(_2d::r2c::row_shuffle, float*, int, int, int, int);
template void row_gather_op(_2d::r2c::row_shuffle, double*, int, int, int, int);

template void row_scatter_op(_2d::r2c::row_scatter_shuffle, float*, int, int, int, int);
template void row_scatter_op(_2d::r2c::row_scatter_shuffle, double*, int, int, int, int);

}
}

namespace _2d {

template<typename F, typename T>
__global__ void compress_row_gather_op(F fn, T* data, int source, size_t batch_size, int d1, int d2) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d2);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2);
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            smem[idx] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            fn.set_i(i);
            data[source + offset + idx] = smem[fn(j) + u * d1];
        }
    }
}

template<typename F, typename T>
__global__ void compress_row_scatter_op(F fn, T* data, int source, size_t batch_size, int d1, int d2) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d2);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2);
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
			int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            fn.set_i(i);
            smem[fn(j) + u * d1] = data[source + offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            data[source + offset + idx] = smem[idx];
        }
    }
}

template<typename F, typename K, typename T>
void compress_row_launch(F fn, K kernel, T* data, int source, int d1, int d2) {
	//PRINT("Smem Compress %s\n", fn.getName().c_str());
	//PRINT("\t(d1, d2) = (%d, %d)\n", d1, d2);
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size = smem_lim / 32;
	int n_threads = max_n_threads_per_sm() / 32;
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	//PRINT("\t# blocks = %d\n", n_blocks);
	size_t batch_size = smem_size / (sizeof(T) * (size_t)d1);
	//PRINT("\tbatch size = %zu\n", batch_size);
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, batch_size, d1, d2);
}

template<typename F, typename T>
__global__ void smem_row_gather_op(F fn, T* data, int source, int d1, int d2) {
    T* smem = shared_memory<T>();
    
    for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
        fn.set_i(i);
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			smem[j] = data[source + j + i * d1];
        }
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			data[source + j + i * d1] = smem[fn(j)];
        }
    }
}

template<typename F, typename T>
__global__ void smem_row_scatter_op(F fn, T* data, int source, int d1, int d2) {
    T* smem = shared_memory<T>();
    
    for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
        fn.set_i(i);
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			smem[fn(j)] = data[source + j + i * d1];
        }
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			data[source + j + i * d1] = smem[j];
        }
    }
}

template<typename F, typename K, typename T>
void smem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2) {
	//PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_size = sizeof(T) * (size_t)d1;
	int n_threads = get_num_thread(d1);
	//PRINT("\t# threads = %d\n", n_threads);
	int n_blocks = min(d2, get_num_block(kernel, n_threads, smem_size));
	kernel<<<n_blocks, n_threads, smem_size>>>(fn, data, source, d1, d2);
}

template<typename F, typename T>
__global__ void gmem_row_gather_op(F fn, T* data, int source, T* tmp, int d1, int d2) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		g.sync();
		for (size_t j = global_id; j < d1; j += grid_size) {
			tmp[j] = data[source + fn(j) + i * d1];
		}
		g.sync();
		for (size_t j = global_id; j < d1; j += grid_size) {
			data[source + j + i * d1] = tmp[j];
		}
	}
}

template<typename F, typename T>
__global__ void gmem_row_scatter_op(F fn, T* data, int source, T* tmp, int d1, int d2) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t grid_size = gridDim.x * blockDim.x;
	for (size_t i = 0; i < d2; i++) {
		fn.set_i(i);
		g.sync();
		for (size_t j = global_id; j < d1; j += grid_size) {
			tmp[fn(j)] = data[source + j + i * d1];
		}
		g.sync();
		for (size_t j = global_id; j < d1; j += grid_size) {
			data[source + j + i * d1] = tmp[j];
		}
	}
}

template<typename F, typename K, typename T>
void gmem_row_launch(F fn, K kernel, T* data, int source, int d1, int d2) {
	//PRINT("Gmem %s\n", fn.getName().c_str());
	T* tmp;
	size_t tmp_size = sizeof(T) * d1;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	int n_threads = 1024;
	int n_blocks = get_num_block(kernel, n_threads, 0) / 2;
	//PRINT("\t# blocks = %d\n", n_blocks);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&source, (void *)&tmp, (void *)&d1, (void *)&d2
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)kernel,
										  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void row_gather_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		//PRINT("compress row\n");
		compress_row_launch(fn, compress_row_gather_op<F, T>, data, source, d1, d2);
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		//PRINT("smem row\n");
		smem_row_launch(fn, smem_row_gather_op<F, T>, data, source, d1, d2);
    }
	else {
		//PRINT("gmem row\n");
        	gmem_row_launch(fn, gmem_row_gather_op<F, T>, data, source, d1, d2);
    }
}

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int source, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, compress_row_scatter_op<F, T>, data, source, d1, d2);
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		smem_row_launch(fn, smem_row_scatter_op<F, T>, data, source, d1, d2);
    }
	else {
        gmem_row_launch(fn, gmem_row_scatter_op<F, T>, data, source, d1, d2);
    }
}

template void row_gather_op(c2r::row_shuffle, float*, int, int, int);
template void row_gather_op(c2r::row_shuffle, double*, int, int, int);

template void row_gather_op(r2c::row_shuffle, float*, int, int, int);
template void row_gather_op(r2c::row_shuffle, double*, int, int, int);

template void row_gather_op(_3d::_213::row_shuffle, float*, int, int, int);
template void row_gather_op(_3d::_213::row_shuffle, double*, int, int, int);

template void row_scatter_op(r2c::row_scatter_shuffle, float*, int, int, int);
template void row_scatter_op(r2c::row_scatter_shuffle, double*, int, int, int);

template void row_gather_op(_4d::_1324::row_shuffle, float*, int, int, int);
template void row_gather_op(_4d::_1324::row_shuffle, double*, int, int, int);

template void row_gather_op(_4d::_3214::row_shuffle, float*, int, int, int);
template void row_gather_op(_4d::_3214::row_shuffle, double*, int, int, int);

template void row_gather_op(_5d::_43215::row_shuffle, float*, int, int, int);
template void row_gather_op(_5d::_43215::row_shuffle, double*, int, int, int);

template void row_gather_op(_5d::_14325::row_shuffle, float*, int, int, int);
template void row_gather_op(_5d::_14325::row_shuffle, double*, int, int, int);

}

}
