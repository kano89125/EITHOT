#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "util.h"
#include "debug.h"
#include "Nd_padding.h"


#define n_threads 1024

/*  
Nd_padding_pre_process -> Dimension_Pre_Padding
Nd_padding_post_process -> Dimension_Post_Padding

if tensor vol is small, we should use shaerd memory as temporary buffer to achieve dimension padding.
and adjust number of threads in experiment to implement optimally. 

*/
namespace inplace {

template <typename T>
__global__ void Dim_Pre_Padding_gmem_op(T *data, T *tmp, int *dim, size_t *stride_s, size_t *padding_stride,
	 int rank, int padding_dim_pos, int padding_dim) {
	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();
	size_t thread_ID = blockIdx.x * blockDim.x + threadIdx.x;
	size_t _ROUND_LENGTH = gridDim.x * blockDim.x;
	size_t handle_vol = stride_s[rank] - stride_s[padding_dim_pos + 1];
	size_t remain_vol = handle_vol % _ROUND_LENGTH, num_round = handle_vol / _ROUND_LENGTH;
	size_t handle_source = stride_s[padding_dim_pos + 1], remain_source = stride_s[padding_dim_pos + 1] + num_round * _ROUND_LENGTH;
	size_t padding_offset = padding_stride[padding_dim_pos] * (padding_dim - dim[padding_dim_pos]);

	if(thread_ID < remain_vol) {
		size_t id_s = remain_source + thread_ID;
		tmp[thread_ID] = data[id_s];
	}

	g.sync();

	if(thread_ID < remain_vol) {
		size_t id_s = remain_source + thread_ID;
		size_t id_d = id_s + id_s / stride_s[padding_dim_pos + 1] * padding_offset;
		data[id_d] = tmp[thread_ID];
	}

	for(int k = num_round - 1; k >= 0; --k) {
		g.sync();
		size_t id_s = handle_source + k * _ROUND_LENGTH + thread_ID;
		tmp[thread_ID] = data[id_s];
		size_t id_d = id_s + id_s / stride_s[padding_dim_pos + 1] * padding_offset;
		g.sync();
		data[id_d] = tmp[thread_ID];
	}
}

template <typename T> // no coalesced, how to ignore the first no movement stride
__global__ void Dim_Post_Padding_gmem_op(T *data, T *tmp, size_t *padding_permuted_stride, 
	int rank, int padding_dim_pos, int padding_dim, int origin_dim) {
	namespace cg = cooperative_groups;
    	cg::grid_group g = cg::this_grid();
	size_t thread_ID = blockIdx.x * blockDim.x + threadIdx.x;
	size_t _ROUND_LENGTH = gridDim.x * blockDim.x;
	size_t handle_vol = padding_permuted_stride[padding_dim_pos] * origin_dim;
	size_t remain_vol = handle_vol % _ROUND_LENGTH, num_round = handle_vol / _ROUND_LENGTH;

	size_t num_handle_vol = padding_permuted_stride[rank] / padding_permuted_stride[padding_dim_pos + 1];
	size_t padding_offset = padding_permuted_stride[padding_dim_pos] * (padding_dim - origin_dim);
	size_t nr_rl = num_round * _ROUND_LENGTH;
	for(size_t j = 1; j < num_handle_vol; ++j) {
		g.sync();
		size_t handle_source = j * padding_permuted_stride[padding_dim_pos + 1];
		for(size_t i = 0; i < num_round; ++i) {
			g.sync();
			size_t id_s = handle_source + i * _ROUND_LENGTH + thread_ID;
			tmp[thread_ID] = data[id_s];
			size_t id_d = id_s - id_s / padding_permuted_stride[padding_dim_pos + 1] * padding_offset;
			g.sync();
			data[id_d] = tmp[thread_ID];
		}

		g.sync();
	
		if(thread_ID < remain_vol) {
			size_t id_s = handle_source + nr_rl + thread_ID;	
			tmp[thread_ID] = data[id_s];
		}
		g.sync();
		if(thread_ID < remain_vol) {
			size_t id_s = handle_source + nr_rl + thread_ID;
			size_t id_d = id_s - id_s / padding_permuted_stride[padding_dim_pos + 1] * padding_offset;
			data[id_d] = tmp[thread_ID];
		}
	}
}


template <typename T>
void Nd_padding_pre_process(T *data, int *h_dim, size_t* h_stride, int rank, int padding_dim_pos, int NUM_TENSOR_BLOCK) {
	int padding_dim = (h_dim[padding_dim_pos] / NUM_TENSOR_BLOCK + 1) * NUM_TENSOR_BLOCK;
	int padding_vol = h_stride[rank] / h_dim[padding_dim_pos] * padding_dim;

	int *d_dim;
	size_t dim_size = rank * sizeof(int);
	CudaSafeCall(cudaMalloc((void **)&d_dim, dim_size));
	CudaSafeCall(cudaMemcpy(d_dim, h_dim, dim_size, cudaMemcpyHostToDevice));

	size_t *h_dim_long = new size_t[rank];
	size_t dim_long_size = rank * sizeof(size_t);
	
	for(int i = 0; i < rank; ++i){ h_dim_long[i] = h_dim[i];}
	h_dim_long[padding_dim_pos] = padding_dim;

	size_t *d_dim_long;
	CudaSafeCall(cudaMalloc((void **)&d_dim_long, dim_long_size));
	CudaSafeCall(cudaMemcpy(d_dim_long, h_dim_long, dim_long_size, cudaMemcpyHostToDevice));

	size_t *d_stride;
	size_t stride_size = (rank + 1) * sizeof(size_t);
	CudaSafeCall(cudaMalloc((void **)&d_stride, stride_size));
	CudaSafeCall(cudaMemcpy(d_stride, h_stride, stride_size, cudaMemcpyHostToDevice));
	

	size_t *h_padding_stride = new size_t[rank + 1];
	h_padding_stride[0] = 1;
	std::partial_sum(h_dim_long, h_dim_long + rank, h_padding_stride + 1, std::multiplies<size_t>());
	size_t *d_padding_stride;
	CudaSafeCall(cudaMalloc((void **)&d_padding_stride, stride_size));
	CudaSafeCall(cudaMemcpy(d_padding_stride, h_padding_stride, stride_size, cudaMemcpyHostToDevice));
	
	int n_blocks = get_num_block(Dim_Pre_Padding_gmem_op<T>, n_threads, 0);
	//PRINT("# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_blocks * n_threads;
	CudaSafeCall(cudaMallocManaged(&tmp, tmp_size));
	prefetch(tmp, tmp_size);

	size_t dataSize = padding_vol * sizeof(T);
	prefetch(&data, dataSize);

	void *kernelArgs[] = {
		(void *)&data, (void *)&tmp, (void *)&d_dim, (void *)&d_stride, 
		(void *)&d_padding_stride, (void *)&rank, (void *)&padding_dim_pos, (void *)&padding_dim
	};
	
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)Dim_Pre_Padding_gmem_op<T>, n_blocks, n_threads, kernelArgs) );
	//CudaSafeCall( cudaLaunchCooperativeKernel((void *)padding_N_dim_pre_process_op<T>, NUM_BLOCKS, NUM_THREADS, kernelArgs) );
	cudaDeviceSynchronize();

	CudaSafeCall(cudaFree(tmp));
	CudaSafeCall(cudaFree(d_dim));
	CudaSafeCall(cudaFree(d_dim_long));
	CudaSafeCall(cudaFree(d_stride));
	CudaSafeCall(cudaFree(d_padding_stride));
	delete [] h_dim_long;
	delete [] h_padding_stride;
}

template void Nd_padding_pre_process(int *, int *, size_t*, int, int, int);
template void Nd_padding_pre_process(long long *, int *, size_t*, int, int, int);
template void Nd_padding_pre_process(float *, int *, size_t*, int, int, int);
template void Nd_padding_pre_process(double *, int *, size_t*, int, int, int);

template <typename T>
void Nd_padding_post_process(T *data, int *h_dim, int *permutation, int rank, int old_padding_dim_pos, int NUM_TENSOR_BLOCK) {
	
	int permuted_padding_dim_pos = -1, permuted_padding_dim_size = -1, origin_dim = -1;
	size_t stride_size = (rank + 1) * sizeof(size_t);

	int *h_permuted_dim = new int[rank];
	size_t *h_permuted_dim_long = new size_t[rank];
	size_t *h_padding_permuted_stride = new size_t[rank + 1];

	size_t *d_padding_permuted_stride;

	for(int i = 0; i < rank; ++i) {
		h_permuted_dim[i] = h_dim[permutation[i]]; 
		h_permuted_dim_long[i] = h_permuted_dim[i];
		if(permutation[i] == old_padding_dim_pos) {
			permuted_padding_dim_pos = i;
			origin_dim = h_dim[permutation[permuted_padding_dim_pos]];
			permuted_padding_dim_size = (h_dim[permutation[permuted_padding_dim_pos]] / NUM_TENSOR_BLOCK + 1) * NUM_TENSOR_BLOCK;
		}
	}

	h_padding_permuted_stride[0] = 1;
	h_permuted_dim_long[permuted_padding_dim_pos] = permuted_padding_dim_size;
	std::partial_sum(h_permuted_dim_long, h_permuted_dim_long + rank, h_padding_permuted_stride + 1, std::multiplies<size_t>());

	CudaSafeCall(cudaMalloc((void **)&d_padding_permuted_stride, stride_size));
	CudaSafeCall(cudaMemcpy(d_padding_permuted_stride, h_padding_permuted_stride, stride_size, cudaMemcpyHostToDevice));

	int n_blocks = get_num_block(Dim_Post_Padding_gmem_op<T>, n_threads, 0);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_blocks * n_threads;
	CudaSafeCall(cudaMallocManaged(&tmp, tmp_size));
	prefetch(tmp, tmp_size);

	if(permuted_padding_dim_pos == -1){ return;}
	//printf("perm = %d %d\n", permutation[0], permutation[1]);
	//printf("dim = %d %d %d %d %d %d\n", new_dim[0], new_dim[1], new_padding_dim, old_padding_dim_pos, new_padding_dim_size, origin_dim);
	//if(new_padding_dim == rank -1) { printf("no migration\n");}
	//printf("padding stride = %ld %ld %ld\n", padding_permutation_stride[0], padding_permutation_stride[1], padding_permutation_stride[2]);
	void *kernelArgs[] = {
		(void *)&data, (void *)&tmp, (void *)&d_padding_permuted_stride, 
		(void *)&rank, (void *)&permuted_padding_dim_pos, (void *)&permuted_padding_dim_size, (void *)&origin_dim
	};
	
	//PRINT("# blocks = %d\n", n_blocks);
	//CudaSafeCall( cudaLaunchCooperativeKernel((void *)padding_N_dim_post_process_op<T>, NUM_BLOCKS, NUM_THREADS, kernelArgs) );
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)Dim_Post_Padding_gmem_op<T>, n_blocks, n_threads, kernelArgs) );
	
	cudaDeviceSynchronize();
	CudaSafeCall(cudaFree(tmp));
	CudaSafeCall(cudaFree(d_padding_permuted_stride));
	delete [] h_permuted_dim_long;
	delete [] h_padding_permuted_stride;
	delete [] h_permuted_dim;
}

template void Nd_padding_post_process(int *, int *, int *, int, int, int);
template void Nd_padding_post_process(long long *, int *, int *, int, int, int);
template void Nd_padding_post_process(float *, int *, int *, int, int, int);
template void Nd_padding_post_process(double *, int *, int *, int, int, int);

}