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
size_t Dimension_Pre_Padding(T *data, int *dim, size_t* stride, int rank, int padding_dim, int NUM_TENSOR_BLOCK) {
	int n_blocks = get_num_block(Dim_Pre_Padding_gmem_op<T>, n_threads, 0);
	size_t tmp = sizeof(T) * n_blocks * n_threads;
	return tmp;
}

template size_t Dimension_Pre_Padding(int *, int *, size_t*, int, int, int);
template size_t Dimension_Pre_Padding(long long *, int *, size_t*, int, int, int);
template size_t Dimension_Pre_Padding(float *, int *, size_t*, int, int, int);
template size_t Dimension_Pre_Padding(double *, int *, size_t*, int, int, int);

template <typename T>
size_t Dimension_Post_Padding(T *data, int *dim, int *permutation, int rank, int old_padding_dim, int NUM_TENSOR_BLOCK) {
	int n_blocks = get_num_block(Dim_Post_Padding_gmem_op<T>, n_threads, 0);
	size_t tmp = sizeof(T) * n_blocks * n_threads;
	return tmp;
}

template size_t Dimension_Post_Padding(int *, int *, int *, int, int, int);
template size_t Dimension_Post_Padding(long long *, int *, int *, int, int, int);
template size_t Dimension_Post_Padding(float *, int *, int *, int, int, int);
template size_t Dimension_Post_Padding(double *, int *, int *, int, int, int);

}