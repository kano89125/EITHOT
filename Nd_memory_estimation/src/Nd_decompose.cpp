#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include "util.h"
#include "col_op.h"
#include "equations.h"
#include "transpose.h"
#include "3dtranspose.h"
#include "Nd_decompose.h"

namespace inplace {


template <typename T>
size_t Linearization_NDPartition(T *data, int decompose_stride, int non_decompose_stride, int rank, int num_block) {
	int linear_permutation[3] = {0, 2, 1};
	int linear_dim[3] = {decompose_stride / num_block, num_block, non_decompose_stride};
	//printf("Linearization_NDPartition d1, d2, d3 = %d %d %d\n", linear_dim[0], linear_dim[1], linear_dim[2]);
	return inplace::transpose(data, 0, 3, linear_dim, linear_permutation, sizeof(T));
}

template size_t Linearization_NDPartition(int*, int , int, int, int);
template size_t Linearization_NDPartition(long long*, int , int, int, int);
template size_t Linearization_NDPartition(float*, int , int, int, int);
template size_t Linearization_NDPartition(double*, int , int, int, int);

template <typename T>
size_t Linearization_NDJoin(T *data, int decompose_stride, int non_decompose_stride, int rank, int num_block) {
	int linear_permutation[3] = {0, 2, 1};
	int linear_dim[3] = {decompose_stride / num_block, non_decompose_stride, num_block};
	//printf("Linearization_NDJoin d1, d2, d3 = %d %d %d\n", linear_dim[0], linear_dim[1], linear_dim[2]);
	return inplace::transpose(data, 0, 3, linear_dim, linear_permutation, sizeof(T));
}

template size_t Linearization_NDJoin(int*, int , int, int, int);
template size_t Linearization_NDJoin(long long*, int , int, int, int);
template size_t Linearization_NDJoin(float*, int , int, int, int);
template size_t Linearization_NDJoin(double*, int , int, int, int);

template <typename T>
size_t NdPartition(T *data, int *dim, int rank, int num_block, int decompose_dim) {
	int decompose_stride = 1, non_decompose_stride = 1;
	for(int i = 0; i <= decompose_dim; ++i) { decompose_stride *= dim[i];}
	for(int i = decompose_dim + 1; i < rank; ++i) {non_decompose_stride *= dim[i];}
	//printf("Nd Partition d, nd = %d %d \n", decompose_stride, non_decompose_stride);
	size_t tmp = Linearization_NDPartition(data, decompose_stride, non_decompose_stride, rank, num_block);
	//printf("Linearization_ND Partition Extra Memory size = %.5f MB\n", (double)tmp / 1e6);
	return tmp;
}

template size_t NdPartition(int*, int *, int, int, int);
template size_t NdPartition(long long*, int *, int, int, int);
template size_t NdPartition(float*, int *, int, int, int);
template size_t NdPartition(double*, int *, int, int, int);

template <typename T>
size_t NdJoin(T *data, int *dim, int *permutation, int rank, int num_block,  int ori_decompose_dim) {
	int *perm_dim = new int[rank];
	int decompose_permutation_dim = -1;
	for(int i = 0; i < rank; ++i) { 
		perm_dim[i] = dim[permutation[i]];
		if(permutation[i] == ori_decompose_dim){ decompose_permutation_dim = i;} 
	}
	assert(decompose_permutation_dim != -1);
	int decompose_stride = 1, non_decompose_stride = 1;
	for(int i = 0; i <= decompose_permutation_dim; ++i) { decompose_stride *= perm_dim[i];}
	for(int i = decompose_permutation_dim + 1; i < rank; ++i) { non_decompose_stride *= perm_dim[i];}
	delete[] perm_dim;
	size_t tmp = Linearization_NDJoin(data, decompose_stride, non_decompose_stride, rank, num_block);
	//printf("Linearization_NDJoin Extra Memory size = %.5f MB\n", (double)tmp / 1e6);
	return tmp;
}

template size_t NdJoin(int*, int *, int *, int, int, int);
template size_t NdJoin(long long*, int *, int *, int, int, int);
template size_t NdJoin(double*, int *, int *, int, int, int);
template size_t NdJoin(float*, int *, int *, int, int, int);

}