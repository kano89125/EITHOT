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
void Linearization_NDPartition(T *data, int decompose_stride, int non_decompose_stride, int rank, int num_block, double LARGE_RATIO) {
	int linear_permutation[3] = {0, 2, 1};
	int linear_dim[3] = {decompose_stride / num_block, num_block, non_decompose_stride};
	assert(linear_dim[0] != 0);
	//printf("Linearization_NDPartition d1, d2, d3 = %d %d %d %f\n", linear_dim[0], linear_dim[1], linear_dim[2], LARGE_RATIO);
	//_3d::_132::transpose(reinterpret_cast<float*>(data), 0, linear_dim[0], linear_dim[1], linear_dim[2]);
	inplace::transpose(data, 0, 3, linear_dim, linear_permutation, sizeof(T), 1, LARGE_RATIO);
}

template void Linearization_NDPartition(int*, int , int, int, int, double);
template void Linearization_NDPartition(long long*, int , int, int, int, double);
template void Linearization_NDPartition(float*, int , int, int, int, double);
template void Linearization_NDPartition(double*, int , int, int, int, double);

template <typename T>
void Linearization_NDJoin(T *data, int decompose_stride, int non_decompose_stride, int rank, int num_block, double LARGE_RATIO) {
	int linear_permutation[3] = {0, 2, 1};
	int linear_dim[3] = {decompose_stride / num_block, non_decompose_stride, num_block};
	assert(linear_dim[0] != 0);
	//printf("Linearization_NDJoin d1, d2, d3 = %d %d %d %f\n", linear_dim[0], linear_dim[1], linear_dim[2], LARGE_RATIO);
	//_3d::_132::transpose(reinterpret_cast<float*>(data), 0, linear_dim[0], linear_dim[1], linear_dim[2]); 
	inplace::transpose(data, 0, 3, linear_dim, linear_permutation, sizeof(T), 1, LARGE_RATIO);
}

template void Linearization_NDJoin(int*, int , int, int, int, double);
template void Linearization_NDJoin(long long*, int , int, int, int, double);
template void Linearization_NDJoin(float*, int , int, int, int, double);
template void Linearization_NDJoin(double*, int , int, int, int, double);

template <typename T>
void NdPartition(T *data, int *dim, int rank, int num_block, int decompose_dim, double LARGE_RATIO) {
	//printf("max dim pos = %d\n", max_dim_pos);
	int decompose_stride = 1, non_decompose_stride = 1;
	for(int i = 0; i <= decompose_dim; ++i) { decompose_stride *= dim[i];}
	for(int i = decompose_dim + 1; i < rank; ++i) {non_decompose_stride *= dim[i];}
	//printf("Nd Partition d, nd = %d %d \n", decompose_stride, non_decompose_stride);
	Linearization_NDPartition(data, decompose_stride, non_decompose_stride, rank, num_block, LARGE_RATIO);
}

template void NdPartition(int*, int *, int, int, int, double);
template void NdPartition(long long*, int *, int, int, int, double);
template void NdPartition(float*, int *, int, int, int, double);
template void NdPartition(double*, int *, int, int, int, double);


// permutation: 0 to rank - 1
template <typename T>
void NdJoin(T *data, int *dim, int *permutation, int rank, int num_block, int ori_decompose_dim, double LARGE_RATIO) {
	//int decompose_permutation_dim = -1;
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
	Linearization_NDJoin(data, decompose_stride, non_decompose_stride, rank, num_block, LARGE_RATIO);
}

template void NdJoin(int*, int *, int *, int, int, int, double);
template void NdJoin(long long*, int *, int *, int, int, int, double);
template void NdJoin(double*, int *, int *, int, int, int, double);
template void NdJoin(float*, int *, int *, int, int, int, double);

}