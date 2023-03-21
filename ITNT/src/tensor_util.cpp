#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include "tensor_util.h"
#include <math.h>
#include <cstring>
#include <vector>
#include <queue>
#include <cassert>
#include <cuda_runtime.h>

template <typename T>
TensorUtil<T>::TensorUtil(FILE *_fp, int _rank, int *_dim, int _source, int *_permutation) : fp(_fp), rank(_rank) {
	dim = new int[rank];
	dim_long = new size_t[rank];
	permutation = new int[rank];
	permutation_long = new size_t[rank];
	std::copy(_dim, _dim + rank, dim);
	std::copy(_permutation, _permutation + rank, permutation);
	std::copy(_dim, _dim + rank, dim_long);
	std::copy(_permutation, _permutation + rank, permutation_long);
	stride = new size_t[rank + 1];
	stride[0] = 1;
	std::partial_sum(dim_long, dim_long + rank, stride + 1, std::multiplies<size_t>());
	vol = stride[rank];
	source = _source;
	num_block = 1;
}

template TensorUtil<int>::TensorUtil(FILE *, int, int *, int, int *);
template TensorUtil<long long>::TensorUtil(FILE *, int, int *, int, int *);
template TensorUtil<float>::TensorUtil(FILE *, int, int *, int, int *);
template TensorUtil<double>::TensorUtil(FILE *, int, int *, int, int *);

template <typename T>
TensorUtil<T>::~TensorUtil() {
	delete[] dim;
	delete[] dim_long;
	delete[] permutation;
	delete[] permutation_long;
	delete[] stride;
}

template TensorUtil<int>::~TensorUtil();
template TensorUtil<long long>::~TensorUtil();
template TensorUtil<float>::~TensorUtil();
template TensorUtil<double>::~TensorUtil();

template <typename T>
void TensorUtil<T>::init_data(T *data) {
	for (size_t i = 0; i < vol; i++) { data[i] = i;}
}

template void TensorUtil<int>::init_data(int *);
template void TensorUtil<long long>::init_data(long long *);
template void TensorUtil<float>::init_data(float *);
template void TensorUtil<double>::init_data(double *);

template <typename T>
void TensorUtil<T>::print_tensor(T *data) {
	assert(vol <= 3000);
	for (size_t i = 0; i < vol; i++) {
		printf("%-6d", (int)data[i]);
		for (int j = 1; j < rank; j++) {
			if ((i + 1) % stride[j] == 0)
				printf("\n");
		}
	}
	printf("\n");
}

template void TensorUtil<int>::print_tensor(int *);
template void TensorUtil<long long>::print_tensor(long long *);
template void TensorUtil<float>::print_tensor(float *);
template void TensorUtil<double>::print_tensor(double *);

template <typename T>
void TensorUtil<T>::write_file(T *data) {
	fwrite(data, sizeof(T), vol, fp);
}

template void TensorUtil<int>::write_file(int *);
template void TensorUtil<long long>::write_file(long long *);
template void TensorUtil<float>::write_file(float *);
template void TensorUtil<double>::write_file(double *);

template <typename T>
void TensorUtil<T>::seq_tt(T *ans, T *data) {
	for (size_t idx_s = 0; idx_s < vol; idx_s++) {
		size_t idx_d = 0;
		for (size_t i = 0; i < rank; i++) {
			size_t stride_d = 1;
			for (size_t j = 0; permutation_long[j] != i; j++) {
				stride_d *= dim_long[permutation_long[j]];
			}
			idx_d += ((idx_s / stride[i]) % dim_long[i]) * stride_d;
		}
		ans[idx_d] = data[idx_s];
	}
}

template void TensorUtil<int>::seq_tt(int *, int *);
template void TensorUtil<long long>::seq_tt(long long *, long long *);
template void TensorUtil<float>::seq_tt(float *, float *);
template void TensorUtil<double>::seq_tt(double *, double *);


template <typename T>
void TensorUtil<T>::verify_inplace(T *h_data) {
	printf("Inplace transpose verify_inplace\n");
	//printf("Dims = (%d %d %d)\n", tu.dim[0], tu.dim[1], tu.dim[2]);
	bool ans_flag = true;
	int *stride_s = new int[rank + 1];
	int *dim_perm = new int[rank];
	for(int i = 0; i < rank; ++i) { dim_perm[i] = dim[permutation[i]];}
	stride_s[0] = 1;
	std::partial_sum(dim, dim + rank, stride_s + 1, std::multiplies<int>());
	for(int id_s = 0; id_s < vol; id_s++) {
		int id_d = 0;
		for(int i = 0; i < rank; ++i) {
			int stride_d = 1;
			for (int j = 0; permutation[j] != i; j++) { stride_d *= dim[permutation[j]];}
			id_d += ((id_s / stride_s[i]) % dim[i]) * stride_d;
		}
		if(h_data[id_d] != id_s) { ans_flag = false; break;}
	}
	if(ans_flag == true) { printf("Correct transpose\n");}
	else {
		printf("Incorrect transpose\n");
		if (vol < 500) {
			//printf("after\n");
   	     		//tu.print_tensor(h_data);
			//printf("ans\n");
        		//tu.print_tensor(o_data);
       		}
	}

}

template void TensorUtil<int>::verify_inplace(int *);
template void TensorUtil<long long>::verify_inplace(long long *);
template void TensorUtil<float>::verify_inplace(float *);
template void TensorUtil<double>::verify_inplace(double *);

template <typename T>
void TensorUtil<T>::verify(T *h_data) {
	//return;
	printf("Inplace transpose verify\n");
	//printf("Dims = (%d %d %d)\n", tu.dim[0], tu.dim[1], tu.dim[2]);
	bool ans_flag = true;
	T* i_data = (T*)malloc(vol * sizeof(T));
	init_data(i_data);
	T* o_data = (T*)malloc(vol * sizeof(T));
	seq_tt(o_data, i_data);
	int print_count = 0;
	for(int count = 0; count < vol; ++count) {
        	if(h_data[count] != o_data[count]) {
			//if(print_count < 10){ printf("Incorrect data = %d %d, ans = %d\n", count, h_data[count], o_data[count]);}
			ans_flag = false;
			print_count++;
			//break;
        	}
    	}
	if(ans_flag == true) { printf("Correct transpose\n");}
	else {
		printf("Incorrect transpose\n");
		if (vol < 500) {
			//printf("after\n");
   	     		//print_tensor(h_data);
			//printf("ans\n");
        		//print_tensor(o_data);
       		}
	}
	//tu.print_tensor(h_data);
	free(i_data);
	free(o_data);
}

template void TensorUtil<int>::verify(int *);
template void TensorUtil<long long>::verify(long long *);
template void TensorUtil<float>::verify(float *);
template void TensorUtil<double>::verify(double *);