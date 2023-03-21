#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include "math.h"
#include "Nd_padding.h"
#include "Nd_decompose.h"
#include "transpose.h"
#include "tensor_util.h"
#include "cudacheck.h"
#include "util.h"

#define WARP_SIZE 32
#define UPPER_BOUND_VOLUME 5500000000

template<typename T>
void transpose(TensorUtil<T>& tu, int NUM_TENSOR_BLOCK, double ALPHA, int TYPE_SIZE)  {	
	size_t& vol = tu.vol;
	size_t dataSize = vol * sizeof(T), smem_size = shared_mem_per_block();
	// check tensor memory usage <= threshold
	assert(dataSize <= UPPER_BOUND_VOLUME);

	int perm_int = 0;
	for(int i = 0; i < tu.rank; ++i) { perm_int += pow(10, i) * (tu.permutation[tu.rank - i - 1] + 1);}

	T* d_data = NULL;
	
	/*
	// If have muliple devices
	int dev = 1;
	CudaSafeCall( cudaSetDevice(dev) );
	*/
	bool decompose_flag = false, padding_flag = false;
	size_t decompose_tmp = 0, padding_tmp = 0, final_tmp = 0, transpose_tmp = 0;
	int decompose_dim = -1, padding_dim_pos = -1, padding_dim = -1;
	int max_dim_pos = find_proper_max_dim_pos(tu.dim, tu.rank);
	// dataSize must larger than smem size	
	transpose_tmp = inplace::transpose(d_data, 0, tu.rank, tu.dim, tu.permutation, sizeof(T));
	int percent = (double)transpose_tmp / (double)dataSize / ALPHA;
	int nb = pow(2, (int)ceil(log2(percent)));
	if(nb == 0){ ++nb;}
	//printf("init Extra Memory size = %.5f MB, take %.5f percent input data size, nb = %d\n", (double)transpose_tmp / 1e6, perc, nb);
	
	if(transpose_tmp > 0) {
		ALPHA = std::max(ALPHA, 1 / sqrt((double)vol / (double)WARP_SIZE));// assure correct for split dim for c2r
		printf("new ratio = %f\n", ALPHA);
	}
	int Large_dim_low_bound = WARP_SIZE / ALPHA;

	if(tu.dim[max_dim_pos] > Large_dim_low_bound && nb > 1) {
		decompose_flag = true;
		decompose_dim = max_dim_pos;
		if(tu.dim[max_dim_pos] % nb != 0) {
			padding_flag = true;
			padding_dim_pos = max_dim_pos;
			padding_dim = (tu.dim[padding_dim_pos] / nb + 1) * nb;
			padding_tmp = tu.vol / tu.dim[padding_dim_pos] * (padding_dim - tu.dim[padding_dim_pos]) * sizeof(T);
			//printf("Padding tensor Extra Memory size = %.5f MB\n", (double)padding_tmp / 1e6);
		}
	} else {  //if(tu.dim[max_dim_pos] <= Large_dim_low_bound)
		ALPHA = ceil(ALPHA * 100.0) / 100.0;
		int single = 1;
		//printf("Large dim bound = %d, nb = %d\n", Large_dim_low_bound, nb);
		printf("\t\"");
		for(size_t i = 0; i < tu.rank; ++i) { printf(" %zu", tu.dim_long[i]);}
		for(size_t i = 0; i < tu.rank; ++i) { printf(" %zu", tu.permutation_long[i] + 1);}
		printf(" %d %d %.2f\"\\\n", TYPE_SIZE, single, ALPHA);
		return;
	}
	
	tu.num_block = nb;
	transpose_tmp /= nb;
	if(decompose_flag == true) {
		if(padding_flag == true) { padding_tmp = std::max(padding_tmp, inplace::Dimension_Pre_Padding(d_data, tu.dim, tu.stride, tu.rank, padding_dim_pos, tu.num_block));}
		decompose_tmp = inplace::NdPartition(d_data, tu.dim, tu.rank, tu.num_block, max_dim_pos);
		decompose_tmp = std::max(decompose_tmp, inplace::NdJoin(d_data, tu.dim, tu.permutation, tu.rank, tu.num_block, max_dim_pos));
		final_tmp = padding_tmp + std::max(decompose_tmp, transpose_tmp);
	}
	else { final_tmp = transpose_tmp;}

	double memRatio = (double)final_tmp / (double)dataSize;
	//printf("Padding Transpose Extra Memory size = %.5f MB\n", (double)padding_tmp / 1e6);
	//printf("Decompose Transpose Extra Memory size = %.5f MB\n", (double)decompose_tmp / 1e6);
	//printf("Sub Transpose Extra Memory size = %.5f MB\n", (double)transpose_tmp / 1e6);
	//printf("mid Extra Memory size = %.5f MB, take %.5f percent input data size, nb = %d\n", (double)final_tmp / 1e6, perc, nb);
	
	if(memRatio > ALPHA) {
		int round = 0;
		size_t new_final_tmp = final_tmp;
		double new_ratio;
		int new_nb = nb; 
		while (final_tmp >= new_final_tmp && nb > 1) {
			round++;
			nb = new_nb;
			final_tmp = new_final_tmp;
			new_nb /= 2;
			padding_dim = (tu.dim[max_dim_pos] / new_nb + 1) * new_nb;
			padding_tmp =  tu.vol / tu.dim[max_dim_pos] * (padding_dim - tu.dim[max_dim_pos]) * sizeof(T);
			transpose_tmp *= 2;
			decompose_tmp /= 2;
			new_final_tmp = padding_tmp + std::max(decompose_tmp, transpose_tmp);
			//printf("round %d, Padding Transpose Extra Memory size = %.5f MB\n", round, (double)padding_tmp / 1e6);
			//printf("round %d, Decompose Transpose Extra Memory size = %.5f MB\n", round, (double)decompose_tmp / 1e6);
			//printf("round %d, Sub Transpose Extra Memory size = %.5f MB\n", round, (double)transpose_tmp / 1e6);
			//printf("round %d Extra Memory size = %.5f MB, take %.5f percent input data size, nb = %d\n", round, (double)new_final_tmp / 1e6, new_perc, new_nb);
		}
		tu.num_block = nb;
		memRatio = (double)final_tmp / (double)dataSize;
		//printf("After optimize, Extra Memory size = %.5f MB, take %.5f percent input data size, nb = %d, ratio = %.3f\n", (double)final_tmp / 1e6, perc, tu.num_block, memRatio);	
		memRatio = ceil(memRatio * 100.0) / 100.0;
		ALPHA = std::max(memRatio, ALPHA);
	}
	
	//printf(" %dD Inplace %d transpose,", tu.rank, perm_int);
	//printf(" Dims =");
	printf("Final Extra Memory Size = %.5f MB, take %.5f percent input data size\n", (double)final_tmp / 1e6, memRatio);
	printf("\t\"");
	for(size_t i = 0; i < tu.rank; ++i) { printf(" %zu", tu.dim_long[i]);}
	for(size_t i = 0; i < tu.rank; ++i) { printf(" %zu", tu.permutation_long[i] + 1);}	
		
	//printf("NUM sub-tensor should be %d\n", nb);
	//printf(" %d\n, nb");
	printf(" %d %d %.2f\"\\\n", TYPE_SIZE, nb, ALPHA);
	
    	//FILE* txtfp = fopen("inplace_space.txt", "a+");
    	//fprintf(txtfp, "%.5f\n", (double)tmp / 1e6);
   	//fclose(txtfp);

    	//if (tu.fp != NULL) tu.write_file(d_data);
	//else tu.print_tensor(d_data);
	
	//CudaSafeCall( cudaFree(d_data) );
}

int main(int argc, char** argv) {
	
	// Input format: [dimensions][permutations][size of data type in bytes][# sub tensor][expect extra mem]
	// filename: optional
	int rank = (argc - 1) / 2 - 1,  k = 1;
	int *dim = new int[rank];
	int *permutation = new int[rank];

	assert(argc == 2 * (rank + 1) + 2);

	for (int i = 0; i < rank; i++) { dim[i] = atoi(argv[k++]);}
	
	for (int i = 0; i < rank; i++) { permutation[i] = atoi(argv[k++]) - 1;}

	int type_size = atoi(argv[k++]);
	int NUM_TENSOR_BLOCK = atoi(argv[k++]);
	double EXTRA_RATIO = atof(argv[k++]);
	
	
	FILE* fp = NULL;
	if (argc == k + 1) { fp = fopen(argv[k], "wb");}
	if (type_size == 4) { 
		// less than 3.7%
		TensorUtil<int> tu(fp, rank, dim, permutation);
		transpose<int>(tu, NUM_TENSOR_BLOCK, EXTRA_RATIO, type_size);
	}
	else { 
		// less than 5.2%
		TensorUtil<long long> tu(fp, rank, dim, permutation);
		transpose<long long>(tu, NUM_TENSOR_BLOCK, EXTRA_RATIO, type_size);
	}
	if (fp != NULL) fclose(fp);
	
	delete [] dim;
	delete [] permutation;

	return 0;
}
