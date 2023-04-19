#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include "math.h"
#include "util.h"
#include "transpose.h"
#include "tensor_util.h"

#define MEMORY_TYPE 1
#define verify_flag 2
#define init_padding_ratio 0.03
#define UPPER_BOUND_VOLUME 22000000000

template <typename T>
void memcpy_transpose(TensorUtil<T> &tu, int NUM_SUB_TENSORS, double ALPHA) {

	T *d_data = NULL;
	T *h_data = NULL;
	size_t dataSize = tu.vol * sizeof(T);
	assert(dataSize <= UPPER_BOUND_VOLUME);

	size_t gpuDataSize = (1 + init_padding_ratio) * dataSize;
	
	CudaSafeCall(cudaMallocHost(&h_data, dataSize));
	CudaSafeCall(cudaMalloc((void **)&d_data, gpuDataSize));
	tu.init_data(h_data);
	tu.num_block = NUM_SUB_TENSORS;	
	
	int perm_int = 0;
	for(int i = 0; i < tu.rank; ++i){ perm_int += pow(10, i) * (tu.permutation[tu.rank - i - 1] + 1);}
	
	printf("%dD Inplace %d transpose\n", tu.rank, perm_int);
	printf("number of sub_tensors = %d\n", tu.num_block);
	printf("Dims =");
	for(size_t i = 0; i < tu.rank; ++i) { printf(" %zu", tu.dim_long[i]);}
	printf(", Data size = %.5f GB\n", (double)dataSize / 1e9);
	//printf("gpu Data size = %.5f GB\n", (double)gpuDataSize / 1e9);
	//assert(gpuDataSize < 11.05 * 1e9);
		
	//int dev = 1;
	//CudaSafeCall( cudaSetDevice(dev) );

	float t, d2h_time, trans_time, h2d_time;
	cudaEvent_t start, stop, d2h, trans, h2d;
	CudaSafeCall(cudaEventCreate(&start));
	CudaSafeCall(cudaEventCreate(&stop));
	CudaSafeCall(cudaEventCreate(&d2h));
	CudaSafeCall(cudaEventCreate(&trans));
	CudaSafeCall(cudaEventCreate(&h2d));

	CudaSafeCall(cudaEventRecord(start, 0));	
	CudaSafeCall(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaEventRecord(h2d, 0));
	
	inplace::transpose(d_data, tu.source, tu.rank, tu.dim, tu.permutation, sizeof(T), tu.num_block, ALPHA);
	CudaSafeCall(cudaEventRecord(trans, 0));

	CudaSafeCall(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaEventRecord(d2h, 0));
	
	//tu.print_tensor(h_data);
	
	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaEventRecord(stop, 0));
	CudaSafeCall(cudaEventSynchronize(stop));

	CudaSafeCall(cudaEventElapsedTime(&h2d_time, start, h2d));
	CudaSafeCall(cudaEventElapsedTime(&d2h_time, trans, d2h));
	CudaSafeCall(cudaEventElapsedTime(&t, start, stop));

	if(dataSize < UPPER_BOUND_VOLUME / 3 && verify_flag == 1) { tu.verify(h_data);}
	else if(verify_flag == 2) { tu.verify_inplace(h_data);}
	
	float throughput = ((double)dataSize * 2) / 1e6 / t;

	printf("Execution Time without h2d2h: %.5fms\n", t - h2d_time - d2h_time);
	printf("Execution Time: %.5fms\n", t);
	FILE *txtfp = fopen("inplace_bench.txt", "a+");
	fprintf(txtfp, "%.5f\n", t); 
	fclose(txtfp);
	txtfp = fopen("inplace_bench_throughput.txt", "a+");
	fprintf(txtfp, "%.5f\n", throughput);
	fclose(txtfp);

	//if (tu.fp != NULL) tu.write_file(d_data);
	CudaSafeCall(cudaFreeHost(h_data));
	CudaSafeCall(cudaFree(d_data));
	cudaDeviceReset();
}

template <typename T> 
void padding_managed_transpose(TensorUtil<T> &tu, int NUM_SUB_TENSORS, double ALPHA) {
	T *d_data = NULL;
	size_t dataSize = tu.vol * sizeof(T);
	assert(dataSize <= UPPER_BOUND_VOLUME);
	size_t gpuDataSize = (1 + init_padding_ratio) * tu.vol * sizeof(T);
	CudaSafeCall( cudaMallocManaged(&d_data, gpuDataSize) );
	tu.init_data(d_data);

	int perm_int = 0;
	tu.num_block = NUM_SUB_TENSORS;
	for(int i = 0; i < tu.rank; ++i){ perm_int += pow(10, i) * (tu.permutation[tu.rank - i - 1] + 1);}
	printf("%dD Inplace %d transpose", tu.rank, perm_int);
	printf(", Data size = %.5f GB\n", (double)gpuDataSize / 1e9);
	printf("number of sub_tensors = %d\n", tu.num_block);
	//int dev = 1;
	//CudaSafeCall( cudaSetDevice(dev) );

	float t;
	cudaEvent_t start, stop;
	CudaSafeCall(cudaEventCreate(&start));
	CudaSafeCall(cudaEventCreate(&stop));

	inplace::transpose(d_data, tu.source, tu.rank, tu.dim, tu.permutation, sizeof(T), tu.num_block, ALPHA);

	//tu.print_tensor(d_data);

	CudaSafeCall(cudaDeviceSynchronize());
	CudaSafeCall(cudaEventRecord(stop, 0));
	CudaSafeCall(cudaEventSynchronize(stop));
	CudaSafeCall(cudaEventElapsedTime(&t, start, stop));

	if(dataSize < UPPER_BOUND_VOLUME / 3 && verify_flag == 1) { tu.verify(d_data);}
	else if(verify_flag == 2) { tu.verify_inplace(d_data);}

	float throughput = ((double)gpuDataSize * 2) / 1e6 / t;

	printf("Execution Time: %.5fms\n", t);
	FILE *txtfp = fopen("inplace_bench.txt", "a+");
	fprintf(txtfp, "%.5f\n", t); 
	fclose(txtfp);
	txtfp = fopen("inplace_bench_throughput.txt", "a+");
	fprintf(txtfp, "%.5f\n", throughput);
	fclose(txtfp);

	//if (tu.fp != NULL) tu.write_file(d_data);
	CudaSafeCall(cudaFree(d_data));
	cudaDeviceReset();
}


int main(int argc, char **argv) {

	// Input format: [dimensions][permutations][size of data type in bytes][# sub tensor][expect extra mem][filename]
	// filename: optional
	int source = 0, rank = (argc - 1) / 2 - 1, k = 1, type_size;
	FILE *fp = NULL;

	int *dim = new int[rank];
	int *permutation = new int[rank];

	assert(argc == 2 * (rank + 1) + 2); 
	
	for (int i = 0; i < rank; i++){ dim[i] = atoi(argv[k++]);}
	for (int i = 0; i < rank; i++){ permutation[i] = atoi(argv[k++]) - 1;}
	type_size = atoi(argv[k++]);
	int NUM_SUB_TENSORS = atoi(argv[k++]);	
	double ALPHA = atof(argv[k++]);
	
	if (argc == k + 1) {fp = fopen(argv[k], "wb");}
	
	if (type_size == 4) {
		TensorUtil<int> tu(fp, rank, dim, source, permutation);
		if(MEMORY_TYPE == 1) { memcpy_transpose<int>(tu, NUM_SUB_TENSORS, ALPHA);}
		else { padding_managed_transpose<int>(tu, NUM_SUB_TENSORS, ALPHA);}
	} else if(type_size > 4) { 
		TensorUtil<long long> tu(fp, rank, dim, source, permutation);
		if(MEMORY_TYPE == 1) { memcpy_transpose<long long>(tu, NUM_SUB_TENSORS, ALPHA);}
		else { padding_managed_transpose<long long>(tu, NUM_SUB_TENSORS, ALPHA);}
	} else { printf("Invalid rank or type size\n");}

	if (fp != NULL){ fclose(fp);}
	
	delete [] dim;
	delete [] permutation;
	return 0;
}