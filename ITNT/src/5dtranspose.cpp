#include <cstdio>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "util.h"
#include "row_op.h"
#include "col_op.h"
#include "equations.h"
#include "Nd_padding.h"
#include "5dtranspose.h"
#include "Nd_transpose.h"
#include "Nd_decompose.h"
#include "memory_estimation.h"

#define fix_nb 1
#define _5d_col_liearization_low_bound 96
#define _5d_row_liearization_low_bound 4

namespace inplace {
namespace _5d {

	void init_dims(void* dim, int& d1, int& d2, int& d3, int& d4, int &d5) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
		d4 = int_dim[3];
		d5 = int_dim[4];
	}

	void stride_generate(size_t *stride, int *int_dim, int rank) { 
		size_t int_dim_long[5];
		std::copy(int_dim, int_dim + rank, int_dim_long);
		stride[0] = 1;
		std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
	}

	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double ALPHA) {
		int d1, d2, d3, d4, d5, n_threads;//, rank = 5;
		init_dims(dim, d1, d2, d3, d4, d5);
		if(msb(d1) <= 18) { n_threads = 32;}
		else { n_threads = 256;}
		//printf("5d transpose.\n");
		switch (type) {
			
			default:
				printf("Invalid rank 5 permutation\n");
				//_Nd::transpose(data, source, dim, type, num_block, tensor_vol, ALPHA);
				return;
		}
	}
	template void transpose(float*, int, void*, int, int, int, double);
	template void transpose(double*, int, void*, int, int, int, double);

	namespace _14325 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double ALPHA) {
			int vol = d1 * d2 * d3 * d4 * d5, rank = 5, max_dim_pos, n_threads, type_size = sizeof(T);
			int d1d2d3d4 = d1 * d2 * d3 * d4, d2d3d4d5 = d2 * d3 * d4 * d5;
			size_t data_size = sizeof(T) * vol;
			int int_dim[5] = {d1, d2, d3, d4, d5};
			int perm[5] = {0, 3, 2, 1, 4};	
			if(msb(d1) <= 18) { n_threads = 32;}
			else { n_threads = 256;}
			prefetch(data, data_size);

			if(d1 >= _5d_col_liearization_low_bound) {
				//if(source == 0){ PRINT("\n5d 14325 col linearization\n");}
				if(n_threads * d2d3d4d5 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);
					int max_dim = *std::max_element(int_dim, int_dim + rank);
					int max_dim_pos = -1;
					for(int i = rank - 1; i >= 1; --i) {
						if(int_dim[i] == max_dim) {
							max_dim_pos = i;
							break;
						}
					}
					assert(max_dim_pos != -1);
					int tmp_vol = n_threads * d2d3d4d5, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
					if(source == 0){printf("\n14325 low-order decomposition, d1 is large enough, nb = %d\n", nb);}
					
					if(nb > 1) {
						if(origin_max_dim_size % nb != 0) {
							size_t stride[6];
							stride_generate(stride, int_dim, rank);
							inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4];
						}
						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) {
							_2d::col_op(_5d::_14325::row_permute(int_dim[4], int_dim[3], int_dim[2], int_dim[1]),
							 data, source + i * sub_vol, int_dim[0], int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4]);
						}
						int_dim[max_dim_pos] *= nb;
						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0) {
							int_dim[max_dim_pos] =  origin_max_dim_size;
							vol = d1 * d2d3d4d5;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else { _2d::col_op(_5d::_14325::row_permute(d5, d4, d3, d2), data, source, d1,  d2d3d4d5);}
				} else { _2d::col_op(_5d::_14325::row_permute(d5, d4, d3, d2), data, source, d1,  d2d3d4d5);}
			} else if(d5 >= _5d_row_liearization_low_bound) { //  && d1d2d3d4 < ALPHA * tensor_vol
				//if(source == 0){ PRINT("\n5d 14325 row linearization\n");}
				if(d1d2d3d4 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);
					int max_dim = *std::max_element(int_dim, int_dim + rank);
					int max_dim_pos = -1;
					for(int i = rank - 2; i >= 0; --i) {
						if(int_dim[i] == max_dim) {
							max_dim_pos = i;
							break;
						}
					}
					assert(max_dim_pos != -1);
					int tmp_vol = d1d2d3d4, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
					if(source == 0){printf("\n14325 low-order decomposition, d5 is large enough, nb = %d\n", nb);}

					if(nb > 1){
						if(origin_max_dim_size % nb != 0) {
							size_t stride[6];
							stride_generate(stride, int_dim, rank);
							inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4];
						}
						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) {
							_2d::row_gather_op(_5d::_14325::row_shuffle(int_dim[3], int_dim[1], int_dim[2], int_dim[0]), 
							data, source + i * sub_vol, int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3], int_dim[4]);
						}
						int_dim[max_dim_pos] *= nb;

						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0) {
							int_dim[max_dim_pos] =  origin_max_dim_size;
							vol = d1 * d2d3d4d5;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else { _2d::row_gather_op(_5d::_14325::row_shuffle(d4, d2, d3, d1), data, source, d1d2d3d4, d5);}
				} else { _2d::row_gather_op(_5d::_14325::row_shuffle(d4, d2, d3, d1), data, source, d1d2d3d4, d5);}
			}
		}
		template void transpose(float*, int, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, int, double);
	}

	namespace _15432 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double ALPHA)
		{
			int vol = d1 * d2 * d3 * d4 * d5, d2d3d4d5 = d2 * d3 * d4 * d5, rank = 5, max_dim_pos, n_threads, type_size = sizeof(T);
			size_t data_size = sizeof(T) * vol;
			int int_dim[5] = {d1, d2, d3, d4, d5};
			int perm[5] = {0, 4, 3, 2, 1};	
			if(msb(d1) <= 18) { n_threads = 32;}
			else { n_threads = 256;}
			prefetch(data, data_size);

			//if(source == 0){ PRINT("\n5d 15432 col linearization\n");}
			if(n_threads * d2d3d4d5 > ALPHA * tensor_vol && fix_nb == 1) {
				assert(source == 0);
				int max_dim = *std::max_element(int_dim, int_dim + rank);
				int max_dim_pos = -1;
				for(int i = rank - 1; i >= 1; --i) {
					if(int_dim[i] == max_dim) {
						max_dim_pos = i;
						break;
					}
				}
				assert(max_dim_pos != -1);
				int tmp_vol = n_threads * d2d3d4d5, origin_max_dim_size = int_dim[max_dim_pos];
				int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
				if(source == 0){printf("\n15432 low-order decomposition, d1 is large enough, nb = %d\n", nb);}
					
				if(nb > 1) {
					if(origin_max_dim_size % nb != 0) {
						size_t stride[6];
						stride_generate(stride, int_dim, rank);
						inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
						int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
						vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4];
					}
					inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

					int_dim[max_dim_pos] /= nb;
					int sub_vol = vol / nb;
					for(int i = 0; i < nb; ++i) {
						_2d::col_op(_5d::_15432::row_permute(int_dim[4], int_dim[3], int_dim[2], int_dim[1]),
						 data, source + i * sub_vol, int_dim[0], int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4]);
					}
					int_dim[max_dim_pos] *= nb;

					inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
					if(origin_max_dim_size % nb != 0) {
						int_dim[max_dim_pos] =  origin_max_dim_size;
						vol = d1 *  d2 * d3 * d4 * d5;
						inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
					}
				} else { _2d::col_op(_5d::_15432::row_permute(d5, d4, d3, d2), data, source, d1, d2 * d3 * d4 * d5);}
			} else { _2d::col_op(_5d::_15432::row_permute(d5, d4, d3, d2), data, source, d1, d2 * d3 * d4 * d5);}	
		}
		template void transpose(float*, int, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, int, double);
	}
	
	namespace _43215 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5, int tensor_vol, double ALPHA)
		{
			int vol = d1 * d2 * d3 * d4 * d5, d1d2d3d4 = d1 * d2 * d3 * d4, rank = 5, max_dim_pos, type_size = sizeof(T);
			size_t data_size = sizeof(T) * vol;
			int int_dim[5] = {d1, d2, d3, d4, d5};
			int perm[5] = {3, 2, 1, 0, 4};	
			prefetch(data, data_size);
			//if(source == 0){ PRINT("\n5d 43215 row linearization\n");}
			if(d1d2d3d4 > ALPHA * tensor_vol && fix_nb == 1) {
				assert(source == 0);
				int max_dim = *std::max_element(int_dim, int_dim + rank);
				int max_dim_pos = -1;
				for(int i = rank - 2; i >= 0; --i) {
					if(int_dim[i] == max_dim) {
						max_dim_pos = i;
						break;
					}
				}
				assert(max_dim_pos != -1);
				int tmp_vol = d1d2d3d4, origin_max_dim_size = int_dim[max_dim_pos];
				int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
				if(source == 0){printf("\n43215 low-order decomposition, d5 is large enough, nb = %d\n", nb);}

				if(nb > 1) {
					if(origin_max_dim_size % nb != 0) {
						size_t stride[6];
						stride_generate(stride, int_dim, rank);
						inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
						int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
						vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3] * int_dim[4];
					}
					inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

					int_dim[max_dim_pos] /= nb;
					int sub_vol = vol / nb;
					for(int i = 0; i < nb; ++i) {
						_2d::row_gather_op(_5d::_43215::row_shuffle(int_dim[3], int_dim[1], int_dim[2], int_dim[0]), 
						data, source + i * sub_vol, int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3], int_dim[4]);
					}
					int_dim[max_dim_pos] *= nb;

					inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
					if(origin_max_dim_size % nb != 0) {
						int_dim[max_dim_pos] =  origin_max_dim_size;
						vol = d1 *  d2 * d3 * d4 * d5;
						inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
					}
				} else { _2d::row_gather_op(_5d::_43215::row_shuffle(d4, d2, d3, d1), data, source, d1 * d2 * d3 * d4, d5);}
			} else { _2d::row_gather_op(_5d::_43215::row_shuffle(d4, d2, d3, d1), data, source, d1 * d2 * d3 * d4, d5);}
		}
		template void transpose(float*, int, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, int, double);
	}
}
}