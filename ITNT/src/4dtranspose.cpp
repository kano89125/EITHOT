#include <cstdio>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "util.h"
#include "debug.h"
#include "row_op.h"
#include "col_op.h"
#include "equations.h"
#include "Nd_padding.h"
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "Nd_transpose.h"
#include "Nd_decompose.h"
#include "memory_estimation.h"


#define fix_nb 0
#define _1324_col_liearization_low_bound 96 // 96
#define _1324_row_liearization_low_bound 4

namespace inplace {

namespace _4d {

	void init_dims(void* dim, int& d1, int& d2, int& d3, int& d4) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
		d4 = int_dim[3];
	}

	void stride_generate(size_t *stride, int *int_dim, int rank) { 
		size_t int_dim_long[4];
		std::copy(int_dim, int_dim + rank, int_dim_long);
		stride[0] = 1;
		std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
	}

	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double ALPHA) {
		int d1, d2, d3, d4, n_threads;//, rank = 4;
		init_dims(dim, d1, d2, d3, d4);
		if(msb(d1) <= 18) { n_threads = 32;}
		else { n_threads = 256;}

		switch (type) {
			case 1324: // fix d1 = 1
				_1324::transpose(data, source, d1, d2, d3, d4, tensor_vol, ALPHA);
				return;
			default:
				printf("Invalid rank 4 permutation\n");
				//_Nd::transpose(data, source, dim, type, num_block, tensor_vol, ALPHA);
				return;
		}

	}
	template void transpose(float*, int, void*, int, int, int, double);
	template void transpose(double*, int, void*, int, int, int, double);

	// col row linearization
	namespace _1324 {

		
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double ALPHA) {
			int vol = d1 * d2 * d3 * d4, rank = 4, max_dim_pos, n_threads, type_size = sizeof(T);
			int int_dim[4] = {d1, d2, d3, d4};
			int perm[4] = {0, 2, 1, 3};	
			size_t data_size = sizeof(T) * vol;
			if(msb(d1) <= 18) { n_threads = 32;}
			else { n_threads = 256;}
			prefetch(data, data_size);


			if(d1 >= _1324_col_liearization_low_bound && n_threads * d2 * d3 * d4 <= ALPHA * tensor_vol)
			{ _2d::col_op(_4d::_1324::row_permute(d3, d2), data, source, d1, d2 * d3 * d4);}
			else if(d4 >= _1324_row_liearization_low_bound && d1 * d2 * d3 > ALPHA * tensor_vol)
			{ _2d::row_gather_op(_4d::_1324::row_shuffle(d2, d3, d1), data, source, d1 * d2 * d3, d4);}
			else if(d4 >= _1324_row_liearization_low_bound) {//d1d2d3, d4 < 100, max nb = 16
				//if(source == 0){ PRINT("\n4d 1324 row linearization\n");}
				if (d1 * d2 * d3 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);

					int max_dim = *std::max_element(int_dim, int_dim + rank - 1);
					int max_dim_pos = -1;
					for(int i = rank - 2; i >= 0; --i) {
						if(int_dim[i] == max_dim) {
							max_dim_pos = i;
							break;
						}
					}
					assert(max_dim_pos != -1);
					int tmp_vol = d1 * d2 * d3, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);

					if(nb > 1) {
						if(origin_max_dim_size % nb != 0) {							
							size_t stride[5];
							stride_generate(stride, int_dim, rank);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3];
						}

						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) {
							_2d::row_gather_op(_4d::_1324::row_shuffle(int_dim[1], int_dim[2], int_dim[0]), data, source + i * sub_vol, int_dim[0] * int_dim[1] * int_dim[2], int_dim[3]);
						}
						int_dim[max_dim_pos] *= nb;

						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0)
						{
							int_dim[max_dim_pos] =  origin_max_dim_size;
							vol = d1 * d2 * d3 * d4;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else { _2d::row_gather_op(_4d::_1324::row_shuffle(d2, d3, d1), data, source, d1 * d2 * d3, d4);}
				} else { _2d::row_gather_op(_4d::_1324::row_shuffle(d2, d3, d1), data, source, d1 * d2 * d3, d4);}	
			} else if(d1 >= _1324_col_liearization_low_bound) { 	
				// cd2d3d4, 31 <= d1 <= 3200, d4 < 8, max nb = 128
				//if(source == 0){ PRINT("\n4d 1324 col linearization\n");}
				if(n_threads * d2 * d3 * d4 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);		
					int max_dim = *std::max_element(int_dim + 1, int_dim + rank);
					int max_dim_pos = -1;
					for(int i = rank - 1; i >= 1; --i) {
						if(int_dim[i] == max_dim) {
							max_dim_pos = i;
							break;
						}
					}
					assert(max_dim_pos != -1);
					int tmp_vol = n_threads * d2 * d3 * d4, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
					
					if(source == 0){printf("\n1324 low-order decomposition, d1 is large enough, nb = %d\n", nb);}

					if(nb > 1) {
						if(origin_max_dim_size % nb != 0) {
							size_t stride[5];
							inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3];
						}
						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) {
							_2d::col_op(_4d::_1324::row_permute(int_dim[2], int_dim[1]), data, source + i * sub_vol, int_dim[0], int_dim[1] * int_dim[2] * int_dim[3]);
						}
						int_dim[max_dim_pos] *= nb;
						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0) {
							int_dim[max_dim_pos] =  origin_max_dim_size;
							vol = d1 * d2 * d3 * d4;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else { _2d::col_op(_4d::_1324::row_permute(d3, d2), data, source, d1, d2 * d3 * d4);}
				} else { _2d::col_op(_4d::_1324::row_permute(d3, d2), data, source, d1, d2 * d3 * d4);}
			} else { //no memory reducing here
				//if(source == 0){PRINT("\n4d 1324 combination method\n");}
				if(d2 >= d3) { // d1, d4 is small, d3
					//if(source == 0){ PRINT("\n4d 1324 d2 > d3\n");}
					// 213 c2r, max {d3, d1d2} = d1d2, split in 3d
					_3d::_213::transpose(data, source, d1 * d2, d3, d4, tensor_vol, ALPHA);
					// row linearization, d1d3 is small enough
					_3d::_213::transpose(data, source, d3, d1, d2 * d4, tensor_vol, ALPHA);

				} else { //if(source == 0){ PRINT("\n4d 1324 d2 <= d3\n");}
					// d1d2
					_2d::row_gather_op(_3d::_213::row_shuffle(d1, d2), data, source, d1 * d2, d3 * d4);
					// max d2, d1d3d4 = d1d3d4, split in 2d
					_3d::_231::transpose(data, source, d2, d1, d3 * d4, tensor_vol, ALPHA);
					// cd2d4
					_2d::col_op(_3d::_132::row_permute(d2, d4), data, source, d1 * d3, d4 * d2);
				}
			}
			
		}
		
		template void transpose(float*, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, double);
	}

	// col linearization
	namespace _1432 { 
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double ALPHA) {
			int vol = d1 * d2 * d3 * d4, d2d3d4 = d2 * d3 * d4, rank = 4, max_dim_pos, n_threads, type_size = sizeof(T);
			size_t data_size = sizeof(T) * vol;
			int perm[4] = {0, 3, 2, 1};
			int int_dim[4] = {d1, d2, d3, d4};
			if(msb(d1) <= 18) { n_threads = 32;}
			else { n_threads = 256;}
			prefetch(data, data_size);
			//if(source == 0){ PRINT("\n4d 1432 col linearization\n");}
			if(n_threads * d2d3d4 > ALPHA * tensor_vol && fix_nb == 1) {
				assert(source == 0);
				int max_dim = *std::max_element(int_dim + 1, int_dim + rank);
				int max_dim_pos = -1;
				for(int i = rank - 1; i >= 1; --i) {
					if(int_dim[i] == max_dim)  {
						max_dim_pos = i;
						break;
					}
				}
				assert(max_dim_pos != -1);
				int tmp_vol = n_threads * d2d3d4, origin_max_dim_size = int_dim[max_dim_pos];
				int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
				
				if(source == 0){PRINT("\n1432 low-order decomposition, d1 is large enough, nb = %d\n", nb);}

				if(nb > 1) {
					if(origin_max_dim_size % nb != 0) {
						//printf("Dimension Pre-padding %d %d\n", int_dim[0], int_dim[1]);
						size_t int_dim_long[4];
						std::copy(int_dim, int_dim + rank, int_dim_long);
						size_t stride[5];
						stride[0] = 1;
						std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
						inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
						int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
						vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3];
					}

					inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

					int_dim[max_dim_pos] /= nb;
					int sub_vol = vol / nb;
					//printf("sub-tensor shape = %d %d\n", int_dim[0], int_dim[1], int_dim[2], int_dim[3]);
					for(int i = 0; i < nb; ++i) {
						_2d::col_op(_4d::_1432::row_permute(int_dim[3] , int_dim[2], int_dim[1]), data, source + i* sub_vol, int_dim[0], int_dim[1] * int_dim[2] * int_dim[3]);
					}
					int_dim[max_dim_pos] *= nb;
					inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
					if(origin_max_dim_size % nb != 0) {
						int_dim[max_dim_pos] =  origin_max_dim_size;
						vol = d1 * d2 * d3 * d4;
						inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
					}
				} else { _2d::col_op(_4d::_1432::row_permute(d4, d3, d2), data, source, d1, d2d3d4);}
			} else { _2d::col_op(_4d::_1432::row_permute(d4, d3, d2), data, source, d1, d2d3d4);}
		}

		template void transpose(float*, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, double);
	}

	// row linearization
	namespace _3214 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int d4, int tensor_vol, double ALPHA) {		
			int vol = d1 * d2 * d3 * d4, rank = 4, max_dim_pos, type_size = sizeof(T);
			int int_dim[4] = {d1, d2, d3, d4};
			int perm[4] = {2, 1, 0, 3};
			size_t data_size = sizeof(T) * vol;
			prefetch(data, data_size);
			//if(source == 0){ PRINT("\n4d 3214 row linearization\n");}

			if(d1 * d2 * d3 > ALPHA * tensor_vol && fix_nb == 1) {
				assert(source == 0);			
				int max_dim = *std::max_element(int_dim, int_dim + rank - 1);
				int max_dim_pos = -1;
				for(int i = rank - 2; i >= 0; --i) {
					if(int_dim[i] == max_dim) {
						max_dim_pos = i;
						break;
					}
				}
				assert(max_dim_pos != -1);
				int tmp_vol = d1 * d2 * d3, origin_max_dim_size = int_dim[max_dim_pos];
				int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);	
				if(source == 0){PRINT("\n3214 low-order decomposition, d4 is large enough, nb = %d\n", nb);}

				if(nb > 1) {
					if(origin_max_dim_size % nb != 0) {
						size_t int_dim_long[4];
						std::copy(int_dim, int_dim + rank, int_dim_long);
						size_t stride[5];
						stride[0] = 1;
						std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
						inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
						int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
						vol = int_dim[0] * int_dim[1] * int_dim[2] * int_dim[3];
					}
					inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

					int_dim[max_dim_pos] /= nb;
					int sub_vol = vol / nb;
					for(int i = 0; i < nb; ++i) {
						_2d::row_gather_op(_4d::_3214::row_shuffle(int_dim[1], int_dim[2], int_dim[0]), 
						data, source + i * sub_vol, int_dim[0]*int_dim[1]*int_dim[2], int_dim[3]);
					}
					int_dim[max_dim_pos] *= nb;
					inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
					if(origin_max_dim_size % nb != 0) {
						int_dim[max_dim_pos] =  origin_max_dim_size;
						vol = d1 * d2 * d3 * d4;
						inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
					}
				} else { _2d::row_gather_op(_4d::_3214::row_shuffle(d2, d3, d1), data, source, d1 * d2 * d3, d4);}
			} else { _2d::row_gather_op(_4d::_3214::row_shuffle(d2, d3, d1), data, source, d1 * d2 * d3, d4);}
		}

		template void transpose(float*, int, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, int, double);
	}
}
}