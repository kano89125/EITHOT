#include <cstdio>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "gcd.h"
#include "util.h"
#include "debug.h"
#include "row_op.h"
#include "col_op.h"
#include "equations.h"
#include "Nd_padding.h"
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "Nd_transpose.h"
#include "Nd_decompose.h"
#include "memory_estimation.h"

#define fix_nb 0
#define _132_col_liearization_low_bound 48
#define _132_catanzaro_low_bound 12
#define _213_row_liearization_low_bound 8

namespace inplace {
namespace _3d {

	void init_dims(void* dim, int& d1, int& d2, int& d3) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
	}

	void stride_generate(size_t *stride, int *int_dim, int rank) { 
		size_t int_dim_long[3];
		std::copy(int_dim, int_dim + rank, int_dim_long);
		stride[0] = 1;
		std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
	}

	template<typename T>
	void transpose(T* data, int source, void* dim, int type, int num_block, int tensor_vol, double ALPHA) {
		int d1, d2, d3;//, rank = 3;
		init_dims(dim, d1, d2, d3);
		switch (type) {
			case 231:
				_231::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				return;
			case 312:
				_312::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				return;
			case 213:
				_213::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				return;
			case 132:
				_132::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				return;
			/*case 321:
				//int perm[3] = {2,1,0};
				//_Nd::transpose(data, source, dim, perm, rank, num_block, tensor_vol, ALPHA);
				return;*/
			default:
				printf("Invalid permutation\n");
				return;
		}
	}
	template void transpose(float*, int, void*, int, int, int, double);
	template void transpose(double*, int, void*, int, int, int, double);

	namespace _231 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double ALPHA) {
			int dim[2];
			dim[0] = d1;
			dim[1] = d2 * d3;
			_2d::transpose(data, source, dim, tensor_vol, ALPHA);
		}
		template void transpose(float*, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, double);
	}
	
	namespace _312 {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double ALPHA) {
			int dim[2];
			dim[0] = d1 * d2;
			dim[1] = d3;
			_2d::transpose(data, source, dim, tensor_vol, ALPHA);
		}
		template void transpose(float*, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, double);
	}

	namespace _213 {
		template<typename T>
		void c2r(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing C2R transpose\n");
			int c, t, k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) { extended_gcd(d2/c, d1/c, t, k);} 
			else { k = t;}
			int a = d2 / c;
			int b = d1 / c;
			if (c > 1) { col_op(_2d::c2r::rotate(d2, b), data, source, d1, d2, d3);}
			row_gather_op(_2d::c2r::row_shuffle(d2, d1, c, k), data, source, d1, d2, d3);
			col_op(_2d::c2r::col_shuffle(d2, d1, c), data, source, d1, d2, d3);
		}
		
		template<typename T>
		void r2c(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d1, d2, c, t);
			if (c > 1) { extended_gcd(d1/c, d2/c, t, q);} 
			else { q = t;}
			
			int a = d2 / c;
			int b = d1 / c;
			int k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) { extended_gcd(d2/c, d1/c, t, k);} 
			else { k = t;}
			
			col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, source, d1, d2, d3);
			row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d1, c, k), data, source, d1, d2, d3);
			if (c > 1) { col_op(_2d::r2c::rotate(d2, b), data, source, d1, d2, d3);}
		}
		
		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double ALPHA) {
			int vol = d1 * d2 * d3, rank = 3, max_dim_pos, type_size = sizeof(T);
			size_t data_size = sizeof(T) * d1 * d2 * d3;
			int int_dim[3] = {d1, d2, d3};
			int perm[3] = {1, 0, 2};
			prefetch(data, data_size);

			if (d3 >= _213_row_liearization_low_bound) { // d1d2, d1/d2 is large
			//if (d1 * d2 / ((double)d1 * d2 * d3) < 0.01) {
				if(d1 * d2 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);				
					if(d1 > d2) { max_dim_pos = 0;}
					else { max_dim_pos = 1;} 
					int tmp_vol = d1 * d2, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);

					if(source == 0){ printf("\n213 low-order decomposition, d3 is large enough, nb = %d\n", nb);}

					if(nb > 1) {
						if(origin_max_dim_size % nb != 0) {
							size_t stride[4];
							stride_generate(stride, int_dim, rank);
							inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2];
						}

						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) { _2d::row_gather_op(_213::row_shuffle(int_dim[0], int_dim[1]), data, source + i * sub_vol, int_dim[0] * int_dim[1], int_dim[2]);}
						int_dim[max_dim_pos] *= nb;
						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0) {
							int_dim[max_dim_pos] =  origin_max_dim_size;
							vol = d1 * d2 * d3;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else { _2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3);}	
				} else { _2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3);}
			}
			else {
				if (d1 > d2) {  // d1, d1 is large
					if(d1 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 0;
						int tmp_vol = d1;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){ printf("\n213 low-order decomposition, c2r, nb = %d\n", nb);}
						
						if(nb > 1) {
							if(d1 % nb != 0) {
								size_t stride[4];
								stride_generate(stride, int_dim, rank);
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) { c2r(data, source + i * sub_vol, int_dim[0], int_dim[1], int_dim[2]);}
							int_dim[max_dim_pos] *= nb;
							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d1 % nb != 0) {
								int_dim[max_dim_pos] =  d1;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						} else { c2r(data, source, d1, d2, d3);}
					} else { c2r(data, source, d1, d2, d3);}
				} else { // d2, d2 is large
					if(d2 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 1;
						int tmp_vol = d2;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){PRINT("\n213 low-order decomposition, r2c, nb = %d\n", nb);}
						
						if(nb > 1) {
							if(d2 % nb != 0) {
								size_t int_dim_long[3];
								std::copy(int_dim, int_dim + rank, int_dim_long);
								size_t stride[4];
								stride[0] = 1;
								std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) { r2c(data, source + i * sub_vol, int_dim[1], int_dim[0], int_dim[2]);}
							int_dim[max_dim_pos] *= nb;

							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d2 % nb != 0) {
								int_dim[max_dim_pos] =  d2;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						} else { r2c(data, source, d2, d1, d3);}
					} else { r2c(data, source, d2, d1, d3);}	
				}
			}
		}
		template void transpose(float*, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, double);
	}
	
	namespace _132 {
		template<typename T>
		void c2r(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing C2R transpose\n");
			
			int c, t, k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d3 / c;
			if (c > 1) {
				col_op(_2d::c2r::rotate(d2, b), data, source, d1, d2, d3);
			}
			row_gather_op(_2d::c2r::row_shuffle(d2, d3, c, k), data, source, d1, d2, d3);
			col_op(_2d::c2r::col_shuffle(d2, d3, c), data, source, d1, d2, d3);
		}

		template<typename T>
		void r2c(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d3, d2, c, t);
			if (c > 1) {
				extended_gcd(d3/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d3 / c;
			
			int k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}
			
			col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, source, d1, d2, d3);
			row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d3, c, k), data, source, d1, d2, d3);
			if (c > 1) {
				col_op(_2d::r2c::rotate(d2, b), data, source, d1, d2, d3);
			}
		}

		template<typename T>
		void transpose(T* data, int source, int d1, int d2, int d3, int tensor_vol, double ALPHA) {
			int vol = d1 * d2 * d3, rank = 3, max_dim_pos, n_threads, type_size = sizeof(T);
			if(msb(d1) <= 18) { n_threads = 32;}
			else { n_threads = 256;}
			size_t data_size = sizeof(T) * d1 * d2 * d3;
			int int_dim[3] = {d1, d2, d3};
			int perm[3] = {0 ,2, 1};
			prefetch(data, data_size);

			if (d1 >= _132_col_liearization_low_bound) { // 32d2d3, d2/d3 is large	
				// if d1 > 2^18, then 256*d2*d3, no need to split
				if(n_threads * d2 * d3 > ALPHA * tensor_vol && fix_nb == 1) {
					assert(source == 0);
					if(d2 > d3) { max_dim_pos = 1;} 
					else { max_dim_pos = 2;} 
					int tmp_vol = d2 * d3 * n_threads, origin_max_dim_size = int_dim[max_dim_pos];
					int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);

					if(nb > 1) {
						if(origin_max_dim_size % nb != 0) {
							size_t stride[4];
							stride_generate(stride, int_dim, rank);
							inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
							int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
							vol = int_dim[0] * int_dim[1] * int_dim[2];
						}

						inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

						int_dim[max_dim_pos] /= nb;
						int sub_vol = vol / nb;
						for(int i = 0; i < nb; ++i) { _2d::col_op(_132::row_permute(int_dim[2], int_dim[1]), data, source + i * sub_vol, int_dim[0], int_dim[1] * int_dim[2]);}
						int_dim[max_dim_pos] *= nb;
						
						inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
						if(origin_max_dim_size % nb != 0) {
							int_dim[max_dim_pos] = origin_max_dim_size;
							vol = d1 * d2 * d3;
							inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
						}
					} else{ _2d::col_op(_132::row_permute(d3, d2), data, source, d1, d2 * d3);}
				} else { _2d::col_op(_132::row_permute(d3, d2), data, source, d1, d2 * d3);}
			}
			else if (d1 > _132_catanzaro_low_bound)
			{ // original d1 > 2, position of each d1, d2, d3???

				if (d2 > d3) { // max{d1d2, cd3}, d2 is large
					if(d1 * d2 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 1;
						int tmp_vol = d1 * d2;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){ PRINT("\n132 low-order decomposition, catanzaro method for d2 > d3, nb = %d\n", nb);}

						if(nb > 1) {
							if(d2 % nb != 0) {
								size_t stride[4];
								stride_generate(stride, int_dim, rank);
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) { c2r(data, source + i * sub_vol, int_dim[0], int_dim[2], int_dim[1]);}
							int_dim[max_dim_pos] *= nb;
							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d2 % nb != 0) {
								int_dim[max_dim_pos] = d2;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						}
						else { c2r(data, source, d1, d3, d2);}
					} else { c2r(data, source, d1, d3, d2);}
				} else { // max{d1d3, cd2}, d3 is large
					if(d1 * d3 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 2;					
						int tmp_vol = d1 * d3;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){ PRINT("\n132 low-order decomposition, catanzaro method for d2 <= d3, nb = %d\n", nb);}

						if(nb > 1) {
							if(d3 % nb != 0) {
								size_t stride[4];
								stride[0] = 1;
								stride_generate(stride, int_dim, rank);
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) { r2c(data, source + i * sub_vol, int_dim[0], int_dim[1], int_dim[2]);}
							int_dim[max_dim_pos] *= nb;
							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d3 % nb != 0) {
								int_dim[max_dim_pos] = d3;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						} else { r2c(data, source, d1, d2, d3);}
					} else { r2c(data, source, d1, d2, d3);}
				}
			} else {	
				if (d2 >= d3) { // max{d1d2, d3, d3d1} = d1d2, d2 is large, d1d3 < alpha * vol
					if(d1 * d2 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 1;
						int tmp_vol = d1 * d2;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){ PRINT("\n132 low-order decomposition, combination method, nb = %d\n", nb);}

						if(nb > 1) {
							if(d2 % nb != 0) {
								size_t stride[4];
								stride_generate(stride, int_dim, rank);
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) {
								_312::transpose(data, source + i * sub_vol, int_dim[0], int_dim[1], int_dim[2], tensor_vol, ALPHA);
								_2d::row_gather_op(_213::row_shuffle(int_dim[2], int_dim[0]), data, source + i * sub_vol, int_dim[2] * int_dim[0], int_dim[1]);
							}
							int_dim[max_dim_pos] *= nb;

							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d2 % nb != 0) {
								int_dim[max_dim_pos] = d2;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						}
						else {
							_312::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
							_2d::row_gather_op(_213::row_shuffle(d3, d1), data, source, d3 * d1, d2);
						}
					}
					else {
						_312::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
						_2d::row_gather_op(_213::row_shuffle(d3, d1), data, source, d3 * d1, d2);
					}
				}
				else { // max{d1d3,d1d2, d2}, combination method
					if(d1 * d3 > ALPHA * tensor_vol && fix_nb == 1) {
						assert(source == 0);
						max_dim_pos = 2;
						int tmp_vol = d1 * d3;
						int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
						if(source == 0){ PRINT("\n132 low-order decomposition, combination method, nb = %d\n", nb);}

						if(nb > 1) {
							if(d3 % nb != 0) {		
								size_t stride[4];	
								stride_generate(stride, int_dim, rank);	
								inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
								int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
								vol = int_dim[0] * int_dim[1] * int_dim[2];
							}

							inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

							int_dim[max_dim_pos] /= nb;
							int sub_vol = vol / nb;
							for(int i = 0; i < nb; ++i) {
								_2d::row_gather_op(_213::row_shuffle(int_dim[0], int_dim[1]), data, source + i * sub_vol, int_dim[0] * int_dim[1], int_dim[2]);
								_231::transpose(data, source + i * sub_vol, int_dim[1], int_dim[0], int_dim[2], tensor_vol, ALPHA);
							}
							int_dim[max_dim_pos] *= nb;
							
							inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
							if(d3 % nb != 0) {
								int_dim[max_dim_pos] = d3;
								vol = d1 * d2 * d3;
								inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
							}
						} else {
							_2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3);
							_231::transpose(data, source, d2, d1, d3, tensor_vol, ALPHA);
						}
					} else {
						_2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3);
						_231::transpose(data, source, d2, d1, d3, tensor_vol, ALPHA);
					}
				}
			}
		}
		template void transpose(float*, int, int, int, int, int, double);
		template void transpose(double*, int, int, int, int, int, double);
	}
}
}
