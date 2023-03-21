#include <cstdio>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "gcd.h"
#include "util.h"
#include "debug.h"
#include "col_op.h"
#include "row_op.h"
#include "equations.h"
#include "Nd_padding.h"
#include "2dtranspose.h"
#include "Nd_decompose.h"
#include "memory_estimation.h"

#define fix_nb 1

// tensor vol * sizeof(T) = input size, vol may be sub-tensor vol

namespace inplace {

namespace _2d {

	void init_dims(void* dim, int& d1, int& d2) {
		int *int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
	}

	void stride_generate(size_t *stride, int *int_dim, int rank) { 
		size_t int_dim_long[2];
		std::copy(int_dim, int_dim + rank, int_dim_long);
		stride[0] = 1;
		std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
	}

	template<typename T>
	void transpose(T* data, int source, void* dim, int tensor_vol, double ALPHA) {
		int d1, d2, rank = 2;
		int perm[2] = {1, 0};
		init_dims(dim, d1, d2);
		int int_dim[2] = {d1, d2};
		int type_size = sizeof(T);
		size_t data_size = type_size * d1 * d2;
		int vol = d1 * d2, max_dim_pos;
		prefetch(data, data_size); // for cudaManaged

/*		for(int numblock = 1; numblock <= 64; numblock *= 4) {
			if(d1 > d2) { Extra_memory_estimation(int_dim, perm, d1 * sizeof(T), data_size, rank, tensor_vol, numblock, 0, sizeof(T));}
			else { Extra_memory_estimation(int_dim, perm, d2 * sizeof(T), data_size, rank, tensor_vol, numblock, 1, sizeof(T));}
		}
		return;*/

		if(d1 > ALPHA * tensor_vol && fix_nb == 1) {
			assert(source == 0);
			max_dim_pos = 0;
			int tmp_vol = d1;
			int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
			printf("21 low-order decomposition, c2r, nb = %d\n", nb);

			if(nb > 1) {
				if(d1 % nb != 0) {
					size_t stride[3]; 
					stride_generate(stride, int_dim, rank);
					inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
					int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
					vol = int_dim[0] * int_dim[1];
				}

				inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);

				int_dim[max_dim_pos] /= nb;
				int sub_vol = vol / nb;
				for(int i = 0; i < nb; ++i) { c2r::transpose(data, source + i * sub_vol, int_dim[0], int_dim[1]);}
				int_dim[max_dim_pos] *= nb;

				inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);
				if(d1 % nb != 0) {
					int_dim[max_dim_pos] = d1;
					vol = d1 * d2;
					inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
				}
			} else { c2r::transpose(data, source, d1, d2);}	
		}
		else if(d2 > ALPHA * tensor_vol && fix_nb == 1) {
			assert(source == 0);
			max_dim_pos = 1;
			int tmp_vol = d2;
			int nb = memory_estimation(int_dim, perm, rank, max_dim_pos, tmp_vol, tensor_vol, ALPHA, type_size);
			printf("21 low-order decomposition, r2c, nb = %d\n", nb);
			
			if(nb > 1) { 
				if(d2 % nb != 0) {
					size_t stride[3]; 
					stride_generate(stride, int_dim, rank);
					inplace::Nd_padding_pre_process(data + source, int_dim, stride, rank, max_dim_pos, nb);
					int_dim[max_dim_pos] = (int_dim[max_dim_pos] / nb + 1) * nb;
					vol = int_dim[0] * int_dim[1];
				}

				inplace::NdPartition(data + source, int_dim, rank, nb, max_dim_pos, ALPHA);
			
				int_dim[max_dim_pos] /= nb;
				int sub_vol = vol / nb;

				for(int i = 0; i < nb; ++i) { r2c::transpose(data, source + i * sub_vol, int_dim[1], int_dim[0]);}
				int_dim[max_dim_pos] *= nb;

				inplace::NdJoin(data + source, int_dim, perm, rank, nb, max_dim_pos, ALPHA);

				if(d2 % nb != 0) {
					int_dim[max_dim_pos] = d2;
					vol = d1 * d2;
					inplace::Nd_padding_post_process(data + source, int_dim, perm, rank, max_dim_pos, nb);	
				}
			} else { r2c::transpose(data, source, d2, d1);}
		}
		else {
			if (d1 >= d2) { c2r::transpose(data, source, d1, d2);}
			else { r2c::transpose(data, source, d2, d1);}
		}
	}
	
	template void transpose(float*, int, void*, int, double);
	template void transpose(double*, int, void*, int, double);

	namespace c2r {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2) {
			//PRINT("Doing C2R transpose\n");
			//PRINT("d1' mod 32 = %d\n", d1 % 32);
			
			int c, t, k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d1 / c;
			if (c > 1) {
				col_op(c2r::rotate(d2, b), data, source, d1, d2);
			}
			row_gather_op(c2r::row_shuffle(d2, d1, c, k), data, source, d1, d2);
			col_op(c2r::col_shuffle(d2, d1, c), data, source, d1, d2);
		}
	}

	namespace r2c {
		template<typename T>
		void transpose(T* data, int source, int d1, int d2) {
			//PRINT("Doing R2C transpose\n");
			//PRINT("d1' mod 32 = %d\n", d1 % 32);

			int c, t, q;
			extended_gcd(d1, d2, c, t);
			if (c > 1) {
				extended_gcd(d1/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d1 / c;
			
			int k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}
			
			col_op(r2c::col_shuffle(a, c, d2, q), data, source, d1, d2);
			row_scatter_op(r2c::row_scatter_shuffle(d2, d1, c, k), data, source, d1, d2);
			if (c > 1) {
				col_op(r2c::rotate(d2, b), data, source, d1, d2);
			}
		}
	}
}

}