#include <cstdio>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "math.h"
#include "util.h"
#include "Nd_padding.h"
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "5dtranspose.h"
#include "Nd_transpose.h"
#include "Nd_decompose.h"
#include "permutation_util.h"
// #include "memory_estimation.h"

/*
	permutation ranks 1 to n here
*/

#define WARP_SIZE 32
#define PRE_ORDER_INDEX_REORDERING 0
#define POST_ORDER_INDEX_REORDERING 1

// use in method selection linearized condition
#define _132_col_liearization_low_bound 48
#define _213_row_liearization_low_bound 8
#define _1324_col_liearization_low_bound 96
#define _1324_row_liearization_low_bound 4

namespace inplace
{
	namespace _Nd
	{

		void init_dims(void *dim, int *int_dim, int rank)
		{
			int *cast_dim = reinterpret_cast<int *>(dim);
			for (int i = 0; i < rank; ++i)
			{
				int_dim[i] = cast_dim[i];
			}
		}

		void init_perm(void *perm, int *int_perm, int rank)
		{
			int *cast_perm = reinterpret_cast<int *>(perm);
			for (int i = 0; i < rank; ++i)
			{
				int_perm[i] = cast_perm[i] + 1;
			}
		}

		void init_stride(size_t *stride, int *int_dim, int rank)
		{
			size_t *int_dim_long = new size_t[rank];
			std::copy(int_dim, int_dim + rank, int_dim_long);
			stride[0] = 1;
			std::partial_sum(int_dim_long, int_dim_long + rank, stride + 1, std::multiplies<size_t>());
			delete[] int_dim_long;
		}

		template <typename T>
		void Low_order_transpose(T *data, int *low_order_dim, int *low_order_perm, int source, int low_order_rank, int round, int tensor_vol, double ALPHA)
		{

			bool valid_perm = true;
			int perm_int = 0, d1 = low_order_dim[0], d2 = low_order_dim[1], d3, d4;
			for (int i = 0; i < low_order_rank; ++i)
			{
				perm_int += pow(10, i) * low_order_perm[low_order_rank - i - 1];
			}
			// printf("%d\n", perm_int);
			switch (perm_int)
			{		
			case 21:
				
				_2d::transpose(data, source, low_order_dim, tensor_vol, ALPHA);
				break;
			case 231:
				d3 = low_order_dim[2];
				_3d::_231::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				break;
			case 312:
				d3 = low_order_dim[2];
				_3d::_312::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				break;
			case 213:
				d3 = low_order_dim[2];
				_3d::_213::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				break;
			case 132:
				d3 = low_order_dim[2];
				_3d::_132::transpose(data, source, d1, d2, d3, tensor_vol, ALPHA);
				break;
			case 1324:
				d3 = low_order_dim[2];
				d4 = low_order_dim[3];
				_4d::_1324::transpose(data, source, d1, d2, d3, d4, tensor_vol, ALPHA);
				break;
			default:
				printf("Unsupported permutation type %d\n", perm_int);
				valid_perm = false;
				assert(valid_perm == true);
				break;
			}
		}

		template void Low_order_transpose(float *, int *, int *, int, int, int, int, double);
		template void Low_order_transpose(double *, int *, int *, int, int, int, int, double);

		int Linearization_condition(int *dim, int *perm, int rank)
		{
			if (rank == 3)
			{
				if (perm[rank - 1] == rank && dim[rank - 1] >= _213_row_liearization_low_bound)
				{
					return 0;
				}
				else if (perm[0] == 1 && dim[0] >= _132_col_liearization_low_bound)
				{
					return 0;
				}
			}
			else if (rank == 4)
			{
				if (perm[rank - 1] == rank && dim[rank - 1] >= _1324_row_liearization_low_bound)
				{
					return 0;
				}
				else if (perm[0] == 1 && dim[0] >= _1324_col_liearization_low_bound)
				{
					return 0;
				}
			}
			return 1;
		}

		int non_linearization_count(int *tensor_dim, int *tensor_indices, int *transpose_permutation, std::vector<int> cur_tensor_indices, int rank, int method_type)
		{
			int non_linearization_method_count = 0;
			std::vector<int> target_perm_vec(rank, 0);
			target_perm_vec.assign(transpose_permutation, transpose_permutation + rank);
			for (int round = 0; round < rank; ++round)
			{
				int low_order_dim[4];
				int low_order_perm[4];
				int *high_order_trans_perm = new int[rank];
				int perm_int = 0, low_order_rank;

				if (method_type == PRE_ORDER_INDEX_REORDERING)
				{
					low_order_rank = pre_order_trans_perm_generation(cur_tensor_indices, tensor_dim, transpose_permutation, low_order_dim, low_order_perm, high_order_trans_perm, rank);
				}
				else
				{
					low_order_rank = post_order_trans_perm_generation(cur_tensor_indices, tensor_dim, transpose_permutation, low_order_dim, low_order_perm, high_order_trans_perm, rank);
				}

				for (int i = 0; i < low_order_rank; ++i)
				{
					perm_int += pow(10, i) * low_order_perm[low_order_rank - i - 1];
				}

				if (low_order_rank == rank + 1)
				{
					break;
				}
				non_linearization_method_count += Linearization_condition(low_order_dim, low_order_perm, low_order_rank);

				int *tmp_tensor_indices = new int[rank];
				for (int i = 0; i < rank; ++i)
				{
					tmp_tensor_indices[i] = cur_tensor_indices.at(high_order_trans_perm[i] - 1);
				}
				std::copy(tmp_tensor_indices, tmp_tensor_indices + rank, tensor_indices);
				cur_tensor_indices.assign(tmp_tensor_indices, tmp_tensor_indices + rank);

				if (cur_tensor_indices == target_perm_vec)
				{
					break;
				}

				delete[] tmp_tensor_indices;
				delete[] high_order_trans_perm;
			}
			return non_linearization_method_count;
		}

		int method_selection(int *tensor_dim, std::vector<int> cur_tensor_indices, int *transpose_permutation, int rank)
		{
			int reordering_method_type;
			int *test_tensor_indices = new int[rank];
			int pre_order_non_linear_count = non_linearization_count(tensor_dim, test_tensor_indices, transpose_permutation, cur_tensor_indices, rank, PRE_ORDER_INDEX_REORDERING);
			int post_order_non_linear_count = non_linearization_count(tensor_dim, test_tensor_indices, transpose_permutation, cur_tensor_indices, rank, POST_ORDER_INDEX_REORDERING);

			// printf("pre, post = %d %d\n", pre_order_non_linear_count, post_order_non_linear_count);
			if (pre_order_non_linear_count < post_order_non_linear_count)
			{
				reordering_method_type = PRE_ORDER_INDEX_REORDERING;
			}
			else
			{
				reordering_method_type = POST_ORDER_INDEX_REORDERING;
			}
			delete[] test_tensor_indices;
			return reordering_method_type;
		}

		template <typename T>
		void Series_Low_order_transpose(T *data, int source, int *tensor_dim, int *transpose_permutation, int rank, int tensor_vol, double ALPHA)
		{
			int *tensor_indices = new int[rank];
			std::vector<int> cur_tensor_indices(rank, 0);
			std::vector<int> target_perm_vec(rank, 0);
			target_perm_vec.assign(transpose_permutation, transpose_permutation + rank);
			for (int index = 0; index < rank; ++index)
			{
				tensor_indices[index] = index + 1;
				cur_tensor_indices[index] = index + 1;
			}
			if (cur_tensor_indices == target_perm_vec)
			{
				return;
			}
			bool PRINT_flag = false;
			int reordering_method_type = method_selection(tensor_dim, cur_tensor_indices, transpose_permutation, rank);

			// reordering_method_type = PRE_ORDER_INDEX_REORDERING;
			// reordering_method_type = POST_ORDER_INDEX_REORDERING;
			if (PRINT_flag == true)
			{
				if (reordering_method_type == PRE_ORDER_INDEX_REORDERING)
				{	
					printf("execute pre order sorting");
				}
				else if (reordering_method_type == POST_ORDER_INDEX_REORDERING)
				{
					printf("execute post order sorting");
				}
			}

			for (int round = 0; round < rank; ++round)
			{
				// if(PRINT_flag == true) { printf("round %d\n", round + 1);}
				int pre_low_order_dim[4];
				int pre_low_order_perm[4];
				int post_low_order_dim[4];
				int post_low_order_perm[4];
				int *pre_round_trans_perm = new int[rank];
				int *post_round_trans_perm = new int[rank];
				int pre_perm_int = 0, post_perm_int = 0;

				std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

				int pre_low_order_rank = pre_order_trans_perm_generation(cur_tensor_indices, tensor_dim, transpose_permutation, pre_low_order_dim, pre_low_order_perm, pre_round_trans_perm, rank);
				int post_low_order_rank = post_order_trans_perm_generation(cur_tensor_indices, tensor_dim, transpose_permutation, post_low_order_dim, post_low_order_perm, post_round_trans_perm, rank);

				for (int i = 0; i < pre_low_order_rank; ++i)
				{
					pre_perm_int += pow(10, i) * pre_low_order_perm[pre_low_order_rank - i - 1];
				}
				for (int i = 0; i < post_low_order_rank; ++i)
				{
					post_perm_int += pow(10, i) * post_low_order_perm[post_low_order_rank - i - 1];
				}

				if (PRINT_flag == true)
				{
					printf("Pre order sorting, %d transpose, catanzaro dim = ( ", pre_perm_int);
					print_arr(pre_low_order_dim, pre_low_order_rank);
					printf(")\nPost order sorting, %d transpose, catanzaro dim = ( ", post_perm_int);
					print_arr(post_low_order_dim, post_low_order_rank);
					printf(")\n");
				}
				// assert(verify_perm(pre_low_order_perm, pre_low_order_rank) == true);
				// assert(verify_perm(post_low_order_perm, post_low_order_rank) == true);

				if (pre_low_order_rank == rank + 1 || post_low_order_rank == rank + 1)
				{
					break;
				}

				if (reordering_method_type == PRE_ORDER_INDEX_REORDERING)
				{
					Low_order_transpose(data, pre_low_order_dim, pre_low_order_perm, source, pre_low_order_rank, round, tensor_vol, ALPHA);
					int *tmp_tensor_indices = new int[rank];
					for (int i = 0; i < rank; ++i)
					{
						tmp_tensor_indices[i] = cur_tensor_indices.at(pre_round_trans_perm[i] - 1);
					}
					std::copy(tmp_tensor_indices, tmp_tensor_indices + rank, tensor_indices);
					cur_tensor_indices.assign(tmp_tensor_indices, tmp_tensor_indices + rank);
					delete[] tmp_tensor_indices;
				}
				else if (reordering_method_type == POST_ORDER_INDEX_REORDERING)
				{
					Low_order_transpose(data, post_low_order_dim, post_low_order_perm, source, post_low_order_rank, round, tensor_vol, ALPHA);
					int *tmp_tensor_indices = new int[rank];
					for (int i = 0; i < rank; ++i)
					{
						tmp_tensor_indices[i] = cur_tensor_indices.at(post_round_trans_perm[i] - 1);
					}
					std::copy(tmp_tensor_indices, tmp_tensor_indices + rank, tensor_indices);
					cur_tensor_indices.assign(tmp_tensor_indices, tmp_tensor_indices + rank);
					delete[] tmp_tensor_indices;
				}

				delete[] pre_round_trans_perm;
				delete[] post_round_trans_perm;
				// assert(verify_perm_vec(cur_tensor_indices, rank) == true);

				std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
				double s_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
				if (PRINT_flag == true)
				{
					printf(", spend %f ms\n", s_time);
					// printf("cur dim perm =");
					// print_vec(cur_tensor_indices);
				}

				if (cur_tensor_indices == target_perm_vec)
				{
					break;
				}
			}
		}

		template void Series_Low_order_transpose(float *, int, int *, int *, int, int, double);
		template void Series_Low_order_transpose(double *, int, int *, int *, int, int, double);


		template <typename T> // permutation, 1~n
		void transpose(T *data, int source, void *dim, void *permutation, int rank, int num_block, double ALPHA)
		{

			int *tensor_dim = new int[rank];
			int *transpose_permutation = new int[rank];
			size_t *stride = new size_t[rank + 1];
			init_dims(dim, tensor_dim, rank);
			init_perm(permutation, transpose_permutation, rank);
			init_stride(stride, tensor_dim, rank);
			// assert(verify_perm(transpose_permutation, rank) == true);
			int vol = stride[rank], LARGE_SIZE = WARP_SIZE * (1 / ALPHA);
			int padding_dim_pos = -1, decompose_dim_pos = -1, padding_dim_size;
			int origin_vol = vol;
			int max_dim_pos = find_proper_max_dim_pos(tensor_dim, rank);
			int origin_max_dim_size = tensor_dim[max_dim_pos];
			// Always consider the largest dimension to optimize
			if (origin_max_dim_size > LARGE_SIZE && num_block > 1)
			{
				// Largest dimension > threshold
				// Hence, divide into multiple sub-tensor along the max dim
				decompose_dim_pos = max_dim_pos;
				if (origin_max_dim_size % num_block != 0)
				{
					// If cannot be evenly divisible, then do padding
					padding_dim_pos = max_dim_pos;
					padding_dim_size = (tensor_dim[padding_dim_pos] / num_block + 1) * num_block;
				}
			}
			else
			{
				num_block = 1;
			}

			if (padding_dim_pos != -1)
			{
				// update tensor information to the form after padding
				tensor_dim[padding_dim_pos] = origin_max_dim_size;
				inplace::Nd_padding_pre_process(data, tensor_dim, stride, rank, padding_dim_pos, num_block);
				tensor_dim[padding_dim_pos] = padding_dim_size;
				vol = vol / origin_max_dim_size * padding_dim_size;
			}
			// The core section for transpose
			if (decompose_dim_pos != -1)
			{
				inplace::NdPartition(data, tensor_dim, rank, num_block, decompose_dim_pos, ALPHA);

				tensor_dim[decompose_dim_pos] /= num_block;
				int sub_tensor_vol = vol / num_block;
				for (int i = 0; i < num_block; ++i)
				{
					Series_Low_order_transpose(data, source + i * sub_tensor_vol, tensor_dim, transpose_permutation, rank, vol, ALPHA);
				}
				tensor_dim[decompose_dim_pos] *= num_block;

				inplace::NdJoin(data, tensor_dim, transpose_permutation, rank, num_block, decompose_dim_pos, ALPHA);
			}
			else
			{
				Series_Low_order_transpose(data, source, tensor_dim, transpose_permutation, rank, vol, ALPHA);
			}

			if (padding_dim_pos != -1)
			{
				// Recover the padding tensor to its origin size
				vol = origin_vol;
				tensor_dim[padding_dim_pos] = origin_max_dim_size;
				inplace::Nd_padding_post_process(data, tensor_dim, transpose_permutation, rank, padding_dim_pos, num_block);
			}

			delete[] tensor_dim;
			delete[] transpose_permutation;
			delete[] stride;
		}

		template void transpose(float *, int, void *, void *, int, int, double);
		template void transpose(double *, int, void *, void *, int, int, double);
	}
}
