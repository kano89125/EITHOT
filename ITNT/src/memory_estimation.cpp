#include "util.h"
#include "math.h"
#include <cstdio>
#include <cassert>
#include <algorithm>

#define max_nb 64

namespace inplace {

	size_t Partition_memory_estimation(int *dim, int rank, int num_block, int max_dim_pos, int type_size)
	{
		if(num_block == 1){ return 0;}
		int decompose_dim = max_dim_pos;
		int decompose_stride = 1, non_decompose_stride = 1;
		for(int i = 0; i <= decompose_dim; ++i) { decompose_stride *= dim[i];}
		for(int i = decompose_dim + 1; i < rank; ++i) {non_decompose_stride *= dim[i];}
		int n_threads;
		if(msb(decompose_stride / num_block) <= 18) { n_threads = 32;} // dim[0] -> decomspose_stide * nb
		else { n_threads = 256;}
		size_t tmp = n_threads * num_block * non_decompose_stride * type_size;
		return tmp;
	}
	
	size_t Join_memory_estimation(int *dim, int *permutation, int rank, int num_block, int ori_max_dim_pos, int type_size) {
		if(num_block == 1){ return 0;}
		int *perm_dim = new int[rank];
		int decompose_permutation_dim = -1; 
		for(int i = 0; i < rank; ++i) { 
			perm_dim[i] = dim[permutation[i]];
			if(permutation[i] == ori_max_dim_pos) { decompose_permutation_dim = i;}
		}
		int decompose_stride = 1, non_decompose_stride = 1;
		for(int i = 0; i <= decompose_permutation_dim; ++i) { decompose_stride *= perm_dim[i];}
		for(int i = decompose_permutation_dim + 1; i < rank; ++i) { non_decompose_stride *= perm_dim[i];}
		delete[] perm_dim;
		int n_threads;
		if(msb(decompose_stride / num_block) <= 18) { n_threads = 32;}
		else { n_threads = 256;}
		size_t tmp = n_threads * num_block * non_decompose_stride * type_size;
		return tmp;
	}
	
	size_t Dimension_Padding_memory_estimation(int *dim, int vol, int nb, int max_dim_pos, int type_size) {
		if(nb == 1){ return 0;}
		int padding_dim_size = (dim[max_dim_pos] / nb + 1) * nb;
		size_t tmp = vol / dim[max_dim_pos] * (padding_dim_size - dim[max_dim_pos]) * type_size;
		return tmp;
	}

	void Extra_memory_estimation(int* int_dim, int *perm, size_t ori_transpose_tmp, size_t data_size, int rank, int tensor_vol, int nb, int max_dim_pos, int type_size) {
		size_t transpose_tmp = ori_transpose_tmp / nb;
		size_t padding_tmp = Dimension_Padding_memory_estimation(int_dim, tensor_vol, nb, max_dim_pos, type_size);
		size_t partition_tmp = Partition_memory_estimation(int_dim, rank, nb, max_dim_pos, type_size);
		size_t join_tmp = Join_memory_estimation(int_dim, perm, rank, nb, max_dim_pos, type_size);
		size_t final_tmp = padding_tmp + std::max(transpose_tmp, std::max(partition_tmp, join_tmp));
		//size_t final_tmp = padding_tmp + std::max(partition_tmp, join_tmp);
		double mem_ratio = (double)final_tmp / (double)data_size;
		printf("f,d, extra memory ratio %.2f, nb = %d\n", 100 * mem_ratio, nb);
	}

	int memory_estimation(int *int_dim, int *perm, int rank, int max_dim_pos, int tmp_vol, int tensor_vol, double ALPHA, int type_size) { 
		size_t data_size = type_size;
		for(int i = 0; i < rank; i++) { data_size *= int_dim[i];}
		int percent = (double)tmp_vol / (double)(tensor_vol) / ALPHA;
		int nb = pow(2, ceil(log2(percent)));
		if(nb == 0){ ++nb;}

		size_t transpose_tmp = (tmp_vol / nb) * type_size;
		size_t padding_tmp = Dimension_Padding_memory_estimation(int_dim, tensor_vol, nb, max_dim_pos, type_size);
		size_t partition_tmp = Partition_memory_estimation(int_dim, rank, nb, max_dim_pos, type_size);
		size_t join_tmp = Join_memory_estimation(int_dim, perm, rank, nb, max_dim_pos, type_size);
		size_t final_tmp = padding_tmp + std::max(transpose_tmp, std::max(partition_tmp, join_tmp));
		double mem_ratio = (double)final_tmp / (double)data_size;
		//printf("nb = %d, trans, pad, final = %ld %ld %ld\n", nb, transpose_tmp, padding_tmp, final_tmp);
		if(mem_ratio > ALPHA) {
			transpose_tmp *= nb;
			nb = 1;
			padding_tmp = Dimension_Padding_memory_estimation(int_dim, tensor_vol, nb, max_dim_pos, type_size);
			partition_tmp = Partition_memory_estimation(int_dim, rank, nb, max_dim_pos, type_size);
			join_tmp = Join_memory_estimation(int_dim, perm, rank, nb, max_dim_pos, type_size);
			final_tmp = padding_tmp + std::max(transpose_tmp, std::max(partition_tmp, join_tmp));

			std::vector<double> tmp_ratio_list(log2(max_nb) + 1, 1.0);
			int count = 0;
			tmp_ratio_list.at(count) = (double)final_tmp / (double)data_size;
			count++;
			
			while(nb < max_nb) {
				//printf("mem ratio = %.2f where nb = %d\n", tmp_ratio_list.at(count-1), nb);
				//printf("nb = %d, trans, pad, part, join, final = %ld %ld %ld %ld %ld\n", nb, transpose_tmp, padding_tmp, partition_tmp, join_tmp, final_tmp);
				nb *= 2;
				padding_tmp = Dimension_Padding_memory_estimation(int_dim, tensor_vol, nb, max_dim_pos, type_size);
				transpose_tmp /= 2;
				partition_tmp = Partition_memory_estimation(int_dim, rank, nb, max_dim_pos, type_size); 
				join_tmp = Join_memory_estimation(int_dim, perm, rank, nb, max_dim_pos, type_size);
				final_tmp = padding_tmp + std::max(transpose_tmp, std::max(partition_tmp, join_tmp));
				tmp_ratio_list.at(count) = (double)final_tmp / (double)data_size;
				//printf("mem_ratio = %.5f\n", tmp_ratio_list.at(count));
				count++;
			}

			int min_index = -1;
			bool less_flag = false;
			for(int i = 0; i < tmp_ratio_list.size(); ++i) {
				if(tmp_ratio_list.at(i) < ALPHA) { 
					min_index = i; 
					less_flag = true;
					break;
				}
			}
			if(less_flag == false) { min_index = std::distance(tmp_ratio_list.begin(), std::min_element(tmp_ratio_list.begin(), tmp_ratio_list.end()));}
			assert(min_index != -1);
				
			nb = pow(2, min_index);
			mem_ratio = tmp_ratio_list.at(min_index);
			//print_vec(tmp_ratio_list);
			printf("revised extra memory percent %.2f, nb = %d\n", 100 * mem_ratio, nb);
			ALPHA = std::max(mem_ratio, ALPHA);	
		}

		return nb;
	}
}