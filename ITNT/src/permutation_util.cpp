#include <cstdio>
#include <vector>
#include "math.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <bits/stdc++.h>
#include "util.h"

//#define MAXIMUM_REVERSE_NUMBER 4

namespace inplace { 
	namespace _Nd {

	void rerank_permutation(int *permutation, int perm_rank) {
		int *rank_permutation = new int[perm_rank];
		std::copy(permutation, permutation + perm_rank, rank_permutation);
   		std::sort(rank_permutation, rank_permutation + perm_rank);

		std::map<int, int> ranks;
		int rank = 1;

		for(int index = 0; index < perm_rank; ++index) {
			int element = rank_permutation[index];
			if (ranks[element] == 0) {
				ranks[element] = rank;
				rank++;
			}
		}

   		for(int index = 0; index < perm_rank; ++index) {
			int element = permutation[index];
			permutation[index] = ranks[permutation[index]];
		}
		
		delete [] rank_permutation;
	}

	int merge_seq_perm_in_dim(int* unmerged_dim, int* merged_dim, int* permutation, int unmerged_rank) {
		int origin_count = 0, merged_count = 0;
		std::vector<int> origin_dim_vec(unmerged_dim, unmerged_dim + unmerged_rank);
		std::vector<int> merged_dim_acc;
		merged_dim_acc.assign(unmerged_rank, 0);

		// find the first one continue integer & accunulative the number of continue integer on its index
		while (origin_count < unmerged_rank) {
			if(origin_count < unmerged_rank - 1) {
				while(permutation[origin_count + 1] == permutation[origin_count] + 1) {
					int init_seq_dim_index = origin_count;
					while (init_seq_dim_index > 0) {
						if(permutation[init_seq_dim_index] != permutation[init_seq_dim_index - 1] + 1) { break;}
						--init_seq_dim_index;
					}
					int merged_dim_index = permutation[init_seq_dim_index] - 1;
					++merged_dim_acc[merged_dim_index];
					++origin_count;
					if(origin_count + 1 >= unmerged_rank) { break;}
				}
			}
			++origin_count;
			++merged_count;
		}

		//  multiply the dim whos corresponding continue integer in the permutation
		int merged_index = 0, merged_rank = 0;
		for(; merged_index < unmerged_rank; ++merged_index) {
			if(merged_dim_acc[merged_index] == 0) { merged_dim[merged_rank] = unmerged_dim[merged_index];}
			else {
				merged_dim[merged_rank] = unmerged_dim[merged_index];
				for(int count = 1; count <= merged_dim_acc[merged_index]; ++count) { merged_dim[merged_rank] *= unmerged_dim[merged_index + count];} 
				merged_index += merged_dim_acc[merged_index];
			}
			++merged_rank;
		}

		for(merged_index = merged_rank; merged_index < unmerged_rank; ++merged_index) { merged_dim[merged_index] = -1;}
		return merged_rank;
	}

	void merge_seq_perm(int *unmerged_perm, int *merged_perm, int unmerged_rank) {
		int origin_count = 0, merged_count = 0;
		// fliter the permutation seq to record all the discontinuous subseq
		while(origin_count < unmerged_rank) {
			merged_perm[merged_count] = unmerged_perm[origin_count];
			if(origin_count < unmerged_rank - 1) {
				while(unmerged_perm[origin_count + 1] == unmerged_perm[origin_count] + 1) {
					++origin_count;
					if(origin_count + 1 >= unmerged_rank) { break;}
				}
			}
			++origin_count;
			++merged_count;
		}
		int merged_rank = merged_count;
		rerank_permutation(merged_perm, merged_rank);
	}

	int find_unmatched_index_pre_order(std::vector<int> cur_tensor_indices, int *transpose_permutation, int perm_rank) {
		// find the first different from the begin
		// the same order => not found will use last index
		int unmatched_index = perm_rank;
		for(int i = 0; i < perm_rank; ++i) {
			if(cur_tensor_indices[i] != transpose_permutation[i]) {
				unmatched_index = i;
				break;
			}
		}
		return unmatched_index;
	}

	int find_unmatched_index_post_order(std::vector<int> cur_tensor_indices, int *transpose_permutation, int perm_rank) {
		// find the first different from the end
		// the same order => not found will use 0 index
		int unmatched_index = 0;
		for(int i = perm_rank - 1; i >= 0; --i) {
			if(cur_tensor_indices[i] != transpose_permutation[i]) {
				unmatched_index = i;
				break;
			}
		}
		return unmatched_index;
	}

	void merge_matched_dim_pre_order(int *origin_dim, int *merged_dim, int init_unmatched_index, int rank) {
		// create a merge dimention, merge the serial sequence 12354 => 132 like
		if(init_unmatched_index != 0) {
			merged_dim[0] = 1;
			for(int i = 0; i < init_unmatched_index; ++i) { merged_dim[0] *= origin_dim[i];}
			std::copy(origin_dim + init_unmatched_index, origin_dim + rank, merged_dim + 1);
		} else { std::copy(origin_dim, origin_dim + rank, merged_dim);}
	}

	void merge_matched_dim_post_order(int *origin_dim, int *merged_dim, int init_unmatched_index, int rank) {
		// create a merge dimention, merge the serial sequence 21345 => 213 like
		int merged_rank;
		if(init_unmatched_index < rank - 2) {
			merged_rank = init_unmatched_index + 2;
			merged_dim[merged_rank - 1] = 1;
			for(int i = init_unmatched_index + 1; i < rank; ++i) { merged_dim[merged_rank - 1] *= origin_dim[i];}
			std::copy(origin_dim, origin_dim + init_unmatched_index + 1, merged_dim);
		} else { std::copy(origin_dim, origin_dim + rank, merged_dim);}
	}

	// target_transpose index > transpose_index
	void pre_round_trans_perm_generation(int *transpose_permutation, std::vector<int> cur_tensor_indices, int *round_trans_perm, int rank, int init_unmatched_index)
	{
		// seek the unmatch perm and its correspond in cur_tensor_indices, whether have a common sequence
		// we must know this has length 1, because contain itself
		// and we can get a permutation to swap this sequence to right place
		// ex: cur_tensor_indices: 568_291_374, transpose_permutation: 568_372_491
		// the we find diff at index=3, and get seq(37)
		// so we can create round_trans_perm = 000_780_000
		// and fill increasing number, the  round_trans_perm = 123_784_569
		// in this form, cur will become 568_372_914

		// transpose index in target permutation
		// target transpose index in current tensor indices
		int match_count = 0, seq_count = 0; // include itself
		int transpose_index = init_unmatched_index, target_transpose_perm = transpose_permutation[transpose_index];
		// find the target index in cur_tensor_indices
		std::vector<int>::iterator itr = std::find(cur_tensor_indices.begin(), cur_tensor_indices.end(), target_transpose_perm);
		int target_transpose_index = std::distance(cur_tensor_indices.begin(), itr);

		while(transpose_index + match_count < rank && target_transpose_index + match_count < rank) { 
			if(transpose_permutation[transpose_index + match_count] == cur_tensor_indices[target_transpose_index + match_count]){ ++match_count;}
			else { break;}
		}

		int seq_part_size = match_count;
		for(int i = 0; i < rank; ++i) { round_trans_perm[i] = 0;}
		int last_seq_index = transpose_index + seq_part_size;
		for(int index = transpose_index; index < last_seq_index; ++index) {
			round_trans_perm[index] = target_transpose_index + seq_count + 1;
			++seq_count;
		}

		int count = 1, index = 0;
		while(index < rank){
			while(round_trans_perm[index] != 0) {index++;}
			if(count < round_trans_perm[transpose_index] || count > round_trans_perm[last_seq_index - 1])
				round_trans_perm[index++] = count;
			count++;
		}
	}

	// transpose index > target_transpose_index
	void post_round_trans_perm_generation(int *transpose_permutation, std::vector<int> cur_tensor_indices, int *round_trans_perm, int rank, int init_unmatched_index) {
		// transpose index in target permutation
		// target transpose index in current tensor indices
		int match_count = 0, seq_count = 0, non_seq_count = 1; // include itself
		int transpose_index = init_unmatched_index, target_transpose_dim = transpose_permutation[transpose_index];
		std::vector<int>::iterator itr = std::find(cur_tensor_indices.begin(), cur_tensor_indices.end(), target_transpose_dim);
		int target_transpose_index = std::distance(cur_tensor_indices.begin(), itr);
 
		while(transpose_index - match_count >= 0 && target_transpose_index - match_count >= 0)  {
			if(transpose_permutation[transpose_index - match_count] == cur_tensor_indices[target_transpose_index - match_count]){ ++match_count;} 
			else { break;}
		}
		
		int seq_part_size = match_count;
		for(int i = 0; i < rank; ++i) { round_trans_perm[i] = 0;}
		int init_seq_index = transpose_index - seq_part_size + 1;
		for(int index = init_seq_index; index <= transpose_index; ++index) {
			round_trans_perm[index] = target_transpose_index + seq_count - seq_part_size + 2;
			++seq_count;
		}

		int count = 1, index = 0;
		while(index < rank){
			while(round_trans_perm[index] != 0) {index++;}
			if(count < round_trans_perm[init_seq_index] || count > round_trans_perm[transpose_index])
				round_trans_perm[index++] = count;
			count++;
		}
	}

	void merge_matched_perm_pre_order(int *round_trans_perm, int *trans_unmatched_perm, int init_unmatched_index, int rank) {
		// Modify perm to low dim, ex: 1235647 => 15647 => 13425
		if(init_unmatched_index != 0) {
			trans_unmatched_perm[0] = 1; // minmum of perm
			std::copy(round_trans_perm + init_unmatched_index, round_trans_perm + rank, trans_unmatched_perm + 1);
			int unmatched_rank = rank - init_unmatched_index + 1;
			int shift = rank - unmatched_rank;
			for(int i = 1; i < unmatched_rank; i++){
				trans_unmatched_perm[i] -= shift;
			}
		} else { std::copy(round_trans_perm, round_trans_perm + rank, trans_unmatched_perm);}
	}

	void merge_matched_perm_post_order(int *round_trans_perm, int *trans_unmatched_perm, int init_unmatched_index, int rank) {
		int unmatched_rank;	
		if(init_unmatched_index < rank - 2) {
			unmatched_rank = init_unmatched_index + 2;
			std::copy(round_trans_perm, round_trans_perm + init_unmatched_index + 1, trans_unmatched_perm);
			trans_unmatched_perm[unmatched_rank - 1] = unmatched_rank; // minmum of perm
		} else { 
			unmatched_rank = rank;
			std::copy(round_trans_perm, round_trans_perm + rank, trans_unmatched_perm);
		}
	}
		

	std::vector<int> tensor_indices_rotation_pre_order(std::vector<int> cur_tensor_indices, int *trans_unmatched_perm, int rank, int init_unmatched_index) {
		int unmatched_rank = rank - init_unmatched_index;
		std::vector<int> tmp_dim_perm;
		tmp_dim_perm.assign(rank, 0);
		if(init_unmatched_index <= 1) {
			for(int i = 0; i < rank; ++i) { 
				//tmp_dim_perm[init_unmatched_index + i] = cur_tensor_indices[trans_unmatched_perm[matched_rank + i] - 1];
				tmp_dim_perm[i] = cur_tensor_indices[trans_unmatched_perm[i] - 1];
			}
		} else {
			for(int i = 0; i < init_unmatched_index; ++i) {
				tmp_dim_perm[i] = cur_tensor_indices[i];
			}
			for(int i = 0; i < unmatched_rank; ++i) { 
				//tmp_dim_perm[init_unmatched_index + i] = cur_tensor_indices[trans_unmatched_perm[matched_rank + i] - 1];
				int target_index = trans_unmatched_perm[i + 1] + init_unmatched_index - 2;
				//printf("target index in cdp = %d\n", target_index);
				tmp_dim_perm[i + init_unmatched_index] = cur_tensor_indices[target_index];
			}
		}
		
		return tmp_dim_perm;
	}

	std::vector<int> tensor_indices_rotation_post_order(std::vector<int> cur_tensor_indices, int *trans_unmatched_perm, int rank, int init_unmatched_index) {
		int unmatched_rank;
		std::vector<int> tmp_dim_perm;
		tmp_dim_perm.assign(rank, 0);
		if(init_unmatched_index >= rank - 2) { 
			unmatched_rank = rank;
			for(int i = 0; i < rank; ++i) { tmp_dim_perm[i] = cur_tensor_indices[trans_unmatched_perm[i] - 1];}
		} else {
			unmatched_rank = init_unmatched_index + 2;
			for(int i = 0; i < rank; ++i) {
				if(i <= init_unmatched_index) { tmp_dim_perm[i] = cur_tensor_indices[trans_unmatched_perm[i] - 1];}
				else { tmp_dim_perm[i] = cur_tensor_indices[i];}
			}
		}

		return tmp_dim_perm;
	}

	int pre_order_trans_perm_generation(std::vector<int> cur_tensor_indices, int *tensor_dim, int *transpose_permutation,
	int *pre_low_order_dim, int *pre_low_order_perm, int *pre_round_trans_perm, int rank) {

		int init_unmatched_index = find_unmatched_index_pre_order(cur_tensor_indices, transpose_permutation, rank);
		if(init_unmatched_index == rank) { return rank + 1;}
		
		// a merge dim need 1 + (rank - init_unmatch_index)
		// (rank - init_unmatch_index) length  of unmatch
		int merged_rank;
		if(init_unmatched_index != 0) { merged_rank = 1 + rank - init_unmatched_index;}
		else { merged_rank = rank;}
		// tensor_dim_reordered: expected dim after permutation
		int *tensor_dim_reordered = new int[rank];
		for(int index = 0; index < rank; index++) { tensor_dim_reordered[index] = tensor_dim[cur_tensor_indices[index] - 1];}
			
		int *merged_dim = new int[merged_rank];
		// get the dimension after merged
		// ex: dim 5 5 5 5 5, do 12354 permutation
		// => 125 5 5, do 132 permutation 
		merge_matched_dim_pre_order(tensor_dim_reordered, merged_dim, init_unmatched_index, rank);
		// create more prefix's permutation		
		pre_round_trans_perm_generation(transpose_permutation, cur_tensor_indices, pre_round_trans_perm, rank, init_unmatched_index);
		
		int *trans_merged_perm = new int[merged_rank];
		merge_matched_perm_pre_order(pre_round_trans_perm, trans_merged_perm, init_unmatched_index, rank);
		// let the continuous seq in perm, combine to low order, also combine their dim
		// ex: 125 5 5 5 5, do 13425, can see that 34 term.
		// so, we get 125 5 25 5 and do 1324  
		int catanzaro_rank = merge_seq_perm_in_dim(merged_dim, pre_low_order_dim, trans_merged_perm, merged_rank);
			
		merge_seq_perm(trans_merged_perm, pre_low_order_perm, merged_rank);
			
		delete [] tensor_dim_reordered;
		delete [] merged_dim;
		delete [] trans_merged_perm;

		return catanzaro_rank;
	}
	
	int post_order_trans_perm_generation(std::vector<int> cur_tensor_indices, int *tensor_dim, int *transpose_permutation,
	 int *post_low_order_dim, int *post_low_order_perm, int *post_round_trans_perm, int rank) {
		int init_unmatched_index = find_unmatched_index_post_order(cur_tensor_indices, transpose_permutation, rank);

		if(init_unmatched_index == 0) { return rank + 1;}
		
		// a merge dim need (init_unmatch_index + 1) + 1 
		// (init_unmatch_index + 1) length  of unmatch
		int merged_rank;
		if(init_unmatched_index < rank - 2) { merged_rank = init_unmatched_index + 2;}
		else { merged_rank = rank;}
		// tensor_dim_reordered: expected dim after permutation
		int *tensor_dim_reordered = new int[rank];
		for(int index = 0; index < rank; index++) { tensor_dim_reordered[index] = tensor_dim[cur_tensor_indices[index] - 1];}
			
		int *merged_dim = new int[merged_rank];
		// get the dimension after merged
		// ex: dim 5 5 5 5 5, do 21345 permutation
		// => 5 5 125, do 213 permutation 
		merge_matched_dim_post_order(tensor_dim_reordered, merged_dim, init_unmatched_index, rank);
		// create more suffix's permutation	
		post_round_trans_perm_generation(transpose_permutation, cur_tensor_indices, post_round_trans_perm, rank, init_unmatched_index);
			
		int *trans_merged_perm = new int[merged_rank];
		merge_matched_perm_post_order(post_round_trans_perm, trans_merged_perm, init_unmatched_index, rank);

		int catanzaro_rank = merge_seq_perm_in_dim(merged_dim, post_low_order_dim, trans_merged_perm, merged_rank);

		merge_seq_perm(trans_merged_perm, post_low_order_perm, merged_rank);
			
		delete [] tensor_dim_reordered;
		delete [] merged_dim;
		delete [] trans_merged_perm;

		return catanzaro_rank;
	}

	
	};
}
