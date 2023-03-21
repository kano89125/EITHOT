#include <vector>
#include <algorithm>

namespace inplace {
	namespace _Nd {

	int pre_order_trans_perm_generation(std::vector<int> cur_tensor_indices, int *origin_dim, int *target_permutation,
	int *pre_catanzaro_dim, int *pre_catanzaro_perm, int *pre_round_transpose_perm, int rank);

	int post_order_trans_perm_generation(std::vector<int> cur_tensor_indices, int *origin_dim, int *target_permutation,
	 int *post_catanzaro_dim, int *post_catanzaro_perm, int *post_round_transpose_perm, int rank);

};
}