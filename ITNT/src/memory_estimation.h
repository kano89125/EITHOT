namespace inplace {

	template <typename T>
	void memory_reducing_pre_process();
	
	template <typename T>
	void memory_reducing_post_process();

	size_t Partition_memory_estimation(int *dim, int rank, int num_block, int max_dim_pos, int type_size);
	size_t Join_memory_estimation(int *dim, int *permutation, int rank, int num_block, int ori_max_dim_pos, int type_size);
	size_t Dimension_Padding_memory_estimation(int *dim, int vol, int nb, int max_dim_pos, int type_size);
	void Extra_memory_estimation(int* int_dim, int *perm, size_t ori_transpose_tmp, size_t data_size, int rank, int tensor_vol, int nb, int max_dim_pos, int type_size);
	int memory_estimation(int *int_dim, int *perm, int rank, int max_dim_pos, int tmp_vol, int tensor_vol, double LARGE_RATIO, int type_size);
}