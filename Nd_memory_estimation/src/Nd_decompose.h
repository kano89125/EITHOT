
namespace inplace {

template <typename T>
size_t Linearization_NDPartition(T *d_data, int decompose_stride, int non_decompose_stride, int rank, int num_block);

template <typename T>
size_t Linearization_NDJoin(T *d_data, int decompose_stride, int non_decompose_stride, int rank, int num_block);


template <typename T>
size_t NdPartition(T *d_data, int *dim, int rank, int num_block, int decompose_dim);

template <typename T>
size_t NdJoin(T *d_data, int *dim, int *permutation, int rank, int num_block, int ori_decompose_dim);
}