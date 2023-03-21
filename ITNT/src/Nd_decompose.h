

namespace inplace {

template <typename T>
void Linearization_NDPartition(T *d_data, int decompose_stride, int non_decompose_stride, int rank, int num_block, double LARGE_RATIO);

template <typename T>
void Linearization_NDJoin(T *d_data, int decompose_stride, int non_decompose_stride, int rank, int num_block, double LARGE_RATIO);


template <typename T>
void NdPartition(T *d_data, int *dim, int rank, int num_block, int decompose_dim, double LARGE_RATIO);

template <typename T>
void NdJoin(T *d_data, int *dim, int *permutation, int rank, int num_block, int ori_decompose_dim, double LARGE_RATIO);
}