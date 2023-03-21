

namespace inplace {
template <typename T>
void Nd_padding_pre_process(T *data, int *dim, size_t* stride, int rank, int padding_dim, int NUM_TENSOR_BLOCK);

template <typename T>
void Nd_padding_post_process(T *data, int *dim, int *permutation, int rank, int old_padding_dim, int NUM_TENSOR_BLOCK);
}