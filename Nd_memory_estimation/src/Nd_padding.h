#include <stdio.h>
#include <stdlib.h>

namespace inplace {

template <typename T>
size_t Dimension_Pre_Padding(T *data, int *dim, size_t* stride, int rank, int padding_dim, int NUM_TENSOR_BLOCK);

template <typename T>
size_t Dimension_Post_Padding(T *data, int *dim, int *permutation, int rank, int old_padding_dim, int NUM_TENSOR_BLOCK);

}