#include <cstdio>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "5dtranspose.h"
#include "Nd_transpose.h"

namespace inplace {

size_t transpose(void* data, int source, int rank, void* dim, void* permutation, size_t sizeofType) {

	switch (rank) {
		default:
			if (sizeofType == 4) return _Nd::transpose(reinterpret_cast<float*>(data), source, dim, permutation, rank);
			else return _Nd::transpose(reinterpret_cast<double*>(data), source, dim, permutation, rank);
	}

}

}