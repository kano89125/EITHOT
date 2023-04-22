#include <cstdio>
#include <math.h>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "5dtranspose.h"
#include "Nd_transpose.h"

namespace inplace {
	void init_dims(void* dim, int* int_dim, int rank) {
		int *cast_dim = reinterpret_cast<int*>(dim);
		for(int i = 0; i < rank; ++i) { int_dim[i] = cast_dim[i];}
	}


	/*void init_dims(void* dim, int* int_dim, int rank) {
		int *cast_dim = reinterpret_cast<int*>(dim);
		for(int i = 0; i < rank; ++i) { int_dim[i] = cast_dim[i];}
	}*/

	void transpose(void* data, int source, int rank, void* dim, void* permutation, size_t sizeofType, int num_block, double ALPHA) {
		
		switch (rank) {
			
			default:
				if (sizeofType == 4) _Nd::transpose(reinterpret_cast<float*>(data), source, dim, permutation, rank, num_block, ALPHA);
				else _Nd::transpose(reinterpret_cast<double*>(data), source, dim, permutation, rank, num_block, ALPHA);
			return;
		}
	}	
}