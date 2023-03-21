#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "5dtranspose.h"
#include "Nd_transpose.h"
#include "util.h"
#include "equations.h"
#include "row_op.h"
#include "col_op.h"
#include "debug.h"
#include "gcd.h"

#define _5d_col_liearization_low_bound 96
#define _5d_row_liearization_low_bound 4

namespace inplace {

namespace _5d {
	void init_dims(void* dim, int& d1, int& d2, int& d3, int& d4, int &d5) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
		d4 = int_dim[3];
		d5 = int_dim[4];
	}

	template<typename T>
	size_t transpose(T* data, int source, void* dim, int type) {
		int d1, d2, d3, d4, d5;
		init_dims(dim, d1, d2, d3, d4, d5);
		//printf("5d transpose.\n");
		switch (type) {
			default:
				printf("Invalid rank 5 permutation\n");
				return 0;
				//return _Nd::transpose(data, source, dim, type);
		}
	}
	template size_t transpose(float*, int, void*, int);
	template size_t transpose(double*, int, void*, int);

	namespace _14325 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5) {
			int d1d2d3d4 = d1 * d2 * d3 * d4, d2d3d4d5 = d2 * d3 * d4 * d5;
			if(d1 >= _5d_col_liearization_low_bound){
				//if(source == 0){ PRINT("\n5d 14325 col linearization\n");}
				return _2d::col_op(_5d::_14325::row_permute(d5, d4, d3, d2), data, source, d1, d2d3d4d5);
			}
			else if(d5 >= _5d_row_liearization_low_bound)
			{
				//if(source == 0){ PRINT("\n5d 14325 row linearization\n");}
				return _2d::row_gather_op(_5d::_14325::row_shuffle(d4, d2, d3, d1), data, source, d1d2d3d4, d5);
			}
			return 0;
		}
		template size_t transpose(float*, int, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int, int);
	}

	namespace _15432 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5) {
			int d2d3d4d5 = d2 * d3 * d4 * d5;
			//if(source == 0){ PRINT("\n5d 15432 col linearization\n");}
			return _2d::col_op(_5d::_15432::row_permute(d5, d4, d3, d2), data, source, d1, d2d3d4d5);
		}
		template size_t transpose(float*, int, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int, int);
	}
	namespace _43215 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4, int d5) {
			int d1d2d3d4 = d1 * d2 * d3 * d4;
			//if(source == 0){ PRINT("\n5d 43215 row linearization\n");}
			return _2d::row_gather_op(_5d::_43215::row_shuffle(d4, d2, d3, d1), data, source, d1d2d3d4, d5);
		}
		template size_t transpose(float*, int, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int, int);
	}
}
}