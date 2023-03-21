#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "4dtranspose.h"
#include "Nd_transpose.h"
#include "util.h"
#include "equations.h"
#include "row_op.h"
#include "col_op.h"
#include "debug.h"
#include "gcd.h"

#define _1324_col_liearization_low_bound 96
#define _1324_row_liearization_low_bound 4

namespace inplace {

namespace _4d {
	void init_dims(void* dim, int& d1, int& d2, int& d3, int& d4) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
		d4 = int_dim[3];
	}

	template<typename T>
	size_t transpose(T* data, int source, void* dim, int type) {
		int d1, d2, d3, d4;
		
		init_dims(dim, d1, d2, d3, d4);
		switch (type) {
			case 1324: // fix d1 = 1
				return _1324::transpose(data, source, d1, d2, d3, d4);
			/*case 1432:
				if(d1 >= _1324_col_liearization_low_bound) { return _1432::transpose(data, source, d1, d2, d3, d4);}
				else { return _Nd::transpose(data, source, dim, type);}

			case 3214:
				if(d4 >= _1324_row_liearization_low_bound) { return _3214::transpose(data, source, d1, d2, d3, d4);}
				else { return _Nd::transpose(data, source, dim, type);}*/
			default:
				printf("Invalid rank 4 permutation\n");
				return 0;
				//return _Nd::transpose(data, source, dim, type);
		}
	}
	template size_t transpose(float*, int, void*, int);
	template size_t transpose(double*, int, void*, int);

	// fix d1 = 1

	namespace _1324 { // 213 block transpose, bl = d1
		
		template<typename T>
		size_t c2r(T* data, int source, int d1, int d2, int d3, int d4) {
			//PRINT("Doing C2R transpose\n");
			
			int c, t, k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d3 / c;
			
			size_t tmp = 0;
			if (c > 1) {
				tmp = std::max(tmp, col_op(_2d::c2r::rotate(d2, b), data, source, d1, d2, d3, d4));
			}
			tmp = std::max(tmp, row_gather_op(_2d::c2r::row_shuffle(d2, d3, c, k), data, source, d1, d2, d3, d4));
			tmp = std::max(tmp, col_op(_2d::c2r::col_shuffle(d2, d3, c), data, source, d1, d2, d3, d4));
			
			return tmp;
		}
		
		template<typename T>
		size_t r2c(T* data, int source, int d1, int d2, int d3, int d4) {
			//PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d3, d2, c, t);
			if (c > 1) {
				extended_gcd(d3/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d3 / c;
			
			int k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}
			
			size_t tmp = 0;
			tmp = std::max(tmp, col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, source, d1, d2, d3, d4));
			tmp = std::max(tmp, row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d3, c, k), data, source, d1, d2, d3, d4));
			if (c > 1) {
				tmp = std::max(tmp, col_op(_2d::r2c::rotate(d2, b), data, source, d1, d2, d3, d4));
			}
			return tmp;
		}
	
		
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4) {
			int d1d2d3 = d1 * d2 * d3, d2d3d4 = d2 * d3 * d4;
			size_t tmp = 0;
			if(d1 >= _1324_col_liearization_low_bound) {
				//PRINT("4d 1324 row permute transpose\n");
				return _2d::col_op(inplace::_4d::_1324::row_permute(d3, d2), data, source, d1, d2d3d4);
			}
			else if(d4 >= _1324_row_liearization_low_bound) {
				//PRINT("4d 1324 row shuffle transpose\n");
				return _2d::row_gather_op(inplace::_4d::_1324::row_shuffle(d2, d3, d1), data, source, d1d2d3, d4);
			}	
			else {
				//PRINT("\n4d 1324 combination method\n");
				if(d2 >= d3) {
					//if(source == 0){ PRINT("\n4d 1324 d2 >= d3\n");}
					tmp = std::max(tmp, _3d::_213::transpose(data, source, d1 * d2, d3, d4));
					tmp = std::max(tmp, _3d::_213::transpose(data, source, d3, d1, d2 * d4));
				}
				else {
					//if(source == 0){ PRINT("\n4d 1324 d2 < d3\n");}
					tmp = std::max(tmp, _2d::row_gather_op(_3d::_213::row_shuffle(d1, d2), data, source, d1 * d2, d3 * d4));
					tmp = std::max(tmp, _3d::_231::transpose(data, source, d2, d1, d3 * d4));
					tmp = std::max(tmp, _2d::col_op(_3d::_132::row_permute(d2, d4), data, source, d1 * d3, d4 * d2));
				}
				return tmp;	
			}
			return tmp;
		}
		
		template size_t transpose(float*, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int);
	}

	namespace _1432 { 
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4) {
			int d2d3d4 = d2 * d3 * d4;
			
			//if(source == 0){ PRINT("\n4d 1432 col linearization\n");}
			return _2d::col_op(_4d::_1432::row_permute(d4, d3, d2), data, source, d1, d2d3d4);
		}
		template size_t transpose(float*, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int);
	}

	namespace _3214 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3, int d4) {
			int d1d2d3 = d1 * d2 * d3, d2d3d4 = d2 * d3 * d4;		
			//if(source == 0){ PRINT("\n4d 3214 row linearization\n");}
			return _2d::row_gather_op(_4d::_3214::row_shuffle(d2, d3, d1), data, source, d1d2d3, d4);
		}
		template size_t transpose(float*, int, int, int, int, int);
		template size_t transpose(double*, int, int, int, int, int);
	}

}
}