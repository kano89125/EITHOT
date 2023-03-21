#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "Nd_transpose.h"
#include "util.h"
#include "equations.h"
#include "row_op.h"
#include "col_op.h"
#include "debug.h"
#include "gcd.h"

#define _132_col_liearization_low_bound 48
#define _132_catanzaro_low_bound 12
#define _213_row_liearization_low_bound 8

namespace inplace {

namespace _3d {

	void init_dims(void* dim, int& d1, int& d2, int& d3) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
	}

	template<typename T>
	size_t transpose(T* data, int source, void* dim, int type) {
		int d1, d2, d3;
		init_dims(dim, d1, d2, d3);
		switch (type) {
			case 213:
				return _213::transpose(data, source, d1, d2, d3);
			case 132:
				return _132::transpose(data, source, d1, d2, d3);
			default:
				printf("Invalid permutation\n");
				return 0;
		}
	}
	template size_t transpose(float*, int, void*, int);
	template size_t transpose(double*, int, void*, int);

	namespace _312 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3) {
			int dim[2];
			dim[0] = d1 * d2;
			dim[1] = d3;
			return _2d::transpose(data, source, dim);
		}
		template size_t transpose(float*, int, int, int, int);
		template size_t transpose(double*, int, int, int, int);

	}

	namespace _231 {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3) {
			int dim[2];
			dim[0] = d1;
			dim[1] = d2 * d3;
			return _2d::transpose(data, source, dim);
		}
		template size_t transpose(float*, int, int, int, int);
		template size_t transpose(double*, int, int, int, int);

	}
	
	namespace _213 {
		template<typename T>
		size_t c2r(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing C2R transpose\n");
			
			int c, t, k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d1 / c;
			
			size_t tmp = 0;
			if (c > 1) {
				tmp = std::max(tmp, col_op(_2d::c2r::rotate(d2, b), data, source, d1, d2, d3));
			}
			tmp = std::max(tmp, row_gather_op(_2d::c2r::row_shuffle(d2, d1, c, k), data, source, d1, d2, d3));
			tmp = std::max(tmp, col_op(_2d::c2r::col_shuffle(d2, d1, c), data, source, d1, d2, d3));
			return tmp;
		}
		
		template<typename T>
		size_t r2c(T* data, int source, int d1, int d2, int d3) {
			//PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d1, d2, c, t);
			if (c > 1) {
				extended_gcd(d1/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d1 / c;
			
			int k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}
			
			size_t tmp = 0;
			tmp = std::max(tmp, col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, source, d1, d2, d3) );
			tmp = std::max(tmp, row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d1, c, k), data, source, d1, d2, d3));
			if (c > 1) {
				tmp = std::max(tmp, col_op(_2d::r2c::rotate(d2, b), data, source, d1, d2, d3));
			}
			return tmp;
		}
		
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3) {
			size_t data_size = sizeof(T) * d1 * d2 * d3;
			if(d3 >= _213_row_liearization_low_bound) {
			//if (d1 * d2 / ((double)d1 * d2 * d3) < 0.01) {
				//PRINT("213 linearization\n");
				return _2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3);
			}
			else {
				//PRINT("213 catanzaro transpose\n");
				if (d1 > d2) return c2r(data, source, d1, d2, d3);
				else return r2c(data, source, d2, d1, d3);
			}
		}
		template size_t transpose(float*, int, int, int, int);
		template size_t transpose(double*, int, int, int, int);
	
	}
	
	namespace _132 {
		template<typename T>
		size_t c2r(T* data, int source, int d1, int d2, int d3) {
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
				tmp = std::max(tmp, col_op(_2d::c2r::rotate(d2, b), data, source, d1, d2, d3));
			}
			tmp = std::max(tmp, row_gather_op(_2d::c2r::row_shuffle(d2, d3, c, k), data, source, d1, d2, d3));
			tmp = std::max(tmp, col_op(_2d::c2r::col_shuffle(d2, d3, c), data, source, d1, d2, d3));
			
			return tmp;
		}
		
		template<typename T>
		size_t r2c(T* data, int source, int d1, int d2, int d3) {
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
			tmp = std::max(tmp, col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, source, d1, d2, d3));
			tmp = std::max(tmp, row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d3, c, k), data, source, d1, d2, d3));
			if (c > 1) {
				tmp = std::max(tmp, col_op(_2d::r2c::rotate(d2, b), data, source, d1, d2, d3));
			}
			return tmp;
		}
	
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2, int d3) {
			
			size_t tmp = 0;
			if (d1 >= _132_col_liearization_low_bound) {
				//PRINT("132 linearization\n");
				tmp = std::max(tmp, _2d::col_op(_132::row_permute(d3, d2), data, source, d1, d2 * d3));
			}
			else if (d1 > _132_catanzaro_low_bound) {
				//PRINT("132 catanzaro transpose\n");
				if (d2 > d3) tmp = std::max(tmp, c2r(data, source, d1, d3, d2));
				else tmp = std::max(tmp, r2c(data, source, d1, d2, d3));
			}
			else {
				//PRINT("132 combination method\n");
				if (d2 >= d3) {
					tmp = std::max(tmp, _312::transpose(data, source, d1, d2, d3));
					tmp = std::max(tmp, _2d::row_gather_op(_213::row_shuffle(d3, d1), data, source, d3 * d1, d2));
				}
				else {
					tmp = std::max(tmp, _2d::row_gather_op(_213::row_shuffle(d1, d2), data, source, d1 * d2, d3));
					tmp = std::max(tmp, _231::transpose(data, source, d2, d1, d3));
				}
			}
			return tmp;
		}
		template size_t transpose(float*, int, int, int, int);
		template size_t transpose(double*, int, int, int, int);
	}

}

}