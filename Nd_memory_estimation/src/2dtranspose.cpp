#include <cuda_runtime.h>
#include <algorithm>
#include "2dtranspose.h"
#include "debug.h"
#include "util.h"
#include "cudacheck.h"
#include "gcd.h"
#include "equations.h"
#include "col_op.h"
#include "row_op.h"

namespace inplace {

namespace _2d {
	void init_dims(void* dim, int& d1, int& d2) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
	}

	template<typename T>
	size_t transpose(T* data, int source, void* dim) {
		int d1, d2;
		init_dims(dim, d1, d2);
		
		//PRINT("(d1, d2) = (%d, %d)\n", d1, d2);
		if (d1 >= d2) {
			return c2r::transpose(data, source, d1, d2);
		}
		else {
			return r2c::transpose(data, source, d2, d1);
		}
	}
	template size_t transpose(float*, int, void*);
	template size_t transpose(double*, int, void*);

	namespace c2r {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2) {
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
				tmp = std::max(tmp, col_op(c2r::rotate(d2, b), data, source, d1, d2));
			}
			tmp = std::max(tmp, row_gather_op(c2r::row_shuffle(d2, d1, c, k), data, source, d1, d2));
			tmp = std::max(tmp, col_op(c2r::col_shuffle(d2, d1, c), data, source, d1, d2));
			
			return tmp;
		}
		template size_t transpose(float*, int, int, int);
		template size_t transpose(double*, int, int, int);
	}

	namespace r2c {
		template<typename T>
		size_t transpose(T* data, int source, int d1, int d2) {
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
			tmp = std::max(tmp, col_op(r2c::col_shuffle(a, c, d2, q), data, source, d1, d2));
			tmp = std::max(tmp, row_scatter_op(r2c::row_scatter_shuffle(d2, d1, c, k), data, source, d1, d2));
			if (c > 1) {
				tmp = std::max(tmp, col_op(r2c::rotate(d2, b), data, source, d1, d2));
			}
			
			return tmp;
		}
		template size_t transpose(float*, int, int, int);
		template size_t transpose(double*, int, int, int);
	}
}

}