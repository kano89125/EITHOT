#pragma once
#include "reduced_math.h"
#ifdef DEBUG
#include <string>
#endif

namespace inplace {


namespace _5d {

namespace _14325 {
	struct row_permute { // col linearization

		int d2, d3, d4, d5;
		
		__host__
		row_permute(int _d5, int _d4, int _d3, int _d2) : d5(_d5), d4(_d4), d3(_d3), d2(_d2) {}
		
		__host__ __device__
		int operator()(const int& iklm, const int& j) {
			int m = iklm / (d4 * d3 * d2);
			int l = iklm / (d4 * d3) % d2;
			int i = (iklm / d4) % d3;
			int k = iklm % d4;
			
			return l + i * d2 + k * d2 * d3 + m * d2 * d3 * d4;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row permute");
				return name;
			}
		#endif
	};

	struct row_shuffle { // row lineaization
		int int_d1, d2, int_d3;
		reduced_divisor d1, d3, d4;
		__host__
		row_shuffle(int _d4, int _d2, int _d3, int _d1) : 
		d4(_d4) , d2(_d2), d3(_d3), d1(_d1), int_d1(_d1), int_d3(_d3) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd1, jmodd1, jdivd1d4, jdivd1modd4, jdivd1d4d3, jdivd1d4modd3;
			d1.divmod(j, jdivd1, jmodd1);
			d4.divmod(jdivd1, jdivd1d4, jdivd1modd4);
			d3.divmod(jdivd1d4, jdivd1d4d3, jdivd1d4modd3);
			return jmodd1 + jdivd1d4d3 * int_d1 + jdivd1d4modd3 * int_d1 * d2 + jdivd1modd4 * int_d1 * d2 * int_d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};
}

namespace _15432 {
	struct row_permute { // col linearization

		int d2, d3, d4, d5;
		
		__host__
		row_permute(int _d5, int _d4, int _d3, int _d2) : d5(_d5), d4(_d4), d3(_d3), d2(_d2) {}
		
		__host__ __device__
		int operator()(const int& iklm, const int& j) {
			int m = iklm / (d5 * d4 * d3);
			int l = iklm / (d5 * d4) % d3;
			int i = (iklm / d5) % d4;
			int k = iklm % d5;
			
			return m + l * d2 + i * d2 * d3 + k * d2 * d3 * d4;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row permute");
				return name;
			}
		#endif
	};
}

namespace _43215 {

	struct row_shuffle { // row lineaization
		int d1, int_d2, int_d3;
		reduced_divisor d2, d3, d4;
		__host__
		row_shuffle(int _d4, int _d2, int _d3, int _d1) : 
		d4(_d4) , d2(_d2), d3(_d3), d1(_d1), int_d2(_d2), int_d3(_d3) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd4, jmodd4, jdivd4d3, jdivd4modd3, jdivd4d3d2, jdivd4d3modd2;
			d4.divmod(j, jdivd4, jmodd4);
			d3.divmod(jdivd4, jdivd4d3, jdivd4modd3);
			d2.divmod(jdivd4d3, jdivd4d3d2, jdivd4d3modd2);
			return jdivd4d3d2 + jdivd4d3modd2 * d1 + jdivd4modd3 * d1 * int_d2 + jmodd4 * d1 * int_d2 * int_d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

}

}

namespace _4d {

namespace _1324 {

	struct row_permute { // col linearization

		int d2, d3;
		
		__host__
		row_permute(int _d3, int _d2) : d3(_d3), d2(_d2) {}
		
		__host__ __device__
		int operator()(const int& ikl, const int& j) {
			int l = ikl / (d2 * d3);
			int i = (ikl / d3) % d2;
			int k = ikl % d3;
			return i + k * d2 + l * d2 * d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row permute");
				return name;
			}
		#endif
	};
	
	struct row_shuffle { // row lineaization
		int int_d1, d2;
		reduced_divisor d1, d3;
		__host__
		row_shuffle(int _d2, int _d3, int _d1) : d2(_d2), d3(_d3), d1(_d1), int_d1(_d1) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd1, jmodd1, jdivd1d3, jdivd1modd3;
			d1.divmod(j, jdivd1, jmodd1);
			d3.divmod(jdivd1, jdivd1d3, jdivd1modd3);
			return jmodd1 + jdivd1d3 * int_d1 + jdivd1modd3 * int_d1 * d2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

}


namespace _1432 {
	struct row_permute { // col linearization

		int d2, d3, d4;
		
		__host__
		row_permute(int _d4, int _d3, int _d2) : d4(_d4), d3(_d3), d2(_d2) {}
		
		__host__ __device__
		int operator()(const int& ikl, const int& j) {
			int l = ikl / (d4 * d3);
			int i = (ikl / d4) % d3;
			int k = ikl % d4;
			return l + i * d2 + k * d2 * d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row permute");
				return name;
			}
		#endif
	};
}

namespace _3214 {

	struct row_shuffle { // row lineaization
		int d1, int_d2;
		reduced_divisor d2, d3;
		__host__
		row_shuffle(int _d2, int _d3, int _d1) : d2(_d2), d3(_d3), d1(_d1), int_d2(_d2) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd3, jmodd3, jdivd3d2, jdivd3modd2;
			d3.divmod(j, jdivd3, jmodd3);
			d2.divmod(jdivd3, jdivd3d2, jdivd3modd2);
			return jdivd3d2 + jdivd3modd2 * d1 + jmodd3 * d1 * int_d2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

}

}

namespace _3d {

namespace _132 {
	
	struct row_permute {
		int d2, d3;
		
		__host__
		row_permute(int _d3, int _d2) : d3(_d3), d2(_d2) {}
		
		__host__ __device__
		int operator()(const int& ik, const int& j) {
			int k = ik % d3;
			int i = ik / d3;
			return i + k * d2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row permute");
				return name;
			}
		#endif
	};
}

namespace _213 {
	struct row_shuffle {
		int d1;
		reduced_divisor d2;
		__host__
		row_shuffle(int _d1, int _d2) : d1(_d1), d2(_d2) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd2, jmodd2;
			d2.divmod(j, jdivd2, jmodd2);
			return jdivd2 + jmodd2 * d1;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};
}

}

namespace _2d {

namespace c2r {
	struct rotate {
		reduced_divisor m;
		reduced_divisor b;
		__host__  rotate(int _m, int _b) : m(_m), b(_b) {}
		__host__  rotate() : m(1), b(1) {}
		
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i + (int)b.div(j));
		}

		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
		#endif
	};
	
	struct row_shuffle {
		int m, n, k;
		reduced_divisor_64 b;
		reduced_divisor c;
		__host__
		row_shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
													   b(_n/_c), c(_c) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		__host__ __device__
		int f(const int& j) {
			int r = j + i * (n - 1);
			//The (int) casts here prevent unsigned promotion
			//and the subsequent underflow: c implicitly casts
			//int - unsigned int to
			//unsigned int - unsigned int
			//rather than to
			//int - int
			//Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}
		
		__host__ __device__
		int operator()(const int& j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divmod(fij, fijdivc, fijmodc);
			// k = a^(-1) ??
			int term_1 = b.mod((long long)k * (long long)fijdivc);
			int term_2 = ((int)fijmodc) * (int)b.get();
			return term_1+term_2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

	struct col_shuffle {
		reduced_divisor m;
		int n, c;
		__host__ 
		col_shuffle(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i * n - (int)m.div(i * c) + j);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
		#endif
	};

} // End of c2r

namespace r2c {
	struct col_shuffle {
		reduced_divisor a;
		reduced_divisor c;
		reduced_divisor m;
		int q;
		__host__ 
		col_shuffle(int _a, int _c, int _m, int _q) : a(_a) , c(_c), m(_m), q(_q) {}
		__host__ __device__ __forceinline__
		int p(const int& i) {
			int cm1 = (int)c.get() - 1;
			int term_1 = int(a.get()) * (int)c.mod(cm1 * i);
			int term_2 = int(a.mod(int(c.div(cm1+i))*q));
			return term_1 + term_2;
			
		}
		__host__ __device__
		int operator()(const int& i, const int& j) {
			int idx = m.mod(i + (int)m.get() - (int)m.mod(j));
			return p(idx);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
		#endif
	};
	
	struct row_shuffle {
		reduced_divisor m;
		reduced_divisor n;
		reduced_divisor b;
		__host__
		row_shuffle(int _m, int _n, int _b) : m(_m), n(_n), b(_b) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}    
		__host__ __device__
		int operator()(const int& j) {
			int r = m.mod(b.div(j) + i) + j * (int)m.get();
			return n.mod(r);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Row Shuffle");
			return name;
		}
		#endif
	};
	
	struct row_scatter_shuffle {
		int m, n, k;
		reduced_divisor_64 b;
		reduced_divisor c;
		__host__
		row_scatter_shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
													   b(_n/_c), c(_c) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		__host__ __device__
		int f(const int& j) {
			int r = j + i * (n - 1);
			//The (int) casts here prevent unsigned promotion
			//and the subsequent underflow: c implicitly casts
			//int - unsigned int to
			//unsigned int - unsigned int
			//rather than to
			//int - int
			//Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}
		
		__host__ __device__
		int operator()(const int& j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divmod(fij, fijdivc, fijmodc);
			int term_1 = b.mod((long long)k * (long long)fijdivc);
			int term_2 = ((int)fijmodc) * (int)b.get();
			return term_1+term_2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};
	
	struct rotate {
		reduced_divisor m;
		reduced_divisor b;
		
		__host__  rotate(int _m, int _b) : m(_m), b(_b) {}
		
		__host__  rotate() : m(1), b(1) {}
		
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i + (int)m.get() - (int)b.div(j));
		}
		
	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
	#endif
	};

}

}


}