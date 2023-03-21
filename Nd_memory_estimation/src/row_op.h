namespace inplace {

namespace _4d {

namespace _1324 {


template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3, int d4);



template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3, int d4);

}

}

namespace _3d {

namespace _132 {

template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3);

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3);

}

namespace _213 {

template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2, int d3);

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2, int d3);

}

}

namespace _2d {

template<typename F, typename T>
size_t row_gather_op(F fn, T* data, int source, int d1, int d2);

template<typename F, typename T>
size_t row_scatter_op(F fn, T* data, int source, int d1, int d2);

}

}
