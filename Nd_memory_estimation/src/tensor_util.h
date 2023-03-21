template <typename T>
class TensorUtil {

/*
cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream);
*/

public:
	FILE* fp;
	int rank;
	int* dim;
	int* permutation;
	size_t* dim_long;
	size_t* permutation_long;
	size_t* stride;
	size_t vol;
	int source;
	int num_block;
	
	TensorUtil(FILE* _fp, int _rank, int* _dim, int* _permutation);
	~TensorUtil();
	void cal_vol();
	void init_data(T* data);
	void print_tensor(T* data);
	void write_file(T* data);
	void seq_tt(T* ans, T* data);
};