#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>
#include "util.h"

int msb(int x) {
	return static_cast<int>(log2(x));
}

int get_num_thread(int d1) {
    //int msb = static_cast<int>(log2(d1)); // most significant bit
    unsigned n_threads = static_cast<unsigned>(2 << msb(d1));
    unsigned lim = 1024;
    return static_cast<int>(std::min(n_threads, lim));
}

void print_vec(std::vector<double> v) {
	std::vector<double>::iterator it;
	for (it = v.begin(); it != v.end(); ++it) {std::cout << " " << *it;}
	std::cout << std::endl;
}

void print_arr(int *arr, int size) {
	for(int i = 0; i < size; ++i) { printf("%d ", arr[i]);}
	//printf("\n");
}

bool verify_perm(int *perm, int rank) {	
		bool ans_flag = true;
		if(rank == 2 && perm[0] == 2 && perm[1] == 1) { return ans_flag;}
		int *tmp = new int[rank + 1];
		for(int j = 1; j <= rank; ++j)
		{
			tmp[j] = 0;
			for(int i = 0; i < rank; ++i)
			{
				if(perm[i] == j)
				{
					tmp[j] = 1;
					break;
				}
			}
		}
		for(int j = 1; j <= rank; ++j)
		{
			if(tmp[j] == 0)
			{
				ans_flag = false;
				break;
			}
		}
		delete [] tmp;
		//if(ans_flag == false) { print_arr(perm, rank);}
		return ans_flag;
	}

bool verify_perm_vec(std::vector<int> perm, int rank) {
		bool ans_flag = true;
		std::vector<int> tmp;
		tmp.assign(rank + 1, 0);
		for(int j = 1; j <= rank; ++j)
		{
			for(int i = 0; i < rank; ++i)
			{
				if(perm.at(i) == j)
				{
					tmp.at(j) = 1;
					break;
				}
			}
		}
		for(int j = 1; j <= rank; ++j)
		{
			if(tmp.at(j) == 0)
			{
				ans_flag = false;
				break;
			}
		}
		return ans_flag;
	}

int find_proper_max_dim_pos(int *dim, int rank) {
	int max_dim = *std::max_element(dim, dim + rank);
	int max_dim_pos = -1;
	for(int i = rank - 1; i >= 0; --i) {
		if(dim[i] == max_dim) { 
			max_dim_pos = i;
			break;
		}
	}
	assert(max_dim_pos != -1);
	return max_dim_pos;
}


void Timer::start() {
    start_tp = clock::now();
}
void Timer::stop() {
    stop_tp = clock::now();
}
double Timer::elapsed_time() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_tp - start_tp).count();
}