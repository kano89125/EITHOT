# ITNT: In-place Transposition of N-order Tensor on Graphics Processing Units

Tensor transposition is a fundamental operation in many applications, but naive implementations require double the memory and aren't suitable for large-scale tensors on memory-limited GPUs. To address this, ITNT is presented as an efficient algorithm for in-place tensor transposition on GPUs, requiring at most 5% additional memory for large tensors.

## Algorithm
---
ITNT is based on the following steps:

1. Factorize high-order tensors into a sequence of low-order transpositions using permutation decomposition.
2. Divide large tensors into smaller ones based on the estimation of required extra memory and transpose each separately.
3. Reassemble the transposed sub-tensors to obtain the desired result.

## Environment
---
The following environment was used to develop and test ITNT:

- Operating System: Ubuntu 22.04
- GPU: NVIDIA RTX 3090
- CUDA Toolkit: 12.0

If you plan on using a GPU that's different from the above one, it's crucial to verify that your GPU is compatible with the CUDA version you're using. Additionally, you may need to make modifications to the "NVCCFLAGS" in the Makefile to ensure that the architecture is properly aligned with your GPU. To assist in this process, you can refer to the website <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/> to find the corresponding architecture for your GPU.

## Getting start
---
To build and install ITNT, follow these steps:
1. Clone the repository:
```
$ git clone https://github.com/kano89125/ITNT.git
```
2. Navigate to the **Nd_memory_estimation** directory and Build the project using the Makefile:
```
$ cd Nd_memory_estimation
$ make
$ cd ..
```
3. Navigate to the **ITNT** directory and Build the project using the Makefile:
```
$ cd ITNT
$ make
$ cd ..
```
## Usage
---
### Input Format
The input to ITNT consists of the following parameters, separated by spaces:
1. Dimensions: A sequence of integers indicating the size of each dimension of the tensor, in order. For example, "2 41961 8192 2" specifies a 4-dimensional tensor with dimensions (2, 41961, 8192, 2).

2. Permutation: A sequence of integers indicating the desired permutation of dimensions for the transposition, using the 1-based index of each dimension. For example, "1 3 2 4" specifies that the tensor should be transposed by swapping the second and third dimensions.

3. Element Size: The size of each element in bytes. For example, "4" indicates that each element is a 32-bit (4-byte).

4. Number of Sub-Tensors: An integer indicating the number of sub-tensors to split the input tensor into.

5. Expected Extra Memory Ratio: A floating-point number indicating the maximum expected ratio of extra memory usage during the transposition.

For example, "2 41961 8192 2 1 3 2 4 4 1 0.01" specifies a 4-dimensional tensor with dimensions (2, 41961, 8192, 2), to be transposed by swapping the second and third dimensions. Each element is a 32-bit integer (4 bytes), and the tensor should be split into 1 sub-tensor. The expected extra memory usage ratio is 1%.

### Test
We provide a script called **test_INT.sh** in both the **Nd_memory_estimation** and **ITNT** folders. To use the script, first navigate to the **Nd_memory_estimation** folder and run the script with the following command:
```
$ ./test_INT.sh
```
In the script, you need to specify the input format, including the dimensions size, permutation of dimensions (a permutation of sequence 1~n), element data size in bytes, number of sub-tensors, and expected ratio of extra memory usage. For the number of sub-tensors, use 1 to start with a single sub-tensor.

After you run the script, the program will estimate the optimal number of sub-tensors for the tensor transposition. 

Next, navigate to the ITNT folder and run the **test_INT.sh** script again with the following command:
```
$ ./test_INT.sh
```
This time, the script will perform the tensor transposition and output the elapsed time.
