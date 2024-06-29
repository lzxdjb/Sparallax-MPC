#pragma once

#include <iostream>
#include <Eigen/Dense>
#include "type.cuh"

using namespace Eigen;

#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    }
}




__device__ double MyatomicAdd(double* address, double val);

__device__ void atomicAddMatrixBlock(UpperDualCache *dst, temp *src) ;


__device__ void copy(double *a, gradient b, int number);


__device__ gradient GetGradient(state state1, control control, final_state final_state, QCost Q, RCost R);

__device__ Hessian GetHessian(QCost Q, RCost R);



__device__ equality GetEquality(state state1, control control, state initial);

__device__ JB GetJB1(state state1, state initial);

__device__ JB GetJB2(state state1, control control);


__device__ Hessian PsedoInverse(Hessian hessian);

// __device__ void mycopy(SharedMatrix *shared, temp temp1, temp temp2 , temp temp3 , temp temp4 ,  int idx);

// __device__ void mycopy2(SharedMatrix *shared, temp temp1, int idx);

// __device__ void DebugCopy(SharedMatrix *shared, SharedMatrix *debug);

__device__ void SecondPhaseCopy(FirstPhaseDual * FirstDual , double * d_x ,int idx) ;


__device__ double MyatomicAdd(double *address, double val);


// __device__ double empty(SharedMatrix *matrix);


__device__  int dense_to_csr1(UpperDualCache h_A_dense, SparseUpperDualCache *sparse,  int Nrows, int Ncols,
                  RowIndices* h_A_RowIndices, ColIndices* h_A_ColIndices);


__device__  void dense_to_csr2(UpperDualCache h_A_dense1, UpperDualCache h_A_dense2 , UpperDualCache h_A_dense3 ,  UpperDualCache h_A_dense4 ,  SparseUpperDualCache * sparse,  int Nrows, int Ncols,
                  RowIndices* h_A_RowIndices, ColIndices* h_A_ColIndices);


__device__  int dense_to_csr1(UpperDualCache h_A_dense, SparseUpperDualCache *sparse,  int Nrows, int Ncols,
                  RowIndices* h_A_RowIndices, ColIndices* h_A_ColIndices);


__device__  int dense_to_csr(UpperDualCache cacheA , UpperDualCache cacheB , UpperDualCache cacheC , SparseUpperDualCache *sparse,  int Nrows, int Ncols,
                  RowIndices* h_A_RowIndices, ColIndices* h_A_ColIndices);

__device__ void SparsecopyToPointer(const SparseUpperDualCache& matrix, tinytype* ptr, int a, int n, int I) ;


__device__ void RowIndicescopyToPointer(const RowIndices& matrix, int* ptr, int a, int n, int I);

__device__ void ColIndicescopyToPointer(const ColIndices& matrix, int* ptr, int a, int n, int I);