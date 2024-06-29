#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include<iostream>
#include <cusparse_v2.h>


#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUDA(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    }


inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H



/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 5;
int N = 5;

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void print_partial_matrix(float *M, int nrows, int ncols, int max_row,
        int max_col)
{
    int row, col;

    for (row = 0; row < max_row; row++)
    {
        for (col = 0; col < max_col; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc, char **argv)
{
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    int *dANnzPerRow;

    float * CsrValA;
    int * CsrRowPtrA;
    int * CsrColIndA;

    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalANnz;
    float alpha = 3.0f;
    float beta = 4.0f;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t Adescr = 0;

    // Generate input
    srand(9384);
    int trueANnz = generate_random_dense_matrix(M, N, &A);
    // int trueBNnz = generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);

    printf("A:\n");
    print_partial_matrix(A, M, N, 2, 2);

    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            printf("%2.2f ", A[row * N + col]);
        }
        printf("\n");
       
    }
    // printf("B:\n");
    // print_partial_matrix(B, N, M, 2, 2);

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));




    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    // CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));
    CHECK(cudaMalloc((void **)&dANnzPerRow, sizeof(int) * M));

    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&Adescr));
    CHECK_CUSPARSE(cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO));

    

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0x00, sizeof(float) * M * M));

    // Compute the number of non-zero elements in A
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
                                dA, M, dANnzPerRow, &totalANnz));

    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }


    // Allocate device memory for vectors and the dense form of the matrix A
    CsrValA = (float *)malloc(sizeof(float) * totalANnz);
    CsrRowPtrA = (int *)malloc(sizeof(int) * (M + 1));
    CsrColIndA = (int *)malloc(sizeof(int) * totalANnz);
    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, Adescr, dA, M, dANnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));

    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    // CHECK_CUSPARSE(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M,
    //                               M, N, totalANnz, &alpha, Adescr, dCsrValA,
    //                               dCsrRowPtrA, dCsrColIndA, dB, N, &beta, dC,
    //                               M));

//   CsrValA = (float *)malloc(sizeof(float) * totalANnz);
//     CsrRowPtrA = (int *)malloc(sizeof(int) * (M + 1));
//     CsrColIndA = (int *)malloc(sizeof(int) * totalANnz);

    CHECK(cudaMemcpy(CsrValA, dCsrValA, sizeof(float) * totalANnz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CsrRowPtrA, dCsrRowPtrA, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CsrColIndA, dCsrColIndA, sizeof(int) * totalANnz, cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));

    std::cout<<"CsrValA "<<std::endl;
    for (int i = 0 ; i < totalANnz ; i ++)
    {
        std::cout<<CsrValA[i]<< " ";
    }
    std::cout<<"CsrRowPtrA "<<std::endl;
    for (int i = 0 ; i < (M + 1) ; i ++)
    {
        std::cout<<CsrRowPtrA[i]<< " ";
    }
    std::cout<<"CsrColIndA "<<std::endl;
    for (int i = 0 ; i < totalANnz ; i ++)
    {
        std::cout<<CsrColIndA[i]<< " ";
    }

    // printf("C:\n");
    // print_partial_matrix(C, M, M, 10, 10);
    float * dX;
    CHECK_CUDA(cudaMalloc((void**)&dX, N * sizeof(float)));

    // cusparseMatDescr_t descrA;
    // CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    // CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    // CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // // Analyze the matrix (needed for solve)
    
    // cusparseSolveAnalysisInfo_t infoA;
    // CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&infoA));
    // CHECK_CUSPARSE(cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, totalANnz,
    //                                        descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, infoA));

    // // Solve the system
    // // const float alpha = 1.0f;
    // CHECK_CUSPARSE(cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &alpha,
    //                                     descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, infoA, dB, dX));

    // // Copy the result back to host
    // float *hX;
    // CHECK_CUDA(cudaMemcpy(hX, dX, N * sizeof(float), cudaMemcpyDeviceToHost));


/////////
 // Create and set up matrix descriptor
    cusparseMatDescr_t descrA;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // Create the analysis info structure
    csric02Info_t infoA;
    CHECK_CUSPARSE(cusparseCreateCsric02Info(&infoA));

    // Buffer size query
    int pBufferSize;
    CHECK_CUSPARSE(cusparseScsric02_bufferSize(handle, M, totalANnz, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, infoA, &pBufferSize));
    
    // Allocate buffer
    void* pBuffer;
    CHECK_CUDA(cudaMalloc(&pBuffer, pBufferSize));
    
    // Perform analysis
    // int structural_zero;
    // CHECK_CUSPARSE(cusparseScsric02_analysis(handle, M, totalANnz, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, infoA, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
    // CHECK_CUSPARSE(cusparseXcsric02_zeroPivot(handle, infoA, &structural_zero));

    // Solve the system
    // const float alpha = 1.0f;
    cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    CHECK_CUSPARSE(cusparseScsric02(handle, M, totalANnz, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, infoA, policy, pBuffer));

    // Copy the result back to host
    float hX[N];
    CHECK_CUDA(cudaMemcpy(hX, dX, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout<<"hX "<<std::endl;
    for (int i = 0 ; i < N ; i ++)
    {
        std::cout<<hX[i]<< " ";
    }

///////
    


    free(A);
    // free(B);
    free(C);

    CHECK(cudaFree(dA));
    // CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(dANnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}