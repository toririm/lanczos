#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <stdlib.h>
#include "crs.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error code: %d\n",         \
               __LINE__, status);                                             \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error code: %d\n",           \
               __LINE__, status);                                             \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("CUSOLVER API failed at line %d with error code: %d\n",         \
               __LINE__, status);                                             \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CURAND(func)                                                     \
{                                                                              \
    curandStatus_t status = (func);                                            \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
        printf("CURAND API failed at line %d with error code: %d\n",           \
               __LINE__, status);                                             \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA_GOTO(func, label)                                           \
do {                                                                           \
	cudaError_t status__ = (func);                                             \
	if (status__ != cudaSuccess) {                                             \
		printf("CUDA API failed at line %d with error: %s (%d)\n",             \
			   __LINE__, cudaGetErrorString(status__), status__);              \
		goto label;                                                            \
	}                                                                          \
} while (0)

#define CHECK_CUSPARSE_GOTO(func, label)                                       \
do {                                                                           \
	cusparseStatus_t status__ = (func);                                        \
	if (status__ != CUSPARSE_STATUS_SUCCESS) {                                 \
		printf("CUSPARSE API failed at line %d with error code: %d\n",         \
			   __LINE__, status__);                                            \
		goto label;                                                            \
	}                                                                          \
} while (0)

#define CHECK_CUBLAS_GOTO(func, label)                                         \
do {                                                                           \
	cublasStatus_t status__ = (func);                                          \
	if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
		printf("CUBLAS API failed at line %d with error code: %d\n",           \
			   __LINE__, status__);                                            \
		goto label;                                                            \
	}                                                                          \
} while (0)

#define CHECK_CUSOLVER_GOTO(func, label)                                       \
do {                                                                           \
	cusolverStatus_t status__ = (func);                                        \
	if (status__ != CUSOLVER_STATUS_SUCCESS) {                                 \
		printf("CUSOLVER API failed at line %d with error code: %d\n",         \
			   __LINE__, status__);                                            \
		goto label;                                                            \
	}                                                                          \
} while (0)

#define CHECK_CURAND_GOTO(func, label)                                         \
do {                                                                           \
	curandStatus_t status__ = (func);                                          \
	if (status__ != CURAND_STATUS_SUCCESS) {                                   \
		printf("CURAND API failed at line %d with error code: %d\n",           \
			   __LINE__, status__);                                            \
		goto label;                                                            \
	}                                                                          \
} while (0)

typedef struct {
    cusparseSpMatDescr_t descr;
    int *d_row_offsets;
    int *d_columns;
    double *d_values;
    int rows;
    int cols;
    int nnz;
} CuSparseMatrix;

extern int create_cusparse_matrix(const Mat_Crs *src, CuSparseMatrix *dist);
extern void destroy_cusparse_matrix(CuSparseMatrix *mat);

extern int lanczos_cusparse_crs(const Mat_Crs *mat,
                                 double eigenvalues[], double *eigenvectors[],
                                 int nth_eig, int max_iter, double threshold);
