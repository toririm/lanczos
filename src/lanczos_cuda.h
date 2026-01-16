#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <stdlib.h>
#include <stdint.h>
#include "crs.h"

static inline const char *cusparse_status_to_string(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        default: return "CUSPARSE_STATUS_<UNKNOWN>";
    }
}

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
        printf("CUSPARSE API failed at line %d: %s -> %s (%d)\n",              \
               __LINE__, #func, cusparse_status_to_string(status), (int)status);\
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
        printf("CUSPARSE API failed at line %d: %s -> %s (%d)\n",              \
               __LINE__, #func, cusparse_status_to_string(status__), (int)status__);\
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
    int64_t *d_row_offsets;
    int64_t *d_columns;
    double *d_values;
    int rows;
    int cols;
    size_t nnz;
} CuSparseMatrix;

extern int create_cusparse_matrix(const Mat_Crs *src, CuSparseMatrix *dist);
extern void destroy_cusparse_matrix(CuSparseMatrix *mat);

extern int lanczos_cuda_crs(const Mat_Crs *mat,
                                 double eigenvalues[], double *eigenvectors[],
                                 int nth_eig, int max_iter, double threshold);
