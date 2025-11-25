#include <cuda_runtime_api.h>
#include <cusparse.h>
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
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

extern int create_cusparse_matrix(const Mat_Crs *src, cusparseSpMatDescr_t *dist);

extern int matvec_cusparse_crs(const cusparseSpMatDescr_t *mat, int dimension,
						       const cusparseDnVecDescr_t *vec,
                               cusparseDnVecDescr_t *dist);

extern int lanczos_cusparse_crs(const Mat_Crs *mat,
                                 double eigenvalues[], double *eigenvectors[],
                                 int nth_eig, int max_iter, double threshold);
