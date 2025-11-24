#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "util.h"
#include "lanczos_cusparse.h"

int create_cusparse_matrix(const Mat_Crs *src, cusparseSpMatDescr_t *dist) {
	int    *dA_csrOffsets, *dA_columns;
	double *dA_values;
	CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
									(src->dimension + 1) * sizeof(int)) );
	CHECK_CUDA( cudaMalloc((void**) &dA_columns,
									src->length * sizeof(int)) );
	CHECK_CUDA( cudaMalloc((void**) &dA_values,
									src->length * sizeof(double)) );
	CHECK_CUDA( cudaMemcpy(dA_csrOffsets, src->row_head_indxes,
									(src->dimension + 1) * sizeof(int),
									cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemcpy(dA_columns, src->column_index,
									src->length * sizeof(int),
									cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemcpy(dA_values, src->values,
									src->length * sizeof(double),
									cudaMemcpyHostToDevice) );
	CHECK_CUSPARSE( cusparseCreateCsr(dist, src->dimension, src->dimension,
									   src->length,
									   dA_csrOffsets, dA_columns, dA_values,
									   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
	return EXIT_SUCCESS;
}

int matvec_cusparse_crs(const cusparseSpMatDescr_t *mat, int dimension,
						 const double *vec, double *dist) {
	const double alpha = 1.0;
	const double beta  = 0.0;
	// Device memory management
	// y = alpha * A * x + beta * y
	// x, y: vectors
	// A: sparse matrix in CSR format
	// alpha, beta: scalars
    double *dX, *dY;
	CHECK_CUDA( cudaMalloc((void**) &dX, dimension * sizeof(double)) );
	CHECK_CUDA( cudaMalloc((void**) &dY, dimension * sizeof(double)) );
	CHECK_CUDA( cudaMemcpy(dX, vec, dimension * sizeof(double),
									cudaMemcpyHostToDevice) );
	// cusparse
	cusparseHandle_t 	 handle = NULL;
	cusparseDnVecDescr_t vecX, vecY;
	void* 				 dBuffer = NULL;
	size_t 				 bufferSize = 0;
	CHECK_CUSPARSE( cusparseCreate(&handle) );
	CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dimension, dX, CUDA_R_64F) );
	CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, dimension, dY, CUDA_R_64F) );
	CHECK_CUSPARSE( cusparseSpMV_bufferSize(
									handle,
									CUSPARSE_OPERATION_NON_TRANSPOSE,
									&alpha, *mat, vecX, &beta, vecY,
									CUDA_R_64F,
									CUSPARSE_SPMV_ALG_DEFAULT,
									&bufferSize) );
	CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );
	CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
									&alpha, *mat, vecX, &beta, vecY,
									CUDA_R_64F,
									CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );
	CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) );
	CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) );
	CHECK_CUSPARSE( cusparseDestroy(handle) );
	// copy result back to host
	CHECK_CUDA( cudaMemcpy(dist, dY, dimension * sizeof(double),
									cudaMemcpyDeviceToHost) );
	// device memory deallcation
	CHECK_CUDA( cudaFree(dBuffer) );
	CHECK_CUDA( cudaFree(dX) );
	CHECK_CUDA( cudaFree(dY) );
	return EXIT_SUCCESS;
}

int lanczos_cusparse_crs(const Mat_Crs *mat,
             			  double eigenvalues[], double *eigenvectors[],
             			  int nth_eig, int max_iter, double threshold) {
	// Device memory management
	// y = alpha * A * x + beta * y
	// x, y: vectors
	// A: sparse matrix in CSR format
	// alpha, beta: scalars
    int    *dA_csrOffsets, *dA_columns;
	double *dA_values;
	cusparseSpMatDescr_t matA = NULL;
	create_cusparse_matrix(mat, &matA);
	// --- Lanczos algorithm ---
    const int mat_dim = mat->dimension;
	double **v, **tmat, *teval_last;
	double norm, alpha, beta = 0;
	v = calloc(max_iter, sizeof(double *));
	for (int i = 0; i < max_iter; i++) {
		v[i] = calloc(mat_dim, sizeof(double));
	}
	tmat = calloc(max_iter, sizeof(double *));
	for (int i = 0; i < max_iter; i++) {
		tmat[i] = calloc(max_iter, sizeof(double));
	}
	eigenvectors = calloc(mat_dim, sizeof(double *));
	for (int i = 0; i < mat_dim; i++) {
		eigenvectors[i] = calloc(mat_dim, sizeof(double));
	}
	teval_last = calloc(mat_dim, sizeof(double));
	
	gaussian_random_vec(mat_dim, v[1]);

	norm = sqrt(dot_product(v[1], v[1], mat_dim));
	for (int i = 0; i < mat_dim; i++) {
		v[1][i] = 1.0 / norm * v[1][i];
	}

	for (int k = 1; k < max_iter - 1; k++) {
		matvec_cusparse_crs(&matA, mat_dim, v[k], v[k + 1]);
		alpha = dot_product(v[k], v[k + 1], mat_dim);
		tmat[k][k] = alpha;
		for (int i = 0; i < nth_eig; i++) {
			teval_last[i] = eigenvalues[i];
		}
		diagonalize_double(tmat, eigenvalues, eigenvectors, k + 1);
		bool all = true;
		for (int i = 0; i < nth_eig; i++) {
			double diff = eigenvalues[i] - teval_last[i];
			if (diff * diff > threshold * threshold) {
				all = false;
			}
		}
		if (all) {
			printf("converged ");
			for (int i = 0; i < nth_eig; i++)
				printf("%.1E ", teval_last[i] - eigenvalues[i]);
			printf("\n");
			return EXIT_SUCCESS;
		}
		printf("%d\t", k);
		for (int i = 0; i < nth_eig; i++)
			printf("%.7f\t", eigenvalues[i]);
		printf("\n");
		for (int i = 0; i < mat_dim; i++) {
			v[k + 1][i] = v[k + 1][i] - beta * v[k - 1][i] - alpha * v[k][i];
		}
		for (int l = 0; l < k - 2; l++) {
			double coeff = dot_product(v[l], v[k + 1], mat_dim);
			for (int i = 0; i < mat_dim; i++) {
				v[k + 1][i] -= v[l][i] * coeff;
			}
		}
		beta = sqrt(dot_product(v[k + 1], v[k + 1], mat_dim));
		if (beta * beta < threshold * threshold * threshold * threshold) {
			printf("%.7f beta converged\n", beta);
			return EXIT_SUCCESS;
		}
		for (int i = 0; i < mat_dim; i++) {
			v[k + 1][i] /= beta;
		}
		tmat[k][k + 1] = beta;
		tmat[k + 1][k] = beta;
	}
	CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
	CHECK_CUDA( cudaFree(dA_csrOffsets) );
	CHECK_CUDA( cudaFree(dA_columns) );
	CHECK_CUDA( cudaFree(dA_values) );
	return EXIT_SUCCESS;
}
