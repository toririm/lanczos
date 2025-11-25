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
				 		const cusparseDnVecDescr_t *vec,
				 		cusparseDnVecDescr_t *dist) {
	(void)dimension;
	const double alpha = 1.0;
	const double beta  = 0.0;
	cusparseHandle_t handle = NULL;
	void *dBuffer = NULL;
	size_t bufferSize = 0;
	CHECK_CUSPARSE( cusparseCreate(&handle) );
	CHECK_CUSPARSE( cusparseSpMV_bufferSize(handle,
								CUSPARSE_OPERATION_NON_TRANSPOSE,
								&alpha, *mat, *vec, &beta, *dist,
								CUDA_R_64F,
								CUSPARSE_SPMV_ALG_DEFAULT,
								&bufferSize) );
	if (bufferSize > 0) {
		CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );
	}
	CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
									&alpha, *mat, *vec, &beta, *dist,
									CUDA_R_64F,
									CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );
	if (dBuffer != NULL) {
		CHECK_CUDA( cudaFree(dBuffer) );
	}
	CHECK_CUSPARSE( cusparseDestroy(handle) );
	return EXIT_SUCCESS;
}

int lanczos_cusparse_crs(const Mat_Crs *mat,
             		  double eigenvalues[], double *eigenvectors[],
             		  int nth_eig, int max_iter, double threshold) {
	cusparseSpMatDescr_t matA = NULL;
	double *dVecIn = NULL;
	double *dVecOut = NULL;
	cusparseDnVecDescr_t vecX = NULL;
	cusparseDnVecDescr_t vecY = NULL;
	double **v = NULL;
	double **tmat = NULL;
	double *teval_last = NULL;
	bool allocated_eigenvectors = false;
	int status = EXIT_FAILURE;

	if (create_cusparse_matrix(mat, &matA) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	const int mat_dim = mat->dimension;
	CHECK_CUDA( cudaMalloc((void**) &dVecIn, mat_dim * sizeof(double)) );
	CHECK_CUDA( cudaMalloc((void**) &dVecOut, mat_dim * sizeof(double)) );
	CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, mat_dim, dVecIn, CUDA_R_64F) );
	CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, mat_dim, dVecOut, CUDA_R_64F) );

	v = calloc(max_iter, sizeof(double *));
	if (v == NULL) {
		goto cleanup;
	}
	for (int i = 0; i < max_iter; i++) {
		v[i] = calloc(mat_dim, sizeof(double));
		if (v[i] == NULL) {
			goto cleanup;
		}
	}
	tmat = calloc(max_iter, sizeof(double *));
	if (tmat == NULL) {
		goto cleanup;
	}
	for (int i = 0; i < max_iter; i++) {
		tmat[i] = calloc(max_iter, sizeof(double));
		if (tmat[i] == NULL) {
			goto cleanup;
		}
	}
	eigenvectors = calloc(mat_dim, sizeof(double *));
	if (eigenvectors == NULL) {
		goto cleanup;
	}
	allocated_eigenvectors = true;
	for (int i = 0; i < mat_dim; i++) {
		eigenvectors[i] = calloc(mat_dim, sizeof(double));
		if (eigenvectors[i] == NULL) {
			goto cleanup;
		}
	}
	teval_last = calloc(mat_dim, sizeof(double));
	if (teval_last == NULL) {
		goto cleanup;
	}

	gaussian_random_vec(mat_dim, v[1]);

	double norm = sqrt(dot_product(v[1], v[1], mat_dim));
	double alpha = 0.0;
	double beta = 0.0;
	for (int i = 0; i < mat_dim; i++) {
		v[1][i] = 1.0 / norm * v[1][i];
	}

	for (int k = 1; k < max_iter - 1; k++) {
		CHECK_CUDA( cudaMemcpy(dVecIn, v[k], mat_dim * sizeof(double),
					cudaMemcpyHostToDevice) );
		CHECK_CUDA( cudaMemset(dVecOut, 0, mat_dim * sizeof(double)) );
		if (matvec_cusparse_crs(&matA, mat_dim, &vecX, &vecY) != EXIT_SUCCESS) {
			goto cleanup;
		}
		CHECK_CUDA( cudaMemcpy(v[k + 1], dVecOut,
					mat_dim * sizeof(double), cudaMemcpyDeviceToHost) );
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
			status = EXIT_SUCCESS;
			goto cleanup;
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
		double beta_next = sqrt(dot_product(v[k + 1], v[k + 1], mat_dim));
		if (beta_next * beta_next <
			threshold * threshold * threshold * threshold) {
			printf("%.7f beta converged\n", beta_next);
			status = EXIT_SUCCESS;
			goto cleanup;
		}
		for (int i = 0; i < mat_dim; i++) {
			v[k + 1][i] /= beta_next;
		}
		tmat[k][k + 1] = beta_next;
		tmat[k + 1][k] = beta_next;
		beta = beta_next;
	}

	status = EXIT_SUCCESS;

cleanup:
	if (vecX != NULL) {
		cusparseDestroyDnVec(vecX);
	}
	if (vecY != NULL) {
		cusparseDestroyDnVec(vecY);
	}
	if (dVecIn != NULL) {
		cudaFree(dVecIn);
	}
	if (dVecOut != NULL) {
		cudaFree(dVecOut);
	}
	if (tmat != NULL) {
		for (int i = 0; i < max_iter; i++) {
			free(tmat[i]);
		}
		free(tmat);
	}
	if (v != NULL) {
		for (int i = 0; i < max_iter; i++) {
			free(v[i]);
		}
		free(v);
	}
	if (allocated_eigenvectors && eigenvectors != NULL) {
		for (int i = 0; i < mat_dim; i++) {
			free(eigenvectors[i]);
		}
		free(eigenvectors);
	}
	free(teval_last);
	if (matA != NULL) {
		cusparseDestroySpMat(matA);
	}
	return status;
}
