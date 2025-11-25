#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "util.h"
#include "lanczos_cusparse.h"

static inline size_t round_up_even(size_t value) {
	return (value % 2 == 0) ? value : value + 1;
}

int create_cusparse_matrix(const Mat_Crs *src, CuSparseMatrix *dist) {
	if (src == NULL || dist == NULL) {
		return EXIT_FAILURE;
	}

	memset(dist, 0, sizeof(*dist));

	dist->rows = src->dimension;
	dist->cols = src->dimension;
	dist->nnz  = src->length;

	cudaError_t cuda_status;

	cuda_status = cudaMalloc((void**) &dist->d_row_offsets,
							 (size_t)(dist->rows + 1) * sizeof(int));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMalloc((void**) &dist->d_columns,
							 (size_t)dist->nnz * sizeof(int));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMalloc((void**) &dist->d_values,
							 (size_t)dist->nnz * sizeof(double));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMemcpy(dist->d_row_offsets, src->row_head_indxes,
							 (size_t)(dist->rows + 1) * sizeof(int),
							 cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMemcpy(dist->d_columns, src->column_index,
							 (size_t)dist->nnz * sizeof(int),
							 cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMemcpy(dist->d_values, src->values,
							 (size_t)dist->nnz * sizeof(double),
							 cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cusparseStatus_t sparse_status = cusparseCreateCsr(&dist->descr,
													   dist->rows,
													   dist->cols,
													   dist->nnz,
													   dist->d_row_offsets,
													   dist->d_columns,
													   dist->d_values,
													   CUSPARSE_INDEX_32I,
													   CUSPARSE_INDEX_32I,
													   CUSPARSE_INDEX_BASE_ZERO,
													   CUDA_R_64F);
	if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
		destroy_cusparse_matrix(dist);
		printf("CUSPARSE API failed at line %d with error code: %d\n",
			   __LINE__, sparse_status);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void destroy_cusparse_matrix(CuSparseMatrix *mat) {
	if (mat == NULL) {
		return;
	}
	if (mat->descr != NULL) {
		cusparseDestroySpMat(mat->descr);
	}
	if (mat->d_row_offsets != NULL) {
		cudaFree(mat->d_row_offsets);
	}
	if (mat->d_columns != NULL) {
		cudaFree(mat->d_columns);
	}
	if (mat->d_values != NULL) {
		cudaFree(mat->d_values);
	}
	memset(mat, 0, sizeof(*mat));
}

int lanczos_cusparse_crs(const Mat_Crs *mat,
			  double eigenvalues[], double *eigenvectors[],
			  int nth_eig, int max_iter, double threshold) {
	(void)eigenvectors;

	if (mat == NULL || eigenvalues == NULL) {
		return EXIT_FAILURE;
	}

	if (max_iter < 2) {
		printf("max_iter must be >= 2\n");
		return EXIT_FAILURE;
	}

	CuSparseMatrix matA;
	if (create_cusparse_matrix(mat, &matA) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	int status = EXIT_FAILURE;
	const int mat_dim = mat->dimension;
	const size_t vec_stride = (size_t)mat_dim;

	cusparseHandle_t sp_handle = NULL;
	cublasHandle_t   blas_handle = NULL;
	cusolverDnHandle_t solver_handle = NULL;
	curandGenerator_t rng = NULL;
	cusparseDnVecDescr_t vecX = NULL;
	cusparseDnVecDescr_t vecY = NULL;
	double *d_V = NULL;
	double *d_T = NULL;
	double *d_W = NULL;
	double *d_work = NULL;
	int *d_info = NULL;
	void *d_buffer = NULL;
	double *d_random = NULL;

	double *h_T = NULL;
	double *teval_last = NULL;

	const int ld = max_iter;
	size_t matrix_bytes = (size_t)ld * (size_t)ld * sizeof(double);
	size_t eigvec_bytes = matrix_bytes;

	double threshold_sq = threshold * threshold;

	memset(eigenvalues, 0, (size_t)nth_eig * sizeof(double));

	teval_last = (double*) calloc((size_t)nth_eig, sizeof(double));
	if (teval_last == NULL) {
		printf("Failed to allocate teval_last\n");
		goto cleanup;
	}

	h_T = (double*) calloc((size_t)ld * (size_t)ld, sizeof(double));
	if (h_T == NULL) {
		printf("Failed to allocate host tridiagonal buffer\n");
		goto cleanup;
	}

	CHECK_CUSPARSE_GOTO(cusparseCreate(&sp_handle), cleanup);
	CHECK_CUBLAS_GOTO(cublasCreate(&blas_handle), cleanup);
	CHECK_CUSOLVER_GOTO(cusolverDnCreate(&solver_handle), cleanup);
	CHECK_CURAND_GOTO(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT), cleanup);
	CHECK_CURAND_GOTO(curandSetPseudoRandomGeneratorSeed(rng, 123456789ULL), cleanup);

	const size_t total_vec_elems = vec_stride * (size_t)max_iter;
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_V, total_vec_elems * sizeof(double)), cleanup);
	CHECK_CUDA_GOTO(cudaMemset(d_V, 0, total_vec_elems * sizeof(double)), cleanup);

	const size_t padded_dim = round_up_even((size_t)mat_dim);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_random, padded_dim * sizeof(double)), cleanup);

	CHECK_CURAND_GOTO(curandGenerateNormalDouble(rng, d_random, padded_dim, 0.0, 1.0), cleanup);
	CHECK_CUDA_GOTO(cudaMemcpy(d_V, d_random, (size_t)mat_dim * sizeof(double), cudaMemcpyDeviceToDevice), cleanup);

	double norm = 0.0;
	CHECK_CUBLAS_GOTO(cublasDnrm2(blas_handle, mat_dim, d_V, 1, &norm), cleanup);
	if (norm == 0.0) {
		printf("Random vector has zero norm\n");
		goto cleanup;
	}
	double inv_norm = 1.0 / norm;
	CHECK_CUBLAS_GOTO(cublasDscal(blas_handle, mat_dim, &inv_norm, d_V, 1), cleanup);

	CHECK_CUSPARSE_GOTO(cusparseCreateDnVec(&vecX, mat_dim, d_V, CUDA_R_64F), cleanup);
	CHECK_CUSPARSE_GOTO(cusparseCreateDnVec(&vecY, mat_dim, d_V + vec_stride, CUDA_R_64F), cleanup);

	size_t buffer_size = 0;
	const double alpha_spmv = 1.0;
	const double beta_spmv = 0.0;
	CHECK_CUSPARSE_GOTO(cusparseSpMV_bufferSize(sp_handle,
												CUSPARSE_OPERATION_NON_TRANSPOSE,
												&alpha_spmv,
												matA.descr,
												vecX,
												&beta_spmv,
												vecY,
												CUDA_R_64F,
												CUSPARSE_SPMV_ALG_DEFAULT,
												&buffer_size), cleanup);
	if (buffer_size > 0) {
		CHECK_CUDA_GOTO(cudaMalloc(&d_buffer, buffer_size), cleanup);
	}

	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_T, matrix_bytes), cleanup);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_W, (size_t)ld * sizeof(double)), cleanup);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_info, sizeof(int)), cleanup);

	int lwork = 0;
	CHECK_CUSOLVER_GOTO(cusolverDnDsyevd_bufferSize(solver_handle,
													CUSOLVER_EIG_MODE_VECTOR,
													CUBLAS_FILL_MODE_UPPER,
													ld,
													d_T,
													ld,
													d_W,
													&lwork), cleanup);
	if (lwork <= 0) {
		printf("Invalid workspace size returned from cuSOLVER\n");
		goto cleanup;
	}
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_work, (size_t)lwork * sizeof(double)), cleanup);

	double beta_prev = 0.0;

#define H_T(i, j) h_T[(size_t)(j) * (size_t)ld + (size_t)(i)]

	for (int k = 0; k < max_iter - 1; k++) {
		double *v_k = d_V + (size_t)k * vec_stride;
		double *v_next = d_V + (size_t)(k + 1) * vec_stride;
		double *v_prev = (k > 0) ? (d_V + (size_t)(k - 1) * vec_stride) : NULL;

		CHECK_CUSPARSE_GOTO(cusparseDnVecSetValues(vecX, v_k), cleanup);
		CHECK_CUSPARSE_GOTO(cusparseDnVecSetValues(vecY, v_next), cleanup);
		CHECK_CUDA_GOTO(cudaMemset(v_next, 0, vec_stride * sizeof(double)), cleanup);
		CHECK_CUSPARSE_GOTO(cusparseSpMV(sp_handle,
										 CUSPARSE_OPERATION_NON_TRANSPOSE,
										 &alpha_spmv,
										 matA.descr,
										 vecX,
										 &beta_spmv,
										 vecY,
										 CUDA_R_64F,
										 CUSPARSE_SPMV_ALG_DEFAULT,
										 d_buffer), cleanup);

		double alpha = 0.0;
		CHECK_CUBLAS_GOTO(cublasDdot(blas_handle, mat_dim, v_k, 1, v_next, 1, &alpha), cleanup);
		H_T(k, k) = alpha;

		for (int i = 0; i < nth_eig; i++) {
			teval_last[i] = eigenvalues[i];
		}

		int current_dim = k + 1;
		size_t copy_cols = (size_t)current_dim;
		CHECK_CUDA_GOTO(cudaMemcpy(d_T,
								   h_T,
								   (size_t)ld * copy_cols * sizeof(double),
								   cudaMemcpyHostToDevice),
						cleanup);
		CHECK_CUSOLVER_GOTO(cusolverDnDsyevd(solver_handle,
											 CUSOLVER_EIG_MODE_VECTOR,
											 CUBLAS_FILL_MODE_UPPER,
											 current_dim,
											 d_T,
											 ld,
											 d_W,
											 d_work,
											 lwork,
											 d_info),
							cleanup);

		int info_host = 0;
		CHECK_CUDA_GOTO(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost), cleanup);
		if (info_host != 0) {
			printf("cuSOLVER Dsyevd failed with info = %d\n", info_host);
			goto cleanup;
		}

		CHECK_CUDA_GOTO(cudaMemcpy(eigenvalues,
								   d_W,
								   (size_t)current_dim * sizeof(double),
								   cudaMemcpyDeviceToHost),
						cleanup);

		bool converged = true;
		if (current_dim < nth_eig) {
			converged = false;
		} else {
			for (int i = 0; i < nth_eig; i++) {
				double diff = eigenvalues[i] - teval_last[i];
				if (diff * diff > threshold_sq) {
					converged = false;
					break;
				}
			}
		}

		if (converged) {
			printf("converged ");
			for (int i = 0; i < nth_eig; i++) {
				printf("%.1E ", teval_last[i] - eigenvalues[i]);
			}
			printf("\n");
			status = EXIT_SUCCESS;
			goto cleanup;
		}

		printf("%d\t", k + 1);
		for (int i = 0; i < nth_eig && i < current_dim; i++) {
			printf("%.7f\t", eigenvalues[i]);
		}
		printf("\n");

		double neg_alpha = -alpha;
		CHECK_CUBLAS_GOTO(cublasDaxpy(blas_handle, mat_dim, &neg_alpha, v_k, 1, v_next, 1), cleanup);
		if (k > 0) {
			double neg_beta_prev = -beta_prev;
			CHECK_CUBLAS_GOTO(cublasDaxpy(blas_handle, mat_dim, &neg_beta_prev, v_prev, 1, v_next, 1), cleanup);
		}

		if (k > 2) {
			for (int j = 0; j < k - 2; j++) {
				double *v_j = d_V + (size_t)j * vec_stride;
				double coeff = 0.0;
				CHECK_CUBLAS_GOTO(cublasDdot(blas_handle, mat_dim, v_j, 1, v_next, 1, &coeff), cleanup);
				double neg_coeff = -coeff;
				CHECK_CUBLAS_GOTO(cublasDaxpy(blas_handle, mat_dim, &neg_coeff, v_j, 1, v_next, 1), cleanup);
			}
		}

		double beta_next = 0.0;
		CHECK_CUBLAS_GOTO(cublasDnrm2(blas_handle, mat_dim, v_next, 1, &beta_next), cleanup);

		if (beta_next * beta_next < threshold_sq * threshold_sq) {
			printf("%.7f beta converged\n", beta_next);
			status = EXIT_SUCCESS;
			goto cleanup;
		}

		if (beta_next == 0.0) {
			printf("beta became zero\n");
			status = EXIT_SUCCESS;
			goto cleanup;
		}

		double inv_beta = 1.0 / beta_next;
		CHECK_CUBLAS_GOTO(cublasDscal(blas_handle, mat_dim, &inv_beta, v_next, 1), cleanup);

		if (k + 1 < ld) {
			H_T(k, k + 1) = beta_next;
			H_T(k + 1, k) = beta_next;
		}

		beta_prev = beta_next;
	}

	status = EXIT_SUCCESS;

cleanup:
	if (vecX != NULL) {
		cusparseDestroyDnVec(vecX);
	}
	if (vecY != NULL) {
		cusparseDestroyDnVec(vecY);
	}
	if (sp_handle != NULL) {
		cusparseDestroy(sp_handle);
	}
	if (blas_handle != NULL) {
		cublasDestroy(blas_handle);
	}
	if (solver_handle != NULL) {
		cusolverDnDestroy(solver_handle);
	}
	if (rng != NULL) {
		curandDestroyGenerator(rng);
	}
	if (d_buffer != NULL) {
		cudaFree(d_buffer);
	}
	if (d_V != NULL) {
		cudaFree(d_V);
	}
	if (d_T != NULL) {
		cudaFree(d_T);
	}
	if (d_W != NULL) {
		cudaFree(d_W);
	}
	if (d_work != NULL) {
		cudaFree(d_work);
	}
	if (d_info != NULL) {
		cudaFree(d_info);
	}
	if (d_random != NULL) {
		cudaFree(d_random);
	}
	free(h_T);
	free(teval_last);
	destroy_cusparse_matrix(&matA);
#undef H_T
	return status;
}
