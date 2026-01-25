#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <omp.h>
#include "lanczos_cuda.h"

static inline size_t round_up_even(size_t value) {
	return (value % 2 == 0) ? value : value + 1;
}

typedef struct {
	const void *registered_values;
	size_t registered_values_bytes;
	int values_registered;
} HostCleanup;

static void host_cleanup_fn(void *data) {
	HostCleanup *cleanup = (HostCleanup*)data;
	if (cleanup == NULL) {
		return;
	}
	if (cleanup->values_registered && cleanup->registered_values != NULL) {
		cudaHostUnregister((void*)cleanup->registered_values);
	}
	free(cleanup);
}

int create_cusparse_matrix(const Mat_Crs *src, CuSparseMatrix *dist, cudaStream_t stream) {
	if (src == NULL || dist == NULL) {
		return EXIT_FAILURE;
	}

	memset(dist, 0, sizeof(*dist));

	double prof_t0 = 0.0;
	double prof_t1 = 0.0;
	double prof_host_register_col = 0.0;
	double prof_dev_alloc = 0.0;
	double prof_h2d_row = 0.0;
	double prof_h2d_col = 0.0;
	double prof_h2d_val = 0.0;
	double prof_host_register = 0.0;
	double prof_descr_create = 0.0;

	cudaEvent_t ev_a = NULL;
	cudaEvent_t ev_b = NULL;
	float ms_tmp = 0.0f;
	int profile_values_registered = 0;

	if (src->dimension > (size_t)INT_MAX) {
		printf("Matrix dimension too large for CUDA path (dim=%zu)\n", src->dimension);
		return EXIT_FAILURE;
	}
	if (src->length > (size_t)INT64_MAX) {
		printf("Matrix nnz too large for cuSPARSE (nnz=%zu)\n", src->length);
		return EXIT_FAILURE;
	}

	dist->rows = (int)src->dimension;
	dist->cols = (int)src->dimension;
	dist->nnz  = src->length;

	const size_t bytes_row_offsets = (size_t)(dist->rows + 1) * sizeof(int64_t);
	const size_t bytes_columns = (size_t)dist->nnz * sizeof(int64_t);
	const size_t bytes_values = (size_t)dist->nnz * sizeof(double);

	cudaError_t cuda_status;

	/* CUDA event timing for H2D copies (accurate, but synchronizes per copy) */
	cuda_status = cudaEventCreate(&ev_a);
	if (cuda_status != cudaSuccess) {
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}
	cuda_status = cudaEventCreate(&ev_b);
	if (cuda_status != cudaSuccess) {
		cudaEventDestroy(ev_a);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	prof_t0 = omp_get_wtime();
	cuda_status = cudaMalloc((void**) &dist->d_row_offsets,
							 (size_t)(dist->rows + 1) * sizeof(int64_t));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMalloc((void**) &dist->d_columns, dist->nnz * sizeof(int64_t));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
				__LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}

	cuda_status = cudaMalloc((void**) &dist->d_values, (size_t)dist->nnz * sizeof(double));
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
				__LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}
	prof_t1 = omp_get_wtime();
	prof_dev_alloc = prof_t1 - prof_t0;

	/* Row offsets H2D */
	CHECK_CUDA(cudaEventRecord(ev_a, stream));

	cuda_status = cudaMemcpyAsync(dist->d_row_offsets, src->row_head_indexes,
						  (size_t)(dist->rows + 1) * sizeof(int64_t),
						  cudaMemcpyHostToDevice,
						  stream);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
			   __LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}
	CHECK_CUDA(cudaEventRecord(ev_b, stream));
	CHECK_CUDA(cudaEventSynchronize(ev_b));
	CHECK_CUDA(cudaEventElapsedTime(&ms_tmp, ev_a, ev_b));
	prof_h2d_row = (double)ms_tmp * 1.0e-3;

	int values_registered = 0;
	prof_t0 = omp_get_wtime();
	CHECK_CUDA(cudaHostRegister((void*)src->column_index, dist->nnz * sizeof(int64_t), cudaHostRegisterDefault));
	prof_t1 = omp_get_wtime();
	prof_host_register_col = prof_t1 - prof_t0;
	
	/* Columns H2D */
	CHECK_CUDA(cudaEventRecord(ev_a, stream));
	cuda_status = cudaMemcpyAsync(dist->d_columns, src->column_index,
						dist->nnz * sizeof(int64_t),
						cudaMemcpyHostToDevice,
						stream);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		printf("CUDA API failed at line %d with error: %s (%d)\n",
				__LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}
	CHECK_CUDA(cudaEventRecord(ev_b, stream));
	CHECK_CUDA(cudaEventSynchronize(ev_b));
	CHECK_CUDA(cudaEventElapsedTime(&ms_tmp, ev_a, ev_b));
	prof_h2d_col = (double)ms_tmp * 1.0e-3;

	prof_t0 = omp_get_wtime();
	cuda_status = cudaHostRegister((void*)src->values, dist->nnz * sizeof(double), cudaHostRegisterDefault);
	if (cuda_status == cudaSuccess) {
		values_registered = 1;
	} else {
		values_registered = 0;
	}
	prof_t1 = omp_get_wtime();
	prof_host_register = prof_t1 - prof_t0;
	profile_values_registered = values_registered;

	/* Values H2D */
	CHECK_CUDA(cudaEventRecord(ev_a, stream));
	cuda_status = cudaMemcpyAsync(dist->d_values, src->values,
						dist->nnz * sizeof(double),
						cudaMemcpyHostToDevice,
						stream);
	if (cuda_status != cudaSuccess) {
		destroy_cusparse_matrix(dist);
		cudaEventDestroy(ev_a);
		cudaEventDestroy(ev_b);
		if (values_registered) {
			cudaHostUnregister((void*)src->values);
		}
		printf("CUDA API failed at line %d with error: %s (%d)\n",
				__LINE__, cudaGetErrorString(cuda_status), cuda_status);
		return EXIT_FAILURE;
	}
	CHECK_CUDA(cudaEventRecord(ev_b, stream));
	CHECK_CUDA(cudaEventSynchronize(ev_b));
	CHECK_CUDA(cudaEventElapsedTime(&ms_tmp, ev_a, ev_b));
	prof_h2d_val = (double)ms_tmp * 1.0e-3;

	HostCleanup *cleanup = (HostCleanup*)calloc(1, sizeof(*cleanup));
	if (cleanup == NULL) {
		/* Fallback: block and free safely */
		cudaStreamSynchronize(stream);
		if (values_registered) {
			cudaHostUnregister((void*)src->values);
		}
	} else {
		cleanup->registered_values = src->values;
		cleanup->registered_values_bytes = dist->nnz * sizeof(double);
		cleanup->values_registered = values_registered;
		cuda_status = cudaLaunchHostFunc(stream, host_cleanup_fn, cleanup);
		if (cuda_status != cudaSuccess) {
			/* Fallback: block and free safely */
			cudaStreamSynchronize(stream);
			host_cleanup_fn(cleanup);
		}
	}

	const int64_t nnz64 = (int64_t)dist->nnz;

	prof_t0 = omp_get_wtime();
	cusparseStatus_t sparse_status = cusparseCreateCsr(&dist->descr,
											  (int64_t)dist->rows,
											  (int64_t)dist->cols,
											   		   nnz64,
													   dist->d_row_offsets,
													   dist->d_columns,
													   dist->d_values,
											   		   CUSPARSE_INDEX_64I,
											   		   CUSPARSE_INDEX_64I,
													   CUSPARSE_INDEX_BASE_ZERO,
													   CUDA_R_64F);
	prof_t1 = omp_get_wtime();
	prof_descr_create = prof_t1 - prof_t0;
	if (sparse_status != CUSPARSE_STATUS_SUCCESS) {
		destroy_cusparse_matrix(dist);
						cudaEventDestroy(ev_a);
						cudaEventDestroy(ev_b);
		printf("CUSPARSE API failed at line %d with error code: %d\n",
			   __LINE__, sparse_status);
		return EXIT_FAILURE;
	}

	/* Profile print (stderr): single-line key=value for easier parsing */
	fprintf(stderr,
			"[CSR] rows=%d nnz=%zu bytes_row=%zu bytes_col=%zu bytes_val=%zu\n",
			dist->rows, dist->nnz, bytes_row_offsets, bytes_columns, bytes_values);
	fprintf(stderr, "[CSR] name=device_alloc sec=%.6f\n", prof_dev_alloc);
	fprintf(stderr, "[CSR] name=h2d_row_offsets sec=%.6f\n", prof_h2d_row);
	fprintf(stderr, "[CSR] name=host_register_columns sec=%.6f registered=%d\n", prof_host_register_col, 1);
	fprintf(stderr, "[CSR] name=h2d_columns sec=%.6f\n", prof_h2d_col);
	fprintf(stderr, "[CSR] name=host_register_values sec=%.6f registered=%d\n", prof_host_register, profile_values_registered);
	fprintf(stderr, "[CSR] name=h2d_values sec=%.6f\n", prof_h2d_val);
	fprintf(stderr, "[CSR] name=cusparseCreateCsr sec=%.6f\n", prof_descr_create);

	cudaEventDestroy(ev_a);
	cudaEventDestroy(ev_b);

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

int lanczos_cuda_crs(const Mat_Crs *mat,
			  double eigenvalues[], double *eigenvectors[],
			  int nth_eig, int max_iter, double threshold) {
	(void)eigenvectors;

	double init_cpu_t0 = 0.0;
	double init_cpu_t1 = 0.0;
	double init_cpu_stream_setup = 0.0;
	double init_cpu_host_alloc = 0.0;
	double init_cpu_handles = 0.0;
	double init_cpu_v_init = 0.0;
	double init_cpu_vec_desc = 0.0;
	double init_cpu_mat_upload = 0.0;
	double init_cpu_spmv_prep = 0.0;
	double init_cpu_solver_prep = 0.0;
	double init_cpu_rng_setup = 0.0;
	double init_cpu_warmup = 0.0;

	cudaEvent_t ev_v_init_start = NULL;
	cudaEvent_t ev_v_init_stop = NULL;
	cudaEvent_t ev_mat_upload_start = NULL;
	cudaEvent_t ev_mat_upload_stop = NULL;
	float ms_v_init = 0.0f;
	float ms_mat_upload = 0.0f;

	double time_memcpy = 0.0;
	double time_matvec = 0.0;
	double time_diag   = 0.0;
	double time_init   = 0.0;
	double time_reorth = 0.0;

	cudaStream_t stream = NULL;
	cudaEvent_t ev_start = NULL;
	cudaEvent_t ev_stop = NULL;
	double init_start = 0.0;

	if (mat == NULL || eigenvalues == NULL) {
		return EXIT_FAILURE;
	}
	if (mat->dimension > (size_t)INT_MAX) {
		printf("Matrix dimension too large for CUDA CRS (dim=%zu)\n", mat->dimension);
		return EXIT_FAILURE;
	}

	if (max_iter < 2) {
		printf("max_iter must be >= 2\n");
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
	CuSparseMatrix matA;
	memset(&matA, 0, sizeof(matA));

	double *h_T = NULL;
	int *h_info_pinned = NULL;
	double *h_eig_pinned = NULL;
	double *teval_last = NULL;

	const int ld = max_iter;
	size_t matrix_bytes = (size_t)ld * (size_t)ld * sizeof(double);

	double threshold_sq = threshold * threshold;

	memset(eigenvalues, 0, (size_t)nth_eig * sizeof(double));

	/* Warm up CUDA context (exclude from init timing/profiling). */
	init_cpu_t0 = omp_get_wtime();
	CHECK_CUDA_GOTO(cudaFree(0), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_warmup = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=cudaFree0_warmup sec=%.6f\n", init_cpu_warmup);

	init_start = omp_get_wtime();

	init_cpu_t0 = omp_get_wtime();
	CHECK_CUDA_GOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_start), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_stop), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_v_init_start), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_v_init_stop), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_mat_upload_start), cleanup);
	CHECK_CUDA_GOTO(cudaEventCreate(&ev_mat_upload_stop), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_stream_setup = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=stream_events_create sec=%.6f\n", init_cpu_stream_setup);

	init_cpu_t0 = omp_get_wtime();
	teval_last = (double*) calloc((size_t)nth_eig, sizeof(double));
	if (teval_last == NULL) {
		printf("Failed to allocate teval_last\n");
		goto cleanup;
	}

	CHECK_CUDA_GOTO(cudaHostAlloc((void**)&h_T, (size_t)ld * (size_t)ld * sizeof(double), cudaHostAllocDefault), cleanup);
	memset(h_T, 0, (size_t)ld * (size_t)ld * sizeof(double));

	CHECK_CUDA_GOTO(cudaHostAlloc((void**)&h_info_pinned, sizeof(int), cudaHostAllocDefault), cleanup);
	*h_info_pinned = 0;
	CHECK_CUDA_GOTO(cudaHostAlloc((void**)&h_eig_pinned, (size_t)nth_eig * sizeof(double), cudaHostAllocDefault), cleanup);
	memset(h_eig_pinned, 0, (size_t)nth_eig * sizeof(double));
	init_cpu_t1 = omp_get_wtime();
	init_cpu_host_alloc = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=host_alloc_pinned_small sec=%.6f\n", init_cpu_host_alloc);

	init_cpu_t0 = omp_get_wtime();
	CHECK_CURAND_GOTO(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT), cleanup);
	CHECK_CURAND_GOTO(curandSetPseudoRandomGeneratorSeed(rng, 123456789ULL), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_rng_setup = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=rng_setup sec=%.6f\n", init_cpu_rng_setup);

	init_cpu_t0 = omp_get_wtime();
	CHECK_CUSPARSE_GOTO(cusparseCreate(&sp_handle), cleanup);
	CHECK_CUBLAS_GOTO(cublasCreate(&blas_handle), cleanup);
	CHECK_CUSOLVER_GOTO(cusolverDnCreate(&solver_handle), cleanup);
	CHECK_CUSPARSE_GOTO(cusparseSetStream(sp_handle, stream), cleanup);
	CHECK_CUBLAS_GOTO(cublasSetStream(blas_handle, stream), cleanup);
	CHECK_CUSOLVER_GOTO(cusolverDnSetStream(solver_handle, stream), cleanup);
	CHECK_CURAND_GOTO(curandSetStream(rng, stream), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_handles = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=handles_setStream sec=%.6f\n", init_cpu_handles);

#define MEASURE_EVENT_ACC(time_var, code) \
	do { \
		CHECK_CUDA_GOTO(cudaEventRecord(ev_start, stream), cleanup); \
		code; \
		CHECK_CUDA_GOTO(cudaEventRecord(ev_stop, stream), cleanup); \
		CHECK_CUDA_GOTO(cudaEventSynchronize(ev_stop), cleanup); \
		float ms__ = 0.0f; \
		CHECK_CUDA_GOTO(cudaEventElapsedTime(&ms__, ev_start, ev_stop), cleanup); \
		time_var += (double)ms__ * 1.0e-3; \
	} while (0)

	init_cpu_t0 = omp_get_wtime();
	const size_t total_vec_elems = vec_stride * (size_t)max_iter;
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_V, total_vec_elems * sizeof(double)), cleanup);
	CHECK_CUDA_GOTO(cudaEventRecord(ev_v_init_start, stream), cleanup);
	CHECK_CUDA_GOTO(cudaMemsetAsync(d_V, 0, total_vec_elems * sizeof(double), stream), cleanup);

	const size_t padded_dim = round_up_even((size_t)mat_dim);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_random, padded_dim * sizeof(double)), cleanup);

	CHECK_CURAND_GOTO(curandGenerateNormalDouble(rng, d_random, padded_dim, 0.0, 1.0), cleanup);
	CHECK_CUDA_GOTO(cudaMemcpyAsync(d_V, d_random, (size_t)mat_dim * sizeof(double), cudaMemcpyDeviceToDevice, stream), cleanup);

	double norm = 0.0;
	CHECK_CUBLAS_GOTO(cublasDnrm2(blas_handle, mat_dim, d_V, 1, &norm), cleanup);
	if (norm == 0.0) {
		printf("Random vector has zero norm\n");
		goto cleanup;
	}
	double inv_norm = 1.0 / norm;
	CHECK_CUBLAS_GOTO(cublasDscal(blas_handle, mat_dim, &inv_norm, d_V, 1), cleanup);
	CHECK_CUDA_GOTO(cudaEventRecord(ev_v_init_stop, stream), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_v_init = init_cpu_t1 - init_cpu_t0;
	{
		const size_t bytes_dV = (size_t)mat_dim * (size_t)max_iter * sizeof(double);
		fprintf(stderr, "[INIT] name=v_init sec_cpu=%.6f bytes_dV=%zu\n", init_cpu_v_init, bytes_dV);
	}

	init_cpu_t0 = omp_get_wtime();
	CHECK_CUSPARSE_GOTO(cusparseCreateDnVec(&vecX, mat_dim, d_V, CUDA_R_64F), cleanup);
	CHECK_CUSPARSE_GOTO(cusparseCreateDnVec(&vecY, mat_dim, d_V + vec_stride, CUDA_R_64F), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_vec_desc = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=vec_descriptors sec=%.6f\n", init_cpu_vec_desc);

	init_cpu_t0 = omp_get_wtime();
	{
		const size_t bytes_row_offsets = (size_t)(mat_dim + 1) * sizeof(int64_t);
		const size_t bytes_cols = mat->length * sizeof(int64_t);
		const size_t bytes_vals = mat->length * sizeof(double);
		fprintf(stderr,
				"[INIT] name=matrix_upload_begin detail_ref=CSR nnz=%zu bytes_row=%zu bytes_col=%zu bytes_val=%zu\n",
				mat->length, bytes_row_offsets, bytes_cols, bytes_vals);
	}
	CHECK_CUDA_GOTO(cudaEventRecord(ev_mat_upload_start, stream), cleanup);
	if (create_cusparse_matrix(mat, &matA, stream) != EXIT_SUCCESS) {
		goto cleanup;
	}
	CHECK_CUDA_GOTO(cudaEventRecord(ev_mat_upload_stop, stream), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_mat_upload = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=matrix_upload_end detail_ref=CSR sec_cpu=%.6f\n", init_cpu_mat_upload);

	init_cpu_t0 = omp_get_wtime();
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
	printf("[SpMV] buffer_size=%zu bytes\n", buffer_size);
	if (buffer_size > 0) {
		CHECK_CUDA_GOTO(cudaMalloc(&d_buffer, buffer_size), cleanup);
	}
	init_cpu_t1 = omp_get_wtime();
	init_cpu_spmv_prep = init_cpu_t1 - init_cpu_t0;
	fprintf(stderr, "[INIT] name=spmv_prep sec=%.6f buffer_bytes=%zu\n", init_cpu_spmv_prep, buffer_size);

	init_cpu_t0 = omp_get_wtime();
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_T, matrix_bytes), cleanup);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_W, (size_t)ld * sizeof(double)), cleanup);
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_info, sizeof(int)), cleanup);

	int meig = 0;
	const int il = 1;
	int iu_max = nth_eig;

	int lwork = 0;
	CHECK_CUSOLVER_GOTO(cusolverDnDsyevdx_bufferSize(solver_handle,
													CUSOLVER_EIG_MODE_NOVECTOR,
													CUSOLVER_EIG_RANGE_I,
													CUBLAS_FILL_MODE_UPPER,
													ld,
													(const double *)d_T,
													ld,
													0.0,
													0.0,
													il,
													iu_max,
													&meig,
													(const double *)d_W,
													&lwork),
										cleanup);
	if (lwork <= 0) {
		printf("Invalid workspace size returned from cuSOLVER\n");
		goto cleanup;
	}
	CHECK_CUDA_GOTO(cudaMalloc((void**) &d_work, (size_t)lwork * sizeof(double)), cleanup);
	init_cpu_t1 = omp_get_wtime();
	init_cpu_solver_prep = init_cpu_t1 - init_cpu_t0;
	{
		const size_t bytes_hT = (size_t)ld * (size_t)ld * sizeof(double);
		const size_t bytes_dT = matrix_bytes;
		const size_t bytes_dwork = (size_t)lwork * sizeof(double);
		fprintf(stderr,
				"[INIT] name=solver_prep sec=%.6f bytes_hT=%zu bytes_dT=%zu bytes_dwork=%zu lwork=%d\n",
				init_cpu_solver_prep, bytes_hT, bytes_dT, bytes_dwork, lwork);
	}

	CHECK_CUDA_GOTO(cudaStreamSynchronize(stream), cleanup);
	if (ev_v_init_start != NULL && ev_v_init_stop != NULL) {
		(void)cudaEventElapsedTime(&ms_v_init, ev_v_init_start, ev_v_init_stop);
	}
	if (ev_mat_upload_start != NULL && ev_mat_upload_stop != NULL) {
		(void)cudaEventElapsedTime(&ms_mat_upload, ev_mat_upload_start, ev_mat_upload_stop);
	}
	fprintf(stderr, "[INIT] name=v_init_gpu sec=%.6f\n", (double)ms_v_init * 1.0e-3);
	fprintf(stderr, "[INIT] name=matrix_upload_gpu detail_ref=CSR sec=%.6f\n", (double)ms_mat_upload * 1.0e-3);
	if (init_start != 0.0) {
		time_init = omp_get_wtime() - init_start;
	}
	fprintf(stderr, "[INIT] name=total_wall sec=%.6f\n", time_init);

	double beta_prev = 0.0;

#define H_T(i, j) h_T[(size_t)(j) * (size_t)ld + (size_t)(i)]

	for (int k = 0; k < max_iter - 1; k++) {
		double *v_k = d_V + (size_t)k * vec_stride;
		double *v_next = d_V + (size_t)(k + 1) * vec_stride;
		double *v_prev = (k > 0) ? (d_V + (size_t)(k - 1) * vec_stride) : NULL;

		CHECK_CUSPARSE_GOTO(cusparseDnVecSetValues(vecX, v_k), cleanup);
		CHECK_CUSPARSE_GOTO(cusparseDnVecSetValues(vecY, v_next), cleanup);
		CHECK_CUDA_GOTO(cudaMemsetAsync(v_next, 0, vec_stride * sizeof(double), stream), cleanup);

		MEASURE_EVENT_ACC(time_matvec,
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
		);

		double alpha = 0.0;
		CHECK_CUBLAS_GOTO(cublasDdot(blas_handle, mat_dim, v_k, 1, v_next, 1, &alpha), cleanup);
		H_T(k, k) = alpha;

		for (int i = 0; i < nth_eig; i++) {
			teval_last[i] = eigenvalues[i];
		}

		int current_dim = k + 1;
		int iu = nth_eig;
		if (iu < 1) {
			iu = 1;
		}
		if (iu > current_dim) {
			iu = current_dim;
		}

		MEASURE_EVENT_ACC(time_memcpy,
		size_t copy_cols = (size_t)current_dim;
		CHECK_CUDA_GOTO(cudaMemcpyAsync(d_T,
								   		h_T,
								   		(size_t)ld * copy_cols * sizeof(double),
							   			cudaMemcpyHostToDevice,
							   			stream),
						cleanup);
		);

		// T_(k,k) の固有値のみ（下位 iu 個）を計算している
		MEASURE_EVENT_ACC(time_diag,
		CHECK_CUSOLVER_GOTO(cusolverDnDsyevdx(solver_handle,
											 CUSOLVER_EIG_MODE_NOVECTOR,
											 CUSOLVER_EIG_RANGE_I,
											 CUBLAS_FILL_MODE_UPPER,
											 current_dim,
											 d_T,
											 ld,
											 0.0,
											 0.0,
											 il,
											 iu,
											 &meig,
											 d_W,
											 d_work,
											 lwork,
											 d_info),
							cleanup);
		);

		MEASURE_EVENT_ACC(time_memcpy,
			CHECK_CUDA_GOTO(cudaMemcpyAsync(h_info_pinned, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream), cleanup);
			CHECK_CUDA_GOTO(cudaMemcpyAsync(h_eig_pinned,
							   d_W,
							   (size_t)iu * sizeof(double),
							   cudaMemcpyDeviceToHost,
							   stream),
					cleanup);
		);
		if (*h_info_pinned != 0) {
			printf("cuSOLVER Dsyevdx failed with info = %d\n", *h_info_pinned);
			goto cleanup;
		}
		memcpy(eigenvalues, h_eig_pinned, (size_t)iu * sizeof(double));

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
		MEASURE_EVENT_ACC(time_reorth,
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
		);

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

	printf("Reached max iterations (%d)\n", max_iter);

cleanup:
	if (ev_start != NULL) {
		cudaEventDestroy(ev_start);
	}
	if (ev_stop != NULL) {
		cudaEventDestroy(ev_stop);
	}
	if (stream != NULL) {
		cudaStreamDestroy(stream);
	}
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
	if (h_T != NULL) {
		cudaFreeHost(h_T);
	}
	if (h_info_pinned != NULL) {
		cudaFreeHost(h_info_pinned);
	}
	if (h_eig_pinned != NULL) {
		cudaFreeHost(h_eig_pinned);
	}
	free(teval_last);
	destroy_cusparse_matrix(&matA);
#undef H_T
	#undef MEASURE_EVENT_ACC
	fprintf(stderr, "[TOTAL] name=init detail_ref=INIT sec=%.6f\n", time_init);
	fprintf(stderr, "[TOTAL] name=cudaMemcpy sec=%.6f\n", time_memcpy);
	fprintf(stderr, "[TOTAL] name=spmv sec=%.6f\n", time_matvec);
	fprintf(stderr, "[TOTAL] name=diagonalization sec=%.6f\n", time_diag);
	fprintf(stderr, "[TOTAL] name=reorthogonalization sec=%.6f\n", time_reorth);
	return status;
}
