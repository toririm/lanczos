#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include "util.h"
#include "lanczos.h"

void lanczos(const Mat_Matvec mat_matvec,
			 double eigenvalues[], double *eigenvectors[],
			 int nth_eig, int max_iter, double threshold) {
    const void *mat = mat_matvec.mat;
    Matvec_General *matvec = mat_matvec.matvec;
    const size_t mat_dim = mat_matvec.dimension;
	double *v = NULL;
	double *tmat = NULL;
	double *eigenvectors_work = NULL;
	double *teval_last = NULL;
	double norm, alpha, beta = 0;
	(void)eigenvectors; /* CPU path currently does not return eigenvectors */

	double time_init   = 0.0;
	double time_matvec = 0.0;
	double time_diag   = 0.0;
	double time_reorth = 0.0;

	double __st = omp_get_wtime();
	if (mat == NULL || matvec == NULL || eigenvalues == NULL) {
		fprintf(stderr, "lanczos: NULL argument\n");
		exit(EXIT_FAILURE);
	}
	if (max_iter < 2 || nth_eig < 1) {
		fprintf(stderr, "lanczos: invalid parameters (nth_eig=%d, max_iter=%d)\n", nth_eig, max_iter);
		exit(EXIT_FAILURE);
	}
	if (mat_dim == 0) {
		fprintf(stderr, "lanczos: mat_dim is 0\n");
		exit(EXIT_FAILURE);
	}
	if (mat_dim > (size_t)INT_MAX) {
		fprintf(stderr, "lanczos: mat_dim=%zu exceeds INT_MAX (util routines use int)\n", mat_dim);
		exit(EXIT_FAILURE);
	}

	const int ld = max_iter;
	const int mat_dim_i = (int)mat_dim;

	if (mat_dim != 0 && (size_t)max_iter > SIZE_MAX / mat_dim) {
		fprintf(stderr, "lanczos: allocation size overflow (v)\n");
		exit(EXIT_FAILURE);
	}
	if ((size_t)ld > 0 && (size_t)ld > SIZE_MAX / (size_t)ld) {
		fprintf(stderr, "lanczos: allocation size overflow (ld)\n");
		exit(EXIT_FAILURE);
	}

	const size_t v_elems = (size_t)max_iter * mat_dim;
	const size_t t_elems = (size_t)ld * (size_t)ld;

	v = calloc(v_elems, sizeof(double));
	if (v == NULL) {
		fprintf(stderr, "lanczos: allocation failed (v, elems=%zu)\n", v_elems);
		exit(EXIT_FAILURE);
	}
	tmat = calloc(t_elems, sizeof(double));
	if (tmat == NULL) {
		fprintf(stderr, "lanczos: allocation failed (tmat, elems=%zu)\n", t_elems);
		goto cleanup;
	}
	/* eigenvectors of (k+1)x(k+1) where (k+1)<=max_iter; keep workspace max_iter^2 */
	eigenvectors_work = calloc(t_elems, sizeof(double));
	if (eigenvectors_work == NULL) {
		fprintf(stderr, "lanczos: allocation failed (eigenvectors_work, elems=%zu)\n", t_elems);
		goto cleanup;
	}


#define V(k, i) v[(size_t)(k) * mat_dim + (size_t)(i)]
#define T(i, j) tmat[(size_t)(j) * (size_t)ld + (size_t)(i)]
#define E(i, j) eigenvectors_work[(size_t)(j) * (size_t)ld + (size_t)(i)]

	teval_last = calloc((size_t)nth_eig, sizeof(double));
	if (teval_last == NULL) {
		fprintf(stderr, "lanczos: allocation failed (teval_last)\n");
		goto cleanup;
	}
	for (int i = 0; i < nth_eig; i++) {
		eigenvalues[i] = 0.0;
	}

	/* 0-based indexing: v[0] is the initial vector */
	gaussian_random_vec(mat_dim_i, &V(0, 0));

	norm = sqrt(dot_product(&V(0, 0), &V(0, 0), mat_dim_i));
	for (int i = 0; i < mat_dim_i; i++) {
		V(0, i) = (1.0 / norm) * V(0, i);
	}

	double __et = omp_get_wtime();

	time_init = __et - __st;

	for (int k = 0; k < max_iter - 1; k++) {
		int eval_count = k + 1;
		if (eval_count > nth_eig) {
			eval_count = nth_eig;
		}
		MEASURE_ACC(time_matvec,
		matvec(mat, &V(k, 0), &V(k + 1, 0));
		);
		alpha = dot_product(&V(k, 0), &V(k + 1, 0), mat_dim_i);
		T(k, k) = alpha;
		for (int i = 0; i < nth_eig; i++) {
			teval_last[i] = eigenvalues[i];
		}
		MEASURE_ACC(time_diag,
		diagonalize_double(tmat, ld, eigenvalues, eigenvectors_work, k + 1);
		);
		bool all = true;
		for (int i = eval_count; i < nth_eig; i++) {
			eigenvalues[i] = 0.0; /* pad unused slots for display */
		}
		for (int i = 0; i < eval_count; i++) {
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
			goto cleanup;
		}
		printf("%d\t", k + 1);
		for (int i = 0; i < nth_eig; i++) {
			printf("%.7f\t", eigenvalues[i]);
		}
		printf("\n");

		double __st_reorth = omp_get_wtime();
		#pragma omp parallel for
		for (size_t i = 0; i < mat_dim; i++) {
			double prev = (k == 0) ? 0.0 : V(k - 1, i);
			V(k + 1, i) = V(k + 1, i) - beta * prev - alpha * V(k, i);
		}
		for (int l = 0; l < k; l++) { /* reorthogonalize to all previous basis */
			double coeff = dot_product(&V(l, 0), &V(k + 1, 0), mat_dim_i);
			#pragma omp parallel for
			for (int i = 0; i < mat_dim_i; i++) {
				V(k + 1, i) -= V(l, i) * coeff;
			}
		}
		double __et_reorth = omp_get_wtime();
		time_reorth += __et_reorth - __st_reorth;
		beta = sqrt(dot_product(&V(k + 1, 0), &V(k + 1, 0), mat_dim_i));
		if (beta * beta < threshold * threshold * threshold * threshold) {
			printf("%.7f beta converged\n", beta);
			goto cleanup;
		}
		#pragma omp parallel for
		for (size_t i = 0; i < mat_dim; i++) {
			V(k + 1, i) /= beta;
		}
		T(k, k + 1) = beta;
		T(k + 1, k) = beta;
	}

cleanup:
	if (teval_last != NULL) {
		free(teval_last);
	}
	if (eigenvectors_work != NULL) {
		free(eigenvectors_work);
	}
	if (tmat != NULL) {
		free(tmat);
	}
	if (v != NULL) {
		free(v);
	}
	fprintf(stderr, "Time spent in init:      %.6f sec\n", time_init);
	fprintf(stderr, "Time spent in SpMV:      %.6f sec\n", time_matvec);
	fprintf(stderr, "Time spent in diagonalization: %.6f sec\n", time_diag);
	fprintf(stderr, "Time spent in reorthogonalization: %.6f sec\n", time_reorth);
}
