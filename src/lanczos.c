#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "util.h"
#include "lanczos.h"

void lanczos(const Mat_Matvec mat_matvec,
			 double eigenvalues[], double *eigenvectors[],
			 int nth_eig, int max_iter, double threshold) {
    const void *mat = mat_matvec.mat;
    Matvec_General *matvec = mat_matvec.matvec;
    const int mat_dim = mat_matvec.dimension;
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
	for (int i = 0; i < nth_eig; i++) {
		eigenvalues[i] = 0.0;
	}

	/* 0-based indexing: v[0] is the initial vector */
	gaussian_random_vec(mat_dim, v[0]);

	norm = sqrt(dot_product(v[0], v[0], mat_dim));
	for (int i = 0; i < mat_dim; i++) {
		v[0][i] = 1.0 / norm * v[0][i];
	}

	for (int k = 0; k < max_iter - 1; k++) {
		int eval_count = k + 1;
		if (eval_count > nth_eig) {
			eval_count = nth_eig;
		}
		matvec(mat, v[k], v[k + 1]);
		alpha = dot_product(v[k], v[k + 1], mat_dim);
		tmat[k][k] = alpha;
		for (int i = 0; i < eval_count; i++) {
			teval_last[i] = eigenvalues[i];
		}
		diagonalize_double(tmat, eigenvalues, eigenvectors, k + 1);
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
			return;
		}
		printf("%d\t", k + 1);
		for (int i = 0; i < nth_eig; i++) {
			printf("%.7f\t", eigenvalues[i]);
		}
		printf("\n");

		for (int i = 0; i < mat_dim; i++) {
			double prev = (k == 0) ? 0.0 : v[k - 1][i];
			v[k + 1][i] = v[k + 1][i] - beta * prev - alpha * v[k][i];
		}
		for (int l = 0; l < k; l++) { /* reorthogonalize to all previous basis */
			double coeff = dot_product(v[l], v[k + 1], mat_dim);
			for (int i = 0; i < mat_dim; i++) {
				v[k + 1][i] -= v[l][i] * coeff;
			}
		}
		beta = sqrt(dot_product(v[k + 1], v[k + 1], mat_dim));
		if (beta * beta < threshold * threshold * threshold * threshold) {
			printf("%.7f beta converged\n", beta);
			return;
		}
		for (int i = 0; i < mat_dim; i++) {
			v[k + 1][i] /= beta;
		}
		tmat[k][k + 1] = beta;
		tmat[k + 1][k] = beta;
	}
}
