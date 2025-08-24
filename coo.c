#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "coo.h"
#include "util.h"

void parse_to_coo(char *src_str, Coo *dist) {
	int row, column;
	double value;
	char *tok, *endptr;

	tok = strtok(src_str, " ");
	row = (int)strtol(tok, &endptr, 10);
	if (endptr == tok || *endptr != '\0') {
		fprintf(stderr, "Error: Invalid integer %s\n", tok);
		exit(EXIT_FAILURE);
	}
	
	tok = strtok(NULL, " ");
	column = (int)strtol(tok, &endptr, 10);
	if (endptr == tok || *endptr != '\0') {
		fprintf(stderr, "Error: Invalid integer %s\n", tok);
		exit(EXIT_FAILURE);
	}

	tok = strtok(NULL, " ");
	sscanf(tok, "%lg", &value);

	// fortran は 1-index, c は 0-index
	dist->index_row = row - 1;
	dist->index_column = column - 1;
	dist->value = value;
}

Mat_Coo read_mat_coo(char *filepath) {
	FILE *fp;
	char *line = NULL;
	size_t len;
	int read;

	const int buf_size = 1000;
	int entry_size = buf_size;
	int index = 0;
	int mat_dim = 0;
	Coo *entries, *tmp;
	
	entries = calloc(entry_size, sizeof(Coo));
	if (entries == NULL) {
		fprintf(stderr, "calloc failed\n");
		exit(1);
	}

	fp = fopen(filepath, "r");
	if (fp == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	
	while ((read = getline(&line, &len, fp)) != -1) {
		if (index >= entry_size) {
			entry_size += buf_size;
			tmp = realloc(entries, sizeof(Coo) * entry_size);
			if (tmp == NULL) {
				fprintf(stderr, "realloc failed\n");
				free(entries);
				exit(1);
			}
			entries = tmp;
		}
		
		parse_to_coo(line, &entries[index]);
		
		index++;
	}
	
	fclose(fp);
	free(line);

	for (int i = 0; i < index; i++) {
		mat_dim = max(mat_dim, entries[i].index_row);
		mat_dim = max(mat_dim, entries[i].index_column);
	}
	// 0-index to size
	mat_dim++;

	Mat_Coo ret = { entries, index, mat_dim };
	return ret;
}

double *matvec(Mat_Coo *mat, double vec[]) {
	double *vec_out = calloc(mat->dimension, sizeof(double));
	
	for (int i = 0; i < mat->length; i++) {
		Coo *entry = &(mat->data[i]);
		vec_out[entry->index_row] += entry->value * vec[entry->index_column];
	}

	return vec_out;
}

void lanczos(Mat_Coo *mat, double eigenvalues[], double *eigenvectors[], int nth_eig, int max_iter, double threshold) {
	int mat_dim = mat->dimension;
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
		v[k + 1] = matvec(mat, v[k]);
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
			return;
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
			return;
		}
		for (int i = 0; i < mat_dim; i++) {
			v[k + 1][i] /= beta;
		}
		tmat[k][k + 1] = beta;
		tmat[k + 1][k] = beta;
	}
}
