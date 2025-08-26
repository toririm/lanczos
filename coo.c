#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h>
#include "coo.h"
#include "util.h"

void parse_to_coo(char *src_line, Coo *dist) {
	int row, column;
	double value;
	char *cur, *endptr;
	cur = src_line;

	while (isspace(*cur)) cur++;
	row = (int)strtol(cur, &endptr, 10);
	if (endptr == cur) {
		fprintf(stderr, "Error: Invalid integer %s\n", cur);
		exit(EXIT_FAILURE);
	}
	cur = endptr;
	
	while (isspace(*cur)) cur++;
	column = (int)strtol(cur, &endptr, 10);
	if (endptr == cur) {
		fprintf(stderr, "Error: Invalid integer %s\n", cur);
		exit(EXIT_FAILURE);
	}
	cur = endptr;

	while (isspace(*cur)) cur++;
	sscanf(cur, "%lg", &value);

	// fortran は 1-index, c は 0-index
	dist->index_row = row - 1;
	dist->index_column = column - 1;
	dist->value = value;
}

Mat_Coo read_mat_coo(char *filepath) {
	Coo *entries;
	int mat_dim = 0;
	
	char **lines, *line, *file_content;
	int allocated_lines = 0;
	int line_count = 0;

	MEASURE(readfile,
		file_content = read_from_file(filepath);
	);
	
	MEASURE(count_line,
	lines = NULL;
	line = strtok(file_content, "\n");
	while (line) {
		if (allocated_lines <= line_count) {
			allocated_lines += BUFSIZ;
			lines = realloc(lines, sizeof(char *) * allocated_lines);
			if (lines == NULL) {
				fprintf(stderr, "realloc failed\n");
				free(file_content);
				exit(EXIT_FAILURE);
			}
		}
		lines[line_count] = line;
		line = strtok(NULL, "\n");
		line_count++;
	}

	entries = malloc(sizeof(Coo) * line_count);
	);

	#pragma omp parallel for
	for (int i = 0; i < line_count; i++) {
		parse_to_coo(lines[i], &entries[i]);
	}

	free(file_content);
	free(lines);

	
	for (int i = 0; i < line_count; i++) {
		mat_dim = MAX(mat_dim, entries[i].index_row);
		mat_dim = MAX(mat_dim, entries[i].index_column);
	}
	// 0-index to size
	mat_dim++;

	Mat_Coo ret = { entries, line_count, mat_dim };
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
