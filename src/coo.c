#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
	dist->index_row 	= row - 1;
	dist->index_column	= column - 1;
	dist->value 		= value;
}

Mat_Coo read_mat_coo(const char *filepath) {
	Coo *entries;
	int mat_dim = 0;
	
	char **lines, **tmp, *line, *file_content;
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
				tmp = realloc(lines, sizeof(char *) * allocated_lines);
				if (tmp == NULL) {
					fprintf(stderr, "realloc failed\n");
					free(file_content);
					free(lines);
					exit(EXIT_FAILURE);
				}
				lines = tmp;
			}
			lines[line_count] = line;
			line = strtok(NULL, "\n");
			line_count++;
		}
	);

	entries = malloc(sizeof(Coo) * line_count);

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

	Mat_Coo ret = {
		.data 		= entries,
		.length		= line_count,
		.dimension	= mat_dim,
	};
	return ret;
}

double *matvec_coo(const Mat_Coo *mat, const double vec[]) {
	double *vec_out = calloc(mat->dimension, sizeof(double));
	
	for (int i = 0; i < mat->length; i++) {
		Coo *entry = &(mat->data[i]);
		vec_out[entry->index_row] += entry->value * vec[entry->index_column];
	}

	return vec_out;
}

/**
 * 行, 列の優先順で比較する比較関数
 */
int compare_coo(const void *a, const void *b) {
	const Coo *mat_a = (Coo *)a;
	const Coo *mat_b = (Coo *)b;
	if (mat_a->index_row == mat_b->index_row && mat_a->index_column == mat_b->index_column) {
		return 0;
	}
	if (mat_a->index_row == mat_b->index_row) {
		return mat_a->index_column - mat_b->index_column;
	} else {
		return mat_a->index_row - mat_b->index_row;
	}
}

/**
 * mat を 行, 列の順でソートする
 */
void sort_matcoo(Mat_Coo *mat) {
	qsort(mat->data, mat->length, sizeof(Coo), compare_coo);
}

