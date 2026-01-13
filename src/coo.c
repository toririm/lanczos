#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
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

Mat_Coo read_mat_coo_incremental(const char *filepath) {
	FILE *fp;
	Coo *entries = NULL;
	size_t total_entries = 0;
	size_t line_count = 0;
	size_t mat_dim = 0;

	char chunk[BUFSIZ];
	char *linebuf = NULL;
	size_t linebuf_cap = 0;
	size_t linebuf_len = 0;

	fp = fopen(filepath, "r");
	if (fp == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	// First pass: count non-empty lines for single allocation
	while (fgets(chunk, sizeof(chunk), fp) != NULL) {
		size_t chunk_len = strlen(chunk);
		size_t needed = linebuf_len + chunk_len + 1;

		if (linebuf_cap < needed) {
			size_t new_cap = linebuf_cap;
			if (new_cap == 0) new_cap = BUFSIZ;
			while (new_cap < needed) new_cap += BUFSIZ;
			char *tmp = realloc(linebuf, new_cap);
			if (tmp == NULL) {
				fprintf(stderr, "realloc failed\n");
				free(entries);
				free(linebuf);
				fclose(fp);
				exit(EXIT_FAILURE);
			}
			linebuf = tmp;
			linebuf_cap = new_cap;
		}

		memcpy(linebuf + linebuf_len, chunk, chunk_len + 1);
		linebuf_len += chunk_len;

		if (chunk_len > 0 && chunk[chunk_len - 1] == '\n') {
			if (linebuf_len > 0 && linebuf[linebuf_len - 1] == '\n') {
				linebuf[--linebuf_len] = '\0';
			}

			// strtok("\n") と同様に、空行(長さ0)だけスキップ
			if (linebuf_len > 0) {
				total_entries++;
			}

			linebuf_len = 0;
			if (linebuf_cap > 0) linebuf[0] = '\0';
		}
	}

	if (ferror(fp)) {
		perror("fgets");
		free(entries);
		free(linebuf);
		fclose(fp);
		exit(EXIT_FAILURE);
	}

	// 改行なし最終行の処理
	if (linebuf_len > 0) {
		total_entries++;
	}

	fclose(fp);

	if (total_entries > 0 && total_entries > (SIZE_MAX / sizeof(Coo))) {
		fprintf(stderr, "Error: allocation size overflow\n");
		free(linebuf);
		exit(EXIT_FAILURE);
	}

	if (total_entries > 0) {
		entries = malloc(sizeof(Coo) * total_entries);
		if (entries == NULL) {
			fprintf(stderr, "malloc failed\n");
			free(linebuf);
			exit(EXIT_FAILURE);
		}
	}

	// Second pass: parse into allocated buffer
	fp = fopen(filepath, "r");
	if (fp == NULL) {
		perror("fopen");
		free(entries);
		free(linebuf);
		exit(EXIT_FAILURE);
	}

	linebuf_len = 0;
	if (linebuf_cap > 0) linebuf[0] = '\0';

	while (fgets(chunk, sizeof(chunk), fp) != NULL) {
		size_t chunk_len = strlen(chunk);
		size_t needed = linebuf_len + chunk_len + 1;

		if (linebuf_cap < needed) {
			size_t new_cap = linebuf_cap;
			if (new_cap == 0) new_cap = BUFSIZ;
			while (new_cap < needed) new_cap += BUFSIZ;
			char *tmp = realloc(linebuf, new_cap);
			if (tmp == NULL) {
				fprintf(stderr, "realloc failed\n");
				free(entries);
				free(linebuf);
				fclose(fp);
				exit(EXIT_FAILURE);
			}
			linebuf = tmp;
			linebuf_cap = new_cap;
		}

		memcpy(linebuf + linebuf_len, chunk, chunk_len + 1);
		linebuf_len += chunk_len;

		if (chunk_len > 0 && chunk[chunk_len - 1] == '\n') {
			if (linebuf_len > 0 && linebuf[linebuf_len - 1] == '\n') {
				linebuf[--linebuf_len] = '\0';
			}

			// strtok("\n") と同様に、空行(長さ0)だけスキップ
			if (linebuf_len > 0) {
				parse_to_coo(linebuf, &entries[line_count]);
				mat_dim = MAX(mat_dim, entries[line_count].index_row);
				mat_dim = MAX(mat_dim, entries[line_count].index_column);
				line_count++;
			}

			linebuf_len = 0;
			if (linebuf_cap > 0) linebuf[0] = '\0';
		}
	}

	if (ferror(fp)) {
		perror("fgets");
		free(entries);
		free(linebuf);
		fclose(fp);
		exit(EXIT_FAILURE);
	}

	// 改行なし最終行の処理
	if (linebuf_len > 0) {
		parse_to_coo(linebuf, &entries[line_count]);
		mat_dim = MAX(mat_dim, entries[line_count].index_row);
		mat_dim = MAX(mat_dim, entries[line_count].index_column);
		line_count++;
	}

	free(linebuf);
	fclose(fp);

	// 0-index to size
	mat_dim++;

	Mat_Coo ret = {
		.data 		= entries,
		.length		= line_count,
		.dimension	= mat_dim,
	};
	return ret;
}

void matvec_coo(const Mat_Coo *mat, const double vec[], double *dist) {
	if (mat == NULL || vec == NULL || dist == NULL) {
		fprintf(stderr, "matvec_coo: NULL argument\n");
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < mat->dimension; i++) {
		dist[i] = 0.0;
	}
	for (size_t i = 0; i < mat->length; i++) {
		Coo *entry = &(mat->data[i]);
		dist[entry->index_row] += entry->value * vec[entry->index_column];
	}
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
