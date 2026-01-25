#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "util.h"
#include "crs.h"

Mat_Crs convert_from_coo(const Mat_Coo *mat_coo, int swap_row_column) {
    if (mat_coo == NULL || mat_coo->data == NULL) {
        fprintf(stderr, "convert_from_coo: mat_coo is NULL\n");
        exit(EXIT_FAILURE);
    }

    const size_t dimension = mat_coo->dimension;
    const size_t length = mat_coo->length;

    if (dimension == 0) {
        fprintf(stderr, "convert_from_coo: dimension is 0\n");
        exit(EXIT_FAILURE);
    }
    if (length > 0 && (length > (SIZE_MAX / sizeof(double)) || length > (SIZE_MAX / sizeof(size_t)))) {
        fprintf(stderr, "convert_from_coo: allocation size overflow (length=%zu)\n", length);
        exit(EXIT_FAILURE);
    }
    if (dimension + 1 < dimension || (dimension + 1) > (SIZE_MAX / sizeof(size_t))) {
        fprintf(stderr, "convert_from_coo: allocation size overflow (dimension=%zu)\n", dimension);
        exit(EXIT_FAILURE);
    }

    double *values = NULL;
    int64_t *column_index = NULL;
    int64_t *row_head_indexes = NULL;
    size_t *row_counts = NULL;
    size_t *next = NULL;

    if (length > 0) {
        values = (double *)malloc(sizeof(double) * length);
        column_index = (int64_t *)malloc(sizeof(int64_t) * length);
        if (values == NULL || column_index == NULL) {
            fprintf(stderr, "convert_from_coo: malloc failed (length=%zu)\n", length);
            free(values);
            free(column_index);
            exit(EXIT_FAILURE);
        }
    }

    row_head_indexes = (int64_t *)malloc(sizeof(int64_t) * (dimension + 1));
    row_counts = (size_t *)calloc(dimension, sizeof(size_t));
    if (row_head_indexes == NULL || row_counts == NULL) {
        fprintf(stderr, "convert_from_coo: allocation failed (dimension=%zu)\n", dimension);
        free(values);
        free(column_index);
        free(row_head_indexes);
        free(row_counts);
        exit(EXIT_FAILURE);
    }

    // 1st pass: count nnz per row (no sorting required)
    for (size_t i = 0; i < length; i++) {
        const Coo *data = &mat_coo->data[i];
        int r = data->index_row;
        int c = data->index_column;
        if (swap_row_column) {
            int tmp = r;
            r = c;
            c = tmp;
        }
        if (r < 0 || c < 0 || (size_t)r >= dimension || (size_t)c >= dimension) {
            fprintf(stderr,
                    "convert_from_coo: index out of range at entry %zu (row=%d, col=%d, dim=%zu)\n",
                    i, r, c, dimension);
            free(values);
            free(column_index);
            free(row_head_indexes);
            free(row_counts);
            exit(EXIT_FAILURE);
        }
        row_counts[(size_t)r]++;
    }

    // Build row pointers (prefix-sum). row_head_indexes[dimension] must be length.
    row_head_indexes[0] = 0;
    for (size_t i = 0; i < dimension; i++) {
        size_t next_val = row_head_indexes[i] + row_counts[i];
        if (next_val < row_head_indexes[i]) {
            fprintf(stderr, "convert_from_coo: prefix-sum overflow at row %zu\n", i);
            free(values);
            free(column_index);
            free(row_head_indexes);
            free(row_counts);
            exit(EXIT_FAILURE);
        }
        row_head_indexes[i + 1] = next_val;
    }
    if (row_head_indexes[dimension] != length) {
        fprintf(stderr,
                "convert_from_coo: internal error (row_ptr[dim]=%zu != length=%zu)\n",
                row_head_indexes[dimension], length);
        free(values);
        free(column_index);
        free(row_head_indexes);
        free(row_counts);
        exit(EXIT_FAILURE);
    }

    // 2nd pass: scatter entries into contiguous per-row segments
    next = (size_t *)malloc(sizeof(size_t) * dimension);
    if (next == NULL) {
        fprintf(stderr, "convert_from_coo: malloc failed (next buffer)\n");
        free(values);
        free(column_index);
        free(row_head_indexes);
        free(row_counts);
        exit(EXIT_FAILURE);
    }
    memcpy(next, row_head_indexes, sizeof(int64_t) * dimension);

    for (size_t i = 0; i < length; i++) {
        const Coo *data = &mat_coo->data[i];
        int r = data->index_row;
        int c = data->index_column;
        if (swap_row_column) {
            int tmp = r;
            r = c;
            c = tmp;
        }
        size_t row = (size_t)r;
        size_t pos = next[row]++;
        if (pos >= row_head_indexes[row + 1]) {
            fprintf(stderr, "convert_from_coo: write overflow at row %zu (pos=%zu)\n", row, pos);
            free(values);
            free(column_index);
            free(row_head_indexes);
            free(row_counts);
            free(next);
            exit(EXIT_FAILURE);
        }
        values[pos] = data->value;
        column_index[pos] = (size_t)c;
    }

    free(row_counts);
    free(next);

    Mat_Crs crs = {
        .values           = values,
        .column_index     = column_index,
        .row_head_indexes = row_head_indexes,
        .length           = length,
        .dimension        = dimension,
    };
    return crs;
}

void matvec_crs(const Mat_Crs *mat, const double vec[], double *dist) {
    if (mat == NULL || vec == NULL || dist == NULL) {
        fprintf(stderr, "matvec_crs: NULL argument\n");
        exit(EXIT_FAILURE);
    }

    /*
     * CRS SpMV is row-parallel and race-free. Keep optimization directive-only:
     * - avoid OpenMP overhead for tiny problems via if(...)
     * - make scheduling explicit (static tends to preserve locality)
     * - encourage SIMD on the inner reduction loop
     */
    #pragma omp parallel for schedule(guided) if(mat->dimension >= 1024)
    for (size_t i = 0; i < mat->dimension; i++) {
        double sum = 0.0;
        const size_t j0 = mat->row_head_indexes[i];
        const size_t j1 = mat->row_head_indexes[i + 1];
        #pragma omp simd reduction(+:sum)
        for (size_t j = j0; j < j1; j++) {
            sum += mat->values[j] * vec[mat->column_index[j]];
        }
        dist[i] = sum;
    }
}
