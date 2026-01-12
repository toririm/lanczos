#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "crs.h"

/**
 * 行,列の順についてソートされていることを前提とする
 * swap_row_column > 0 で列優先でも動作する
 */
Mat_Crs convert_from_coo(const Mat_Coo *mat_coo, int swap_row_column) {
    size_t dimension         = mat_coo->dimension;
    size_t length            = mat_coo->length;
    double *values           = malloc(sizeof(double) * length);
    size_t *column_index     = malloc(sizeof(size_t) * length);
    size_t *row_head_indexes = malloc(sizeof(size_t) * (dimension + 1));
    size_t tmp;

    #pragma omp parallel for
    for (size_t i = 0; i < dimension + 1; i++)
        row_head_indexes[i] = length;

    size_t cur_row = -1;
    for (size_t i = 0; i < length; i++) {
        Coo *data = &mat_coo->data[i];
        size_t index_row    = data->index_row;
        size_t index_column = data->index_column;

        if (swap_row_column) {
            tmp = index_row;
            index_row = index_column;
            index_column = tmp;
        }

        values[i]       = data->value;
        column_index[i] = index_column;

        if (cur_row < index_row) {
            row_head_indexes[index_row] = i;
            cur_row = index_row;
        }
    }

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
    size_t i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < mat->dimension; i++) {
        for (j = mat->row_head_indexes[i];
            j < mat->row_head_indexes[i + 1]; j++) {
                dist[i] += mat->values[j] * vec[mat->column_index[j]];
        }
    }
}
