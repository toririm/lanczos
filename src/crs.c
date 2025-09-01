#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "crs.h"

/**
 * 行,列の順についてソートされていることを前提とする
 */
Mat_Crs convert_from_coo(const Mat_Coo *mat_coo) {
    int dimension         = mat_coo->dimension;
    int length            = mat_coo->length;
    double *values        = malloc(sizeof(double) * length);
    int *column_index     = malloc(sizeof(int) * length);
    int *row_head_indexes = malloc(sizeof(int) * (dimension + 1));

    for (int i = 0; i < dimension + 1; i++)
        row_head_indexes[i] = length;

    int cur_row = -1;
    for (int i = 0; i < length; i++) {
        Coo *data = &mat_coo->data[i];
        int index_row = data->index_row;

        values[i]       = data->value;
        column_index[i] = data->index_column;

        if (cur_row < index_row) {
            row_head_indexes[index_row] = i;
            cur_row = index_row;
        }
    }

    Mat_Crs crs = {
        .values          = values,
        .column_index    = column_index,
        .row_head_indxes = row_head_indexes,
        .length          = length,
        .dimension       = dimension,
    };
    return crs;
}

void matvec_crs(const Mat_Crs *mat, const double vec[], double *dist) {
    int i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < mat->dimension; i++) {
        for (j = mat->row_head_indxes[i];
            j < mat->row_head_indxes[i + 1]; j++) {
                dist[i] += mat->values[j] * vec[mat->column_index[j]];
        }
    }
}
