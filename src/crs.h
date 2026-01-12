#include "coo.h"
#pragma once

typedef struct crs {
    double  *values;
    size_t     *column_index;
    /**
     * 各行の最初の非ゼロ要素を value の index で格納
     * 長さは dimension + 1
     * row_head_indexes[dimension] = length
     */
    size_t     *row_head_indexes;
    /**
     * value, column_index の長さ
     */
    size_t     length;
    /**
     * 正方行列の次元数、行数
     */
    size_t     dimension;
} Mat_Crs;

extern Mat_Crs convert_from_coo(const Mat_Coo *mat_coo, int swap_row_column);
extern void matvec_crs(const Mat_Crs *mat, const double vec[], double *dist);
