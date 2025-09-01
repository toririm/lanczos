#include "coo.h"

typedef struct crs {
    double  *values;
    int     *column_index;
    /**
     * 各行の最初の非ゼロ要素を value の index で格納
     * 長さは dimension + 1
     * row_head_indexes[dimension] = length
     */
    int     *row_head_indxes;
    /**
     * value, column_index の長さ
     */
    int     length;
    /**
     * 正方行列の次元数、行数
     */
    int     dimension;
} Mat_Crs;

extern Mat_Crs convert_from_coo(const Mat_Coo *mat_coo);
extern double *matvec_crs(const Mat_Crs *mat, const double vec[]);
