#include <stdio.h>
#include <omp.h>

#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) <  (b) ? (a) : (b))
/**
 * 時間計測をするマクロ
 */
#define MEASURE(label, code) \
    do { \
        double __st = omp_get_wtime(); \
        code \
        double __en = omp_get_wtime(); \
        fprintf(stderr, "[TIME] %s: %lf sec\n", #label, __en - __st); \
    } while(0)

/**
 * 累積時間計測をするマクロ
 */
#define MEASURE_ACC(time_var, code) \
    do { \
        double __st = omp_get_wtime(); \
        code \
        double __en = omp_get_wtime(); \
        time_var += (__en - __st); \
    } while(0)

extern char *read_from_file(const char *filepath);
extern void gaussian_random_vec(int n, double *r);
extern double dot_product(const double *a, const double *b, int size);
extern void diagonalize_double(const double *symmetric_matrix, int ld,
                          double *eigenvalues, double *eigenvectors, int n,
                          double *work_nxn);
