
#define max(x, y) (x >= y ? x : y)

extern void gaussian_random_vec(int n, double *r);
extern double dot_product(double *a, double *b, int size);
extern void diagonalize_double(double **symmetric_matrix, double *eigenvalues, double **eigenvectors, int n);
