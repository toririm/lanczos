#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mkl_lapacke.h>
#include "util.h"

void gaussian_random_vec(int n, double *r) {
  int iseed[4];
  srand((unsigned int)time(NULL));
  for (int i = 0; i < 4; i++) {
    iseed[i] = rand();
  }

  LAPACKE_dlarnv(3, iseed, n, r);
}

double dot_product(double *a, double *b, int size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void diagonalize_double(double **symmetric_matrix, double *eigenvalues, double **eigenvectors, int n) {
  double *u_flat = calloc(n * n, sizeof(double));
  
  if (u_flat == NULL) {
    fprintf(stderr, "Memory allocation failed.\n");
    exit(1);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      u_flat[i * n + j] = symmetric_matrix[i][j];
    }
  }
  
  int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, u_flat, n, eigenvalues);
  
  if (info != 0) {
    fprintf(stderr, "diagonalize error in diagonalize_double. info = %d\n", info);
    exit(1);
  }

  for (int j = 0; j < n; j++) {
    if (u_flat[j] < 0.0) {
      for (int i = 0; i < n; i++) {
        u_flat[i * n + j] *= -1.0;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      eigenvectors[i][j] = u_flat[i * n + j];
    }
  }
    
  free(u_flat);
}
