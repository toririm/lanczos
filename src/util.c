#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mkl_lapacke.h>
#include "util.h"

char *read_from_file(const char *filepath) {
  FILE *fp;
  char *head, *tmp;
  size_t read_bytes, total_bytes = 0;

  fp = fopen(filepath, "r");
  if (fp == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  head = NULL;

  read_bytes = -1;
  while (read_bytes != 0) {
    tmp = realloc(head, total_bytes + BUFSIZ + 1);
    if (tmp == NULL) {
      fprintf(stderr, "realloc failed\n");
      free(head);
      fclose(fp);
      exit(EXIT_FAILURE);
    }
    head = tmp;

    read_bytes = fread(head + total_bytes, sizeof(char), BUFSIZ, fp);
    if (ferror(fp)) {
      perror("fread");
      free(head);
      fclose(fp);
      exit(EXIT_FAILURE);
    }
    total_bytes += read_bytes;
  }
  head[total_bytes] = '\0';

  return head;
}

void gaussian_random_vec(int n, double *r) {
  int iseed[4];
  srand((unsigned int)time(NULL));
  for (int i = 0; i < 4; i++) {
    iseed[i] = rand();
  }

  LAPACKE_dlarnv(3, iseed, n, r);
}

double dot_product(const double *a, const double *b, int size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void diagonalize_double(const double *symmetric_matrix, int ld,
						  double *eigenvalues, double *eigenvectors, int n) {
	if (symmetric_matrix == NULL || eigenvalues == NULL || eigenvectors == NULL) {
		fprintf(stderr, "diagonalize_double: NULL argument\n");
		exit(1);
	}
	if (n <= 0 || ld < n) {
		fprintf(stderr, "diagonalize_double: invalid dimensions (n=%d, ld=%d)\n", n, ld);
		exit(1);
	}

	double *u_flat = calloc((size_t)n * (size_t)n, sizeof(double));
  
  if (u_flat == NULL) {
    fprintf(stderr, "Memory allocation failed.\n");
    exit(1);
  }

  // Copy top-left n x n from column-major symmetric_matrix (ld leading dim)
  const size_t col_bytes_in = (size_t)n * sizeof(double);
  for (int j = 0; j < n; j++) {
    const double *src_col = symmetric_matrix + (size_t)j * (size_t)ld;
    double *dst_col = u_flat + (size_t)j * (size_t)n;
    memcpy(dst_col, src_col, col_bytes_in);
  }
  
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, u_flat, n, eigenvalues);
  
  if (info != 0) {
    fprintf(stderr, "diagonalize error in diagonalize_double. info = %d\n", info);
  	free(u_flat);
    exit(1);
  }

  for (int j = 0; j < n; j++) {
    if (u_flat[(size_t)j * (size_t)n] < 0.0) {
      for (int i = 0; i < n; i++) {
        u_flat[(size_t)j * (size_t)n + (size_t)i] *= -1.0;
      }
    }
  }

  // Write eigenvectors into column-major output with leading dimension ld
  const size_t col_bytes_out = (size_t)n * sizeof(double);
  for (int j = 0; j < n; j++) {
    memcpy(eigenvectors + (size_t)j * (size_t)ld,
           u_flat + (size_t)j * (size_t)n,
           col_bytes_out);
  }
    
  free(u_flat);
}
