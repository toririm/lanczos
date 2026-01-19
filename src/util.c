#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
  #pragma omp simd reduction(+:sum)
  for (int i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

void diagonalize_double(const double *symmetric_matrix, int ld,
              double *eigenvalues, double *eigenvectors, int n,
              double *work_nxn) {
  if (symmetric_matrix == NULL || eigenvalues == NULL || work_nxn == NULL) {
    fprintf(stderr, "diagonalize_double_ws: NULL argument\n");
    exit(1);
  }
  if (n <= 0 || ld < n) {
    fprintf(stderr, "diagonalize_double_ws: invalid dimensions (n=%d, ld=%d)\n", n, ld);
    exit(1);
  }

  double *u_flat = work_nxn;

  // Copy top-left n x n from column-major symmetric_matrix (ld leading dim)
  const size_t col_bytes_in = (size_t)n * sizeof(double);
  for (int j = 0; j < n; j++) {
    const double *src_col = symmetric_matrix + (size_t)j * (size_t)ld;
    double *dst_col = u_flat + (size_t)j * (size_t)n;
    memcpy(dst_col, src_col, col_bytes_in);
  }

  const char jobz = (eigenvectors != NULL) ? 'V' : 'N';
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, 'U', n, u_flat, n, eigenvalues);
  if (info != 0) {
    fprintf(stderr, "diagonalize error in diagonalize_double. info = %d\n", info);
    exit(1);
  }

  if (eigenvectors == NULL) {
    return;
  }

  for (int j = 0; j < n; j++) {
    if (u_flat[(size_t)j * (size_t)n] < 0.0) {
      #pragma omp simd
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
}

void diagonalize_double_partial(const double *symmetric_matrix, int ld,
              double *eigenvalues, int n, int nth_eig,
              double *work_nxn) {
  if (symmetric_matrix == NULL || eigenvalues == NULL || work_nxn == NULL) {
    fprintf(stderr, "diagonalize_double_partial: NULL argument\n");
    exit(1);
  }
  if (n <= 0 || ld < n) {
    fprintf(stderr, "diagonalize_double_partial: invalid dimensions (n=%d, ld=%d)\n", n, ld);
    exit(1);
  }
  if (nth_eig < 1) {
    fprintf(stderr, "diagonalize_double_partial: invalid nth_eig=%d\n", nth_eig);
    exit(1);
  }

  const lapack_int n_lap = (lapack_int)n;
  lapack_int iu = (lapack_int)nth_eig;
  if (iu > n_lap) {
    iu = n_lap;
  }
  if (iu < 1) {
    iu = 1;
  }

  double *a = work_nxn;

  // Copy top-left n x n from column-major symmetric_matrix (ld leading dim)
  const size_t col_bytes = (size_t)n * sizeof(double);
  for (int j = 0; j < n; j++) {
    const double *src_col = symmetric_matrix + (size_t)j * (size_t)ld;
    double *dst_col = a + (size_t)j * (size_t)n;
    memcpy(dst_col, src_col, col_bytes);
  }

  // Cache LAPACKE work buffers to avoid alloc/free in every iteration.
  static lapack_int *iwork = NULL;
  static lapack_int *isuppz = NULL;
  static double *work = NULL;
  static lapack_int lwork = 0;
  static lapack_int liwork = 0;
  static lapack_int n_cached = 0;
  static lapack_int iu_cached = 0;

  const bool need_resize = (n_lap > n_cached) || (iu > iu_cached) || (work == NULL) || (iwork == NULL) || (isuppz == NULL);
  if (need_resize) {
    lapack_int n_query = n_lap;
    lapack_int iu_query = iu;
    if (iu_query < 1) {
      iu_query = 1;
    }
    if (iu_query > n_query) {
      iu_query = n_query;
    }

    if (n_query > n_cached) {
      lapack_int *new_isuppz = (lapack_int *)realloc(isuppz, (size_t)2 * (size_t)n_query * sizeof(lapack_int));
      if (new_isuppz == NULL) {
        fprintf(stderr, "diagonalize_double_partial: realloc failed (isuppz)\n");
        exit(1);
      }
      isuppz = new_isuppz;
    }

    double work_query = 0.0;
    lapack_int iwork_query = 0;
    lapack_int m_query = 0;
    double z_dummy = 0.0;
    lapack_int info_q = LAPACKE_dsyevr_work(LAPACK_COL_MAJOR,
                                           'N',
                                           'I',
                                           'U',
                                           n_query,
                                           a,
                                           n_query,
                                           0.0,
                                           0.0,
                                           1,
                                           iu_query,
                                           0.0,
                                           &m_query,
                                           eigenvalues,
                                           &z_dummy,
                                           1,
                                           isuppz,
                                           &work_query,
                                           -1,
                                           &iwork_query,
                                           -1);
    if (info_q != 0) {
      fprintf(stderr, "diagonalize_double_partial: LAPACKE_dsyevr_work query failed (info=%d)\n", (int)info_q);
      exit(1);
    }

    lapack_int new_lwork = (lapack_int)work_query;
    if (new_lwork < 1) {
      new_lwork = 1;
    }
    lapack_int new_liwork = iwork_query;
    if (new_liwork < 1) {
      new_liwork = 1;
    }

    double *new_work = (double *)realloc(work, (size_t)new_lwork * sizeof(double));
    lapack_int *new_iwork = (lapack_int *)realloc(iwork, (size_t)new_liwork * sizeof(lapack_int));
    if (new_work == NULL || new_iwork == NULL) {
      fprintf(stderr, "diagonalize_double_partial: realloc failed (work buffers)\n");
      free(new_work);
      free(new_iwork);
      exit(1);
    }
    work = new_work;
    iwork = new_iwork;
    lwork = new_lwork;
    liwork = new_liwork;
    n_cached = n_query;
    iu_cached = iu_query;
  }

  lapack_int m = 0;
  double z_dummy = 0.0;
  lapack_int info = LAPACKE_dsyevr_work(LAPACK_COL_MAJOR,
                                       'N',
                                       'I',
                                       'U',
                                       n_lap,
                                       a,
                                       n_lap,
                                       0.0,
                                       0.0,
                                       1,
                                       iu,
                                       0.0,
                                       &m,
                                       eigenvalues,
                                       &z_dummy,
                                       1,
                                       isuppz,
                                       work,
                                       lwork,
                                       iwork,
                                       liwork);
  if (info != 0) {
    fprintf(stderr, "diagonalize_double_partial: LAPACKE_dsyevr_work failed (info=%d)\n", (int)info);
    exit(1);
  }
  if (m != iu) {
    fprintf(stderr, "diagonalize_double_partial: unexpected m=%d (expected %d)\n", (int)m, (int)iu);
    exit(1);
  }

  for (int i = (int)iu; i < nth_eig; i++) {
    eigenvalues[i] = 0.0;
  }
}
