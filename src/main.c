#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "main.h"
#include "coo.h"
#include "crs.h"
#include "util.h"
#include "lanczos.h"
#include "lanczos_cuda.h"

#define buf_size 1000

int main(int argc, const char *argv[]) {
	const char *filename;
	Mat_Type mat_type;
	Mat_Coo mat;
	Mat_Crs mat_crs;
	double eigenvalues[buf_size];
	double *eigenvectors[buf_size];
	int number_of_eigenvalues;

	printf("Using %d threads\n", omp_get_max_threads());

	if (argc != 5) {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs|crs_cuda> <number_of_eigenvalues> <max_iter>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];
	if (strcmp(argv[2], "coo") == 0) {
		mat_type = COO;
	} else if (strcmp(argv[2], "crs") == 0) {
		mat_type = CRS;
	} else if (strcmp(argv[2], "crs_cuda") == 0) {
		mat_type = CRS_CUDA;
	} else {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs|crs_cuda> <number_of_eigenvalues> <max_iter>\n", argv[0]);
		return EXIT_FAILURE;
	}

	number_of_eigenvalues = atoi(argv[3]);
	int max_iter = atoi(argv[4]);

	MEASURE(read_mat,
		mat = read_mat_coo_incremental(filename);
	);

	printf("mat dim: %ld\n", mat.dimension);
	printf("mat num-of-non-zero: %ld\n", mat.length);
	printf("sparsity: %f\n", ((double)mat.length / ((double)mat.dimension * (double)mat.dimension)));

	if (mat_type == CRS) {
		printf("[MODE] CRS selected\n");
		MEASURE(convert_from_coo,
			mat_crs = convert_from_coo(&mat, 1);
		);
		free(mat.data);
		MEASURE(lanczos,
			lanczos(MAKE_MAT_MATVEC(&mat_crs), eigenvalues, eigenvectors, number_of_eigenvalues, max_iter, 10e-5);
		);
	} else if (mat_type == CRS_CUDA) {
		printf("[MODE] CRS_CUDA selected\n");
		MEASURE(convert_from_coo,
			mat_crs = convert_from_coo(&mat, 1);
		);
		free(mat.data);
		MEASURE(lanczos_cuda_crs,
			lanczos_cuda_crs(&mat_crs, eigenvalues, eigenvectors, number_of_eigenvalues, max_iter, 10e-5);
		);
	} else {
		printf("[MODE] COO selected\n");
		MEASURE(lanczos,
			lanczos(MAKE_MAT_MATVEC(&mat), eigenvalues, eigenvectors, number_of_eigenvalues, max_iter, 10e-5);
		);
	}

	for (int i = 0; i < number_of_eigenvalues; i++) {
		printf("eigenvalue\t\t%d\t%.12f\n", i + 1, eigenvalues[i]);
	}

  return EXIT_SUCCESS;
}
