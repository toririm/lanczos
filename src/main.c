#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "main.h"
#include "coo.h"
#include "crs.h"
#include "util.h"
#include "lanczos.h"
#include "lanczos_cusparse.h"

#define number_of_eigenvalues 5
#define buf_size 1000

int main(int argc, const char *argv[]) {
	const char *filename;
	Mat_Type mat_type;
	Mat_Coo mat;
	Mat_Crs mat_crs;
	double eigenvalues[buf_size];
	double *eigenvectors[buf_size];

	printf("Using %d threads\n", omp_get_max_threads());

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs|crs_cusparse>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];
	if (strcmp(argv[2], "coo") == 0) {
		mat_type = COO;
	} else if (strcmp(argv[2], "crs") == 0) {
		mat_type = CRS;
	} else if (strcmp(argv[2], "crs_cusparse") == 0) {
		mat_type = CRS_CUSPARSE;
	} else {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs|crs_cusparse>\n", argv[0]);
		return EXIT_FAILURE;
	}

	MEASURE(read_mat,
		mat = read_mat_coo(filename);
	);

	printf("mat dim: %d\n", mat.dimension);

	if (mat_type == CRS) {
		printf("[MODE] CRS selected\n");
		MEASURE(convert_from_coo,
			mat_crs = convert_from_coo(&mat, 1);
		);
		MEASURE(lanczos,
			lanczos(MAKE_MAT_MATVEC(&mat_crs), eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
		);
	} else if (mat_type == CRS_CUSPARSE) {
		printf("[MODE] CRS_CUSPARSE selected\n");
		MEASURE(convert_from_coo,
			mat_crs = convert_from_coo(&mat, 1);
		);
		MEASURE(lanczos_cusparse_crs,
			lanczos_cusparse_crs(&mat_crs, eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
		);
	} else {
		printf("[MODE] COO selected\n");
		MEASURE(lanczos,
			lanczos(MAKE_MAT_MATVEC(&mat), eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
		);
	}

	for (int i = 0; i < number_of_eigenvalues; i++) {
		printf("eigenvalue\t\t%d\t%.12f\n", i + 1, eigenvalues[i]);
	}

  return EXIT_SUCCESS;
}
