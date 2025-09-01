#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "main.h"
#include "coo.h"
#include "crs.h"
#include "util.h"
#include "lanczos.h"

#define number_of_eigenvalues 5
#define buf_size 1000

int main(int argc, char *argv[]) {
	char *filename;
	Mat_Type mat_type;
	Mat_Coo mat;
	Mat_Crs mat_crs;
	double eigenvalues[buf_size];
	double *eigenvectors[buf_size];

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];
	if (strcmp(argv[2], "coo") == 0) {
		mat_type = COO;
	} else if (strcmp(argv[2], "crs") == 0) {
		mat_type = CRS;
	} else {
		fprintf(stderr, "Usage: %s <input_file> <coo|crs>\n", argv[0]);
		return EXIT_FAILURE;
	}

	MEASURE(read_mat,
		mat = read_mat_coo(filename);
	);

	printf("mat dim: %d\n", mat.dimension);

	if (mat_type == CRS) {
		printf("[MODE] CRS selected\n");
		// Fortran から出力されたファイルは 列優先 (Column Major) なので、CRSとして読み込むためソートする
		MEASURE(sort_matcoo,
			sort_matcoo(&mat);
		);
		MEASURE(convert_from_coo,
			mat_crs = convert_from_coo(&mat);
		);
		MEASURE(lanczos,
			lanczos(MAKE_MAT_MATVEC(&mat_crs), eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
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
