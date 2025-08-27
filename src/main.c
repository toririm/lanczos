#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "main.h"
#include "coo.h"
#include "util.h"
#include "lanczos.h"

#define number_of_eigenvalues 5
#define buf_size 1000

int main(int argc, char *argv[]) {
	char *filename;
	Mat_Coo mat;
	double eigenvalues[buf_size];
	double *eigenvectors[buf_size];

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];

	MEASURE(read_mat,
		mat = read_mat_coo(filename);
	);


	printf("mat dim: %d\n", mat.dimension);
	
	for (int i = 0; i < number_of_eigenvalues; i++) {
		eigenvectors[i] = calloc(mat.dimension, sizeof(double));
	}
	
	MEASURE(lanczos,
		lanczos(MAKE_MAT_MATVEC(&mat), eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
	);

	for (int i = 0; i < number_of_eigenvalues; i++) {
		printf("eigenvalue\t\t%d\t%.12f\n", i + 1, eigenvalues[i]);
	}

  return EXIT_SUCCESS;
}
