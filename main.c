#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "main.h"
#include "coo.h"
#include "util.h"

#define number_of_eigenvalues 5
#define buf_size 1000

int main(int argc, char *argv[]) {
	char *filename;
	Mat_Coo mat;
	double eigenvalues[buf_size];
	double *eigenvectors[buf_size];

	double start_time, end_time;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];

	start_time = omp_get_wtime();
	mat = read_mat_coo(filename);
	end_time = omp_get_wtime();
	printf("[TIME] read_mat_coo elapsed time: %.6f sec\n", end_time - start_time);

	printf("mat dim: %d\n", mat.dimension);
	
	for (int i = 0; i < number_of_eigenvalues; i++) {
		eigenvectors[i] = calloc(mat.dimension, sizeof(double));
	}
	
	start_time = omp_get_wtime();
	lanczos(&mat, eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
	end_time = omp_get_wtime();
	printf("[TIME] lanczos elapsed time: %.6f sec\n", end_time - start_time);

	for (int i = 0; i < number_of_eigenvalues; i++) {
		printf("eigenvalue\t\t%d\t%.12f\n", i + 1, eigenvalues[i]);
	}

  return EXIT_SUCCESS;
}
