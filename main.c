#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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

	struct timeval st, en;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
		return EXIT_FAILURE;
	}
	filename = argv[1];

	mat = read_mat_coo(filename);
	
	printf("mat dim: %d\n", mat.dimension);
	
	for (int i = 0; i < number_of_eigenvalues; i++) {
		eigenvectors[i] = calloc(mat.dimension, sizeof(double));
	}
	
	gettimeofday(&st, NULL);
	lanczos(&mat, eigenvalues, eigenvectors, number_of_eigenvalues, 100, 10e-5);
	gettimeofday(&en, NULL);
	printf("elapsed time: %.6f sec\n", (double)(en.tv_sec - st.tv_sec) + (double)(en.tv_usec - st.tv_usec) * 1.0e-6);

	for (int i = 0; i < number_of_eigenvalues; i++) {
		printf("eigenvalue\t\t%d\t%.12f\n", i + 1, eigenvalues[i]);
	}

  return EXIT_SUCCESS;
}
