#pragma once

typedef struct coo {
	int 	index_row;
	int 	index_column;
	double 	value;
} Coo;

typedef struct mat_coo {
	Coo		*data;
	int 	length;
	int 	dimension;
} Mat_Coo;

extern void parse_to_coo(char *src_str, Coo *dist);
extern Mat_Coo read_mat_coo(const char *filepath);
extern void matvec_coo(const Mat_Coo *mat, const double vec[], double *dist);
extern int compare_coo(const void *a, const void *b);
extern void sort_matcoo(Mat_Coo *mat);
