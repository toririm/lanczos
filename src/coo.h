
typedef struct coo {
	int 		index_row;
	int 		index_column;
	double 		value;
} Coo;

typedef struct mat_coo {
	Coo		*data;
	int 	length;
	int 	dimension;
} Mat_Coo;

extern void parse_to_coo(char *src_str, Coo *dist);
extern Mat_Coo read_mat_coo(char *filepath);
extern double *matvec_coo(Mat_Coo *mat, double vec[]);
