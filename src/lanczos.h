/**
 * matvec 関数の一般的な型
 * 第一引数に mat を、第二引数に vec を取る
 */
typedef void Matvec_General(const void *mat, const double *vec, double *dist);
/**
 * mat と matvec をまとめた一般的な型
 * matvec の第一引数に mat を渡して動作することが期待される
 */
typedef struct mat_matvec {
    void            *mat;
    Matvec_General  *matvec;
    int             dimension;
} Mat_Matvec;

/**
 * Mat_Matvec を型安全に生成するためのマクロ
 * mat のポインタを渡すと、mat の型に応じて適切な matvec 関数をあてがってくれる
 */
#define MAKE_MAT_MATVEC(mat_ptr) \
    _Generic((mat_ptr), \
        Mat_Coo *: ((Mat_Matvec){ \
            .mat       = (void *)(mat_ptr), \
            .matvec    = (Matvec_General *)matvec_coo, \
            .dimension = (mat_ptr)->dimension \
        }), \
        Mat_Crs *: ((Mat_Matvec){ \
            .mat       = (void *)(mat_ptr), \
            .matvec    = (Matvec_General *)matvec_crs, \
            .dimension = (mat_ptr)->dimension \
        }) \
    )

extern void lanczos(const Mat_Matvec mat_matvec,
                    double eigenvalues[], double *eigenvectors[],
                    int nth_eig, int max_iter, double threshold);
