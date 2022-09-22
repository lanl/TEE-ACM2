#ifndef __CPU_MAT_MULT__
#define __CPU_MAT_MULT__


//round up value to xxx.xx
float round_fl(float var)
{
    float value = (int)(var * 100 + .5);
    return (float)value / 100;
}

/**
 * Two matrices multiplication (keep all matrices on shared memory)
 * @param A Matrix A.
 * @param B Matrix B.
 * @param C Matrix multiplication results (i.e. C = A*B)
 * @param n_row_A number of rows of the A matrix
 * @param n_col_A number of collumns of the A and B matrices
 * @param n_row_B number of rows of the B matrix
 */
template <typename T>
void cpu_mat_mult(std::vector<T> & A, std::vector<T> & B, std::vector<T> & C, size_t n_row_A, size_t n_col_A, size_t n_row_B){
    //size_t i = 0;
    //size_t j = 0;
    T sum = 0;
    /*
    for (size_t k = 0; k < n_col_A; ++k) {
        T a = A[i * n_col_A + k];
        T b = B[k * n_row_B + j];
        sum += a * b;
    }
    C[i * n_row_B + j] = (T)sum;
        */
    

    
    for (size_t i = 0; i < n_row_A; ++i){
        for (size_t j = 0; j < n_row_B; ++j) {
            sum = 0;
            for (size_t k = 0; k < n_col_A; ++k) {
                T a = A[i * n_col_A + k];
                T b = B[k * n_row_B + j];
                sum += a * b;
            }
            C[i * n_row_B + j] = (T)sum;
        }
    }
    
}
#endif
