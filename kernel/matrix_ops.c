#include "matrix_ops.h"

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols) 
{
    // TODO: Implement matrix multiplication
    if (A_cols != B_rows) { return NULL; }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float));
    }

    for (int i = 0; i < A_rows; i++)
    {
	    for (int j = 0; j < B_cols; j++)
	    {
		    for (int k = 0; k < A_cols; k++)
		    {
			    C[i][j] += A[i][k] * B[k][j];
		    }
	    }
    }
    return C;
}
