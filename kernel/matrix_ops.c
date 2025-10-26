#include "matrix_ops.h"

uint8_t extract_be(uint32_t a, int idx) 
{
    // big-endian logical order
    return (uint8_t)((a >> ((3 - idx) * 8)) & 0xFF);
}

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

float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // sources:
    // https://iitd.github.io/col729/lec/matrix_multiplication.html
    // https://www.reddit.com/r/learnprogramming/comments/le4ve/how_does_blocking_increase_speedup_of_matrix/

    if (A_cols != B_rows) { return NULL; }

    // allocate memory for the rows (each size of a float array)
    float **C = (float**)malloc((size_t)A_rows * sizeof(float*)); 

    // allocate memory for the cols of each row (each size of a float)
    for (int i = 0; i < A_rows; i++)
    {
	    C[i] = (float*)calloc((size_t)B_cols, sizeof(float)); // calloc initializes to zero.
    }

    int bi=0;
    int bj=0;
    int bk=0;

    int i=0;
    int j=0;
    int k=0;

    int blockSize = 0; // to be updated to be n/N where n is the smallest dimension of either matrix
    int N = 2;

    if (A_rows <= A_cols && A_rows <= B_rows && A_rows <= B_cols)
    {
        blockSize = A_rows / N; // if A_rows is the smallest
    }
    else if (A_cols <= A_rows && A_cols <= B_rows && A_cols <= B_cols)
    {
        blockSize = A_cols / N; // if A_cols is the smallest
    }
    else if (B_rows <= A_rows && B_rows <= A_cols && B_rows <= B_cols)
    {
        blockSize = B_rows / N; // if B_rows is the smallest
    }
    else 
    {
        blockSize = B_cols / N; // if B_cols is the smallest (casts to int...)
    }
	
    // implementation based on reddit post cited above.
    // basically blocking should be faster than naive because we are improving data locality (otherwise it's essentially the same)
    // by blocking, we only access one local block of a matrix at a time, so we have better cache hit rates
    for(bi = 0; bi<A_rows; bi+=blockSize)
    {
	    for(bj = 0; bj<B_cols; bj+=blockSize)
		{   
            for(bk = 0; bk<A_cols; bk+=blockSize)
            {
                for(i = 0; i < blockSize; i++)
                {
                    for(j = 0; j < blockSize; j++)
                    {
                        for(k = 0; k < blockSize; k++)
                        {
                            if (bi+i < A_rows && bk+k < A_cols && bj+j < B_cols) // no need to check B_cols because it = A_cols
                            {
                                C[bi+i][bj+j] +=  A[bi+i][bk+k] * B[bk+k][bj+j];
                            }
                        }
                    }
                }

            }
        }
    }
    return C;
}

float **matmul_quant(unsigned int **A, int **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_rows <= 0 || A_cols <= 0 || B_rows <= 0 || B_cols <= 0) { return NULL;}
    if (4 * A_cols != B_rows) { return NULL; }

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
		    for (int k = 0; k < B_rows; k++)
		    {
			    C[i][j] += (float)(extract_be(A[i][k/4], (k%4)) * B[k][j]);
		    }
	    }
    }
    return C;
}